"""The main window: state, threading and the wiring between panels.

The central idea is the two-stage split from :mod:`hdrmerge.pipeline`. A merged
scene is cached; structural changes throw it away and re-merge, display changes
just re-render it. That is what makes ten grading sliders usable on a bracket
that takes seconds to merge.

Live dragging renders small and fast; letting go renders at full preview size.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..grouping import Bracket, group_frames
from ..loaders import LoadError, expand_inputs
from ..metadata import FrameMeta, read_exif
from ..pipeline import MergedScene, MergeOptions
from ..writers import WriteError
from . import theme
from .controls import MergePanel, OutputPanel, TonePanel
from .filmstrip import Filmstrip
from .preview import PreviewPane, paths_from
from .thumbnails import load_thumbnail
from .workers import ExportJob, JobRunner, MergeJob, RenderJob

log = logging.getLogger(__name__)

#: Render size while a slider is being dragged. Small enough that tone mapping
#: keeps up with the mouse (~70ms), which is the whole point of a live preview.
INTERACTIVE_EDGE = 640

#: How long to wait after the last change before rendering at full preview size.
SETTLE_MS = 220


class MainWindow(QMainWindow):
    def __init__(self, initial: Optional[Sequence[str]] = None):
        super().__init__()
        self.setWindowTitle("hdrmerge")
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self.jobs = JobRunner(self)
        self._token = 0
        self._scene: Optional[MergedScene] = None
        self._paths: List[str] = []
        self._brackets: List[Bracket] = []
        self._thumbnails: Dict[str, object] = {}
        self._dragging = False
        self._busy = False

        self._build_ui()
        self._build_menu()

        # One timer, restarted on every change, so a burst of slider moves ends
        # in a single full-quality render rather than a queue of them.
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.timeout.connect(self._render_full)

        if initial:
            self.load_paths(list(initial))

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        self.preview = PreviewPane()

        # The filmstrip is meaningless before anything is loaded, and an empty
        # panel beside the drop zone reads as a rendering fault rather than an
        # empty list, so it stays hidden until there are frames to show.
        self.frames_panel = QWidget()
        self.frames_panel.setObjectName("Panel")
        frames_layout = QVBoxLayout(self.frames_panel)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(0)

        strip_header = QLabel("FRAMES  ·  DARKEST FIRST")
        strip_header.setProperty("heading", True)
        strip_header.setContentsMargins(12, 10, 12, 6)

        self.filmstrip = Filmstrip()
        frames_layout.addWidget(strip_header)
        frames_layout.addWidget(self.filmstrip, 1)

        self.frames_panel.setMinimumWidth(250)
        self.frames_panel.setMaximumWidth(400)
        self.frames_panel.hide()

        self.merge_panel = MergePanel()
        self.tone_panel = TonePanel()
        self.output_panel = OutputPanel()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("Panel")
        self.tabs.addTab(self.tone_panel, "Tone")
        self.tabs.addTab(self.merge_panel, "Merge")
        self.tabs.addTab(self.output_panel, "Output")

        right = QWidget()
        right.setObjectName("Panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.tabs, 1)

        self.export_button = QPushButton("Export")
        self.export_button.setProperty("primary", True)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        self.export_hint = QLabel()
        self.export_hint.setProperty("dim", True)
        self.export_hint.setAlignment(Qt.AlignCenter)

        footer = QVBoxLayout()
        footer.setContentsMargins(12, 10, 12, 12)
        footer.setSpacing(6)
        footer.addWidget(self.export_hint)
        footer.addWidget(self.export_button)
        right_layout.addLayout(footer)
        right.setFixedWidth(330)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.frames_panel)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 900])

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter, 1)
        layout.addWidget(right)
        self.setCentralWidget(central)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)          # indeterminate
        self.progress.setFixedWidth(150)
        self.progress.hide()
        self.status = QLabel("Drop a bracket to begin")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.progress)

        self.merge_panel.changed.connect(self._on_structural_changed)
        self.tone_panel.changed.connect(self._on_display_changed)
        self.tone_panel.slider_released.connect(self._on_display_changed)
        self.tone_panel.tonemap.currentIndexChanged.connect(self._on_operator_changed)
        self.output_panel.changed.connect(self._update_export_hint)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_files = QAction("&Open Images…", self)
        open_files.setShortcut(QKeySequence.Open)
        open_files.triggered.connect(self.choose_files)

        open_folder = QAction("Open &Folder…", self)
        open_folder.triggered.connect(self.choose_folder)

        self.export_action = QAction("&Export", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export)

        close = QAction("&Clear", self)
        close.setShortcut(QKeySequence("Ctrl+W"))
        close.triggered.connect(self.clear)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

        for action in (open_files, open_folder):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(close)
        file_menu.addAction(quit_action)

    # ------------------------------------------------------------ drag & drop

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if paths_from(event):
            event.acceptProposedAction()
            self.preview.set_drag_hover(True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.preview.set_drag_hover(False)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.preview.set_drag_hover(False)
        paths = paths_from(event)
        if paths:
            event.acceptProposedAction()
            self.load_paths(paths)

    def choose_files(self) -> None:
        from ..loaders import SUPPORTED_EXTENSIONS

        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open bracketed exposures", "",
            f"Images ({patterns});;All files (*)",
        )
        if paths:
            self.load_paths(paths)

    def choose_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open a folder of brackets")
        if directory:
            self.load_paths([directory])

    # -------------------------------------------------------------- loading

    def load_paths(self, paths: Sequence[str]) -> None:
        """Take dropped files or folders and prepare a bracket to preview."""
        try:
            files = expand_inputs(list(paths))
        except LoadError as exc:
            self._error("Could not read those files", str(exc))
            return

        folder_dropped = any(os.path.isdir(p) for p in paths)
        if folder_dropped:
            self._brackets = group_frames(files)
            if not self._brackets:
                self._error(
                    "No brackets found",
                    "That folder has no group of two or more frames that looks "
                    "like a bracket.",
                )
                return
            self._paths = self._brackets[0].paths
            self.status.setText(
                f"{len(self._brackets)} brackets detected — previewing the first"
            )
        else:
            self._brackets = [Bracket([read_exif(path) for path in files])]
            self._paths = files
            self.status.setText(f"{len(files)} frames loaded")

        if len(self._paths) < 2:
            self._error(
                "Not a bracket",
                "A merge needs at least two frames at different exposures.",
            )
            return

        self._show_frames()
        self._update_export_hint()
        self._on_structural_changed()

    def _show_frames(self) -> None:
        metas = [read_exif(path) for path in self._paths]
        # Sorted by exposure so the strip reads darkest to brightest, matching
        # what the merge does and what the reference-frame numbers refer to.
        order = sorted(range(len(metas)),
                       key=lambda i: (metas[i].relative_exposure or 0.0, metas[i].name))
        metas = [metas[i] for i in order]
        self._paths = [self._paths[i] for i in order]

        thumbnails = [self._thumbnail(meta.path) for meta in metas]
        reference = len(metas) // 2
        self.filmstrip.set_frames(metas, thumbnails, reference)
        self.merge_panel.set_frame_count(len(metas), reference)
        self.frames_panel.show()

    def _thumbnail(self, path: str):
        if path not in self._thumbnails:
            self._thumbnails[path] = load_thumbnail(path)
        return self._thumbnails[path]

    def clear(self) -> None:
        self._scene = None
        self._paths = []
        self._brackets = []
        self._thumbnails.clear()
        self.filmstrip.clear_frames()
        self.frames_panel.hide()
        self.preview.show_empty()
        self.export_button.setEnabled(False)
        self.export_action.setEnabled(False)
        self.export_hint.clear()
        self.status.setText("Drop a bracket to begin")

    # -------------------------------------------------------------- options

    def options(self, *, for_export: bool = False) -> MergeOptions:
        """Assemble a MergeOptions from every panel.

        Preview and export differ in exactly two ways: export runs at full
        resolution and actually writes files.
        """
        return MergeOptions(
            preview_scale=1.0 if for_export else self.merge_panel.preview_scale.currentData(),
            max_frames=self.merge_panel.max_frames.value(),
            reference=self.merge_panel.reference.currentData(),
            crf_method=self.merge_panel.crf.currentText(),
            align_method=self.merge_panel.align.currentText(),
            deghost_threshold=self.merge_panel.threshold(),
            formats=self.output_panel.selected() if for_export else [],
            out_dir=self.output_panel.out_dir.text().strip() or None,
            suffix=self.output_panel.suffix.text() or "_hdr",
            jpeg_quality=self.output_panel.jpeg_quality.value(),
            tonemap=self.tone_panel.tonemap.currentText(),
            gamma=self.tone_panel.gamma.value(),
            adjustments=self.tone_panel.adjustments(),
        )

    def _on_operator_changed(self) -> None:
        fusion = self.tone_panel.is_fusion
        self.output_panel.set_radiance_enabled(
            not fusion,
            "Exposure fusion blends the frames directly and never builds a "
            "radiance map, so it cannot write this format.",
        )
        # Fusion is the one display setting that is structurally different: it
        # skips the radiance map entirely, so the cached scene cannot serve it.
        self._on_structural_changed()

    # --------------------------------------------------------------- merging

    def _on_structural_changed(self) -> None:
        if not self._paths:
            return
        self._token += 1
        token = self._token
        self._set_busy(True, "Merging…")

        job = MergeJob(token, self._paths, self.options())
        job.signals.scene_ready.connect(self._on_scene_ready)
        job.signals.failed.connect(self._on_failed)
        job.signals.progress.connect(self._on_progress)
        self.jobs.start(job)

    @Slot(int, object)
    def _on_scene_ready(self, token: int, scene: MergedScene) -> None:
        if token != self._token:
            return          # a newer request has already superseded this one
        self._scene = scene
        self._set_busy(False)
        self._describe_scene(scene)
        self.export_button.setEnabled(True)
        self.export_action.setEnabled(True)
        self._render_full()

    def _describe_scene(self, scene: MergedScene) -> None:
        width, height = scene.size
        parts = [f"{len(scene.metas)} frames", f"{width}×{height} preview"]
        if scene.stats is not None:
            parts.append(f"{scene.stats.dynamic_range_stops:.1f} stops recovered")
        if len(self._brackets) > 1:
            parts.append(f"{len(self._brackets)} brackets queued")
        self.status.setText("  ·  ".join(parts))

    # -------------------------------------------------------------- rendering

    def _on_display_changed(self) -> None:
        """A grading change: re-render, cheaply, without re-merging."""
        if self._scene is None:
            return
        self._dragging = any(
            slider.slider.isSliderDown() for slider in self.tone_panel.sliders.values()
        )
        self._render(INTERACTIVE_EDGE if self._dragging else None)
        self._settle.start(SETTLE_MS)

    def _render_full(self) -> None:
        self._settle.stop()
        self._render(self.preview.longest_edge)

    def _render(self, longest_edge: Optional[int]) -> None:
        if self._scene is None:
            return
        self._token += 1
        job = RenderJob(self._token, self._scene, self.options(), longest_edge)
        job.signals.render_ready.connect(self._on_render_ready)
        job.signals.failed.connect(self._on_failed)
        self.jobs.start(job)

    @Slot(int, object)
    def _on_render_ready(self, token: int, display) -> None:
        if token != self._token or display is None:
            return
        self.preview.show_image(display)

    # --------------------------------------------------------------- export

    def export(self) -> None:
        if not self._paths:
            return
        formats = self.output_panel.selected()
        if not formats:
            self._error("Nothing to export", "Choose at least one output format.")
            return

        brackets = [b.paths for b in self._brackets] if self._brackets else [self._paths]
        options = self.options(for_export=True)
        try:
            options.validate()
        except (ValueError, WriteError) as exc:
            self._error("Those settings cannot be exported", str(exc))
            return

        self._token += 1
        self._set_busy(True, "Exporting…")
        job = ExportJob(self._token, brackets, options)
        job.signals.export_done.connect(self._on_export_done)
        job.signals.failed.connect(self._on_failed)
        job.signals.progress.connect(self._on_progress)
        self.jobs.start(job)

    @Slot(int, list)
    def _on_export_done(self, token: int, written: List[str]) -> None:
        self._set_busy(False)
        if not written:
            self.status.setText("Nothing was written")
            return
        where = os.path.dirname(written[0])
        self.status.setText(f"Wrote {len(written)} file(s) to {where}")
        QMessageBox.information(
            self, "Export complete",
            f"Wrote {len(written)} file(s):\n\n"
            + "\n".join(os.path.basename(p) for p in written[:12])
            + ("\n…" if len(written) > 12 else ""),
        )

    def _update_export_hint(self) -> None:
        formats = self.output_panel.selected()
        if not self._paths:
            self.export_hint.clear()
            return
        count = max(1, len(self._brackets))
        noun = "bracket" if count == 1 else "brackets"
        self.export_hint.setText(
            f"{count} {noun} → {', '.join(formats) if formats else 'no format selected'}"
        )

    # ---------------------------------------------------------------- status

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self.progress.setVisible(busy)
        self.export_button.setEnabled(not busy and self._scene is not None)
        if message:
            self.status.setText(message)

    @Slot(int, str)
    def _on_progress(self, token: int, message: str) -> None:
        if token == self._token:
            self.status.setText(message)

    @Slot(int, str)
    def _on_failed(self, token: int, message: str) -> None:
        if token != self._token:
            return
        self._set_busy(False)
        self._error("That did not work", message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Let in-flight jobs go before the widgets they report to disappear."""
        self.jobs.shutdown()
        super().closeEvent(event)

    def _error(self, title: str, message: str) -> None:
        self.status.setText(message.splitlines()[0])
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(title)
        box.setInformativeText(message)
        box.exec()
