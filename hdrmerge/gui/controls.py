"""Every command-line option, as controls.

The panels mirror the CLI exactly -- anything `hdrmerge merge` accepts is here,
grouped the way you actually work rather than the way argparse lists it: what
the merge does, how it is tone mapped, how it is graded, and where it goes.

Two signals, deliberately separate:

* ``structural_changed`` -- settings that require re-merging (alignment,
  response curve, deghosting, reference frame, preview scale, and the fusion
  operator, which never builds a radiance map).
* ``display_changed`` -- settings that only re-tone-map a merge we already
  have, which is nearly free and can therefore run live as a slider moves.

Getting a setting into the wrong group either makes the UI sluggish or makes it
show a stale image, so each one is placed explicitly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..adjust import Adjustments
from ..align import METHODS as ALIGN_METHODS
from ..crf import METHODS as CRF_METHODS
from ..tonemap import OPERATORS as TONEMAP_OPERATORS
from ..writers import FORMATS, available
from . import theme


def _heading(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setProperty("heading", True)
    return label


def _rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background: {theme.BORDER}; max-height: 1px; border: none;")
    return line


class LabelledSlider(QWidget):
    """A -100..100 photo slider with a live value and double-click to reset.

    Sliders are integers in Qt, which suits every adjustment here except
    exposure, so ``scale`` converts to the real units (stops for exposure).
    """

    value_changed = Signal(float)
    released = Signal()

    def __init__(self, label: str, minimum: int = -100, maximum: int = 100,
                 default: int = 0, scale: float = 1.0, suffix: str = "",
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self._scale = scale
        self._default = default
        self._suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        self.readout = QLabel()
        self.readout.setProperty("dim", True)
        self.readout.setAlignment(Qt.AlignRight)
        header.addWidget(self.name)
        header.addStretch()
        header.addWidget(self.readout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(default)
        if tooltip:
            self.setToolTip(tooltip)
            self.slider.setToolTip(tooltip)

        layout.addLayout(header)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_change)
        self.slider.sliderReleased.connect(self.released)
        self._update_readout()

    def _on_change(self, _value: int) -> None:
        self._update_readout()
        self.value_changed.emit(self.value())

    def _update_readout(self) -> None:
        value = self.value()
        text = f"{value:+.2f}" if self._scale != 1.0 else f"{value:+.0f}"
        if value == 0:
            text = "0"
        self.readout.setText(text + self._suffix)

    def value(self) -> float:
        return self.slider.value() * self._scale

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round(value / self._scale)))

    def is_default(self) -> bool:
        return self.slider.value() == self._default

    def reset(self) -> None:
        self.slider.setValue(self._default)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.reset()
        super().mouseDoubleClickEvent(event)


class MergePanel(QWidget):
    """Settings that change the merge itself, and so require re-merging."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setSpacing(8)

        self.align = QComboBox()
        self.align.addItems(list(ALIGN_METHODS))
        self.align.setToolTip(
            "mtb — exposure-invariant alignment for handheld brackets (default)\n"
            "ecc — also corrects slight rotation; slower\n"
            "none — tripod; skips alignment entirely and is faster"
        )

        self.deghost = QCheckBox("Remove ghosts from moving subjects")
        self.deghost.setChecked(True)
        self.deghost.setToolTip(
            "Rejects pixels that disagree with the reference frame, so a moving "
            "subject resolves to one position instead of smearing across frames."
        )

        self.deghost_threshold = QDoubleSpinBox()
        self.deghost_threshold.setRange(0.1, 5.0)
        self.deghost_threshold.setSingleStep(0.1)
        self.deghost_threshold.setValue(0.7)
        self.deghost_threshold.setSuffix(" stops")
        self.deghost_threshold.setToolTip(
            "How far a pixel may differ from the reference before it is treated "
            "as movement. Lower rejects more."
        )

        self.crf = QComboBox()
        self.crf.addItems(list(CRF_METHODS))
        self.crf.setToolTip(
            "How to linearise rendered images. RAW is already linear, so 'auto' "
            "leaves it alone and uses Debevec for JPEG/TIFF/PNG."
        )

        self.reference = QComboBox()
        self.reference.setToolTip(
            "Which frame alignment and deghosting are measured against. "
            "The middle exposure usually has the most usable pixels."
        )

        self.preview_scale = QComboBox()
        for label, value in (("Full", 1.0), ("1/2", 0.5), ("1/4", 0.25), ("1/8", 0.125)):
            self.preview_scale.addItem(label, value)
        self.preview_scale.setCurrentIndex(2)
        self.preview_scale.setToolTip(
            "Resolution the preview merges at. Exports always run at full "
            "resolution regardless of this."
        )

        self.max_frames = QSpinBox()
        self.max_frames.setRange(2, 99)
        self.max_frames.setValue(15)
        self.max_frames.setToolTip(
            "Refuse brackets larger than this. A 24MP frame costs roughly 300MB "
            "in memory, so this is a guard against running out."
        )

        form.addRow("Alignment", self.align)
        form.addRow("Response curve", self.crf)
        form.addRow("Reference frame", self.reference)
        form.addRow("Preview quality", self.preview_scale)
        form.addRow("Max frames", self.max_frames)
        layout.addLayout(form)

        layout.addSpacing(4)
        layout.addWidget(self.deghost)
        ghost_row = QFormLayout()
        ghost_row.setContentsMargins(20, 0, 0, 0)
        ghost_row.addRow("Threshold", self.deghost_threshold)
        layout.addLayout(ghost_row)
        layout.addStretch()

        for widget in (self.align, self.crf, self.reference, self.preview_scale):
            widget.currentIndexChanged.connect(self.changed)
        self.max_frames.valueChanged.connect(self.changed)
        self.deghost_threshold.valueChanged.connect(self.changed)
        self.deghost.toggled.connect(self._on_deghost_toggled)
        self.deghost_threshold.setEnabled(True)

    def _on_deghost_toggled(self, enabled: bool) -> None:
        self.deghost_threshold.setEnabled(enabled)
        self.changed.emit()

    def set_frame_count(self, count: int, reference: int) -> None:
        """Repopulate the reference chooser for a newly loaded bracket."""
        blocked = self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItem("Auto (middle exposure)", None)
        for index in range(count):
            self.reference.addItem(f"Frame {index + 1}", index)
        if 0 <= reference < count:
            self.reference.setItemText(reference + 1, f"Frame {reference + 1}  ·  current")
        self.reference.blockSignals(blocked)

    def threshold(self) -> float:
        return self.deghost_threshold.value() if self.deghost.isChecked() else 0.0


class TonePanel(QWidget):
    """Tone mapping and the full grading set. All cheap to re-apply."""

    changed = Signal()
    slider_released = Signal()

    #: (attribute, label, min, max, scale, suffix, tooltip)
    SLIDERS = [
        ("exposure", "Exposure", -50, 50, 0.1, " EV",
         "Overall brightness, in stops."),
        ("temp", "Temperature", -100, 100, 1.0, "",
         "Cooler (−) to warmer (+)."),
        ("tint", "Tint", -100, 100, 1.0, "",
         "Green (−) to magenta (+)."),
        ("highlights", "Highlights", -100, 100, 1.0, "",
         "Recover blown highlights (−) or lift them (+)."),
        ("shadows", "Shadows", -100, 100, 1.0, "",
         "Open up shadows (+) or deepen them (−)."),
        ("whites", "Whites", -100, 100, 1.0, "",
         "Move the white point. Positive brightens the brightest tones."),
        ("blacks", "Blacks", -100, 100, 1.0, "",
         "Move the black point. Negative crushes the darkest tones."),
        ("contrast", "Contrast", -100, 100, 1.0, "",
         "S-curve around mid-grey."),
        ("saturation", "Saturation", -100, 100, 1.0, "",
         "Scales all colour equally. −100 is monochrome."),
        ("vibrance", "Vibrance", -100, 100, 1.0, "",
         "Boosts muted colours more than already-saturated ones, so skies lift "
         "without skin tones going orange."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(8)
        self.tonemap = QComboBox()
        self.tonemap.addItems(list(TONEMAP_OPERATORS))
        self.tonemap.setToolTip(
            "reinhard — natural and predictable (default)\n"
            "drago — gentle highlight rolloff\n"
            "mantiuk — stronger local contrast\n"
            "linear — no curve; gamma and clip\n"
            "fusion — blends frames directly, no radiance map "
            "(cannot write EXR/HDR/32-bit TIFF)"
        )
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(2.2)
        self.gamma.setToolTip("Tone-mapping gamma.")
        form.addRow("Operator", self.tonemap)
        form.addRow("Gamma", self.gamma)
        layout.addLayout(form)

        layout.addSpacing(6)
        header = QHBoxLayout()
        header.addWidget(_heading("Adjustments"))
        header.addStretch()
        self.auto = QPushButton("Auto")
        self.auto.setToolTip(
            "A tuned starting grade: mild shadow lift, highlight recovery, a "
            "little contrast and vibrance."
        )
        self.auto.setMinimumWidth(58)
        self.reset = QPushButton("Reset")
        self.reset.setMinimumWidth(58)
        header.addWidget(self.auto)
        header.addWidget(self.reset)
        layout.addLayout(header)
        layout.addWidget(_rule())

        self.sliders: Dict[str, LabelledSlider] = {}
        for name, label, low, high, scale, suffix, tip in self.SLIDERS:
            slider = LabelledSlider(label, low, high, 0, scale, suffix, tip)
            self.sliders[name] = slider
            layout.addWidget(slider)

        layout.addSpacing(6)
        gamma_form = QFormLayout()
        self.output_gamma = QDoubleSpinBox()
        self.output_gamma.setRange(0.1, 5.0)
        self.output_gamma.setSingleStep(0.05)
        self.output_gamma.setValue(1.0)
        self.output_gamma.setToolTip("Final gamma trim, applied after everything else.")
        gamma_form.addRow("Output gamma", self.output_gamma)
        layout.addLayout(gamma_form)

        layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self.tonemap.currentIndexChanged.connect(self.changed)
        self.gamma.valueChanged.connect(self.changed)
        self.output_gamma.valueChanged.connect(self.changed)
        for slider in self.sliders.values():
            slider.value_changed.connect(lambda _v: self.changed.emit())
            slider.released.connect(self.slider_released)
        self.auto.clicked.connect(self.apply_auto)
        self.reset.clicked.connect(self.reset_all)

    def apply_auto(self) -> None:
        self.reset_all(emit=False)
        for name, value in Adjustments.AUTO.items():
            if name in self.sliders:
                self.sliders[name].set_value(value)
        self.slider_released.emit()

    def reset_all(self, emit: bool = True) -> None:
        for slider in self.sliders.values():
            slider.blockSignals(True)
            slider.reset()
            slider.blockSignals(False)
        self.output_gamma.blockSignals(True)
        self.output_gamma.setValue(1.0)
        self.output_gamma.blockSignals(False)
        if emit:
            self.slider_released.emit()

    def adjustments(self) -> Adjustments:
        values = {name: slider.value() for name, slider in self.sliders.items()}
        values["gamma"] = self.output_gamma.value()
        return Adjustments(**values)

    @property
    def is_fusion(self) -> bool:
        return self.tonemap.currentText() == "fusion"


class OutputPanel(QWidget):
    """Formats and where they go. Purely an export concern."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)

        layout.addWidget(_heading("Formats"))
        self.formats: Dict[str, QCheckBox] = {}
        grid = QGridLayout()
        grid.setSpacing(4)
        for row, (key, spec) in enumerate(FORMATS.items()):
            box = QCheckBox(key)
            box.setToolTip(spec[2])
            usable = available(key)
            if not usable:
                box.setEnabled(False)
                box.setToolTip(
                    spec[2] + "\n\nUnavailable: needs the optional OpenEXR "
                    'package (pip install "hdrmerge[exr]").'
                )
            box.setChecked(key == "jpg")
            self.formats[key] = box
            description = QLabel(spec[2] + ("" if usable else "   (unavailable)"))
            description.setProperty("dim", True)
            grid.addWidget(box, row, 0)
            grid.addWidget(description, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        layout.addSpacing(8)
        layout.addWidget(_rule())
        form = QFormLayout()
        form.setSpacing(8)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("Alongside the source images")
        self.browse = QPushButton("Choose folder…")

        self.suffix = QLineEdit("_hdr")
        self.suffix.setToolTip("Appended to the output filename.")

        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setValue(95)

        form.addRow("Suffix", self.suffix)
        form.addRow("JPEG quality", self.jpeg_quality)
        layout.addLayout(form)

        # Destination gets the full panel width: a path is long, and squeezing
        # it beside a button truncates it to uselessness.
        layout.addSpacing(6)
        layout.addWidget(_heading("Destination"))
        layout.addWidget(self.out_dir)
        layout.addWidget(self.browse)
        layout.addStretch()

        self.browse.clicked.connect(self._choose_directory)
        for box in self.formats.values():
            box.toggled.connect(self.changed)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose an output folder")
        if directory:
            self.out_dir.setText(directory)

    def selected(self) -> List[str]:
        return [key for key, box in self.formats.items() if box.isChecked()]

    def set_radiance_enabled(self, enabled: bool, reason: str = "") -> None:
        """Grey out radiance formats when the operator cannot produce them.

        Exposure fusion never builds a radiance map, so offering EXR alongside
        it would only produce an error at export time.
        """
        from ..writers import RADIANCE_FORMATS

        for key in RADIANCE_FORMATS:
            box = self.formats[key]
            if not available(key):
                continue
            box.setEnabled(enabled)
            if not enabled:
                box.setChecked(False)
                box.setToolTip(reason)
            else:
                box.setToolTip(FORMATS[key][2])
