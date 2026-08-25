"""Background work.

Merging a full-resolution bracket takes seconds and rendering takes hundreds of
milliseconds. Neither may run on the GUI thread: a frozen window during a merge
looks like a crash. Everything here runs on Qt's thread pool and reports back
through signals.

Each worker carries a ``token``. Settings change faster than merges finish, so
results arrive out of order and stale ones have to be recognised and dropped --
the window compares the token against what it currently wants.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ..pipeline import MergedScene, MergeOptions, merge_scene, render

log = logging.getLogger(__name__)


class _Signals(QObject):
    """QRunnable cannot carry signals itself, so they live on a helper."""

    scene_ready = Signal(int, object)        # token, MergedScene
    render_ready = Signal(int, object)       # token, display image
    export_done = Signal(int, list)          # token, written paths
    progress = Signal(int, str)              # token, message
    failed = Signal(int, str)                # token, message


class _Job(QRunnable):
    def __init__(self, token: int):
        super().__init__()
        self.token = token
        self.cancelled = False
        self.signals = _Signals()
        # Qt deletes an auto-delete QRunnable as soon as run() returns, but the
        # Python wrapper -- and the _Signals object hanging off it -- can be
        # garbage collected even sooner, because QThreadPool.start() does not
        # keep a Python reference. The signal then fires into a deleted object
        # ("Signal source has been deleted") and the result is silently lost.
        # JobRunner below holds the reference; Qt must not free it underneath.
        self.setAutoDelete(False)

    def _fail(self, exc: BaseException) -> None:
        log.debug("Worker failed: %s", traceback.format_exc())
        self._emit(self.signals.failed, self.token, str(exc) or exc.__class__.__name__)

    def _emit(self, signal, *args) -> None:
        """Emit unless the receiver is gone.

        Closing the window while a merge is still running destroys the objects
        this job is about to emit into. Qt raises RuntimeError from the worker
        thread when that happens, which is noise at best -- the answer is no
        longer wanted either way.
        """
        if self.cancelled:
            return
        try:
            signal.emit(*args)
        except RuntimeError:
            log.debug("Dropped a result for a receiver that no longer exists")


class MergeJob(_Job):
    """The expensive stage: load, align, linearise, merge."""

    def __init__(self, token: int, paths: Sequence[str], options: MergeOptions):
        super().__init__(token)
        self.paths = list(paths)
        self.options = options

    @Slot()
    def run(self) -> None:
        try:
            self._emit(self.signals.progress, self.token, f"Merging {len(self.paths)} frames…")
            scene = merge_scene(self.paths, self.options)
        except BaseException as exc:  # noqa: BLE001 - reported, never raised into Qt
            self._fail(exc)
        else:
            self._emit(self.signals.scene_ready, self.token, scene)


class RenderJob(_Job):
    """The cheap stage: tone map and grade an already-merged scene."""

    def __init__(self, token: int, scene: MergedScene, options: MergeOptions,
                 longest_edge: Optional[int] = None):
        super().__init__(token)
        self.scene = scene
        self.options = options
        self.longest_edge = longest_edge

    @Slot()
    def run(self) -> None:
        try:
            scene = _downscaled(self.scene, self.longest_edge)
            _, display = render(scene, self.options)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
        else:
            self._emit(self.signals.render_ready, self.token, display)


class ExportJob(_Job):
    """Full-resolution merge and write, for one bracket or a whole batch."""

    def __init__(self, token: int, brackets: Sequence[Sequence[str]], options: MergeOptions):
        super().__init__(token)
        self.brackets = [list(b) for b in brackets]
        self.options = options

    @Slot()
    def run(self) -> None:
        from ..pipeline import merge_bracket

        written: List[str] = []
        try:
            for index, paths in enumerate(self.brackets, start=1):
                if len(self.brackets) > 1:
                    self._emit(
                        self.signals.progress,
                        self.token, f"Exporting bracket {index} of {len(self.brackets)}…",
                    )
                else:
                    self._emit(self.signals.progress, self.token, "Exporting at full resolution…")
                written.extend(merge_bracket(paths, self.options).written)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
        else:
            self._emit(self.signals.export_done, self.token, written)


def _downscaled(scene: MergedScene, longest_edge: Optional[int]) -> MergedScene:
    """A smaller copy of a scene, for the fast render while a slider is moving.

    Shrinking the radiance map is what makes live dragging feel live: tone
    mapping cost scales with pixel count, so a 640px render lands in about
    70ms where a 1200px one takes 300ms and drags visibly behind the mouse.
    """
    if not longest_edge:
        return scene

    import cv2

    image = scene.radiance if scene.radiance is not None else scene.fused
    if image is None:
        return scene
    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    if scale >= 1.0:
        return scene

    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale)))) 
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return MergedScene(
        metas=scene.metas,
        reference=scene.reference,
        radiance=small if scene.radiance is not None else None,
        fused=small if scene.fused is not None else None,
        stats=scene.stats,
    )


class JobRunner(QObject):
    """Runs jobs and keeps them alive until they report back.

    QThreadPool.start() takes the QRunnable but not a Python reference to it,
    so without this every job is a race between finishing and being collected.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._running = set()

    def start(self, job: _Job) -> None:
        self._running.add(job)
        # Every job ends in exactly one of these. progress is not terminal, so
        # it must not release the reference.
        for signal in (
            job.signals.scene_ready,
            job.signals.render_ready,
            job.signals.export_done,
            job.signals.failed,
        ):
            signal.connect(lambda *_args, finished=job: self._release(finished))
        self._pool.start(job)

    def _release(self, job: _Job) -> None:
        self._running.discard(job)

    def wait(self, milliseconds: int = 30_000) -> bool:
        """Block until everything finishes. For tests and shutdown, not the UI."""
        return self._pool.waitForDone(milliseconds)

    def shutdown(self, milliseconds: int = 5_000) -> None:
        """Stop caring about in-flight results, then give them a moment to end.

        Called when the window closes. Without it a merge still running emits
        into half-destroyed widgets and Qt raises from the worker thread.
        """
        for job in self._running:
            job.cancelled = True
        self._running.clear()
        self._pool.waitForDone(milliseconds)
