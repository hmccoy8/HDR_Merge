"""The image area: an empty drop target, or the merged result.

One widget with two faces. Before anything is loaded it is the drop zone, which
is where the whole workflow starts, so it has to read as an invitation rather
than an empty panel. Once a merge exists it becomes the preview canvas.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from ..loaders import is_supported
from . import theme
from .thumbnails import float_to_qimage


def paths_from(event) -> List[str]:
    """Supported image paths from a drag event, folders included."""
    if not event.mimeData().hasUrls():
        return []
    found: List[str] = []
    for url in event.mimeData().urls():
        path = url.toLocalFile()
        if not path:
            continue
        if os.path.isdir(path):
            found.append(path)
        elif is_supported(path):
            found.append(path)
    return found


class DropZone(QWidget):
    """The empty state: a dashed target that lights up under a drag."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        self.setAcceptDrops(False)  # the window handles drops for the whole app

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        title = QLabel("Drop your bracketed exposures here")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 17px; font-weight: 500;")

        hint = QLabel(
            "RAW, JPEG, TIFF or PNG — two frames or more.\n"
            "Drop a folder to detect every bracket in a shoot."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setProperty("dim", True)

        self.browse_hint = QLabel("or use  File ▸ Open…")
        self.browse_hint.setAlignment(Qt.AlignCenter)
        self.browse_hint.setProperty("dim", True)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(4)
        layout.addWidget(self.browse_hint)

    def set_hover(self, hover: bool) -> None:
        if hover != self._hover:
            self._hover = hover
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        from PySide6.QtGui import QColor, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        inset = 28
        rect = self.rect().adjusted(inset, inset, -inset, -inset)

        pen = QPen(QColor(theme.ACCENT if self._hover else theme.BORDER))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([6, 5])
        painter.setPen(pen)
        if self._hover:
            painter.setBrush(QColor(74, 158, 255, 18))
        painter.drawRoundedRect(rect, 12, 12)
        painter.end()

        super().paintEvent(event)


class ImageCanvas(QLabel):
    """Shows the merged image, scaled to fit and re-fit on resize."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._source: Optional[QPixmap] = None

    def set_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self._source = None
            self.clear()
            return
        self._source = QPixmap.fromImage(float_to_qimage(image))
        self._rescale()

    def has_image(self) -> bool:
        return self._source is not None

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None:
            return
        # Scale from the original every time rather than from the last scaled
        # copy, or repeated resizes compound the resampling into mush.
        self.setPixmap(
            self._source.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )


class PreviewPane(QStackedWidget):
    """Swaps between the drop zone and the image canvas."""

    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Canvas")
        self.drop_zone = DropZone()
        self.canvas = ImageCanvas()
        self.addWidget(self.drop_zone)
        self.addWidget(self.canvas)

    def show_image(self, image: Optional[np.ndarray]) -> None:
        self.canvas.set_image(image)
        self.setCurrentWidget(self.canvas if image is not None else self.drop_zone)

    def show_empty(self) -> None:
        self.canvas.set_image(None)
        self.setCurrentWidget(self.drop_zone)

    def set_drag_hover(self, hover: bool) -> None:
        self.drop_zone.set_hover(hover)

    @property
    def longest_edge(self) -> int:
        """Preview render size, capped so a huge window does not crawl."""
        return max(320, min(1600, max(self.width(), self.height())))
