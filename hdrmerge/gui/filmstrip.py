"""The frame list: which photo is which, and what exposure it was.

This is the GUI's answer to `hdrmerge inspect`. Each row carries the thumbnail
plus the two things that decide whether a merge will work -- the camera
settings and the detected EV -- and says whether the EV came from EXIF or was
estimated from the pixels, because that distinction is exactly what you want to
check when a merge looks wrong.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..metadata import SOURCE_EXIF, FrameMeta
from . import theme
from .thumbnails import to_qpixmap

THUMB_WIDTH = 104
THUMB_HEIGHT = 70


class FrameRow(QWidget):
    """One frame: thumbnail, filename, settings, EV."""

    def __init__(self, meta: FrameMeta, thumbnail: Optional[np.ndarray],
                 is_reference: bool = False, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 8, 6)
        layout.setSpacing(10)

        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setPixmap(_thumb_pixmap(thumbnail))
        layout.addWidget(self.thumb)

        text = QVBoxLayout()
        text.setSpacing(1)
        text.setContentsMargins(0, 2, 0, 2)

        name = QLabel(meta.name)
        name.setStyleSheet("font-weight: 500;")
        name.setToolTip(meta.path)

        settings = QLabel(meta.describe())
        settings.setProperty("dim", True)

        text.addWidget(name)
        text.addWidget(settings)
        text.addWidget(_ev_label(meta))
        text.addStretch()
        layout.addLayout(text, 1)

        if is_reference:
            badge = QLabel("REF")
            badge.setToolTip(
                "Reference frame: alignment and deghosting are measured against "
                "this one. It defaults to the middle exposure."
            )
            badge.setStyleSheet(
                f"color: {theme.ACCENT}; font-size: 10px; font-weight: 700;"
                f" letter-spacing: 1px;"
            )
            badge.setAlignment(Qt.AlignTop | Qt.AlignRight)
            layout.addWidget(badge)


def _ev_label(meta: FrameMeta) -> QLabel:
    if meta.ev is None:
        label = QLabel("exposure unknown")
        label.setStyleSheet("color: #d08770; font-size: 12px;")
        return label

    estimated = meta.source != SOURCE_EXIF
    label = QLabel(f"EV {meta.ev:+.2f}" + ("  · estimated" if estimated else ""))
    # Estimated exposures are worth a second look, so they are not styled the
    # same as measured ones -- this is the number a bad merge usually traces to.
    colour = "#d0a070" if estimated else theme.TEXT_DIM
    label.setStyleSheet(f"color: {colour}; font-size: 12px;")
    label.setToolTip(
        "Estimated from pixel brightness because this file carries no usable "
        "EXIF exposure data." if estimated else "Read from the file's EXIF data."
    )
    return label


def _thumb_pixmap(thumbnail: Optional[np.ndarray]) -> QPixmap:
    if thumbnail is None:
        pixmap = QPixmap(THUMB_WIDTH, THUMB_HEIGHT)
        pixmap.fill(QColor(theme.SURFACE_RAISED))
        painter = QPainter(pixmap)
        painter.setPen(QColor(theme.TEXT_DIM))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "no preview")
        painter.end()
        return pixmap

    pixmap = to_qpixmap(thumbnail)
    return pixmap.scaled(
        THUMB_WIDTH, THUMB_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


class Filmstrip(QListWidget):
    """The loaded frames, darkest first."""

    reference_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setUniformItemSizes(False)
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._metas: List[FrameMeta] = []

    def set_frames(
        self,
        metas: Sequence[FrameMeta],
        thumbnails: Sequence[Optional[np.ndarray]],
        reference: int = -1,
    ) -> None:
        self.clear()
        self._metas = list(metas)

        for index, meta in enumerate(metas):
            thumbnail = thumbnails[index] if index < len(thumbnails) else None
            row = FrameRow(meta, thumbnail, is_reference=(index == reference))
            item = QListWidgetItem(self)
            item.setSizeHint(QSize(0, THUMB_HEIGHT + 18))
            item.setToolTip(meta.path)
            self.addItem(item)
            self.setItemWidget(item, row)

    def clear_frames(self) -> None:
        self.clear()
        self._metas = []

    @property
    def metas(self) -> List[FrameMeta]:
        return list(self._metas)
