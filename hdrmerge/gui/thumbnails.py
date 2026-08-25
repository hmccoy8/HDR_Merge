"""Fast thumbnails for the filmstrip.

Decoding a 24MP RAW takes seconds, and a filmstrip that makes you wait seconds
per frame to see which shot is which is useless. Every camera embeds a
ready-made JPEG preview in the RAW file, so the fast path reads that instead
and falls back to a real decode only when it is missing.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from ..loaders import is_raw_path

log = logging.getLogger(__name__)


def load_thumbnail(path: str, longest_edge: int = 320) -> Optional[np.ndarray]:
    """An 8-bit RGB thumbnail of ``path``, or None if it cannot be read.

    Never raises: a filmstrip should show a placeholder for the one file it
    cannot read, not refuse to display the other eight.
    """
    try:
        if is_raw_path(path):
            image = _raw_thumbnail(path)
            if image is None:
                image = _raw_fallback(path)
        else:
            image = _ldr_thumbnail(path, longest_edge)
    except Exception as exc:  # noqa: BLE001 - one bad file must not break the strip
        log.debug("Thumbnail failed for %s: %s", os.path.basename(path), exc)
        return None

    if image is None:
        return None
    return _fit(image, longest_edge)


def _raw_thumbnail(path: str) -> Optional[np.ndarray]:
    """The JPEG the camera already embedded -- milliseconds instead of seconds."""
    import rawpy

    try:
        with rawpy.imread(path) as raw:
            thumb = raw.extract_thumb()
    except Exception as exc:  # noqa: BLE001 - not every RAW carries a preview
        log.debug("No embedded thumbnail in %s: %s", os.path.basename(path), exc)
        return None

    if thumb.format == rawpy.ThumbFormat.JPEG:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(thumb.data)) as img:
            return np.asarray(img.convert("RGB"))
    if thumb.format == rawpy.ThumbFormat.BITMAP:
        return np.asarray(thumb.data)
    return None


def _raw_fallback(path: str) -> Optional[np.ndarray]:
    """Decode the RAW at half size. Slow, so only when there is no preview."""
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(half_size=True, output_bps=8, use_camera_wb=True)
    return np.asarray(rgb)


def _ldr_thumbnail(path: str, longest_edge: int) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        # draft() lets the JPEG decoder skip straight to a smaller size while
        # decoding, which is far cheaper than decoding full size and shrinking.
        try:
            img.draft("RGB", (longest_edge * 2, longest_edge * 2))
        except Exception:  # noqa: BLE001 - only JPEG supports it
            pass
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"))


def _fit(image: np.ndarray, longest_edge: int) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    if scale >= 1.0:
        return np.ascontiguousarray(image)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.ascontiguousarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def to_qimage(image: np.ndarray):
    """Wrap an 8-bit RGB array as a QImage that owns its pixels.

    QImage does not copy the buffer it is handed, so the array has to outlive
    it -- `.copy()` at the end is what stops the widget painting freed memory.
    """
    from PySide6.QtGui import QImage

    array = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    return QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()


def to_qpixmap(image: np.ndarray):
    from PySide6.QtGui import QPixmap

    return QPixmap.fromImage(to_qimage(image))


def float_to_qimage(image: np.ndarray):
    """Convert a display-referred float image in [0, 1] to a QImage."""
    array = np.clip(np.nan_to_num(np.asarray(image, dtype=np.float32)), 0.0, 1.0)
    return to_qimage((array * 255.0).round().astype(np.uint8))
