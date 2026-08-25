"""Reading bracket frames into float arrays.

Two classes of input, handled differently for a reason that matters:

* **Camera RAW** is linear -- a sensor value is proportional to the light that
  hit the photosite. LibRaw is asked for a demosaiced 16-bit image with no
  gamma and no auto-brightening, and the result can be merged directly.
* **JPEG/TIFF/PNG** have been through an unknown camera tone curve. They must
  be linearised (see :mod:`hdrmerge.crf`) before the same merge maths applies.

Mixing the two in one bracket is rejected rather than silently averaged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .metadata import FrameMeta, read_exif

log = logging.getLogger(__name__)

#: Extensions LibRaw handles. Not exhaustive -- anything LibRaw opens works,
#: this list only decides which reader to try first.
RAW_EXTENSIONS = frozenset(
    """.3fr .arw .cr2 .cr3 .crw .dcr .dng .erf .fff .iiq .k25 .kdc .mef .mos
       .mrw .nef .nrw .orf .pef .raf .raw .rw2 .rwl .sr2 .srf .srw .x3f""".split()
)

LDR_EXTENSIONS = frozenset(
    ".jpg .jpeg .jpe .png .tif .tiff .webp .bmp".split()
)

SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | LDR_EXTENSIONS


class LoadError(RuntimeError):
    """Raised when a frame cannot be read, or a bracket is inconsistent."""


@dataclass
class Frame:
    """One loaded bracket frame."""

    meta: FrameMeta
    #: Float32 RGB in [0, 1]. Linear for RAW, still tone-curve encoded for LDR.
    data: np.ndarray
    is_raw: bool

    @property
    def is_linear(self) -> bool:
        return self.is_raw

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]


def is_raw_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def load_frame(path: str, preview_scale: float = 1.0) -> Frame:
    """Load a single frame as float32 RGB in [0, 1], with its metadata."""
    if not os.path.isfile(path):
        raise LoadError(f"No such file: {path}")

    raw = is_raw_path(path)
    try:
        data = _load_raw(path) if raw else _load_ldr(path)
    except LoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced with the filename attached
        raise LoadError(f"Could not read {os.path.basename(path)}: {exc}") from exc

    if preview_scale and preview_scale != 1.0:
        data = _downscale(data, preview_scale)

    return Frame(meta=read_exif(path), data=data, is_raw=raw)


def _load_raw(path: str) -> np.ndarray:
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),              # keep the data linear
            no_auto_bright=True,       # never rescale -- it would destroy the ratios
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            highlight_mode=rawpy.HighlightMode.Clip,
        )
    return np.asarray(rgb, dtype=np.float32) / 65535.0


#: Formats that can carry more than 8 bits per channel. Pillow cannot
#: represent 16-bit RGB and silently hands back 8-bit data for these, throwing
#: away exactly the shadow detail that makes deep files worth using, so they go
#: to readers that preserve the depth.
DEEP_EXTENSIONS = frozenset(".tif .tiff .png".split())


def _load_ldr(path: str) -> np.ndarray:
    extension = os.path.splitext(path)[1].lower()
    array = (
        _read_deep(path, extension)
        if extension in DEEP_EXTENSIONS
        else _read_with_pillow(path)
    )

    if array is None:
        array = _read_with_pillow(path)

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]

    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype in (np.uint16, np.int32, np.uint32):
        return np.clip(array.astype(np.float32) / 65535.0, 0.0, 1.0)
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array.astype(np.float32), 0.0, 1.0)
    raise LoadError(f"Unsupported pixel format {array.dtype} in {os.path.basename(path)}")


def _read_deep(path: str, extension: str):
    """Read a TIFF or PNG at its real bit depth. Returns None to fall back."""
    try:
        if extension in (".tif", ".tiff"):
            import tifffile

            return np.asarray(tifffile.imread(path))

        import cv2

        array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if array is None:
            return None
        if array.ndim == 2:
            return np.asarray(array)
        # OpenCV gives BGR, or BGRA when there is an alpha channel. Drop alpha
        # before reversing -- reversing all four channels first would leave the
        # alpha where red belongs.
        return np.asarray(array[..., :3][..., ::-1])
    except Exception as exc:  # noqa: BLE001 - Pillow gets a turn before we give up
        log.debug("Deep read of %s failed (%s); falling back to Pillow", path, exc)
        return None


def _read_with_pillow(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        img = _apply_exif_orientation(img)
        if img.mode not in ("RGB", "I;16", "I;16B", "I;16L"):
            img = img.convert("RGB")
        return np.asarray(img)


def _apply_exif_orientation(img):
    """Rotate per the EXIF orientation tag so a bracket lines up pixel-for-pixel."""
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - orientation is a nicety, not a requirement
        return img


def _downscale(data: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    if scale <= 0 or scale > 1:
        raise LoadError(f"preview-scale must be in (0, 1], got {scale}")
    height = max(1, int(round(data.shape[0] * scale)))
    width = max(1, int(round(data.shape[1] * scale)))
    return cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA)


def load_bracket(
    paths: Sequence[str],
    preview_scale: float = 1.0,
    max_frames: int = 15,
) -> List[Frame]:
    """Load and validate a whole bracket.

    Enforces the guards that keep a merge meaningful: at least two frames, a
    frame count that will actually fit in memory, matching dimensions, and a
    single input class throughout.
    """
    paths = list(paths)
    if len(paths) < 2:
        raise LoadError(
            f"A bracket needs at least 2 frames, got {len(paths)}. "
            "Pass every exposure of the bracket."
        )
    if max_frames and len(paths) > max_frames:
        raise LoadError(
            f"{len(paths)} frames exceeds the --max-frames limit of {max_frames}. "
            "A 24MP frame costs roughly 300MB in memory; raise the limit only if "
            "you have the RAM for it, or use --preview-scale to work smaller."
        )

    frames = [load_frame(path, preview_scale) for path in paths]

    kinds = {frame.is_raw for frame in frames}
    if len(kinds) > 1:
        raw = [f.meta.name for f in frames if f.is_raw]
        ldr = [f.meta.name for f in frames if not f.is_raw]
        raise LoadError(
            "Cannot mix RAW and rendered images in one bracket -- RAW is linear "
            "and JPEG/TIFF/PNG are tone-curve encoded, so they merge on different "
            f"scales.\n  RAW: {', '.join(raw)}\n  rendered: {', '.join(ldr)}"
        )

    shapes = {frame.shape for frame in frames}
    if len(shapes) > 1:
        detail = ", ".join(f"{f.meta.name}={f.shape[1]}x{f.shape[0]}" for f in frames)
        raise LoadError(f"All frames must have the same dimensions. Got: {detail}")

    return frames


def expand_inputs(paths: Sequence[str]) -> List[str]:
    """Turn a mix of files and directories into a sorted list of image files."""
    found: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            entries = [
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if is_supported(os.path.join(path, name))
            ]
            if not entries:
                log.warning("No supported images found in %s", path)
            found.extend(entries)
        elif os.path.isfile(path):
            if not is_supported(path):
                raise LoadError(
                    f"Unsupported file type: {os.path.basename(path)}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            found.append(path)
        else:
            raise LoadError(f"No such file or directory: {path}")
    if not found:
        raise LoadError("No input images found")
    return found
