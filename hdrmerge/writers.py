"""Writing merged results out.

Formats split into two groups, and the difference is not cosmetic:

* **Display formats** (jpg, png16, tif16) hold tone-mapped, gamma-encoded
  pixels in [0, 1] and are what you look at or hand to an editor.
* **Radiance formats** (exr, hdr, tif32) hold the linear 32-bit merge result
  with no tone mapping at all -- the whole point of them is that the highlight
  and shadow data survives for grading elsewhere.

The pipeline never tone maps a radiance output, and never writes unbounded
linear data into a display format.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

#: format key -> (file extension, is-radiance, human description)
FORMATS: Dict[str, tuple] = {
    "jpg":   (".jpg",  False, "8-bit JPEG, tone-mapped, full resolution"),
    "png16": (".png",  False, "16-bit PNG, tone-mapped, lossless"),
    "tif16": (".tif",  False, "16-bit TIFF, tone-mapped, for Lightroom/Photoshop"),
    "tif32": (".tif",  True,  "32-bit float TIFF, linear radiance"),
    "exr":   (".exr",  True,  "OpenEXR, 32-bit float linear radiance"),
    "hdr":   (".hdr",  True,  "Radiance RGBE, linear"),
}

RADIANCE_FORMATS = frozenset(key for key, spec in FORMATS.items() if spec[1])
DISPLAY_FORMATS = frozenset(key for key, spec in FORMATS.items() if not spec[1])

#: Formats whose backing library is optional, and the module each one needs.
#: Only EXR is optional today -- OpenEXR publishes no wheel for Python 3.14, and
#: making that take down the whole install would be absurd for one writer among
#: six. Written as a table so a future optional writer slots in without new
#: plumbing.
OPTIONAL_BACKENDS = {"exr": "OpenEXR"}

_EXR_HELP = (
    'Writing EXR needs the OpenEXR package, which has no wheel for Python 3.14 '
    'yet, so pip cannot install it there without CMake and a C++ compiler.\n'
    '  - Install it anyway:  pip install "hdrmerge[exr]"  (needs that toolchain)\n'
    "  - Or use --format tif32 or --format hdr: both hold the same 32-bit "
    "linear radiance and need nothing extra."
)


class WriteError(RuntimeError):
    """Raised when an output file cannot be produced."""


def unavailable_reason(fmt: str) -> Optional[str]:
    """Why ``fmt`` cannot be written here, or None if it can.

    Checked up front rather than at write time so a long batch fails in the
    first second instead of after every bracket has been merged.
    """
    _check(fmt)
    module = OPTIONAL_BACKENDS.get(fmt)
    if module is None:
        return None
    try:
        importlib.import_module(module)
    except ImportError:
        return _EXR_HELP if fmt == "exr" else f"Writing {fmt} needs the {module} package"
    return None


def available(fmt: str) -> bool:
    """Whether ``fmt`` can actually be written in this environment."""
    return unavailable_reason(fmt) is None


def is_radiance(fmt: str) -> bool:
    _check(fmt)
    return FORMATS[fmt][1]


def extension(fmt: str) -> str:
    _check(fmt)
    return FORMATS[fmt][0]


def _check(fmt: str) -> None:
    if fmt not in FORMATS:
        raise WriteError(
            f"Unknown output format {fmt!r}. Choose from: {', '.join(sorted(FORMATS))}"
        )


def output_path(base: str, fmt: str, out_dir: Optional[str] = None,
                suffix: str = "_hdr") -> str:
    """Build ``<dir>/<base><suffix>.<ext>``, disambiguating tif16 from tif32."""
    _check(fmt)
    stem = os.path.splitext(os.path.basename(base))[0] + suffix
    ext = extension(fmt)
    if fmt == "tif32":
        # tif16 and tif32 share an extension, so the linear one is marked.
        stem += "_linear"
    directory = out_dir if out_dir else os.path.dirname(os.path.abspath(base))
    return os.path.join(directory, f"{stem}{ext}")


def write(image: np.ndarray, path: str, fmt: str, jpeg_quality: int = 95) -> str:
    """Write ``image`` to ``path`` in ``fmt`` and return the path written."""
    _check(fmt)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise WriteError(f"Expected an HxWx3 image, got shape {image.shape}")

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    if is_radiance(fmt):
        # Radiance data is unbounded above but must not be negative or NaN --
        # both make downstream tools misbehave in confusing ways.
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.maximum(image, 0.0)
    else:
        image = np.clip(np.nan_to_num(image, nan=0.0), 0.0, 1.0)

    writer = {
        "jpg": _write_jpeg,
        "png16": _write_png16,
        "tif16": _write_tif16,
        "tif32": _write_tif32,
        "exr": _write_exr,
        "hdr": _write_hdr,
    }[fmt]
    writer(image, path, jpeg_quality)
    return path


def _write_jpeg(image: np.ndarray, path: str, quality: int) -> None:
    from PIL import Image

    array = np.rint(image * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(
        path,
        format="JPEG",
        quality=int(quality),
        subsampling=0,       # 4:4:4 -- no chroma loss on a final deliverable
        optimize=True,
        icc_profile=_srgb_profile(),
    )


def _write_png16(image: np.ndarray, path: str, quality: int) -> None:
    import cv2

    # Pillow cannot write 16-bit RGB PNG; OpenCV can. It expects BGR.
    array = np.rint(image * 65535.0).astype(np.uint16)[..., ::-1]
    if not cv2.imwrite(path, array):
        raise WriteError(f"Failed to write {path}")


def _write_tif16(image: np.ndarray, path: str, quality: int) -> None:
    import tifffile

    array = np.rint(image * 65535.0).astype(np.uint16)
    tifffile.imwrite(path, array, photometric="rgb", compression="adobe_deflate")


def _write_tif32(image: np.ndarray, path: str, quality: int) -> None:
    import tifffile

    tifffile.imwrite(path, image, photometric="rgb", compression="adobe_deflate")


def _write_exr(image: np.ndarray, path: str, quality: int) -> None:
    # OpenCV wheels are frequently built without OpenEXR support (the headless
    # builds in particular report "OpenEXR: NO"), so the dedicated bindings are
    # used instead of cv2.imwrite here.
    try:
        import OpenEXR
    except ImportError as exc:
        raise WriteError(_EXR_HELP) from exc

    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    exr = OpenEXR.File(header, {"RGB": np.ascontiguousarray(image, dtype=np.float32)})
    exr.write(path)


def _write_hdr(image: np.ndarray, path: str, quality: int) -> None:
    import cv2

    if not cv2.imwrite(path, np.ascontiguousarray(image[..., ::-1])):
        raise WriteError(f"Failed to write {path}")


_SRGB_PROFILE: Optional[bytes] = None


def _srgb_profile() -> Optional[bytes]:
    """sRGB ICC profile bytes, so JPEGs open with the right colours elsewhere."""
    global _SRGB_PROFILE
    if _SRGB_PROFILE is None:
        try:
            from PIL import ImageCms

            _SRGB_PROFILE = ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes()
        except Exception as exc:  # noqa: BLE001 - profile is optional metadata
            log.debug("Could not build an sRGB ICC profile: %s", exc)
            _SRGB_PROFILE = b""
    return _SRGB_PROFILE or None
