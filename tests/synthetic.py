"""Building synthetic brackets with known ground truth.

Every test that checks merge accuracy needs a scene whose true radiance is
known, plus frames rendered from it at known exposures. Doing that here keeps
the tests free of binary fixtures and lets them assert against real numbers
rather than eyeballed tolerances.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hdrmerge.crf import linear_to_srgb


def scene(height: int = 96, width: int = 128, stops: float = 14.0, seed: int = 0) -> np.ndarray:
    """A smooth linear scene spanning ``stops`` of dynamic range.

    Smooth rather than random: a real photograph has spatial structure, and
    alignment and deghosting both depend on it, so noise would not exercise
    them meaningfully.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    # A bright window in a dim room -- the classic case for bracketing.
    gradient = (x / width) * 0.6 + (y / height) * 0.4
    window = np.exp(-(((x - width * 0.75) / (width * 0.13)) ** 2
                      + ((y - height * 0.3) / (height * 0.16)) ** 2))
    # Smoothed noise rather than sinusoids: a periodic texture gives alignment
    # many equally good matches, so a registration test against it would be
    # measuring the fixture's aliasing rather than the aligner.
    texture = _smooth_noise(height, width, rng)

    low, high = 2.0 ** -stops, 1.0
    base = low * (high / low) ** (gradient * 0.55 + texture * 0.45)
    base = base + window * 4.0

    tint = np.array([1.0, 0.92, 0.84], dtype=np.float32)
    image = base[..., None] * tint
    image *= (1.0 + 0.01 * rng.standard_normal(image.shape).astype(np.float32))
    return np.maximum(image, low).astype(np.float32)


def _smooth_noise(height: int, width: int, rng, scale: int = 6) -> np.ndarray:
    """A smooth, non-repeating field in [0, 1] -- stand-in for scene texture."""
    import cv2

    coarse = rng.random((max(2, height // scale), max(2, width // scale))).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), 1.2)
    low, high = float(field.min()), float(field.max())
    return (field - low) / max(high - low, 1e-6)


def render(
    truth: np.ndarray,
    exposure: float,
    encode: bool = False,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render one frame of the bracket at ``exposure``, clipping like a sensor."""
    frame = np.clip(truth * float(exposure), 0.0, 1.0)
    if noise:
        rng = np.random.default_rng(seed)
        frame = np.clip(frame + noise * rng.standard_normal(frame.shape).astype(np.float32), 0.0, 1.0)
    return linear_to_srgb(frame) if encode else frame.astype(np.float32)


def bracket(
    truth: Optional[np.ndarray] = None,
    exposures: Sequence[float] = (0.25, 1.0, 4.0),
    encode: bool = False,
    noise: float = 0.0,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """``(truth, frames, exposures)`` -- everything a merge test needs."""
    truth = scene() if truth is None else truth
    frames = [
        render(truth, exposure, encode=encode, noise=noise, seed=index)
        for index, exposure in enumerate(exposures)
    ]
    return truth, frames, list(exposures)


def write_jpeg_bracket(
    directory: str,
    exposures: Sequence[float] = (1 / 200, 1 / 50, 1 / 12.5),
    start: Optional[_dt.datetime] = None,
    interval: float = 0.4,
    prefix: str = "IMG",
    index_from: int = 1,
    truth: Optional[np.ndarray] = None,
    iso: int = 100,
    f_number: float = 8.0,
) -> List[str]:
    """Write a real JPEG bracket to disk, EXIF and all.

    Exposures are given as shutter speeds so the files carry the same metadata a
    camera would write, which is what the EXIF and grouping paths are tested
    against.
    """
    import piexif
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    truth = scene() if truth is None else truth
    start = start or _dt.datetime(2026, 4, 12, 9, 30, 0)

    # Normalise so the middle shutter speed renders a well-exposed frame.
    gain = 1.0 / (sorted(exposures)[len(exposures) // 2] * 4.0)

    paths: List[str] = []
    for offset, shutter in enumerate(exposures):
        frame = render(truth, shutter * gain, encode=True, seed=offset)
        stamp = start + _dt.timedelta(seconds=interval * offset)
        path = os.path.join(directory, f"{prefix}_{index_from + offset:04d}.jpg")

        exif = piexif.dump({
            "0th": {piexif.ImageIFD.Make: b"HDRMERGE", piexif.ImageIFD.Model: b"Synthetic"},
            "Exif": {
                piexif.ExifIFD.ExposureTime: _ratio(shutter),
                piexif.ExifIFD.FNumber: _ratio(f_number),
                piexif.ExifIFD.ISOSpeedRatings: int(iso),
                piexif.ExifIFD.DateTimeOriginal: stamp.strftime("%Y:%m:%d %H:%M:%S").encode(),
                # DateTimeOriginal only has second resolution, but brackets are
                # shot faster than that, so sub-seconds carry the real spacing.
                piexif.ExifIFD.SubSecTimeOriginal: f"{stamp.microsecond // 10000:02d}".encode(),
            },
        })
        Image.fromarray((frame * 255).round().astype(np.uint8), "RGB").save(
            path, quality=97, subsampling=0, exif=exif
        )
        paths.append(path)
    return paths


def _ratio(value: float, precision: int = 100000) -> Tuple[int, int]:
    """EXIF stores rationals; keep enough precision for fast shutter speeds."""
    from fractions import Fraction

    fraction = Fraction(float(value)).limit_denominator(precision)
    return fraction.numerator, fraction.denominator
