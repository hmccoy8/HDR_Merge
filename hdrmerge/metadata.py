"""Exposure metadata: read it from EXIF, or recover it from the pixels.

Merging a bracket only ever needs the *ratios* between frame exposures, never
their absolute values, so everything here is expressed as a unitless
``relative_exposure`` where a frame twice as bright as another has twice the
value.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Where a frame's exposure figure came from.
SOURCE_EXIF = "exif"
SOURCE_ESTIMATED = "estimated"
SOURCE_UNKNOWN = "unknown"


@dataclass
class FrameMeta:
    """Everything we know about one frame of a bracket."""

    path: str
    exposure_time: Optional[float] = None   # seconds
    f_number: Optional[float] = None
    iso: Optional[float] = None
    timestamp: Optional[_dt.datetime] = None
    relative_exposure: Optional[float] = None
    source: str = SOURCE_UNKNOWN

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def ev(self) -> Optional[float]:
        """Exposure value. Lower EV means a brighter (more exposed) frame."""
        if self.relative_exposure is None or self.relative_exposure <= 0:
            return None
        return -math.log2(self.relative_exposure)

    def describe(self) -> str:
        parts = []
        if self.exposure_time:
            parts.append(
                f"1/{1 / self.exposure_time:.0f}s" if self.exposure_time < 1
                else f"{self.exposure_time:g}s"
            )
        if self.f_number:
            parts.append(f"f/{self.f_number:g}")
        if self.iso:
            parts.append(f"ISO{self.iso:g}")
        return " ".join(parts) if parts else "-"


def _to_float(value) -> Optional[float]:
    """Coerce an exifread tag value (Ratio, int, list, str) to a float."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _to_float(value[0])
    for attr in ("num", "numerator"):
        if hasattr(value, attr):
            den = getattr(value, "den", None) or getattr(value, "denominator", None)
            try:
                den = float(den)
                if den == 0:
                    return None
                return float(getattr(value, attr)) / den
            except (TypeError, ValueError):
                return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if "/" in text:
        head, _, tail = text.partition("/")
        try:
            den = float(tail)
            return float(head) / den if den else None
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_timestamp(date_text, subsec_text) -> Optional[_dt.datetime]:
    if not date_text:
        return None
    try:
        stamp = _dt.datetime.strptime(str(date_text).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    if subsec_text:
        digits = "".join(ch for ch in str(subsec_text) if ch.isdigit())
        if digits:
            stamp = stamp.replace(microsecond=int(f"{digits:0<6.6}"))
    return stamp


def read_exif(path: str) -> FrameMeta:
    """Read exposure metadata from ``path``.

    Never raises: a file whose EXIF is missing, stripped, or in a container
    exifread cannot parse (notably Canon CR3, which is not TIFF-based) comes
    back with ``source == SOURCE_UNKNOWN`` and the pixel-based estimator in
    :func:`estimate_relative_exposures` takes over.
    """
    meta = FrameMeta(path=path)
    try:
        import exifread
    except ImportError:  # pragma: no cover - exifread is a hard dependency
        log.warning("exifread is not installed; falling back to pixel estimation")
        return meta

    try:
        with open(path, "rb") as handle:
            tags = exifread.process_file(handle, details=False)
    except Exception as exc:  # noqa: BLE001 - any read failure is non-fatal here
        log.debug("EXIF read failed for %s: %s", path, exc)
        return meta

    def tag(*names):
        for name in names:
            if name in tags:
                return tags[name].values if hasattr(tags[name], "values") else tags[name]
        return None

    meta.exposure_time = _to_float(tag("EXIF ExposureTime", "Image ExposureTime"))
    meta.f_number = _to_float(tag("EXIF FNumber", "Image FNumber"))
    meta.iso = _to_float(
        tag(
            "EXIF ISOSpeedRatings",
            "EXIF PhotographicSensitivity",
            "Image ISOSpeedRatings",
            "MakerNote ISOSetting",
        )
    )
    meta.timestamp = _parse_timestamp(
        tag("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"),
        tag("EXIF SubSecTimeOriginal", "EXIF SubSecTime"),
    )

    rel = relative_exposure(meta.exposure_time, meta.f_number, meta.iso)
    if rel is not None:
        meta.relative_exposure = rel
        meta.source = SOURCE_EXIF
    return meta


def relative_exposure(
    exposure_time: Optional[float],
    f_number: Optional[float],
    iso: Optional[float],
) -> Optional[float]:
    """``t * (ISO/100) / N^2`` -- proportional to the light each frame received.

    Shutter speed alone is enough for the overwhelming majority of brackets
    (cameras hold aperture and ISO fixed when bracketing), so aperture and ISO
    are treated as optional refinements rather than requirements.
    """
    if not exposure_time or exposure_time <= 0:
        return None
    value = float(exposure_time)
    if iso and iso > 0:
        value *= iso / 100.0
    if f_number and f_number > 0:
        value /= f_number ** 2
    return value


def estimate_relative_exposures(
    images: Sequence[np.ndarray],
    low: float = 0.05,
    high: float = 0.9,
    linear: bool = True,
) -> List[float]:
    """Recover relative exposures from pixel data alone.

    For each adjacent pair of frames we take the pixels that are well exposed
    in *both* and use the median of their per-pixel ratios as the exposure step
    between them; chaining those steps gives a full relative scale. The median
    is what makes this robust -- noise, clipping and moving subjects all
    perturb individual pixels, but not the middle of the distribution.

    ``images`` must be in capture order. Pass ``linear=False`` for rendered
    frames: brightness ratios are only proportional to exposure ratios in
    linear light, so gamma-encoded input is undone with the sRGB curve first.
    That is an assumption rather than a measurement -- the real response curve
    cannot be recovered without the exposures we are trying to estimate -- but
    it is close enough for every camera JPEG in practice, and far closer than
    ignoring the encoding altogether.
    """
    if not images:
        return []

    if not linear:
        from .crf import srgb_to_linear

        images = [srgb_to_linear(image) for image in images]

    scale = [1.0]
    for previous, current in zip(images[:-1], images[1:]):
        ratio = _pairwise_ratio(previous, current, low, high)
        scale.append(scale[-1] * ratio)

    # Normalise to the geometric mean so the numbers stay near 1.0 for long
    # brackets rather than drifting off to 2^-8.
    centre = float(np.exp(np.mean(np.log(scale))))
    return [value / centre for value in scale]


def _pairwise_ratio(a: np.ndarray, b: np.ndarray, low: float, high: float) -> float:
    """Median brightness ratio b/a over pixels well exposed in both frames."""
    a = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
    b = b.reshape(-1, b.shape[-1]) if b.ndim == 3 else b.reshape(-1, 1)

    # Subsample large frames; a few hundred thousand pixels pin the median down
    # far more precisely than we need, and this keeps the estimate fast.
    stride = max(1, a.shape[0] // 200_000)
    a = a[::stride]
    b = b[::stride]

    usable = (a > low) & (a < high) & (b > low) & (b < high)
    ratios = b[usable] / a[usable]
    if ratios.size < 64:
        # Not enough overlap between the two frames to say anything; assume the
        # bracket's most common step so the chain does not collapse.
        log.warning("Too little tonal overlap to estimate an exposure step; assuming 2 EV")
        return 4.0
    return float(np.median(ratios))


def resolve_exposures(
    metas: Sequence[FrameMeta],
    images: Optional[Sequence[np.ndarray]] = None,
    images_are_linear: bool = True,
) -> List[FrameMeta]:
    """Fill in every frame's ``relative_exposure``, preferring EXIF.

    EXIF is used whenever *all* frames have it -- mixing measured and estimated
    exposures in one bracket would put them on two different scales. Otherwise
    the whole bracket is estimated from pixels together.
    """
    metas = list(metas)
    if all(m.relative_exposure for m in metas):
        return metas

    missing = [m.name for m in metas if not m.relative_exposure]
    if images is None:
        raise ValueError(
            "No exposure metadata for "
            + ", ".join(missing)
            + " and no pixel data supplied to estimate it from"
        )

    log.info(
        "Estimating exposures from pixel data (no usable EXIF for: %s)",
        ", ".join(missing),
    )
    estimated = estimate_relative_exposures(images, linear=images_are_linear)
    for meta, value in zip(metas, estimated):
        meta.relative_exposure = value
        meta.source = SOURCE_ESTIMATED
    return metas
