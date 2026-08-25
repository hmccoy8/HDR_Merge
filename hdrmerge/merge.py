"""The merge itself: many exposures in, one radiance map out.

For each pixel the estimate is a weighted average of what every frame says the
scene radiance was:

    R = sum_i w(Z_i) * g_i * (L_i / t_i)  /  sum_i w(Z_i) * g_i

where ``L_i`` is the linearised pixel value, ``t_i`` the frame's relative
exposure, ``g_i`` the deghosting weight, and ``w`` a hat function that discounts
values near black (buried in noise) and near saturation (clipped, so they carry
no information about how bright the scene really was).

Doing this well is mostly about the weighting and the degenerate cases, not the
average.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

_EPSILON = 1e-9


class MergeError(RuntimeError):
    """Raised when a bracket cannot be merged."""


@dataclass
class MergeStats:
    """What happened during a merge, for reporting and for tests."""

    frames: int
    reference: int
    dynamic_range_stops: float
    #: Pixels no frame could describe -- clipped or ghost-rejected everywhere.
    fallback_pixels: int
    fallback_fraction: float


def hat_weight(
    encoded: np.ndarray,
    low: float = 0.01,
    high: float = 0.99,
    power: int = 12,
) -> np.ndarray:
    """Confidence in each pixel, from the *encoded* (pre-linearisation) value.

    Broad and flat through the midtones, falling off fast at both ends. The
    weight is computed from the brightest channel rather than per-channel, so a
    pixel with one clipped channel is discounted as a whole -- weighting
    channels independently is what gives naive implementations magenta or cyan
    fringes in blown highlights.
    """
    value = np.asarray(encoded, dtype=np.float32)
    if value.ndim == 3:
        value = value.max(axis=-1)

    weight = 1.0 - np.power(np.clip((value - 0.5) / 0.5, -1.0, 1.0), power)
    weight = np.clip(weight, 0.0, 1.0)
    weight[(value <= low) | (value >= high)] = 0.0
    return weight.astype(np.float32)


def merge_radiance(
    linear: Sequence[np.ndarray],
    encoded: Sequence[np.ndarray],
    exposures: Sequence[float],
    reference: int = 0,
    masks: Optional[Sequence[np.ndarray]] = None,
    saturation_threshold: float = 0.99,
):
    """Merge a bracket into a linear radiance map.

    ``linear`` holds the linearised frames, ``encoded`` the original pixel
    values the weights are derived from (identical to ``linear`` for RAW). The
    result is scaled into the reference frame's exposure units, so a well
    exposed midtone comes back near the value it had in that frame while
    recovered highlights run above 1.0 -- a scale that is still perfectly
    linear, just anchored somewhere intuitive.

    Returns ``(radiance, MergeStats)``.
    """
    if len(linear) != len(encoded) or len(linear) != len(exposures):
        raise MergeError("Frame, encoded-frame and exposure counts must match")
    if len(linear) < 2:
        raise MergeError(f"Need at least 2 frames to merge, got {len(linear)}")
    if not 0 <= reference < len(linear):
        raise MergeError(f"Reference frame {reference} is out of range")
    if any(exposure <= 0 for exposure in exposures):
        raise MergeError("Every frame needs a positive relative exposure")

    shape = linear[0].shape
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape[:2], dtype=np.float32)

    for index, (lin, enc, exposure) in enumerate(zip(linear, encoded, exposures)):
        weight = hat_weight(enc, high=saturation_threshold)
        if masks is not None:
            weight = weight * masks[index]
        numerator += (lin / float(exposure)) * weight[..., None]
        denominator += weight

    empty = denominator <= _EPSILON
    fallback_count = int(np.count_nonzero(empty))

    radiance = numerator / np.maximum(denominator, _EPSILON)[..., None]

    if fallback_count:
        # Every frame was clipped or ghost-rejected here. Leaving these as
        # zero-divided garbage is what produces the black speckle in blown
        # skies; falling back to the frame that best describes each pixel keeps
        # the result continuous.
        radiance[empty] = _fallback(linear, encoded, exposures, reference, empty)
        log.info(
            "%d pixel(s) (%.4f%%) had no usable exposure and fell back to a single frame",
            fallback_count, 100.0 * fallback_count / denominator.size,
        )

    radiance *= float(exposures[reference])
    radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(radiance, 0.0, out=radiance)

    stats = MergeStats(
        frames=len(linear),
        reference=reference,
        dynamic_range_stops=dynamic_range(radiance),
        fallback_pixels=fallback_count,
        fallback_fraction=fallback_count / float(denominator.size),
    )
    return radiance, stats


def _fallback(linear, encoded, exposures, reference, empty):
    """Best single-frame estimate for pixels no weighted average could cover.

    A pixel with nothing usable is either blown in every frame (take the
    darkest, which clipped least) or black in every frame (take the brightest,
    which saw the most). Deciding per pixel handles a bracket containing both.
    """
    order_dark_first = sorted(range(len(linear)), key=lambda i: exposures[i])
    brightest = order_dark_first[-1]
    darkest = order_dark_first[0]

    reference_value = np.asarray(encoded[reference], dtype=np.float32)[empty].max(axis=-1)
    blown = reference_value >= 0.5

    result = np.empty((int(np.count_nonzero(empty)), linear[0].shape[-1]), dtype=np.float32)
    for source, selector in ((darkest, blown), (brightest, ~blown)):
        if np.any(selector):
            values = np.asarray(linear[source], dtype=np.float32)[empty][selector]
            result[selector] = values / float(exposures[source])
    return result


def dynamic_range(radiance: np.ndarray) -> float:
    """Stops between the 1st and 99.9th percentile of non-zero radiance."""
    values = radiance[radiance > 0]
    if values.size == 0:
        return 0.0
    low, high = np.percentile(values, [1.0, 99.9])
    if low <= 0 or high <= 0:
        return 0.0
    return float(np.log2(high / low))


def exposure_fusion(
    encoded: Sequence[np.ndarray],
    contrast: float = 1.0,
    saturation: float = 1.0,
    exposure_weight: float = 1.0,
) -> np.ndarray:
    """Mertens exposure fusion -- straight to a display image, no radiance map.

    Fusion blends the source frames by local contrast, colour saturation and
    how well exposed each pixel is. It never builds a radiance map, so it needs
    no camera response and no exposure metadata at all, which makes it the
    natural fallback when either of those is unavailable or untrustworthy. It
    also tends to simply look good.

    Returns a display-referred image in [0, 1].
    """
    import cv2

    if len(encoded) < 2:
        raise MergeError(f"Need at least 2 frames to fuse, got {len(encoded)}")

    frames8 = [
        np.ascontiguousarray(
            (np.clip(np.asarray(f, dtype=np.float32), 0, 1)[..., ::-1] * 255).round().astype(np.uint8)
        )
        for f in encoded
    ]
    merger = cv2.createMergeMertens(
        float(contrast), float(saturation), float(exposure_weight)
    )
    fused = merger.process(frames8)
    fused = np.clip(np.nan_to_num(np.asarray(fused, dtype=np.float32)), 0.0, 1.0)
    return np.ascontiguousarray(fused[..., ::-1])   # back to RGB
