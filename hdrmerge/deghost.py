"""Suppressing ghosts from subject motion.

Anything that moved between frames -- leaves, water, people, cars -- appears in
a different place in each exposure, and averaging them together smears it into a
translucent streak. The fix used here is the reference-frame approach: pick one
frame as the truth, and for every other frame reject the pixels that disagree
with it once exposure is accounted for. A moving subject then resolves to its
position in the reference frame instead of a blend of all its positions.

The cost is that the moving region ends up with the dynamic range of a single
frame. That is the right trade -- a slightly noisier moving subject beats a
ghosted one -- and it is why the reference defaults to the middle exposure.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

log = logging.getLogger(__name__)

_EPSILON = 1e-6


def ghost_masks(
    linear: Sequence[np.ndarray],
    exposures: Sequence[float],
    reference: int,
    threshold: float = 0.7,
    feather: int = 5,
) -> List[np.ndarray]:
    """Per-frame weights in [0, 1]; low where a frame disagrees with the reference.

    ``threshold`` is measured in stops of disagreement: a pixel differing from
    the reference by more than this many stops is rejected outright, and the
    transition below it is smooth so masks do not print visible edges.
    """
    if threshold <= 0:
        return [np.ones(image.shape[:2], dtype=np.float32) for image in linear]

    ref_radiance = _radiance(linear[reference], exposures[reference])
    ref_log = np.log2(ref_radiance + _EPSILON)

    # Where the reference itself is clipped or buried in noise it cannot judge
    # anything, so those pixels are exempt and every frame is trusted there.
    ref_encoded = np.asarray(linear[reference], dtype=np.float32).max(axis=-1)
    ref_usable = ((ref_encoded > 0.02) & (ref_encoded < 0.98)).astype(np.float32)

    masks: List[np.ndarray] = []
    for index, image in enumerate(linear):
        if index == reference:
            masks.append(np.ones(image.shape[:2], dtype=np.float32))
            continue

        difference = np.abs(np.log2(_radiance(image, exposures[index]) + _EPSILON) - ref_log)
        difference = difference.max(axis=-1)     # any channel disagreeing is a ghost

        # 1.0 below the threshold, ramping to 0.0 by twice it.
        mask = np.clip(2.0 - difference / threshold, 0.0, 1.0).astype(np.float32)
        mask = np.maximum(mask, 1.0 - ref_usable)
        masks.append(_clean(mask, feather))

    _log_coverage(masks, reference)
    return masks


def _radiance(image: np.ndarray, exposure: float) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / max(float(exposure), _EPSILON)


def _clean(mask: np.ndarray, feather: int) -> np.ndarray:
    """Drop isolated speckle, then soften the edges of what remains."""
    import cv2

    if feather <= 0:
        return mask

    binary = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove noise specks
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill pinholes

    mask = np.minimum(mask, binary.astype(np.float32))
    radius = feather * 2 + 1
    return cv2.GaussianBlur(mask, (radius, radius), 0).astype(np.float32)


def _log_coverage(masks, reference):
    for index, mask in enumerate(masks):
        if index == reference:
            continue
        rejected = float(np.mean(mask < 0.5))
        if rejected > 0.001:
            log.info("Deghosting rejected %.1f%% of frame %d", rejected * 100.0, index)
        if rejected > 0.5:
            log.warning(
                "Frame %d disagrees with the reference across %.0f%% of the image. "
                "Check that alignment worked and that the exposures are correct; "
                "use --deghost-threshold to loosen this, or --deghost none.",
                index, rejected * 100.0,
            )
