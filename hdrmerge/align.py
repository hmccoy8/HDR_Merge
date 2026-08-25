"""Aligning handheld brackets.

Frames shot handheld drift by a few pixels between exposures, which shows up in
the merge as soft, doubled edges. Alignment has to work across frames of wildly
different brightness, which rules out ordinary intensity-based registration --
hence median-threshold bitmaps, which compare each frame against its own median
and are therefore exposure-invariant by construction.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

METHODS = ("mtb", "ecc", "none")


class AlignError(RuntimeError):
    """Raised when alignment cannot be performed."""


def align(
    images: Sequence[np.ndarray],
    reference: int,
    method: str = "mtb",
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Align every frame onto ``reference``.

    Returns the aligned frames and the per-frame (dx, dy) shift applied, so the
    caller can report what happened. ``method="none"`` is a no-op for tripod
    sets and is meaningfully faster.
    """
    if method not in METHODS:
        raise AlignError(f"Unknown alignment method {method!r}. Choose from: {', '.join(METHODS)}")

    images = [np.asarray(image, dtype=np.float32) for image in images]
    if method == "none" or len(images) < 2:
        return images, [(0.0, 0.0)] * len(images)

    if method == "mtb":
        return _align_mtb(images, reference)
    return _align_ecc(images, reference)


def _align_mtb(images, reference):
    import cv2

    aligned: List[np.ndarray] = []
    shifts: List[Tuple[float, float]] = []

    height, width = images[reference].shape[:2]
    mtb = cv2.createAlignMTB(_max_bits(height, width))

    ref8 = _to_uint8(images[reference])
    ref_bitmap, ref_mask = _bitmap(ref8)

    for index, image in enumerate(images):
        if index == reference:
            aligned.append(image)
            shifts.append((0.0, 0.0))
            continue

        image8 = _to_uint8(image)
        try:
            shift = mtb.calculateShift(ref8, image8)
            dx, dy = float(shift[0]), float(shift[1])
        except cv2.error as exc:
            log.warning("MTB alignment failed for frame %d (%s); using it unshifted", index, exc)
            dx = dy = 0.0

        dx, dy = _validate(ref_bitmap, ref_mask, image8, dx, dy, index)
        aligned.append(_translate(image, dx, dy))
        shifts.append((dx, dy))

    _log_shifts(shifts, reference)
    return aligned, shifts


def _max_bits(height: int, width: int) -> int:
    """Pyramid depth that still leaves usable pixels at the coarsest level.

    OpenCV's default of 6 assumes a full-size photograph. On anything smaller
    the top of the pyramid collapses to a handful of pixels and calculateShift
    returns garbage rather than failing, so the depth is derived from the image
    instead of assumed.
    """
    smallest = max(1, min(height, width))
    return int(np.clip(np.floor(np.log2(smallest / 6.0)), 1, 6))


def _validate(ref_bitmap, ref_mask, image8, dx, dy, index):
    """Keep a proposed shift only if it genuinely improves frame agreement.

    Median-threshold alignment has several ways of confidently returning a
    wrong answer -- a frame so dark or so blown that its median carries no
    structure, a repeating texture, a pyramid that bottomed out. All of them
    look the same from outside, so rather than trying to detect each, the shift
    is simply measured: if moving the frame does not reduce disagreement with
    the reference, it is discarded.
    """
    if dx == 0 and dy == 0:
        return 0.0, 0.0

    before = _mismatch(ref_bitmap, ref_mask, image8, 0.0, 0.0)
    after = _mismatch(ref_bitmap, ref_mask, image8, dx, dy)
    if after < before:
        return dx, dy

    log.info(
        "Rejected an implausible alignment shift for frame %d "
        "(%+.0f,%+.0f would have made agreement worse: %.3f -> %.3f)",
        index, dx, dy, before, after,
    )
    return 0.0, 0.0


def _bitmap(image8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ward's median-threshold bitmap and its exclusion mask.

    Thresholding at the frame's own median is what makes the comparison
    exposure-invariant; the mask drops pixels close to the median, whose side
    of the threshold is decided by noise rather than by scene content.
    """
    median = float(np.median(image8))
    bitmap = image8 > median
    mask = np.abs(image8.astype(np.int16) - median) > 4
    return bitmap, mask


def _mismatch(ref_bitmap, ref_mask, image8, dx, dy) -> float:
    """Fraction of confidently-different pixels between two frames at a shift."""
    import cv2

    shifted = image8 if (dx == 0 and dy == 0) else _translate(image8, dx, dy)
    bitmap, mask = _bitmap(np.asarray(shifted, dtype=np.uint8))

    both = ref_mask & mask
    total = int(np.count_nonzero(both))
    if total == 0:
        return 1.0
    return float(np.count_nonzero((ref_bitmap != bitmap) & both)) / total


def _align_ecc(images, reference):
    """Refine with an intensity-based Euclidean fit, catching slight rotation.

    Runs on top of MTB rather than instead of it: ECC needs a good starting
    point, and MTB provides one that is exposure-invariant.
    """
    import cv2

    images, shifts = _align_mtb(images, reference)
    ref_gray = _to_gray(images[reference])
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)

    refined: List[np.ndarray] = []
    for index, image in enumerate(images):
        if index == reference:
            refined.append(image)
            continue
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_gray, _to_gray(image), warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
            )
            refined.append(
                cv2.warpAffine(
                    image, warp, (image.shape[1], image.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            )
        except cv2.error as exc:
            # ECC diverges on low-texture or heavily-clipped frames. The MTB
            # result is still valid, so keep it rather than failing the merge.
            log.warning("ECC refinement did not converge for frame %d (%s)", index, exc)
            refined.append(image)

    return refined, shifts


def _translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    import cv2

    if dx == 0 and dy == 0:
        return image
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    interpolation = cv2.INTER_NEAREST if image.dtype == np.uint8 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        flags=interpolation, borderMode=cv2.BORDER_REPLICATE,
    )


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Single-channel 8-bit view, which is what AlignMTB.calculateShift needs.

    Gamma-encoded first so shadow detail survives the 8-bit quantisation --
    median-threshold bitmaps split at the median, and a linear encoding pushes
    almost every pixel of a dark frame into the bottom few code values.
    """
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if array.ndim == 3:
        array = array.mean(axis=-1, dtype=np.float32)
    array = np.power(array, 1.0 / 2.2)

    # Stretch each frame across the full 8-bit range before comparing. The
    # darkest frame of a bracket otherwise occupies only the bottom few code
    # values, leaving the median-threshold bitmap with almost no structure to
    # match against.
    low, high = np.percentile(array, [1.0, 99.0])
    if high - low < 1e-3:
        low, high = float(array.min()), max(float(array.max()), float(array.min()) + 1e-3)
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.ascontiguousarray((array * 255.0).round().astype(np.uint8))


def _to_gray(image: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    array = np.power(array, 1.0 / 2.2)
    return np.ascontiguousarray(array.mean(axis=-1, dtype=np.float32))


def _log_shifts(shifts, reference):
    moved = [
        f"#{i}:({dx:+.0f},{dy:+.0f})"
        for i, (dx, dy) in enumerate(shifts)
        if i != reference and (dx or dy)
    ]
    if moved:
        log.info("Alignment shifted %d frame(s): %s", len(moved), " ".join(moved))
    else:
        log.info("Alignment found no drift between frames")
