"""Linearising rendered (non-RAW) frames.

A JPEG straight out of a camera has been through a tone curve nobody published,
so its pixel values are not proportional to light and cannot be merged as-is.
Recovering that curve -- the camera response function -- is what makes an HDR
merge from JPEGs possible at all.

RAW frames skip this module entirely: LibRaw already handed us linear data.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

log = logging.getLogger(__name__)

METHODS = ("auto", "debevec", "robertson", "srgb", "none")


class CRFError(RuntimeError):
    """Raised when a camera response function cannot be recovered."""


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Inverse of the sRGB transfer function."""
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Forward sRGB transfer function."""
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def choose_method(is_raw: bool, requested: str = "auto") -> str:
    """Pick a linearisation method for the bracket at hand.

    ``auto`` resolves to ``none`` for RAW (already linear) and ``debevec`` for
    rendered frames.
    """
    if requested not in METHODS:
        raise CRFError(f"Unknown CRF method {requested!r}. Choose from: {', '.join(METHODS)}")
    if requested != "auto":
        if is_raw and requested != "none":
            log.warning(
                "RAW frames are already linear; applying %s will distort them. "
                "Use --crf none unless you know why you want this.",
                requested,
            )
        return requested
    return "none" if is_raw else "debevec"


def estimate_response(
    images: Sequence[np.ndarray],
    exposures: Sequence[float],
    method: str = "debevec",
    samples: int = 128,
) -> np.ndarray:
    """Recover a 256-entry per-channel response curve.

    Returns an array of shape (256, 3) mapping 8-bit code value to linear
    radiance. Raises :class:`CRFError` if the estimate does not converge, which
    the pipeline treats as a signal to fall back to exposure fusion.
    """
    import cv2

    if method not in ("debevec", "robertson"):
        raise CRFError(f"{method!r} is not a response-estimation method")

    frames8 = [_to_uint8(image) for image in images]
    times = np.asarray(exposures, dtype=np.float32)

    calibrator = (
        cv2.createCalibrateDebevec(samples=samples)
        if method == "debevec"
        else cv2.createCalibrateRobertson()
    )
    try:
        response = calibrator.process(frames8, times=times)
    except cv2.error as exc:
        raise CRFError(f"{method} calibration failed: {exc}") from exc

    response = np.asarray(response, dtype=np.float32).reshape(256, -1)
    if response.shape[1] == 1:
        response = np.repeat(response, 3, axis=1)

    if not np.all(np.isfinite(response)):
        raise CRFError(f"{method} produced a non-finite response curve")

    # A usable response curve must rise with code value. Debevec in particular
    # can return something wildly non-monotonic when the bracket is short, has
    # little tonal overlap, or has wrong exposure metadata -- better to bail out
    # and let the caller pick a different strategy than to merge with it.
    for channel in range(response.shape[1]):
        column = response[:, channel]
        rising = np.count_nonzero(np.diff(column) >= 0)
        if rising < 0.9 * (len(column) - 1):
            raise CRFError(
                f"{method} produced a non-monotonic response curve "
                f"(channel {channel}); the bracket may have too little overlap "
                "or incorrect exposure metadata"
            )

    return _normalise(response)


def _normalise(response: np.ndarray) -> np.ndarray:
    """Scale the curve so mid-grey maps to mid-grey, keeping values sane."""
    response = np.maximum(response, 0.0)
    midpoint = response[128].mean()
    if midpoint > 0:
        response = response / midpoint * 0.5

    # Nothing in the data constrains code 0: a pixel that is black in every
    # frame says nothing about what radiance produced it, so the estimator is
    # free to return a different value for each channel there. Left alone that
    # difference prints as a colour cast across every region the bracket could
    # not reach. Tie the channels together at that end rather than inventing a
    # hue for black.
    response[0] = response[0].min()
    return response.astype(np.float32)


def apply_response(image: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Map an encoded image through a response curve to linear radiance."""
    codes = np.clip(np.rint(image * 255.0), 0, 255).astype(np.int32)
    linear = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        linear[..., channel] = response[:, channel][codes[..., channel]]
    return linear


def linearize(
    images: Sequence[np.ndarray],
    exposures: Sequence[float],
    method: str,
) -> List[np.ndarray]:
    """Linearise a whole bracket with the chosen method."""
    if method == "none":
        return [np.asarray(image, dtype=np.float32) for image in images]
    if method == "srgb":
        return [srgb_to_linear(image) for image in images]

    response = estimate_response(images, exposures, method=method)
    log.info("Recovered a %s camera response curve", method)
    return [apply_response(image, response) for image in images]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """OpenCV's calibrators only accept 8-bit BGR input."""
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.ascontiguousarray((array[..., ::-1] * 255.0).round().astype(np.uint8))
