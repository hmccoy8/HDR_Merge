"""Tone mapping: linear radiance to something a screen can show.

A merged radiance map routinely spans 15+ stops. A display has about 8. Tone
mapping is the compression step in between, and it is where most of the
"HDR look" -- good or bad -- actually comes from.

Every operator here takes linear radiance and returns display-referred,
gamma-encoded values in [0, 1].
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

OPERATORS = ("reinhard", "drago", "mantiuk", "linear", "fusion")

#: ``fusion`` is produced by :func:`hdrmerge.merge.exposure_fusion`, which never
#: builds a radiance map, so it is handled by the pipeline rather than here.
RADIANCE_OPERATORS = ("reinhard", "drago", "mantiuk", "linear")


class TonemapError(RuntimeError):
    """Raised when tone mapping fails or is asked for something impossible."""


def tonemap(
    radiance: np.ndarray,
    operator: str = "reinhard",
    gamma: float = 2.2,
    intensity: float = 0.0,
    light_adapt: float = 0.9,
    colour_adapt: float = 0.0,
    saturation: float = 1.0,
    bias: float = 0.85,
) -> np.ndarray:
    """Compress ``radiance`` to display range.

    Note that every operator here is scale-invariant: each normalises against
    the scene's own brightness, so multiplying the radiance map beforehand
    changes nothing about the result. Exposure is therefore applied afterwards,
    in :func:`hdrmerge.adjust.apply_display`.

    ``intensity``, ``light_adapt`` and ``colour_adapt`` are Reinhard's
    parameters and ``bias`` is Drago's; ``saturation`` is applied by Drago and
    Mantiuk but has no equivalent in OpenCV's Reinhard, so with that operator
    use the ``--saturation`` adjustment instead.
    """
    if operator not in RADIANCE_OPERATORS:
        raise TonemapError(
            f"{operator!r} is not a radiance tone-mapping operator. "
            f"Choose from: {', '.join(RADIANCE_OPERATORS)}"
        )

    radiance = np.nan_to_num(np.asarray(radiance, dtype=np.float32), nan=0.0,
                             posinf=0.0, neginf=0.0)
    radiance = np.maximum(radiance, 0.0)

    scaled = _normalise(radiance)

    if operator == "linear":
        return _linear(scaled, gamma)

    result = _opencv_tonemap(
        scaled, operator, gamma, intensity, light_adapt, colour_adapt,
        saturation, bias,
    )
    if result is None:
        log.warning("%s tone mapping failed; falling back to the linear operator", operator)
        return _linear(scaled, gamma)
    return result


def _opencv_tonemap(scaled, operator, gamma, intensity, light_adapt,
                    colour_adapt, saturation, bias) -> Optional[np.ndarray]:
    import cv2

    bgr = np.ascontiguousarray(scaled[..., ::-1])

    try:
        if operator == "reinhard":
            mapper = cv2.createTonemapReinhard(
                float(gamma), float(intensity), float(light_adapt), float(colour_adapt)
            )
        elif operator == "drago":
            mapper = cv2.createTonemapDrago(float(gamma), float(saturation), float(bias))
        else:
            mapper = cv2.createTonemapMantiuk(float(gamma), 0.85, float(saturation))
        mapped = mapper.process(bgr)
    except cv2.error as exc:
        log.debug("%s tone mapping raised: %s", operator, exc)
        return None

    if mapped is None:
        return None
    mapped = np.asarray(mapped, dtype=np.float32)[..., ::-1]
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)

    if not np.isfinite(mapped).all() or mapped.max() <= 0:
        return None
    return np.clip(mapped, 0.0, 1.0)


def _normalise(radiance: np.ndarray, target: float = 0.18) -> np.ndarray:
    """Put the scene's midtone at 18% grey before an operator sees it.

    OpenCV's operators calibrate against the image's own brightness, so without
    this the same settings would give different results for every bracket.
    """
    values = radiance[radiance > 0]
    if values.size == 0:
        return radiance
    # Geometric mean is the standard log-average luminance; it tracks the
    # midtone rather than being dragged around by specular highlights.
    log_average = float(np.exp(np.mean(np.log(values + 1e-8))))
    if log_average <= 0:
        return radiance
    return (radiance * (target / log_average)).astype(np.float32)


def _linear(scaled: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma encode and clip. No curve at all -- highlights above white blow out.

    This is the operator for someone who wants to grade the result themselves
    and would rather see honestly clipped highlights than a curve they did not
    choose. Expect a lot of clipping on a wide-range scene -- that is the point,
    and ``--exposure`` is how you decide which end to keep.
    """
    exponent = 1.0 / gamma if gamma > 0 else 1.0
    return np.power(np.clip(scaled, 0.0, 1.0), exponent).astype(np.float32)
