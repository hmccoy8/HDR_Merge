"""Core post-processing adjustments.

Split across the tone-mapping boundary, because where an adjustment belongs
changes what it does:

* **Linear adjustments** (exposure, white balance) run on the radiance map
  before tone mapping. Exposure applied to linear data is a genuine change in
  how much light the image represents, so the tone curve reacts to it the way
  it would to a differently-exposed capture.
* **Display adjustments** (highlights, shadows, contrast, colour) run after,
  on [0, 1] values, because they are shaping an image somebody is looking at.

Everything defaults to a no-op, so specifying nothing changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Tuple

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709


@dataclass
class Adjustments:
    """Every adjustment, in the units a photographer expects.

    ``exposure`` is in stops; ``temp`` and ``tint`` run -100..100 (negative
    temp is cooler); the tonal and colour controls run -100..100 with 0 as
    no-op, matching the feel of the sliders in a raw editor.
    """

    exposure: float = 0.0
    temp: float = 0.0
    tint: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    gamma: float = 1.0

    #: A tuned default that makes a no-flags merge look finished: opens the
    #: shadows a little, pulls the highlights back off the clip point, and adds
    #: just enough contrast and vibrance to undo tone mapping's usual flatness.
    AUTO = dict(highlights=-25.0, shadows=25.0, contrast=10.0, vibrance=15.0)

    @classmethod
    def auto(cls, **overrides) -> "Adjustments":
        """The ``--auto`` preset, with any explicit settings taking precedence."""
        values = dict(cls.AUTO)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def is_identity(self) -> bool:
        return all(
            getattr(self, f.name) == f.default
            for f in fields(self)
            if isinstance(f.default, float)
        )

    def describe(self) -> str:
        active = [
            f"{f.name}={getattr(self, f.name):g}"
            for f in fields(self)
            if isinstance(f.default, float) and getattr(self, f.name) != f.default
        ]
        return ", ".join(active) if active else "none"


def apply_linear(radiance: np.ndarray, adjustments: Adjustments) -> np.ndarray:
    """Exposure and white balance, applied to linear radiance.

    Note this is what a *radiance* output receives. The display path takes its
    exposure inside tone mapping instead -- see :func:`hdrmerge.tonemap.tonemap`
    for why applying it here would have no effect there.
    """
    result = np.asarray(radiance, dtype=np.float32)

    if adjustments.exposure:
        result = result * np.float32(2.0 ** adjustments.exposure)

    if adjustments.temp or adjustments.tint:
        result = result * _white_balance_gains(adjustments.temp, adjustments.tint)

    return np.maximum(result, 0.0)


def _white_balance_gains(temp: float, tint: float) -> np.ndarray:
    """Per-channel gains, normalised so overall brightness does not shift.

    An approximation, not a colorimetric temperature conversion: warming pushes
    red up and blue down, tint trades green against magenta. That is what the
    sliders need to feel like, and it composes correctly with the camera white
    balance LibRaw already applied.
    """
    warm = float(temp) / 100.0
    green = float(tint) / 100.0

    gains = np.array(
        [1.0 + 0.3 * warm, 1.0 - 0.3 * green, 1.0 - 0.3 * warm],
        dtype=np.float32,
    )
    gains = np.maximum(gains, 0.05)
    return (gains / float(np.dot(gains, _LUMA))).astype(np.float32)


def apply_exposure_display(image: np.ndarray, stops: float) -> np.ndarray:
    """Apply an exposure change to a display-referred image.

    Brightness only means anything in linear light, so the image is decoded,
    scaled and re-encoded rather than multiplied where it stands -- scaling
    gamma-encoded values directly would wash out the shadows.
    """
    if not stops:
        return np.asarray(image, dtype=np.float32)

    from .crf import linear_to_srgb, srgb_to_linear

    linear = srgb_to_linear(image) * np.float32(2.0 ** float(stops))
    return linear_to_srgb(np.clip(linear, 0.0, 1.0))


def apply_display(image: np.ndarray, adjustments: Adjustments) -> np.ndarray:
    """The tonal and colour adjustments, on display-referred [0, 1] values.

    Order matters and follows the usual editing sequence: exposure first, then
    the tonal range (blacks and whites), then the ends (highlights and
    shadows), then contrast, then colour, then the final gamma trim.

    Exposure lands here rather than upstream because every tone-mapping
    operator is scale-invariant -- see :func:`hdrmerge.tonemap.tonemap`.
    """
    result = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)

    if adjustments.exposure:
        result = apply_exposure_display(result, adjustments.exposure)
    if adjustments.blacks or adjustments.whites:
        result = _endpoints(result, adjustments.blacks, adjustments.whites)
    if adjustments.highlights or adjustments.shadows:
        result = _highlights_shadows(result, adjustments.highlights, adjustments.shadows)
    if adjustments.contrast:
        result = _contrast(result, adjustments.contrast)
    if adjustments.saturation or adjustments.vibrance:
        result = _colour(result, adjustments.saturation, adjustments.vibrance)
    if adjustments.gamma and adjustments.gamma != 1.0:
        result = np.power(np.clip(result, 0.0, 1.0), 1.0 / float(adjustments.gamma))

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    return np.tensordot(image, _LUMA, axes=([-1], [0])).astype(np.float32)


def _endpoints(image: np.ndarray, blacks: float, whites: float) -> np.ndarray:
    """Move the black and white points, remapping everything between them.

    Signs follow the convention of a raw editor's sliders: positive whites
    brighten the highlights (by pulling the white point *down* onto them) and
    negative blacks deepen the shadows.
    """
    black_point = -float(blacks) / 100.0 * 0.15
    white_point = 1.0 - float(whites) / 100.0 * 0.15
    span = max(white_point - black_point, 1e-3)
    return np.clip((image - black_point) / span, 0.0, 1.0)


def _highlights_shadows(image: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    """Lift shadows and pull highlights, each masked to its own end of the range.

    Both work on luminance and are re-applied to the channels as a ratio, so
    hue and saturation survive a heavy recovery instead of drifting the way
    per-channel curves make them.
    """
    luma = luminance(image)
    adjusted = luma.copy()

    if shadows:
        amount = float(shadows) / 100.0
        mask = np.power(1.0 - luma, 3.0)               # strongest at black
        adjusted = adjusted + amount * 0.45 * mask * (1.0 - luma)

    if highlights:
        amount = float(highlights) / 100.0
        mask = np.power(np.clip(luma, 0.0, 1.0), 3.0)  # strongest at white
        adjusted = adjusted + amount * 0.45 * mask * luma

    adjusted = np.clip(adjusted, 0.0, 1.0)
    return _relight(image, luma, adjusted)


def _relight(image: np.ndarray, old_luma: np.ndarray, new_luma: np.ndarray) -> np.ndarray:
    """Apply a luminance change to RGB while preserving colour ratios."""
    scale = new_luma / np.maximum(old_luma, 1e-4)
    scaled = image * scale[..., None]
    # Where the original was essentially black there is no ratio to preserve,
    # so add the change flat rather than multiplying up sensor noise.
    flat = np.broadcast_to((new_luma - old_luma)[..., None], image.shape)
    return np.where(old_luma[..., None] < 1e-3, image + flat, scaled)


def _contrast(image: np.ndarray, amount: float) -> np.ndarray:
    """S-curve around mid-grey, applied to luminance."""
    strength = float(amount) / 100.0
    luma = luminance(image)
    centred = np.clip(luma, 0.0, 1.0) - 0.5
    # A smooth cubic S-curve: gentle in the midtones, tapering at both ends so
    # it never clips on its own.
    curved = np.clip(0.5 + centred + strength * (0.5 - 2.0 * centred * centred) * centred * 2.0, 0.0, 1.0)
    return _relight(image, luma, curved)


#: Below this luminance a pixel's colour is mostly sensor noise and quantisation
#: rather than scene content, so colour boosts are faded out.
_SHADOW_COLOUR_FLOOR = 0.06


def _colour(image: np.ndarray, saturation: float, vibrance: float) -> np.ndarray:
    """Saturation scales everything; vibrance spares what is already saturated."""
    luma = luminance(image)[..., None]
    difference = image - luma

    boost = np.float32(float(saturation) / 100.0)

    if vibrance:
        peak = image.max(axis=-1, keepdims=True)
        trough = image.min(axis=-1, keepdims=True)
        current = (peak - trough) / np.maximum(peak, 1e-4)   # HSV-style saturation
        # Weight the boost towards muted pixels, which is what stops vibrance
        # from turning skin tones orange the way plain saturation does.
        boost = boost + np.float32(float(vibrance) / 100.0) * (1.0 - current)

    # Fade the boost out in the deepest shadows. A merged HDR image has real
    # detail down where a single exposure would be black, and the colour of
    # those pixels is dominated by noise -- boosting it prints coloured speckle
    # into what should read as shadow.
    boost = boost * np.clip(luma / _SHADOW_COLOUR_FLOOR, 0.0, 1.0)

    return np.clip(luma + difference * (1.0 + boost), 0.0, 1.0)
