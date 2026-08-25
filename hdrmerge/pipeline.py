"""The merge pipeline: the one place the stages are wired together.

Deliberately free of any CLI or UI concern, so a future GUI can drive exactly
the same code path the command line does.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import align as align_module
from . import crf, deghost, merge as merge_module, tonemap as tonemap_module, writers
from .adjust import Adjustments, apply_display, apply_linear
from .loaders import Frame, LoadError, load_bracket
from .metadata import FrameMeta, resolve_exposures

log = logging.getLogger(__name__)


@dataclass
class MergeOptions:
    """Everything that steers a merge. Defaults produce a good result unaided."""

    # Input handling
    preview_scale: float = 1.0
    max_frames: int = 15
    reference: Optional[int] = None

    # Merge
    crf_method: str = "auto"
    align_method: str = "mtb"
    deghost_threshold: float = 0.7
    saturation_threshold: float = 0.99

    # Output
    formats: List[str] = field(default_factory=lambda: ["jpg"])
    out_dir: Optional[str] = None
    suffix: str = "_hdr"
    jpeg_quality: int = 95

    # Post-processing
    tonemap: str = "reinhard"
    gamma: float = 2.2
    adjustments: Adjustments = field(default_factory=Adjustments)

    def validate(self) -> None:
        for fmt in self.formats:
            writers._check(fmt)
        if self.tonemap not in tonemap_module.OPERATORS:
            raise ValueError(
                f"Unknown tone-mapping operator {self.tonemap!r}. "
                f"Choose from: {', '.join(tonemap_module.OPERATORS)}"
            )
        if self.tonemap == "fusion":
            radiance = [f for f in self.formats if writers.is_radiance(f)]
            if radiance:
                raise ValueError(
                    "--tonemap fusion cannot produce "
                    f"{', '.join(radiance)}: exposure fusion blends the frames "
                    "directly and never builds a radiance map. Use a radiance "
                    "operator (reinhard, drago, mantiuk, linear) for those formats."
                )

        # Checked last, and deliberately after the conflicts above: those are
        # wrong no matter where they run, so they should report the same thing
        # on every machine. Missing-backend is a property of this environment.
        # Still ahead of any merging though -- discovering a missing writer
        # after a 20-bracket batch has run wastes the user's time.
        for fmt in self.formats:
            reason = writers.unavailable_reason(fmt)
            if reason:
                raise ValueError(reason)

        if not 0 < self.preview_scale <= 1.0:
            raise ValueError(f"--preview-scale must be in (0, 1], got {self.preview_scale}")


@dataclass
class MergeResult:
    """The outcome of one merge."""

    metas: List[FrameMeta]
    reference: int
    #: Linear radiance in reference-frame exposure units. None for fusion.
    radiance: Optional[np.ndarray]
    #: Display-referred result in [0, 1]. None when only radiance was asked for.
    display: Optional[np.ndarray]
    stats: Optional[merge_module.MergeStats]
    written: List[str] = field(default_factory=list)


def merge_bracket(paths: Sequence[str], options: Optional[MergeOptions] = None) -> MergeResult:
    """Merge one bracket. The public entry point of the library."""
    options = options or MergeOptions()
    options.validate()

    frames = load_bracket(paths, options.preview_scale, options.max_frames)
    log.info("Loaded %d frames (%dx%d)", len(frames), frames[0].shape[1], frames[0].shape[0])

    frames = _order_by_exposure(frames)
    reference = _pick_reference(frames, options.reference)
    log.info("Reference frame: %s", frames[reference].meta.name)

    encoded = [frame.data for frame in frames]
    exposures = [frame.meta.relative_exposure for frame in frames]

    if options.tonemap == "fusion":
        display = merge_module.exposure_fusion(encoded)
        display = apply_display(display, options.adjustments)
        return _write(
            MergeResult(
                metas=[f.meta for f in frames], reference=reference,
                radiance=None, display=display, stats=None,
            ),
            paths[0], options,
        )

    aligned, _ = align_module.align(encoded, reference, options.align_method)

    method = crf.choose_method(frames[0].is_raw, options.crf_method)
    try:
        linear = crf.linearize(aligned, exposures, method)
    except crf.CRFError as exc:
        # Without a trustworthy response curve the radiance map would be wrong
        # in ways that are hard to see and impossible to undo. Fusion needs no
        # curve at all, so it is the honest fallback.
        raise MergeFallback(
            f"Could not recover a camera response curve ({exc}). "
            "Retry with --tonemap fusion, which needs no response curve, "
            "or --crf srgb to assume a standard sRGB tone curve."
        ) from exc

    masks = (
        deghost.ghost_masks(linear, exposures, reference, options.deghost_threshold)
        if options.deghost_threshold > 0
        else None
    )

    radiance, stats = merge_module.merge_radiance(
        linear, aligned, exposures, reference, masks, options.saturation_threshold
    )
    log.info(
        "Merged %d frames spanning %.1f stops of dynamic range",
        stats.frames, stats.dynamic_range_stops,
    )

    radiance = apply_linear(radiance, options.adjustments)

    display = None
    # An empty format list means "merge but write nothing" -- a library caller
    # wanting the result in memory -- so it still needs the display image.
    if not options.formats or any(not writers.is_radiance(fmt) for fmt in options.formats):
        display = tonemap_module.tonemap(radiance, options.tonemap, options.gamma)
        display = apply_display(display, options.adjustments)

    return _write(
        MergeResult(
            metas=[f.meta for f in frames], reference=reference,
            radiance=radiance, display=display, stats=stats,
        ),
        paths[0], options,
    )


class MergeFallback(RuntimeError):
    """Raised when a merge cannot proceed but a different option would work."""


def _order_by_exposure(frames: List[Frame]) -> List[Frame]:
    """Sort darkest to brightest, resolving exposures first if EXIF was absent.

    Filename order is not exposure order in general -- and the merge, the
    reference choice and deghosting all assume a known ordering -- so this
    happens before anything else touches the pixels.
    """
    metas = resolve_exposures(
        [frame.meta for frame in frames],
        images=[frame.data for frame in frames],
        images_are_linear=frames[0].is_linear,
    )
    for frame, meta in zip(frames, metas):
        frame.meta = meta

    values = [frame.meta.relative_exposure for frame in frames]
    if len(set(values)) == 1:
        raise LoadError(
            "Every frame has the same exposure, so there is no bracket to merge. "
            "Check that these are actually bracketed captures."
        )

    return sorted(frames, key=lambda frame: frame.meta.relative_exposure)


def _pick_reference(frames: Sequence[Frame], requested: Optional[int]) -> int:
    """Default to the middle exposure, which usually has the most usable pixels."""
    if requested is None:
        return len(frames) // 2
    if not 0 <= requested < len(frames):
        raise ValueError(
            f"--reference {requested} is out of range; this bracket has "
            f"{len(frames)} frames (0-{len(frames) - 1}, ordered darkest first)"
        )
    return requested


def _write(result: MergeResult, base: str, options: MergeOptions) -> MergeResult:
    for fmt in options.formats:
        image = result.radiance if writers.is_radiance(fmt) else result.display
        if image is None:
            continue
        path = writers.output_path(base, fmt, options.out_dir, options.suffix)
        writers.write(image, path, fmt, options.jpeg_quality)
        result.written.append(path)
        log.info("Wrote %s", path)
    return result
