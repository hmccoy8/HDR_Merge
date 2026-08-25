"""Splitting a folder of frames into individual brackets.

Point the tool at a shoot containing twenty 3-frame brackets and it should
produce twenty HDR images, not one merge of sixty unrelated photos. Two signals
do the work, and they cover different failure modes:

* **Time gaps.** Frames within a bracket land within a second or so of each
  other; the pause before the next bracket is much longer. This is the primary
  signal and it handles the normal case on its own.
* **Exposure pattern.** A bracket's exposures step monotonically and then reset
  for the next one. This catches brackets shot back-to-back with no pause,
  which the time signal alone would run together.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .metadata import FrameMeta, read_exif

log = logging.getLogger(__name__)

DEFAULT_MAX_GAP = 2.0        # seconds
DEFAULT_GAP_FACTOR = 3.0     # a gap this much larger than typical starts a bracket


@dataclass
class Bracket:
    """One detected bracket."""

    frames: List[FrameMeta] = field(default_factory=list)

    @property
    def paths(self) -> List[str]:
        return [frame.path for frame in self.frames]

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def ev_span(self) -> Optional[float]:
        evs = [frame.ev for frame in self.frames if frame.ev is not None]
        return max(evs) - min(evs) if len(evs) > 1 else None

    def describe(self) -> str:
        span = f", {self.ev_span:.1f} EV span" if self.ev_span else ""
        names = ", ".join(frame.name for frame in self.frames)
        return f"{self.size} frames{span}: {names}"


def group_frames(
    paths: Sequence[str],
    max_gap: float = DEFAULT_MAX_GAP,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    min_size: int = 2,
    expected_size: Optional[int] = None,
) -> List[Bracket]:
    """Group ``paths`` into brackets.

    ``expected_size`` short-circuits detection entirely and chops the (capture-
    ordered) frames into fixed-size groups -- worth using when the heuristics
    guess wrong, which they can on brackets shot with long exposures.
    """
    metas = [read_exif(path) for path in paths]
    metas = _sort_by_capture(metas)

    if expected_size:
        if expected_size < min_size:
            raise ValueError(f"--bracket-size must be at least {min_size}")
        groups = [
            Bracket(metas[i:i + expected_size])
            for i in range(0, len(metas), expected_size)
        ]
    else:
        groups = _detect(metas, max_gap, gap_factor)

    return _report(groups, min_size)


def _sort_by_capture(metas: Sequence[FrameMeta]) -> List[FrameMeta]:
    """Capture order, falling back to filename where timestamps are missing."""
    if all(meta.timestamp for meta in metas):
        return sorted(metas, key=lambda m: m.timestamp)
    if any(meta.timestamp for meta in metas):
        log.warning("Some frames have no capture time; grouping by filename order")
    return sorted(metas, key=lambda m: m.name)


def _detect(metas: Sequence[FrameMeta], max_gap: float, gap_factor: float) -> List[Bracket]:
    if len(metas) < 2:
        return [Bracket(list(metas))] if metas else []

    gaps = _gaps(metas)
    threshold = _threshold(gaps, max_gap, gap_factor)

    groups: List[Bracket] = []
    current = [metas[0]]

    for index in range(1, len(metas)):
        gap = gaps[index - 1]
        time_break = gap is not None and gap > threshold
        pattern_break = _pattern_resets(current, metas[index])

        if time_break or pattern_break:
            groups.append(Bracket(current))
            current = [metas[index]]
            log.debug(
                "Split before %s (%s)",
                metas[index].name,
                "time gap" if time_break else "exposure pattern reset",
            )
        else:
            current.append(metas[index])

    groups.append(Bracket(current))
    return groups


def _gaps(metas: Sequence[FrameMeta]) -> List[Optional[float]]:
    result: List[Optional[float]] = []
    for previous, current in zip(metas[:-1], metas[1:]):
        if previous.timestamp and current.timestamp:
            result.append((current.timestamp - previous.timestamp).total_seconds())
        else:
            result.append(None)
    return result


def _threshold(gaps: Sequence[Optional[float]], max_gap: float, gap_factor: float) -> float:
    """Adapt to the shoot: scale the typical in-bracket gap, floored at ``max_gap``.

    Adapting matters because a bracket of 30-second night exposures has
    in-bracket gaps far longer than the pause between two daylight brackets.
    """
    known = sorted(gap for gap in gaps if gap is not None)
    if not known:
        return max_gap
    median = known[len(known) // 2]
    return max(max_gap, median * gap_factor)


def _pattern_resets(current: Sequence[FrameMeta], candidate: FrameMeta) -> bool:
    """True when ``candidate`` restarts the exposure sequence.

    A bracket steps consistently in one direction. When the next frame steps
    back the other way -- or repeats an exposure already in this group -- the
    camera has started a new bracket.
    """
    evs = [frame.ev for frame in current if frame.ev is not None]
    if len(evs) < 2 or candidate.ev is None:
        return False

    steps = [b - a for a, b in zip(evs[:-1], evs[1:])]
    if not all(step > 0.05 for step in steps) and not all(step < -0.05 for step in steps):
        return False   # this group is not a clean ramp; do not read into it

    next_step = candidate.ev - evs[-1]
    ascending = steps[0] > 0
    reversed_direction = (next_step < -0.05) if ascending else (next_step > 0.05)
    repeats = any(abs(candidate.ev - ev) < 0.05 for ev in evs)
    return reversed_direction or repeats


def _report(groups: Sequence[Bracket], min_size: int) -> List[Bracket]:
    kept: List[Bracket] = []
    for group in groups:
        if group.size < min_size:
            log.warning(
                "Skipping %d orphan frame(s) that form no bracket: %s",
                group.size, ", ".join(frame.name for frame in group.frames),
            )
            continue
        kept.append(group)

    sizes = {group.size for group in kept}
    if len(sizes) > 1:
        log.warning(
            "Detected brackets of differing sizes (%s). If that is wrong, pass "
            "--bracket-size to split into fixed-size groups instead.",
            ", ".join(str(size) for size in sorted(sizes)),
        )
    return kept


def output_base(bracket: Bracket) -> str:
    """The path a bracket's outputs are named after: its first frame."""
    return bracket.frames[0].path if bracket.frames else "merged"
