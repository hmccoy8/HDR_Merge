"""Splitting a folder of frames into individual brackets."""

import datetime as dt

import pytest
import synthetic

from hdrmerge.grouping import group_frames


def shoot(tmp_path, brackets, gap=8.0, size=3, interval=0.4):
    """Write ``brackets`` consecutive brackets separated by ``gap`` seconds."""
    start = dt.datetime(2026, 4, 12, 9, 30, 0)
    exposures = [1 / 200, 1 / 50, 1 / 12.5, 1 / 3][:size]
    paths = []
    for index in range(brackets):
        paths += synthetic.write_jpeg_bracket(
            str(tmp_path),
            exposures=exposures,
            start=start + dt.timedelta(seconds=index * gap),
            interval=interval,
            index_from=1 + index * size,
        )
    return paths


def test_a_single_bracket_stays_whole(tmp_path):
    paths = shoot(tmp_path, brackets=1)

    groups = group_frames(paths)

    assert len(groups) == 1
    assert groups[0].size == 3


def test_pauses_between_brackets_split_them(tmp_path):
    paths = shoot(tmp_path, brackets=4, gap=10.0)

    groups = group_frames(paths)

    assert [group.size for group in groups] == [3, 3, 3, 3]


def test_brackets_shot_back_to_back_split_on_the_exposure_pattern(tmp_path):
    """The case a time-gap heuristic alone would run together."""
    paths = shoot(tmp_path, brackets=3, gap=0.8, interval=0.4)

    groups = group_frames(paths)

    assert len(groups) == 3
    assert all(group.size == 3 for group in groups)


def test_each_bracket_reports_its_ev_span(tmp_path):
    paths = shoot(tmp_path, brackets=2)

    groups = group_frames(paths)

    assert all(group.ev_span == pytest.approx(4.0, abs=0.05) for group in groups)


def test_orphan_frames_are_dropped_not_merged(tmp_path):
    paths = shoot(tmp_path, brackets=1)
    stray = synthetic.write_jpeg_bracket(
        str(tmp_path), exposures=[1 / 100],
        start=dt.datetime(2026, 4, 12, 10, 0, 0), index_from=90,
    )

    groups = group_frames(paths + stray)

    assert len(groups) == 1
    assert stray[0] not in groups[0].paths


def test_explicit_bracket_size_overrides_detection(tmp_path):
    """The escape hatch for when the heuristics guess wrong."""
    paths = shoot(tmp_path, brackets=2, gap=0.5, interval=0.1)

    groups = group_frames(paths, expected_size=3)

    assert [group.size for group in groups] == [3, 3]


def test_frames_are_grouped_in_capture_order_not_argument_order(tmp_path):
    paths = shoot(tmp_path, brackets=2, gap=10.0)

    groups = group_frames(list(reversed(paths)))

    assert [group.size for group in groups] == [3, 3]
    assert groups[0].frames[0].name < groups[1].frames[0].name


def test_long_exposure_brackets_are_not_split_by_their_own_pauses(tmp_path):
    """A night bracket's in-bracket gaps exceed a daylight bracket's pauses.

    A fixed time threshold would shatter this into single frames, which is why
    the threshold scales with the shoot's own typical spacing.
    """
    paths = shoot(tmp_path, brackets=2, gap=90.0, interval=25.0)

    groups = group_frames(paths)

    assert [group.size for group in groups] == [3, 3]


def test_a_bracket_of_four_is_detected_as_one(tmp_path):
    paths = shoot(tmp_path, brackets=2, size=4, gap=12.0)

    groups = group_frames(paths)

    assert [group.size for group in groups] == [4, 4]
