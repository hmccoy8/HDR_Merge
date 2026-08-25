"""Exposure detection from EXIF, and the pixel fallback when EXIF is absent."""

import datetime as dt
import os

import numpy as np
import pytest
import synthetic

from hdrmerge.metadata import (
    SOURCE_ESTIMATED,
    SOURCE_EXIF,
    FrameMeta,
    read_exif,
    relative_exposure,
    resolve_exposures,
)


@pytest.fixture
def jpeg_bracket(tmp_path):
    return synthetic.write_jpeg_bracket(str(tmp_path), exposures=[1 / 200, 1 / 50, 1 / 12.5])


def test_exif_gives_exposure_and_capture_time(jpeg_bracket):
    metas = [read_exif(path) for path in jpeg_bracket]

    assert all(meta.source == SOURCE_EXIF for meta in metas)
    assert [round(meta.exposure_time, 6) for meta in metas] == [0.005, 0.02, 0.08]
    assert all(meta.f_number == 8.0 and meta.iso == 100 for meta in metas)
    assert all(meta.timestamp is not None for meta in metas)


def test_two_stop_bracket_reads_as_two_stop_steps(jpeg_bracket):
    evs = sorted(read_exif(path).ev for path in jpeg_bracket)

    steps = [b - a for a, b in zip(evs[:-1], evs[1:])]
    assert steps == pytest.approx([2.0, 2.0], abs=0.01)


def test_sub_second_timestamps_separate_frames_within_a_bracket(jpeg_bracket):
    """DateTimeOriginal only has second resolution; brackets are shot faster."""
    stamps = sorted(read_exif(path).timestamp for path in jpeg_bracket)

    gaps = [(b - a).total_seconds() for a, b in zip(stamps[:-1], stamps[1:])]
    assert gaps == pytest.approx([0.4, 0.4], abs=0.02)


def test_relative_exposure_accounts_for_shutter_aperture_and_iso():
    base = relative_exposure(1 / 100, 8.0, 100)

    assert relative_exposure(1 / 50, 8.0, 100) == pytest.approx(2 * base)   # 1 stop slower
    assert relative_exposure(1 / 100, 8.0, 200) == pytest.approx(2 * base)  # 1 stop more ISO
    # f-numbers are printed rounded, so one stop from f/8 is f/5.657 exactly
    # and the familiar "f/5.6" is 2% off. Compare against the real stop.
    assert relative_exposure(1 / 100, 8.0 / 2 ** 0.5, 100) == pytest.approx(2 * base)


def test_relative_exposure_needs_a_shutter_speed():
    assert relative_exposure(None, 8.0, 100) is None
    assert relative_exposure(0.0, 8.0, 100) is None


def test_ev_is_higher_for_darker_frames():
    dark = FrameMeta("a", relative_exposure=relative_exposure(1 / 1000, 8.0, 100))
    bright = FrameMeta("b", relative_exposure=relative_exposure(1 / 30, 8.0, 100))

    assert dark.ev > bright.ev


def test_missing_exif_falls_back_to_pixel_estimation():
    exposures = [0.25, 1.0, 4.0]
    truth, frames, _ = synthetic.bracket(exposures=exposures)
    metas = [FrameMeta(path=f"frame{i}.tif") for i in range(len(frames))]

    resolved = resolve_exposures(metas, images=frames)

    assert all(meta.source == SOURCE_ESTIMATED for meta in resolved)
    values = np.array([meta.relative_exposure for meta in resolved])
    assert np.allclose(values / values[0], np.array(exposures) / exposures[0], rtol=0.02)


def test_pixel_estimation_matches_exif_on_the_same_files(jpeg_bracket):
    """The fallback should land within a quarter stop of the measured truth."""
    from hdrmerge.loaders import load_frame
    from hdrmerge.metadata import estimate_relative_exposures

    metas = [read_exif(path) for path in jpeg_bracket]
    order = sorted(range(len(metas)), key=lambda i: metas[i].relative_exposure)
    images = [load_frame(jpeg_bracket[i]).data for i in order]

    estimated = np.array(estimate_relative_exposures(images, linear=False))
    measured = np.array([metas[i].relative_exposure for i in order])

    estimated_ev = -np.log2(estimated / estimated[0])
    measured_ev = -np.log2(measured / measured[0])
    assert np.abs(estimated_ev - measured_ev).max() < 0.25


def test_unreadable_file_returns_empty_metadata_rather_than_raising(tmp_path):
    path = tmp_path / "not-an-image.jpg"
    path.write_bytes(b"definitely not a jpeg")

    meta = read_exif(str(path))

    assert meta.relative_exposure is None
    assert meta.ev is None
    assert meta.describe() == "-"


def test_resolve_without_exif_or_pixels_explains_itself():
    metas = [FrameMeta(path="a.cr3"), FrameMeta(path="b.cr3")]

    with pytest.raises(ValueError, match="a.cr3"):
        resolve_exposures(metas, images=None)
