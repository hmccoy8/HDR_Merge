"""End-to-end behaviour of the merge pipeline and its output formats."""

import os

import numpy as np
import pytest
import synthetic

from hdrmerge import Adjustments, MergeOptions, merge_bracket, writers
from hdrmerge.loaders import LoadError


@pytest.fixture
def bracket(tmp_path):
    return synthetic.write_jpeg_bracket(str(tmp_path))


def options(tmp_path, **overrides):
    settings = dict(out_dir=str(tmp_path / "out"), align_method="none")
    settings.update(overrides)
    return MergeOptions(**settings)


def test_merge_writes_the_requested_formats(tmp_path, bracket):
    result = merge_bracket(bracket, options(tmp_path, formats=["jpg", "exr", "tif16"]))

    assert len(result.written) == 3
    assert all(os.path.getsize(path) > 0 for path in result.written)
    assert {os.path.splitext(p)[1] for p in result.written} == {".jpg", ".exr", ".tif"}


@pytest.mark.parametrize("fmt", sorted(writers.FORMATS))
def test_every_format_writes_a_readable_file(tmp_path, bracket, fmt):
    result = merge_bracket(tmp_path and bracket, options(tmp_path, formats=[fmt]))

    assert len(result.written) == 1
    assert _read_back(result.written[0], fmt).shape[:2] == (96, 128)


def _read_back(path, fmt):
    if fmt == "exr":
        import OpenEXR

        with OpenEXR.File(path) as handle:
            return handle.channels()["RGB"].pixels
    if fmt in ("tif16", "tif32"):
        import tifffile

        return tifffile.imread(path)

    import cv2

    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def test_radiance_formats_keep_linear_data_and_display_formats_do_not(tmp_path, bracket):
    """The distinction the whole writer split exists for."""
    result = merge_bracket(tmp_path and bracket, options(tmp_path, formats=["exr", "jpg"]))

    radiance = _read_back(result.written[0], "exr")
    display = _read_back(result.written[1], "jpg")

    assert radiance.dtype == np.float32
    assert radiance.max() > 1.0, "linear radiance should exceed the display ceiling"
    assert display.dtype == np.uint8


def test_bit_depth_matches_the_format(tmp_path, bracket):
    result = merge_bracket(
        tmp_path and bracket, options(tmp_path, formats=["jpg", "png16", "tif16", "tif32"])
    )
    depths = [_read_back(path, fmt).dtype for path, fmt in
              zip(result.written, ["jpg", "png16", "tif16", "tif32"])]

    assert depths == [np.uint8, np.uint16, np.uint16, np.float32]


def test_frames_are_sorted_by_exposure_regardless_of_input_order(tmp_path, bracket):
    result = merge_bracket(list(reversed(bracket)), options(tmp_path))

    exposures = [meta.relative_exposure for meta in result.metas]
    assert exposures == sorted(exposures)


def test_reference_defaults_to_the_middle_exposure(tmp_path, bracket):
    result = merge_bracket(bracket, options(tmp_path))

    assert result.reference == 1
    assert result.metas[result.reference].name.endswith("0002.jpg")


def test_reference_can_be_chosen(tmp_path, bracket):
    result = merge_bracket(bracket, options(tmp_path, reference=0))

    assert result.reference == 0


def test_out_of_range_reference_is_reported(tmp_path, bracket):
    with pytest.raises(ValueError, match="out of range"):
        merge_bracket(bracket, options(tmp_path, reference=7))


@pytest.mark.parametrize("operator", ["reinhard", "drago", "mantiuk", "linear", "fusion"])
def test_every_tone_mapping_operator_produces_a_usable_image(tmp_path, bracket, operator):
    result = merge_bracket(
        tmp_path and bracket, options(tmp_path, tonemap=operator, formats=["jpg"])
    )

    display = result.display
    assert display.min() >= 0.0 and display.max() <= 1.0
    assert np.isfinite(display).all()
    assert 0.05 < display.mean() < 0.95, "the image should be neither black nor blown"


def test_fusion_needs_no_radiance_map(tmp_path, bracket):
    result = merge_bracket(tmp_path and bracket, options(tmp_path, tonemap="fusion"))

    assert result.radiance is None
    assert result.display is not None


def test_fusion_with_a_radiance_format_is_refused_with_an_explanation(tmp_path, bracket):
    with pytest.raises(ValueError, match="never builds a radiance map"):
        merge_bracket(bracket, options(tmp_path, tonemap="fusion", formats=["exr"]))


def test_auto_grade_changes_the_image_without_breaking_it(tmp_path, bracket):
    plain = merge_bracket(bracket, options(tmp_path)).display
    graded = merge_bracket(
        bracket, options(tmp_path, adjustments=Adjustments.auto())
    ).display

    assert not np.allclose(plain, graded)
    assert graded.min() >= 0.0 and graded.max() <= 1.0
    assert np.isfinite(graded).all()


def test_no_adjustments_means_no_change(tmp_path, bracket):
    first = merge_bracket(bracket, options(tmp_path)).display
    second = merge_bracket(
        bracket, options(tmp_path, adjustments=Adjustments())
    ).display

    assert np.array_equal(first, second)


def test_exposure_adjustment_brightens_the_result(tmp_path, bracket):
    darker = merge_bracket(
        bracket, options(tmp_path, adjustments=Adjustments(exposure=-1.0))
    ).display
    brighter = merge_bracket(
        bracket, options(tmp_path, adjustments=Adjustments(exposure=1.0))
    ).display

    assert brighter.mean() > darker.mean()


def test_merging_with_no_formats_still_returns_the_images(tmp_path, bracket):
    """The library path: merge, hand back the result, write nothing."""
    result = merge_bracket(bracket, options(tmp_path, formats=[]))

    assert result.written == []
    assert result.display is not None
    assert result.radiance is not None
    assert not os.path.exists(str(tmp_path / "out"))


def test_requesting_only_radiance_skips_tone_mapping(tmp_path, bracket):
    result = merge_bracket(tmp_path and bracket, options(tmp_path, formats=["exr"]))

    assert result.radiance is not None
    assert result.display is None, "tone mapping is wasted work for radiance output"


def test_preview_scale_shrinks_the_output(tmp_path, bracket):
    result = merge_bracket(bracket, options(tmp_path, preview_scale=0.5))

    assert result.display.shape[:2] == (48, 64)


def test_a_single_frame_is_not_a_bracket(tmp_path, bracket):
    with pytest.raises(LoadError, match="at least 2 frames"):
        merge_bracket(bracket[:1], options(tmp_path))


def test_max_frames_guards_memory(tmp_path, bracket):
    with pytest.raises(LoadError, match="max-frames"):
        merge_bracket(bracket, options(tmp_path, max_frames=2))


def test_mismatched_dimensions_are_refused(tmp_path, bracket):
    from PIL import Image

    with Image.open(bracket[-1]) as img:
        img.resize((64, 48)).save(bracket[-1], quality=97)

    with pytest.raises(LoadError, match="same dimensions"):
        merge_bracket(bracket, options(tmp_path))


def test_identical_exposures_are_not_a_bracket(tmp_path):
    paths = synthetic.write_jpeg_bracket(
        str(tmp_path), exposures=[1 / 100, 1 / 100, 1 / 100]
    )

    with pytest.raises(LoadError, match="same exposure"):
        merge_bracket(paths, options(tmp_path))


def test_unknown_format_is_rejected_before_any_work_happens(tmp_path, bracket):
    with pytest.raises(writers.WriteError, match="Unknown output format"):
        merge_bracket(bracket, options(tmp_path, formats=["gif"]))
