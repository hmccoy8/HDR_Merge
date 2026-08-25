"""The central accuracy test: does a merge actually recover the scene?

Rendering a bracket from a scene whose radiance we chose ourselves means the
merge can be checked against the truth rather than against itself. Errors are
measured in stops, which is the unit the error actually matters in.
"""

import numpy as np
import pytest
import synthetic

from hdrmerge.crf import linearize
from hdrmerge.merge import MergeError, hat_weight, merge_radiance
from hdrmerge.metadata import estimate_relative_exposures

EXPOSURES = [0.125, 0.5, 2.0, 8.0]
REFERENCE = 1


def covered_mask(truth, exposures):
    """Pixels at least one frame in the bracket can actually describe."""
    return (truth * max(exposures) > 0.03) & (truth * min(exposures) < 0.95)


def stop_error(recovered, truth, mask):
    return np.abs(np.log2(recovered[mask] / truth[mask]))


def test_linear_merge_recovers_the_scene_exactly():
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES)

    radiance, stats = merge_radiance(frames, frames, exposures, REFERENCE)
    scene = radiance / exposures[REFERENCE]

    error = stop_error(scene, truth, covered_mask(truth, exposures))
    assert np.median(error) < 0.01
    assert np.percentile(error, 99) < 0.05
    assert stats.frames == len(EXPOSURES)
    assert stats.dynamic_range_stops > 10


def test_merge_is_anchored_to_the_reference_exposure():
    """Well-exposed midtones should come back near their reference-frame value."""
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES)

    radiance, _ = merge_radiance(frames, frames, exposures, REFERENCE)

    reference_frame = frames[REFERENCE]
    midtones = (reference_frame > 0.2) & (reference_frame < 0.8)
    assert np.allclose(radiance[midtones], reference_frame[midtones], rtol=0.02)


@pytest.mark.parametrize("method,tolerance", [("srgb", 0.01), ("debevec", 0.05)])
def test_encoded_frames_merge_through_a_response_curve(method, tolerance):
    """JPEG-style input must be linearised before it can be merged."""
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES, encode=True)

    linear = linearize(frames, exposures, method)
    radiance, _ = merge_radiance(linear, frames, exposures, REFERENCE)

    # A recovered response curve fixes radiance only up to a global scale, so
    # the comparison is of shape, not of absolute level.
    mask = covered_mask(truth, exposures)
    scene = radiance / exposures[REFERENCE]
    scale = np.median(scene[mask] / truth[mask])

    error = stop_error(scene / scale, truth, mask)
    assert np.median(error) < tolerance


def test_merging_encoded_frames_without_linearising_is_visibly_wrong():
    """Guards the reason the two paths exist at all."""
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES, encode=True)
    mask = covered_mask(truth, exposures)

    naive, _ = merge_radiance(frames, frames, exposures, REFERENCE)
    correct, _ = merge_radiance(
        linearize(frames, exposures, "srgb"), frames, exposures, REFERENCE
    )

    def error_of(radiance):
        scene = radiance / exposures[REFERENCE]
        return np.median(stop_error(scene / np.median(scene[mask] / truth[mask]), truth, mask))

    assert error_of(naive) > 10 * max(error_of(correct), 1e-3)


def test_a_recovered_response_curve_is_neutral_at_black():
    """Regions black in every frame must not pick up a colour cast."""
    from hdrmerge.crf import estimate_response

    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES, encode=True)

    response = estimate_response(frames, exposures, "debevec")

    assert response[0, 0] == response[0, 1] == response[0, 2]


def test_regions_black_in_every_frame_stay_neutral():
    truth = synthetic.scene(stops=20.0)          # far wider than the bracket
    _, frames, exposures = synthetic.bracket(truth, exposures=[1.0, 2.0], encode=True)
    # Quantise to 8 bits, as a camera does: that is what turns "very dark" into
    # the exactly-zero pixels this test is about.
    frames = [np.round(frame * 255.0).astype(np.float32) / 255.0 for frame in frames]

    linear = linearize(frames, exposures, "debevec")
    radiance, _ = merge_radiance(linear, frames, exposures, 0)

    black_everywhere = np.all([frame.max(axis=-1) <= 0.0 for frame in frames], axis=0)
    assert black_everywhere.any(), "this scene should exceed the bracket's reach"

    values = radiance[black_everywhere]
    spread = values.max(axis=-1) - values.min(axis=-1)
    assert np.allclose(spread, 0.0, atol=1e-9), "black should have no hue"


def test_exposures_are_recovered_from_pixels_alone():
    """The fallback for frames with no usable EXIF."""
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES)

    estimated = np.array(estimate_relative_exposures(frames))
    truth_ratios = np.array(exposures) / exposures[0]

    assert np.allclose(estimated / estimated[0], truth_ratios, rtol=0.02)


def test_exposures_are_recovered_from_encoded_pixels():
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES, encode=True)

    estimated = np.array(estimate_relative_exposures(frames, linear=False))

    assert np.allclose(
        estimated / estimated[0], np.array(exposures) / exposures[0], rtol=0.05
    )


def test_noise_does_not_break_exposure_estimation():
    truth, frames, exposures = synthetic.bracket(exposures=EXPOSURES, noise=0.01)

    estimated = np.array(estimate_relative_exposures(frames))

    assert np.allclose(
        estimated / estimated[0], np.array(exposures) / exposures[0], rtol=0.05
    )


def test_more_frames_recover_more_dynamic_range():
    truth = synthetic.scene(stops=16.0)
    _, few, few_exposures = synthetic.bracket(truth, exposures=[1.0, 4.0])
    _, many, many_exposures = synthetic.bracket(
        truth, exposures=[0.06, 0.25, 1.0, 4.0, 16.0]
    )

    _, narrow = merge_radiance(few, few, few_exposures, 0)
    _, wide = merge_radiance(many, many, many_exposures, 2)

    assert wide.dynamic_range_stops > narrow.dynamic_range_stops
    assert wide.fallback_fraction < narrow.fallback_fraction


def test_hat_weight_discards_clipped_and_black_pixels():
    values = np.array([[0.0, 0.005, 0.2, 0.5, 0.8, 0.995, 1.0]], dtype=np.float32)
    weights = hat_weight(np.repeat(values[..., None], 3, axis=-1))

    assert weights[0, 0] == 0.0 and weights[0, 1] == 0.0     # black
    assert weights[0, 5] == 0.0 and weights[0, 6] == 0.0     # clipped
    assert weights[0, 3] == pytest.approx(1.0, abs=1e-3)     # mid-grey
    assert weights[0, 2] > 0.5 and weights[0, 4] > 0.5


def test_weight_is_shared_across_channels():
    """One clipped channel must discount the whole pixel, not just that channel.

    Weighting channels independently is what gives blown highlights a colour
    cast, because the unclipped channels keep contributing after the clipped
    one has stopped.
    """
    pixel = np.array([[[1.0, 0.5, 0.5]]], dtype=np.float32)

    assert hat_weight(pixel)[0, 0] == 0.0


def test_pixels_no_frame_can_describe_fall_back_instead_of_going_black():
    """The usual source of black speckle in blown skies."""
    truth = synthetic.scene(stops=20.0)          # far wider than the bracket
    _, frames, exposures = synthetic.bracket(truth, exposures=[1.0, 2.0])

    radiance, stats = merge_radiance(frames, frames, exposures, 0)

    assert stats.fallback_pixels > 0, "this scene should exceed the bracket's range"
    assert np.isfinite(radiance).all()
    assert (radiance >= 0).all()
    assert not np.any(np.all(radiance == 0, axis=-1)), "no pixel should be pure black"


def test_merge_rejects_impossible_input():
    truth, frames, exposures = synthetic.bracket()

    with pytest.raises(MergeError):
        merge_radiance(frames[:1], frames[:1], exposures[:1], 0)
    with pytest.raises(MergeError):
        merge_radiance(frames, frames, exposures, reference=99)
    with pytest.raises(MergeError):
        merge_radiance(frames, frames, [0.0] * len(frames), 0)
    with pytest.raises(MergeError):
        merge_radiance(frames, frames, exposures[:-1], 0)
