"""Alignment and deghosting -- the two things handheld brackets need."""

import numpy as np
import pytest
import synthetic

from hdrmerge.align import align
from hdrmerge.deghost import ghost_masks
from hdrmerge.merge import merge_radiance

EXPOSURES = [0.25, 1.0, 4.0]
REFERENCE = 1


def shift(image, dx, dy):
    return np.roll(np.roll(image, dx, axis=1), dy, axis=0)


def interior(image):
    """Crop the border, where a shift pulls in replicated edge pixels."""
    return image[14:-14, 14:-14]


def test_alignment_recovers_a_known_shift():
    truth, frames, _ = synthetic.bracket(exposures=EXPOSURES)
    drifted = [shift(f, 3, -2) if i != REFERENCE else f for i, f in enumerate(frames)]

    aligned, shifts = align(drifted, REFERENCE, "mtb")

    for index in (0, 2):
        assert shifts[index] == pytest.approx((-3.0, 2.0), abs=1.0)
        assert np.abs(interior(aligned[index]) - interior(frames[index])).mean() < 0.005


def test_alignment_improves_agreement_between_frames():
    truth, frames, _ = synthetic.bracket(exposures=EXPOSURES)
    drifted = [shift(f, 5, 4) if i != REFERENCE else f for i, f in enumerate(frames)]

    aligned, _ = align(drifted, REFERENCE, "mtb")

    for index in (0, 2):
        before = np.abs(interior(drifted[index]) - interior(frames[index])).mean()
        after = np.abs(interior(aligned[index]) - interior(frames[index])).mean()
        assert after < before / 2


def test_alignment_leaves_an_already_aligned_bracket_alone():
    """A spurious one-pixel shift on a tripod set is pure softening."""
    truth, frames, _ = synthetic.bracket(exposures=EXPOSURES)

    aligned, shifts = align(frames, REFERENCE, "mtb")

    assert all(shift == (0.0, 0.0) for shift in shifts)
    assert all(np.array_equal(a, b) for a, b in zip(aligned, frames))


def test_alignment_can_be_disabled():
    truth, frames, _ = synthetic.bracket(exposures=EXPOSURES)
    drifted = [shift(f, 4, 4) for f in frames]

    aligned, shifts = align(drifted, REFERENCE, "none")

    assert all(s == (0.0, 0.0) for s in shifts)
    assert all(np.array_equal(a, b) for a, b in zip(aligned, drifted))


def test_alignment_survives_a_featureless_frame():
    """A blank frame gives median-threshold alignment nothing to work with."""
    truth, frames, _ = synthetic.bracket(exposures=EXPOSURES)
    frames[0] = np.full_like(frames[0], 0.5)

    aligned, shifts = align(frames, REFERENCE, "mtb")

    assert np.isfinite(aligned[0]).all()
    assert shifts[0] == (0.0, 0.0), "a shift derived from nothing should be rejected"


#: Positions in the well-exposed part of the test scene. Reference-frame
#: deghosting can only judge a pixel the reference itself describes -- where the
#: reference is black, a brighter frame showing detail is indistinguishable from
#: a ghost, and the frame is trusted. So a meaningful test has to put its movers
#: where the reference can actually see them.
MOVER_POSITIONS = [(0, 78), (18, 60), (36, 66)]
MOVER_SIZE = 12
MOVER_STOPS = 3.0


def moving_bracket():
    """A bracket with a dark occluder in a different place in every frame.

    The occluder is defined as a fixed number of stops below the scene behind
    it, rather than as a fixed pixel value. That keeps it equally visible in
    every frame -- a fixed value would clip in the brightest exposure and could
    coincidentally match the scene in another, which makes it undetectable for
    reasons that say nothing about the deghosting.
    """
    truth = synthetic.scene()
    frames = []
    for exposure, (row, column) in zip(EXPOSURES, MOVER_POSITIONS):
        frame = synthetic.render(truth, exposure)
        patch = (slice(row, row + MOVER_SIZE), slice(column, column + MOVER_SIZE))
        frame[patch] = frame[patch] / (2.0 ** MOVER_STOPS)
        frames.append(frame)
    return truth, frames, MOVER_POSITIONS


def test_the_movers_sit_where_the_reference_can_judge_them():
    """Precondition for every deghosting test below."""
    truth, frames, positions = moving_bracket()
    reference = synthetic.render(truth, EXPOSURES[REFERENCE]).max(axis=-1)

    for row, column in positions:
        patch = reference[row:row + MOVER_SIZE, column:column + MOVER_SIZE]
        assert patch.min() > 0.02 and patch.max() < 0.98


def test_deghosting_rejects_the_moving_subject():
    truth, frames, positions = moving_bracket()

    masks = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=0.7)

    for index in (0, 2):
        row, column = positions[index]
        assert masks[index][row + 6, column + 6] < 0.5, "the mover should be rejected"


def test_deghosting_keeps_the_static_scene():
    truth, frames, positions = moving_bracket()

    masks = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=0.7)

    # A corner none of the squares reaches must survive in every frame.
    assert all(mask[-8, -8] > 0.5 for mask in masks)
    assert np.mean(masks[0] > 0.5) > 0.9, "only the mover should be rejected"


def test_the_reference_frame_is_never_masked():
    truth, frames, _ = moving_bracket()

    masks = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=0.7)

    assert np.all(masks[REFERENCE] == 1.0)


def test_deghosting_removes_the_ghost_from_the_merged_result():
    truth, frames, positions = moving_bracket()

    ghosted, _ = merge_radiance(frames, frames, EXPOSURES, REFERENCE)
    masks = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=0.7)
    cleaned, _ = merge_radiance(frames, frames, EXPOSURES, REFERENCE, masks)

    # Where frame 0's square sat, the merge should now follow the reference,
    # which has no square there.
    # Frame 2 is the brightest, so its occluder is still well above the noise
    # floor and carries real weight in the merge -- frame 0's would be
    # discarded by the exposure weighting alone, proving nothing.
    row, column = positions[2]
    patch = (slice(row + 3, row + 9), slice(column + 3, column + 9))
    # merge_radiance anchors its output to the reference exposure, so the
    # reference frame's own values are the target here.
    target = frames[REFERENCE][patch]

    ghost_error = np.abs(ghosted[patch] - target).mean()
    clean_error = np.abs(cleaned[patch] - target).mean()
    assert clean_error < ghost_error / 2


def test_deghosting_can_be_disabled():
    truth, frames, _ = moving_bracket()

    masks = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=0.0)

    assert all(np.all(mask == 1.0) for mask in masks)


def test_a_looser_threshold_rejects_less():
    truth, frames, _ = moving_bracket()

    strict = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=0.3)
    loose = ghost_masks(frames, EXPOSURES, REFERENCE, threshold=2.0)

    assert np.mean(loose[0] < 0.5) < np.mean(strict[0] < 0.5)
