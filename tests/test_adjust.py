"""The in-program post-processing controls."""

import numpy as np
import pytest

from hdrmerge.adjust import Adjustments, apply_display, apply_linear, luminance


@pytest.fixture
def image():
    """A smooth ramp with colour, so tonal and colour controls both bite."""
    ramp = np.linspace(0.02, 0.98, 64, dtype=np.float32)
    base = np.repeat(ramp[None, :], 48, axis=0)
    return np.stack([base, base * 0.85, base * 0.7], axis=-1).astype(np.float32)


def test_defaults_change_nothing(image):
    assert np.array_equal(apply_display(image, Adjustments()), image)
    assert np.array_equal(apply_linear(image, Adjustments()), image)
    assert Adjustments().is_identity()


def test_every_result_stays_in_range(image):
    extreme = Adjustments(
        exposure=3.0, temp=100, tint=-100, highlights=-100, shadows=100,
        whites=100, blacks=-100, contrast=100, saturation=100, vibrance=100,
    )

    result = apply_display(image, extreme)

    assert result.min() >= 0.0 and result.max() <= 1.0
    assert np.isfinite(result).all()


def test_exposure_is_a_doubling_per_stop():
    linear = np.full((4, 4, 3), 0.1, dtype=np.float32)

    assert np.allclose(apply_linear(linear, Adjustments(exposure=1.0)), 0.2)
    assert np.allclose(apply_linear(linear, Adjustments(exposure=-1.0)), 0.05)


def test_shadows_lift_the_dark_end_and_leave_the_bright_end(image):
    result = apply_display(image, Adjustments(shadows=60))

    before, after = luminance(image), luminance(result)
    dark, bright = before < 0.2, before > 0.8
    assert after[dark].mean() > before[dark].mean() + 0.01
    assert after[bright].mean() == pytest.approx(before[bright].mean(), abs=0.02)


def test_highlights_pull_the_bright_end_and_leave_the_dark_end(image):
    result = apply_display(image, Adjustments(highlights=-60))

    before, after = luminance(image), luminance(result)
    dark, bright = before < 0.2, before > 0.8
    assert after[bright].mean() < before[bright].mean() - 0.01
    assert after[dark].mean() == pytest.approx(before[dark].mean(), abs=0.02)


def test_contrast_spreads_the_tones_apart(image):
    more = apply_display(image, Adjustments(contrast=60))
    less = apply_display(image, Adjustments(contrast=-60))

    assert luminance(more).std() > luminance(image).std()
    assert luminance(less).std() < luminance(image).std()


def test_saturation_moves_colours_away_from_grey(image):
    saturated = apply_display(image, Adjustments(saturation=50))
    desaturated = apply_display(image, Adjustments(saturation=-100))

    def spread(array):
        return float((array.max(axis=-1) - array.min(axis=-1)).mean())

    assert spread(saturated) > spread(image)
    assert spread(desaturated) < 0.01, "-100 should be monochrome"


def test_vibrance_spares_already_saturated_colours():
    """The difference between vibrance and saturation, stated as a test."""
    muted = np.full((8, 8, 3), 0.5, dtype=np.float32)
    muted[..., 0] = 0.55                                  # barely coloured
    vivid = np.tile(np.array([0.9, 0.1, 0.1], dtype=np.float32), (8, 8, 1))

    def boost(image, adjustments):
        """How much wider the colour spread got, as a ratio.

        A ratio rather than a difference: the vivid patch starts eight times
        more colourful than the muted one, so it gains more in absolute terms
        under any setting at all, which would say nothing about vibrance.
        """
        result = apply_display(image, adjustments)
        before = float((image.max(axis=-1) - image.min(axis=-1)).mean())
        after = float((result.max(axis=-1) - result.min(axis=-1)).mean())
        return after / before

    vibrance = Adjustments(vibrance=80)
    saturation = Adjustments(saturation=80)

    # Vibrance treats the two very differently; saturation treats them alike.
    assert boost(muted, vibrance) > 1.5 * boost(vivid, vibrance)
    assert boost(vivid, vibrance) < boost(vivid, saturation)

    # Saturation, by contrast, applies its full multiplier regardless of how
    # colourful the pixel already was.
    assert boost(muted, saturation) == pytest.approx(1.8, rel=0.02)


def test_white_balance_shifts_colour_without_shifting_brightness(image):
    warm = apply_display(apply_linear(image, Adjustments(temp=60)), Adjustments())
    cool = apply_display(apply_linear(image, Adjustments(temp=-60)), Adjustments())

    assert warm[..., 0].mean() > image[..., 0].mean()
    assert warm[..., 2].mean() < image[..., 2].mean()
    assert cool[..., 2].mean() > cool[..., 0].mean() - image[..., 0].mean() + image[..., 2].mean()
    assert luminance(warm).mean() == pytest.approx(luminance(image).mean(), rel=0.1)


def test_tint_trades_green_against_magenta(image):
    green = apply_linear(image, Adjustments(tint=-60))
    magenta = apply_linear(image, Adjustments(tint=60))

    assert green[..., 1].mean() > magenta[..., 1].mean()


def test_black_and_white_points_move_the_ends(image):
    """Positive whites brighten and negative blacks deepen, as in a raw editor."""
    result = apply_display(image, Adjustments(blacks=-80, whites=80))

    assert result.min() < image.min(), "blacks should crush the dark end"
    assert result.max() >= image.max(), "whites should brighten the light end"
    assert luminance(result).std() > luminance(image).std()


def test_gamma_brightens_or_darkens_the_midtones(image):
    brighter = apply_display(image, Adjustments(gamma=2.0))
    darker = apply_display(image, Adjustments(gamma=0.5))

    assert brighter.mean() > image.mean() > darker.mean()


def test_colour_boosts_fade_out_in_the_deepest_shadows():
    """Colour down there is noise, and boosting it prints speckle."""
    noisy_shadow = np.zeros((16, 16, 3), dtype=np.float32)
    noisy_shadow[..., 0] = 0.012          # a one-code-value channel imbalance
    noisy_shadow[..., 1] = 0.004
    noisy_shadow[..., 2] = 0.004

    boosted = apply_display(noisy_shadow, Adjustments(vibrance=100, saturation=100))

    spread_before = float((noisy_shadow.max(axis=-1) - noisy_shadow.min(axis=-1)).mean())
    spread_after = float((boosted.max(axis=-1) - boosted.min(axis=-1)).mean())
    assert spread_after < spread_before * 1.3

    # Midtones must be unaffected by the guard.
    midtone = np.tile(np.array([0.55, 0.5, 0.5], dtype=np.float32), (16, 16, 1))
    lifted = apply_display(midtone, Adjustments(saturation=100))
    assert (lifted.max(axis=-1) - lifted.min(axis=-1)).mean() > 0.08


def test_auto_preset_is_a_real_grade(image):
    auto = Adjustments.auto()

    assert not auto.is_identity()
    assert auto.shadows > 0 and auto.highlights < 0
    assert not np.allclose(apply_display(image, auto), image)


def test_explicit_settings_override_the_auto_preset():
    auto = Adjustments.auto(shadows=0.0, contrast=50.0)

    assert auto.shadows == 0.0
    assert auto.contrast == 50.0
    assert auto.highlights == Adjustments.AUTO["highlights"]   # untouched


def test_describe_lists_only_what_changed():
    assert Adjustments().describe() == "none"
    assert Adjustments(contrast=20).describe() == "contrast=20"
