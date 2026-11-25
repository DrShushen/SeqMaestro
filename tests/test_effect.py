"""Measuring one feature against the label, and the two ways a confounder gets into the answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqmaestro.harness.effect import (
    direct_effect,
    match_within_strata,
    matched_effect,
    proportion_ratio_interval,
    stratified_permutation_p,
)


def confounded(n: int = 2000, seed: int = 0):
    """A dataset where the label is a property of the stratum and of nothing else.

    `is_the_stratum` describes the stratum instead of the row, yet an unstratified test calls it
    the strongest effect present. The stratified tools under test exist to catch this.
    """
    rng = np.random.default_rng(seed)
    strata = rng.integers(0, 20, n)
    label = (rng.random(n) < (strata / 30 + 0.1)).astype(int)
    values = np.column_stack([
        strata.astype(float),                      # is_the_stratum
        rng.normal(size=n),                        # noise
        label * 0.5 + rng.normal(size=n),          # a real within-stratum effect
    ])
    return values, label, strata, ["is_the_stratum", "noise", "real_effect"]


# --------------------------------------------------------------------------------- matching

def test_matching_leaves_every_stratum_with_equal_classes():
    _, label, strata, _ = confounded()
    kept = match_within_strata(strata, label, seed=1)
    counts = pd.crosstab(strata[kept], label[kept])
    assert (counts[0] == counts[1]).all()


def test_matching_drops_a_stratum_holding_one_class():
    label = np.array([1, 1, 0, 0, 1, 1])
    strata = np.array(["a", "a", "a", "a", "b", "b"])       # b is positives only
    kept = match_within_strata(strata, label, seed=0)
    assert set(strata[kept]) == {"a"}


def test_matching_keeps_the_whole_minority_class():
    """Matching drops rows from the abundant class only, so the minority class limits the matched
    size."""
    label = np.array([1, 0, 0, 0, 0])
    strata = np.zeros(5)
    kept = match_within_strata(strata, label, seed=0)
    assert label[kept].sum() == 1
    assert len(kept) == 2


def test_a_ratio_keeps_more_of_the_abundant_class_and_no_more_of_the_scarce_one():
    """Matching 1:2 instead of 1:1 gives a split more rows, but no more positives."""
    label = np.array([1] * 10 + [0] * 40)
    strata = np.zeros(50)
    one_to_one = match_within_strata(strata, label, seed=0, ratio=1)
    one_to_two = match_within_strata(strata, label, seed=0, ratio=2)
    assert label[one_to_one].sum() == label[one_to_two].sum() == 10
    assert len(one_to_one) == 20 and len(one_to_two) == 30


def test_a_ratio_the_stratum_cannot_supply_drops_it():
    label = np.array([1, 1, 0, 0, 0])
    assert len(match_within_strata(np.zeros(5), label, seed=0, ratio=4)) == 0


def test_a_ratio_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        match_within_strata(np.zeros(4), np.array([1, 1, 0, 0]), ratio=0)


def test_matching_returns_nothing_when_no_stratum_holds_both_classes():
    """Both classes are present overall, yet the design has no contrast anywhere in it."""
    kept = match_within_strata(np.array(["a", "a", "b", "b"]), np.array([1, 1, 0, 0]), seed=0)
    assert len(kept) == 0


# --------------------------------------------------------------------------- ratio interval

def test_a_ratio_of_equal_proportions_is_one_and_its_interval_spans_it():
    ratio, low, high = proportion_ratio_interval(50, 200, 50, 200)
    assert ratio == pytest.approx(1.0)
    assert low < 1.0 < high


def test_a_larger_sample_gives_a_tighter_interval():
    _, small_low, small_high = proportion_ratio_interval(50, 200, 40, 200)
    _, large_low, large_high = proportion_ratio_interval(500, 2000, 400, 2000)
    assert (large_high - large_low) < (small_high - small_low)


def test_a_rate_of_zero_has_no_interval_on_the_log_scale():
    assert all(np.isnan(v) for v in proportion_ratio_interval(0, 200, 40, 200))


# ------------------------------------------------------------------------------ the null

def test_the_null_sees_a_confounder_when_it_is_not_told_about_the_strata():
    values, label, _, _ = confounded()
    p = stratified_permutation_p(values, label, None, draws=400, seed=0)
    assert p[0] < 0.01                                        # is_the_stratum
    assert p[2] < 0.01                                        # real_effect


def test_the_null_absorbs_a_confounder_once_it_is_permuted_within_the_strata():
    """Permuting within strata tests whether a column differs between classes inside a stratum, so
    the same column and estimate stop looking significant."""
    values, label, strata, _ = confounded()
    p = stratified_permutation_p(values, label, strata, draws=400, seed=0)
    assert p[0] > 0.2                                         # is_the_stratum, now unremarkable
    assert p[2] < 0.01                                        # real_effect survives


def test_a_within_stratum_permutation_preserves_each_stratum_class_counts():
    """If it did not, the class sizes within a stratum would vary between draws and the contrast
    would drift."""
    from seqmaestro.harness.effect import _within_stratum_permutations

    label = np.array([1.0, 0, 0, 1, 1, 0, 0, 0])
    codes = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    drawn = _within_stratum_permutations(label, codes, 32, np.random.default_rng(0))
    for code in (0, 1):
        assert (drawn[:, codes == code].sum(axis=1) == label[codes == code].sum()).all()


def test_no_p_value_is_ever_zero():
    """A permutation test cannot support a p-value of zero, because the observed arrangement is one
    of the possible arrangements."""
    values, label, strata, _ = confounded()
    p = stratified_permutation_p(values, label, strata, draws=100, seed=0)
    assert (p >= 1 / 101).all()


def test_asking_for_no_draws_returns_nothing_rather_than_a_p_of_one():
    """With no draws, `1 / (0 + 1)` would look like a measured null result backed by no
    evidence."""
    values, label, strata, _ = confounded()
    assert np.isnan(stratified_permutation_p(values, label, strata, draws=0)).all()


def test_a_missing_stratum_is_refused():
    with pytest.raises(ValueError, match="belongs to a stratum"):
        stratified_permutation_p(np.zeros((3, 1)), np.array([0, 1, 0]),
                                 np.array(["a", None, "b"], dtype=object), draws=10)


def test_a_single_class_is_not_a_contrast():
    with pytest.raises(ValueError, match="two-class contrast"):
        direct_effect(np.zeros((4, 1)), np.ones(4), ["x"], draws=10)


# ---------------------------------------------------------------------------- the table

def test_a_binary_feature_reports_rates_and_an_interval():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 4000)
    values = (rng.random(4000) < 0.2 + 0.15 * label).astype(float)[:, None]
    row = direct_effect(values, label, ["flag"], draws=200, seed=0).iloc[0]
    assert row.binary
    assert row.mean_positive == pytest.approx(0.35, abs=0.05)
    assert row.ci_lo < row.ratio < row.ci_hi


def test_a_continuous_feature_gets_no_proportion_interval():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 500)
    row = direct_effect(rng.normal(size=500)[:, None], label, ["x"], draws=100).iloc[0]
    assert not row.binary
    assert np.isnan(row.ci_lo) and np.isnan(row.ci_hi)


def test_a_feature_that_takes_both_signs_reports_no_ratio():
    """For a feature that takes both signs, a class mean can pass through zero and flip the ratio's
    sign, so the ratio's magnitude would mean nothing."""
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 500)
    row = direct_effect(rng.normal(size=500)[:, None], label, ["x"], draws=100).iloc[0]
    assert np.isnan(row.ratio)


def test_the_table_refuses_to_guess_which_name_belongs_to_which_column():
    with pytest.raises(ValueError, match="column"):
        direct_effect(np.zeros((10, 3)), np.array([0, 1] * 5), ["only", "two"], draws=10)


def test_the_design_travels_with_the_table():
    values, label, strata, names = confounded()
    frame = direct_effect(values, label, names, strata=strata, draws=50, seed=3)
    assert frame.attrs["design"]["strata"] == 20
    assert frame.attrs["design"]["n"] == len(label)


# ------------------------------------------------------------------------- matched_effect

def test_the_matched_estimate_reports_how_far_redrawing_moves_it():
    values, label, strata, names = confounded()
    frame = matched_effect(values, label, strata, names, repeats=8, draws=200, seed=0)
    assert frame.attrs["design"]["match_repeats"] == 8
    assert (frame.ratio_draw_min <= frame.ratio_draw_max).all()
    assert (frame.difference_draw_sd >= 0).all()


def test_a_rare_feature_moves_more_between_draws_than_a_common_one():
    """The spread across draws is reported because matching samples the abundant class down, and a
    feature with few events is the most sensitive to which rows are kept."""
    rng = np.random.default_rng(0)
    n = 4000
    strata = rng.integers(0, 10, n)
    label = (rng.random(n) < 0.15).astype(int)
    values = np.column_stack([(rng.random(n) < 0.5).astype(float),
                              (rng.random(n) < 0.005).astype(float)])
    frame = matched_effect(values, label, strata, ["common", "rare"],
                           repeats=12, draws=100, seed=0)
    spread = frame.set_index("feature").difference_draw_sd
    assert spread["rare"] < spread["common"]        # smaller in absolute terms, larger relatively
    relative = frame.set_index("feature")
    assert ((relative.ratio_draw_max - relative.ratio_draw_min)["rare"]
            > (relative.ratio_draw_max - relative.ratio_draw_min)["common"])


def test_matching_inside_the_estimate_removes_what_the_null_also_removes():
    """Matching inside the estimate and permuting within strata both remove the confounder, so each
    correction checks the other."""
    values, label, strata, names = confounded()
    frame = matched_effect(values, label, strata, names, repeats=2, draws=400, seed=0)
    stratum_row = frame.set_index("feature").loc["is_the_stratum"]
    assert abs(stratum_row.difference) < 1.0        # matched away from the estimate
    assert stratum_row.p_permutation > 0.2          # and absent from the null
