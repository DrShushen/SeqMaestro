"""Which way a feature points: the shared vocabulary and each model's route to it.

A direction is worth recording only if two things hold. Every model must reach the same word,
whatever its machinery (a coefficient, a shape function, an attribution). A model must also be able
to say that there is no single direction, instead of reporting a net trend that hides a reversal.
Both are checked here on data where the answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqmaestro.features import make_featuriser
from seqmaestro.harness.direction import (
    BINARY_WORDS,
    CONTINUOUS_WORDS,
    DIRECTION_KINDS,
    DIRECTIONS,
    HIGHER_IN_NEGATIVE,
    HIGHER_IN_POSITIVE,
    MONOTONICITY_FLOOR,
    NON_MONOTONE,
    dependence_curve,
    from_curve,
    from_increments,
    from_sign,
)
from seqmaestro.harness.effect import rank_biserial
from seqmaestro.models import MODELS, make_model

SEQUENCE, TARGET = "window", "outcome"


def planted(n=400, length=40, seed=0):
    """Sequences whose class is decided by the GC content of their first quarter.

    The signal runs in one direction and in known columns, so a direction that comes out the other
    way is a bug instead of an artefact of the data.
    """
    rng = np.random.default_rng(seed)
    seqs = pd.Series(["".join(rng.choice(list("ACGT"), length)) for _ in range(n)])
    head = seqs.str.slice(0, length // 4)
    y = ((head.str.count("G") + head.str.count("C")) > length // 8).astype(int).to_numpy()
    return seqs, y


# ------------------------------------------------------------------------------ the primitive


@pytest.mark.parametrize("increments,expected", [
    ([1.0], HIGHER_IN_POSITIVE),
    ([-1.0], HIGHER_IN_NEGATIVE),
    ([1.0, 2.0, 3.0], HIGHER_IN_POSITIVE),
    ([-1.0, -2.0], HIGHER_IN_NEGATIVE),
    # Rises, then falls back the whole way. A net of zero over real movement is what NON_MONOTONE
    # exists for, and calling it "no direction" would hide that the feature does something.
    ([3.0, -3.0], NON_MONOTONE),
    ([1.0, -0.9], NON_MONOTONE),
    # Doubles back, but not far enough to stop the curve being described by its trend.
    ([3.0, -0.5], HIGHER_IN_POSITIVE),
])
def test_the_shared_primitive_reads_a_sequence_of_steps(increments, expected):
    assert from_increments(increments, BINARY_WORDS) == expected


def test_a_feature_that_moves_the_model_nowhere_has_no_direction():
    """A flat response is the absence of a direction, so it gives `None` instead of one of the
    words."""
    assert from_increments([0.0, 0.0], BINARY_WORDS) is None
    assert from_increments([], BINARY_WORDS) is None


def test_non_finite_steps_are_dropped_rather_than_poisoning_the_answer():
    assert from_increments([2.0, np.nan], BINARY_WORDS) == HIGHER_IN_POSITIVE
    assert from_increments([np.nan, np.inf], BINARY_WORDS) is None


def test_the_floor_is_the_stated_convention_and_nothing_else_decides():
    """Monotonicity exactly at the floor counts as a direction, and below it does not. The test
    states this so that a change to `MONOTONICITY_FLOOR` has to update it, instead of surprising a
    reader of the output."""
    up, down = 1.0, -(1.0 - MONOTONICITY_FLOOR) / (1.0 + MONOTONICITY_FLOOR)
    assert from_increments([up, down], BINARY_WORDS) == HIGHER_IN_POSITIVE
    assert from_increments([1.0, -0.999], BINARY_WORDS) == NON_MONOTONE


def test_a_curve_is_read_as_its_steps():
    assert from_curve([0.0, 1.0, 2.0], BINARY_WORDS) == HIGHER_IN_POSITIVE
    assert from_curve([0.0, 3.0, 0.0], BINARY_WORDS) == NON_MONOTONE
    assert from_curve([5.0], BINARY_WORDS) is None


def test_a_single_signed_number_reads_as_its_sign():
    assert from_sign(0.3, BINARY_WORDS) == HIGHER_IN_POSITIVE
    assert from_sign(-0.3, BINARY_WORDS) == HIGHER_IN_NEGATIVE
    # Zero is a tie and None means nothing was recorded. Neither is an answer of "neither class".
    assert from_sign(0.0, BINARY_WORDS) is None
    assert from_sign(None, BINARY_WORDS) is None
    assert from_sign(float("nan"), BINARY_WORDS) is None


# -------------------------------------------------------------------------- the marginal effect


def test_the_marginal_effect_is_positive_when_the_feature_is_higher_in_the_positive_class():
    y = np.array([0, 0, 0, 1, 1, 1])
    values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    assert rank_biserial(values, y) == pytest.approx([1.0])
    assert rank_biserial(-values, y) == pytest.approx([-1.0])


def test_a_feature_that_separates_nothing_scores_zero():
    y = np.array([0, 1, 0, 1])
    assert rank_biserial(np.array([[1.0], [1.0], [2.0], [2.0]]), y) == pytest.approx([0.0])


def test_the_marginal_effect_is_unmoved_by_a_monotone_reparameterisation():
    """A count and a fraction of the same thing over the same window are one feature. A rank
    statistic scores them alike, whereas a difference of means does not, so the effect uses
    ranks."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    values = rng.normal(size=(200, 1)) + y[:, None]
    assert rank_biserial(values, y) == pytest.approx(rank_biserial(np.exp(values) * 7, y))


def test_a_column_carrying_a_non_finite_value_is_refused_rather_than_repaired():
    y = np.array([0, 0, 1, 1])
    values = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 3.0], [4.0, 4.0]])
    scored = rank_biserial(values, y)
    assert scored[0] == pytest.approx(1.0)
    assert np.isnan(scored[1])


def test_one_class_leaves_the_separation_undefined_rather_than_zero():
    assert np.isnan(rank_biserial(np.array([[1.0], [2.0]]), np.array([1, 1]))).all()


def test_the_marginal_effect_ignores_which_way_the_labels_are_encoded():
    """Positive means higher in the positive class, which is taken to be the larger label, so a
    dataset labelled 1/2 instead of 0/1 does not silently invert every direction in the report."""
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert rank_biserial(values, np.array([0, 0, 1, 1])) == \
           pytest.approx(rank_biserial(values, np.array([1, 1, 2, 2])))


# ------------------------------------------------------------------------------- the dependence curve


def test_a_dependence_curve_follows_the_feature_it_is_binned_on():
    x = np.linspace(0, 1, 100)
    assert from_curve(dependence_curve(x, 3 * x), BINARY_WORDS) == HIGHER_IN_POSITIVE
    assert from_curve(dependence_curve(x, -3 * x), BINARY_WORDS) == HIGHER_IN_NEGATIVE
    # Helps at both ends and hurts in the middle, which a net trend would report as no effect.
    assert from_curve(dependence_curve(x, (x - 0.5) ** 2), BINARY_WORDS) == NON_MONOTONE


def test_a_two_valued_feature_yields_two_points_rather_than_eight_copies():
    x = np.array([0.0] * 50 + [1.0] * 50)
    assert len(dependence_curve(x, x)) == 2


def test_every_point_of_a_curve_is_a_bin_of_rows_and_the_largest_value_is_in_the_last():
    """The row at a feature's largest value belongs to the top bin. As a point of its own, one
    row's contribution would count for as much as a whole bin's mean."""
    x = np.arange(80.0)
    contributions = np.where(x == x.max(), -100.0, x)
    curve = dependence_curve(x, contributions)
    assert len(curve) == 8
    assert from_curve(curve, BINARY_WORDS) == HIGHER_IN_POSITIVE


def test_rows_that_share_a_value_share_a_bin():
    """A feature that is zero on most rows gives one point for its zeros and the others for the
    rows above them, whereas edges at the quantiles of its values would all fall on zero."""
    x = np.array([0.0] * 90 + list(range(1, 11)))
    curve = dependence_curve(x, np.where(x > 0, 1.0, 0.0))
    assert curve.tolist() == [0.0, 1.0]
    assert from_curve(curve, BINARY_WORDS) == HIGHER_IN_POSITIVE


def test_a_constant_feature_has_no_curve_to_read():
    assert dependence_curve(np.ones(20), np.arange(20.0)).size == 0


def test_a_curve_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="against"):
        dependence_curve(np.arange(5.0), np.arange(4.0))


# ------------------------------------------------------------------------------- each model's route


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_every_model_reports_a_direction_for_the_features_it_used(model_name):
    seqs, y = planted()
    X = make_featuriser("positional").fit_transform(seqs, y)
    model = make_model(model_name, target_kind="binary", seed=0).fit(X, y)

    assert model.direction_kind in DIRECTION_KINDS
    found = model.directions(X)
    assert found, f"{model_name} reported no directions at all"
    assert set(found) <= set(model.importances()), "a direction for a feature the model did not use"
    assert set(found.values()) <= set(DIRECTIONS)


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_a_model_agrees_with_the_data_where_the_signal_was_planted(model_name):
    """Each model reaches a direction by different machinery, so the shared vocabulary is useful
    only if the words come out the same on a signal every model can see.

    The check uses the features each model relied on most, as ranked by its importance, because a
    direction taken from a feature the model barely used reflects noise.
    """
    seqs, y = planted()
    X = make_featuriser("positional").fit_transform(seqs, y)
    model = make_model(model_name, target_kind="binary", seed=0).fit(X, y)

    importances = model.importances()
    found = model.directions(X)
    leant_on = sorted(found, key=lambda name: -abs(importances[name]))[:10]
    marginal = rank_biserial(X[leant_on].to_numpy(), y)

    agreed = sum(found[name] == from_sign(value, BINARY_WORDS)
                 for name, value in zip(leant_on, marginal, strict=True))
    # A sparse model may lean on fewer than ten, and an equation on only a few.
    assert agreed >= 0.8 * len(leant_on), (
        f"{model_name} disagreed with the data on {len(leant_on) - agreed} of its top {len(leant_on)}")


def test_a_linear_fit_can_never_be_non_monotone():
    """A linear fit's direction is the sign of its coefficient, which is why `l1_logistic` needs no
    rows to give one. If a change made `non_monotone` reachable, the sign would not decide the
    direction on its own."""
    seqs, y = planted()
    X = make_featuriser("positional").fit_transform(seqs, y)
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(X, y)
    assert NON_MONOTONE not in set(model.directions(X).values())


def test_the_boosted_model_refuses_a_frame_it_was_not_fitted_on():
    """TreeSHAP's columns are matched to features by position, so a frame with different columns
    would attribute one feature's contributions to another. XGBoost's feature-name check prevents
    that."""
    seqs, y = planted()
    featuriser = make_featuriser("positional")
    X = featuriser.fit_transform(seqs, y)
    model = make_model("xgboost", target_kind="binary", seed=0).fit(X, y)

    with pytest.raises(Exception, match=r"(?i)feature|name|mismatch"):
        model.directions(X.rename(columns={X.columns[0]: "not_a_feature"}))


def test_a_continuous_target_takes_its_own_words_and_their_opposites():
    from seqmaestro.harness.direction import DECREASES_TARGET, INCREASES_TARGET, OPPOSITE

    assert from_sign(0.3, CONTINUOUS_WORDS) == INCREASES_TARGET
    assert from_sign(-0.3, CONTINUOUS_WORDS) == DECREASES_TARGET
    assert from_curve([0.0, 3.0, 0.0], CONTINUOUS_WORDS) == NON_MONOTONE
    assert OPPOSITE[INCREASES_TARGET] == DECREASES_TARGET and set(CONTINUOUS_WORDS) <= set(DIRECTIONS)
