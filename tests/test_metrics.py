"""The classification metrics (`seqmaestro.harness.metrics`).

The expected values come from cases whose answer is known independently of the implementation (a
perfect ranker, a perfectly wrong one, a coin flip), so the tests do not simply pin whatever the
code returns. Several tests check the guards that stop a meaningless number from being reported as
a valid one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seqmaestro.harness.metrics import choose_threshold, classification_metrics

PERFECT_TRUE = np.array([0, 0, 1, 1])
PERFECT_SCORE = np.array([0.1, 0.2, 0.8, 0.9])


# ------------------------------------------------------------------------------ known answers


def test_a_perfect_ranker_scores_one_everywhere():
    m = classification_metrics(PERFECT_TRUE, PERFECT_SCORE)

    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["mcc"] == 1.0
    assert (m["fp"], m["fn"]) == (0, 0)


def test_a_perfectly_wrong_ranker_scores_zero_auroc():
    m = classification_metrics(PERFECT_TRUE, 1 - PERFECT_SCORE)

    assert m["auroc"] == 0.0
    assert m["accuracy"] == 0.0
    assert m["mcc"] == -1.0


def test_a_coin_flip_scores_near_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    m = classification_metrics(y, rng.random(4000))

    assert 0.45 < m["auroc"] < 0.55
    assert abs(m["auprc"] - m["prevalence"]) < 0.05, "AUPRC of a random ranker is the prevalence"
    assert abs(m["auprc_lift"] - 1.0) < 0.12


def test_the_confusion_counts_reconstruct_the_thresholded_metrics():
    """The stored confusion counts reproduce the thresholded metrics, so a thresholded metric the
    code does not compute can still be recovered from them."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    m = classification_metrics(y, rng.random(500))

    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["n"]
    assert m["tp"] + m["fn"] == m["n_positive"]
    assert m["accuracy"] == pytest.approx((m["tp"] + m["tn"]) / m["n"])
    assert m["recall"] == pytest.approx(m["tp"] / (m["tp"] + m["fn"]))
    assert m["specificity"] == pytest.approx(m["tn"] / (m["tn"] + m["fp"]))
    assert m["balanced_accuracy"] == pytest.approx((m["recall"] + m["specificity"]) / 2)


def test_brier_and_log_loss_reward_confidence_only_when_it_is_right():
    confident_right = classification_metrics(PERFECT_TRUE, np.array([0.01, 0.01, 0.99, 0.99]))
    hedged = classification_metrics(PERFECT_TRUE, np.array([0.4, 0.4, 0.6, 0.6]))

    assert confident_right["brier"] < hedged["brier"]
    assert confident_right["log_loss"] < hedged["log_loss"]
    assert confident_right["auroc"] == hedged["auroc"] == 1.0, "ranking cannot tell them apart"


def test_calibration_error_separates_a_calibrated_model_from_an_overconfident_one():
    """Calibration error is recorded because it tells apart two models that AUROC cannot: both rank
    identically, but one overstates its certainty."""
    rng = np.random.default_rng(3)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)  # By construction, p is the true probability

    # Sharpen by cubing the odds. The map is strictly increasing on [0, 1] and stays inside it, so
    # the ranking, and therefore AUROC, is unchanged. Clipping a linear stretch would not work: it
    # ties every value it saturates, which changes AUROC and would test the wrong thing.
    sharpened = p**3 / (p**3 + (1 - p) ** 3)

    calibrated = classification_metrics(y, p)
    overconfident = classification_metrics(y, sharpened)

    assert calibrated["expected_calibration_error"] < 0.05
    assert overconfident["expected_calibration_error"] > calibrated["expected_calibration_error"]
    assert overconfident["auroc"] == pytest.approx(calibrated["auroc"], abs=1e-9)


# ------------------------------------------------- the DEFT paper's metrics and the whole set


def test_the_deft_papers_metrics_are_present_and_named_as_it_names_them():
    """Accuracy and AUPRC are the metrics the DEFT paper (arXiv 2604.12060) reports in its Table 1,
    whereas AUROC is SeqMaestro's addition."""
    m = classification_metrics(PERFECT_TRUE, PERFECT_SCORE)

    assert {"accuracy", "auprc", "auroc"} <= set(m)


def test_the_result_is_json_serialisable():
    """The result is written straight into a cell's `metrics.json`, and a numpy scalar in it would
    break the dump."""
    m = classification_metrics(PERFECT_TRUE, PERFECT_SCORE, n_bootstrap=20)

    assert json.loads(json.dumps(m))["auroc"] == 1.0


def test_the_context_needed_to_interpret_the_numbers_travels_with_them():
    m = classification_metrics(PERFECT_TRUE, PERFECT_SCORE)

    assert m["n"] == 4 and m["n_positive"] == 2 and m["prevalence"] == 0.5
    assert m["threshold"] == 0.5


# ------------------------------------------------------------------------------- bootstrap


def test_bootstrap_intervals_bracket_the_point_estimate():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 800)
    score = np.clip(y * 0.3 + rng.random(800) * 0.7, 0, 1)
    m = classification_metrics(y, score, n_bootstrap=200, seed=7)

    for name, (low, high) in m["ci"].items():
        assert low <= m[name] <= high, name


def test_bootstrap_is_reproducible_and_seed_dependent():
    rng = np.random.default_rng(5)
    y, score = rng.integers(0, 2, 300), rng.random(300)

    same = [classification_metrics(y, score, n_bootstrap=100, seed=1)["ci"]["auroc"] for _ in range(2)]
    other = classification_metrics(y, score, n_bootstrap=100, seed=2)["ci"]["auroc"]

    assert same[0] == same[1]
    assert same[0] != other


def test_no_bootstrap_means_no_interval_block():
    assert "ci" not in classification_metrics(PERFECT_TRUE, PERFECT_SCORE)


# ------------------------------------------------------------------------ threshold selection


def test_the_chosen_threshold_beats_the_default_when_scores_are_not_centred():
    """A shallow tree's leaf proportions need not straddle 0.5. At a fixed 0.5 threshold such an
    arm would get a poor accuracy for a reason unrelated to what it learned."""
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.full(50, 0.05), np.full(50, 0.20)])  # Separable, but all below 0.5

    default = classification_metrics(y, score)["balanced_accuracy"]
    tuned = classification_metrics(y, score, threshold=choose_threshold(y, score))["balanced_accuracy"]

    assert default == 0.5, "everything predicted negative"
    assert tuned == 1.0


def test_the_threshold_can_be_chosen_for_a_named_metric():
    y = np.array([0, 0, 0, 1])
    score = np.array([0.1, 0.2, 0.6, 0.7])

    assert 0.0 <= choose_threshold(y, score, metric="f1") <= 1.0


def test_an_unknown_metric_for_selection_is_refused():
    with pytest.raises(ValueError, match="Unknown metric"):
        choose_threshold(PERFECT_TRUE, PERFECT_SCORE, metric="auroc_at_dawn")


# ----------------------------------------------------------------------------------- guards


def test_an_unbounded_decision_function_is_refused_not_silently_scored():
    """Brier score and log loss computed on a raw margin look like valid numbers but mean
    nothing, so scores outside [0, 1] are refused."""
    with pytest.raises(ValueError, match="probabilities in"):
        classification_metrics(PERFECT_TRUE, np.array([-2.0, -1.0, 1.0, 2.0]))


def test_multiclass_labels_are_refused():
    with pytest.raises(ValueError, match="Binary labels"):
        classification_metrics([0, 1, 2], [0.1, 0.5, 0.9])


def test_a_single_class_yields_nan_ranking_rather_than_a_crash():
    """A single class occurs on a degenerate split and in bootstrap resamples of a tiny set, so it
    must give NaN ranking metrics instead of raising."""
    m = classification_metrics([1, 1, 1], [0.2, 0.6, 0.9])

    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"])
    assert m["accuracy"] == pytest.approx(2 / 3), "thresholded metrics are still well defined"


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="labels but"):
        classification_metrics([0, 1], [0.5])


def test_no_samples_is_refused():
    with pytest.raises(ValueError, match="No samples"):
        classification_metrics([], [])


def test_non_finite_scores_are_refused():
    with pytest.raises(ValueError, match="NaN or infinity"):
        classification_metrics([0, 1], [0.5, np.nan])


def test_boolean_labels_are_accepted():
    m = classification_metrics([False, False, True, True], PERFECT_SCORE)

    assert m["auroc"] == 1.0
