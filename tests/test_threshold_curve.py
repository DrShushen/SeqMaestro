"""`choose_threshold` must return the same threshold as a naive implementation, only faster.

The fast form derives the four confusion counts from one sort, whereas the naive form calls the
whole metric suite once per candidate cut point. The two must be indistinguishable, so the naive
form is kept here as the oracle and the two are compared over many random and adversarial inputs.
The oracle is defined in this file and not imported from the module, because a reference that
changes with the code it checks cannot catch a regression in it.
"""

from __future__ import annotations

import numpy as np
import pytest

from seqmaestro.harness.metrics import (
    HIGHER_IS_BETTER,
    THRESHOLDED_METRICS,
    _metrics,
    choose_threshold,
)

METRICS = sorted(THRESHOLDED_METRICS)


def naive_choose_threshold(y_true, y_score, metric="balanced_accuracy"):
    """The reference the fast path must match: the full metric suite, recomputed per candidate."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    candidates = np.unique(y_score)
    candidates = np.concatenate([[0.0], (candidates[:-1] + candidates[1:]) / 2, [1.0]])
    scored = [(_metrics(y_true, y_score, t).get(metric, np.nan), t) for t in candidates]
    scored = [(value, t) for value, t in scored if np.isfinite(value)]
    if not scored:
        return 0.5
    best = max(scored) if HIGHER_IS_BETTER[metric] else min(scored)
    return float(best[1])


def random_case(n, seed, separation=1.4, prevalence=0.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    s = rng.normal(np.where(y == 1, separation, -separation), 1.0)
    return y, 1 / (1 + np.exp(-s))


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("seed", range(6))
def test_matches_the_naive_implementation(metric, seed):
    y, p = random_case(180, seed)
    assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "prevalence,separation",
    [(0.02, 1.4), (0.98, 1.4), (0.5, 0.0), (0.5, 6.0)],
    ids=["rare-positives", "rare-negatives", "no-signal", "separable"],
)
def test_matches_under_skew_and_degenerate_signal(metric, prevalence, separation):
    """Skewed classes and a useless ranker are where zero-division and tie-breaking errors show
    up."""
    y, p = random_case(200, 11, separation=separation, prevalence=prevalence)
    assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


@pytest.mark.parametrize("metric", METRICS)
def test_matches_with_heavy_score_ties(metric):
    """Coarse scores, such as a tree's leaf proportions, collapse many rows onto one cut point."""
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.5).astype(int)
    p = np.round(rng.random(300) * 4) / 4  # only 0, 0.25, 0.5, 0.75, 1
    assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


@pytest.mark.parametrize("metric", METRICS)
def test_matches_when_every_score_is_identical(metric):
    y = np.array([0, 1, 0, 1, 1])
    p = np.full(5, 0.3)
    assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("label", [0, 1])
def test_matches_when_one_class_is_absent(metric, label):
    """`balanced_accuracy` averages over the classes present in `y_true`, so the two cases (no
    positives, no negatives) are not symmetric."""
    y = np.full(20, label)
    p = np.linspace(0.05, 0.95, 20)
    assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


def test_scores_at_the_interval_boundaries_are_handled():
    # 0.0 and 1.0 are also the first and last candidates, so an off-by-one in `searchsorted`
    # would surface here and nowhere else.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.4, 0.6, 1.0])
    for metric in METRICS:
        assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


@pytest.mark.parametrize("metric", sorted(set(HIGHER_IS_BETTER) - THRESHOLDED_METRICS))
def test_threshold_free_metrics_behave_as_before(metric):
    """AUROC and the other threshold-free metrics are identical at every cut point, so the choice
    is vacuous. It must still return the candidate the oracle returns (the largest)."""
    y, p = random_case(120, 7)
    assert choose_threshold(y, p, metric) == pytest.approx(naive_choose_threshold(y, p, metric))


def test_the_chosen_threshold_really_is_optimal():
    """Independently of the oracle, the threshold returned scores at least as well as every other
    candidate."""
    y, p = random_case(150, 5)
    chosen = choose_threshold(y, p, "balanced_accuracy")
    best = _metrics(y, p, chosen)["balanced_accuracy"]
    candidates = np.unique(p)
    for t in np.concatenate([[0.0], (candidates[:-1] + candidates[1:]) / 2, [1.0]]):
        assert _metrics(y, p, t)["balanced_accuracy"] <= best + 1e-12


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="No samples"):
        choose_threshold(np.array([]), np.array([]))


def test_is_fast_enough_to_use_on_a_real_validation_split():
    """The time bound guards against quadratic complexity and is not a performance target.

    At 40,000 rows a quadratic implementation would take roughly half an hour, whereas the linear
    one takes milliseconds. A time between the two indicates a bug, so the bound sits orders of
    magnitude from both, which also keeps the test from failing spuriously when the suite runs in
    parallel and every core is busy.
    """
    import time

    y, p = random_case(40_000, 1)
    started = time.time()
    choose_threshold(y, p, "balanced_accuracy")
    elapsed = time.time() - started
    assert elapsed < 30.0, f"took {elapsed:.1f}s: threshold selection has gone superlinear"
