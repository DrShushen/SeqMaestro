"""Stage 3B: the statistics of the reliability panel, the sweep of Tier 1's selections, and the
panel end to end.

The statistics are checked against independent computations: Tier 1's bound against its formula,
the vectorised permutation null against `roc_auc_score` one column and one permutation at a time,
and the multiple-testing corrections against `statsmodels`. Each is the kind of code that can be
wrong by a constant or an off-by-one and still look plausible. The panel is checked on a planted
signal, from a run's cells through the reveal to its tables, and across the environment boundary
when DEFT's interpreter is installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from seqmaestro.execution import THREADS_FILE_VARIABLE, THREADS_VARIABLE
from seqmaestro.harness.heldout import TEST_ARRAYS, TEST_SPLIT, SealError, reveal_record
from seqmaestro.harness.stability import impute_non_finite
from seqmaestro.models import MODELS, make_model, one_core
from seqmaestro.models.base import FILLED_ROWS
from seqmaestro.pipeline.rows import Rows
from seqmaestro.reduction import eliminate
from seqmaestro.reduction.arm import fit_step_for, rebuild_model
from seqmaestro.reliability import checks, multitest, prefilter, selections, tier1, tier2, tier3
from seqmaestro.reliability import pool as pools
from seqmaestro.reliability.fits import one_thread
from seqmaestro.reliability.panel import NOT_REPRODUCIBLE, Panel, procedure
from tests.planted import ANALYSES, LANGUAGE_MODEL, SEQUENCE, TARGET, planted_dataset

REPO = Path(__file__).resolve().parents[1]
ADAPTERS = REPO / "experiments" / "adapters"

#: Descriptions as a cell records them, which a rebuild must reproduce in full.
L1 = make_model("l1_logistic", target_kind="binary", class_weight="balanced").describe()
EBM = make_model("ebm", target_kind="binary", class_weight="balanced").describe()

# --------------------------------------------------------------------------------------------
# Tier 1: the bound and its bookkeeping
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("q", "p", "tau", "expected"), [
    (10.0, 800, 0.8, 100 / (0.6 * 800)),
    (10.0, 30, 0.9, 100 / (0.8 * 30)),
    (7.5, 404, 0.55, 56.25 / (0.1 * 404)),
    (10.0, 1000, 1.0, 100 / 1000),
])
def test_the_bound_is_the_formula(q, p, tau, expected):
    assert tier1.expected_false_positives(q, p, tau) == pytest.approx(expected)


def test_the_bound_says_nothing_at_or_below_one_half():
    assert tier1.expected_false_positives(10.0, 800, 0.5) == float("inf")
    assert tier1.expected_false_positives(10.0, 800, 0.4) == float("inf")


def test_the_bound_falls_with_tau_and_with_p():
    taus = [tier1.expected_false_positives(10, 500, t) for t in (0.6, 0.7, 0.8, 0.9)]
    assert taus == sorted(taus, reverse=True)
    ps = [tier1.expected_false_positives(10, p, 0.8) for p in (100, 500, 2000)]
    assert ps == sorted(ps, reverse=True)


def test_the_smallest_tau_meeting_a_bound():
    # q = 10, p = 800: bound <= 1 needs 2 tau - 1 >= 100/800 = 0.125, so tau >= 0.5625.
    assert tier1.smallest_tau_with_bound(10.0, 800, 1.0) == pytest.approx(0.565, abs=0.005)
    assert tier1.smallest_tau_with_bound(10.0, 30, 1.0) is None


@pytest.mark.parametrize("n", [11, 87, 705, 1000])
def test_half_samples_are_disjoint_and_equal(n):
    halves = tier1.half_samples(n, n_pairs=7, seed=3)
    assert halves.shape == (14, n // 2)
    tier1.check_half_samples(halves, n)


def test_half_samples_are_reproducible_and_depend_on_the_seed():
    a = tier1.half_samples(101, 5, seed=1)
    assert np.array_equal(a, tier1.half_samples(101, 5, seed=1))
    assert not np.array_equal(a, tier1.half_samples(101, 5, seed=2))


def test_an_overlapping_pair_is_refused():
    halves = tier1.half_samples(50, 2, seed=0)
    halves[1] = halves[0]
    with pytest.raises(AssertionError):
        tier1.check_half_samples(halves, 50)


def test_selection_frequency_and_q_hat():
    selected = np.zeros((10, 6), dtype=bool)
    selected[:, 0] = True
    selected[:5, 1] = True
    summary = tier1.summarise(selected, p=6)
    assert summary["q_hat"] == pytest.approx(1.5)
    assert summary["curve"][0]["n_selected"] == 1        # tau = 0.55: only column 0
    assert tier1.selection_frequency(selected)[1] == pytest.approx(0.5)


def test_a_group_is_selected_when_any_member_is():
    selected = np.array([[True, False, False], [False, True, False], [False, False, True]])
    collapsed, unique = tier1.collapse(selected, np.array([0, 0, 1]))
    assert list(unique) == [0, 1]
    assert collapsed.tolist() == [[True, False], [True, False], [False, True]]


# --------------------------------------------------------------------------------------------
# Tier 2: the permutation null
# --------------------------------------------------------------------------------------------


def test_the_vectorised_null_is_the_direct_auc():
    """The vectorised rank-sum statistic (`R'Y`) is checked against `roc_auc_score`, one column and
    one permutation at a time. The test rebuilds the whole null matrix with sklearn and requires the
    p-values to agree exactly."""
    rng = np.random.default_rng(11)
    n, p, n_permutations, block = 60, 7, 40, 7
    X = np.column_stack([
        rng.integers(0, 2, n), rng.integers(0, 2, n), rng.poisson(1.5, n),
        rng.poisson(4.0, n), rng.normal(size=n), rng.normal(size=n),
        np.repeat([0.0, 1.0, 2.0], n // 3),          # heavy ties, to exercise tie handling
    ])
    y = (rng.random(n) < 0.45).astype(np.int8)

    out = tier2.profile(X, y, [f"f{j}" for j in range(p)], target_kind="binary", seed=5,
                        n_permutations=n_permutations, block=block)
    direct = np.array([roc_auc_score(y, X[:, j]) for j in range(p)])
    assert out["auc"] == pytest.approx(direct, abs=1e-12)

    rebuilt = np.empty((p, n_permutations))
    rng2 = np.random.default_rng(5)
    for column in range(n_permutations):
        permuted = rng2.permutation(y)
        for j in range(p):
            rebuilt[j, column] = abs(roc_auc_score(permuted, X[:, j]) - 0.5)
    scale = 2 * y.sum() * (n - y.sum())
    at_least = (np.round(rebuilt * scale).astype(np.int64)
                >= np.round(np.abs(direct - 0.5) * scale).astype(np.int64)[:, None]).sum(axis=1)
    assert out["p_value"] == pytest.approx((1 + at_least) / (n_permutations + 1), abs=1e-12)


def test_the_auc_identity_the_vectorisation_rests_on():
    rng = np.random.default_rng(3)
    n = 41
    x = rng.poisson(2.0, n).astype(float)
    y = (rng.random(n) < 0.4).astype(int)
    ranks = rankdata(x)
    n1, n0 = y.sum(), n - y.sum()
    assert (ranks @ y - n1 * (n1 + 1) / 2) / (n1 * n0) == pytest.approx(roc_auc_score(y, x))
    assert ((ranks - (n + 1) / 2) @ y) / (n1 * n0) == pytest.approx(roc_auc_score(y, x) - 0.5)


def test_ties_in_the_null_are_counted_exactly():
    """A binary feature on few rows puts much of its null exactly on the observed value. Counting
    those ties as less extreme would shrink the p-value."""
    rng = np.random.default_rng(17)
    n, n_permutations = 80, 500
    X = rng.integers(0, 2, size=(n, 6)).astype(float)
    y = (rng.random(n) < 0.5).astype(np.int8)
    out = tier2.profile(X, y, [f"f{j}" for j in range(6)], target_kind="binary", seed=2,
                        n_permutations=n_permutations,
                        block=100)

    scale = 2 * int(y.sum()) * (n - int(y.sum()))
    exact = np.round(np.abs(np.array([roc_auc_score(y, X[:, j]) for j in range(6)]) - 0.5) * scale)
    rng2 = np.random.default_rng(2)
    counts = np.zeros(6, dtype=int)
    for _ in range(n_permutations):
        permuted = rng2.permutation(y)
        drawn = np.round(np.abs(
            np.array([roc_auc_score(permuted, X[:, j]) for j in range(6)]) - 0.5) * scale)
        counts += (drawn >= exact).astype(int)
    assert out["p_value"] == pytest.approx((1 + counts) / (n_permutations + 1), abs=1e-12)


def test_the_statistic_is_an_integer_on_any_split():
    """Doubled centred average ranks are integers on a split of any size, so every statistic is an
    integer and every comparison is exact."""
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, size=(5001, 4)).astype(float)
    ranks = tier2.doubled_centred_ranks(X)
    assert np.array_equal(ranks, np.round(ranks))
    y = (rng.random(5001) < 0.5).astype(float)
    assert np.array_equal(ranks.T @ y, np.round(ranks.T @ y))


def test_null_p_values_are_uniform_without_a_signal():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 60))
    y = (rng.random(200) < 0.5).astype(np.int8)
    out = tier2.profile(X, y, [f"f{j}" for j in range(60)], target_kind="binary", seed=1,
                        n_permutations=2000)
    assert out["called"][0.05]["bh"] <= 3
    assert 0.4 < out["p_value"].mean() < 0.6


def test_a_weighted_design_is_refused_by_tier_2():
    tier2.refuse_weighted(None)
    with pytest.raises(tier2.WeightedDesignError, match="per-row weights"):
        tier2.refuse_weighted("fit_weight")


# --------------------------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 37, 785, 6187])
def test_bh_by_and_holm_are_statsmodels(n):
    rng = np.random.default_rng(n)
    p = np.clip(rng.random(n) ** 3, 1e-12, 1.0)
    p[: max(1, n // 10)] = p[0]                       # ties, which is where these go wrong
    for mine, code in ((multitest.benjamini_hochberg(p), "fdr_bh"),
                       (multitest.benjamini_yekutieli(p), "fdr_by"),
                       (multitest.holm(p), "holm")):
        assert mine == pytest.approx(multipletests(p, method=code)[1], rel=1e-12, abs=1e-14)


def test_by_is_never_below_bh():
    p = np.random.default_rng(0).random(500)
    assert (multitest.benjamini_yekutieli(p) >= multitest.benjamini_hochberg(p) - 1e-12).all()


# --------------------------------------------------------------------------------------------
# Step 0
# --------------------------------------------------------------------------------------------


def test_the_prefilter_drops_constant_and_near_constant_columns():
    n = 1000
    X = np.zeros((n, 5))
    X[:, 0] = 1.0                                     # constant
    X[:5, 1] = 1.0                                    # prevalence 0.5%
    X[:, 2] = 1.0
    X[:5, 2] = 0.0                                    # prevalence 99.5%
    X[: n // 2, 3] = 1.0                              # prevalence 50%
    X[:, 4] = np.arange(n)                            # continuous, never modal
    out = prefilter.prefilter(X, [f"f{j}" for j in range(5)], max_modal_share=0.99)
    assert out["kept"].tolist() == [False, False, False, True, True]
    assert out["n_after"] == 2


def test_the_prefilter_keeps_a_continuous_column_that_is_never_zero():
    """The usual binary rule (drop a non-zero prevalence below 1% or above 99%) would delete this
    column. The modal-share rule that generalises it must keep it."""
    X = np.random.default_rng(0).normal(loc=5.0, size=(500, 1))
    assert prefilter.prefilter(X, ["x"], max_modal_share=0.99)["kept"].tolist() == [True]


def test_clusters_honour_their_own_threshold():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(400, 4))
    X = np.column_stack([base[:, 0], base[:, 0] + 0.05 * rng.normal(size=400),
                         base[:, 1], base[:, 2], base[:, 3]])
    rho = prefilter.spearman_abs(X, seed=0)
    labels = prefilter.cluster(rho, threshold=0.8)
    assert labels[0] == labels[1]
    assert prefilter.worst_within_cluster(rho, labels) >= 0.8


def test_an_undefined_value_is_ranked_at_its_columns_median():
    """A DEFT feature written for one node can be NaN on most rows of D (training plus validation).
    The clusters and Tier 2 rank such a value as the median of the column's defined values, as the
    matching across repeats does, because a NaN would otherwise reach the linkage and stop the
    panel."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 3))
    X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=300)
    undefined = rng.random(300) < 0.85
    X[undefined, 2] = np.nan
    imputed = impute_non_finite(X)
    assert (imputed[undefined, 2] == np.median(X[~undefined, 2])).all()

    rho = prefilter.spearman_abs(X, seed=0)
    assert np.isfinite(rho).all()
    np.testing.assert_array_equal(rho, prefilter.spearman_abs(imputed, seed=0))
    labels = prefilter.cluster(rho, threshold=0.8)
    assert labels[0] == labels[1] != labels[2]

    y = X[:, 0] + rng.normal(size=300)
    names = ["a", "b", "c"]
    out = tier2.profile(X, y, names, target_kind="continuous", seed=1, n_permutations=200)
    reference = tier2.profile(imputed, y, names, target_kind="continuous", seed=1, n_permutations=200)
    for key in ("rho", "p_value", "q_bh", "mean"):
        np.testing.assert_array_equal(out[key], reference[key])
    assert np.isfinite(out["p_value"]).all()
    assert prefilter.undefined(X, names) == {"c": round(float(undefined.mean()), 4)}


def test_a_representative_is_the_most_stable_member_then_the_simpler_kind():
    labels = np.array([0, 0, 0])
    kinds = ["llm_generated", "positional", "kmer"]
    assert prefilter.representatives(labels, np.array([0.5, 0.5, 0.9]), kinds, list("abc"))[0] == 2
    assert prefilter.representatives(labels, np.array([0.9, 0.9, 0.9]), kinds, list("abc"))[0] == 1


# --------------------------------------------------------------------------------------------
# Tier 3
# --------------------------------------------------------------------------------------------


def test_the_refits_drop_each_feature_and_each_cluster_of_several():
    plan = tier3.refits(["a", "b", "c", "d"], {"a": 1, "b": 1, "c": 2, "d": 3})
    labels = [entry["label"] for entry in plan]
    assert labels[:4] == ["a", "b", "c", "d"]
    assert "cluster_1" in labels and "cluster_2" not in labels
    assert next(e for e in plan if e["label"] == "cluster_1")["columns"] == ["c", "d"]


def test_a_model_made_worse_is_detected_and_an_unchanged_one_is_not():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.5).astype(int)
    good = np.where(y == 1, 0.8, 0.2)
    worse = np.where(y == 1, 0.55, 0.45)
    hurt = tier3.compare(y, good, worse, target_kind="binary", seed=0, n_sign_flips=2000)
    assert hurt["mean_delta_log_loss"] > 0
    assert hurt["p_t_one_sided"] < 1e-6 and hurt["p_sign_flip"] < 0.01
    same = tier3.compare(y, good, good.copy(), target_kind="binary", seed=0, n_sign_flips=2000)
    assert same["mean_delta_log_loss"] == pytest.approx(0.0)
    assert same["p_t_one_sided"] >= 0.5


def test_per_row_log_loss_is_sklearns():
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(9)
    y = (rng.random(50) < 0.5).astype(int)
    p = rng.uniform(0.05, 0.95, 50)
    assert tier3.per_sample_log_loss(y, p).mean() == pytest.approx(log_loss(y, p, labels=[0, 1]))


# --------------------------------------------------------------------------------------------
# The selection and the fits behind it
# --------------------------------------------------------------------------------------------


def test_the_selection_is_the_reductions_own_loop_and_returns_at_most_n():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 40))
    y = (rng.random(120) < 0.5).astype(np.int8)
    names = [f"f{j}" for j in range(40)]
    chosen = selections.selector(eliminate.run, names, 10)(
        fit_step_for(L1, 0, None, target_kind="binary"), X, y)
    assert len(chosen) <= 10 and set(chosen) <= set(names)
    assert selections.selected_mask(names, chosen).sum() == len(chosen)


class UsesTheFirstOf:
    """A stand-in fit whose model uses only the columns named, so the elimination pads to N."""

    def __init__(self, used: set[str], columns: list[str]):
        self.values = np.array([1.0 if name in used else 0.0 for name in columns])

    def importances(self) -> np.ndarray:
        return self.values

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


def test_a_selection_counts_only_the_features_its_model_uses():
    """A model that uses two columns leaves eight survivors held in place by the tie-break alone,
    which picks the same eight on any rows. If they were counted, they would look stable under any
    labels."""
    names = [f"f{j}" for j in range(40)]
    X, y = np.zeros((6, 40)), np.zeros(6)
    select = selections.selector(eliminate.run, names, 10)
    assert select(lambda X, y, columns: UsesTheFirstOf({"f7", "f30"}, columns), X, y) == ["f7", "f30"]
    assert select(lambda X, y, columns: UsesTheFirstOf(set(), columns), X, y) == []


def test_the_panel_reports_the_features_m_star_uses():
    reduction = {"final_columns": ["a", "b", "c", "d"], "final_importance": [0.4, 0.0, 0.6, 0.0],
                 "n_target": 4}
    panel = Panel("arm", run_dir="run", reduction=reduction, reduced_cell="run/cell", source=None,
                  fits=None, settings=None, discovery=None, n_train=0, columns=None,
                  is_check_arm=False)
    assert panel.reported == ["a", "c"]


def test_the_record_names_the_elimination_it_mirrors():
    digest = procedure()
    assert digest["eliminate_sha256"] == hashlib.sha256(Path(eliminate.__file__).read_bytes()).hexdigest()
    assert digest["keep_fraction"] == eliminate.KEEP_FRACTION


def test_an_ebm_on_one_core_is_the_same_model():
    """Tier 1's workers fit EBMs on one core, while the reported models are fitted without that
    limit. The importances must be identical either way."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 12)).astype(float)
    y = (rng.random(200) < 1 / (1 + np.exp(-(X[:, 0] - X[:, 1])))).astype(int)
    columns = [f"f{j}" for j in range(12)]
    default = fit_step_for(EBM, 3, None, target_kind="binary")(X, y, columns).importances()
    held = fit_step_for(EBM, 3, None, one_core("ebm"), target_kind="binary")(X, y, columns).importances()
    assert one_core("ebm") and np.array_equal(default, held)
    assert one_core("l1_logistic") == {}, "a model that fits on one core anyway needs no option"


def test_a_tier1_fit_holds_the_numeric_libraries_to_one_thread_and_then_lets_go():
    """Tier 1's workers already fill the cores, so a fit in one of them must not start a BLAS
    pool's worth of threads beside it. Once the fit ends, the process's previous thread limits are
    restored."""
    from threadpoolctl import threadpool_info

    def threads() -> int:
        return max((info["num_threads"] for info in threadpool_info()), default=1)

    before = threads()
    seen = []
    step = one_thread(lambda X, y, names: seen.append(threads()) or "fitted")
    assert step(None, None, []) == "fitted"
    assert seen == [1]
    assert threads() == before


def test_only_models_that_can_be_read_are_profiled():
    assert {name for name, model in MODELS["binary"].items() if model.interpretable} == {
        "l1_logistic", "ebm", "pysr"}
    assert {name for name, model in MODELS["continuous"].items() if model.interpretable} == {
        "l1_linear", "ebm", "pysr"}


def planted_matrix(n: int = 150, p: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * X[:, 0] + 2 * X[:, 1]))).astype(np.int8)
    return X, y, [f"f{j}" for j in range(p)]


def test_the_sweep_resumes_where_it_stopped_and_fans_out_to_the_same_answer(tmp_path):
    X, y, names = planted_matrix()
    halves = tier1.half_samples(len(y), 4, seed=1)
    select = selections.selector(eliminate.run, names, 3)

    def make(weights):
        return fit_step_for(L1, 0, weights, target_kind="binary")

    whole, _ = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                              out_path=tmp_path / "whole.npz", workers=1)
    fanned, timing = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                                    out_path=tmp_path / "fanned.npz", workers=3)
    assert np.array_equal(whole, fanned) and timing["n_half_samples"] == 8

    interrupted = dict(np.load(tmp_path / "whole.npz"))
    interrupted["done"][5:] = False
    interrupted["selected"][5:] = False
    np.savez_compressed(tmp_path / "resumed.npz", **interrupted)
    messages = []
    resumed, _ = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                                out_path=tmp_path / "resumed.npz", workers=1, log=messages.append)
    assert np.array_equal(resumed, whole)
    assert "resuming: 5/8" in messages[0]

    with pytest.raises(ValueError, match="other candidates or other half-samples"):
        selections.run(select, X, y, names, tier1.half_samples(len(y), 4, seed=2),
                       make_fit_step=make, weights=None, out_path=tmp_path / "whole.npz", workers=1)


def test_the_workers_are_as_many_as_fit_beside_the_panel():
    """Given the memory one worker was measured to hold, each worker is expected to hold a quarter
    more, beside what the panel holds. The sweep never runs more workers than it was given, nor
    fewer than one."""
    gb = 2**30
    assert selections.workers_for(memory=10 * gb, beside=2 * gb, held=gb, workers=32) == 6
    assert selections.workers_for(memory=10 * gb, beside=2 * gb, held=gb, workers=4) == 4
    assert selections.workers_for(memory=2 * gb, beside=2 * gb, held=gb, workers=32) == 1
    assert selections.workers_for(memory=gb, beside=0, held=0, workers=8) == 8, \
        "a measurement of nothing changes nothing"


def test_the_sweep_measures_one_selection_and_runs_the_workers_that_fit(tmp_path):
    """Given the memory the panel may hold, the sweep runs the first selection alone in a worker
    whose memory it measures, then runs the rest in as many workers as fit. The selections are the
    same however many workers ran."""
    import time

    X, y, names = planted_matrix()
    halves = tier1.half_samples(len(y), 4, seed=1)
    elimination = selections.selector(eliminate.run, names, 3)

    def select(fit_step, X, y):
        """The elimination, in a worker that holds 40 MB for half a second, long enough to be
        measured."""
        held = bytearray(40 * 2**20)
        time.sleep(0.5)
        return elimination(fit_step, X, y) if held else []

    def make(weights):
        return fit_step_for(L1, 0, weights, target_kind="binary")

    whole, _ = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                              out_path=tmp_path / "whole.npz", workers=1)
    messages = []
    roomy, timing = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                                   out_path=tmp_path / "roomy.npz", workers=3,
                                   memory=selections.resident_bytes() + 2**40,
                                   log=messages.append)
    assert np.array_equal(roomy, whole) and timing["workers"] == 3
    assert "so 3 of 3 workers fit" in messages[0]
    messages.clear()
    tight, timing = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                                   out_path=tmp_path / "tight.npz", workers=3,
                                   memory=selections.resident_bytes(), log=messages.append)
    assert np.array_equal(tight, whole) and timing["workers"] == 1
    assert "so 1 of 3 workers fit" in messages[0]


def test_the_sweep_follows_the_threads_it_may_use_as_they_rise(tmp_path):
    """Given a threads file, the sweep runs as many selections at once as the file allows when each
    selection starts. Here the first selection raises the number from one to three, as the driver
    would when threads free up. The selections are the same however many workers ran."""
    X, y, names = planted_matrix()
    halves = tier1.half_samples(len(y), 5, seed=1)
    elimination = selections.selector(eliminate.run, names, 3)
    threads = tmp_path / "threads"

    def select(fit_step, X, y):
        chosen = elimination(fit_step, X, y)
        threads.write_text("3\n")
        return chosen

    def make(weights):
        return fit_step_for(L1, 0, weights, target_kind="binary")

    whole, _ = selections.run(select, X, y, names, halves, make_fit_step=make, weights=None,
                              out_path=tmp_path / "whole.npz", workers=1)
    threads.write_text("1\n")
    messages = []
    followed, timing = selections.run(select, X, y, names, halves, make_fit_step=make,
                                      weights=None, out_path=tmp_path / "followed.npz", workers=1,
                                      threads_file=threads, log=messages.append)
    assert np.array_equal(followed, whole) and timing["workers"] == 3
    assert any("may now use 3 threads, so at most 3 selections run at once" in line
               for line in messages)

    threads.write_text("3\n")
    messages.clear()
    capped, timing = selections.run(select, X, y, names, halves, make_fit_step=make,
                                    weights=None, out_path=tmp_path / "capped.npz", workers=1,
                                    memory=selections.resident_bytes(), threads_file=threads,
                                    log=messages.append)
    assert np.array_equal(capped, whole) and timing["workers"] == 1
    assert "so 1 of 3 workers fit" in messages[0], "the memory still decides"


def test_the_sweep_starts_none_while_more_run_than_the_threads_it_may_use(tmp_path):
    """When the driver takes threads back, the sweep lets the running selections finish and starts
    the next only once fewer run than the file allows. Here the first selection to end lowers the
    number from three to one, so every selection started after that runs alone."""
    X, y, names = planted_matrix()
    halves = tier1.half_samples(len(y), 5, seed=1)
    elimination = selections.selector(eliminate.run, names, 3)
    threads, running, seen = tmp_path / "threads", tmp_path / "running", tmp_path / "seen"
    running.mkdir()

    def select(fit_step, X, y):
        marker = running / str(os.getpid())
        marker.touch()
        with seen.open("a") as handle:
            handle.write(f"{len(list(running.iterdir()))}\n")
        chosen = elimination(fit_step, X, y)
        threads.write_text("1\n")
        marker.unlink()
        return chosen

    threads.write_text("3\n")
    selections.run(select, X, y, names, halves, make_fit_step=lambda weights: fit_step_for(
        L1, 0, weights, target_kind="binary"), weights=None, out_path=tmp_path / "s.npz",
        workers=3, threads_file=threads, log=lambda message: None)
    at_start = [int(line) for line in seen.read_text().split()]
    assert len(at_start) == 10 and at_start[3:] == [1] * 7


def test_the_threads_file_holds_a_number_or_leaves_the_one_given(tmp_path):
    threads = tmp_path / "threads"
    assert selections.threads_in(None, 4) == 4
    assert selections.threads_in(threads, 4) == 4, "no file yet"
    threads.write_text("7\n")
    assert selections.threads_in(threads, 4) == 7
    threads.write_text("")
    assert selections.threads_in(threads, 4) == 4, "a file being written holds no number"


def test_a_failure_in_the_measured_selection_is_raised_with_its_traceback(tmp_path):
    X, y, names = planted_matrix()
    halves = tier1.half_samples(len(y), 2, seed=1)

    def broken(fit_step, X, y):
        raise ValueError("a selection that cannot run")

    with pytest.raises(RuntimeError, match="a selection that cannot run"):
        selections.run(broken, X, y, names, halves, make_fit_step=lambda weights: None,
                       weights=None, out_path=tmp_path / "s.npz", workers=2, memory=2**40)


def test_what_a_worker_adds_is_measured_with_what_it_started(tmp_path):
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "x = bytearray(60 * 2**20); "
                              "open(__import__('sys').argv[1], 'w').close(); "
                              "__import__('time').sleep(30)", str(tmp_path / "ready")])
    try:
        while not (tmp_path / "ready").exists():
            pass
        assert selections.unshared_bytes(child.pid) >= 60 * 2**20
        assert selections.unshared_bytes(child.pid + 2**22) == 0, "a process that is not there"
    finally:
        child.kill()
        child.wait()


def test_every_selection_is_fitted_with_its_own_rows_weights(tmp_path):
    X, y, names = planted_matrix()
    weights = np.arange(len(y), dtype=float)
    halves = tier1.half_samples(len(y), 2, seed=0)
    seen = []

    def make(given):
        seen.append(given)
        return fit_step_for(L1, 0, given, target_kind="binary")

    selections.run(selections.selector(eliminate.run, names, 3), X, y, names, halves,
                   make_fit_step=make, weights=weights, out_path=tmp_path / "s.npz", workers=1)
    assert [list(w) for w in seen] == [list(weights[rows]) for rows in halves]


# --------------------------------------------------------------------------------------------
# The planted-signal recipe
# --------------------------------------------------------------------------------------------


def test_the_planted_recipe_draws_the_same_model_on_any_rows():
    X, _, names = planted_matrix(p=12)
    recipe = checks.planted_recipe(X, names, np.full(12, 0.1), seed=3, n_planted=4, signal_sd=1.5,
                                   target_kind="binary")
    assert len(recipe["names"]) == 4 and recipe["beta"] == [1.0, -1.0, 1.0, -1.0]
    first = checks.draw_planted(X, recipe, seed=5)
    assert np.array_equal(first, checks.draw_planted(X, recipe, seed=5))
    assert set(np.unique(first)) <= {0, 1}
    score = ((X[:, recipe["columns"]] - recipe["mean"]) / recipe["sd"]) @ recipe["beta"]
    assert np.std(score * recipe["scale"]) == pytest.approx(1.5)


# --------------------------------------------------------------------------------------------
# The shared files the vendored side loads
# --------------------------------------------------------------------------------------------


def load_cart_adapter():
    spec = importlib.util.spec_from_file_location("deft_cart_under_test", ADAPTERS / "deft_cart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_sides_name_the_pools_alike():
    adapter = load_cart_adapter()
    assert adapter.POOLS == {"cart": pools.NAIVE_CANDIDATES, "default": pools.RECORDED_FEATURES}
    assert adapter.SHARED["selections"] == REPO / "seqmaestro" / "reliability" / "selections.py"


# --------------------------------------------------------------------------------------------
# The config section
# --------------------------------------------------------------------------------------------

RELIABILITY = {"pairs": 3, "permutations": 200, "cluster_rho": 0.8, "max_modal_share": 0.99,
               "planted_count": 3, "planted_signal_sd": 1.5, "check_arm": "l1_logistic/positional",
               "seed": 0}


def a_config(tmp_path, dataset, models=("l1_logistic",), entry=None, **reliability) -> Path:
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10,
                 "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60, "guidance": None}},
        "language_model": LANGUAGE_MODEL,
        "analyses": ANALYSES,
        "arms": {"positional": {"models": list(models), **(entry or {})}},
        "reduction": {"n_features": 3},
        "reliability": {**RELIABILITY, **reliability},
    }))
    return source


def test_the_panel_profiles_what_can_be_read_less_what_is_excluded(runners, tmp_path):
    dataset = planted_dataset(tmp_path)
    config = runners.config.load(a_config(tmp_path, dataset, models=("l1_logistic", "ebm", "xgboost"),
                                          entry={"not_profiled": {"ebm": "too slow for a test"}}))
    assert config.profiled_arms() == [("l1_logistic", "positional")]
    assert config.not_profiled()["ebm/positional"] == "too slow for a test"
    assert runners.config.load(a_config(tmp_path, dataset)).to_mapping()["reliability"]["check_arm"] \
        == "l1_logistic/positional"


@pytest.mark.parametrize(("change", "message"), [
    ({"exclude": {"ebm/positional": "why"}}, "has no key exclude"),
    ({"arms": ["positional"]}, "has no key arms"),
    ({"check_arm": "xgboost/positional"}, "does not profile"),
    ({"cluster_rho": 1.5}, r"in \(0, 1\]"),
    ({"pairs": 0}, ">= 1"),
    ({"planted": 3}, "has no key planted"),
])
def test_a_reliability_section_that_cannot_run_is_refused(runners, tmp_path, change, message):
    dataset = planted_dataset(tmp_path)
    with pytest.raises(runners.config.ConfigError, match=message):
        runners.config.load(a_config(tmp_path, dataset, models=("l1_logistic", "xgboost"), **change))


def test_a_panel_follows_the_threads_the_driver_gives_it_unless_told_its_workers(
        runners, tmp_path, monkeypatch):
    monkeypatch.setenv(THREADS_VARIABLE, "4")
    monkeypatch.setenv(THREADS_FILE_VARIABLE, str(tmp_path / "threads"))
    parse = runners.reliability.build_parser().parse_args
    assert runners.reliability.tier1_workers(parse(["--run-dir", "run"])) == (4, tmp_path / "threads")
    assert runners.reliability.tier1_workers(parse(["--run-dir", "run", "--workers", "2"])) == (2, None)
    monkeypatch.delenv(THREADS_VARIABLE)
    monkeypatch.delenv(THREADS_FILE_VARIABLE)
    assert runners.reliability.tier1_workers(parse(["--run-dir", "run"])) == (os.cpu_count(), None)


def test_a_panel_needs_a_reduction(runners, tmp_path):
    dataset = planted_dataset(tmp_path)
    source = a_config(tmp_path, dataset)
    document = yaml.safe_load(source.read_text())
    del document["reduction"]
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(runners.config.ConfigError, match="needs a `reduction` section"):
        runners.config.load(source)


# --------------------------------------------------------------------------------------------
# A panel, end to end: a run's cells, reduced, profiled, revealed, tabulated
# --------------------------------------------------------------------------------------------


def a_profiled_run(tmp_path, runners, models=("l1_logistic",), **reliability):
    dataset = planted_dataset(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = runners.config.load(a_config(tmp_path, dataset, models=models, **reliability))
    (run_dir / "config.yaml").write_text(config.to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main(flags) == 0
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    return dataset, run_dir


def test_a_pysr_panel_records_the_reproduction_check_it_did_not_run(runners, tmp_path):
    """PySR's searches are not reproducible, so rerunning the reduction on its fitting rows would
    prove nothing. The panel records that it skipped this check and runs everything else, with one
    Julia thread per search in Tier 1."""
    assert rebuild_model(make_model("pysr", target_kind="binary", seed=1).describe(), 1, "binary",
                         one_core("pysr")).threads == 1
    _, run_dir = a_profiled_run(tmp_path, runners, models=("pysr",), check_arm="pysr/positional",
                                pairs=2)
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "2"]) == 0

    arm = run_dir / "reliability" / "pysr" / "positional"
    prepared = json.loads((arm / "prepare.json").read_text())
    assert prepared["reproduction"] == {"run": False, "reason": NOT_REPRODUCIBLE}
    record = json.loads((arm / "reliability.json").read_text())
    assert all((arm / f"tier1{suffix}.json").exists() for suffix in ("", "_null", "_planted"))
    for entry in record["loco"]:
        metrics = json.loads((arm / entry["cell"] / "metrics.json").read_text())
        assert set(metrics[FILLED_ROWS]) == {"val"}

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    summary = pd.read_csv(run_dir / "reliability" / "summary.csv")
    assert summary["model"].tolist() == ["pysr"] and summary["reproduced_from_pool"].isna().all()


def test_a_panel_is_profiled_revealed_and_tabulated(runners, tmp_path):
    _, run_dir = a_profiled_run(tmp_path, runners)
    assert runners.reveal.main(["--run-dir", str(run_dir), "--check"]) == 2, \
        "the reveal waits for the panel"
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "2"]) == 0

    arm = run_dir / "reliability" / "l1_logistic" / "positional"
    record = json.loads((arm / "reliability.json").read_text())
    reduced = json.loads((run_dir / "reduction" / "l1_logistic" / "positional" /
                          "reduction.json").read_text())
    assert record["reported"] == reduced["final_columns"] and record["is_check_arm"]
    assert record["tier2"] == {"refused": None, "label_sets_at_reveal": ["real", "null", "planted"]}
    prepared = json.loads((arm / "prepare.json").read_text())
    assert prepared["reproduction"]["from_recorded"]["identical"]
    assert prepared["candidates"]["p"] == 80, "the positional vocabulary, not the features the arm used"
    assert all((arm / f"tier1{suffix}.json").exists() for suffix in ("", "_null", "_planted"))
    assert all((arm / f"tier2_discovery{suffix}.json").exists() for suffix in ("", "_planted"))
    assert not any(TEST_SPLIT in path.name for path in arm.iterdir()), "nothing of the test split yet"
    for entry in record["loco"]:
        run = json.loads((arm / entry["cell"] / "run.json").read_text())
        assert run["columns"] == [name for name in record["reported"]
                                  if name not in entry["dropped"]]
        assert run["seed"] == reduced["seed"] and (arm / entry["cell"] / "fitted.pkl").exists()

    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 2, \
        "nothing to tabulate before the reveal"

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    sealed = reveal_record(run_dir)["sealed"]
    assert "reliability/l1_logistic/positional" in sealed
    assert {f"reliability/l1_logistic/positional/{entry['cell']}" for entry in record["loco"]} \
        <= set(sealed)
    assert reveal_record(run_dir)["scored"]["reliability_tier2"] == 1
    for suffix in ("", "_null", "_planted"):
        assert (arm / f"tier2_{TEST_SPLIT}{suffix}.json").exists()

    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    tier3_result = json.loads((arm / "tier3.json").read_text())
    assert tier3_result["n_tests"] == len(record["loco"])
    with np.load(run_dir / "reduction" / "l1_logistic" / "positional" / "reduced" /
                 "predictions.npz") as model:
        assert tier3_result["auroc_model"] == pytest.approx(
            roc_auc_score(model[TEST_ARRAYS[0]], model[TEST_ARRAYS[1]]))
    features = pd.read_csv(arm / "profile_features.csv")
    assert len(features) == 80 and features["in_reported_model"].sum() == len(record["reported"])
    assert features.loc[features["filtered_out"], "pi_hat"].isna().all(), \
        "a filtered candidate has no selection frequency, not a zero"
    summary = pd.read_csv(run_dir / "reliability" / "summary.csv")
    assert summary["model"].tolist() == ["l1_logistic"]
    checks_result = json.loads((arm / "checks.json").read_text())
    assert {"global_null_tier2", "global_null_tier1", "planted_tier1", "planted_tier2",
            "planted_tier2_discovery"} <= set(checks_result)


def test_a_panel_resumes_and_refuses_a_revealed_run(runners, tmp_path, capsys):
    _, run_dir = a_profiled_run(tmp_path, runners)
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "1"]) == 0
    arm = run_dir / "reliability" / "l1_logistic" / "positional"
    written = {path: path.read_bytes() for path in arm.rglob("*") if path.is_file()}
    capsys.readouterr()
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "1"]) == 0
    assert "already profiled" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in written} == written

    (arm / "reliability.json").unlink()
    (arm / "tier1.json").unlink()
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "1"]) == 0
    printed = capsys.readouterr().out
    assert "tier 1 [real]" in printed and "LOCO" not in printed and "pool" not in printed, \
        "an interrupted panel does only what is missing"

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    with pytest.raises(SealError):
        runners.reliability.main(["--run-dir", str(run_dir)])


def test_a_weighted_design_is_profiled_without_tier_2(runners, tmp_path):
    """Tier 1 and the refits use the per-row weights, whereas Tier 2 is refused and the record says
    why."""
    _, run_dir = a_profiled_run(tmp_path, runners)
    config = runners.config.load(run_dir / "config.yaml")
    rows = Rows.of(config)
    for frame in (rows.train, rows.val):
        frame["fit_weight"] = 1.0 + (frame[TARGET] == 1)
    discovery = rows.discovery()
    panel = runners.reliability.panel_for(run_dir, config, ("l1_logistic", "positional"), rows)
    panel.columns = replace(panel.columns, weight="fit_weight")
    record = panel.before_reveal(workers=1, log=lambda message: None)
    assert "per-row weights" in record["tier2"]["refused"]
    assert record["tier2"]["label_sets_at_reveal"] == []
    arm = run_dir / "reliability" / "l1_logistic" / "positional"
    assert not list(arm.glob("tier2_*")), "no unweighted Tier 2 for a weighted design"
    with np.load(arm / "prepare.npz") as prepared:
        assert np.array_equal(prepared["weights"], discovery["fit_weight"].to_numpy())
    loco = json.loads((arm / record["loco"][0]["cell"] / "run.json").read_text())
    assert loco["model_settings"]["per_row_weights"] is True


def test_a_panel_that_does_not_reproduce_its_reduction_stops(runners, tmp_path):
    _, run_dir = a_profiled_run(tmp_path, runners)
    config = runners.config.load(run_dir / "config.yaml")
    panel = runners.reliability.panel_for(run_dir, config, ("l1_logistic", "positional"),
                                          Rows.of(config))
    panel.reported = ["pos_0_A", "pos_1_C", "pos_2_G"]
    with pytest.raises(RuntimeError, match="not the reduction's M"):
        panel.prepare(log=lambda message: None)


# --------------------------------------------------------------------------------------------
# A DEFT arm's panel, across the environment boundary, when DEFT's interpreter is installed
# --------------------------------------------------------------------------------------------


def deft_python() -> Path:
    from seqmaestro.pipeline.vendored import deft as run_deft
    return run_deft.DEFT_PYTHON


@pytest.mark.skipif(not deft_python().exists(), reason="DEFT's interpreter is not installed here")
def test_a_deft_arms_panel_is_fitted_on_defts_side_and_revealed(runners, tmp_path, monkeypatch):
    """A CART cell fitted by the Stage 2 adapter is reduced, profiled and revealed with DEFT's real
    interpreter. Its candidates are the naive population DEFT's finder offers the root, its
    selections and refits are DEFT's trees, and its refits and Tier 2 are revealed through DEFT's
    side. Its selections follow a threads file, as they do under the driver."""
    import subprocess

    from seqmaestro.harness.runner import ingest_bundle, subsample_training_rows
    from seqmaestro.pipeline.vendored import deft as run_deft

    dataset = planted_dataset(tmp_path)
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1], "labels": ["real"], "train_fraction": 0.8,
                 "n_bootstrap": 10, "permutation_scope": "seed",
                 "class_weight": "balanced", "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60, "guidance": None}},
        "language_model": LANGUAGE_MODEL,
        "analyses": ANALYSES,
        "arms": {"positional": {"models": ["deft_or_cart"]}},
        "reduction": {"n_features": 2},
        "reliability": {**RELIABILITY, "check_arm": "deft_or_cart/positional"},
    }))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = runners.config.load(source)
    (run_dir / "config.yaml").write_text(config.to_yaml())
    assert runners.baseline.main(["--run-dir", str(run_dir)]) == 0

    train, val = pd.read_csv(dataset.train_path), pd.read_csv(dataset.val_path)
    cell = run_dir / "results" / "deft_or_cart" / "positional" / "real" / "seed_1"
    rows, y_fit = subsample_training_rows(train, seed=1, columns=dataset.columns,
                                          train_fraction=0.8, permute_labels=False,
                                          permutation_key=None)
    paths = run_deft.write_inputs("positional", rows, y_fit, train, val, cell / "inputs",
                                  columns=dataset.columns)
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    (descriptions / "planted.json").write_text(json.dumps({"task": "Predicting a planted motif."}))
    completed = subprocess.run(
        [str(run_deft.DEFT_PYTHON), str(run_deft.ADAPTER), "fit", "--train", str(paths["train"]),
         "--val", str(paths["val"]), "--effect", str(paths["effect"]),
         "--out-bundle", str(cell / "bundle"), "--feature-finder", "cart",
         "--target-column", TARGET, "--target-kind", "binary", "--sequence-column", SEQUENCE,
         "--task", "planted",
         "--dataset-dir", str(descriptions), "--permutation-repeats", "0",
         "--permutation-seed", "0", "--config-override", "tree.class_weight=balanced"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    ingest_bundle(cell / "bundle", val, target_column=TARGET, target_kind="binary", out_dir=cell,
                  seed=1, n_bootstrap=10,
                  labels="real", arm=("deft_or_cart", "positional"),
                  permutation_scope="seed", train_fraction=0.8, n_fit_rows=len(rows),
                  weight_column=None, predict_with=run_deft.PREDICT_WITH)

    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    threads = tmp_path / "threads"
    threads.write_text("2\n")
    monkeypatch.setenv(THREADS_FILE_VARIABLE, str(threads))
    assert runners.reliability.main(["--run-dir", str(run_dir)]) == 0
    arm = run_dir / "reliability" / "deft_or_cart" / "positional"
    record = json.loads((arm / "reliability.json").read_text())
    assert record["candidates"]["source"] == pools.NAIVE_CANDIDATES
    assert record["candidates"]["p"] == 80, "every (position, base) DEFT's finder offers the root"
    assert (record["model"], record["pool"]) == ("deft_or_cart", "positional")
    assert json.loads((arm / "prepare.json").read_text())["reproduction"]["from_recorded"]["identical"]
    for entry in record["loco"]:
        assert (arm / entry["cell"] / "model.dill").exists()

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert (arm / f"tier2_{TEST_SPLIT}.json").exists()
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    assert json.loads((arm / "tier3.json").read_text())["n_tests"] == len(record["loco"])


def test_a_refit_left_with_no_feature_predicts_the_weighted_prevalence():
    """Where one feature or one cluster is the whole of M* (the arm's reduced model), its Tier 3
    refit has no feature left and predicts the positive rate the fit saw, weighted as M* was."""
    from seqmaestro.reduction.arm import NoFeatures, fit_step_for

    [only] = tier3.refits(["gc"], {"gc": 0})
    assert only["columns"] == []
    y = np.array([1, 1, 1, 0])
    fitted = fit_step_for(L1, 0, None, target_kind="binary")(np.zeros((4, 0)), y, only["columns"])
    assert isinstance(fitted.model, NoFeatures)
    assert fitted.predict(np.zeros((2, 0))).tolist() == [0.5, 0.5]
    unweighted = fit_step_for({**L1, "class_weight": None}, 0, None, target_kind="binary")
    assert unweighted(np.zeros((4, 0)), y, []).predict(np.zeros((1, 0))).tolist() == [0.75]
