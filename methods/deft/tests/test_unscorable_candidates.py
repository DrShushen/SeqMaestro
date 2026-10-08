"""A candidate that could not be scored must not become a split.

`optimize_features` leaves a candidate it could not score at `score = -inf, threshold = 0` and
**keeps it in the pool** — that is its failure sentinel, set when the feature raised, when the
90-second threshold-search timeout fired, or when no cut point satisfied `min_samples_leaf`.

So the best-scoring candidate could be one that never scored, and `tree.py`'s guard is
`if not optimal_feature:`, which a `FeatureInfo` always passes. Two things then went wrong:
the tree split on a `threshold = 0` that no search ever chose, and — measured — a feature that
raises took the *whole run* down at the split step, after its tokens had been spent. That call
was the one step in the pipeline with nothing around it.

No finite score means no valid split at this node, which is what CART concludes too: the node
becomes a leaf.

All offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.features.feature_finder import (
    CartFeatureFinder,
    FeatureFinder,
    _best_scoring_feature,
)
from deft.provenance import IntegrityPolicy, RunProvenance
from deft.trees.split_criterion import GiniCriterion
from deft.trees.tree import AdaptiveDecisionTree
from deft.utils.feature import FeatureInfo


def scored(name, score, threshold=0.5):
    f = FeatureInfo(fn=lambda d: d["a"], name=name, string=f"def {name}(X): ...")
    f.score, f.threshold = score, threshold
    return f


@pytest.fixture
def data():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.DataFrame({"y": [0, 0, 0, 1, 1, 1]})
    return X, y


# --------------------------------------------------------------------- choosing a candidate


def test_the_best_scoring_candidate_is_chosen():
    chosen = _best_scoring_feature([scored("best", 0.9), scored("worse", 0.1)])

    assert chosen.name == "best"


def test_a_pool_where_nothing_scored_yields_no_split():
    assert _best_scoring_feature([scored("a", -np.inf), scored("b", -np.inf)]) is None


def test_an_unscorable_candidate_is_skipped_for_one_that_scored():
    """Sorted best-first, so the sentinel sorts last — but it must be stepped over, not used."""
    chosen = _best_scoring_feature([scored("real", 0.4), scored("failed", -np.inf)])

    assert chosen.name == "real"


def test_a_candidate_that_was_never_scored_at_all_is_not_chosen():
    """`score` is None before `optimize_features` runs."""
    assert _best_scoring_feature([FeatureInfo(name="unscored")]) is None


def test_an_empty_pool_yields_no_split():
    assert _best_scoring_feature([]) is None


def test_the_cart_finder_applies_the_same_rule(data):
    """Every column is constant, so no cut point exists and nothing can score."""
    X = pd.DataFrame({"a": [1.0] * 6, "b": [2.0] * 6})
    _, y = data

    chosen = CartFeatureFinder().get_optimal_feature(
        X, y, min_samples_leaf=1, splitting_criterion=GiniCriterion(), sequence_column=None
    )

    assert chosen is None


# ------------------------------------------------------------------------- inside a fit


class ReturnsOnly(FeatureFinder):
    """A finder that hands back exactly the feature it is given."""

    def __init__(self, feature):
        super().__init__()
        self.feature = feature

    def generate_features(self, X, y=None):
        return []

    def get_optimal_feature(self, X, y, **kwargs):
        return self.feature


def fit_with(feature, X, y, provenance=None):
    tree = AdaptiveDecisionTree(
        feature_finder=ReturnsOnly(feature),
        splitting_criterion=GiniCriterion(),
        max_depth=2,
        min_samples_leaf=1,
        save_locally=False,
    )
    tree.provenance = provenance
    tree.fit(X, y)
    return tree


def test_a_feature_that_raises_at_the_split_step_costs_a_node_not_the_run(data):
    """This is the measured failure: `ValueError` straight out of `fit`, mid-run."""
    X, y = data

    def explodes(d):
        raise ValueError("boom")

    feature = FeatureInfo(fn=explodes, name="explodes", string="def explodes(X): ...")
    feature.score, feature.threshold = 0.4, 0.5   # scored fine, then misbehaves

    tree = fit_with(feature, X, y)

    assert tree.root._is_leaf(), "the node should have become a leaf"
    assert tree.root.prediction == pytest.approx(0.5)


def test_that_node_is_recorded_as_having_no_split(data):
    """So the integrity policy sees it, rather than it looking like a successful split."""
    X, y = data
    provenance = RunProvenance(policy=IntegrityPolicy(enabled=False))

    def explodes(d):
        raise ValueError("boom")

    feature = FeatureInfo(fn=explodes, name="explodes", string="x")
    feature.score, feature.threshold = 0.4, 0.5

    fit_with(feature, X, y, provenance=provenance)

    assert provenance.nodes[0].split_feature_name is None
    assert provenance.summary()["nodes_split"] == 0


def test_a_fit_where_nothing_can_be_scored_produces_a_single_leaf(data):
    """The whole tree is one leaf rather than a crash or a fabricated split."""
    X, y = data
    tree = fit_with(None, X, y)

    assert tree.root._is_leaf()


def test_a_working_feature_still_splits(data):
    """The guard must not swallow the normal path."""
    X, y = data
    feature = FeatureInfo(fn=lambda d: d["a"], name="a", string="def a(X): return X['a']")
    feature.score, feature.threshold = 0.4, 3.5

    tree = fit_with(feature, X, y)

    assert not tree.root._is_leaf()
    assert tree.root.feature.name == "a"
