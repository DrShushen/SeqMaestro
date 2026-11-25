"""Class weighting: its definition, and a check that every model applies it.

The main risk is a model that accepts `class_weight`, records it in its manifest, and fits
unweighted anyway. A results table would not show that failure, so each model is checked by its
behaviour on data where weighting has to change the answer, instead of by the configuration it
reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqmaestro.models import make_model
from seqmaestro.models.weights import (
    resolve_class_weight,
    resolve_fit_weights,
    sample_weight,
)


def test_balanced_matches_sklearns_definition():
    y = np.array([0] * 90 + [1] * 10)
    weights = sample_weight(y, "balanced")
    # n_samples / (n_classes * count): 100/(2*90) and 100/(2*10).
    assert weights[y == 0] == pytest.approx(100 / 180)
    assert weights[y == 1] == pytest.approx(100 / 20)
    # Each class has the same total weight, which is the property the rest of the code relies on.
    assert weights[y == 0].sum() == pytest.approx(weights[y == 1].sum())


def test_balanced_weights_are_one_when_classes_are_equal():
    y = np.array([0] * 50 + [1] * 50)
    assert sample_weight(y, "balanced") == pytest.approx(np.ones(100))


def test_unweighted_returns_none_rather_than_ones():
    """A model handed uniform weights is not always identical to one handed none."""
    y = np.array([0] * 90 + [1] * 10)
    assert sample_weight(y, None) is None
    assert sample_weight(y, "none") is None


def test_single_class_has_no_defined_ratio():
    assert sample_weight(np.zeros(20), "balanced") is None


def test_unknown_class_weight_is_refused():
    with pytest.raises(ValueError, match="Unknown class_weight"):
        resolve_class_weight("inverse")


def _imbalanced(n: int = 400, seed: int = 0):
    """One informative column, one uninformative column and 10% positives, with enough noise that a
    model has to choose between the classes."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * (n // 10) + [0] * (n - n // 10))
    signal = rng.normal(loc=y * 1.2, scale=1.0)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    return X, y


@pytest.mark.parametrize("name", ["l1_logistic", "ebm", "xgboost"])
def test_every_model_actually_applies_the_weights(name):
    """Weighting must change the fitted model as well as its manifest.

    The comparison uses the mean predicted probability, because upweighting the minority class
    raises it in all three model families, whereas accuracy can move either way.
    """
    X, y = _imbalanced()
    unweighted = make_model(name, target_kind="binary", seed=0, class_weight="none").fit(X, y).predict(X)
    weighted = make_model(name, target_kind="binary", seed=0, class_weight="balanced").fit(X, y).predict(X)
    assert weighted.mean() > unweighted.mean() + 0.05


@pytest.mark.parametrize("name", ["l1_logistic", "ebm", "xgboost"])
def test_class_weight_is_recorded_in_the_manifest(name):
    """A result has to say which configuration produced it."""
    assert make_model(name, target_kind="binary").describe()["class_weight"] == "balanced"
    assert make_model(name, target_kind="binary", class_weight="none").describe()["class_weight"] is None


@pytest.mark.parametrize("name", ["l1_logistic", "ebm", "xgboost"])
def test_balanced_is_the_default(name):
    """A caller that states no class weighting gets the default, so the default is pinned here."""
    assert make_model(name, target_kind="binary").class_weight == "balanced"


# --- per-row fitting weights ---------------------------------------------------------------------


def test_per_row_weights_replace_class_weighting_rather_than_compounding():
    """A matched design expressed as per-row weights already balances the classes, so applying a
    class correction on top would correct twice."""
    y = np.array([0] * 90 + [1] * 10)
    given = np.full(100, 2.0)
    # Rescaled to mean 1, so the constant vector comes back as ones rather than twos.
    assert resolve_fit_weights(y, "balanced", given) == pytest.approx(np.ones(100))
    assert resolve_fit_weights(y, None, given) == pytest.approx(np.ones(100))
    # Without per-row weights the class-weight path is unchanged.
    assert resolve_fit_weights(y, "balanced", None) == pytest.approx(sample_weight(y, "balanced"))
    assert resolve_fit_weights(y, None, None) is None


@pytest.mark.parametrize("bad,message", [
    (np.ones(5), "entries for"),
    (np.array([1.0, -1.0] + [1.0] * 98), "finite and non-negative"),
    (np.array([np.nan] + [1.0] * 99), "finite and non-negative"),
    (np.zeros(100), "sums to zero"),
])
def test_per_row_weights_are_validated(bad, message):
    with pytest.raises(ValueError, match=message):
        resolve_fit_weights(np.array([0] * 90 + [1] * 10), None, bad)


@pytest.mark.parametrize("name", ["l1_logistic", "ebm", "xgboost"])
def test_every_model_applies_per_row_weights(name):
    """Weighting only half of the positives must change the fit, and the result must differ from
    the class-weighted fit, because a `class_weight` mapping cannot express that weighting."""
    X, y = _imbalanced(seed=1)
    half = np.zeros(len(y), dtype=bool)
    half[: len(y) // 2] = True
    rows = np.where((y == 1) & half, 30.0, 1.0)

    by_rows = make_model(name, target_kind="binary", seed=0).fit(X, y, sample_weight=rows).predict(X)
    by_class = make_model(name, target_kind="binary", seed=0, class_weight="balanced").fit(X, y).predict(X)
    unweighted = make_model(name, target_kind="binary", seed=0, class_weight="none").fit(X, y).predict(X)
    assert not np.allclose(by_rows, by_class)
    assert not np.allclose(by_rows, unweighted)


@pytest.mark.parametrize("name", ["l1_logistic", "ebm", "xgboost"])
def test_per_row_weights_are_recorded_in_the_manifest(name):
    X, y = _imbalanced()
    model = make_model(name, target_kind="binary", seed=0)
    assert model.describe()["per_row_weights"] is False
    model.fit(X, y, sample_weight=np.ones(len(y)))
    assert model.describe()["per_row_weights"] is True


def test_per_row_weights_are_rescaled_to_mean_one():
    """The total weight sets how much the data counts against a regularisation penalty. Weights that
    balance strata are typically tiny, so without rescaling they would make the effective
    regularisation many times stronger. The rescaling must leave the ratios between weights
    unchanged."""
    y = np.array([0] * 8 + [1] * 2)
    given = np.array([0.01] * 8 + [0.04] * 2)
    out = resolve_fit_weights(y, None, given)
    assert out.sum() == pytest.approx(len(y))
    assert out[8] / out[0] == pytest.approx(given[8] / given[0])
    # `balanced` weights also sum to the number of rows, which keeps the two kinds comparable.
    assert sample_weight(y, "balanced").sum() == pytest.approx(len(y))
