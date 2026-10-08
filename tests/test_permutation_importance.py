"""Permutation importance, the importance that means the same thing for every arm.

Each model's importance cannot be compared with another model's. That is fine while an
analysis compares an arm with itself, but not when it asks whether several arms found the same
thing. These tests cover the measure that answers the second question: what it reports, what it
refuses, and that it leaves the caller's frame alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.features import RANK_BY, top_features
from seqmaestro.harness.importance import describe, permutation_importance
from seqmaestro.models import make_model
from seqmaestro.models.sparse import LeastSquares
from seqmaestro.target import BINARY, CONTINUOUS, target_for


class UsesOneColumn:
    """A model whose score is one column, so that column is all of its importance."""

    target_kind = "binary"

    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


def frame(n: int = 200) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    signal = rng.random(n)
    return pd.DataFrame({"signal": signal, "ignored": rng.random(n)}), (signal > 0.5).astype(int)


def test_the_column_a_model_uses_carries_the_drop():
    X, y = frame()
    scores = permutation_importance(UsesOneColumn("signal"), X, y,
                                    repeats=3, rng=np.random.default_rng(1))
    assert scores["signal"] > 0.3, "shuffling what the model scores on must cost it"
    assert abs(scores["ignored"]) < 0.05, "a column the model never reads cannot matter"


def test_the_callers_frame_is_untouched():
    """The measure shuffles columns, and a caller reusing the frame afterwards must not inherit
    the shuffle."""
    X, y = frame()
    before = X.copy()
    permutation_importance(UsesOneColumn("signal"), X, y, repeats=2, rng=np.random.default_rng(1))
    pd.testing.assert_frame_equal(X, before)


def test_one_class_has_no_measure_rather_than_a_zero():
    """With one class there is no measure, so the result is empty. A mapping of zeros would be
    indistinguishable from a model that ignores every feature."""
    X, _ = frame()
    assert permutation_importance(UsesOneColumn("signal"), X, np.zeros(len(X)),
                                  repeats=3, rng=np.random.default_rng(1)) == {}


def test_no_repeats_means_no_measure():
    X, y = frame()
    assert permutation_importance(UsesOneColumn("signal"), X, y, repeats=0) == {}


def test_the_same_generator_gives_the_same_answer():
    X, y = frame()
    first = permutation_importance(UsesOneColumn("signal"), X, y,
                                   repeats=3, rng=np.random.default_rng(7))
    second = permutation_importance(UsesOneColumn("signal"), X, y,
                                    repeats=3, rng=np.random.default_rng(7))
    assert first == second


def test_only_the_columns_asked_for_are_measured():
    X, y = frame()
    scores = permutation_importance(UsesOneColumn("signal"), X, y, columns=["signal"],
                                    repeats=2, rng=np.random.default_rng(1))
    assert set(scores) == {"signal"}


# ------------------------------------------------------------------ the models it is computed on


def pool(n: int = 240) -> pd.DataFrame:
    """Indicators, counts and continuous columns, the types a pool's columns come in."""
    rng = np.random.default_rng(3)
    return pd.DataFrame({"indicator": rng.integers(0, 2, n).astype(np.uint8),
                         "count": rng.integers(0, 6, n),
                         "ratio": rng.random(n),
                         "noise": rng.random(n)})


def target(X: pd.DataFrame, kind: str) -> np.ndarray:
    rng = np.random.default_rng(4)
    signal = X["indicator"] + X["ratio"] + rng.normal(0, 0.5, len(X))
    return (signal > 1).astype(int).to_numpy() if kind == BINARY else signal.to_numpy()


MODELS = [(BINARY, "l1_logistic"), (BINARY, "ebm"), (BINARY, "xgboost"),
          (CONTINUOUS, "l1_linear"), (CONTINUOUS, "ebm"), (CONTINUOUS, "xgboost")]


def fitted(kind: str, name: str):
    X = pool()
    weighting = {"class_weight": "balanced"} if kind == BINARY else {}
    if name == "least_squares":
        return LeastSquares().fit(X, target(X, kind)), X
    return make_model(name, target_kind=kind, seed=0, **weighting).fit(X, target(X, kind)), X


@pytest.mark.parametrize(("kind", "name"), [*MODELS, (CONTINUOUS, "least_squares")])
def test_a_model_predicts_the_same_from_any_frame_of_its_columns(kind, name):
    """A model predicts from its columns as one array of floats, so a frame of several column types
    and a frame over one array give the same scores."""
    model, X = fitted(kind, name)
    over_one_array = pd.DataFrame(X.to_numpy(dtype=float), columns=X.columns)
    np.testing.assert_array_equal(model.predict(X), model.predict(over_one_array))


@pytest.mark.parametrize(("kind", "name"), [*MODELS, (CONTINUOUS, "least_squares")])
def test_a_model_refuses_columns_other_than_its_own(kind, name):
    """The names and their order are checked before the values are used, as the libraries checked
    them when they were handed the frame."""
    model, X = fitted(kind, name)
    with pytest.raises(ValueError, match="another order"):
        model.predict(X[list(reversed(X.columns))])
    with pytest.raises(ValueError, match=r"missing \['noise'\]"):
        model.predict(X.drop(columns="noise"))


@pytest.mark.parametrize(("kind", "name"), MODELS)
def test_the_importance_is_what_shuffling_the_frame_gives(kind, name):
    """The importance shuffles one array and predicts from it. Shuffling a copy of the frame one
    column at a time, with the same generator, gives the same numbers."""
    model, X = fitted(kind, name)
    y = target(X, kind)
    computed = permutation_importance(model, X, y, repeats=2, rng=np.random.default_rng(5))

    rng, score = np.random.default_rng(5), target_for(kind).score
    baseline = score(y, model.predict(X))
    expected = {}
    for column in X.columns:
        shuffled = X.copy()
        drops = []
        for _ in range(2):
            shuffled[column] = rng.permutation(X[column].to_numpy())
            drops.append(baseline - score(y, model.predict(shuffled)))
        expected[column] = float(np.mean(drops))
    assert computed == expected


def test_what_it_was_measured_against_is_recorded():
    """A number whose metric, repeats and split are not written down cannot be compared later."""
    assert describe(5, "val", "auroc") == {"metric": "auroc", "repeats": 5, "split": "val"}


# ------------------------------------------------------------------ choosing which one to rank by


def records() -> list[FeatureRecord]:
    return [
        FeatureRecord(name="a", description="", kind="positional", importance=0.1,
                      permutation_importance=0.9),
        FeatureRecord(name="b", description="", kind="positional", importance=0.9,
                      permutation_importance=0.1),
    ]


def test_ranking_defaults_to_the_models_importance():
    assert [record.name for record in top_features(records(), 2)] == ["b", "a"]


def test_ranking_can_be_asked_for_the_comparable_one():
    assert [record.name for record in top_features(records(), 2, by="permutation_importance")] \
        == ["a", "b"]


def test_ranking_by_something_that_is_not_an_importance_is_refused():
    with pytest.raises(ValueError, match="Cannot rank by"):
        top_features(records(), 2, by="name")
    assert "permutation_importance" in RANK_BY
