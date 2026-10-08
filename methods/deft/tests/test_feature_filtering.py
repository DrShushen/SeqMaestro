"""`filter_features` and `screen_features`: the gate every generated feature passes before it
can be scored.

It is what stands between a model-written function and the candidate pool. A regression here
would quietly reduce how much of a tree is DEFT's, which is exactly the failure mode
`src/deft/provenance.py` exists to catch. Each rejection also says why, in words the repair round
sends back to the model.

Naive features do not pass through here (they are generated locally and go straight to
`optimize_features`), so this is the LLM path only.

All offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.features.helpers import filter_features, screen_features
from deft.utils.feature import FeatureInfo


@pytest.fixture
def X():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "s": list("ACGT")})


def feature(fn, name="f"):
    return FeatureInfo(fn=fn, name=name, string=f"def {name}(X): ...")


def kept(fn, X):
    return bool(filter_features([feature(fn)], X))


# ------------------------------------------------------------------------------ what survives


def test_a_well_behaved_feature_is_kept(X):
    assert kept(lambda d: d["a"], X)


def test_a_boolean_feature_is_kept(X):
    """Positional nucleotide-style features return booleans; they must survive the float cast."""
    assert kept(lambda d: d["s"] == "A", X)


def test_a_numpy_array_is_kept(X):
    assert kept(lambda d: np.arange(len(d), dtype=float), X)


def test_a_plain_list_is_kept(X):
    """The contract with the model is 'return the column'; it does not promise a Series."""
    assert kept(lambda d: [1.0, 2.0, 3.0, 4.0], X)


def test_a_column_shaped_frame_is_kept(X):
    assert kept(lambda d: d[["a"]], X)


# ------------------------------------------------------------------------------ what is rejected


def test_a_feature_that_raises_is_dropped(X):
    """The commonest case: generated code that compiles but does not run on this node's data."""
    def explodes(d):
        raise ValueError("boom")

    assert not kept(explodes, X)


def test_a_scalar_is_dropped(X):
    """A feature that aggregates the whole column instead of returning one value per row."""
    assert not kept(lambda d: 1.0, X)


def test_a_wrong_length_result_is_dropped(X):
    assert not kept(lambda d: pd.Series([1.0, 2.0]), X)


def test_a_result_containing_nan_is_dropped(X):
    """At the default `max_undefined_share` of 0. NaN compares false against every threshold, so
    those rows go right at every split, which a run has to choose to allow."""
    assert not kept(lambda d: pd.Series([1.0, np.nan, 3.0, 4.0]), X)


def test_a_constant_result_is_dropped(X):
    """It cannot separate anything, and would still cost a full threshold search."""
    assert not kept(lambda d: pd.Series([7.0] * 4), X)


def test_a_non_numeric_result_is_dropped(X):
    """Splitting is by threshold, so a feature returning labels has nothing to threshold on."""
    assert not kept(lambda d: pd.Series(list("abcd")), X)


def test_none_is_dropped(X):
    assert not kept(lambda d: None, X)


# ------------------------------------------------------------------------------------- batching


def test_one_bad_feature_does_not_lose_the_batch(X):
    """A whole generation round must not be lost to a single unusable candidate."""
    features = [
        feature(lambda d: d["a"], "good_one"),
        feature(lambda d: 1.0, "scalar"),
        feature(lambda d: d["a"] * 2, "good_two"),
    ]

    surviving = [f.name for f in filter_features(features, X)]

    assert surviving == ["good_one", "good_two"]


def test_the_input_data_is_not_mutated(X):
    """Features are run on a defensive copy, so one that writes to `X` cannot corrupt training
    data for every feature scored after it."""
    def hostile(d):
        d["a"] = 0.0
        return d["a"] + 1

    filter_features([feature(hostile)], X)

    assert X["a"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_an_empty_batch_is_fine(X):
    assert filter_features([], X) == []


# ------------------------------------------------------------------------------ the reasons


def reason(fn, X, **settings):
    [(_, failure, detail)] = screen_features([feature(fn)], X, **settings)[1]
    return failure, detail


def explodes(d):
    raise ValueError("boom")


def test_each_rejection_says_why(X):
    assert reason(explodes, X) == ("raised", "raised ValueError: boom")
    assert reason(lambda d: 1.0, X)[0] == "wrong_shape"
    assert reason(lambda d: pd.Series([1.0, 2.0]), X) == ("wrong_shape",
                                                          "returned 2 values for 4 rows")
    assert reason(lambda d: pd.Series(list("abcd")), X)[0] == "wrong_shape"
    assert reason(lambda d: pd.Series([7.0] * 4), X)[0] == "constant"
    assert reason(lambda d: pd.Series([np.nan] * 4), X) == ("undefined",
                                                            "returned NaN on every row")


def undefined_on_one_row_in_ten(d):
    return d["a"].where(d["a"] > 0)


def test_a_feature_undefined_on_a_few_rows_is_kept_up_to_the_allowed_share():
    X = pd.DataFrame({"a": np.arange(10.0)})
    candidate = feature(undefined_on_one_row_in_ten)
    assert filter_features([candidate], X, max_undefined_share=0.2)
    assert not filter_features([candidate], X, max_undefined_share=0.0)
    assert reason(undefined_on_one_row_in_ten, X) == (
        "undefined", "returned NaN on 10% of the rows, more than the 0% allowed")


def test_a_feature_constant_where_it_is_defined_is_constant_whatever_the_share():
    X = pd.DataFrame({"a": np.arange(10.0)})
    assert reason(lambda d: pd.Series([1.0] * 9 + [np.nan]), X,
                  max_undefined_share=0.5)[0] == "constant"
