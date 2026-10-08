"""Threshold search (`optimize_threshold`) and the timeout that guards it.

B4 in the plan expected this to be the wall-clock bottleneck of a real pilot run, and to need
vectorising. Measurement said otherwise (~2% of a node's cost) so it was left alone. These tests
cover the correctness properties it had none for, and the one real hazard in the code around it.

All offline.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

from deft.code_guard import compile_feature_function
from deft.features import helpers
from deft.features.helpers import optimize_threshold, timeout
from deft.trees.split_criterion import GiniCriterion


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n)
    # A label correlated with the feature, so there is a real best split to find.
    labels = (values > 0.3).astype(int)
    X = pd.DataFrame({"v": values})
    return X, pd.DataFrame({"label": labels})


def feature(fn=lambda d: d["v"]):
    from deft.utils.feature import FeatureInfo

    return FeatureInfo(name="v", string="def v(X):\n    return X['v']", fn=fn)


# --------------------------------------------------------------------------------------------
# Correctness properties that had no test
# --------------------------------------------------------------------------------------------


def test_finds_a_split_that_separates_the_classes():
    X, y = data()
    f = optimize_threshold(GiniCriterion(), feature(), X, y, min_samples_leaf=5)

    assert f.score > -np.inf
    # The planted boundary is 0.3; the search should land near it.
    assert 0.2 < f.threshold < 0.4


def test_min_samples_leaf_is_respected():
    """Every threshold considered must leave at least `min_samples_leaf` on both sides."""
    X, y = data(n=100)
    msl = 30
    f = optimize_threshold(GiniCriterion(), feature(), X, y, min_samples_leaf=msl)

    n_left = int((X["v"].to_numpy() <= f.threshold).sum())
    assert n_left >= msl and (len(X) - n_left) >= msl


def test_no_admissible_split_leaves_the_feature_unselectable():
    """`min_samples_leaf` larger than half the data admits nothing; the score must stay -inf so
    the feature can never win, rather than defaulting to something plausible."""
    X, y = data(n=40)
    f = optimize_threshold(GiniCriterion(), feature(), X, y, min_samples_leaf=30)

    assert f.score == -np.inf


def test_a_constant_feature_is_unselectable():
    X, y = data(n=100)
    f = optimize_threshold(
        GiniCriterion(), feature(fn=lambda d: pd.Series(1.0, index=d.index)), X, y, 5
    )

    assert f.score == -np.inf


def test_the_feature_is_computed_on_a_copy():
    """A generated feature that mutates its argument must not corrupt the training data."""
    X, y = data(n=100)
    original = X["v"].copy()

    def vandal(d):
        d["v"] = 0.0
        return pd.Series(np.arange(len(d)), index=d.index, dtype=float)

    optimize_threshold(GiniCriterion(), feature(fn=vandal), X, y, 5)

    pd.testing.assert_series_equal(X["v"], original)


def test_only_a_naive_candidate_runs_on_the_rows_themselves():
    """DEFT's own candidates only read the rows, so they are not given a copy, which on a frame of
    thousands of columns costs more than the search. Any other candidate still is."""
    from deft.provenance import ORIGIN_LLM, ORIGIN_NAIVE

    X, y = data(n=100)
    seen = {}

    def note(origin):
        def fn(d):
            seen[origin] = d is X
            return d["v"]
        return fn

    for origin in (ORIGIN_NAIVE, ORIGIN_LLM):
        candidate = feature(fn=note(origin))
        candidate.origin = origin
        optimize_threshold(GiniCriterion(), candidate, X, y, 5)
    assert seen == {ORIGIN_NAIVE: True, ORIGIN_LLM: False}


def test_scoring_is_deterministic():
    X, y = data()
    first = optimize_threshold(GiniCriterion(), feature(), X, y, 5)
    second = optimize_threshold(GiniCriterion(), feature(), X, y, 5)

    assert (first.threshold, first.score) == (second.threshold, second.score)


# --------------------------------------------------------------------------------------------
# The timeout, and the one real hazard around it
# --------------------------------------------------------------------------------------------


def test_the_timeout_bounds_a_runaway_feature():
    @timeout(seconds=1, default="gave up")
    def never_returns():
        import time

        time.sleep(5)
        return "finished"

    assert never_returns() == "gave up"



def test_the_timeout_holds_where_the_call_catches_every_error():
    """A generated feature often catches `Exception` row by row, and so does the threshold search's
    loop. A timeout either of them could catch would stop one row and let the rest run on, with no
    limit left. Bounded, so that such a failure costs this test seconds."""
    @timeout(seconds=1, default="gave up")
    def catches_every_error():
        import time

        caught = 0
        for _ in range(20):
            try:
                time.sleep(0.25)
            except Exception:
                caught += 1
        return f"finished, having caught {caught}"

    assert catches_every_error() == "gave up"


def test_a_feature_out_of_time_is_screened_as_out_of_time():
    """The screen's `except Exception` used to catch the timeout before `timeout` could, so a
    feature that ran out of time was dropped as having raised an error. This one also catches with
    a bare `except:`, which the guard compiles as `except Exception:`."""
    slow = compile_feature_function(
        "def slow(X):\n"
        "    total = 0\n"
        "    for _ in range(100):\n"
        "        try:\n"
        "            for _ in range(10 ** 6):\n"
        "                total += 1\n"
        "        except:\n"
        "            total = -1\n"
        "    return X['v'] * total\n"
    )
    screened = timeout(seconds=1, default=helpers._TIMED_OUT)(helpers._compute_feature.__wrapped__)
    assert screened(slow, pd.DataFrame({"v": [1.0, 2.0]})) is helpers._TIMED_OUT

def test_off_the_main_thread_the_missing_timeout_is_announced(caplog):
    """SIGALRM is main-thread only, so off it the guard silently is not there.

    Wrap `fit` in a thread pool and every per-feature timeout disappears without a word, and one
    pathological generated feature can hang the run forever. It now says so.
    """
    import deft.features.helpers as helpers

    helpers._OFF_THREAD_TIMEOUT_WARNED = False

    @timeout(seconds=1, default=None)
    def quick():
        return "done"

    result = {}
    with caplog.at_level(logging.WARNING):
        thread = threading.Thread(target=lambda: result.update(value=quick()))
        thread.start()
        thread.join()

    assert result["value"] == "done", "it must still run, just unguarded"
    assert "timeout is NOT in effect" in caplog.text


def test_the_warning_is_emitted_only_once():
    """Per-feature, per-node, this would otherwise flood the log."""
    import deft.features.helpers as helpers

    helpers._OFF_THREAD_TIMEOUT_WARNED = False

    @timeout(seconds=1, default=None)
    def quick():
        return 1

    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record)  # type: ignore[assignment]
    helpers.logger.addHandler(handler)
    try:
        for _ in range(3):
            thread = threading.Thread(target=quick)
            thread.start()
            thread.join()
    finally:
        helpers.logger.removeHandler(handler)

    warnings = [r for r in seen if r.levelno == logging.WARNING]
    assert len(warnings) == 1
