import functools
import logging
import signal
import threading
from collections.abc import Mapping
from copy import copy
from typing import Any

import numpy as np

from deft.provenance import ORIGIN_NAIVE

logger = logging.getLogger(__name__)

_OFF_THREAD_TIMEOUT_WARNED = False


class _OutOfTime(BaseException):
    """Raised inside a call when its time runs out.

    A `BaseException`, so that no `except Exception` between the signal and `timeout` can catch
    it: not one in a generated feature that handles its own errors row by row, not the one in
    `_compute_feature`, and not the one in `optimize_threshold`'s search loop. Any of them would
    take the timeout for one more error and carry on with no limit left. The guard compiles a bare
    `except:` in generated code as `except Exception:` for the same reason. It never leaves
    `timeout`, which returns the default in its place.
    """


def timeout(seconds=60, default=None):
    """Bound a call with SIGALRM, which only works on the main thread.

    Off the main thread the bound is silently unavailable, so the call runs unguarded. That
    matters because this decorator is what stops one pathological generated feature from hanging
    a whole run: wrap `fit` in a thread pool and every per-feature timeout disappears without a
    word. The first such call now says so.

    When the time runs out, the call is stopped with `_OutOfTime`, and the decorated function
    returns `default`.
    """

    def decorator(func):
        """Handle decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # SIGALRM only works in the main thread; fallback to direct execution
            # to avoid runtime crashes in worker threads.
            """Handle wrapper."""
            if threading.current_thread() is not threading.main_thread():
                global _OFF_THREAD_TIMEOUT_WARNED
                if not _OFF_THREAD_TIMEOUT_WARNED:
                    _OFF_THREAD_TIMEOUT_WARNED = True
                    logger.warning(
                        "%s is running off the main thread, where SIGALRM is unavailable, so "
                        "its %ds timeout is NOT in effect. A generated feature that never "
                        "returns will hang this run.",
                        func.__name__,
                        seconds,
                    )
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.info(f"Got the error: {e}")
                    return default

            armed = True

            def handle_timeout(signum, frame):
                # Only while armed: a signal already on its way when the call ends must not raise
                # outside it.
                if armed:
                    raise _OutOfTime

            previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
            try:
                try:
                    signal.alarm(seconds)
                    return func(*args, **kwargs)
                finally:
                    armed = False
                    signal.alarm(0)
            except _OutOfTime:
                return default
            except Exception as e:
                logger.info(f"Got the error: {e}")
                return default
            finally:
                signal.signal(signal.SIGALRM, previous_handler)

        return wrapper

    return decorator


#: Why a generated candidate cannot be used at a node, as `screen_features` names it.
RAISED, WRONG_SHAPE, UNDEFINED, CONSTANT = "raised", "wrong_shape", "undefined", "constant"

#: Which entry of `params_generation.repair_guidance` says how to correct each failure. Code that
#: raises or returns the wrong shape is at fault; a constant or mostly undefined feature is not.
GUIDANCE_FOR = {RAISED: "code_fault", WRONG_SHAPE: "code_fault", CONSTANT: "constant",
                UNDEFINED: "undefined"}

#: What a call returns when it does not finish in time, told apart from a feature returning None.
_TIMED_OUT = object()


class _Raised:
    """What a call returns when the feature raised, so the error survives the timeout wrapper."""

    def __init__(self, error: Exception):
        self.error = error


@timeout(seconds=60, default=_TIMED_OUT)
def _compute_feature(fn, data):
    """One feature on these rows, with any error it raises returned, not raised. Running out of
    time passes through the `except` below, so `timeout` returns `_TIMED_OUT` for it."""
    try:
        return fn(data)
    except Exception as error:  # generated code can raise anything
        return _Raised(error)


def _screen(feature, X, max_undefined_share):
    """Why `feature` cannot be used on `X`, as `(failure, detail)`, or None where it can."""
    values = _compute_feature(feature.fn, copy(X))
    if values is _TIMED_OUT:
        return RAISED, "did not finish within 60 seconds"
    if isinstance(values, _Raised):
        return RAISED, f"raised {type(values.error).__name__}: {values.error}"
    try:
        values = np.asarray(values, dtype=float)
    except Exception as error:
        return WRONG_SHAPE, f"did not return numbers: {error}"
    if values.ndim == 0:
        return WRONG_SHAPE, f"returned a single value, not one per row ({len(X)})"
    values = values.flatten()
    if len(values) != len(X):
        return WRONG_SHAPE, f"returned {len(values)} values for {len(X)} rows"

    # NaN counts as undefined, and the threshold search and the routing send it right of every
    # threshold, at fit and at predict alike. An infinite value is a value, ordered past every
    # finite one.
    undefined = np.isnan(values)
    share = float(undefined.mean())
    if undefined.all():
        return UNDEFINED, "returned NaN on every row"
    if share > max_undefined_share:
        return UNDEFINED, (f"returned NaN on {share:.0%} of the rows, more than the "
                           f"{max_undefined_share:.0%} allowed")
    if len(np.unique(values[~undefined])) == 1:
        return CONSTANT, ("returned the same value on every row where it is defined, so it "
                          "cannot tell one row from another")
    return None


def screen_features(list_features, X, max_undefined_share=0.0):
    """The candidates usable on these rows, and each of the others with why it is not.

    A candidate is dropped when it raises (or runs past 60 seconds), returns other than one
    number per row, is undefined (NaN) on more than `max_undefined_share` of the rows, or is
    constant where it is defined. The reasons are phrased for the model a repair round sends them
    back to. Returns `(kept, failed)`, with `failed` as `(feature, failure, detail)`.
    """
    kept, failed = [], []
    for feature in list_features:
        reason = _screen(feature, X, max_undefined_share)
        if reason is None:
            logger.info("Successfully added feature")
            kept.append(feature)
        else:
            logger.info("Dropped feature `%s`: it %s", feature.name, reason[1])
            failed.append((feature, *reason))
    return kept, failed


def filter_features(list_features, X, max_undefined_share=0.0):
    """The candidates usable on these rows (`screen_features`), without the reasons."""
    return screen_features(list_features, X, max_undefined_share)[0]


@timeout(seconds=90, default=None)
def optimize_threshold(
    splitting_criterion,
    feature,
    X,
    y,
    min_samples_leaf,
    class_weight: Mapping[Any, float] | None = None,
):
    """
    Given a single feature, find the optimal threshold to split the data for this feature,
    based on the splitting criterion, and update the fields of the feature accordingly.
    """
    n_samples = len(y)

    # A candidate a language model wrote runs on a copy of the rows, so code that alters its frame
    # cannot change what the next candidate sees. A naive candidate is DEFT's own, which only reads
    # a column or the sequence, so it runs on the rows as they are: a frame handed thousands of
    # columns holds one block per column, and copying it for each of thousands of candidates cost
    # several times the threshold search itself.
    rows = X if feature.origin == ORIGIN_NAIVE else copy(X)
    feature_values = feature.fn(rows)

    # Convert to numpy array if not already
    if not isinstance(feature_values, np.ndarray):
        feature_values = np.array(feature_values)

    # convert to float
    feature_values = feature_values.astype(float)

    # Sort values and get unique thresholds between different values
    sorted_idx = np.argsort(feature_values)

    sorted_values = feature_values[sorted_idx]

    # Find unique points to evaluate thresholds
    value_changes = np.where(np.abs(np.diff(sorted_values)) > 1e-7)[0]
    # Make sure we don't go beyond array bounds
    valid_changes = value_changes[value_changes + 1 < len(sorted_values)]

    thresholds = (sorted_values[valid_changes] + sorted_values[valid_changes + 1]) / 2

    best_score = -np.inf
    feature.score = best_score
    feature.threshold = 0

    for threshold in thresholds:
        try:
            left_mask = feature_values <= threshold
            n_left = np.sum(left_mask)
            n_right = n_samples - n_left

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            score = splitting_criterion(
                y[left_mask],
                y[~left_mask],
                class_weight=class_weight,
            )

            if score > best_score:
                best_score = score

                feature.threshold = threshold
                feature.score = score
        except Exception as e:
            logger.info(f"Got the error: {e}")
            continue

    return feature


def optimize_features(
    splitting_criterion,
    list_features,
    X,
    y,
    min_samples_leaf,
    class_weight: Mapping[Any, float] | None = None,
):

    """Handle optimize features."""
    for i in range(len(list_features)):
        feature = list_features[i]
        logging.info(f"Optimizing feature {feature.name}")
        try:
            result = optimize_threshold(
                splitting_criterion,
                feature,
                X,
                y,
                min_samples_leaf,
                class_weight=class_weight,
            )
            if result is None:
                logger.info(f"Feature {feature.name} timed out after 90 seconds")
                feature.score = -np.inf
                feature.threshold = 0
        except Exception as e:
            logger.info(f"Got the error: {e}")
            # Set the score to -inf if we encounter an error
            feature.score = -np.inf
            feature.threshold = 0
            continue
    # Now find the best feature based on the score once we have optimized all the feature thresholds

    return list_features
