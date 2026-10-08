"""What a generated function did on the rows, and whether a model can use the result.

A function a language model wrote can fail in five ways once it meets real rows: it raises, it does
not finish within its time limit, it returns something other than one number per row, it is
undefined (NaN or infinite) on too many rows, or it returns one value for every row. `check` runs a
function on a frame and says which, in words a language model can act on.

Both places that run generated code use it. The proposer checks each candidate on the rows it
proposed from, so a failure can go back to the model with its failure attached (`Proposer`'s repair
round). `GeneratedFeaturiser` checks again when it fits, as a second guard. One definition means the
two cannot disagree about what is usable.

**How much undefined is too much** is a setting, `max_undefined_share`. Below it the feature is
kept and its undefined rows are filled with its median, as a model needs a number in every row.
Above it the feature is mostly a statement about which rows it cannot measure, and is dropped. A
feature undefined on every row is never kept.

**How long is too long** is a setting too, `max_seconds`: the longest one function may run on the
rows it is checked on. Generated code can cost time that grows with the square or the cube of a
sequence's length, which finishes in seconds on sequences of a few hundred bases and takes days on
sequences of several thousand. The limit turns such a function into a failure like any other, so
the proposer can send it back to the model. On more rows than it was checked on, a function is
given proportionally more time (`scaled_limit`).
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Significant digits a generated feature's values are rounded to. Generated code can compute one
#: value two ways (a sum taken in another order gives 0.15 against 0.15000000000000002), and every
#: exact comparison downstream (a rank, a count of distinct values, a tree's or an EBM's cut point)
#: would treat the two as different. Twelve digits is far finer than any meaningful difference in a
#: feature, and far coarser than float64 arithmetic noise, which sits near the sixteenth digit.
SIGNIFICANT_DIGITS = 12


def to_significant_digits(values: np.ndarray, digits: int) -> np.ndarray:
    """`values` rounded to `digits` significant digits; zero, NaN and infinities pass unchanged."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        magnitude = np.floor(np.log10(np.abs(values)))
        scale = 10.0 ** (digits - 1 - magnitude)
        rounded = np.round(values * scale) / scale
    # Zero, non-finite values, and magnitudes so extreme the scale overflows, are left as they are.
    return np.where(np.isfinite(rounded) & (values != 0), rounded, values)


def as_float_column(values, n_rows: int) -> np.ndarray:
    """Coerce whatever a generated function returned into one float per row, or raise.

    Generated functions return Series, arrays, lists and occasionally a scalar. A scalar is refused
    instead of broadcast, because a feature that is the same for every row carries no information,
    and widening it into a constant column would hide that behind a column that looks real.
    """
    if np.isscalar(values):
        raise ValueError(f"returned a scalar, expected one value per row ({n_rows}).")

    array = np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values, dtype=float)
    if array.shape != (n_rows,):
        raise ValueError(f"returned shape {array.shape}, expected ({n_rows},).")
    return array


#: Why a generated function cannot be used, as `Check.failure` names it.
RAISED, SLOW, WRONG_SHAPE, UNDEFINED, CONSTANT = "raised", "slow", "wrong_shape", "undefined", "constant"
FAILURES = (RAISED, SLOW, WRONG_SHAPE, UNDEFINED, CONSTANT)


class FeatureTimeout(TimeoutError):
    """A generated function did not finish within its time limit."""


class TimeLimitUnavailable(RuntimeError):
    """A time limit cannot be set for a generated function about to run: the limit is not above
    zero, the code is off the main thread, or another timer is set. This says nothing about the
    function, so every caller that catches the function's errors lets this through."""


class _OutOfTime(BaseException):
    """Raised inside a generated function when its time runs out.

    A `BaseException`, so that generated code catching `Exception` to handle its own edge cases
    cannot catch this too and run on, row after row, with no limit left. A bare `except:` would
    catch it, which is why the guard compiles one as `except Exception:`. It never leaves
    `time_limit`, which raises `FeatureTimeout` in its place.
    """


@contextmanager
def time_limit(seconds: float | None) -> Iterator[None]:
    """Stop the code inside with `FeatureTimeout` once it has run for `seconds`. None sets no limit.

    It uses the process's real-time timer and SIGALRM, which reaches only the main thread, so it is
    refused on any other thread. It is also refused while another timer is set, which it would
    cancel. The signal interrupts Python code and regular-expression matching as they run, and a
    single long call into numpy when that call returns.
    """
    if seconds is None:
        yield
        return
    if not seconds > 0:
        # The timer takes zero to mean "cancel", so a limit of zero would be no limit at all.
        raise TimeLimitUnavailable(f"A time limit must be above 0 seconds, or None for no limit, not "
                                   f"{seconds!r}.")
    if threading.current_thread() is not threading.main_thread():
        raise TimeLimitUnavailable("A generated function's time limit needs the main thread, the "
                                   "only one SIGALRM reaches, so the function cannot be run on this "
                                   "thread.")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise TimeLimitUnavailable("A real-time timer is already set in this process, and the time "
                                   "limit on a generated function would cancel it.")
    armed = True

    def out_of_time(signum, frame):
        # Only while armed: a signal already on its way when the limit is lifted must not raise
        # outside the generated code.
        if armed:
            raise _OutOfTime

    previous = signal.signal(signal.SIGALRM, out_of_time)
    try:
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            yield
        finally:
            armed = False
            signal.setitimer(signal.ITIMER_REAL, 0)
    except _OutOfTime:
        raise FeatureTimeout(f"did not finish within {seconds:g} seconds") from None
    finally:
        signal.signal(signal.SIGALRM, previous)


def scaled_limit(max_seconds: float | None, checked_rows: int | None, rows: int) -> float | None:
    """The time limit on `rows` rows for a function allowed `max_seconds` on the `checked_rows` it
    was checked on: longer in proportion to the rows, and never shorter than `max_seconds`, since a
    few rows still pay the function's fixed costs. None where either is unknown, which sets no
    limit."""
    if max_seconds is None or checked_rows is None:
        return None
    return max_seconds * max(1.0, rows / max(checked_rows, 1))


@dataclass(frozen=True)
class Check:
    """One function on one set of rows: its values where it ran, and why it cannot be used, if not."""

    failure: str | None
    #: The failure in words, for the model asked to repair it, and for a log.
    detail: str
    #: One float per row, rounded as a model would be given them. None where the function did not
    #: return one value per row.
    values: np.ndarray | None = None

    @property
    def usable(self) -> bool:
        return self.failure is None

    @property
    def undefined_share(self) -> float:
        """The share of rows with no finite value, 0 where there are no values."""
        return 0.0 if self.values is None else float(1 - np.isfinite(self.values).mean())


def run(function, frame: pd.DataFrame, significant_digits: int | None,
        max_seconds: float | None) -> np.ndarray:
    """One generated function on these rows, within `max_seconds`: a float per row, to the kept
    precision. Raises what the function raises, `FeatureTimeout` where it runs out of time, and
    `ValueError` where it returns other than one number per row."""
    with time_limit(max_seconds):
        returned = function(frame)
    values = as_float_column(returned, len(frame))
    return values if significant_digits is None else to_significant_digits(values, significant_digits)


def check(function, frame: pd.DataFrame, *, max_undefined_share: float,
          significant_digits: int | None, max_seconds: float | None) -> Check:
    """Run `function` on `frame` within `max_seconds`, and say whether its result is usable (see the
    module docstring)."""
    try:
        with time_limit(max_seconds):
            returned = function(frame)
    except TimeLimitUnavailable:
        raise
    except FeatureTimeout:
        return Check(SLOW, f"did not finish within {max_seconds:g} seconds on these {len(frame)} "
                           f"rows")
    except Exception as error:  # generated code can raise anything
        return Check(RAISED, f"raised {type(error).__name__}: {error}")
    try:
        values = as_float_column(returned, len(frame))
    except (TypeError, ValueError) as error:
        return Check(WRONG_SHAPE, f"did not return one number per row: it {error}")
    if significant_digits is not None:
        values = to_significant_digits(values, significant_digits)

    finite = np.isfinite(values)
    share = float(1 - finite.mean())
    if not finite.any():
        return Check(UNDEFINED, "returned NaN or an infinite value on every row", values)
    if share > max_undefined_share:
        return Check(UNDEFINED, f"returned NaN or an infinite value on {share:.0%} of the rows, "
                                f"more than the {max_undefined_share:.0%} allowed", values)
    if np.ptp(values[finite]) == 0:
        return Check(CONSTANT, "returned the same value on every row where it is defined, so it "
                               "cannot tell one row from another", values)
    return Check(None, "usable", values)
