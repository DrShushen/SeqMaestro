"""IGSR's second term language: each term a Python function of the rows.

IGSR's native term is a numpy expression over named numeric columns (`igsr_utils.safe_eval`). In the
function form a term is any computation a short function can express over the rows as a frame: a
count of a motif in a text column of any length, a fraction of it over a range, a transform of a
numeric column, a combination of both. The search is unchanged. It needs only that each term
evaluates to one number per row, so the influence, the pruning, the tree search and the history
treat a function term as they treat an expression, by its name.

A term is a `FunctionTerm`: the name the model gave it, a description of what it measures, a
rationale for why it might predict the target, and its code, `def <name>(X)` over a frame `X` of the
input columns. The `FunctionTermRegistry` holds every term accepted during one search, so that a
node's terms, the pruner's decisions and the final model refer to terms by name.

**A term is run on the training rows before it is accepted** (`check`), and refused if it raises,
returns other than one number per row, is undefined (NaN or infinite) on more than
`max_undefined_share` of the rows, or is constant. Up to that share its undefined rows are filled
with its median on the training rows, and that fill is kept, so every other set of rows is filled
with the same value. A term that fails on other rows cannot be refused there, because a model may
already use it, so its whole column on those rows takes the fill, and the failure is recorded.

**Values are rounded to `SIGNIFICANT_DIGITS` significant digits** as they are computed. Code can
compute one value two ways that differ in the last bits, and a fit, a rank or a count of distinct
values would treat the two as different.

**Terms are deduplicated by their normalised source** (`normalised_source`): the code with the
term's function renamed to one placeholder, docstrings removed and the layout canonical, so the
same computation proposed twice under two names is one term.

**Each call runs under the code guard** (`igsr.code_guard`) **and a timeout.** The timeout uses
SIGALRM, which exists only on the main thread; off it, a term runs without one, with a warning.
"""

from __future__ import annotations

import ast
import keyword
import logging
import signal
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from igsr.code_guard import CodeGuardPolicy, UnsafeCodeError, compile_feature_function

logger = logging.getLogger(__name__)

#: The two term languages (`experiment.term_language`).
EXPRESSION, FUNCTION = "expression", "function"
TERM_LANGUAGES = (EXPRESSION, FUNCTION)

#: Significant digits a function term's values are rounded to. Twelve is far finer than any
#: meaningful difference between two rows, and far coarser than float64 arithmetic noise, which
#: sits near the sixteenth digit.
SIGNIFICANT_DIGITS = 12

#: Why a proposed term cannot be used: its code was refused by the guard or does not compile, its
#: name is not usable, or running it on the training rows failed in one of four ways.
REFUSED, NAMING, RAISED, WRONG_SHAPE, UNDEFINED, CONSTANT = (
    "refused", "naming", "raised", "wrong_shape", "undefined", "constant")

#: The settings of the function form, each of which a config states.
SETTINGS = ("max_undefined_share", "repair_rounds", "n_examples", "max_example_length", "timeout_seconds")


@dataclass(frozen=True)
class FunctionTermSettings:
    """How function terms are shown to the model, screened and repaired."""

    #: A term undefined on more of the training rows than this share is refused.
    max_undefined_share: float
    #: Times the terms of one proposal that failed go back to the model with their failures.
    repair_rounds: int
    #: Training rows shown to the model as examples, with their target.
    n_examples: int
    #: Characters of a text value an example shows; a longer one is shown by its ends. None shows
    #: every value whole.
    max_example_length: Optional[int]
    #: Seconds one term may take on one set of rows.
    timeout_seconds: int
    guard: CodeGuardPolicy = field(default_factory=CodeGuardPolicy)

    def __post_init__(self) -> None:
        if not 0 <= self.max_undefined_share < 1:
            raise ValueError(f"max_undefined_share must be in [0, 1), not {self.max_undefined_share!r}.")
        if self.repair_rounds < 0 or self.n_examples < 1 or self.timeout_seconds < 1:
            raise ValueError("repair_rounds must be at least 0, and n_examples and timeout_seconds at least 1.")

    @classmethod
    def from_config(cls, block: Mapping[str, Any]) -> "FunctionTermSettings":
        """From `experiment.function_terms`, which must state every setting."""
        missing = [key for key in SETTINGS if key not in block]
        if missing:
            raise ValueError(f"experiment.function_terms must state {', '.join(missing)}.")
        return cls(
            max_undefined_share=float(block["max_undefined_share"]),
            repair_rounds=int(block["repair_rounds"]),
            n_examples=int(block["n_examples"]),
            max_example_length=None if block["max_example_length"] is None else int(block["max_example_length"]),
            timeout_seconds=int(block["timeout_seconds"]),
            guard=CodeGuardPolicy.from_config(block.get("code_guard")),
        )


# ================================================================================
# Running a term on rows
# ================================================================================


def to_significant_digits(values: np.ndarray, digits: int) -> np.ndarray:
    """`values` rounded to `digits` significant digits; zero, NaN and infinities pass unchanged."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        magnitude = np.floor(np.log10(np.abs(values)))
        scale = 10.0 ** (digits - 1 - magnitude)
        rounded = np.round(values * scale) / scale
    return np.where(np.isfinite(rounded) & (values != 0), rounded, values)


def as_float_column(values: Any, n_rows: int) -> np.ndarray:
    """Whatever a term returned as one float per row, or raise. A scalar is refused instead of
    broadcast, because a term that is the same on every row carries nothing."""
    if np.isscalar(values):
        raise ValueError(f"returned a scalar, expected one value per row ({n_rows}).")
    array = np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values, dtype=float)
    if array.shape != (n_rows,):
        raise ValueError(f"returned shape {array.shape}, expected ({n_rows},).")
    return array


_warned_off_main_thread = False


class _OutOfTime(BaseException):
    """Raised inside a term when its time runs out.

    A `BaseException`, so that a term catching `Exception` to handle its own errors row by row
    cannot catch this too and run on with no limit left. A bare `except:` would catch it, which is
    why the guard compiles one as `except Exception:`. It never leaves `call_with_timeout`, which
    raises `TimeoutError` in its place.
    """


def call_with_timeout(function: Callable, frame: pd.DataFrame, seconds: int) -> Any:
    """`function(frame)`, stopped with `TimeoutError` after `seconds` on the main thread."""
    global _warned_off_main_thread
    if threading.current_thread() is not threading.main_thread():
        if not _warned_off_main_thread:
            logger.warning("Function terms run off the main thread, where SIGALRM is unavailable, so their "
                           "%d s timeout is not in effect.", seconds)
            _warned_off_main_thread = True
        return function(frame)
    armed = True

    def expired(signum, stack):
        # Only while armed: a signal already on its way when the term ends must not raise outside it.
        if armed:
            raise _OutOfTime

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        try:
            signal.alarm(seconds)
            return function(frame)
        finally:
            armed = False
            signal.alarm(0)
    except _OutOfTime:
        raise TimeoutError(f"took longer than {seconds} s") from None
    finally:
        signal.signal(signal.SIGALRM, previous)


def values_of(function: Callable, frame: pd.DataFrame, seconds: int) -> np.ndarray:
    """One term on these rows: a float per row, rounded. Raises what the term raises, `TimeoutError`
    when it runs too long, and `ValueError` when it returns other than one number per row."""
    return to_significant_digits(as_float_column(call_with_timeout(function, frame, seconds), len(frame)),
                                 SIGNIFICANT_DIGITS)


@dataclass(frozen=True)
class Check:
    """One term on the training rows: its values where it ran, and why it cannot be used, if not."""

    failure: Optional[str]
    #: The failure in words, for the model asked to repair it.
    detail: str
    values: Optional[np.ndarray] = None

    @property
    def usable(self) -> bool:
        return self.failure is None


def check(function: Callable, frame: pd.DataFrame, settings: FunctionTermSettings) -> Check:
    """Run a term on the training rows and say whether its values can be used (module docstring)."""
    try:
        returned = call_with_timeout(function, frame, settings.timeout_seconds)
    except Exception as error:  # model-written code can raise anything
        return Check(RAISED, f"raised {type(error).__name__}: {error}")
    try:
        values = to_significant_digits(as_float_column(returned, len(frame)), SIGNIFICANT_DIGITS)
    except (TypeError, ValueError) as error:
        return Check(WRONG_SHAPE, f"did not return one number per row: it {error}")

    finite = np.isfinite(values)
    share = float(1 - finite.mean())
    if not finite.any():
        return Check(UNDEFINED, "returned NaN or an infinite value on every row", values)
    if share > settings.max_undefined_share:
        return Check(UNDEFINED, f"returned NaN or an infinite value on {share:.0%} of the rows, more than "
                                f"the {settings.max_undefined_share:.0%} allowed", values)
    if np.ptp(values[finite]) == 0:
        return Check(CONSTANT, "returned the same value on every row where it is defined, so it cannot "
                               "tell one row from another", values)
    return Check(None, "usable", values)


# ================================================================================
# What a proposal is, and when two are the same term
# ================================================================================


@dataclass(frozen=True)
class Proposal:
    """One term as the model wrote it. `replaces` names the term a correction is for."""

    name: str
    description: str
    rationale: str
    code: str
    replaces: Optional[str] = None


def naming_problem(proposal: Proposal) -> Optional[str]:
    """Why a proposal's name cannot be used, or None. The name must be a legal identifier, and the
    code must define a function of that name taking `X`, so that a term's name and its code agree."""
    if not proposal.name.isidentifier() or keyword.iskeyword(proposal.name):
        return f"its name {proposal.name!r} is not a legal Python identifier"
    try:
        module = ast.parse(proposal.code)
    except SyntaxError:
        return None  # compiling it says what is wrong
    defined = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == proposal.name
               and [argument.arg for argument in node.args.args] == ["X"]]
    if not defined:
        return f"its code defines no function `{proposal.name}(X)`"
    return None


def compile_term(proposal: Proposal, policy: CodeGuardPolicy) -> Callable:
    """The term's function, compiled behind the guard. Raises `UnsafeCodeError` when the guard
    refuses the code and `ValueError` when it does not compile."""
    return compile_feature_function(proposal.code, policy=policy)


class _Canonical(ast.NodeTransformer):
    """Rename the term's function to one placeholder and drop docstrings."""

    PLACEHOLDER = "term"

    def __init__(self, name: str) -> None:
        self.name = name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        if node.name == self.name:
            node.name = self.PLACEHOLDER
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self.name:
            node.id = self.PLACEHOLDER
        return node


def normalised_source(code: str, name: str) -> str:
    """The code as terms are compared: parsed, the function `name` renamed to a placeholder,
    docstrings dropped, and printed back in one layout, so comments, formatting and the name do not
    count. Code that does not parse is compared as written."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code
    return ast.unparse(_Canonical(name).visit(module))


# ================================================================================
# The terms of one search
# ================================================================================


@dataclass
class FunctionTerm:
    """One accepted function term: what the model wrote, and what the search keeps about it."""

    name: str
    description: str
    rationale: str
    code: str
    #: The median of its finite values on the training rows, which fills its undefined rows.
    fill: float
    #: The share of the training rows it was undefined on.
    undefined_share: float
    #: The iteration or node whose proposal it came from.
    first_seen: str
    #: Accepted as the correction of a term that failed.
    repaired: bool = False
    #: The name it was proposed under, where it was renamed.
    renamed_from: Optional[str] = None
    function: Optional[Callable] = field(default=None, repr=False, compare=False)

    def record(self) -> Dict[str, Any]:
        """Everything about the term but its compiled function."""
        return {"name": self.name, "description": self.description, "rationale": self.rationale,
                "code": self.code, "fill": self.fill, "undefined_share": self.undefined_share,
                "first_seen": self.first_seen, "repaired": self.repaired, "renamed_from": self.renamed_from}


@dataclass
class FunctionTermRows:
    """One split's rows as terms are run on them: the frame of input columns, and the split's name,
    under which each term's values are computed once."""

    split: str
    frame: pd.DataFrame
    registry: "FunctionTermRegistry"

    def matrix(self, names: List[str]) -> np.ndarray:
        """The named terms on these rows, one column each, in order."""
        return np.column_stack([self.registry.column(name, self) for name in names])


@dataclass
class ProposalStats:
    """What came of a search's proposals, so a run can say how much of what the model wrote was
    usable: every term proposed or sent as a correction, those accepted and those repaired, the
    repair requests sent, and the failures that left a term out, by kind."""

    proposed: int = 0
    corrections: int = 0
    accepted: int = 0
    repaired: int = 0
    repair_requests: int = 0
    left_out: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"proposed": self.proposed, "corrections": self.corrections, "accepted": self.accepted,
                "repaired": self.repaired, "repair_requests": self.repair_requests, "left_out": dict(self.left_out)}


class FunctionTermRegistry:
    """Every function term accepted during one search, by name, with its values on each split."""

    def __init__(self, settings: FunctionTermSettings) -> None:
        self.settings = settings
        self._terms: Dict[str, FunctionTerm] = {}
        self._by_source: Dict[str, str] = {}
        self._values: Dict[Tuple[str, str], np.ndarray] = {}
        #: Terms that failed on a split other than the training rows: name -> split -> the failure.
        self.failures: Dict[str, Dict[str, str]] = {}
        self.stats = ProposalStats()

    def __contains__(self, name: str) -> bool:
        return name in self._terms

    def __getitem__(self, name: str) -> FunctionTerm:
        return self._terms[name]

    def __len__(self) -> int:
        return len(self._terms)

    def same_as(self, proposal: Proposal) -> Optional[str]:
        """The name of an accepted term that computes what `proposal` computes, or None."""
        return self._by_source.get(normalised_source(proposal.code, proposal.name))

    def free_name(self, name: str) -> str:
        """`name` if no accepted term has it, else the first of `name_2`, `name_3`, ... that is free."""
        if name not in self._terms:
            return name
        suffix = 2
        while f"{name}_{suffix}" in self._terms:
            suffix += 1
        return f"{name}_{suffix}"

    def accept(self, proposal: Proposal, function: Callable, checked: Check, training: FunctionTermRows,
               first_seen: str, repaired: bool = False) -> FunctionTerm:
        """Add a term that passed its check on the training rows, under a free name. A term whose name
        another term already has is renamed, and records the name it was proposed under."""
        values = checked.values
        finite = np.isfinite(values)
        fill = float(np.median(values[finite]))
        name = self.free_name(proposal.name)
        renamed_from = proposal.replaces if proposal.replaces not in (None, proposal.name) else None
        if name != proposal.name:
            renamed_from = renamed_from or proposal.name
        term = FunctionTerm(
            name=name, description=proposal.description, rationale=proposal.rationale,
            code=proposal.code if name == proposal.name else _renamed(proposal.code, proposal.name, name),
            fill=fill, undefined_share=float(1 - finite.mean()), first_seen=first_seen, repaired=repaired,
            renamed_from=renamed_from, function=function)
        self._terms[name] = term
        self._by_source[normalised_source(proposal.code, proposal.name)] = name
        self._values[(name, training.split)] = np.where(finite, values, fill)
        self.stats.accepted += 1
        self.stats.repaired += int(repaired)
        return term

    def column(self, name: str, rows: FunctionTermRows) -> np.ndarray:
        """A term's values on a split, computed once and filled where undefined. A term that fails on
        a split is filled there entirely, and the failure is recorded in `failures`."""
        key = (name, rows.split)
        if key not in self._values:
            term = self._terms[name]
            try:
                values = values_of(term.function, rows.frame, self.settings.timeout_seconds)
            except Exception as error:  # model-written code can raise anything
                self.failures.setdefault(name, {})[rows.split] = f"{type(error).__name__}: {error}"
                values = np.full(len(rows.frame), np.nan)
            self._values[key] = np.where(np.isfinite(values), values, term.fill)
        return self._values[key]

    def records(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """The records of the named terms, or of every accepted term in the order accepted."""
        return [self._terms[name].record() for name in (names if names is not None else self._terms)]


def _renamed(code: str, old: str, new: str) -> str:
    """The code with its function `old` renamed `new`, so a renamed term's code defines its name."""
    module = ast.parse(code)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == old:
            node.name = new
        elif isinstance(node, ast.Name) and node.id == old:
            node.id = new
    return ast.unparse(module)


# ================================================================================
# Screening one reply's proposals
# ================================================================================


@dataclass
class Screened:
    """What came of one reply: the names of the terms accepted, and the proposals that fail, each with
    its failure, under the name a correction must give as `replaces`."""

    accepted: List[str] = field(default_factory=list)
    failing: Dict[str, Tuple[Proposal, Check]] = field(default_factory=dict)


def screen(proposals: List[Proposal], registry: FunctionTermRegistry, training: FunctionTermRows,
           first_seen: str) -> Screened:
    """Accept the proposals that work on the training rows, and say why each of the others fails. A
    proposal that computes what an accepted term computes is that term."""
    found = Screened()
    for proposal in proposals:
        name, failed = _screen_one(proposal, registry, training, first_seen, repaired=False)
        if name is not None:
            found.accepted.append(name)
        else:
            found.failing[proposal.name] = failed
    found.accepted = list(dict.fromkeys(found.accepted))
    return found


def repair(corrections: List[Proposal], failing: Dict[str, Tuple[Proposal, Check]],
           registry: FunctionTermRegistry, training: FunctionTermRows, first_seen: str) -> Screened:
    """Screen corrections of failing terms. Each is matched to the failing term its `replaces` names,
    or its own name where it names none; a correction that matches no term still failing is ignored.
    A term whose correction fails keeps the correction and its failure, so the next request shows
    the code last tried."""
    found = Screened(failing=dict(failing))
    for correction in corrections:
        target = correction.replaces or correction.name
        if target not in found.failing:
            continue
        name, failed = _screen_one(correction, registry, training, first_seen, repaired=True)
        if name is not None:
            found.accepted.append(name)
            del found.failing[target]
        else:
            found.failing[target] = failed
    found.accepted = list(dict.fromkeys(found.accepted))
    return found


def _screen_one(proposal: Proposal, registry: FunctionTermRegistry, training: FunctionTermRows,
                first_seen: str, repaired: bool) -> Tuple[Optional[str], Optional[Tuple[Proposal, Check]]]:
    """The name the proposal is accepted under, or its failure."""
    existing = registry.same_as(proposal)
    if existing is not None:
        return existing, None
    problem = naming_problem(proposal)
    if problem is not None:
        return None, (proposal, Check(NAMING, problem))
    try:
        function = compile_term(proposal, registry.settings.guard)
    except UnsafeCodeError as error:
        return None, (proposal, Check(REFUSED, f"was refused: {error}"))
    except ValueError as error:
        return None, (proposal, Check(REFUSED, f"does not compile: {error}"))
    checked = check(function, training.frame, registry.settings)
    if not checked.usable:
        return None, (proposal, checked)
    return registry.accept(proposal, function, checked, training, first_seen, repaired).name, None
