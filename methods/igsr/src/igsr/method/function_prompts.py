"""What the model is asked for in the function form, and how its replies are read back.

The expression form's prompt (`igsr.method.igsr.prompt_generate_terms`) previews the first values of
every column, which describes a numeric table and nothing else. A function term reads the rows as a
frame, so the model is shown what it needs to write code over them: each column of the frame, the
range of a text column's lengths, each column's description where the dataset gives one, and example
rows with their target, a long text value shown by its ends. The dataset's description, the current
terms and equation and the history of keep and drop decisions are shown as the expression form
shows them.

**The reply is JSON**: a list of terms, each with a name, a description, a rationale and the code,
after the `TERMS:` sentinel IGSR's parser looks for. A term that fails its check goes back to the
model with its code and its failure (`render_repair_request`), up to `repair_rounds` times per
proposal, through IGSR's own agent loop (`postprocessor_generate_function_terms`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from igsr.agent.agent import PostprocessorOutput
from igsr.method.function_terms import (
    CONSTANT,
    NAMING,
    RAISED,
    REFUSED,
    UNDEFINED,
    WRONG_SHAPE,
    Check,
    FunctionTermRegistry,
    FunctionTermRows,
    FunctionTermSettings,
    Proposal,
    repair,
    screen,
)
from igsr.parser import extract_code_blocks

#: The fields every term in a reply carries.
FIELDS = ("name", "description", "rationale", "code")

#: How a repair request asks for a failed term to be corrected, by what went wrong. A request shows
#: only the lines for the failures it sends back.
REPAIR_GUIDANCE: Tuple[Tuple[frozenset, str], ...] = (
    (frozenset({REFUSED, NAMING}),
     "If its code was refused or does not compile, or its name is unusable, write it again within "
     "the rules: a function `def <name>(X):` whose name is the term's name, importing only the "
     "modules allowed."),
    (frozenset({RAISED, WRONG_SHAPE}),
     "If it raised an error, or did not return one number per row, the code is at fault. Keep what "
     "the term measures, and change only what makes the code fail."),
    (frozenset({CONSTANT}),
     "If it returned the same value on every row, the term as written cannot tell one row from "
     "another in this data. Change what it measures as little as you can so that it varies from row "
     "to row, for example by loosening a condition that no row meets."),
    (frozenset({UNDEFINED}),
     "If it returned NaN or an infinite value on too many rows, the term as written is undefined for "
     "too much of the data. Change what it measures as little as you can so that it is defined on "
     "more rows, for example by reading a part of a text value that every row has, or by avoiding a "
     "division that is often by zero."),
)

#: What a repair request says about names, whatever went wrong.
REPAIR_NAMING = (
    "A corrected term that still measures what the original measured keeps the original's name. One "
    "that now measures something different needs a new name and a description, both saying what it "
    "measures now. In every entry of your answer, \"replaces\" is the name of the term it corrects."
)


# ================================================================================
# What the model is shown
# ================================================================================


def is_text(values: pd.Series) -> bool:
    """Whether a column holds text, which a term reads with string operations."""
    return pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values)


def describe_columns(frame: pd.DataFrame, descriptions: Optional[Mapping[str, str]]) -> str:
    """One line per column of the frame a term is given: its name, whether it is text (with the range
    of lengths) or numeric, and its description where the dataset gives one."""
    lines = []
    for column in frame.columns:
        values = frame[column]
        if is_text(values):
            lengths = values.astype(str).str.len()
            shortest, longest = int(lengths.min()), int(lengths.max())
            shape = (f"text, {shortest} characters in every row" if shortest == longest
                     else f"text, from {shortest} to {longest} characters: lengths differ between rows")
        else:
            shape = "numeric"
        description = (descriptions or {}).get(column)
        lines.append(f"- `{column}` ({shape})" + (f": {description}" if description else ""))
    return "\n".join(lines)


def shown(value: Any, max_length: Optional[int]) -> str:
    """A value as an example shows it: a text value longer than `max_length` by its two ends."""
    text = str(value)
    if max_length is None or len(text) <= max_length:
        return text
    head = max_length // 2
    return f"{text[:head]}...{text[len(text) - (max_length - head):]} ({len(text)} characters)"


def sample_examples(y: np.ndarray, n_examples: int, seed: int) -> np.ndarray:
    """Positions of `n_examples` training rows spread over the target's range: the rows are split
    into up to ten groups by the rank of the (first) target, and drawn from the groups in turn, with
    the seed. Returned in the order of the target, low to high."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:
        y = y[:, 0]
    n = min(n_examples, len(y))
    rng = np.random.default_rng(seed)
    groups = pd.qcut(pd.Series(y).rank(method="first"), q=min(10, n), labels=False).to_numpy()
    pools = [list(rng.permutation(np.flatnonzero(groups == group))) for group in np.unique(groups)]
    chosen: List[int] = []
    while len(chosen) < n:
        for pool in pools:
            if pool and len(chosen) < n:
                chosen.append(int(pool.pop()))
    return np.array(sorted(chosen, key=lambda row: (y[row], row)), dtype=int)


def render_examples(frame: pd.DataFrame, y: Mapping[str, np.ndarray], rows: np.ndarray,
                    max_length: Optional[int]) -> str:
    """The example rows: each target's value, then every column of the frame."""
    blocks = []
    for number, row in enumerate(rows, start=1):
        targets = ", ".join(f"{name} = {float(values[row]):g}" for name, values in y.items())
        columns = "\n".join(f"  {column}: {shown(frame[column].iloc[row], max_length)}" for column in frame.columns)
        blocks.append(f"Example {number}: {targets}\n{columns}")
    return "\n\n".join(blocks)


def worked_example(frame: pd.DataFrame) -> str:
    """A term written to the contract, over the first text column if there is one, else over the
    first numeric column, so the model sees the shape its code must have."""
    text_columns = [column for column in frame.columns if is_text(frame[column])]
    if text_columns:
        column = text_columns[0]
        return (f'def distinct_characters_first_20(X):\n'
                f'    """Number of distinct characters among the first 20 of `{column}`, or NaN where it is '
                f'shorter than 20."""\n'
                f'    text = X["{column}"].astype(str)\n'
                f'    return text.map(lambda s: len(set(s[:20])) if len(s) >= 20 else float("nan"))')
    column = frame.columns[0]
    return (f'def log1p_{column}(X):\n'
            f'    """log(1 + {column}), or NaN where {column} is -1 or below."""\n'
            f'    values = X["{column}"].astype(float)\n'
            f'    return np.log1p(values.where(values > -1))')


def answer_format(terms_sentinel: str, repair: bool = False) -> str:
    """The shape of a reply: the sentinel, then one JSON list in a fenced block."""
    replaces = ', "replaces": "<the name of the term this corrects>"' if repair else ""
    return (f'{terms_sentinel}\n```json\n[\n  {{"name": "<a Python identifier>", "description": "<what it measures>", '
            f'"rationale": "<why it might predict the target>", "code": "def <name>(X):\\n    ..."{replaces}}}\n]\n```')


@dataclass(frozen=True)
class FunctionTermContext:
    """What the function form shows the model about the data, fixed for one search."""

    columns: str
    examples: str
    worked_example: str
    target_names: Tuple[str, ...]


def build_context(frame: pd.DataFrame, y: Mapping[str, np.ndarray], descriptions: Optional[Mapping[str, str]],
                  settings: FunctionTermSettings, seed: int) -> FunctionTermContext:
    """The columns, the example rows and the worked example, from the training rows, drawn once per
    search with its seed."""
    rows = sample_examples(np.column_stack(list(y.values())), settings.n_examples, seed)
    return FunctionTermContext(
        columns=describe_columns(frame, descriptions),
        examples=render_examples(frame, y, rows, settings.max_example_length),
        worked_example=worked_example(frame),
        target_names=tuple(y),
    )


def prompt_generate_function_terms(
    *,
    data_dictionary: Optional[str],
    context: FunctionTermContext,
    settings: FunctionTermSettings,
    registry: FunctionTermRegistry,
    current_terms: List[str],
    current_equation: Optional[str],
    history_text: str,
    terms_per_round: int,
    first_round_n_candidates: int,
    terms_sentinel: str,
) -> str:
    """The request for new terms in the function form."""
    current = "\n".join(f"- `{name}`: {registry[name].description}" for name in current_terms) or "(none yet)"
    targets = ", ".join(f"`{name}`" for name in context.target_names)
    modules = ", ".join(settings.guard.modules)
    return f"""You are an automated assistant proposing terms for a linear model in a symbolic regression pipeline. Each term is a Python function of the input rows, and the model predicts the target ({targets}) as a weighted sum of the terms.

Your proposed terms will be:
1. Added to the current terms, and fitted with them by least squares.
2. Kept or dropped by how much the error on held-out rows rises when each one is removed (its influence).

# Instructions
Propose around {terms_per_round} new terms, or around {first_round_n_candidates} if there are no current terms yet. Make use of the dataset and problem description and the example rows, and of the history of previous rounds. Each term should be a distinct hypothesis a domain expert could name and reason about, and none should repeat a current term.

Each term is a function `def <name>(X):`, where `<name>` is the term's name and `X` is a pandas DataFrame with one row per example and the columns listed below. It returns one number per row, as a pandas Series or a numpy array of length `len(X)`, and NaN for a row where the term is undefined. A text column's lengths may differ between rows, so code that reads it must work at every length. The code may import only {modules}, and `np` and `pd` are available without an import. It must not read files or the network, and nothing but imports and function definitions may stand at the top level.

A term written to these rules:
```python
{context.worked_example}
```

Answer with {terms_sentinel} followed by one JSON list in a fenced block, one entry per term:

{answer_format(terms_sentinel)}

You may return an empty list if no new term seems worth adding.

DATASET AND PROBLEM DESCRIPTION
------------------------------
{data_dictionary or "(no description given)"}

THE COLUMNS OF X
------------------------------
{context.columns}

EXAMPLE ROWS, WITH THE TARGET
------------------------------
{context.examples}

CURRENT TERMS
------------------------------
{current}

# Current equation:
{current_equation if current_equation is not None else "N/A"}

HISTORY OF KEEP/DROP DECISIONS
------------------------------
{history_text}
"""


def render_repair_request(failing: Dict[str, Tuple[Proposal, Check]], terms_sentinel: str) -> str:
    """Send the terms that failed back to the model: each with the code last tried and its failure,
    with how to correct each kind of failure and how to name a correction."""
    kinds = {checked.failure for _, checked in failing.values()}
    guidance = [text for failures, text in REPAIR_GUIDANCE if failures & kinds]
    listed = []
    for name, (proposal, checked) in failing.items():
        listed.append(f"Term `{name}` {checked.detail}.\n```python\n{proposal.code}\n```")
    return "\n\n".join([
        "Some of the terms you proposed cannot be used. Each is listed below with its code and what went "
        "wrong when it was run on the training rows.",
        *guidance,
        REPAIR_NAMING,
        *listed,
        f"Answer in the same format as before, with one entry per corrected term, and leave out a term you "
        f"cannot correct:\n\n{answer_format(terms_sentinel, repair=True)}",
    ])


# ================================================================================
# Reading the replies back
# ================================================================================


def parse_function_terms(text: str, terms_sentinel: str) -> Tuple[List[Proposal], Optional[str]]:
    """The terms in a reply, or why the reply cannot be read."""
    if terms_sentinel not in text:
        return [], f"{terms_sentinel} was not found in the reply."
    blocks = extract_code_blocks(text, sentinel=f"{terms_sentinel}\n", return_info_string=False)
    if len(blocks) != 1:
        return [], f"Expected exactly one fenced block right after {terms_sentinel}, found {len(blocks)}."
    try:
        entries = json.loads(blocks[0])
    except json.JSONDecodeError as error:
        return [], f"The block after {terms_sentinel} is not valid JSON: {error}."
    if not isinstance(entries, list):
        return [], f"The block after {terms_sentinel} must be a JSON list of terms."
    proposals = []
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) for key in FIELDS):
            return [], f"Entry {number} must be an object with the text fields {', '.join(FIELDS)}."
        replaces = entry.get("replaces")
        if replaces is not None and not isinstance(replaces, str):
            return [], f"Entry {number}'s \"replaces\" must be the name of a term."
        proposals.append(Proposal(name=entry["name"].strip(), description=entry["description"].strip(),
                                  rationale=entry["rationale"].strip(), code=entry["code"],
                                  replaces=replaces.strip() if replaces else None))
    return proposals, None


@dataclass
class ProposalRound:
    """One proposal's progress through the agent's turns: the terms accepted so far, the terms still
    failing, and how many repair requests were sent."""

    accepted: List[str] = field(default_factory=list)
    failing: Dict[str, Tuple[Proposal, Check]] = field(default_factory=dict)
    repairs_sent: int = 0
    #: Every failure seen, as (name, kind), for the search's record.
    failures: List[Tuple[str, str]] = field(default_factory=list)


def postprocessor_generate_function_terms(agent, *args, **kwargs) -> PostprocessorOutput:
    """Read a reply of the function form, screen its terms, and ask for repairs while any fail.

    The first reply's terms are screened; each later reply is read as the corrections of the terms
    still failing. While terms fail and repair requests remain, the failures go back to the model
    (`inform_agent`); after that, the proposal ends with the terms accepted, and a term still
    failing is left out. A reply that cannot be read goes back with the format, as IGSR's own
    postprocessor sends it, without counting as a repair.
    """
    registry: FunctionTermRegistry = kwargs["registry"]
    training: FunctionTermRows = kwargs["training"]
    progress: ProposalRound = kwargs["progress"]
    terms_sentinel: str = kwargs["terms_sentinel"]
    first_seen: str = kwargs["first_seen"]
    correct_format: str = kwargs["correct_format"]

    proposals, error = parse_function_terms(agent.conversation_history[-1]["content"], terms_sentinel)
    if error is not None:
        return PostprocessorOutput(success=False, inform_agent=True, feedback=f"Output error. {error}\n{correct_format}")

    if progress.repairs_sent == 0:
        registry.stats.proposed += len(proposals)
        found = screen(proposals, registry, training, first_seen)
    else:
        registry.stats.corrections += len(proposals)
        found = repair(proposals, progress.failing, registry, training, first_seen)
    progress.accepted = list(dict.fromkeys(progress.accepted + found.accepted))
    progress.failing = found.failing
    progress.failures.extend((name, checked.failure) for name, (_, checked) in found.failing.items())

    if progress.failing and progress.repairs_sent < registry.settings.repair_rounds:
        progress.repairs_sent += 1
        registry.stats.repair_requests += 1
        return PostprocessorOutput(success=False, inform_agent=True,
                                   feedback=render_repair_request(progress.failing, terms_sentinel))
    for _, checked in progress.failing.values():
        registry.stats.left_out[checked.failure] = registry.stats.left_out.get(checked.failure, 0) + 1
    return PostprocessorOutput(success=True, inform_agent=False, parsed_output=list(progress.accepted))
