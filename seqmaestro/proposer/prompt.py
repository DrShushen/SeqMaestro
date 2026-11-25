"""What the language model is asked for and how its reply is parsed.

This is kept apart from the proposer, so that the wording can be inspected, compared with `diff`
and tried out without a credential: `run_arm.py --show-prompt` prints what would be sent.

**The language model is shown labelled rows as well as the task.** The labels ground a proposal in
this dataset instead of in sequence biology generally, and they give a permuted cell its meaning.
Shown shuffled labels, the language model proposes features that tell the shuffled labels apart,
which makes the permuted cell a negative control. A proposer that never saw a label would give the
permuted cell the features it gives the real one.

**The request mentions covariates only where the rows have some.** It then lists the columns
(`seqmaestro.llm.columns`), shows the examples' covariate values as one table after their
sequences, and tells the language model that the frame holds them. Rows with no covariates and a
single length get a request with none of these additions.

**A repair request says how to correct each kind of failure** (`REPAIR_GUIDANCE`). Code that fails,
or runs too slowly for these sequences, is fixed without changing what the feature measures. A
feature that cannot work on these rows (one constant on all of them, or undefined on too many) can
only be corrected by measuring something a little different, and the language model is asked to
name and describe it for what it then measures, so a record's name never says one thing while its
code computes another.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from seqmaestro.features.base import as_inputs
from seqmaestro.llm.columns import render_columns
from seqmaestro.llm.context import TaskContext
from seqmaestro.proposer.checks import CONSTANT, FAILURES, RAISED, SLOW, UNDEFINED, WRONG_SHAPE, Check
from seqmaestro.target import BINARY, CONTINUOUS, target_for

#: The system prompt states the role only. Nothing about the data belongs here, including which
#: nucleic acid it is.
#:
#: **The molecule is stated by the dataset**, in its `sequence_layout`, because it differs between
#: datasets and this string does not. Naming one here would contradict the description of every
#: dataset of the other kind, and the contradiction does harm: a language model told it is looking
#: at DNA proposes fewer of the features that only make sense for RNA, and on an RNA dataset those
#: are most of what the run is meant to find.
SYSTEM_PROMPT = (
    "You are an expert computational molecular biologist. You propose interpretable features of "
    "nucleic acid sequence that might predict a biological outcome, and you write each one as a "
    "short Python function. You favour features a biologist could name and reason about -- "
    "composition over a defined window, motif occurrence, periodicity, skew, positional context -- "
    "over opaque statistics. Each feature should be a distinct hypothesis, not a restatement of "
    "another."
)

#: What the features are asked to do, by the kind of target: tell the classes apart, or predict the
#: number.
REQUEST = {
    BINARY: "distinguish the classes",
    CONTINUOUS: "predict the target's value",
}

#: The contract the generated code must satisfy, stated as a worked example because a language
#: model follows a demonstrated shape more reliably than a described one. Every rejected function
#: wastes tokens and leaves the candidate pool smaller than it looks.
#:
#: The example says nothing about any dataset: it measures a window at the start of the sequence,
#: whose size is taken from the rows it is shown with (`example_window`), so the code is correct for
#: them.
CODE_CONTRACT = '''\
def gc_fraction_first_{window}nt(X):
    """Fraction of G or C in the first {window} nt of the sequence."""
    seqs = X["{sequence_column}"].astype(str).str.upper().str.slice(0, {window})
    return (seqs.str.count("G") + seqs.str.count("C")) / {window}.0\
'''

#: The largest number of positions the worked example's window covers.
EXAMPLE_WINDOW = 20


def example_window(length: int) -> int:
    """The window the worked example measures, for sequences of one `length`: `EXAMPLE_WINDOW`
    positions, or half the sequence where that is shorter."""
    return max(1, min(EXAMPLE_WINDOW, length // 2))


#: The worked example where sequences differ in length, or their length is not known. It has the
#: same shape, and it shows the two things such code must do: work for any length, and return NaN
#: where a row lacks what the feature measures, instead of raising or measuring a shorter window
#: without saying so.
CODE_CONTRACT_ANY_LENGTH = '''\
def gc_fraction_last_40nt(X):
    """Fraction of G or C in the last 40 nt, or NaN where the sequence is shorter than 40 nt."""
    seqs = X["{sequence_column}"].astype(str).str.upper()
    last = seqs.str.slice(-40)
    return ((last.str.count("G") + last.str.count("C")) / 40.0).where(seqs.str.len() >= 40)\
'''

#: How a repair request asks for a failed function to be corrected, by what went wrong
#: (`seqmaestro.proposer.checks`). A request shows only the lines for the failures it sends back.
REPAIR_GUIDANCE: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({RAISED, WRONG_SHAPE}),
     "If it raised an error, or did not return one number per row, the code is at fault. Keep what "
     "the feature measures, and change only what makes the code fail."),
    (frozenset({SLOW}),
     "If it did not finish in time, the code is too slow for sequences of this length. Keep what "
     "the feature measures, and compute it in time that grows in proportion to the length of a "
     "sequence, for example by passing along each sequence once instead of comparing every pair of "
     "positions or testing every substring."),
    (frozenset({CONSTANT}),
     "If it returned the same value on every row, the feature as written cannot tell one row from "
     "another in this data, so it cannot help. Change what it measures as little as you can so "
     "that it varies from row to row, for example by loosening a condition that no row meets, or "
     "tightening one that every row meets."),
    (frozenset({UNDEFINED}),
     "If it returned NaN or an infinite value on too many rows, the feature as written is "
     "undefined for too much of the data. Change what it measures as little as you can so that it "
     "is defined on more rows, for example by reading a window that fits inside every sequence, "
     "or by avoiding a division that is often by zero."),
)

#: What a repair request says about names, whatever went wrong. The answer names the function each
#: correction replaces, so a correction that measures something new can be named for it.
REPAIR_NAMING = (
    "A corrected function that still measures what the original measured keeps the original's "
    "name. One that now measures something different needs a new name and a description, both "
    "saying what it measures now. In every entry of your answer, \"replaces\" is the name of the "
    "function that entry corrects."
)


def wording_digest() -> str:
    """A digest of the wording every proposal is made under, for the cache key.

    `ProposalCache` promises that anything which would change a proposal is in its key. Without
    this digest, editing the wording of a request would change what a language model is asked
    while the cache went on serving replies to the older question. Only a resumed run can meet that
    problem, because a cache lives inside one run directory and a new run starts empty, but a stale
    proposal is hardest to notice after a resume that follows an edit.

    **The digest is taken over requests rendered from fixed stand-in inputs**
    (`_stand_in_requests`), which between them show every part of the wording that only some
    requests show, with the system prompt that is sent beside each of them. Any change to what a
    language model is asked, whether in a constant of this module or in the code that assembles a
    request, therefore changes the digest, whereas a change to a comment or a docstring does not.
    """
    text = "\n\n".join([SYSTEM_PROMPT, *_stand_in_requests()])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _stand_in_requests() -> list[str]:
    """The requests `wording_digest` is taken over: one for a binary target on rows of one length,
    long enough for the worked example to measure `EXAMPLE_WINDOW` positions;
    one for a continuous target on rows of several lengths, with a covariate, guidance, earlier
    proposals, a fit's report and a shortened example; and the repair of one function for every
    kind of failure. The inputs are stand-ins, since only the wording around them matters here."""
    context = TaskContext(task="T", sequence_layout="L", notes="N")
    one_length = {"sequence": {"name": "s", "length": 100}, "covariates": []}
    several = {"sequence": {"name": "s", "length": {"min": 3, "max": 5}},
               "covariates": [{"name": "c", "min": 0.0, "max": 1.0, "description": "D"}]}
    rows = [Example("ACGT", 1), Example("TGCA", 0)]
    measured = [Example("ACGTA", 0.5, (("c", 0.25),)), Example("TGC", -0.5, (("c", 0.75),))]
    failures = [(f"f_{failure}", "def f(X): pass", Check(failure, "failed")) for failure in FAILURES]
    return [render_request(context, 1, rows, [], "s", "y", BINARY, columns=one_length),
            render_request(context, 1, measured, ["g"], "s", "y", CONTINUOUS, feedback={"k": 1},
                           columns=several, max_example_length=4, guidance="G"),
            render_repair_request(context, failures, "s", columns=several)]


@dataclass(frozen=True)
class Proposal:
    """One feature as the language model returned it, before it has been compiled or run.
    `replaces` is set in the answer to a repair request: the name of the function this one
    corrects."""

    name: str
    description: str
    rationale: str
    code: str
    replaces: str | None = None


@dataclass(frozen=True)
class Example:
    """One row shown to the language model: its sequence, its label, and its covariates by name."""

    sequence: str
    label: int | float | None
    covariates: tuple[tuple[str, float], ...] = ()


def sample_examples(inputs, y: np.ndarray | None, n_examples: int, seed: int,
                    target_kind: str) -> list[Example]:
    """A sample of rows to show the language model, from `Inputs` or the sequences alone, balanced
    over the strata of the target's kind: the classes of a binary target, and the quantile bins of a
    continuous one (`seqmaestro.target`).

    The sample is balanced instead of uniformly random, because at this sample size a random draw
    can be almost all one class, and a language model shown almost no positives proposes features
    about the negatives. A continuous target's tails are thin in the same way, and a random draw of
    a heavy-tailed target shows the language model almost nothing but values near its centre.
    """
    inputs = as_inputs(inputs)
    rng = np.random.default_rng(seed)

    def example(row: int, label: int | None) -> Example:
        return Example(inputs.sequences.iloc[row], label,
                       tuple((name, float(inputs.covariates[name].iloc[row]))
                             for name in inputs.covariate_columns))

    if y is None:
        chosen = rng.choice(len(inputs), size=min(n_examples, len(inputs)), replace=False)
        return [example(int(i), None) for i in sorted(chosen)]

    target = target_for(target_kind)
    y = target.values(y)
    strata = target.strata(y)
    picks: list[int] = []
    groups = np.unique(strata)
    per_group = max(1, n_examples // max(len(groups), 1))
    for group in groups:
        rows = np.flatnonzero(strata == group)
        take = min(per_group, len(rows))
        picks.extend(int(i) for i in rng.choice(rows, size=take, replace=False))

    # The rows are interleaved instead of grouped by stratum, so a row's position in the list says
    # nothing about its label.
    rng.shuffle(picks)
    return [example(i, y[i].item()) for i in picks]


def render_examples(examples: list[Example], target_column: str,
                    max_example_length: int | None) -> list[str]:
    """The example rows as a request shows them, one line per row: the sequence, then its label.

    Rows that have covariates are numbered, and their covariates follow as one table of
    comma-separated values: the covariates are named once, in its header, and each row's values
    are on the line with its number. Naming each covariate once keeps the prompt short when a
    dataset has many of them, since names written beside every value would take several times the
    space of the values.
    """
    def label(example: Example) -> str:
        return f"   -> {target_column}={example.label:g}" if example.label is not None else ""

    if not any(example.covariates for example in examples):
        return [f"  {shown_sequence(example.sequence, max_example_length)}{label(example)}"
                for example in examples]
    names = [name for name, _ in examples[0].covariates]
    numbered = list(enumerate(examples, start=1))
    return [
        *(f"  row {row}: {shown_sequence(example.sequence, max_example_length)}{label(example)}"
          for row, example in numbered),
        "",
        "The covariates of these rows, one line per row, as comma-separated values:",
        "  " + ",".join(["row", *names]),
        *("  " + ",".join([str(row), *(f"{value:g}" for _, value in example.covariates)])
          for row, example in numbered),
    ]


def shown_sequence(sequence: str, max_length: int | None) -> str:
    """A sequence as an example row shows it: whole, or where it is longer than `max_length`, its
    first and last bases with its length stated, since a whole transcript would fill the prompt."""
    if max_length is None or len(sequence) <= max_length:
        return sequence
    head = max_length // 2
    return (f"{sequence[:head]}...{sequence[len(sequence) - (max_length - head):]}"
            f"  ({len(sequence):,} characters, the middle not shown)")


#: How a request introduces the guidance the person running the analysis gave. DEFT's and IGSR's
#: adapters use the same words in the prompts they build.
GUIDANCE_INTRODUCTION = ("The person running this analysis gives this guidance on the form of the "
                         "features:")


def _guidance(guidance: str | None) -> list[str]:
    """The lines of a request that show the guidance on the form of the features, or none."""
    return ["", GUIDANCE_INTRODUCTION, guidance] if guidance else []


def render_request(
    context,
    n_features: int,
    examples: list[Example],
    already_proposed: list[str],
    sequence_column: str,
    target_column: str,
    target_kind: str,
    feedback: dict | None = None,
    columns: Mapping[str, Any] | None = None,
    max_example_length: int | None = None,
    guidance: str | None = None,
) -> str:
    """The user message for one proposal round.

    `columns` is `seqmaestro.llm.columns.describe_columns` of the rows the examples come from. The
    columns block, the covariates in the examples and the requirements, and what the requirements
    say about sequences of different lengths, appear only where they add something the task
    description does not say. An example longer than `max_example_length` is shown shortened
    (`shown_sequence`). `guidance` is what the person running the analysis asks of the form of the
    features, shown word for word after the request, and nothing is shown where there is none.
    """
    parts = [*_task(context, columns),
             f"Below are {len(examples)} example rows from the training data.",
             *render_examples(examples, target_column, max_example_length),
             "",
             f"Propose {n_features} candidate features that might {REQUEST[target_kind]}. For each, "
             f"write a Python function computing it for every row of a pandas DataFrame.",
             *_guidance(guidance),
             "",
             *_code_requirements(sequence_column, columns)]

    if already_proposed:
        parts += [
            "",
            "These have already been proposed. Propose different ones -- not restatements, "
            "rescalings or neighbouring windows of these:",
            "  " + ", ".join(already_proposed),
        ]

    if feedback:
        # What a fitted model reported back, for an arm that proposes in rounds. It is rendered as
        # JSON because what a model can say about its fit differs by model (a gain table, a shape
        # summary, a signed coefficient), and the proposer should not need to know which it is
        # looking at.
        parts += [
            "",
            "A model was fitted on the features proposed so far. What it reported:",
            json.dumps(feedback, indent=2, default=str),
            "",
            "Propose features that address what this shows -- what the model could not use, what "
            "it leaned on and might be over-relying on, or what the errors suggest is missing.",
        ]

    return "\n".join([*parts, "", *_answer_format()])


def render_repair_request(context, failures: list[tuple[str, str, Check]], sequence_column: str,
                          columns: Mapping[str, Any] | None = None) -> str:
    """The user message sending back functions that failed on the training rows.

    `failures` holds each function's name, the code last tried for it, and how that code failed
    (`seqmaestro.proposer.checks.Check`). The language model is told how to correct each kind of
    failure (`REPAIR_GUIDANCE`) and how to name a correction (`REPAIR_NAMING`), under the same task,
    columns and code requirements as the request that produced the functions.
    """
    present = {checked.failure for _, _, checked in failures}
    parts = [*_task(context, columns),
             f"{len(failures)} of the functions you proposed failed when they were run on the "
             f"training rows. Each is shown below with its code and what went wrong. Return one "
             f"corrected function for each.",
             "",
             "How to correct a function depends on what went wrong:",
             *(f"  - {text}" for failures_of, text in REPAIR_GUIDANCE if failures_of & present),
             "",
             REPAIR_NAMING,
             "",
             *_code_requirements(sequence_column, columns),
             "",
             "The functions that failed:"]
    for name, code, checked in failures:
        parts += ["", f"Function: {name}", f"What went wrong: it {checked.detail.rstrip('.')}.",
                  "Code:", code]
    return "\n".join([*parts, "", *_answer_format(repair=True)])


def _task(context, columns: Mapping[str, Any] | None) -> list[str]:
    """The task, and the columns block where there is one, each followed by a blank line."""
    block = render_columns(columns) if columns is not None else None
    return [context.render(), "", *([block, ""] if block is not None else [])]


def _code_requirements(sequence_column: str, columns: Mapping[str, Any] | None) -> list[str]:
    """What the code must do, and the worked example of its shape. What it says about covariates
    and about sequences of different lengths is there only where the rows have them."""
    frame = f"a DataFrame with a {sequence_column!r} column of strings"
    if columns is not None and columns["covariates"]:
        frame += ", and a column of numbers for each covariate listed above"
    lengths = None if columns is None else columns["sequence"]["length"]
    varies = isinstance(lengths, Mapping)
    lines = [
        "Requirements for the code:",
        f"  - exactly one argument, named X: {frame}",
        "  - return one number per row -- a pandas Series or numpy array of length len(X)",
        "  - you may use numpy, pandas, re, math, collections and itertools; np and pd are already "
        "bound, and nothing else may be imported",
        "  - no top-level code: definitions only, no example usage, no prints",
        "  - the feature name must be a legal Python identifier: letters, digits and underscores, "
        "not starting with a digit",
    ]
    if varies:
        lines += [
            f"  - the sequences differ in length, from {lengths['min']:,} to {lengths['max']:,} "
            f"characters, so the code must work for every length",
            "  - where a feature is undefined for a row, such as a window that runs past the end "
            "of a short sequence, return NaN for that row instead of raising an error",
        ]
    if isinstance(lengths, int):
        contract = CODE_CONTRACT.format(sequence_column=sequence_column,
                                        window=example_window(lengths))
    else:
        contract = CODE_CONTRACT_ANY_LENGTH.format(sequence_column=sequence_column)
    return [*lines, "", "Example of the expected shape:", "", contract]


def _answer_format(repair: bool = False) -> list[str]:
    """How the language model is to answer: a JSON array of named functions and nothing else. In a
    repair, each entry also names the function it corrects."""
    entry = {
        **({"replaces": "<the name of the function this entry corrects>"} if repair else {}),
        "name": "<identifier>",
        "description": "<one sentence, what it computes>",
        "rationale": "<one sentence, why it might predict the outcome>",
        "code": "<the function definition>",
    }
    return ["Answer with a JSON array only, no other text:", json.dumps([entry], indent=2)]


def parse_proposals(text: str) -> tuple[list[Proposal], str | None]:
    """Parse a reply into proposals, returning what parsed and why the rest did not.

    Prose and code fences around the JSON are tolerated, because insisting on a clean reply would
    discard good features for a formatting slip. A proposal missing its code or its name is dropped
    without repair, because a guessed name or code would give a feature the language model did not
    write.
    """
    if not text or not text.strip():
        return [], "empty reply"

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1).strip()

    array = re.search(r"\[.*\]", candidate, re.S)
    if not array:
        return [], f"no JSON array in reply: {text[:120]!r}"

    try:
        payload = json.loads(array.group(0))
    except json.JSONDecodeError as exc:
        return [], f"unparseable JSON: {exc}"

    if not isinstance(payload, list):
        return [], f"expected a JSON array, got {type(payload).__name__}"

    proposals, skipped = [], 0
    for entry in payload:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("code"):
            skipped += 1
            continue
        proposals.append(
            Proposal(
                name=str(entry["name"]).strip(),
                description=str(entry.get("description") or "").strip(),
                rationale=str(entry.get("rationale") or "").strip(),
                code=str(entry["code"]),
                replaces=str(entry["replaces"]).strip() if entry.get("replaces") else None,
            )
        )

    error = f"{skipped} entries lacked a name or code" if skipped else None
    return proposals, error
