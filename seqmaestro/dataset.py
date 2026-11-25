"""The Stage 1 standard: a prepared dataset and the `dataset.json` that describes it.

Each dataset is built by a script written for it, and whichever script built it, `finalise`
describes the result in the same form, in a file written beside the three split files:

    <splits>/train.csv, val.csv, test.csv
    <splits>/dataset.json

Later stages read `dataset.json` alone, never a record a builder wrote, so a stage never needs code
for one builder's conventions. The file holds what the builder decided, copied from the declaration,
and what `finalise` counts from the files themselves, with the result of every check.
`load_dataset` refuses a dataset with a failed blocking check that was not acknowledged, whereas an
advisory finding is reported, and settled in Stage 1.

**This module opens the held-out split, which later stages do not open before the reveal.** Stage 1
makes the split, so it counts the test rows and measures what they can detect, whereas
`load_dataset` re-checks the digests of the training and validation files only (see
`seqmaestro.harness.heldout`).

`validate_declared` lists the declaration's fields, and `run_checks` lists the checks.
`docs/developer/dataset.md` explains the two kinds of fact and why a check is blocking or advisory,
and `experiments/prepare/finalise_dataset.py` is the command-line front end for this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqmaestro import __version__
from seqmaestro.features import COVARIATES, parse_pool
from seqmaestro.features.base import IDENTIFIER, Inputs
from seqmaestro.features.positional import PositionalWindow
from seqmaestro.target import BINARY, CONTINUOUS, TARGET_KINDS, Target, target_for

#: The file `finalise` writes beside the splits, and the only record of them `load_dataset` reads.
DATASET_FILE = "dataset.json"

#: The group map, written beside the splits: which rows a grouped split must keep together, one row
#: per sequence and group. It is keyed on the sequence because that is the only key the split files
#: are sure to hold. A record keyed on a transcript or a site could not be joined back to them, so
#: it could only restate what the builder claimed.
GROUP_FILE = "groups.csv"

#: The group map's two columns. `group` is the connected component over all the declared relations
#: together. One relation's raw value is not enough: two rows can share a gene without sharing a
#: window, or share a window without sharing a gene, and a split keyed on either alone leaves the
#: other relation free to straddle the split. A sequence belonging to more than one group gets a
#: row for each, which records that the declared grouping alone does not decide which split that
#: sequence goes to.
GROUP_COLUMNS = ("sequence", "group")

#: Incremented when a field changes meaning or moves, so that a file written under an earlier
#: version, which would be misread, is refused.
SCHEMA_VERSION = 1

#: The three splits every prepared dataset has, in the order they are reported.
SPLITS = ("train", "val", "test")

#: How each check can come out. `not_applicable` is distinct from `passed`, because a grouping check
#: that could not run for lack of a declared grouping has not shown that the grouping holds.
#: `flagged` marks an advisory check that found something, and `acknowledged` marks a check whose
#: finding the declaration has answered.
PASSED, FAILED, NOT_APPLICABLE = "passed", "failed", "not_applicable"
FLAGGED, ACKNOWLEDGED = "flagged", "acknowledged"

#: Every check `finalise` runs, in the order they are reported. `run_checks` says what each one tests.
CHECKS = (
    "readme_present",
    "target_valid",
    "target_varies_in_every_split",
    "target_distribution",
    "alphabet_seen_in_training",
    "lengths_as_declared",
    "sequence_lengths",
    "positional_window_fits",
    "covariates_valid",
    "no_sequence_in_two_splits",
    "groups_disjoint",
    "weights_valid",
    "filters_account_for_rows",
)

#: The checks that report a finding and do not refuse the dataset. Whether a validation or test
#: sequence that is also a training sequence is a leak or the shape of the data depends on the
#: experiment. One dataset measures the same site in two sequencing replicates, and another repeats
#: a context because its unit is an event and not a window. Counting cannot tell those cases apart,
#: so `no_sequence_in_two_splits` states what it found and Stage 1 decides. Lengths that go with
#: the target (`sequence_lengths`) are the same kind of finding: a model can take a length from
#: counts and from padding, and whether that is the biology or how the rows were collected is for
#: Stage 1 to say. So is a continuous target whose tails or a single value hold an unusual share of
#: the rows (`target_distribution`): whether the tails are the signal or the noise of the
#: measurement is known to whoever measured it. Every other check finds a defect, whatever the
#: dataset.
ADVISORY = frozenset({"target_distribution", "no_sequence_in_two_splits", "sequence_lengths"})

#: A continuous target's tails are heavy when more than this share of its training rows lies beyond
#: three robust standard deviations of its median. A normal distribution puts 0.27% there.
HEAVY_TAIL_SHARE = 0.01

#: A continuous target is piled on one value when that value repeats and covers more than this
#: share of its training rows.
PILED_SHARE = 0.10

_DECLARED_KEYS = {
    "required": {"columns", "target", "sequences", "split"},
    "optional": {"classes", "covariates", "filters", "sources", "builders", "acknowledged", "notes"},
}

#: What the declaration's `target` section holds for each kind. A binary target's labels are
#: described in `classes`, and a continuous target by its name and what it measures.
_TARGET_KEYS = {BINARY: {"kind"}, CONTINUOUS: {"kind", "name", "meaning"}}

#: What the declaration's `sequences` section may hold. `positional_window` is optional.
_SEQUENCE_KEYS = {"window", "anchor", "orientation", "length", "positional_window"}


class DatasetError(ValueError):
    """A prepared dataset that cannot be used as it stands. The message says what to fix."""


@dataclass(frozen=True)
class ClassInfo:
    """What one label value means, in the words a report prints beside every direction."""

    name: str
    meaning: str


@dataclass(frozen=True)
class Columns:
    """Which column of a split file holds what, and what kind of target the target column holds:
    the `columns` section of `dataset.json`, with its `target.kind`.

    Every field is required, `covariates` included, so a dataset with none says so, and a dataset's
    covariates cannot be dropped by forgetting them. `inputs` is the only place where a split frame
    becomes the rows a featuriser is given, so a stage never picks the sequence or the covariates
    out of a frame itself. The kind is kept with the columns because every stage that is handed
    them fits or scores against the target, and the two kinds are fitted and scored differently.
    """

    sequence: str
    target: str
    weight: str | None
    covariates: tuple[str, ...]
    #: One of `seqmaestro.target.TARGET_KINDS`.
    target_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariates", tuple(self.covariates))
        target_for(self.target_kind)

    def inputs(self, frame: pd.DataFrame) -> Inputs:
        """The sequences and covariates of these rows, indexed as the frame is. The label and the
        weight stay behind, so code that is handed `Inputs` cannot see them."""
        missing = [column for column in (self.sequence, *self.covariates) if column not in frame.columns]
        if missing:
            raise DatasetError(f"The rows have no column {', '.join(missing)}; they have "
                               f"{', '.join(map(str, frame.columns))}.")
        return Inputs(frame[self.sequence], frame[list(self.covariates)])

    def without_covariates(self) -> Columns:
        """These columns without the covariates: what an arm is given when its pool does not hold
        them."""
        return Columns(self.sequence, self.target, self.weight, (), self.target_kind)


@dataclass(frozen=True)
class Dataset:
    """A finalised dataset, read back from its `dataset.json`.

    The columns, the target's kind and the classes are typed fields because every stage uses them.
    `classes` is empty for a continuous target. Everything else is in `record`, which is the file
    as written.
    """

    directory: Path
    sequence_column: str
    target_column: str
    weight_column: str | None
    covariate_columns: tuple[str, ...]
    target_kind: str
    classes: dict[int, ClassInfo]
    record: dict[str, Any] = field(repr=False)

    @property
    def columns(self) -> Columns:
        return Columns(self.sequence_column, self.target_column, self.weight_column,
                       self.covariate_columns, self.target_kind)

    @property
    def target(self) -> Target:
        """The arithmetic of this dataset's kind of target (`seqmaestro.target`)."""
        return target_for(self.target_kind)

    def inputs(self, frame: pd.DataFrame) -> Inputs:
        """The rows a featuriser is given, from one of this dataset's split frames
        (`Columns.inputs`)."""
        return self.columns.inputs(frame)

    def columns_for(self, pool: str) -> Columns:
        """The columns an arm on `pool` is given. The covariates are among them only when the pool
        holds them, so an arm whose pool does not hold them never sees them, in its model or in its
        prompts."""
        return self.columns if COVARIATES in parse_pool(pool) else self.columns.without_covariates()

    def covariates_for(self, pool: str) -> dict[str, str] | None:
        """The covariates' descriptions an arm on `pool` is handed, or None where the pool does not
        hold them. Raises `DatasetError` for a pool that holds them on a dataset that declares
        none."""
        if COVARIATES not in parse_pool(pool):
            return None
        if not self.covariate_columns:
            raise DatasetError(f"Pool {pool!r} holds the covariates, and {self.directory} declares "
                               f"none.")
        return self.covariates

    def shown_columns(self, pools) -> dict[str, str | None]:
        """Each column a language model on any of `pools` is shown by name, with its declared
        description, or None where it has none: the sequence and target columns always, and the
        covariates where a pool holds them. In a blinded run, these names and descriptions must not
        mention what the run withholds (`seqmaestro.llm.blinding_leaks`)."""
        shown: dict[str, str | None] = {self.sequence_column: None, self.target_column: None}
        for pool in pools:
            shown |= self.covariates_for(pool) or {}
        return shown

    @property
    def covariates(self) -> dict[str, str]:
        """Each covariate's declared description, in the declared order."""
        described = self.record.get("covariates") or {}
        return {name: described[name]["description"] for name in self.covariate_columns}

    @property
    def declared_length(self) -> int | str:
        """The length every sequence has, or `variable`, as the declaration states it."""
        return self.record["sequences"]["length"]

    @property
    def positional_window(self) -> PositionalWindow | None:
        """The declared window for positional features, or None where none is declared."""
        window = self.record["sequences"].get("positional_window")
        return None if window is None else PositionalWindow(**window)

    @property
    def train_path(self) -> Path:
        return self.directory / "train.csv"

    @property
    def val_path(self) -> Path:
        return self.directory / "val.csv"

    @property
    def split_counts(self) -> dict[str, dict[str, Any]]:
        return self.record["splits"]

    @property
    def alphabet(self) -> list[str]:
        return self.record["sequences"]["alphabet"]


# ------------------------------------------------------------------------------ the declaration


def _require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise DatasetError(f"`{where}` must be a mapping, not {type(value).__name__}.")
    return dict(value)


def _require_text(section: Mapping, key: str, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"`{where}.{key}` must be non-empty text.")
    return value.strip()


def _class_key(key: Any, where: str) -> str:
    """A label value as the string JSON keys it under, refusing anything but 0 and 1."""
    text = str(key).strip()
    if text not in {"0", "1"}:
        raise DatasetError(f"`{where}` has key {key!r}; the labels are 1 and 0.")
    return text


def _whole_number(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetError(f"`{where}` must be a whole number of rows, not {value!r}.")
    return value


def _row_count(section: Any, where: str, target: Target) -> dict[str, int] | int:
    """A filter's rows before or after it: per class for a binary target, whose filters can remove
    different numbers of each class, and one count for a continuous target."""
    if not target.labels:
        return _whole_number(section, where)
    counts = _require_mapping(section, where)
    out = {_class_key(key, where): value for key, value in counts.items()}
    if set(out) != {"0", "1"}:
        raise DatasetError(f"`{where}` must give a count for both labels, 1 and 0.")
    return {key: _whole_number(value, f"{where}.{key}") for key, value in out.items()}


def _classes(section: Any) -> dict[str, dict[str, str]]:
    classes = _require_mapping(section, "classes")
    normalised = {}
    for key, value in classes.items():
        label = _class_key(key, "classes")
        entry = _require_mapping(value, f"classes.{label}")
        normalised[label] = {"name": _require_text(entry, "name", f"classes.{label}"),
                             "meaning": _require_text(entry, "meaning", f"classes.{label}")}
    if set(normalised) != {"0", "1"}:
        raise DatasetError("`classes` must describe both labels, 1 and 0.")
    if normalised["0"]["name"] == normalised["1"]["name"]:
        raise DatasetError("`classes` gives both labels the same name.")
    return normalised


def _target(section: Any, classes: Any) -> tuple[dict[str, str], dict[str, dict[str, str]] | None]:
    """The declared target and, for a binary one, what its two classes mean.

    A binary target is described by `classes` (one name and meaning per label), which a report
    prints beside every direction. A continuous target is described by `target.name` (a short noun
    phrase a report uses in a sentence) and `target.meaning` (what the number measures, in which
    units, and what a higher value means).
    """
    target = _require_mapping(section, "target")
    kind = target.get("kind")
    if kind not in TARGET_KINDS:
        raise DatasetError(f"`target.kind` must be {' or '.join(TARGET_KINDS)}, not {kind!r}. A "
                           f"multi-class or ordinal target is not supported.")
    unknown = sorted(set(target) - _TARGET_KEYS[kind])
    if unknown:
        raise DatasetError(f"`target` of a {kind} target has no key {', '.join(unknown)}; "
                           f"expected {', '.join(sorted(_TARGET_KEYS[kind]))}.")
    if kind == BINARY:
        if classes is None:
            raise DatasetError("A binary target needs `classes`, describing labels 1 and 0.")
        return {"kind": BINARY}, _classes(classes)
    if classes is not None:
        raise DatasetError("`classes` describes the labels of a binary target. A continuous target "
                           "is described by `target.name` and `target.meaning`.")
    return {"kind": CONTINUOUS, "name": _require_text(target, "name", "target"),
            "meaning": _require_text(target, "meaning", "target")}, None


def _string_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DatasetError(f"`{where}` must be a list of column names.")
    return list(value)


def _covariate_columns(columns: Mapping, weight: str | None) -> list[str]:
    """The declared covariate columns, in order. Each must be a legal Python identifier, because a
    generated function indexes a covariate by name and a covariate's feature is named after it."""
    names = _string_list(columns.get("covariates"), "columns.covariates")
    illegal = [name for name in names if not IDENTIFIER.match(name)]
    if illegal:
        raise DatasetError(f"`columns.covariates` names {', '.join(illegal)}, which are not legal "
                           f"Python identifiers. Rename the columns when the dataset is built.")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise DatasetError(f"`columns.covariates` names {', '.join(repeated)} more than once.")
    taken = {columns.get("sequence"), columns.get("target"), weight} & set(names)
    if taken:
        raise DatasetError(f"`columns.covariates` names {', '.join(sorted(taken))}, which is already "
                           f"the sequence, the label or the weight.")
    return names


def _covariate_descriptions(section: Any, names: list[str]) -> dict[str, dict[str, str]]:
    """What each covariate measures, one entry per declared column and none besides.

    The description is required: it is what the language models are told about the column and what
    a report prints beside a feature built on it, and a name alone rarely says what was measured.
    """
    described = _require_mapping(section or {}, "covariates")
    missing = [name for name in names if name not in described]
    if missing:
        raise DatasetError(f"`covariates` must describe every declared covariate; missing "
                           f"{', '.join(missing)}.")
    extra = sorted(set(described) - set(names))
    if extra:
        raise DatasetError(f"`covariates` describes {', '.join(map(str, extra))}, which "
                           f"`columns.covariates` does not declare.")
    normalised = {}
    for name in names:
        entry = _require_mapping(described[name], f"covariates.{name}")
        unknown = sorted(set(entry) - {"description"})
        if unknown:
            raise DatasetError(f"`covariates.{name}` has no key {', '.join(unknown)}; expected "
                               f"description.")
        normalised[name] = {"description": _require_text(entry, "description", f"covariates.{name}")}
    return normalised


def _positional_window(section: Any, length: int | str) -> dict[str, Any] | None:
    """The declared positional window after checking, or None where there is none.

    `pad` has no default: whether a sequence shorter than the window is padded or refused is a
    decision the user confirms at Stage 1, so the declaration must state it.
    """
    if section is None:
        return None
    window = _require_mapping(section, "sequences.positional_window")
    keys = {"length", "anchor", "pad"}
    if set(window) != keys:
        raise DatasetError(f"`sequences.positional_window` must state length, anchor and pad, and "
                           f"nothing else; it has {', '.join(sorted(window)) or 'nothing'}.")
    try:
        checked = PositionalWindow(**window)
    except ValueError as error:
        raise DatasetError(f"`sequences.positional_window`: {error}") from error
    if isinstance(length, int) and checked.length > length:
        raise DatasetError(f"`sequences.positional_window` is {checked.length} long, and every "
                           f"sequence is {length}; the window cannot be longer than the sequences.")
    return checked.as_dict()


def validate_declared(declared: Mapping[str, Any]) -> dict[str, Any]:
    """The builder's declaration after checking and normalising, or `DatasetError` naming the bad
    field.

    Required fields have no defaults. Which column is the sequence, what kind of target it has and
    what it means, and how the split was grouped are facts about one dataset, and a default here
    would apply one dataset's answer to the next one silently. A required field whose answer is none
    is stated all the same: `columns.weight` and `split.seed` as null, and `split.grouped_by` and
    `split.stratified_on` as null or an empty list. The fields marked optional below may be left
    out, and each then comes back empty or null. An unknown section is refused, and so is an unknown
    key in `columns`, `target`, `sequences`, `split` or `covariates`.

    - `columns.sequence` and `columns.target`: the split files' sequence and target columns.
    - `columns.weight`: the column of per-row fitting weights, or null. Weights are used in fitting
      only, and every metric is unweighted.
    - `columns.covariates` (optional): the covariate columns, in the order the models and the
      language models are given them. Each name must be a legal Python identifier, given once, and
      none may be the sequence, target or weight column.
    - `covariates.<name>.description`: required for each declared covariate, and for no other
      column. It says what the covariate measures, in the words a domain expert would use.
    - `target.kind`: `binary` or `continuous`.
    - `target.name` and `target.meaning`: required for a continuous target, and refused for a binary
      one. The name is a short noun phrase a report uses in a sentence, and the meaning says what
      the number measures, in which units, and what a higher value means.
    - `classes`: required for a binary target, and refused for a continuous one. It gives each label
      (`1` and `0`) a short `name`, different for the two, and a one-line `meaning`. A report
      prints these beside every direction, so they must say which way round the labels run.
    - `sequences.window`: what each sequence is, in words: its width and what it covers.
    - `sequences.anchor`: the landmark the window is taken around, so that a position means the same
      kind of site in every row.
    - `sequences.orientation`: the strand or reading direction, and how it was verified.
    - `sequences.length`: the length of every sequence, or `variable`.
    - `sequences.positional_window` (optional): the `length`, `anchor` and `pad` of the frame in
      which positional features are measured (`seqmaestro.features.positional.PositionalWindow`).
      It cannot be longer than sequences of a fixed length. Where the lengths vary, a run needs one
      if any of its pools names `positional`, as every tree arm's pool does.
    - `split.method`: how the rows were divided, in words.
    - `split.grouped_by`: the columns whose values make two rows dependent, or an empty list for a
      split that is not grouped. A grouped split needs its group map (`GROUP_FILE`) beside it.
    - `split.stratified_on`: the columns the split balances, or an empty list.
    - `split.seed`: the seed the split used, or null.
    - `filters` (optional, but expected): each step that could remove rows, in order, with `step`,
      `why`, `rows_in` and `rows_out`. The rows are counted per label for a binary target, and as
      one count for a continuous one. A step that removed nothing is recorded too.
    - `sources` and `builders` (optional): the delivered files with their digests, and the scripts
      that built the dataset.
    - `acknowledged` (optional): the answer to a check's finding, under the check's name. For an
      advisory finding the answer says what the finding means for this dataset, and for a failure
      it gives the reason the failure is accepted.
    - `notes` (optional): anything a reader of the dataset needs that has no field.
    """
    declared = _require_mapping(declared, "the declaration")
    missing = sorted(_DECLARED_KEYS["required"] - set(declared))
    if missing:
        raise DatasetError(f"The declaration is missing {', '.join(missing)}.")
    unknown = sorted(set(declared) - _DECLARED_KEYS["required"] - _DECLARED_KEYS["optional"])
    if unknown:
        raise DatasetError(f"The declaration has no field {', '.join(unknown)}. See docs/developer/dataset.md.")

    columns = _require_mapping(declared["columns"], "columns")
    unknown = sorted(set(columns) - {"sequence", "target", "weight", "covariates"})
    if unknown:
        raise DatasetError(f"`columns` has no key {', '.join(unknown)}; expected sequence, target, "
                           f"weight, covariates.")
    # Stated even where the answer is none: a key left out would be taken for an answer nobody gave.
    if "weight" not in columns:
        raise DatasetError("`columns` is missing weight: the column of per-row weights, or null "
                           "where the design has none.")
    weight = columns["weight"]
    if weight is not None and (not isinstance(weight, str) or not weight):
        raise DatasetError("`columns.weight` must be a column name or null.")
    normalised_columns = {
        "sequence": _require_text(columns, "sequence", "columns"),
        "target": _require_text(columns, "target", "columns"),
        "weight": weight,
        "covariates": _covariate_columns(columns, weight),
    }
    normalised_covariates = _covariate_descriptions(declared.get("covariates"),
                                                    normalised_columns["covariates"])

    normalised_target, normalised_classes = _target(declared["target"], declared.get("classes"))
    target = target_for(normalised_target["kind"])

    sequences = _require_mapping(declared["sequences"], "sequences")
    unknown = sorted(set(sequences) - _SEQUENCE_KEYS)
    if unknown:
        raise DatasetError(f"`sequences` has no key {', '.join(unknown)}; expected "
                           f"{', '.join(sorted(_SEQUENCE_KEYS))}.")
    length = sequences.get("length")
    if not (length == "variable" or (isinstance(length, int) and not isinstance(length, bool)
                                     and length > 0)):
        raise DatasetError("`sequences.length` must be a positive whole number or 'variable'.")
    normalised_sequences = {
        "window": _require_text(sequences, "window", "sequences"),
        "anchor": _require_text(sequences, "anchor", "sequences"),
        "orientation": _require_text(sequences, "orientation", "sequences"),
        "length": length,
        "positional_window": _positional_window(sequences.get("positional_window"), length),
    }

    split = _require_mapping(declared["split"], "split")
    unknown = sorted(set(split) - {"method", "grouped_by", "stratified_on", "seed"})
    if unknown:
        raise DatasetError(f"`split` has no key {', '.join(unknown)}.")
    # Each is stated even where the answer is none. A split with no `grouped_by` is taken as
    # ungrouped, and its rows are then not checked for a group on both sides, so a key that was
    # only forgotten must not pass for that answer.
    absent = [key for key in ("grouped_by", "stratified_on", "seed") if key not in split]
    if absent:
        raise DatasetError(f"`split` is missing {', '.join(absent)}: state each, with an empty list "
                           f"for a split that is not grouped or not stratified, and null for one "
                           f"that used no seed.")
    seed = split["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DatasetError("`split.seed` must be a whole number, or null for a split that used none.")
    normalised_split = {
        "method": _require_text(split, "method", "split"),
        "grouped_by": _string_list(split["grouped_by"], "split.grouped_by"),
        "stratified_on": _string_list(split["stratified_on"], "split.stratified_on"),
        "seed": seed,
    }

    filters = []
    for index, step in enumerate(declared.get("filters") or []):
        where = f"filters[{index}]"
        step = _require_mapping(step, where)
        filters.append({"step": _require_text(step, "step", where),
                        "why": _require_text(step, "why", where),
                        "rows_in": _row_count(step.get("rows_in"), f"{where}.rows_in", target),
                        "rows_out": _row_count(step.get("rows_out"), f"{where}.rows_out", target)})

    acknowledged = _require_mapping(declared.get("acknowledged") or {}, "acknowledged")
    for check, reason in acknowledged.items():
        if check not in CHECKS:
            raise DatasetError(f"`acknowledged` names {check!r}, which is not a check. "
                               f"Checks: {', '.join(CHECKS)}.")
        if not isinstance(reason, str) or not reason.strip():
            raise DatasetError(
                f"`acknowledged.{check}` must give the reason: what the finding means for this "
                f"dataset, in the words of whoever knows it.")

    notes = declared.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise DatasetError("`notes` must be text.")

    return {
        "columns": normalised_columns,
        "covariates": normalised_covariates,
        "target": normalised_target,
        "classes": normalised_classes,
        "sequences": normalised_sequences,
        "split": normalised_split,
        "filters": filters,
        "sources": list(declared.get("sources") or []),
        "builders": _string_list(declared.get("builders"), "builders"),
        "acknowledged": {check: reason.strip() for check, reason in acknowledged.items()},
        "notes": notes,
    }


# ------------------------------------------------------------------------------ what can be counted


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes, so a split edited after it was finalised is caught."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_splits(directory: Path, columns: dict[str, Any]
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """The three split files, with only the columns the declaration names, as strings and numbers,
    and each file's full header.

    A missing sequence, label or weight column is refused here. A missing covariate is skipped and
    left to `covariates_valid`, so the record is still written and says which split lacks it.
    """
    wanted = [columns["sequence"], columns["target"]] + ([columns["weight"]] if columns["weight"] else [])
    frames, headers = {}, {}
    for split in SPLITS:
        path = directory / f"{split}.csv"
        if not path.exists():
            raise DatasetError(f"{path} is missing; every prepared dataset has {', '.join(SPLITS)}.")
        header = list(pd.read_csv(path, nrows=0).columns)
        absent = [column for column in wanted if column not in header]
        if absent:
            raise DatasetError(f"{path} has no column {', '.join(absent)}; it has {', '.join(header)}.")
        present = [column for column in columns["covariates"] if column in header]
        frames[split] = pd.read_csv(path, usecols=wanted + present,
                                    dtype={columns["sequence"]: str}, keep_default_na=False)
        headers[split] = header
    return frames, headers


def covariate_values(column: pd.Series) -> tuple[np.ndarray, list[str]]:
    """A covariate column as floats, and the raw values that are not numbers.

    The split files are read without turning empty fields into NaN, so a missing value is caught
    here and named, instead of passing as a number. `true` and `false`, in upper or lower case,
    become 1 and 0. Anything else that does not parse as a number (an empty field, `NA`, a word)
    comes back as NaN in the array and by its text in the list.
    """
    text = column.astype(str).str.strip()
    lowered = text.str.lower()
    booleans = lowered.isin(["true", "false"])
    values = pd.to_numeric(text.where(~booleans), errors="coerce")
    values = values.where(~booleans, (lowered == "true").astype(float))
    return values.to_numpy(dtype=float), text[values.isna()].tolist()


def _symbols(sequences: pd.Series) -> set[str]:
    return set("".join(sequences.tolist()))


def _check(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def _significant(value: float, digits: int = 6) -> float:
    """A number to `digits` significant digits. Covariates come on any scale, so a fixed number of
    decimals would print one as 0.0 and another with a dozen meaningless digits."""
    return float(f"{float(value):.{digits}g}")


def run_checks(directory: Path, declared: dict[str, Any],
               frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Run every check in `CHECKS` against the split files and the declaration together.

    Each result holds a `status` and a `detail` that says what the check found. A blocking check
    fails on a broken dataset, whereas an advisory check (`ADVISORY`) is flagged when it finds a
    property that only somebody who knows the data can judge. A check that cannot apply to the
    dataset, such as a grouping check on a split that declares no grouping, is `not_applicable`. A
    failure or a finding that the declaration answers under `acknowledged` becomes `acknowledged`,
    with the answer beside the finding.

    - `readme_present` fails when the dataset's `README.md`, in the folder above the splits, is
      missing.
    - `target_valid` fails when a binary target holds anything but 0 or 1, or a continuous one holds
      anything but a finite number (an empty field, a word, NaN or infinity).
    - `target_varies_in_every_split` fails when a split holds only one class of a binary target, or
      only one value of a continuous one.
    - `target_distribution` (advisory, for a continuous target) reports each split's spread,
      quantiles, skewness and tails. It is flagged when more than `HEAVY_TAIL_SHARE` of the training
      rows lie beyond three robust standard deviations of the median, or when one repeated value
      covers more than `PILED_SHARE` of them.
    - `alphabet_seen_in_training` fails when validation or test uses a symbol that no training
      sequence holds, since a featuriser fitted on the training rows has no column for it.
    - `lengths_as_declared` fails when a sequence is not the declared fixed length.
    - `sequence_lengths` (advisory, for sequences of variable length) is flagged whenever the lengths
      differ. It reports the quartiles of the lengths in each split (and in each class, for a binary
      target), the rank correlation of length with a continuous target, and, with a positional
      window, the share of rows the window pads (or would refuse) and the share it truncates.
    - `positional_window_fits` fails when a sequence is shorter than a positional window declared
      without padding.
    - `covariates_valid` fails when a declared covariate is absent from a split, missing or not a
      number, infinite, or constant on the training rows.
    - `no_sequence_in_two_splits` (advisory) is flagged when a validation or test sequence is also a
      training sequence, or when a test sequence is also a validation one (`test_in_val`). It
      reports the share of each split affected (by class, for a binary target) and, where the
      covariates are valid, how many of the repeated rows differ in their covariates from every
      training copy.
    - `groups_disjoint` fails when two different sequences of one declared group are in different
      splits, or when the group map is missing or does not cover the splits.
    - `weights_valid` fails when a weight is missing, negative or not finite, or when the training
      rows' weights sum to zero (for a binary target, those of either class).
    - `filters_account_for_rows` fails when the last declared filter leaves a different number of
      rows (per class, for a binary target) from the number the splits hold.
    """
    columns, sequence = declared["columns"], declared["columns"]["sequence"]
    target = target_for(declared["target"]["kind"])
    results: dict[str, dict[str, str]] = {}

    readme = directory.parent / "README.md"
    results["readme_present"] = (_check(PASSED, f"{readme.name} is present") if readme.exists()
                                 else _check(FAILED, f"no {readme} describing the dataset"))

    parsed = {split: target.parse(frame[columns["target"]]) for split, frame in frames.items()}
    values = {split: y for split, (y, _) in parsed.items()}
    bad = {split: unusable for split, (_, unusable) in parsed.items() if unusable}
    passed = "every label is 0 or 1" if target.labels else "every value is a finite number"
    results["target_valid"] = (
        _check(PASSED, passed) if not bad
        else _check(FAILED, "; ".join(f"{split} has {', '.join(unusable[:5])}"
                                      for split, unusable in bad.items())))
    results["target_varies_in_every_split"] = _target_varies(values, target)
    results["target_distribution"] = _target_distribution(values, target)

    trained_on = _symbols(frames["train"][sequence])
    unseen = {split: sorted(_symbols(frames[split][sequence]) - trained_on) for split in ("val", "test")}
    unseen = {split: symbols for split, symbols in unseen.items() if symbols}
    results["alphabet_seen_in_training"] = (
        _check(PASSED, f"validation and test use only {''.join(sorted(trained_on))}") if not unseen
        else _check(FAILED, "; ".join(f"{split} uses {''.join(symbols)}, absent from training"
                                      for split, symbols in unseen.items())))

    lengths = pd.concat([frame[sequence].str.len() for frame in frames.values()])
    declared_length = declared["sequences"]["length"]
    window = declared["sequences"]["positional_window"]
    if declared_length == "variable":
        results["lengths_as_declared"] = _check(
            NOT_APPLICABLE, f"declared variable; observed {lengths.min()} to {lengths.max()}")
    else:
        off = int((lengths != declared_length).sum())
        results["lengths_as_declared"] = (
            _check(PASSED, f"every sequence is {declared_length} long") if not off
            else _check(FAILED, f"{off} sequence(s) are not {declared_length} long "
                                f"(observed {lengths.min()} to {lengths.max()})"))

    results["sequence_lengths"] = _sequence_lengths(frames, sequence, values, target,
                                                    declared_length, window)
    results["positional_window_fits"] = _window_fits(frames, sequence, window)
    results["covariates_valid"] = _covariates_valid(frames, columns["covariates"])

    results["no_sequence_in_two_splits"] = _sequences_across_splits(
        frames, sequence, values, target,
        columns["covariates"] if results["covariates_valid"]["status"] == PASSED else ())

    if not declared["split"]["grouped_by"]:
        results["groups_disjoint"] = _check(NOT_APPLICABLE, "the split declares no grouping")
    else:
        results["groups_disjoint"] = _groups_disjoint(directory, declared, frames)

    weight = columns["weight"]
    if weight is None:
        results["weights_valid"] = _check(NOT_APPLICABLE, "the design has no per-row weights")
    else:
        results["weights_valid"] = _weights_valid(frames, weight, values, target)

    filters = declared["filters"]
    if not filters:
        results["filters_account_for_rows"] = _check(
            NOT_APPLICABLE, "no filters declared; a build that removed no rows still records one step")
    else:
        held = (sum(len(y) for y in values.values()) if not target.labels else
                {str(label): sum(int((y == label).sum()) for y in values.values())
                 for label in target.labels})
        last = filters[-1]["rows_out"]
        results["filters_account_for_rows"] = (
            _check(PASSED, f"the last step leaves {_rows_text(last)}, which is what the splits hold")
            if last == held
            else _check(FAILED, f"the last declared step leaves {_rows_text(last)}, but the splits "
                                f"hold {_held_text(held)}; a step that changed the row count is "
                                "missing"))

    for check, reason in declared["acknowledged"].items():
        if results[check]["status"] in (FAILED, FLAGGED):
            results[check] = {**results[check], "status": ACKNOWLEDGED, "acknowledged": reason,
                              "was": results[check]["status"]}
    return {check: results[check] for check in CHECKS}


def _rows_text(rows: dict[str, int] | int) -> str:
    """A filter's row count in a sentence: per class for a binary target, one number otherwise."""
    if isinstance(rows, dict):
        return f"{rows['1']} positive and {rows['0']} negative rows"
    return f"{rows:,} rows"


def _held_text(rows: dict[str, int] | int) -> str:
    return f"{rows['1']} and {rows['0']}" if isinstance(rows, dict) else f"{rows:,}"


def _target_varies(values: dict[str, np.ndarray], target: Target) -> dict[str, str]:
    """Whether every split holds both classes of a binary target, or more than one value of a
    continuous one. Nothing can be fitted or scored on a split whose target does not vary."""
    if target.labels:
        thin = [f"{split} ({int((y == 1).sum())} positive, {int((y == 0).sum())} negative)"
                for split, y in values.items() if (y == 1).sum() == 0 or (y == 0).sum() == 0]
        return (_check(PASSED, "every split holds both classes") if not thin
                else _check(FAILED, f"a split missing a class: {', '.join(thin)}"))
    constant = [f"{split} (every value {_significant(np.nanmax(y)):g})" for split, y in values.items()
                if np.isfinite(y).any() and np.nanmin(y) == np.nanmax(y)]
    return (_check(PASSED, "the target takes more than one value in every split") if not constant
            else _check(FAILED, f"a split whose target is constant: {', '.join(constant)}"))


def _distribution(y: np.ndarray) -> dict[str, Any]:
    """A continuous target's distribution over one split, in the terms `target_distribution` needs:
    its spread and quantiles, its shape, the share of rows in its tails, and its commonest value.

    The tails are measured against a robust standard deviation (the interquartile range divided by
    1.349), which equals the standard deviation for a normal distribution and is not inflated by the
    tails it is used to find. Their share is undefined where that deviation is zero, which happens
    only when at least the middle half of the rows share one value.
    """
    y = y[np.isfinite(y)]
    quantiles = np.percentile(y, [1, 5, 25, 50, 75, 95, 99])
    median, iqr = quantiles[3], quantiles[4] - quantiles[2]
    spread = y.std()
    centred = (y - y.mean()) / spread if spread > 0 else np.zeros_like(y)
    values, counts = np.unique(y, return_counts=True)
    commonest = int(np.argmax(counts))
    robust_sd = iqr / 1.349
    return {
        "rows": len(y),
        "mean": _significant(y.mean()),
        "sd": _significant(y.std(ddof=1)) if len(y) > 1 else None,
        "min": _significant(y.min()),
        "max": _significant(y.max()),
        "quantiles": {f"{q:02d}": _significant(value)
                      for q, value in zip((1, 5, 25, 50, 75, 95, 99), quantiles, strict=True)},
        "skewness": round(float((centred ** 3).mean()), 3),
        "excess_kurtosis": round(float((centred ** 4).mean() - 3), 3),
        "share_beyond_3_robust_sd": (round(float((np.abs(y - median) > 3 * robust_sd).mean()), 4)
                                     if robust_sd > 0 else None),
        "commonest_value": _significant(values[commonest]),
        "commonest_rows": int(counts[commonest]),
        "commonest_share": round(float(counts[commonest] / len(y)), 4),
    }


def _target_distribution(values: dict[str, np.ndarray], target: Target) -> dict[str, Any]:
    """How a continuous target is distributed in each split, and whether its tails or a single value
    hold an unusual share of the training rows.

    This is an advisory check. R-squared weighs a row by its squared distance from the mean, so a
    few rows in a long tail can decide a model's score, and a value that covers much of a split (a
    zero that means no change, or a detection floor) can mix two populations. Whether the tails are
    the signal or the noise of the measurement, and whether a transformation or a filter is
    warranted, can only be answered by whoever knows how the target was measured. The check reports
    the distribution in the terms that decision needs.
    """
    if target.labels:
        return _check(NOT_APPLICABLE, "the target is binary; each split's class counts are recorded "
                                      "with the split")
    if not all(np.isfinite(y).any() for y in values.values()):
        return _check(NOT_APPLICABLE, "a split holds no finite value; see target_valid")
    measured = {split: _distribution(y) for split, y in values.items()}
    train = measured["train"]
    tails, piled = train["share_beyond_3_robust_sd"], train["commonest_share"]
    shape = (f"on the training rows the target runs from {train['min']:g} to {train['max']:g} around "
             f"a median of {train['quantiles']['50']:g}, with "
             + (f"{tails:.1%} of rows beyond three robust standard deviations of the median (a normal "
                f"distribution puts 0.3% there)" if tails is not None else
                "half the rows or more on one value")
             + f"; its commonest value, {train['commonest_value']:g}, covers {piled:.1%}")
    heavy = tails is None or tails > HEAVY_TAIL_SHARE
    if heavy or (train["commonest_rows"] > 1 and piled > PILED_SHARE):
        return {**_check(FLAGGED, shape), "measured": measured}
    return {**_check(PASSED, shape), "measured": measured}


def _sequences_across_splits(frames, sequence: str, targets, target: Target,
                             covariates=()) -> dict[str, Any]:
    """How much of the validation and test splits a run has already seen verbatim.

    This is an advisory check. It reports the share of each split's rows affected, because the
    judgement turns on the share: a count of repeated sequences means little until it is set
    against the size of the split, and the same count can be a leak on one dataset and expected on
    another that repeats by design. For a binary target the share by class is reported too, because
    a leak concentrated in one class affects a score differently from one spread evenly over both.

    A validation or a test row that repeats a training sequence was seen by every model fitted on
    the training rows. A test row that repeats a validation sequence, and no training one, was
    seen by what a run chooses on validation: a threshold, the seed an arm is reduced from, what
    a search or a feedback round keeps, and the reliability panel's selections, which are made on
    training and validation together. Such rows are counted separately, under `test_in_val`.

    Where the dataset declares covariates, the check also counts the repeated rows whose covariates
    differ from every training row with that sequence. Two rows with one sequence and different
    measurements are different samples, so those rows are not seen verbatim. `covariates` is empty
    when the covariates did not pass `covariates_valid`, since they then cannot be compared.
    """
    trained_on = set(frames["train"][sequence])
    twins: dict[str, set[tuple]] = {}
    if covariates:
        values = np.column_stack([covariate_values(frames["train"][name])[0] for name in covariates])
        for text, row in zip(frames["train"][sequence], map(tuple, values), strict=True):
            twins.setdefault(text, set()).add(row)
    detail, measured = [], {}
    for split in ("val", "test"):
        frame = frames[split]
        seen = frame[sequence].isin(trained_on).to_numpy()
        y = targets[split]
        share = float(seen.mean()) if len(seen) else 0.0
        by_class = {str(label): (round(float(seen[y == label].mean()), 4)
                                 if int((y == label).sum()) else None) for label in target.labels}
        measured[split] = {"rows": int(seen.sum()), "of": len(frame), "share": round(share, 4)}
        if target.labels:
            measured[split]["share_by_class"] = by_class
        differing = ""
        if covariates:
            values = np.column_stack([covariate_values(frame[name])[0] for name in covariates])
            differ = sum(tuple(values[row]) not in twins[frame[sequence].iloc[row]]
                         for row in np.flatnonzero(seen))
            measured[split]["differ_in_covariates"] = int(differ)
            differing = f"; {differ:,} of them with covariates unlike any training copy"
        if seen.any():
            classes = "".join(f"; {by_class[str(label)]:.1%} of class {label}" if index == 0 else
                              f", {by_class[str(label)]:.1%} of class {label}"
                              for index, label in enumerate(label for label in target.labels
                                                            if by_class[str(label)] is not None))
            detail.append(f"{int(seen.sum()):,} of {len(frame):,} {split} rows ({share:.1%}"
                          f"{classes}{differing})")
    test = frames["test"]
    chosen_on = set(frames["val"][sequence]) - trained_on
    repeats = test[sequence].isin(chosen_on).to_numpy()
    measured["test_in_val"] = {"rows": int(repeats.sum()), "of": len(test),
                               "share": round(float(repeats.mean()) if len(repeats) else 0.0, 4)}
    found = []
    if detail:
        found.append("a model fitted on the training rows has already seen, verbatim, "
                     + "; ".join(detail))
    if repeats.any():
        found.append(f"{int(repeats.sum()):,} of {len(test):,} test rows "
                     f"({float(repeats.mean()):.1%}) repeat a validation sequence that is no "
                     f"training sequence, and a run makes its choices on validation")
    if not found:
        return _check(PASSED, "no validation or test sequence is also a training sequence, and "
                              "no test sequence is also a validation sequence")
    return {**_check(FLAGGED, ". ".join(found)), "measured": measured}


def read_group_map(directory: Path) -> pd.DataFrame:
    """The group map beside the splits, or `DatasetError` saying what is wrong with it."""
    path = Path(directory) / GROUP_FILE
    if not path.exists():
        raise DatasetError(
            f"{path} is missing. The split declares `grouped_by`, so the grouping has to be checked "
            f"against a record of it: write one with experiments/prepare/write_group_map.py.")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    absent = [column for column in GROUP_COLUMNS if column not in table.columns]
    if absent:
        raise DatasetError(f"{path} has no column {', '.join(absent)}; it has {', '.join(table.columns)}.")
    return table.drop_duplicates()


def _groups_disjoint(directory: Path, declared: dict[str, Any],
                     frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Check that each group of the declared relations sits in one split.

    The group map is joined to the split files on the sequence, so the check sees where the rows
    went instead of relying on what a record of the split says about itself.
    """
    sequence, relations = declared["columns"]["sequence"], declared["split"]["grouped_by"]
    try:
        table = read_group_map(directory)
    except DatasetError as error:
        return _check(FAILED, str(error).replace("\n", " "))

    split_of: dict[str, set[str]] = {}
    for split, frame in frames.items():
        for value in frame[sequence].unique():
            split_of.setdefault(value, set()).add(split)
    unmapped = set(split_of) - set(table[GROUP_COLUMNS[0]])
    if unmapped:
        return _check(FAILED, f"{len(unmapped)} sequence(s) in the splits are absent from "
                              f"{GROUP_FILE}, so their group is unknown; write it again from the "
                              f"dataset these splits were cut from")

    # This check looks for the leak that only grouping can close: two *different* sequences of one
    # group, placed in different splits. Two kinds of sequence cannot show that, so both are set
    # aside and not counted against the grouping:
    #
    # - a sequence sitting in two splits. Counting it here would make every group it belongs to look
    #   as though the grouping had failed. The advisory check `no_sequence_in_two_splits` reports it
    #   instead, both where one of the two splits is training and where a test sequence repeats a
    #   validation one.
    # - a sequence belonging to two groups. The split files name no row, so when a window shared by
    #   two genes appears once in each split it reaches, this check cannot say which gene's row went
    #   where. Taking it as evidence about either gene would report a failure over a placement that
    #   was never observed.
    groups_of: dict[str, set[str]] = {}
    for value, group in zip(table[GROUP_COLUMNS[0]], table[GROUP_COLUMNS[1]], strict=True):
        groups_of.setdefault(value, set()).add(group)

    reach: dict[str, set[str]] = {}
    in_two_splits, in_two_groups = set(), set()
    for value, splits in split_of.items():
        if len(splits) > 1:
            in_two_splits.add(value)
        elif len(groups_of.get(value, ())) > 1:
            in_two_groups.add(value)
        else:
            reach.setdefault(next(iter(groups_of[value])), set()).add(next(iter(splits)))

    named = ", ".join(relations)
    aside = "; ".join(
        part for part in
        (f"{len(in_two_splits)} sequence(s) sit in more than one split and are left to "
         f"no_sequence_in_two_splits" if in_two_splits else "",
         f"{len(in_two_groups)} belong to more than one group, which the split files cannot tell "
         f"apart" if in_two_groups else "") if part)
    aside = f"; {aside}" if aside else ""

    straddling = sorted(group for group, splits in reach.items() if len(splits) > 1)
    if straddling:
        return _check(FAILED, f"{len(straddling)} group(s) of {named} hold sequences in more than "
                              f"one split, e.g. group {straddling[0]!r}{aside}")
    return _check(PASSED, f"no two sequences of one {named} group are in different splits "
                          f"({len(reach):,} group(s) over {len(split_of) - len(in_two_splits) - len(in_two_groups):,} "
                          f"sequences){aside}")


def _weights_valid(frames, weight: str, values, target: Target) -> dict[str, str]:
    problems = []
    for split, frame in frames.items():
        weights = pd.to_numeric(frame[weight], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            problems.append(f"{split} has missing, negative or non-finite weights")
    train_weights = pd.to_numeric(frames["train"][weight], errors="coerce").to_numpy(dtype=float)
    if target.labels:
        for label in target.labels:
            if np.nansum(train_weights[values["train"] == label]) <= 0:
                problems.append(f"the training rows labelled {label} carry no weight")
        carried = "both classes carry weight"
    else:
        if np.nansum(train_weights) <= 0:
            problems.append("the training rows carry no weight")
        carried = "the training rows carry weight"
    return (_check(PASSED, f"weights are finite and non-negative, and {carried}")
            if not problems else _check(FAILED, "; ".join(problems)))


def _quartiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    low, q1, median, q3, high = np.percentile(values, [0, 25, 50, 75, 100])
    return {"min": int(low), "q1": round(float(q1), 1), "median": round(float(median), 1),
            "q3": round(float(q3), 1), "max": int(high)}


def _sequence_lengths(frames, sequence: str, values, target: Target, declared_length,
                      window: dict[str, Any] | None) -> dict[str, Any]:
    """How the lengths of a variable-length dataset are spread, whether they go with the target, and
    what the declared positional window does to them.

    This is an advisory check. A length that goes with the target is something a model can use: a
    k-mer count grows with length, and a padded position says the sequence stopped before it.
    Whether such a relation is the biology or how the rows were collected cannot be settled by
    counting, so the check reports the lengths in the terms that decision needs and Stage 1 answers
    it. For a binary target that is their quartiles by split and class. For a continuous one it is
    their quartiles by split and the rank correlation of length with the target on the training
    rows. With a window, it also reports the share of rows the window pads (or, without padding,
    would have to refuse) and the share it cuts short, by class where the target has classes.
    """
    if declared_length != "variable":
        return _check(NOT_APPLICABLE, f"the dataset declares every sequence {declared_length} long")
    by_split = {}
    for split, frame in frames.items():
        lengths, y = frame[sequence].str.len().to_numpy(), values[split]
        by_split[split] = ({str(label): _quartiles(lengths[y == label]) for label in target.labels}
                           if target.labels else _quartiles(lengths))
    lengths = np.concatenate([frame[sequence].str.len().to_numpy() for frame in frames.values()])
    y = np.concatenate([values[split] for split in frames])
    if lengths.min() == lengths.max():
        return {**_check(PASSED, f"declared variable, and every sequence is {lengths.min()} long"),
                "measured": {"by_split": by_split}}
    measured: dict[str, Any] = {"by_split": by_split}
    if target.labels:
        medians = {label: ("no" if quartiles is None else _length(quartiles["median"]))
                   for label, quartiles in by_split["train"].items()}
        detail = (f"lengths run from {lengths.min():,} to {lengths.max():,}; the median training "
                  f"sequence is {medians['1']} long in class 1 and {medians['0']} in class 0")
    else:
        train_lengths = frames["train"][sequence].str.len()
        rho = train_lengths.corr(pd.Series(values["train"], index=train_lengths.index),
                                 method="spearman")
        measured["spearman_with_target"] = None if pd.isna(rho) else round(float(rho), 4)
        detail = (f"lengths run from {lengths.min():,} to {lengths.max():,}; the median training "
                  f"sequence is {_length(by_split['train']['median'])} long, and on the training "
                  f"rows length has a rank correlation of {rho:.3f} with the target")
    if window is not None:
        def share(rows: np.ndarray) -> dict[str, float | None] | float:
            if not target.labels:
                return round(float(rows.mean()), 4)
            return {str(label): (round(float(rows[y == label].mean()), 4)
                                 if (y == label).any() else None) for label in target.labels}

        shorter, longer = share(lengths < window["length"]), share(lengths > window["length"])
        suffix = "_by_class" if target.labels else ""
        measured["window"] = {**window, f"shorter{suffix}": shorter, f"longer{suffix}": longer}
        fate = "padded" if window["pad"] else "shorter than the window, which is declared without padding"
        if target.labels:
            portions = (f"{_percent(shorter['1'])} of class 1 and {_percent(shorter['0'])} of class 0 "
                        f"are {fate}, and {_percent(longer['1'])} and {_percent(longer['0'])} are")
        else:
            portions = f"{_percent(shorter)} of the sequences are {fate}, and {_percent(longer)} are"
        detail += (f". The positional window reads {window['length']} bases from the "
                   f"{window['anchor']}: {portions} cut at the far end")
    return {**_check(FLAGGED, detail), "measured": measured}


def _length(value: float) -> str:
    """A length for a sentence: whole where it is whole, since a median can fall between two."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _percent(share: float | None) -> str:
    return "none" if share is None else f"{share:.1%}"


def _window_fits(frames, sequence: str, window: dict[str, Any] | None) -> dict[str, str]:
    """Whether every sequence fills the positional window, when the window is declared without
    padding. Shorter sequences are filtered out at Stage 1 in that case, as a recorded step, so
    every arm is given the same rows, and a shorter sequence that remains is a defect."""
    if window is None:
        return _check(NOT_APPLICABLE, "no positional window is declared")
    if window["pad"]:
        return _check(NOT_APPLICABLE, f"the {window['length']}-base window pads shorter sequences, "
                                      f"as declared")
    short = {split: int((frame[sequence].str.len() < window["length"]).sum())
             for split, frame in frames.items()}
    if not any(short.values()):
        return _check(PASSED, f"every sequence is at least {window['length']} long, so each fills "
                              f"the window")
    listing = ", ".join(f"{count:,} in {split}" for split, count in short.items() if count)
    return _check(FAILED, f"sequences shorter than the {window['length']}-base window, which is "
                          f"declared without padding: {listing}. Filter them out as a recorded "
                          f"step, so every arm reads the same rows, or declare the window with "
                          f"padding")


def _covariates_valid(frames, covariates: list[str]) -> dict[str, str]:
    """Whether every declared covariate is present, numeric and finite in every split, and varies
    on the training rows.

    Missing values are refused, not imputed. Imputation, where it is wanted, is a Stage 1 decision
    recorded as a preparation step, and refusing here keeps every later stage simple: an L1 fit
    cannot take a missing value, and a tree sorts one to a side without saying so.
    """
    if not covariates:
        return _check(NOT_APPLICABLE, "the dataset declares no covariates")
    problems = []
    for name in covariates:
        for split, frame in frames.items():
            if name not in frame.columns:
                problems.append(f"{name} is absent from {split}")
                continue
            values, unreadable = covariate_values(frame[name])
            if unreadable:
                problems.append(f"{name} has {len(unreadable):,} missing or non-numeric value(s) in "
                                f"{split}, e.g. {unreadable[0]!r}")
            elif not np.isfinite(values).all():
                problems.append(f"{name} is infinite in {split}")
            elif split == "train" and np.ptp(values) == 0:
                problems.append(f"{name} is constant on the training rows")
    if problems:
        return _check(FAILED, "; ".join(problems))
    return _check(PASSED, f"{len(covariates)} covariate(s), numeric and finite in every split and "
                          f"varying on the training rows")


def _covariate_summaries(frames, covariates: list[str], descriptions: dict[str, dict[str, str]]
                         ) -> dict[str, dict[str, Any]]:
    """Each covariate's description, and its spread on the training rows where its values parse as
    finite numbers."""
    summaries = {}
    for name in covariates:
        training = None
        if name in frames["train"].columns:
            values, unreadable = covariate_values(frames["train"][name])
            if not unreadable and np.isfinite(values).all():
                training = {"min": _significant(values.min()), "max": _significant(values.max()),
                            "mean": _significant(values.mean()),
                            "sd": _significant(values.std(ddof=1)) if len(values) > 1 else None,
                            "distinct": len(np.unique(values))}
        summaries[name] = {**descriptions[name], "training": training}
    return summaries


def _group_counts(directory: Path, declared: dict[str, Any],
                  frames: dict[str, pd.DataFrame]) -> dict[str, int | None]:
    """How many groups each split holds, or None per split where the split declares no grouping or
    its group map cannot be read.

    The value is None, never zero, because a split whose group map is missing or unreadable has an
    unknown number of groups, and zero would be taken for a count. The grouping check reports why
    the map is unusable.
    """
    if not declared["split"]["grouped_by"]:
        return dict.fromkeys(SPLITS)
    try:
        table = read_group_map(directory)
    except DatasetError:
        return dict.fromkeys(SPLITS)
    sequence = declared["columns"]["sequence"]
    groups: dict[str, set[str]] = {}
    for value, group in zip(table[GROUP_COLUMNS[0]], table[GROUP_COLUMNS[1]], strict=True):
        groups.setdefault(value, set()).add(group)
    # Only sequences that belong to one group, as the check counts them. Counting a sequence into
    # every group that claims it would make these sum to more than the dataset has groups.
    single = {value: next(iter(found)) for value, found in groups.items() if len(found) == 1}
    return {split: len({single[value] for value in frame[sequence].unique() if value in single})
            for split, frame in frames.items()}


# ------------------------------------------------------------------------------ writing and reading


def finalise(splits_dir: str | Path, declared: Mapping[str, Any],
             provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Count, check and measure a prepared split, and write its `dataset.json`. Returns the record.

    `declared` is the builder's declaration, whose fields `validate_declared` lists.
    `provenance` is anything the caller wants recorded about how the dataset was made, such as the
    commit. The caller supplies it because this module does not know where the repository is.

    The record is written even when a check fails, so the failure can be read and fixed. Refusing
    the dataset is left to `load_dataset`.
    """
    directory = Path(splits_dir)
    normalised = validate_declared(declared)
    frames, headers = _read_splits(directory, normalised["columns"])
    sequence, target, weight, covariates = (normalised["columns"][key]
                                            for key in ("sequence", "target", "weight", "covariates"))
    declared_columns = {sequence, target, weight, *covariates}
    unused = sorted({column for header in headers.values() for column in header} - declared_columns)

    kind = target_for(normalised["target"]["kind"])
    values = {split: kind.parse(frame[target])[0] for split, frame in frames.items()}
    groups = _group_counts(directory, normalised, frames)
    splits = {}
    total = sum(len(frame) for frame in frames.values())
    for split, frame in frames.items():
        splits[split] = {
            "rows": len(frame),
            "share": round(len(frame) / total, 4) if total else 0.0,
            **kind.summary(values[split]),
            "groups": groups[split],
            "sha256": file_digest(directory / f"{split}.csv"),
        }

    lengths = pd.concat([frame[sequence].str.len() for frame in frames.values()])
    train_weights = (pd.to_numeric(frames["train"][weight], errors="coerce").to_numpy(dtype=float)
                     if weight else None)
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        # `unused` is counted, not declared: columns in a split file that nothing reads, listed so
        # that a column cannot sit in the files unnoticed.
        "columns": {**normalised["columns"], "unused": unused},
        "covariates": _covariate_summaries(frames, covariates, normalised["covariates"]),
        "target": normalised["target"],
        "classes": normalised["classes"],
        "sequences": {
            **normalised["sequences"],
            "alphabet": sorted(_symbols(frames["train"][sequence])),
            "observed_length": {"min": int(lengths.min()), "max": int(lengths.max())},
        },
        "split": normalised["split"],
        "splits": splits,
        "filters": normalised["filters"],
        "checks": run_checks(directory, normalised, frames),
        # Two numbers, because a design can fail on either: what the held-out split can detect,
        # and how much information the training split holds.
        "design_power": {"test": kind.held_out_power(values["test"]),
                         "train": kind.training_information(values["train"], train_weights)},
        "notes": normalised["notes"],
        "provenance": {
            "sources": normalised["sources"],
            "builders": normalised["builders"],
            "seqmaestro_version": __version__,
            **dict(provenance or {}),
        },
    }
    (directory / DATASET_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record


def target_kind(record: Mapping[str, Any]) -> str:
    """The kind of target a `dataset.json` records."""
    return record["target"]["kind"]


def failed_checks(record: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """The blocking checks that failed and were not acknowledged. These stop Stage 2."""
    return {name: result for name, result in record["checks"].items() if result["status"] == FAILED}


def flagged_checks(record: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """The advisory checks whose findings the declaration has not answered yet.

    These do not stop a run. They are the questions Stage 1 puts to whoever knows the data, and each
    stays flagged until the declaration answers it.
    """
    return {name: result for name, result in record["checks"].items() if result["status"] == FLAGGED}


def load_dataset(splits_dir: str | Path) -> Dataset:
    """A finalised dataset, or `DatasetError` saying why it cannot be used.

    Refused when `dataset.json` is missing, was written under another schema version, or records a
    failed check that was not acknowledged, or when the training or validation file differs from
    the digest recorded for it. The test file's digest is checked at the reveal instead, because the
    file stays closed until then.
    """
    directory = Path(splits_dir)
    path = directory / DATASET_FILE
    if not path.exists():
        raise DatasetError(
            f"{path} does not exist. Every dataset is finalised at the end of Stage 1: run "
            f"experiments/prepare/finalise_dataset.py --splits {directory} --declared <file>.")
    record = json.loads(path.read_text())
    if record.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(
            f"{path} was written at schema version {record.get('schema_version')!r}; this reader is "
            f"version {SCHEMA_VERSION}. Finalise the dataset again.")
    failed = failed_checks(record)
    if failed:
        listing = "; ".join(f"{name}: {result['detail']}" for name, result in failed.items())
        raise DatasetError(f"{path} records failed checks that the declaration does not acknowledge: {listing}")
    for split in ("train", "val"):
        recorded = record["splits"][split]["sha256"]
        if file_digest(directory / f"{split}.csv") != recorded:
            raise DatasetError(
                f"{directory / (split + '.csv')} has changed since the dataset was finalised. "
                "Finalise it again, or restore the file.")
    columns = record["columns"]
    return Dataset(
        directory=directory,
        sequence_column=columns["sequence"],
        target_column=columns["target"],
        weight_column=columns["weight"],
        covariate_columns=tuple(columns["covariates"]),
        target_kind=target_kind(record),
        classes={int(label): ClassInfo(**entry)
                 for label, entry in (record.get("classes") or {}).items()},
        record=record,
    )
