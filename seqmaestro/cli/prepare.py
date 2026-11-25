"""Stage 1 from one table: the columns, the target, the rows kept, the split and the files.

`seqmaestro run` prepares a dataset from a table of sequences with the defaults below, which the
agent would decide with the user one by one (`process/task-construction.md`). It writes the files
a dataset prepared with an agent has, where the agent writes them, and finalises the dataset
(`seqmaestro.dataset.finalise`), so every later stage reads it as it reads any other:

    data/<name>/README.md, raw/<the table>, splits/{train,val,test}.csv, splits/groups.csv
    experiments/prepare/declarations/<name>__splits.yaml
    experiments/datasets/<name>.json

The defaults:

- The sequences are used as delivered, in upper case. A row whose sequence is missing or holds a
  symbol other than A, C, G, T (U in place of T for RNA) and N is dropped, as one recorded step.
- The target is binary where its column holds two values, and continuous where it holds more, all
  of them numbers. The user names the value of the positive class.
- Sequences of different lengths keep their lengths, and positional features read a window as long
  as the shortest sequence, from the end the user chooses, so that no row is padded or dropped.
- The split assigns 7, 1 and 2 of 10 folds to training, validation and the held-out rows. It keeps
  together the rows that share a value of a grouping column, and the rows whose sequences are
  identical, and balances the classes across the folds, or five bins of a continuous target.

Covariates, weights, a window cut around a landmark, and filters on the target are left to the
agent's way, as is a check of the sequences' orientation.
"""

from __future__ import annotations

import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from seqmaestro import files
from seqmaestro.dataset import GROUP_FILE, SPLITS, file_digest, finalise
from seqmaestro.grouping import connected_groups, group_map
from seqmaestro.pipeline.repository import REPO
from seqmaestro.provenance import git_is_dirty, git_sha
from seqmaestro.target import BINARY, CONTINUOUS

#: How a table's columns are separated, by the extension of its file (before any `.gz`).
SEPARATORS = {".csv": ",", ".tsv": "\t", ".tab": "\t"}

#: What a cell holds where it holds no value, compared in lower case.
MISSING = frozenset({"", "na", "n/a", "nan", "null", "none"})

#: The symbols a kept sequence may hold: the bases of DNA, or of RNA where the sequences hold U and
#: no T, with N for an unknown base.
ALPHABETS = {"DNA": "ACGTN", "RNA": "ACGUN"}

#: A column's values read as sequences when they are made of these letters alone.
SEQUENCE_TEXT = re.compile(r"[ACGTUN]+", re.IGNORECASE)

#: Ten folds, assigned seven, one and two to training, validation and the held-out rows, as the
#: example dataset's are. Assigning whole folds keeps the grouping and the balance together.
FOLDS, TRAIN_FOLDS, VAL_FOLDS = 10, 7, 1
SPLIT_SEED = 0
SPLIT_METHOD = (f"StratifiedGroupKFold over {FOLDS} folds; {TRAIN_FOLDS} train, {VAL_FOLDS} validation, "
                f"{FOLDS - TRAIN_FOLDS - VAL_FOLDS} test")

#: A continuous target is balanced across the folds in this many bins, each holding as many rows.
TARGET_BINS = 5

#: The builder a declaration names for a dataset the command line prepared.
BUILDER = "seqmaestro run (seqmaestro/cli/prepare.py)"

#: A dataset's name, which names its folder, its files and its runs.
NAME = re.compile(r"[a-z0-9][a-z0-9_]*")


class PrepareError(ValueError):
    """A table that cannot be prepared as the command line prepares one. The message says why."""


# ------------------------------------------------------------------------------ names and places


@dataclass(frozen=True)
class Paths:
    """Where a dataset's files go: under the repository, where the agent writes them, or under
    another root in a test."""

    name: str
    root: Path = REPO

    @property
    def data(self) -> Path:
        return self.root / "data" / self.name

    @property
    def splits(self) -> Path:
        return self.data / "splits"

    @property
    def declaration(self) -> Path:
        return self.root / "experiments" / "prepare" / "declarations" / f"{self.name}__splits.yaml"

    @property
    def description(self) -> Path:
        return self.root / "experiments" / "datasets" / f"{self.name}.json"

    @property
    def config(self) -> Path:
        return self.root / "experiments" / "configs" / f"{self.name}.yaml"

    def taken(self) -> list[Path]:
        """The dataset's files that exist already, which the command line does not replace."""
        return [path for path in (self.data, self.declaration, self.description, self.config)
                if path.exists()]

    def shown(self, path: Path) -> str:
        """A path as the files and the messages give it: relative to the root where it can be."""
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)


def name_problem(name: str) -> str | None:
    """What is wrong with a dataset's name, or None where nothing is."""
    if not NAME.fullmatch(name):
        return "a name is lower-case letters, digits and underscores, starting with a letter or a digit"
    return None


def name_for(table: Path) -> str:
    """A dataset's name made from its table's file name: `sites_60nt` for `Sites 60nt.csv`."""
    stem = table.name.lower().removesuffix(".gz")
    stem = stem[:-len(Path(stem).suffix)] if Path(stem).suffix else stem
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "dataset"


# ------------------------------------------------------------------------------ the table


def read_table(path: Path) -> pd.DataFrame:
    """The table as text: one column per field, each cell as written, none read as a number or as
    missing yet."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    extension = suffixes[-1] if suffixes else ""
    if extension not in SEPARATORS:
        raise PrepareError(f"{path.name}: give the table as a .csv or a .tsv file, which may be "
                           f"gzipped.")
    if not path.is_file():
        raise PrepareError(f"{path} is not a file.")
    try:
        frame = pd.read_csv(path, sep=SEPARATORS[extension], dtype=str, keep_default_na=False)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise PrepareError(f"{path.name} could not be read as a table: {error}") from error
    if len(frame.columns) < 2 or frame.empty:
        raise PrepareError(f"{path.name} needs a column of sequences, a column of what was measured, "
                           f"and a row for each sequence.")
    return frame


def is_missing(column: pd.Series) -> pd.Series:
    """Whether each cell holds no value."""
    return column.str.strip().str.lower().isin(MISSING)


def sequence_share(column: pd.Series) -> float:
    """The share of a column's values that read as sequences of bases, at least 8 long."""
    present = column[~is_missing(column)].str.strip()
    if present.empty:
        return 0.0
    return float((present.str.fullmatch(SEQUENCE_TEXT) & (present.str.len() >= 8)).mean())


def sequence_columns(frame: pd.DataFrame) -> list[str]:
    """The columns, those that read most as sequences first."""
    return sorted(frame.columns, key=lambda column: -sequence_share(frame[column]))


# ------------------------------------------------------------------------------ the target


@dataclass(frozen=True)
class TargetColumn:
    """What a target column holds: its kind and, for a binary target, its two values as the table
    writes them."""

    column: str
    kind: str
    #: A binary target's two values, in order; empty for a continuous one.
    values: tuple[str, ...]
    #: Whether every value is a number, as it is for a continuous target.
    numeric: bool
    #: The rows with a usable value, and those without one.
    usable: int
    unusable: int

    def value_of(self, text: str) -> str | None:
        """One of a binary target's values as written in the table, given as typed (`1.0` for
        `1`), or None where it is neither."""
        text = text.strip()
        if self.numeric:
            try:
                text = _number_text(float(text))
            except ValueError:
                return None
        return text if text in self.values else None


def _number_text(value: float) -> str:
    return f"{value:g}" if float(value).is_integer() else repr(float(value))


def _usable(column: pd.Series) -> tuple[pd.Series, pd.Series, bool]:
    """Which cells hold a usable value, the values as numbers where all of them are, and whether
    they are. A number must be finite to be usable."""
    text = column.str.strip()
    usable = ~is_missing(text)
    numbers = pd.to_numeric(text.where(usable), errors="coerce")
    numeric = bool(numbers[usable].notna().all()) and bool(usable.any())
    if numeric:
        usable &= np.isfinite(numbers.astype(float))
    return usable, numbers, numeric


def inspect_target(frame: pd.DataFrame, column: str) -> TargetColumn:
    """The kind of target a column holds: binary where it holds two values, continuous where it
    holds more and every one is a number."""
    usable, numbers, numeric = _usable(frame[column])
    count = int(usable.sum())
    if numeric:
        distinct = np.unique(numbers[usable].astype(float))
        values = tuple(_number_text(value) for value in distinct)
    else:
        values = tuple(sorted(frame.loc[usable, column].str.strip().unique()))
    if len(values) < 2:
        raise PrepareError(f"`{column}` holds {'one value' if values else 'no value'}, so there is "
                           f"nothing to predict.")
    if len(values) == 2:
        return TargetColumn(column, BINARY, values, numeric, count, len(frame) - count)
    if not numeric:
        raise PrepareError(f"`{column}` holds {len(values)} different values, such as "
                           f"{', '.join(map(repr, values[:3]))}, and not every one is a number. "
                           f"A target is two classes, or a number for each row.")
    return TargetColumn(column, CONTINUOUS, (), True, count, len(frame) - count)


# ------------------------------------------------------------------------------ the rows kept


@dataclass(frozen=True)
class Rows:
    """The rows of the dataset: the sequence column in upper case, the target as 1 and 0 or as a
    number, and the grouping columns, with a missing group value as NaN."""

    frame: pd.DataFrame
    sequence: str
    target: str
    groups: tuple[str, ...]
    kind: str
    #: DNA or RNA.
    alphabet: str
    #: The table's rows that have no usable target value, which belong to no class.
    without_target: int
    #: The declaration's record of the rows dropped for their sequence.
    step: dict[str, Any]
    #: A binary target's values as the table writes them: the positive class's, then the other's.
    values: tuple[str, str] | None = None

    @property
    def table_rows(self) -> int:
        """The rows of the table the dataset was made from."""
        before = self.step["rows_in"]
        return (sum(before.values()) if isinstance(before, dict) else before) + self.without_target

    @property
    def lengths(self) -> tuple[int, int]:
        lengths = self.frame[self.sequence].str.len()
        return int(lengths.min()), int(lengths.max())

    @property
    def symbols(self) -> str:
        """The symbols the sequences hold, in the alphabet's order."""
        held = set("".join(self.frame[self.sequence]))
        return "".join(letter for letter in ALPHABETS[self.alphabet] if letter in held)

    def counts(self, frame: pd.DataFrame | None = None) -> dict[int, int] | int:
        """The rows of each class (1 first) for a binary target, or the rows for a continuous one."""
        return _counts(self.frame if frame is None else frame, self.target, self.kind)


def _counts(frame: pd.DataFrame, target: str, kind: str) -> dict[int, int] | int:
    if kind == BINARY:
        return {label: int((frame[target] == label).sum()) for label in (1, 0)}
    return len(frame)


def spelled(letters: str) -> str:
    """Letters as a sentence lists them: `A, C, G, T and N`."""
    return letters if len(letters) < 2 else f"{', '.join(letters[:-1])} and {letters[-1]}"


def keep_rows(frame: pd.DataFrame, sequence: str, target: TargetColumn, positive: str | None,
              groups: list[str]) -> Rows:
    """The rows a dataset is made of: those with a usable target value, whose sequence is made of
    one alphabet's symbols. `positive` is the value of a binary target's positive class."""
    usable, numbers, _ = _usable(frame[target.column])
    rows = frame.loc[usable, [sequence, *groups]].copy()
    if target.kind == BINARY:
        if positive is None or target.value_of(positive) is None:
            raise PrepareError(f"The positive class must be one of {', '.join(target.values)}.")
        values = (_number_text_series(numbers[usable]) if target.numeric
                  else frame.loc[usable, target.column].str.strip())
        rows[target.column] = (values == target.value_of(positive)).astype(int)
    else:
        rows[target.column] = numbers[usable].astype(float)
    rows[sequence] = rows[sequence].str.strip().str.upper()
    for column in groups:
        rows[column] = rows[column].where(~is_missing(rows[column]))

    sequences = rows[sequence]
    alphabet = ("RNA" if sequences.str.contains("U").any() and not sequences.str.contains("T").any()
                else "DNA")
    letters = ALPHABETS[alphabet]
    good = sequences.str.fullmatch(f"[{letters}]+").fillna(False).astype(bool)
    step = {
        "step": f"rows whose sequence is missing or holds a symbol other than {spelled(letters)} "
                f"dropped",
        "why": "every featuriser reads a sequence as a string of one alphabet's symbols, and a row "
               "without a sequence has nothing to read. The sequences are otherwise used as "
               "delivered, in upper case",
        "rows_in": _counts(rows, target.column, target.kind),
        "rows_out": _counts(rows[good], target.column, target.kind),
    }
    kept = rows[good]
    if kept.empty:
        raise PrepareError(f"No row has both a value in `{target.column}` and a sequence of "
                           f"{spelled(letters)} in `{sequence}`.")
    values = None
    if target.kind == BINARY:
        chosen = target.value_of(positive)
        values = (chosen, next(value for value in target.values if value != chosen))
    return Rows(kept, sequence, target.column, tuple(groups), target.kind, alphabet,
                target.unusable, step, values)


def _number_text_series(numbers: pd.Series) -> pd.Series:
    return numbers.astype(float).map(_number_text)


def positional_window(rows: Rows, anchor: str) -> dict[str, Any] | None:
    """The window positional features read where the sequences differ in length: as long as the
    shortest, from `anchor` (`start` or `end`), so that no sequence is padded or dropped. None where
    every sequence has one length."""
    shortest, longest = rows.lengths
    if shortest == longest:
        return None
    return {"length": shortest, "anchor": anchor, "pad": False}


# ------------------------------------------------------------------------------ the split


#: Each split as the command line names it to the user.
SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Held out"}


def split_summary(rows: Rows, split: pd.Series) -> list[tuple[str, int, str, str]]:
    """Each split's name, its rows, and its positive and negative rows (binary) or the target's
    mean and median (continuous), as the command line shows the split."""
    summary = []
    for name in SPLITS:
        part = rows.frame.loc[split == name, rows.target]
        middle = ((f"{int((part == 1).sum()):,}", f"{int((part == 0).sum()):,}") if rows.kind == BINARY
                  else (f"{part.mean():.4g}", f"{part.median():.4g}"))
        summary.append((SPLIT_NAMES[name], len(part), *middle))
    return summary


def detection_floor(record: dict[str, Any]) -> str:
    """What the held-out rows can show to be above chance, from the dataset's record, in a
    sentence."""
    power = record["design_power"]["test"]
    if "smallest_detectable_auroc" in power:
        return (f"The held-out rows ({power['n_positive']:,} positive, {power['n_negative']:,} "
                f"negative) can show an AUROC of {power['smallest_detectable_auroc']} or more to be "
                f"above chance.")
    return (f"The held-out rows ({power['rows']:,}) can show a correlation of "
            f"{power['smallest_detectable_r']} or more (R² {power['smallest_detectable_r2']}) to be "
            f"above chance.")


def assign_splits(rows: Rows) -> pd.Series:
    """Each row's split (`train`, `val` or `test`): whole folds of `StratifiedGroupKFold`, grouped
    on the grouping columns and the sequence together, balanced on the class or on bins of a
    continuous target."""
    relations = [*rows.groups, rows.sequence]
    groups = connected_groups(rows.frame, relations)
    if groups.nunique() < FOLDS:
        raise PrepareError(f"The grouping leaves {groups.nunique()} groups of rows, and the split "
                           f"needs at least {FOLDS}, one for each fold.")
    target = rows.frame[rows.target]
    strata = (target if rows.kind == BINARY else
              pd.qcut(target.rank(method="first"), TARGET_BINS, labels=False))
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold = np.full(len(rows.frame), -1)
    with warnings.catch_warnings():
        # A class with fewer rows than folds draws a warning. The checks of the finished dataset
        # say whether every split still holds both classes, which is what matters.
        warnings.simplefilter("ignore", UserWarning)
        for number, (_, held) in enumerate(splitter.split(rows.frame, strata, groups)):
            fold[held] = number
    names = np.where(fold < TRAIN_FOLDS, "train",
                     np.where(fold < TRAIN_FOLDS + VAL_FOLDS, "val", "test"))
    return pd.Series(names, index=rows.frame.index)


# ------------------------------------------------------------------------------ the files


@dataclass(frozen=True)
class Described:
    """What the user and the language model said about the data, as the files record it: the task
    description's three fields, and each class's name and meaning, or what the number measures."""

    task: str
    sequence_layout: str
    target_name: str
    #: For a binary target: the name and the meaning of class 1 and of class 0.
    classes: dict[int, tuple[str, str]] | None
    #: For a continuous target: what the number measures.
    target_meaning: str | None

    def as_task(self) -> dict[str, str]:
        """The task description file's content (`seqmaestro.llm.TaskContext`)."""
        return {"task": self.task, "sequence_layout": self.sequence_layout,
                "target_name": self.target_name}


def declaration(table: Path, rows: Rows, window: dict[str, Any] | None,
                described: Described) -> dict[str, Any]:
    """What the command line decided, as a declaration states it (`docs/developer/dataset.md`)."""
    shortest, longest = rows.lengths
    if rows.kind == BINARY:
        target: dict[str, Any] = {"kind": BINARY}
        classes = {label: {"name": name, "meaning": meaning}
                   for label, (name, meaning) in described.classes.items()}
    else:
        target = {"kind": CONTINUOUS, "name": described.target_name,
                  "meaning": described.target_meaning}
        classes = None
    if window is None:
        layout = f"each {shortest} nucleotides"
        frame = "the command line chose no landmark, so position i is the i-th base of each sequence"
    else:
        layout = f"from {shortest} to {longest} nucleotides"
        frame = (f"the command line chose no landmark. Positional features use the "
                 f"{'first' if window['anchor'] == 'start' else 'last'} {window['length']} bases of "
                 f"each sequence, as many as the shortest has")
    declared = {
        "columns": {"sequence": rows.sequence, "target": rows.target, "weight": None},
        "target": target,
        **({"classes": classes} if classes else {}),
        "sequences": {
            "window": f"the sequences of {rows.alphabet} as delivered in {table.name}, {layout}",
            "anchor": frame,
            "orientation": "as delivered, in upper case. The command line does not check the "
                           "orientation, so a feature must not assume which end is upstream "
                           "unless the task description says so",
            "length": shortest if window is None else "variable",
            **({"positional_window": window} if window else {}),
        },
        "split": {
            "method": SPLIT_METHOD,
            "grouped_by": [*rows.groups, rows.sequence],
            "stratified_on": [rows.target if rows.kind == BINARY
                              else f"{TARGET_BINS} bins of {rows.target}"],
            "seed": SPLIT_SEED,
        },
        "filters": [rows.step],
        "sources": [{"path": f"raw/{table.name}", "sha256": file_digest(table)}],
        "builders": [BUILDER],
        "notes": f"Prepared by `seqmaestro run` from {table.name} on {date.today():%Y-%m-%d}, with "
                 f"its defaults (seqmaestro/cli/prepare.py).",
    }
    return declared


def capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]


def readme(table: Path, paths: Paths, rows: Rows, split: pd.Series, window: dict[str, Any] | None,
           described: Described) -> str:
    """The dataset's README, with the sections `docs/developer/dataset.md` lists for one. Each
    paragraph is one line, as markdown allows."""
    shortest, longest = rows.lengths
    sizes = " / ".join(f"{int((split == name).sum()):,}" for name in SPLITS)
    together = (f"the rows that share a value of {', '.join(f'`{column}`' for column in rows.groups)}, "
                f"and the rows whose sequences are identical" if rows.groups
                else "the rows whose sequences are identical")
    before, after = rows.step["rows_in"], rows.step["rows_out"]
    step = capitalised(rows.step["step"])
    if rows.kind == BINARY:
        (name_1, meaning_1), (name_0, meaning_0) = described.classes[1], described.classes[0]
        positive, negative = rows.values
        target = f"`{rows.target}` is 1 for {name_1} ({meaning_1}) and 0 for {name_0} ({meaning_0})."
        if (positive, negative) != ("1", "0"):
            target += f" The table writes them `{positive}` and `{negative}`."
        steps = [f"| Step | {name_1} | {name_0} |", "|---|---:|---:|",
                 f"| With a value in `{rows.target}` | {before[1]:,} | {before[0]:,} |",
                 f"| {step} | {after[1]:,} | {after[0]:,} |"]
    else:
        target = f"`{rows.target}` is {described.target_name}: {described.target_meaning}."
        steps = ["| Step | Rows |", "|---|---:|",
                 f"| With a value in `{rows.target}` | {before:,} |", f"| {step} | {after:,} |"]
    lengths = (f"{shortest} nt" if window is None else
               f"{shortest} to {longest} nt, of which positional features read the "
               f"{'first' if window['anchor'] == 'start' else 'last'} {window['length']}")
    paragraphs = [
        f"# {paths.name}",
        f"Prepared by `seqmaestro run` from `{table.name}`, with the command line's defaults "
        f"(`seqmaestro/cli/prepare.py`). An agent's preparation would decide each of them with you.",
        "\n".join([
            "| | |", "|---|---|",
            f"| Rows | {len(rows.frame):,} |",
            f"| Sequence | {lengths}, upper case {spelled(rows.symbols)} |",
            f"| Split | {sizes} rows in training, validation and test |",
            f"| Declaration | `{paths.shown(paths.declaration)}` |",
            f"| Description | `{paths.shown(paths.description)}` |"]),
        "## The source",
        f"`raw/{table.name}` is the table as it was given, with {rows.table_rows:,} rows. The "
        f"sequences are in `{rows.sequence}`, and the target in `{rows.target}`.",
        "## The target",
        target,
        "## The sequences",
        "The sequences are used as delivered, in upper case, with no window cut around a landmark. "
        "Their orientation was not checked.",
        "## The preparation steps and the rows each removed",
        "\n".join(steps),
        *([f"{rows.without_target:,} rows of the table have no usable value in `{rows.target}`, and "
           f"are left out before this step."] if rows.without_target else []),
        "## The grouping",
        f"The split keeps together {together}, so that no held-out sequence is also a training "
        f"sequence. It assigns whole folds of `StratifiedGroupKFold` ({FOLDS} folds: {TRAIN_FOLDS} "
        f"training, {VAL_FOLDS} validation, {FOLDS - TRAIN_FOLDS - VAL_FOLDS} test, seed "
        f"{SPLIT_SEED}), balanced on "
        f"{'the class' if rows.kind == BINARY else f'{TARGET_BINS} bins of the target'}, and "
        f"`splits/{GROUP_FILE}` records the group of each sequence.",
        "## What the language models are told",
        f"`{paths.shown(paths.description)}`:",
        f"> {described.task}\n>\n> {described.sequence_layout}",
        "Nothing is withheld from them, so the run is not blinded.",
    ]
    return "\n\n".join(paragraphs) + "\n"


def write(table: Path, paths: Paths, rows: Rows, split: pd.Series, window: dict[str, Any] | None,
          described: Described) -> dict[str, Any]:
    """Write the dataset's files and finalise it. Returns the record `finalise` wrote."""
    taken = paths.taken()
    if taken:
        raise PrepareError(f"{', '.join(paths.shown(path) for path in taken)} exist already.")
    (paths.data / "raw").mkdir(parents=True)
    shutil.copy2(table, paths.data / "raw" / table.name)
    paths.splits.mkdir()
    for name in SPLITS:
        rows.frame.loc[split == name, [rows.sequence, rows.target]].to_csv(
            paths.splits / f"{name}.csv", index=False)
    group_map(rows.frame, rows.sequence, [*rows.groups, rows.sequence]).to_csv(
        paths.splits / GROUP_FILE, index=False)
    files.write_text(paths.data / "README.md", readme(table, paths, rows, split, window, described))

    declared = declaration(table, rows, window, described)
    paths.declaration.parent.mkdir(parents=True, exist_ok=True)
    files.write_text(paths.declaration,
                     f"# What `seqmaestro run` decided when it prepared {paths.shown(paths.splits)}.\n"
                     f"# finalise_dataset.py counts and checks everything else. Fields: "
                     f"docs/developer/dataset.md.\n\n"
                     + yaml.safe_dump(declared, sort_keys=False, allow_unicode=True, width=100))
    paths.description.parent.mkdir(parents=True, exist_ok=True)
    files.write_json(paths.description, described.as_task())
    return finalise(paths.splits, declared, provenance={
        "finalised_by": BUILDER,
        "declaration": paths.shown(paths.declaration),
        "git_sha": git_sha(REPO),
        "git_dirty": git_is_dirty(REPO),
    })
