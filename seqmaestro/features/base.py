"""What a featuriser is and what it produces.

`FeatureRecord` describes each field of the record below, and `docs/developer/feature_record.md`
says which producer fills each field, how the records reach a run and how a field is added. The
page is maintained alongside this module, so a change to one needs a change to the other.

Four ideas shape the module.

**A feature is a record, not a column.** A record holds what the feature computes, the reasoning
behind it, its code and the evidence for it, which is the shape in which the proposer returns the
features a language model writes. The fixed-vocabulary featurisers produce the same shape, so
*k*-mers, written features and motif scans all go into one library with no adapter between them.

**The rows a featuriser is given hold more than the sequences.** A dataset may declare covariates:
numbers measured beside each sequence, such as a chromatin signal. `Inputs` holds the sequences and
the covariates together, and every featuriser takes an `Inputs`. A featuriser asks for
`inputs.sequences` or `inputs.covariates`, never for a column by name, so which columns of a split
file are which is decided in one place (`seqmaestro.dataset.Columns`). A bare sequence column is
accepted too, as rows with no covariates.

**Nothing is assumed about the data.** Sequence length, alphabet and vocabulary are learned at `fit`
and recorded, and the dtype of a count is derived from a bound on the largest count the input allows
(`count_dtype`). Datasets differ in sequence length and alphabet, so an implementation that
hard-codes them breaks, or silently truncates, on the next dataset. Where a fixed property is
required (positional features need equal lengths or a declared window), it is checked, and data
without it are refused.

**`fit` is separate from `transform`, even for featurisers that do not need it.** The vocabulary of
exact *k*-mers follows from the sequences alone, so the separation hardly matters for them.
Discovered motifs, by contrast, are found with the labels, and if they were found in any rows
besides those the model fits on, the evaluation would leak. Because the interface separates the two
steps for every featuriser, one that learns from the labels needs no change to it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Whose method produced a feature (`FeatureRecord.origin`). The values are a convention and are not
# enumerated here: a vendored method uses its name (`"deft"`), so features pooled from several runs
# stay attributable without this module knowing every method. The value for SeqMaestro is below.
ORIGIN_SEQMAESTRO = "seqmaestro"

#: The two values of `FeatureRecord.provenance`, whose docstring says what each means.
PROVENANCE_SELECTED = "selected"
PROVENANCE_GENERATED = "generated"
PROVENANCES = (PROVENANCE_SELECTED, PROVENANCE_GENERATED)


@dataclass(frozen=True)
class FeatureRecord:
    """The record of one feature: the type every arm produces and every analysis uses.

    A *k*-mer count, a positional indicator, a motif scan and a function a language model wrote all
    take this shape, so the stability estimator, the judge and the results table use one format and
    need not know which arm produced a record. Only `name`, `description` and `kind` are required.
    The optional fields are filled in at different steps: `source` and `rationale` when a language
    model proposes the feature, `importance` and `direction` when a fitted model says how much it
    used the feature and which way, and `marginal_effect` when its cell is scored.

    - `name`: a legal Python identifier, which `__post_init__` checks. A language model copies the
      naming conventions it is shown into the function names it writes, so a name such as
      `upstream_-15nt_GC` would lead to code that does not compile.
    - `description`: what the feature computes, in words. For a written feature it is the language
      model's own description, which the judge and the recurrence analysis are shown.
    - `kind`: what the feature computes, as a category such as `kmer`, `pwm_scan` or
      `llm_generated`, which the analyses filter on. It also separates the candidates DEFT's
      language model writes from the naive positional ones DEFT adds at every node, which DEFT's
      `llm_split_share` counts.
    - `params`: the recipe of a feature that is not code, such as `{"kmer": "GGT", "k": 3}`. It is
      empty for written code.
    - `source`: the function's code, for a feature that is code.
    - `rationale`: for a feature a language model proposed, the model's stated reason why it might
      predict the target. The judge is shown it.
    - `origin`: whose method produced the feature: `seqmaestro` (`ORIGIN_SEQMAESTRO`), or the name
      of a vendored method, such as `deft` or `igsr`.
    - `importance`: the fitted model's own measure of how much it used the feature. Its unit
      differs by model and is named once per file (`importance_kind`), so importances are compared
      between arms only by rank (`seqmaestro.harness.features.rank_by_importance`).
    - `permutation_importance`: how far the arm's headline metric falls when the feature's column
      is shuffled on the validation rows: AUROC for a binary target, R-squared for a continuous one.
      It is computed the same way for every arm, so unlike `importance` it means the same thing
      across arms (`seqmaestro.harness.importance`). `None` where it was not computed.
    - `direction`: which class the fitted model uses the feature to argue for, or which way it
      moves a continuous target, in words every arm shares (`seqmaestro.harness.direction`). Each
      model arrives at a direction by a different method, so the file names the basis once, in
      `direction_kind`. `None` where the model has no basis for a direction, or where the feature
      does not change its predictions.
    - `marginal_effect`: how strongly the feature is associated with the target on its own, over
      every training row against the real target: the rank-biserial correlation (`2 * AUC - 1`)
      for a binary target, positive where higher values are more common in the positive class, and
      Spearman's rho for a continuous one, positive where higher values go with a higher target. It
      needs no model, so it describes the data, whereas `direction` describes what was fitted to
      it. A disagreement between the two is a finding (`seqmaestro.harness.effect`).
    - `provenance`: how the arm came to have the feature. `selected` means that the arm took it
      from the pool it was handed, and `generated` that the arm constructed it while fitting, so
      that no other arm on the same pool had it. Provenance is relative to the arm: a written
      feature handed to a model in a pool is `selected` for that arm, because the arm only chose
      it, whereas the same feature written during the fit of an arm on the iterative source is
      `generated`. Whoever produces the record declares it, because inferring it would need a test
      of whether a function lies in the span of the pool's columns, which is undecidable in general
      and ill-defined for nonlinear features. `None` where the producer did not declare it.
    - `extra`: what one method knows and the others do not, such as the nodes a DEFT feature splits
      at or a discovered motif's matrix. It has no schema.

    `docs/developer/feature_record.md` says which producer fills each field, how the records reach
    a run, and how a field is added.
    """

    name: str
    description: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    rationale: str | None = None
    origin: str | None = None
    importance: float | None = None
    permutation_importance: float | None = None
    direction: str | None = None
    marginal_effect: float | None = None
    provenance: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not IDENTIFIER.match(self.name):
            raise ValueError(
                f"Feature name {self.name!r} is not a legal Python identifier. A model copies the "
                f"naming conventions it is shown into the function names it writes, so an illegal "
                f"name here produces code that will not compile downstream."
            )


@dataclass(frozen=True, eq=False)
class Inputs:
    """The rows a featuriser takes: each row's sequence and the covariates measured beside it.

    `seqmaestro.dataset.Columns.inputs` builds one from a split frame, using the columns
    `dataset.json` records, so a stage never picks the sequence or covariate columns out of a frame
    itself. `from_sequences` builds rows with no covariates, which is what a featuriser makes of a
    bare sequence column.

    Both parts are checked once, here. Every sequence must be present and be a string. The
    covariates must be finite numbers under names that are legal Python identifiers, because a
    generated function indexes a covariate by name and a covariate's feature is named after it. They
    are held as float64 (a boolean becomes 0 or 1), in the order given, which is the order the
    dataset declares.

    No column name is held for the sequences, because a featuriser that uses `sequences` needs none.
    One whose code indexes a column by name knows the name that code was written against, and builds
    the frame it hands the code with `frame`.
    """

    sequences: pd.Series
    covariates: pd.DataFrame

    def __post_init__(self) -> None:
        sequences = as_sequences(self.sequences)
        covariates = self.covariates
        if not isinstance(covariates, pd.DataFrame):
            raise TypeError(f"The covariates must be a DataFrame, not {type(covariates).__name__}.")
        if not covariates.index.equals(sequences.index):
            raise ValueError("The covariates and the sequences must share one index, row for row.")
        names = list(covariates.columns)
        illegal = [name for name in names if not isinstance(name, str) or not IDENTIFIER.match(name)]
        if illegal:
            raise ValueError(
                f"Covariate name(s) {illegal} are not legal Python identifiers. Generated code indexes "
                f"a covariate by its name and a feature is named after it, so rename the column at "
                f"Stage 1.")
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Covariate(s) {repeated} are named more than once.")
        non_numeric = [name for name in names if not is_numeric_dtype(covariates[name])]
        if non_numeric:
            raise TypeError(
                f"Covariate(s) {non_numeric} are not numeric. A categorical covariate is encoded as "
                f"numbers at Stage 1.")
        values = covariates.astype(np.float64)
        undefined = [name for name in names if not np.isfinite(values[name].to_numpy()).all()]
        if undefined:
            raise ValueError(
                f"Covariate(s) {undefined} hold missing or infinite values. Stage 1 refuses these "
                f"(the `covariates_valid` check), and any imputation is decided there.")
        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "covariates", values)

    @classmethod
    def from_sequences(cls, sequences) -> Inputs:
        """Rows with no covariates, from a Series, a list or an array of sequences."""
        sequences = as_sequences(sequences)
        return cls(sequences, pd.DataFrame(index=sequences.index))

    @property
    def covariate_columns(self) -> tuple[str, ...]:
        return tuple(self.covariates.columns)

    def __len__(self) -> int:
        return len(self.sequences)

    def take(self, positions) -> Inputs:
        """These rows, by position: a slice or a list of positions, as `DataFrame.iloc` takes them."""
        return Inputs(self.sequences.iloc[positions], self.covariates.iloc[positions])

    def select(self, covariates) -> Inputs:
        """These rows with only the named covariates, in the order named.

        A featuriser that uses covariates records at `fit` which ones it used, and selects them by
        name here when it transforms. A fitted featuriser can therefore be handed every covariate
        the dataset declares, and rows that lack one it used are refused instead of featurised
        wrongly.
        """
        covariates = list(covariates)
        missing = [name for name in covariates if name not in self.covariates.columns]
        if missing:
            raise ValueError(
                f"These rows have no covariate {', '.join(missing)}, which the featuriser was fitted "
                f"on; they have {', '.join(self.covariate_columns) or 'none'}.")
        return Inputs(self.sequences, self.covariates[covariates])

    def frame(self, sequence_column: str) -> pd.DataFrame:
        """The rows as one frame: the sequences under `sequence_column`, then the covariates under
        their names, indexed from 0. This is what a generated function is handed as `X`.

        The frame is built in one step and copied, so the covariates' columns are held together in
        memory. A frame read from a CSV holds each column separately, and adding a column to such a
        frame with more than a hundred covariates makes pandas warn, on every call, that the frame
        is fragmented.
        """
        if sequence_column in self.covariates.columns:
            raise ValueError(f"{sequence_column!r} names a covariate as well as the sequences.")
        sequences = pd.DataFrame({sequence_column: self.sequences.to_numpy()})
        return pd.concat([sequences, self.covariates.reset_index(drop=True)], axis=1).copy()


class Featuriser(ABC):
    """Turns rows into a named numeric matrix. A row is a sequence and, where the dataset declares
    them, its covariates (`Inputs`).

    Subclasses implement `_fit` and `_transform`. The base class owns the fitted/unfitted
    bookkeeping, the input validation, and composition via `+`.
    """

    def __init__(self) -> None:
        self._records: list[FeatureRecord] | None = None

    # ------------------------------------------------------------------ subclass contract

    @abstractmethod
    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        """Learn from the training rows and return this featuriser's feature records, in column order.

        `y` is the training labels, which most featurisers ignore, because a *k*-mer vocabulary and
        a positional one-hot layout are properties of the sequences alone. It exists for featurisers
        that cannot work without it. A language-model proposer shown no labels would propose the
        same features for a permuted cell as for a real one, so the permuted cells would have no
        features fitted to noise, and the judge would have no negative controls.

        `sample_weight` is the weight each training row is fitted with, where the design has
        weights. Only a featuriser that fits a model while it builds its columns uses it (the
        iterative source), so that the fit it shows the language model is weighted as the arm's
        model is.
        """

    @abstractmethod
    def _transform(self, inputs: Inputs) -> np.ndarray:
        """Produce the value matrix, columns in the same order as the records from `_fit`."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Configuration and fitted state, which a cell records under `featuriser`, so a result says
        how its features were computed."""

    # ------------------------------------------------------------------------------ public

    @property
    def records(self) -> list[FeatureRecord]:
        if self._records is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first.")
        return self._records

    @property
    def feature_names(self) -> list[str]:
        return [record.name for record in self.records]

    @property
    def is_fitted(self) -> bool:
        return self._records is not None

    def fit(self, inputs, y=None, sample_weight=None) -> Featuriser:
        """Learn from the training rows: an `Inputs`, or the sequences alone, taken as rows with no
        covariates. `y` and `sample_weight` are optional, and featurisers that have no use for
        them ignore them (`_fit` explains which need them)."""
        records = self._fit(as_inputs(inputs), y, sample_weight)

        counted = Counter(record.name for record in records)
        duplicates = {name for name, count in counted.items() if count > 1}
        if duplicates:
            raise ValueError(f"{type(self).__name__} produced duplicate feature names: {sorted(duplicates)}")

        self._records = records
        return self

    def transform(self, inputs) -> pd.DataFrame:
        """The feature matrix for these rows, indexed as they are. Takes what `fit` takes."""
        # Fetched before `_transform`, so an unfitted featuriser fails here with a clear message.
        # Otherwise an unfitted subclass reaches for fitted state and fails on whatever happens to
        # be None first, which says nothing about the real mistake.
        records = self.records
        inputs = as_inputs(inputs)
        values = self._transform(inputs)

        if values.shape != (len(inputs), len(records)):
            raise ValueError(
                f"{type(self).__name__} produced {values.shape} for {len(inputs)} rows and "
                f"{len(records)} features; the matrix and the records must agree."
            )
        return pd.DataFrame(values, columns=[record.name for record in records],
                            index=inputs.sequences.index)

    def fit_transform(self, inputs, y=None, sample_weight=None) -> pd.DataFrame:
        inputs = as_inputs(inputs)
        return self.fit(inputs, y, sample_weight).transform(inputs)

    def __add__(self, other: Featuriser) -> ComposedFeaturiser:
        return ComposedFeaturiser([self, other])


class ComposedFeaturiser(Featuriser):
    """Several featurisers whose columns are concatenated side by side.

    Because featurisers compose, a pool of several sources needs no featuriser written for it:
    `positional+kmer` is the composition of `positional` and `kmer`.
    """

    def __init__(self, parts: list[Featuriser]) -> None:
        super().__init__()
        flattened: list[Featuriser] = []
        for part in parts:
            # Keep `a + b + c` flat rather than nesting, so `describe()` stays readable.
            flattened.extend(part.parts if isinstance(part, ComposedFeaturiser) else [part])
        self.parts = flattened

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        records: list[FeatureRecord] = []
        taken: set[str] = set()
        for part in self.parts:
            part.fit(inputs, y, sample_weight)
            for record in part.records:
                if record.name in taken and record.source:
                    record = _under_a_free_name(record, taken)
                records.append(record)
                taken.add(record.name)
        return records

    def _transform(self, inputs: Inputs) -> np.ndarray:
        blocks = [part.transform(inputs).to_numpy() for part in self.parts]
        # Concatenation forces one dtype across blocks. Take the one that holds every block
        # losslessly, so numpy's default does not silently widen everything to float64.
        dtype = np.result_type(*(block.dtype for block in blocks))
        return np.concatenate([block.astype(dtype, copy=False) for block in blocks], axis=1)

    def describe(self) -> dict[str, Any]:
        return {"kind": "composed", "parts": [part.describe() for part in self.parts]}


# ----------------------------------------------------------------------------------- helpers

#: What is added to the name of a written feature when an earlier column already has that name.
WRITTEN_SUFFIX = "written"


def _under_a_free_name(record: FeatureRecord, taken: set[str]) -> FeatureRecord:
    """A written feature renamed so that its name differs from every earlier column of its pool.

    A feature that is code is named by whoever wrote it, and a language model shown the pool's
    columns can name a function after one of them, such as `kmer_CG`. The two compute different
    things, so both are kept: the written one takes the suffix `_written`, numbered where that is
    taken too, and its record keeps the name it was proposed under in `extra["name_as_proposed"]`.
    Two enumerated columns with one name are a different case, a conflict between recipes, which
    `Featuriser.fit` refuses.
    """
    name, count = f"{record.name}_{WRITTEN_SUFFIX}", 1
    while name in taken:
        count += 1
        name = f"{record.name}_{WRITTEN_SUFFIX}_{count}"
    return replace(record, name=name, extra={**record.extra, "name_as_proposed": record.name})



def as_inputs(inputs) -> Inputs:
    """`inputs` as `Inputs`: an `Inputs` unchanged, and anything else as sequences alone (rows with
    no covariates)."""
    return inputs if isinstance(inputs, Inputs) else Inputs.from_sequences(inputs)


def as_sequences(sequences) -> pd.Series:
    """Accept a Series, a list or an array, and refuse anything that is not usable sequence data."""
    if isinstance(sequences, pd.DataFrame):
        raise TypeError(
            "Pass `Inputs` or the sequence column itself, not the frame. Which columns of a frame "
            "are the sequences and the covariates is recorded in dataset.json and read by "
            "`seqmaestro.dataset.Columns.inputs`, not guessed at here."
        )
    if not isinstance(sequences, pd.Series):
        sequences = pd.Series(list(sequences))

    if len(sequences) == 0:
        raise ValueError("No sequences given.")
    if sequences.isna().any():
        missing = sequences.index[sequences.isna()].tolist()
        raise ValueError(f"{len(missing)} sequence(s) are missing, first at index {missing[0]}.")
    if not all(isinstance(value, str) for value in sequences):
        offender = next(value for value in sequences if not isinstance(value, str))
        raise TypeError(f"Sequences must be strings; found {type(offender).__name__}.")

    return sequences


def alphabet_of(sequences: pd.Series) -> list[str]:
    """The symbols present in the sequences, sorted. They are taken from the data instead of
    assumed to be ACGT."""
    return sorted(set("".join(sequences)))


def count_dtype(maximum_possible_count: int) -> np.dtype:
    """The narrowest unsigned integer type that can hold `maximum_possible_count`.

    It is derived from the theoretical bound and not the observed maximum, so `transform` on a
    validation split cannot overflow a dtype chosen on training data. Counts are small and the
    matrices are wide, so a narrow dtype saves up to an order of magnitude in memory, which matters
    most to the boosting arm. The bound depends on the input, so it is computed and never assumed.
    """
    if maximum_possible_count < 0:
        raise ValueError(f"A count bound cannot be negative; got {maximum_possible_count}.")
    return np.dtype(np.min_scalar_type(int(maximum_possible_count)))


def check_symbols_are_nameable(alphabet: list[str]) -> None:
    """Refuse any symbol that cannot be part of a column name, because column names are Python
    identifiers.

    Such a symbol is refused instead of transliterated, because the right replacement depends on
    what the symbol means in the data (a gap, an ambiguity code, a mask), and one invented here
    would be arbitrary and invisible.
    """
    illegal = [symbol for symbol in alphabet if not IDENTIFIER.match(f"x{symbol}")]
    if illegal:
        raise ValueError(
            f"Symbol(s) {illegal} cannot appear in a Python identifier, so they cannot be part of a "
            f"column name. Re-encode them upstream, because what they should become depends on what they "
            f"mean in your data."
        )


def check_symbols(sequences: pd.Series, known: set[str], owner: str) -> None:
    """Refuse symbols unseen at fit time, rather than encoding them as silent zeros."""
    unknown = sorted(set("".join(sequences)) - known)
    if unknown:
        raise ValueError(
            f"{owner} met symbol(s) {unknown} that were absent when it was fitted. They would be "
            f"encoded as all-zero, which is indistinguishable from a real absence. Either refit on "
            f"data containing them, or clean them upstream."
        )
