"""What a featuriser is, and what it produces.

`docs/feature_record.md` documents the record below field by field. It is maintained alongside this
module, so a change to one needs a change to the other.

Four ideas shape the module.

**A feature is a record, not a column.** The write-up specifies a library entry as description,
reasoning, code and evidence, and a language-model proposer emits that shape. The
fixed-vocabulary featurisers emit the same shape, so *k*-mers, model-proposed features and later
additions such as PWM scans all go into one library with no adapter between them.

**A featuriser reads rows, and a row is more than its sequence.** A dataset may declare covariates:
numbers measured beside each sequence, such as a chromatin signal. `Inputs` carries the sequences
and the covariates together, and every featuriser takes one. A featuriser asks for
`inputs.sequences` or `inputs.covariates` and never for a column by name, so which columns of a
split file are which is decided in one place (`seqmaestro.dataset.Columns`). A bare sequence column
is accepted too, as rows with no covariates.

**Nothing is assumed about the data.** Sequence length, alphabet and vocabulary are learned at `fit`
and recorded, and a count's dtype is derived from a bound on what the input can hold
(`count_dtype`). Datasets differ in width and alphabet, and an implementation that hard-codes them
either breaks or silently truncates on the next dataset. Where a fixed property is genuinely
required (positional features need equal lengths or a declared window), it is checked, and data
without it are refused.

**`fit` is separate from `transform` even where it need not be.** Exact *k*-mers have a vocabulary
derivable from the data alone, so the separation hardly matters for them. STREME motifs are
*discovered* and PWM thresholds are *calibrated*, both from data, and if either is derived from
anything but the training split, the experiment leaks. With the separation built into the
interface, such featurisers fit in without changing it.
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

# Whose method produced a feature. The values are a convention and are not enumerated here: a
# vendored method uses its own name (`"deft"`), so features pooled from several runs stay
# attributable without this module knowing about every method. SeqMaestro's own name is below.
ORIGIN_SEQMAESTRO = "seqmaestro"

#: How a feature entered an arm's answer. Compare `kind`, which is what the feature computes.
#:
#: `selected`: the arm took it from the input pool it was handed. `generated`: the arm constructed
#: it while fitting, so it was not available to any other arm drawing on the same pool.
#:
#: **Declared by whoever produced the record, never inferred.** An extensional test ("is this
#: function inside the pool's span") is undecidable in general and ill-defined for nonlinear
#: features, so the field records what the producer did.
#:
#: It is relative to the *arm*. A language-model pool handed to a model is `selected` there,
#: because that arm only chose from it, while the same feature constructed mid-fit by an iterative
#: arm is `generated`. Two arms can report the same feature under different provenance and both be
#: right.
PROVENANCE_SELECTED = "selected"
PROVENANCE_GENERATED = "generated"
PROVENANCES = (PROVENANCE_SELECTED, PROVENANCE_GENERATED)


@dataclass(frozen=True)
class FeatureRecord:
    """One feature, however it was arrived at. The whole system uses this record type.

    A *k*-mer count, a positional indicator, a PWM scan and a function a language model wrote all
    take this shape, so the stability estimator, the judge and the results table consume one format
    and need not know which arm produced a given row.

    `kind` and `params` are the extension point. Adding gapped *k*-mers, mismatch *k*-mers, STREME
    motifs or CentriMo-style positional concentration means adding a `Featuriser` that emits a new
    `kind` with its own `params`, with no change here or downstream:

        gapped_kmer   {"pattern": "GG.TCA", "k": 5, "n_gaps": 1}
        pwm_scan      {"motif_id": "MA0139.1", "db": "JASPAR", "mode": "max"}

    The optional fields fill in as a feature acquires history: `source` and `rationale` when a
    model proposes it, `importance` and `direction` when a fitted model says how much it used it and
    which way, and `marginal_effect` when its cell is scored.

    **`kind` and `origin` are independent, and both are needed.** `kind` is what the feature
    computes, and `origin` is whose method produced it (`"deft"`, `"seqmaestro"`). They differ where
    the research question lies. An `llm_generated` feature proposed by DEFT and one proposed by our
    own shared proposer have the same kind and different origins, and once features from several
    runs pool into one library, `origin` is the only field that tells them apart. `kind` in turn
    separates DEFT's LLM candidates from the naive positional ones it seeds every node with, which
    is what DEFT's `llm_split_share` (in its provenance record) counts.

    `extra` carries whatever a method knows and others do not, such as a tree depth, a split
    threshold or the path from the root. It has no schema, because most methods have no depth or
    threshold to record.

    The record is flat and plain-typed because the arm that runs in a vendored method's own conda
    environment cannot import this module, and has to write conforming JSON by hand (see
    `seqmaestro.harness.features`).
    """

    name: str
    description: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    rationale: str | None = None
    origin: str | None = None
    importance: float | None = None
    #: How far the arm's headline metric falls when this feature's column is shuffled on the
    #: validation rows: AUROC for a binary target, R-squared for a continuous one. It is computed
    #: the same way for every arm, so unlike `importance` it means the same thing across arms (see
    #: `seqmaestro.harness.importance`). `None` where it was not computed.
    permutation_importance: float | None = None
    #: Which class the fitted model uses this feature to argue for, or which way it moves a
    #: continuous target, in words every arm shares (see `seqmaestro.harness.direction`). Each model
    #: answers from different machinery, so the file names the basis once, in `direction_kind`, as
    #: it does for `importance`. `None` where the model has no basis for a direction, or the feature
    #: does not move the model.
    direction: str | None = None
    #: How far the feature goes with the target *on its own*, over every training row against the
    #: real target: the rank-biserial correlation (`2 * AUC - 1`) for a binary target, positive
    #: meaning higher in the positive class, and Spearman's rho for a continuous one, positive
    #: meaning higher values go with a higher target. It is computed the same way for every arm and
    #: needs no model, so unlike `direction` it describes the data and not what was fitted to it. A
    #: disagreement between the two is a finding (see `seqmaestro.harness.effect`).
    marginal_effect: float | None = None
    #: `selected` or `generated` (see `PROVENANCES`). `None` where the producer did not declare it.
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
    """The rows a featuriser reads: each row's sequence, and the covariates measured beside it.

    `seqmaestro.dataset.Columns.inputs` builds one from a split frame, using the columns
    `dataset.json` records, so no stage picks the sequence or covariate columns out of a frame by
    itself. `from_sequences` builds rows with no covariates, which is what a featuriser handed a
    bare sequence column reads.

    Both parts are checked once, here. The sequences must be strings, none missing. The covariates
    must be numbers, all finite, under names that are legal Python identifiers, because a generated
    function indexes a covariate by name and a covariate's feature is named after it. They are held
    as float64, in the order given, which is the order the dataset declares. A boolean becomes 0
    or 1.

    No column name is held for the sequences. A featuriser that reads `sequences` needs none. One
    whose code indexes a column by name knows the name that code was written against, and builds
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

        A featuriser that reads covariates records at `fit` which ones it read, and selects them
        by name here when it transforms. So a fitted featuriser can be handed every covariate the
        dataset declares, and rows that lack one it read are refused instead of misread.
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
        their own names, indexed from 0. This is what a generated function is handed as `X`.

        Built in one step and copied, which holds the covariates' columns together in memory. A
        frame read from a CSV holds each column separately, and adding a column to one with more
        than a hundred covariates makes pandas warn, on every call, that the frame is fragmented.
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

        `y` is the training labels, and most featurisers ignore it: a *k*-mer vocabulary and a
        positional one-hot layout are properties of the sequences alone. It exists for featurisers
        that cannot work without it. A language-model proposer shown no labels would propose the same
        features for a permuted-label arm as for a real one, which would leave the permuted runs
        with no noise-derived features and the judge with no negative controls.

        `sample_weight` is the weight each training row is fitted with, where the design has
        weights. Only a featuriser that fits a model while it builds its columns uses it (the
        iterative source), so that the fit it reports on is weighted as the arm's model is.
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
        # Read before `_transform`, so an unfitted featuriser fails here with a clear message.
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
    """Several featurisers side by side, their columns concatenated.

    Composition is why a pool of several members needs no implementation of its own:
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
    """A written feature under a name no earlier column of its pool has.

    A feature that is code is named by whoever wrote it, and a language model shown the pool's
    columns can name a function after one of them, such as `kmer_CG`. The two compute different
    things, so both are kept: the written one takes the suffix `_written`, numbered where that is
    taken too, and its record keeps the name it was proposed under in `extra["name_as_proposed"]`.
    Two enumerated columns with one name are a different case, a conflict between recipes, and
    `Featuriser.fit` refuses them.
    """
    name, count = f"{record.name}_{WRITTEN_SUFFIX}", 1
    while name in taken:
        count += 1
        name = f"{record.name}_{WRITTEN_SUFFIX}_{count}"
    return replace(record, name=name, extra={**record.extra, "name_as_proposed": record.name})



def as_inputs(inputs) -> Inputs:
    """`inputs` as `Inputs`: an `Inputs` as it is, and anything else as the sequences alone, rows
    with no covariates."""
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
    """The symbols actually present, sorted. Taken from the data rather than assumed to be ACGT."""
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
    """Refuse any symbol that cannot be part of a column name, since column names are identifiers.

    Such a symbol is refused and not transliterated. The right replacement depends on what the
    symbol means in the data (a gap, an ambiguity code, a mask), and one invented here would be
    arbitrary and invisible.
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
