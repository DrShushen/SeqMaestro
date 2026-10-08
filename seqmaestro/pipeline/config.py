"""What one run is: the dataset it binds to, the grid it sweeps, and the arms it runs.

A full experiment is a dozen settings that have to agree across the stages. Passed loose, as
environment variables or as flags retyped for each stage, they cause three problems. They are
invisible unless you read the driver. One left unset takes a default that lives somewhere else
entirely. And nothing records what a finished run was configured with, so a resumed stage cannot
tell a finished cell from one computed under different settings.

A config file solves all three by being the single statement of a run. The grid and the arms that
every dataset shares are in git, in the configs a dataset's config extends, and the config is
written, resolved, into the run directory when the run starts, so the run describes itself
afterwards.

**Nothing here defaults a fact about the data.** The `dataset` section names the prepared split and
the description the language-model consumers read, and both are required. A config need not state
the columns. They come from the split's `dataset.json`, which Stage 1 writes, and a config that
states one anyway, as the config a run records does, must agree with that file, so a config cannot
disagree with the data it names. A default is one dataset's answer applied silently to another's,
and for the description that means a model is told the wrong thing and answers confidently anyway.

**An unknown key is refused**, as in `experiments/datasets/`, because a misspelled key is a setting
silently missing from a run that completes without it. A model or a pool that does not exist is
refused too (below), so a misspelled one stops the run before it starts.

**A config may extend another**, so the grid, the arms and the settings that every dataset shares
are written once and each dataset states only its own. Merging is recursive, a list replaces the
list it extends, and `null` removes a key, so a fragment can drop a pool and its arms as easily as
it can change a setting. A run records the *resolved* config, so extending costs nothing in how
completely a finished run describes itself.

**Vocabularies are imported from the library.** Model names, class weights and pool names are
checked against the library's tuples, so a config cannot name a model that does not exist, and
this file does not become a second place to update when one is added.

**`arms` lists each pool and the models fitted on it**, and `seqmaestro.arms` says which model can
take which pool, so a combination no model can run is refused here, before anything is spent.
`sources` states how each source the pools hold is computed, once for the run, and `models` the
settings of the models that have any. Where the config has a `reduction` and a `reliability`
section, every arm is reduced and profiled, unless its entry leaves it out with a reason
(`not_reduced`, `not_profiled`) or its model is a black box, which the panel leaves out with the
reason the registry gives.

**This module also says which cells a config produces** (`ExperimentConfig.expected_cells`), using
the naming helpers in `seqmaestro.pipeline.layout` that the stages write their cells with, so the
reveal's list of what a run should hold cannot drift from what the stages wrote.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seqmaestro import arms as registry
from seqmaestro.dataset import Columns, DatasetError, load_dataset
from seqmaestro.features import (
    COVARIATES,
    LLM_GENERATED,
    LLM_ITERATIVE,
    PositionalWindow,
    parse_pool,
    pool_name,
)
from seqmaestro.features.kmer import NORMALISATIONS
from seqmaestro.harness.cell import LABELS
from seqmaestro.harness.runner import PERMUTATION_SCOPES
from seqmaestro.models.weights import CLASS_WEIGHT_CHOICES
from seqmaestro.pipeline.layout import (
    CONFIG_FILE,
    REDUCED_ROLES,
    baseline_dir,
    cell_dir,
    reduction_dir,
)
from seqmaestro.pipeline.repository import REPO
from seqmaestro.reliability.panel import Settings
from seqmaestro.target import BINARY

#: The version of a config's shape: its sections, their keys and what each means. A run records it
#: with its config, and a stage refuses a run that recorded another, since it could not say which
#: of that run's settings mean something else today.
CONFIG_VERSION = 2

#: The sources whose computation a config states under `sources`, with the settings each takes. A
#: source some pool holds is stated in full, and one no pool holds may not be stated, since a setting
#: nobody reads is a mistake the run would not show. The proposer's settings serve both sources it
#: writes, so the iterative one's first round makes the one-round source's proposal.
SOURCE_SETTINGS = {
    "kmer": ("k_range", "normalise"),
    LLM_GENERATED: ("proposal_rounds", "n_per_round", "n_examples"),
    LLM_ITERATIVE: ("feedback_rounds",),
}

#: The models whose settings a config may state under `models`: their methods' configurations,
#: passed verbatim. A model no arm fits may not be given any.
MODEL_SETTINGS = {registry.DEFT_OR_CART: ("config_overrides",), registry.IGSR: ("config_overrides",)}


class ConfigError(ValueError):
    """A config that cannot be run as written. Raised with the path and the offending key."""


@dataclass(frozen=True)
class Dataset:
    """Which data, and what its columns mean.

    `task`, `splits` and `blinded` come from the config. The columns are read from the split's
    `dataset.json` when the config is loaded, and a config that states one anyway must agree with it.

    **`blinded` says whether the run withholds a fact from the language models**: the task
    description's `judge_only`, told to the judge alone. It is stated in every config, so a reader
    of the config, or of the run's report, sees whether the run was blinded without opening the
    description. The driver refuses a run whose flag disagrees with its description, and refuses a
    blinded run whose column names or covariate descriptions give away a withheld word
    (`seqmaestro.llm.blinding_leaks`).
    """

    task: str
    splits: Path
    blinded: bool
    sequence_column: str
    target_column: str
    weight_column: str | None
    covariate_columns: tuple[str, ...]
    #: What the target column holds, `binary` or `continuous` (`seqmaestro.target`).
    target_kind: str
    #: The length every sequence has, or `variable`, and the declared positional window. Read to
    #: refuse an arm that cannot read the sequences, and not recorded, since the dataset states them.
    declared_length: int | str
    positional_window: PositionalWindow | None

    @property
    def directory(self) -> Path:
        """The split directory itself, wherever the command runs from."""
        return splits_directory(self.splits)

    @property
    def columns(self) -> Columns:
        """The columns as the stages read them: see `seqmaestro.dataset.Columns`."""
        return Columns(self.sequence_column, self.target_column, self.weight_column,
                       self.covariate_columns, self.target_kind)


def splits_directory(splits: Path) -> Path:
    """A config's `dataset.splits`, which is written relative to the repository, as a real path."""
    return splits if splits.is_absolute() else REPO / splits


@dataclass(frozen=True)
class GeneratedFeatures:
    """How the arms that run a language model's code treat a feature it wrote that fails on the
    rows, stated once for all of them.

    `repair_rounds` is how many times failed candidates go back to the model, `max_undefined_share`
    the share of rows on which a candidate may be undefined and still be kept, `max_example_length`
    how much of one sequence an example in a prompt shows, and `max_seconds` the longest one
    candidate may run on the rows it is checked on. DEFT takes the first three: the time limit on a
    feature its language model writes is fixed in its code.
    """

    repair_rounds: int
    max_undefined_share: float
    max_example_length: int
    max_seconds: int


@dataclass(frozen=True)
class Grid:
    """What is swept, and how the null and the intervals are drawn. Every field is required.

    There are no defaults because these settings are what make two runs comparable. A grid that
    filled in its own bootstrap count would have intervals that mean something different from those
    of the run it is compared with, and nothing in either run would say so.
    """

    seeds: tuple[int, ...]
    labels: tuple[str, ...]
    #: The share of the training rows each seed fits on, drawn anew per seed and kept in the
    #: target's proportions. It is what makes two seeds differ for a model that is a function of
    #: its rows, so every arm and the baseline draw the same rows at a seed.
    train_fraction: float
    n_bootstrap: int
    permutation_scope: str
    #: None for a continuous target, which has no classes to weight.
    class_weight: str | None
    permutation_repeats: int
    #: Whether the judge runs. It asks a language model to rate the most important features of
    #: each cell, several times each, as `analyses` states. Its scores fill one column of the
    #: report's Stage 2 table and no stage needs them, so a run may leave it out. It is stated like
    #: everything else here, so a run's record says whether the judge was meant to run.
    judge: bool
    #: Whether the judge is told which class each feature runs with, or which way it moves a
    #: continuous target. It is a setting of the grid, like `n_bootstrap`, because it changes what
    #: the judge's scale measures: two runs that differ in it cannot be compared on the judge axis.
    judge_direction: bool
    #: One policy for a written feature that fails on the rows, across every arm that runs a
    #: language model's code.
    generated_features: GeneratedFeatures


@dataclass(frozen=True)
class KmerSettings:
    """How the k-mers of every pool that holds them are counted (`seqmaestro.features.kmer`)."""

    k_range: tuple[int, int]
    normalise: str


@dataclass(frozen=True)
class ProposerSettings:
    """How the proposer writes a proposal, for both sources it writes: calls per proposal, features
    asked for per call, and labelled rows shown."""

    proposal_rounds: int
    n_per_round: int
    n_examples: int


@dataclass(frozen=True)
class SourceSettings:
    """`sources`: how each source some pool holds is computed. None for a source no pool holds."""

    kmer: KmerSettings | None = None
    proposer: ProposerSettings | None = None
    feedback_rounds: int | None = None


@dataclass(frozen=True)
class ArmEntry:
    """One pool of `arms`: the models fitted on it, and those left out of the reduction or the
    reliability panel, each with the reason the run records."""

    pool: str
    models: tuple[str, ...]
    not_reduced: dict[str, str] = field(default_factory=dict)
    not_profiled: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeSettings:
    """How the judge rates the features, in a run whose grid has it run (`grid.judge`)."""

    #: Independent ratings of each feature. Their spread says how repeatable a rating is.
    repeats: int
    #: Whether the judge is shown each feature's code beside its description.
    show_code: bool
    #: The label conditions whose features are rated.
    labels: tuple[str, ...]
    #: The share of the panel that must get a rating. Below it the judge's stage fails.
    min_answered: float
    #: Seeds the order the panel is presented in.
    seed: int


@dataclass(frozen=True)
class RecurrenceSettings:
    """How the recurrence analysis groups the features an arm wrote."""

    #: The kinds of feature that are grouped.
    kinds: tuple[str, ...]
    #: The size of group the grouping request asks for, the same for both label conditions.
    items_per_group: int
    #: Permutations behind the recurrence null.
    draws: int
    #: Whether every arm's features are also grouped together in one pass, which is what can say
    #: that two arms reached one idea.
    across_arms: bool
    #: Seeds the order the features are listed in, and the null.
    seed: int


@dataclass(frozen=True)
class AnalysisSettings:
    """How the Stage 2 analyses take a run's cells. They are settings of the run, since each
    decides what a table of the run holds."""

    #: The features of a cell an analysis takes: its most important, by the model's importance.
    #: Models differ by orders of magnitude in how many features they use, so the analyses compare
    #: cells on one cut.
    top_n: int
    judge: JudgeSettings
    recurrence: RecurrenceSettings


@dataclass(frozen=True)
class ReductionSettings:
    """Stage 3A: how many features each arm is reduced to. Every arm is reduced but those its entry
    lists under `not_reduced`, and only real-label cells, at each arm's best validation seed."""

    n_features: int


@dataclass(frozen=True)
class ReliabilitySettings:
    """Stage 3B: the reliability panel's settings, and the arm that carries its checks.

    Every setting is required, as in `grid`, because they decide what the panel's numbers mean. The
    panel profiles every reduced arm but those of a black-box model and those its entry lists under
    `not_profiled`. The check arm carries the global-null and planted-signal reruns of Tier 1 and
    must be profiled.
    """

    panel: Settings
    check_arm: tuple[str, str]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: Dataset
    grid: Grid
    analyses: AnalysisSettings
    #: Each pool the run fits, by its canonical name, in the order the config lists them.
    arms: dict[str, ArmEntry]
    source: Path
    sources: SourceSettings = SourceSettings()
    #: Each model's settings, where the config states any (`MODEL_SETTINGS`).
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str | None = None
    reduction: ReductionSettings | None = None
    reliability: ReliabilitySettings | None = None
    #: What the config may mean that it does not say, for the driver to print.
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        """The config as it would be written back out, after overrides and with nothing implied.

        This mapping, and not the file on disk, is what a run records and what a resume is compared
        against, because `--set` overrides are part of what a run was configured with. Model names
        are recorded resolved, so `l1` is written as the model the dataset's target called for.
        """
        dataset = {
            "task": self.dataset.task,
            "splits": str(self.dataset.splits),
            "blinded": self.dataset.blinded,
            "sequence_column": self.dataset.sequence_column,
            "target_column": self.dataset.target_column,
        }
        if self.dataset.weight_column is not None:
            dataset["weight_column"] = self.dataset.weight_column
        if self.dataset.covariate_columns:
            dataset["covariate_columns"] = list(self.dataset.covariate_columns)
        dataset["target_kind"] = self.dataset.target_kind
        mapping: dict[str, Any] = {"version": CONFIG_VERSION, "name": self.name}
        if self.description is not None:
            mapping["description"] = self.description
        mapping["dataset"] = dataset
        mapping["grid"] = {
            "seeds": list(self.grid.seeds),
            "labels": list(self.grid.labels),
            "train_fraction": self.grid.train_fraction,
            "n_bootstrap": self.grid.n_bootstrap,
            "permutation_scope": self.grid.permutation_scope,
            **({"class_weight": self.grid.class_weight}
               if self.dataset.target_kind == BINARY else {}),
            "permutation_repeats": self.grid.permutation_repeats,
            "judge": self.grid.judge,
            "judge_direction": self.grid.judge_direction,
            "generated_features": asdict(self.grid.generated_features),
        }
        mapping["analyses"] = {
            "top_n": self.analyses.top_n,
            "judge": {**asdict(self.analyses.judge), "labels": list(self.analyses.judge.labels)},
            "recurrence": {**asdict(self.analyses.recurrence),
                           "kinds": list(self.analyses.recurrence.kinds)},
        }
        sources: dict[str, Any] = {}
        if self.sources.kmer is not None:
            sources["kmer"] = {"k_range": list(self.sources.kmer.k_range),
                               "normalise": self.sources.kmer.normalise}
        if self.sources.proposer is not None:
            sources[LLM_GENERATED] = asdict(self.sources.proposer)
        if self.sources.feedback_rounds is not None:
            sources[LLM_ITERATIVE] = {"feedback_rounds": self.sources.feedback_rounds}
        if sources:
            mapping["sources"] = sources
        if self.models:
            mapping["models"] = {name: {key: list(value) if isinstance(value, (list, tuple)) else value
                                        for key, value in settings.items()}
                                 for name, settings in self.models.items()}
        mapping["arms"] = {
            pool: {"models": list(entry.models),
                   **({"not_reduced": dict(entry.not_reduced)} if entry.not_reduced else {}),
                   **({"not_profiled": dict(entry.not_profiled)} if entry.not_profiled else {})}
            for pool, entry in self.arms.items()}
        if self.reduction is not None:
            mapping["reduction"] = {"n_features": self.reduction.n_features}
        if self.reliability is not None:
            mapping["reliability"] = {**asdict(self.reliability.panel),
                                      "check_arm": "/".join(self.reliability.check_arm)}
        return mapping

    def to_yaml(self) -> str:
        """Canonical text, so two configs can be compared as strings rather than key by key."""
        return yaml.safe_dump(self.to_mapping(), sort_keys=False, default_flow_style=False)

    def arm_cells(self) -> list[tuple[str, str]]:
        """Every `(model, pool)` arm of the grid, in the order the config lists them."""
        return [(model, pool) for pool, entry in self.arms.items() for model in entry.models]

    def reduced_arms(self) -> list[tuple[str, str]]:
        """The arms Stage 3A reduces; empty without a `reduction`."""
        if self.reduction is None:
            return []
        return [(model, pool) for model, pool in self.arm_cells()
                if model not in self.arms[pool].not_reduced]

    def profiled_arms(self) -> list[tuple[str, str]]:
        """The arms the reliability panel profiles; empty without a `reliability` section. Every
        reduced arm, less a black-box model's and those its entry leaves out."""
        if self.reliability is None:
            return []
        return [(model, pool) for model, pool in self.reduced_arms()
                if not registry.black_box(model) and model not in self.arms[pool].not_profiled]

    def not_reduced(self) -> dict[str, str]:
        """Every arm the reduction leaves out, as `model/pool`, and why. Empty without a
        `reduction`."""
        if self.reduction is None:
            return {}
        return {f"{model}/{pool}": self.arms[pool].not_reduced[model]
                for model, pool in self.arm_cells() if model in self.arms[pool].not_reduced}

    def not_profiled(self) -> dict[str, str]:
        """Every arm the panel leaves out, as `model/pool`, and why. Empty without a `reliability`
        section."""
        if self.reliability is None:
            return {}
        reasons = {}
        for model, pool in self.arm_cells():
            entry = self.arms[pool]
            if model in entry.not_reduced:
                reasons[f"{model}/{pool}"] = f"not reduced: {entry.not_reduced[model]}"
            elif registry.black_box(model):
                reasons[f"{model}/{pool}"] = registry.BLACK_BOX_REASON
            elif model in entry.not_profiled:
                reasons[f"{model}/{pool}"] = entry.not_profiled[model]
        return reasons

    def expected_cells(self) -> list[Path]:
        """Every cell this config produces, relative to the run directory: one baseline cell per
        seed, every arm's cells for every label and seed of the grid, and each reduced arm's three
        models.

        The reveal checks a run against this list before it opens the test split. A cell that
        failed, or an arm that was skipped, is missing from disk but still listed here, so the
        reveal sees the gap.
        """
        cells = [baseline_dir("", seed) for seed in self.grid.seeds]
        for model, pool in self.arm_cells():
            cells.extend(cell_dir("", model, pool, labels, seed)
                         for labels in self.grid.labels for seed in self.grid.seeds)
        for model, pool in self.reduced_arms():
            cells.extend(reduction_dir("", model, pool) / role for role in REDUCED_ROLES)
        return cells


def _mapping(value: Any, where: str, source: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: `{where}` must be a mapping, not {type(value).__name__}.")
    return value


def _keys(section: dict[str, Any], required: frozenset[str], optional: frozenset[str],
          where: str, source: Path) -> None:
    """Refuse a section that is missing a required key or carries one nobody reads."""
    present = set(section)
    missing = sorted(required - present)
    if missing:
        raise ConfigError(f"{source}: `{where}` is missing {', '.join(missing)}.")
    unknown = sorted(present - required - optional)
    if unknown:
        known = ", ".join(sorted(required | optional))
        raise ConfigError(f"{source}: `{where}` has no key {', '.join(unknown)}. Known: {known}.")


def _string(section: dict[str, Any], key: str, where: str, source: Path) -> str:
    value = section[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{source}: `{where}.{key}` must be a non-empty string.")
    return value


def _positive_int(section: dict[str, Any], key: str, where: str, source: Path, floor: int = 1) -> int:
    value = section[key]
    # `bool` is an `int` in Python, and `n_bootstrap: true` should not be taken as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
        raise ConfigError(f"{source}: `{where}.{key}` must be an integer >= {floor}, not {value!r}.")
    return value


def _fraction(section: dict[str, Any], key: str, where: str, source: Path) -> float:
    """A number strictly between 0 and 1, or exactly 1."""
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ConfigError(f"{source}: `{where}.{key}` must be a number in (0, 1], not {value!r}.")
    return float(value)


def _choice(section: dict[str, Any], key: str, allowed: tuple[str, ...], where: str, source: Path) -> str:
    value = _string(section, key, where, source)
    if value not in allowed:
        raise ConfigError(f"{source}: `{where}.{key}` is {value!r}; expected one of {', '.join(allowed)}.")
    return value


def _bool(section: dict[str, Any], key: str, where: str, source: Path) -> bool:
    value = section[key]
    # `isinstance(True, int)` is true, so an unguarded read would silently accept `1` and `0`.
    if not isinstance(value, bool):
        raise ConfigError(f"{source}: `{where}.{key}` must be true or false, not {value!r}.")
    return value


def _list_of(section: dict[str, Any], key: str, kind: type, where: str, source: Path) -> list:
    value = section[key]
    if not isinstance(value, list) or not value or any(not isinstance(item, kind) for item in value):
        raise ConfigError(
            f"{source}: `{where}.{key}` must be a non-empty list of {kind.__name__}, not {value!r}.")
    return value


#: What a config may restate about its dataset: the columns and the target's kind. A recorded run
#: config carries them so that it describes the run on its own. A hand-written config need not.
STATED = ("sequence_column", "target_column", "weight_column", "covariate_columns", "target_kind")


def _dataset(section: dict[str, Any], source: Path) -> Dataset:
    """The prepared dataset the config names, with its columns read from its `dataset.json`.

    A config may restate a column, as the copy recorded beside a run does so that the run
    describes itself, but only if it agrees. A config that disagrees with the data it names is
    refused, since one of the two is wrong and nothing downstream could say which.
    """
    _keys(section, required=frozenset({"task", "splits", "blinded"}), optional=frozenset(STATED),
          where="dataset", source=source)
    splits = Path(_string(section, "splits", "dataset", source))
    try:
        prepared = load_dataset(splits_directory(splits))
    except DatasetError as error:
        raise ConfigError(f"{source}: `dataset.splits` is not a usable dataset. {error}") from error
    for key in STATED:
        # The covariates are a list in a config and a tuple once read, so they compare as lists.
        recorded = getattr(prepared, key)
        recorded = list(recorded) if isinstance(recorded, tuple) else recorded
        if key in section and section[key] != recorded:
            raise ConfigError(
                f"{source}: `dataset.{key}` is {section[key]!r}, but {splits}/dataset.json records "
                f"{recorded!r}. The dataset states its own columns; remove the key.")
    return Dataset(
        task=_string(section, "task", "dataset", source),
        splits=splits,
        blinded=_bool(section, "blinded", "dataset", source),
        sequence_column=prepared.sequence_column,
        target_column=prepared.target_column,
        weight_column=prepared.weight_column,
        covariate_columns=prepared.covariate_columns,
        target_kind=prepared.target_kind,
        declared_length=prepared.declared_length,
        positional_window=prepared.positional_window,
    )


def _grid(section: dict[str, Any], target_kind: str, source: Path) -> Grid:
    """The grid. `class_weight` is required for a binary target and refused for a continuous one,
    which has no classes to weight."""
    weighted = frozenset({"class_weight"}) if target_kind == BINARY else frozenset()
    _keys(section,
          required=frozenset({"seeds", "labels", "train_fraction", "n_bootstrap", "permutation_scope",
                              "permutation_repeats", "judge", "judge_direction",
                              "generated_features"})
          | weighted,
          optional=frozenset(), where="grid", source=source)
    seeds = _list_of(section, "seeds", int, "grid", source)
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"{source}: `grid.seeds` repeats a seed; a repeat is not a second draw.")
    labels = _list_of(section, "labels", str, "grid", source)
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ConfigError(f"{source}: `grid.labels` has {', '.join(unknown)}; expected "
                          f"{', '.join(LABELS)}.")
    train_fraction = _fraction(section, "train_fraction", "grid", source)
    if train_fraction == 1:
        raise ConfigError(f"{source}: `grid.train_fraction` must be below 1. A seed that fitted on "
                          f"every training row would give most models the same fit as every other "
                          f"seed, and the run would have no stability to measure.")
    return Grid(
        seeds=tuple(seeds),
        labels=tuple(labels),
        train_fraction=train_fraction,
        # Zero is allowed: it turns the percentile intervals off for a smoke run.
        n_bootstrap=_positive_int(section, "n_bootstrap", "grid", source, floor=0),
        permutation_scope=_choice(section, "permutation_scope", PERMUTATION_SCOPES, "grid", source),
        class_weight=(_choice(section, "class_weight", tuple(CLASS_WEIGHT_CHOICES), "grid", source)
                      if weighted else None),
        # Zero skips the cross-arm importance entirely.
        permutation_repeats=_positive_int(section, "permutation_repeats", "grid", source, floor=0),
        judge=_bool(section, "judge", "grid", source),
        judge_direction=_bool(section, "judge_direction", "grid", source),
        generated_features=_generated_features(section["generated_features"], source),
    )


def _generated_features(value: Any, source: Path) -> GeneratedFeatures:
    where = "grid.generated_features"
    section = _mapping(value, where, source)
    _keys(section, required=frozenset(GeneratedFeatures.__dataclass_fields__), optional=frozenset(),
          where=where, source=source)
    share = section["max_undefined_share"]
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
        raise ConfigError(f"{source}: `{where}.max_undefined_share` must be a number in [0, 1], "
                          f"not {share!r}.")
    return GeneratedFeatures(
        # Zero sends nothing back to the model.
        repair_rounds=_positive_int(section, "repair_rounds", where, source, floor=0),
        max_undefined_share=float(share),
        max_example_length=_positive_int(section, "max_example_length", where, source),
        # Whole seconds, as IGSR takes its limit.
        max_seconds=_positive_int(section, "max_seconds", where, source),
    )


def _analyses(section: dict[str, Any], source: Path) -> AnalysisSettings:
    """The `analyses` section, required in full: a table computed under a cut or a count the
    config does not state could not be compared with another run's."""
    _keys(section, required=frozenset({"top_n", "judge", "recurrence"}), optional=frozenset(),
          where="analyses", source=source)
    judge = _mapping(section["judge"], "analyses.judge", source)
    _keys(judge, required=frozenset(JudgeSettings.__dataclass_fields__), optional=frozenset(),
          where="analyses.judge", source=source)
    labels = _list_of(judge, "labels", str, "analyses.judge", source)
    unknown = [label for label in labels if label not in LABELS]
    if unknown or len(set(labels)) != len(labels):
        raise ConfigError(f"{source}: `analyses.judge.labels` must hold each of "
                          f"{', '.join(LABELS)} at most once, not {labels!r}.")
    recurrence = _mapping(section["recurrence"], "analyses.recurrence", source)
    _keys(recurrence, required=frozenset(RecurrenceSettings.__dataclass_fields__),
          optional=frozenset(), where="analyses.recurrence", source=source)
    return AnalysisSettings(
        top_n=_positive_int(section, "top_n", "analyses", source),
        judge=JudgeSettings(
            repeats=_positive_int(judge, "repeats", "analyses.judge", source),
            show_code=_bool(judge, "show_code", "analyses.judge", source),
            labels=tuple(labels),
            min_answered=_fraction(judge, "min_answered", "analyses.judge", source),
            seed=_positive_int(judge, "seed", "analyses.judge", source, floor=0)),
        recurrence=RecurrenceSettings(
            kinds=tuple(_list_of(recurrence, "kinds", str, "analyses.recurrence", source)),
            items_per_group=_positive_int(recurrence, "items_per_group", "analyses.recurrence",
                                          source),
            draws=_positive_int(recurrence, "draws", "analyses.recurrence", source),
            across_arms=_bool(recurrence, "across_arms", "analyses.recurrence", source),
            seed=_positive_int(recurrence, "seed", "analyses.recurrence", source, floor=0)))


def _reduction(section: dict[str, Any], source: Path) -> ReductionSettings:
    _keys(section, required=frozenset({"n_features"}), optional=frozenset(), where="reduction",
          source=source)
    return ReductionSettings(n_features=_positive_int(section, "n_features", "reduction", source))


#: The keys of the `reliability` section that set how the panel computes, as `Settings` names them.
RELIABILITY_SETTINGS = frozenset(Settings.__dataclass_fields__)


def _reliability(section: dict[str, Any], reduction: ReductionSettings | None, target_kind: str,
                 source: Path) -> ReliabilitySettings:
    where = "reliability"
    _keys(section, required=RELIABILITY_SETTINGS | {"check_arm"}, optional=frozenset(), where=where,
          source=source)
    if reduction is None:
        raise ConfigError(f"{source}: `reliability` profiles each arm's reduced model, so it needs a "
                          f"`reduction` section.")
    signal_sd = section["planted_signal_sd"]
    if isinstance(signal_sd, bool) or not isinstance(signal_sd, (int, float)) or signal_sd <= 0:
        raise ConfigError(f"{source}: `{where}.planted_signal_sd` must be a positive number.")
    panel = Settings(
        pairs=_positive_int(section, "pairs", where, source),
        permutations=_positive_int(section, "permutations", where, source),
        cluster_rho=_fraction(section, "cluster_rho", where, source),
        max_modal_share=_fraction(section, "max_modal_share", where, source),
        planted_count=_positive_int(section, "planted_count", where, source),
        planted_signal_sd=float(signal_sd),
        seed=_positive_int(section, "seed", where, source, floor=0),
    )
    model, _, pool = _string(section, "check_arm", where, source).partition("/")
    return ReliabilitySettings(panel=panel, check_arm=(registry.resolve(model, target_kind), pool))


def _check_reliability_scope(config: ExperimentConfig, source: Path) -> None:
    """The check arm must be one the panel profiles."""
    check_arm = config.reliability.check_arm
    if check_arm not in config.profiled_arms():
        profiled = ", ".join("/".join(arm) for arm in config.profiled_arms()) or "none"
        raise ConfigError(f"{source}: `reliability.check_arm` is {'/'.join(check_arm)}, which the "
                          f"panel does not profile. It profiles: {profiled}.")


def _reasons(entry: dict[str, Any], key: str, models: list[str], target_kind: str, where: str,
             source: Path) -> dict[str, str]:
    """An entry's `not_reduced` or `not_profiled`: models of the entry, each with its reason."""
    reasons = {}
    for name, reason in _mapping(entry.get(key, {}), f"{where}.{key}", source).items():
        model = registry.resolve(name, target_kind)
        if model not in models:
            raise ConfigError(f"{source}: `{where}.{key}` names {name}, which `{where}.models` does "
                              f"not list.")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"{source}: `{where}.{key}.{name}` needs a reason, which the run "
                              f"records.")
        reasons[model] = reason
    return reasons


def _refuse_pool(pool: str, dataset: Dataset, where: str, source: Path) -> None:
    """A pool no model could be fitted on here: one that does not parse or is out of order, one with
    two written sources, covariates the dataset does not have, or positions on sequences of several
    lengths with no window to read them in."""
    try:
        members = parse_pool(pool)
    except ValueError as error:
        raise ConfigError(f"{source}: `{where}` is not a pool: {error}") from error
    if pool != pool_name(members):
        raise ConfigError(f"{source}: `{where}` names its sources out of order. Write it "
                          f"`{pool_name(members)}`, so that one pool has one name.")
    for problem in registry.pool_problems(pool):
        raise ConfigError(f"{source}: `{where}`: {problem}.")
    if COVARIATES in members and not dataset.covariate_columns:
        raise ConfigError(f"{source}: `{where}` holds the covariates, and {dataset.splits} declares "
                          f"none.")
    if "positional" in members and dataset.declared_length == "variable" \
            and dataset.positional_window is None:
        raise ConfigError(
            f"{source}: `{where}` reads positions, and {dataset.splits} holds sequences of "
            f"different lengths with no positional window. Declare one at Stage 1 "
            f"(docs/dataset.md), or leave the pool out.")


def _arms(section: dict[str, Any], dataset: Dataset, source: Path
          ) -> tuple[dict[str, ArmEntry], list[str]]:
    """`arms`: each pool, the models fitted on it, and what Stage 3 leaves out of each. Returns the
    entries, and warnings about what the arms leave unused."""
    entries: dict[str, ArmEntry] = {}
    for pool, raw in section.items():
        where = f"arms.{pool}"
        _refuse_pool(pool, dataset, where, source)
        entry = _mapping(raw, where, source)
        _keys(entry, required=frozenset({"models"}),
              optional=frozenset({"not_reduced", "not_profiled"}), where=where, source=source)
        models: list[str] = []
        for name in _list_of(entry, "models", str, where, source):
            model = registry.resolve(name, dataset.target_kind)
            try:
                found = registry.problems(model, pool, dataset.target_kind,
                                          weighted=dataset.weight_column is not None)
            except ValueError as error:
                raise ConfigError(f"{source}: `{where}.models`: {error}") from error
            if found:
                raise ConfigError(f"{source}: `{where}`: {'; '.join(found)}.")
            if model in models:
                raise ConfigError(f"{source}: `{where}.models` names {model} twice.")
            models.append(model)
        not_reduced = _reasons(entry, "not_reduced", models, dataset.target_kind, where, source)
        not_profiled = _reasons(entry, "not_profiled", models, dataset.target_kind, where, source)
        both = sorted(set(not_reduced) & set(not_profiled))
        if both:
            raise ConfigError(f"{source}: `{where}` leaves {', '.join(both)} out under both "
                              f"`not_reduced` and `not_profiled`. An arm the reduction leaves out "
                              f"is left out of the panel too, so name it under `not_reduced` only.")
        boxed = sorted(model for model in not_profiled if registry.black_box(model))
        if boxed:
            raise ConfigError(f"{source}: `{where}.not_profiled` names {', '.join(boxed)}, which the "
                              f"panel leaves out anyway: {registry.BLACK_BOX_REASON}.")
        entries[pool] = ArmEntry(pool, tuple(models), not_reduced, not_profiled)
    warnings = []
    if dataset.covariate_columns and not any(COVARIATES in parse_pool(pool) for pool in entries):
        warnings.append(f"{dataset.splits} declares {len(dataset.covariate_columns)} covariates, and "
                        f"no pool in `arms` holds them, so no arm is handed them.")
    return entries, warnings


def _sources(section: dict[str, Any], pools: list[str], source: Path) -> SourceSettings:
    """`sources`, against the pools: every source some pool holds is stated in full, and no other."""
    held = {member for pool in pools for member in parse_pool(pool)}
    needed = {name for name in SOURCE_SETTINGS if name in held}
    if LLM_ITERATIVE in held:
        needed.add(LLM_GENERATED)
    unknown = sorted(set(section) - set(SOURCE_SETTINGS))
    if unknown:
        raise ConfigError(f"{source}: `sources` has no source {', '.join(unknown)}. The sources with "
                          f"settings are {', '.join(SOURCE_SETTINGS)}.")
    unused = sorted(set(section) - needed)
    if unused:
        raise ConfigError(f"{source}: `sources` sets {', '.join(unused)}, which no pool in `arms` "
                          f"uses. Remove it, or set it to null in a config that extends this one.")
    missing = sorted(needed - set(section))
    if missing:
        raise ConfigError(f"{source}: a pool in `arms` needs {', '.join(missing)}, and `sources` "
                          f"does not say how it is computed.")
    settings = {}
    for name in needed:
        where = f"sources.{name}"
        settings[name] = _mapping(section[name], where, source)
        _keys(settings[name], required=frozenset(SOURCE_SETTINGS[name]), optional=frozenset(),
              where=where, source=source)
    kmer = proposer = feedback_rounds = None
    if "kmer" in settings:
        k_range = _list_of(settings["kmer"], "k_range", int, "sources.kmer", source)
        if len(k_range) != 2 or k_range[0] < 1 or k_range[1] < k_range[0]:
            raise ConfigError(f"{source}: `sources.kmer.k_range` must be [min, max] with "
                              f"1 <= min <= max.")
        kmer = KmerSettings(k_range=(k_range[0], k_range[1]),
                            normalise=_choice(settings["kmer"], "normalise", NORMALISATIONS,
                                              "sources.kmer", source))
    if LLM_GENERATED in settings:
        where = f"sources.{LLM_GENERATED}"
        proposer = ProposerSettings(**{key: _positive_int(settings[LLM_GENERATED], key, where, source)
                                       for key in SOURCE_SETTINGS[LLM_GENERATED]})
    if LLM_ITERATIVE in settings:
        feedback_rounds = _positive_int(settings[LLM_ITERATIVE], "feedback_rounds",
                                        f"sources.{LLM_ITERATIVE}", source)
    return SourceSettings(kmer=kmer, proposer=proposer, feedback_rounds=feedback_rounds)


def _model_settings(section: dict[str, Any], fitted: set[str], source: Path
                    ) -> dict[str, dict[str, Any]]:
    """`models`: each model's settings, for models some arm fits."""
    unknown = sorted(set(section) - set(MODEL_SETTINGS))
    if unknown:
        raise ConfigError(f"{source}: `models` has no settings for {', '.join(unknown)}. The models "
                          f"with settings are {', '.join(MODEL_SETTINGS)}.")
    unused = sorted(set(section) - fitted)
    if unused:
        raise ConfigError(f"{source}: `models` sets {', '.join(unused)}, which no arm fits. Remove "
                          f"it, or set it to null in a config that extends this one.")
    settings = {}
    for name, raw in section.items():
        where = f"models.{name}"
        entry = _mapping(raw, where, source)
        _keys(entry, required=frozenset(MODEL_SETTINGS[name]), optional=frozenset(), where=where,
              source=source)
        overrides = _list_of(entry, "config_overrides", str, where, source)
        malformed = [item for item in overrides if "=" not in item]
        if malformed:
            raise ConfigError(f"{source}: `{where}.config_overrides` needs key=value: "
                              f"{', '.join(malformed)}.")
        settings[name] = {"config_overrides": overrides}
    return settings


def _apply_override(document: dict[str, Any], assignment: str, source: Path) -> None:
    """Apply one `section.key=value` override in place, parsing the value as YAML.

    Overrides let the whole pipeline be exercised cheaply (a tiny split, one seed, no bootstrap)
    without a second config file that would drift from the real one. They are applied before
    validation, so an override is checked as a written-in value would be, and they are part of what
    the run records.

    An intermediate section must already exist: `--set grid.n_bootstrap=0` edits the grid, but
    `--set gird.n_bootstrap=0` is a typo and would otherwise create a section nobody reads.
    """
    path, separator, raw = assignment.partition("=")
    if not separator or not path:
        raise ConfigError(f"--set needs section.key=value, not {assignment!r}.")
    *sections, key = path.split(".")
    target = document
    for index, name in enumerate(sections):
        if name not in target or not isinstance(target[name], dict):
            known = ", ".join(sorted(k for k, v in target.items() if isinstance(v, dict))) or "none"
            raise ConfigError(
                f"{source}: --set {assignment}: no section `{'.'.join(sections[:index + 1])}`. "
                f"Sections here: {known}.")
        target = target[name]
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ConfigError(f"{source}: --set {assignment}: cannot read {raw!r} as a value: {error}") from error
    # `null` removes the key, as it does in a fragment, so `--set arms.llm_generated=null` drops the
    # arms on that pool. Removing a key that is not there is treated as a misspelling, because
    # silently doing nothing would leave those arms running with nothing to say so.
    if value is None:
        if key not in target:
            raise ConfigError(
                f"{source}: --set {assignment}: there is no `{path}` to remove. "
                f"Here: {', '.join(sorted(target)) or 'nothing'}.")
        del target[key]
    else:
        target[key] = value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fold `override` onto `base`: mappings merge, `null` removes, everything else replaces.

    A list replaces the base list instead of appending to it, which is what a reader expects of
    `models` or `config_overrides`: if lists appended, the extending file could never shorten one.
    """
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(source: Path) -> dict[str, Any]:
    try:
        text = source.read_text()
    except OSError as error:
        raise ConfigError(f"cannot read {source}: {error}") from error
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{source} is not readable as YAML: {error}") from error
    return _mapping(raw, "the config", source)


def resolve(source: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a config and fold it onto whatever it extends, innermost first.

    `extends` is a path relative to the extending file, so a config directory can be moved whole.
    It is consumed here, so a resolved config has no `extends`, and the copy recorded beside a run
    is a complete statement on its own.
    """
    here = source.resolve()
    if here in seen:
        chain = " -> ".join(path.name for path in (*seen, here))
        raise ConfigError(f"{source}: `extends` is circular: {chain}.")
    document = _read(source)
    parent = document.pop("extends", None)
    if parent is None:
        return document
    if not isinstance(parent, str) or not parent:
        raise ConfigError(f"{source}: `extends` must be a path relative to {source.parent}.")
    return _merge(resolve(source.parent / parent, (*seen, here)), document)


def load(path: Path | str, overrides: tuple[str, ...] = ()) -> ExperimentConfig:
    """Read, extend, override and validate a config, or raise `ConfigError` naming the bad key.

    A config may state the `version` of the shape it is written in, and a recorded one always
    does. A config that states another version than this code reads is refused.
    """
    source = Path(path)
    document = resolve(source)
    for assignment in overrides:
        _apply_override(document, assignment, source)

    version = document.pop("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigError(f"{source} is written as config version {version!r}, and this code "
                          f"reads version {CONFIG_VERSION}.")
    _keys(document, required=frozenset({"name", "dataset", "grid", "analyses", "arms"}),
          optional=frozenset({"description", "sources", "models", "reduction", "reliability"}),
          where="the config", source=source)
    description = document.get("description")
    if description is not None and not isinstance(description, str):
        raise ConfigError(f"{source}: `description` must be text.")
    dataset = _dataset(_mapping(document["dataset"], "dataset", source), source)
    arms, warnings = _arms(_mapping(document["arms"], "arms", source), dataset, source)
    if not arms:
        raise ConfigError(f"{source}: `arms` is empty, so the run would do nothing but analyse.")
    sources = _sources(_mapping(document.get("sources", {}), "sources", source), list(arms), source)
    models = _model_settings(_mapping(document.get("models", {}), "models", source),
                             {model for entry in arms.values() for model in entry.models}, source)
    reduction = reliability = None
    if "reduction" in document:
        reduction = _reduction(_mapping(document["reduction"], "reduction", source), source)
    if "reliability" in document:
        reliability = _reliability(_mapping(document["reliability"], "reliability", source),
                                   reduction, dataset.target_kind, source)
    config = ExperimentConfig(
        name=_string(document, "name", "the config", source),
        dataset=dataset,
        grid=_grid(_mapping(document["grid"], "grid", source), dataset.target_kind, source),
        analyses=_analyses(_mapping(document["analyses"], "analyses", source), source),
        arms=arms,
        source=source,
        sources=sources,
        models=models,
        description=description,
        reduction=reduction,
        reliability=reliability,
        warnings=tuple(warnings),
    )
    if reliability is not None:
        _check_reliability_scope(config, source)
    return config


def load_recorded(run_dir: Path | str) -> ExperimentConfig:
    """The config a run recorded as it started (`<run>/config.yaml`), from which every stage takes
    its settings.

    The driver records a config with its version. A file that states none is refused, since it is
    no record of a run, and reading it as today's version would be a guess.
    """
    path = Path(run_dir) / CONFIG_FILE
    if "version" not in _read(path):
        raise ConfigError(f"{path} states no `version`, so it is not a config the driver recorded "
                          f"for a run. A stage takes its settings from a recorded config alone.")
    return load(path)


def main(argv: list[str] | None = None) -> int:
    """Validate a config and print it as it would be recorded. No run, no cost."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)
    try:
        print(load(args.config, tuple(args.overrides)).to_yaml(), end="")
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    return 0
