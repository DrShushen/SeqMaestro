"""The config of a run: the dataset it names, the grid it sweeps and the arms it fits.

`experiments/configs/README.md` says why a run is stated in a file, how a config shares settings
through `extends`, and which configs are refused and why. This module loads a config, folds in the
files it extends, and refuses one it cannot run as written, such as one that names an unknown key
or a model or pool that does not exist. The resolved config is written into the run's directory
when the run starts, so that the finished run describes itself.

**Facts about the data are never defaulted.** The `dataset` section names the prepared split and
the task description, and both are required. A config need not state the columns, because they
come from the split's `dataset.json`, which Stage 1 writes, and a config that states one anyway (as
the config a run records does) must agree with that file. A default would be one dataset's answer
applied silently to another, and for the task description it would mean that a language model is
told the wrong thing and answers confidently anyway.

**Vocabularies are imported from the library.** Model names, class weights and pool names are
checked against the library's tuples, so a config cannot name a model that does not exist, and
adding a model needs no change to this file.

**`language_model` states how every call to the language model is made**: the reasoning effort,
the retries and the wait, for the proposer, the judge, the recurrence analysis, DEFT and IGSR alike.
Which provider and model are called is not a setting of the config: the environment names them,
with their credentials (`seqmaestro.llm.client`), and each part that calls the model records the
model it called.

**`arms` lists each pool and the models fitted on it**, and `seqmaestro.arms` says which model can
take which pool, so a combination that cannot run is refused here, before any work is done.
`sources` states how each feature source in the pools is computed, once for the whole run, and
`models` holds the settings a run changes: a Core model's own settings (those its class lists,
`seqmaestro.models.settings`), or overrides of a vendored method's configuration. Where the config
has a `reduction` and a `reliability` section, every arm is reduced and profiled, unless its entry
leaves it out with a reason (`not_reduced`, `not_profiled`) or its model is a black box, which the
panel leaves out with the reason the registry gives.

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
    MOTIF_DISCOVERED,
    MOTIF_KNOWN,
    PositionalWindow,
    parse_pool,
    pool_name,
)
from seqmaestro.features.kmer import NORMALISATIONS
from seqmaestro.harness.cell import LABELS
from seqmaestro.harness.runner import PERMUTATION_SCOPES
from seqmaestro.llm.client import REASONING_EFFORTS, CallSettings
from seqmaestro.models import effective_settings, make_model, models_for
from seqmaestro.models import settings as core_settings
from seqmaestro.models.weights import CLASS_WEIGHT_CHOICES
from seqmaestro.motifs.discovery import DIRECTION_SETTINGS, STREME_MAX_WIDTH, DiscoverySettings
from seqmaestro.motifs.scan import STRANDS, SUMMARIES, ScanSettings
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
#: with its config, and a stage refuses a run that recorded another version, because it could not
#: tell which of that run's settings now mean something else.
CONFIG_VERSION = 4

#: The feature sources whose computation a config states under `sources`, with the settings each
#: takes. A source that some pool holds must be stated in full, whereas one that is in no pool may
#: not be stated, because a setting that nothing uses is a mistake the run would not show. The
#: proposer's settings serve both sources it writes, so the iterative source's first round is a
#: proposal made as for the one-round source.
SOURCE_SETTINGS = {
    "kmer": ("k_range", "normalise"),
    MOTIF_KNOWN: ("collection", "strands", "summaries", "p_threshold", "normalise",
                  "min_match_share"),
    MOTIF_DISCOVERED: ("strands", "summaries", "p_threshold", "normalise", "min_match_share",
                       "directions", "continuous_fraction", "min_width", "max_width", "max_motifs"),
    LLM_GENERATED: ("proposal_rounds", "n_per_round", "n_examples"),
    LLM_ITERATIVE: ("feedback_rounds",),
}
#: The optional settings of a source, each with what leaving it out means.
SOURCE_OPTIONAL = {MOTIF_DISCOVERED: {"annotate_with": "no found motif is compared with a collection"}}

#: The models fitted in their methods' environments, whose settings a config states under `models`
#: as overrides of the method's own configuration, passed verbatim. A Core model takes the settings
#: its class lists instead (`seqmaestro.models.settings`). A model that no arm fits may not be
#: given any.
VENDORED_SETTINGS = {registry.DEFT_OR_CART: ("config_overrides",), registry.IGSR: ("config_overrides",)}


class ConfigError(ValueError):
    """A config that cannot be run as written. Raised with the path and the offending key."""


@dataclass(frozen=True)
class Dataset:
    """The dataset of a run and the meaning of its columns.

    `task`, `splits` and `blinded` come from the config. The columns are read from the split's
    `dataset.json` when the config is loaded, and a config that states one anyway must agree with it.

    **`blinded` says whether the run withholds a fact from the language models**: the task
    description's `judge_only`, which only the judge is told. It is stated in every config, so a
    reader of the config, or of the run's report, sees whether the run was blinded without opening
    the description. The driver refuses a run whose flag disagrees with its description, and it
    refuses a blinded run whose column names or covariate descriptions give away a withheld word
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
    #: The length of every sequence (or `variable`) and the declared positional window, which let
    #: the config refuse an arm that cannot take the dataset's sequences. They are not recorded,
    #: because the dataset states them.
    declared_length: int | str
    positional_window: PositionalWindow | None

    @property
    def directory(self) -> Path:
        """The split directory as an absolute path, whichever directory the command runs from."""
        return splits_directory(self.splits)

    @property
    def columns(self) -> Columns:
        """The columns in the form the stages use (`seqmaestro.dataset.Columns`)."""
        return Columns(self.sequence_column, self.target_column, self.weight_column,
                       self.covariate_columns, self.target_kind)


def splits_directory(splits: Path) -> Path:
    """A config's `dataset.splits` (written relative to the repository) as an absolute path."""
    return splits if splits.is_absolute() else REPO / splits


@dataclass(frozen=True)
class GeneratedFeatures:
    """How features are asked for, and how a written feature that fails on the rows is treated,
    stated once for every arm on a written source.

    `repair_rounds` is how many times failed candidates go back to the model, `max_undefined_share`
    the share of rows on which a candidate may be undefined and still be kept, `max_example_length`
    how much of one sequence an example in a prompt shows, and `max_seconds` the longest one
    candidate may run on the rows it is checked on. DEFT takes the first three, because the time
    limit on a feature its language model writes is fixed in DEFT's code.

    `guidance` is what the person running the analysis asks of the form of the features, such as
    a functional form or a stretch of sequence to measure in, or None for no guidance. Every arm
    that writes features is shown it word for word: the proposer in its requests, and DEFT's
    generative finder and IGSR in the prompts their adapters build from the task description.
    """

    repair_rounds: int
    max_undefined_share: float
    max_example_length: int
    max_seconds: int
    guidance: str | None


@dataclass(frozen=True)
class Grid:
    """What a run sweeps, and how the null and the intervals are drawn.

    Every field is required, with no default, because these settings are what make two runs
    comparable. A grid that defaulted the bootstrap count would have intervals that mean something
    different from those of the run it is compared with, and neither run would show it.
    """

    seeds: tuple[int, ...]
    labels: tuple[str, ...]
    #: The share of the training rows each seed fits on, drawn afresh for each seed and kept in the
    #: target's proportions. It is what makes two seeds differ for a model that is a function of
    #: its rows. At a given seed, every arm and the baseline draw the same rows.
    train_fraction: float
    n_bootstrap: int
    permutation_scope: str
    #: None for a continuous target, which has no classes to weight.
    class_weight: str | None
    permutation_repeats: int
    #: Whether the judge runs. The judge rates the most important features of each cell, several
    #: times each, as `analyses` states. Its scores fill one column of the report's Stage 2 table
    #: and the stages do not depend on them, so a run may leave the judge out. The setting is
    #: required all the same, so that a run's record says whether the judge was meant to run.
    judge: bool
    #: Whether the judge is told which class each feature is associated with, or which way it moves
    #: a continuous target. It is a setting of the grid (like `n_bootstrap`) because it changes what
    #: the judge's scale measures, so the judge's scores of two runs that differ in it cannot be
    #: compared.
    judge_direction: bool
    #: How features are asked for, and how a written feature that fails on the rows is treated, for
    #: every arm on a written source.
    generated_features: GeneratedFeatures


@dataclass(frozen=True)
class KmerSettings:
    """How the k-mers of every pool that holds them are counted (`seqmaestro.features.kmer`)."""

    k_range: tuple[int, int]
    normalise: str


@dataclass(frozen=True)
class KnownMotifSettings:
    """How the motifs of a published collection are scanned (`seqmaestro.features.motif`): the
    collection, by the name it was fetched under (`data/motif_collections/<name>/`), and how its
    scans are summarised. Which collection applies and which strands are scanned are facts about a
    dataset, so a dataset's config states them, whereas the other settings come from
    `_motifs.yaml`. `min_match_share` is the share of the fitting rows that must contain a match
    for a count to be kept."""

    collection: str
    strands: str
    summaries: tuple[str, ...]
    p_threshold: float
    normalise: str
    min_match_share: float

    def scan(self) -> ScanSettings:
        return ScanSettings(strands=self.strands, p_threshold=self.p_threshold,
                            summaries=self.summaries, normalise=self.normalise)


@dataclass(frozen=True)
class DiscoveredMotifSettings:
    """How motifs are discovered in the fitting rows and then scanned
    (`seqmaestro.motifs.discovery`). The scan takes the same settings as for the known motifs, and
    STREME (the MEME Suite's motif discovery tool) takes the directions searched, the share of a
    continuous target's rows at each end, the range of motif widths, and how many motifs are kept
    per direction. `annotate_with` names the collection a found motif is compared with, or is None.
    `strands` is a fact about the dataset, which the dataset's config states."""

    strands: str
    summaries: tuple[str, ...]
    p_threshold: float
    normalise: str
    min_match_share: float
    directions: str
    continuous_fraction: float
    min_width: int
    max_width: int
    max_motifs: int
    annotate_with: str | None = None

    def scan(self) -> ScanSettings:
        return ScanSettings(strands=self.strands, p_threshold=self.p_threshold,
                            summaries=self.summaries, normalise=self.normalise)

    def discovery(self) -> DiscoverySettings:
        return DiscoverySettings(directions=self.directions,
                                 continuous_fraction=self.continuous_fraction,
                                 min_width=self.min_width, max_width=self.max_width,
                                 max_motifs=self.max_motifs, strands=self.strands)


@dataclass(frozen=True)
class ProposerSettings:
    """How the proposer writes a proposal for both sources it writes: the calls per proposal, the
    features asked for per call, and the labelled rows shown."""

    proposal_rounds: int
    n_per_round: int
    n_examples: int


@dataclass(frozen=True)
class SourceSettings:
    """`sources`: how each feature source that some pool holds is computed, or None for a source
    that is in no pool."""

    kmer: KmerSettings | None = None
    motif_known: KnownMotifSettings | None = None
    motif_discovered: DiscoveredMotifSettings | None = None
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
    """How the judge rates the features, in a run whose grid turns it on (`grid.judge`)."""

    #: The number of independent ratings of each feature, whose spread shows how repeatable a
    #: rating is.
    repeats: int
    #: Whether the judge is shown each feature's code beside its description.
    show_code: bool
    #: The label conditions whose features are rated.
    labels: tuple[str, ...]
    #: The share of the judge's panel that must receive a rating, below which the judge's stage
    #: fails.
    min_answered: float
    #: The seed for the order in which the judge's panel is presented.
    seed: int


@dataclass(frozen=True)
class RecurrenceSettings:
    """How the recurrence analysis groups the features an arm wrote."""

    #: The kinds of feature that are grouped.
    kinds: tuple[str, ...]
    #: The group size the grouping request asks for, which is the same for both label conditions.
    items_per_group: int
    #: The number of permutations drawn for the recurrence null.
    draws: int
    #: Whether the features of all arms are also grouped together in one pass, which is what shows
    #: that two arms reached the same idea.
    across_arms: bool
    #: The seed for the order in which the features are listed, and for the null.
    seed: int


@dataclass(frozen=True)
class AnalysisSettings:
    """How the Stage 2 analyses use a run's cells. These are settings of the run because each
    decides what a table of the run holds."""

    #: How many features of a cell an analysis takes: the most important ones, by the model's
    #: importance. Models differ by orders of magnitude in how many features they use, so the
    #: analyses compare cells at the same cutoff.
    top_n: int
    judge: JudgeSettings
    recurrence: RecurrenceSettings


@dataclass(frozen=True)
class ReductionSettings:
    """Stage 3A: how many features each arm is reduced to. Every arm is reduced except those its
    entry lists under `not_reduced`, and only from its real-label cell at its best validation
    seed."""

    n_features: int


@dataclass(frozen=True)
class ReliabilitySettings:
    """Stage 3B: the reliability panel's settings, and the arm its checks are made on.

    Every setting is required, as in `grid`, because the settings decide what the panel's numbers
    mean. The panel profiles every reduced arm except those of a black-box model and those its
    entry lists under `not_profiled`. The global-null and planted-signal reruns of Tier 1 are made
    on the check arm, which must be profiled.
    """

    panel: Settings
    check_arm: tuple[str, str]


@dataclass(frozen=True)
class ExperimentConfig:
    """A loaded config, with a field for each section of the file. The type of a section's field,
    such as `Grid` for `grid`, lists the section's keys.

    `name`, `dataset`, `grid`, `language_model`, `analyses` and `arms` are required. `sources`
    must state each source of `SOURCE_SETTINGS` that some pool holds, and `models` is needed only
    where the run changes a model's settings. A run without `reduction` has no Stage 3A, and
    `reliability` (Stage 3B) needs `reduction`. `source` is the file the config was loaded from,
    and `warnings` are what the driver prints about it.
    """

    name: str
    dataset: Dataset
    grid: Grid
    #: How every part of the run calls the language model.
    language_model: CallSettings
    analyses: AnalysisSettings
    #: Each pool the run fits, by its canonical name, in the order the config lists them.
    arms: dict[str, ArmEntry]
    source: Path
    sources: SourceSettings = SourceSettings()
    #: Each model's settings, where the config states any, by the name a cell records: a Core
    #: model's own settings, or a vendored model's `config_overrides` (`VENDORED_SETTINGS`).
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str | None = None
    reduction: ReductionSettings | None = None
    reliability: ReliabilitySettings | None = None
    #: Consequences of the config that it does not state, for the driver to print.
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        """The config as it would be written back out, after the overrides and with every implied
        value stated.

        A run records this mapping instead of the file on disk, and a resume is compared with it,
        because the `--set` overrides are part of the run's configuration. Model names are recorded
        resolved, so `l1` is written as the model that the dataset's target calls for.
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
        mapping["language_model"] = asdict(self.language_model)
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
        if self.sources.motif_known is not None:
            known = self.sources.motif_known
            sources[MOTIF_KNOWN] = {"collection": known.collection, "strands": known.strands,
                                    "summaries": list(known.summaries),
                                    "p_threshold": known.p_threshold, "normalise": known.normalise,
                                    "min_match_share": known.min_match_share}
        if self.sources.motif_discovered is not None:
            discovered = asdict(self.sources.motif_discovered)
            discovered["summaries"] = list(discovered["summaries"])
            if discovered["annotate_with"] is None:
                del discovered["annotate_with"]
            sources[MOTIF_DISCOVERED] = discovered
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
        """The config as canonical text, so that two configs can be compared as strings instead of
        key by key."""
        return yaml.safe_dump(self.to_mapping(), sort_keys=False, default_flow_style=False)

    def settings_for(self, model: str) -> dict[str, Any]:
        """The settings a Core model is built with in every cell of the run: those the config
        states for it, or none, which leaves the model's defaults."""
        if model in VENDORED_SETTINGS:
            raise ValueError(f"{model} is fitted in its method's environment, which takes its "
                             f"settings as overrides of the method's configuration.")
        return dict(self.models.get(model, {}))

    def effective_model_settings(self) -> dict[str, dict[str, Any]]:
        """Each Core model the run fits, with the value of every setting a config may change for
        it: the config's, or the model's default. A vendored model's overrides are in `models`."""
        fitted = dict.fromkeys(model for model, _ in self.arm_cells())
        return {model: effective_settings(model, self.dataset.target_kind, self.settings_for(model))
                for model in fitted if model not in VENDORED_SETTINGS}

    def arm_cells(self) -> list[tuple[str, str]]:
        """Every `(model, pool)` arm of the grid, in the order the config lists them."""
        return [(model, pool) for pool, entry in self.arms.items() for model in entry.models]

    def reduced_arms(self) -> list[tuple[str, str]]:
        """The arms Stage 3A reduces, or an empty list without a `reduction` section."""
        if self.reduction is None:
            return []
        return [(model, pool) for model, pool in self.arm_cells()
                if model not in self.arms[pool].not_reduced]

    def profiled_arms(self) -> list[tuple[str, str]]:
        """The arms the reliability panel profiles: every reduced arm except those of a black-box
        model and those its entry leaves out. Empty without a `reliability` section."""
        if self.reliability is None:
            return []
        return [(model, pool) for model, pool in self.reduced_arms()
                if not registry.black_box(model) and model not in self.arms[pool].not_profiled]

    def not_reduced(self) -> dict[str, str]:
        """Every arm the reduction leaves out, as `model/pool`, with the reason. Empty without a
        `reduction` section."""
        if self.reduction is None:
            return {}
        return {f"{model}/{pool}": self.arms[pool].not_reduced[model]
                for model, pool in self.arm_cells() if model in self.arms[pool].not_reduced}

    def not_profiled(self) -> dict[str, str]:
        """Every arm the panel leaves out, as `model/pool`, with the reason. Empty without a
        `reliability` section."""
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
        seed, every arm's cells for every label condition and seed of the grid, and each reduced
        arm's three models.

        The reveal checks a run against this list before it opens the held-out split. A cell that
        failed, or an arm that was skipped, is missing from disk but still listed here, so the
        reveal finds the gap.
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
    """Refuse a section that is missing a required key or has a key that nothing uses."""
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
    # `bool` is a subclass of `int` in Python, so without this check `1` and `0` would pass
    # silently as true and false.
    if not isinstance(value, bool):
        raise ConfigError(f"{source}: `{where}.{key}` must be true or false, not {value!r}.")
    return value


def _list_of(section: dict[str, Any], key: str, kind: type, where: str, source: Path) -> list:
    value = section[key]
    if not isinstance(value, list) or not value or any(not isinstance(item, kind) for item in value):
        raise ConfigError(
            f"{source}: `{where}.{key}` must be a non-empty list of {kind.__name__}, not {value!r}.")
    return value


#: What a config may restate about its dataset: the columns and the kind of target. The config
#: recorded beside a run states them so that it describes the run on its own, whereas a hand-written
#: config need not.
STATED = ("sequence_column", "target_column", "weight_column", "covariate_columns", "target_kind")


def _dataset(section: dict[str, Any], source: Path) -> Dataset:
    """The prepared dataset the config names, with its columns read from its `dataset.json`.

    A config may restate a column (as the copy recorded beside a run does, so that the run
    describes itself), but only if it agrees with the file. A config that disagrees with the data
    it names is refused, because one of the two is wrong and nothing downstream could tell which.
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
    """The `grid` section, in which `class_weight` is required for a binary target and refused for
    a continuous one, which has no classes to weight."""
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
    guidance = section["guidance"]
    # Empty text is refused because it could be taken for guidance that was given and lost. A run
    # without guidance states null.
    if guidance is not None and (not isinstance(guidance, str) or not guidance.strip()):
        raise ConfigError(f"{source}: `{where}.guidance` must be the text of the guidance, or null "
                          f"for none, not {guidance!r}.")
    return GeneratedFeatures(
        # Zero sends nothing back to the model.
        repair_rounds=_positive_int(section, "repair_rounds", where, source, floor=0),
        max_undefined_share=float(share),
        max_example_length=_positive_int(section, "max_example_length", where, source),
        # Whole seconds, because IGSR takes its limit in whole seconds.
        max_seconds=_positive_int(section, "max_seconds", where, source),
        guidance=guidance,
    )


def _language_model(section: dict[str, Any], source: Path) -> CallSettings:
    """The `language_model` section, required in full whether or not the run calls a model (as
    `grid.generated_features` is), because a setting left to a client's default would not be
    stated in any record of the run."""
    where = "language_model"
    _keys(section, required=frozenset(CallSettings.__dataclass_fields__), optional=frozenset(),
          where=where, source=source)
    return CallSettings(
        reasoning_effort=_choice(section, "reasoning_effort", REASONING_EFFORTS, where, source),
        # Zero makes each call once.
        retries=_positive_int(section, "retries", where, source, floor=0),
        timeout_seconds=_positive_int(section, "timeout_seconds", where, source),
    )


def _analyses(section: dict[str, Any], source: Path) -> AnalysisSettings:
    """The `analyses` section, required in full, because a table computed with a cutoff or a count
    that the config does not state could not be compared with another run's."""
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
    """Refuse a check arm that the panel does not profile."""
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
    """Refuse a pool that cannot be fitted here: one that does not parse or names its sources out
    of order, one with two written sources, one with covariates the dataset does not have, one
    that puts known motifs beside a written source in a blinded run, or one with positions on
    sequences of different lengths and no window to take them in."""
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
    if dataset.blinded and MOTIF_KNOWN in members and registry.written(pool):
        raise ConfigError(
            f"{source}: `{where}` holds {MOTIF_KNOWN} beside a source a language model writes, and "
            f"the run is blinded. A known motif is named after its factor, and the language model "
            f"would be shown those names. Leave {MOTIF_KNOWN} out of the pool, or the written source.")
    if "positional" in members and dataset.declared_length == "variable" \
            and dataset.positional_window is None:
        raise ConfigError(
            f"{source}: `{where}` reads positions, and {dataset.splits} holds sequences of "
            f"different lengths with no positional window. Declare one at Stage 1 "
            f"(docs/developer/dataset.md), or leave the pool out.")


def _arms(section: dict[str, Any], dataset: Dataset, source: Path
          ) -> tuple[dict[str, ArmEntry], list[str]]:
    """`arms`: each pool, the models fitted on it, and what Stage 3 leaves out of each. Returns the
    entries and warnings about what the arms leave unused."""
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
    """`sources`, checked against the pools: every feature source that some pool holds must be
    stated in full, and no other source may be stated."""
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
    # In the order SOURCE_SETTINGS lists them, so that a config with problems in several sources
    # always has the same one reported first.
    for name in sorted(needed, key=list(SOURCE_SETTINGS).index):
        where = f"sources.{name}"
        settings[name] = _mapping(section[name], where, source)
        _keys(settings[name], required=frozenset(SOURCE_SETTINGS[name]),
              optional=frozenset(SOURCE_OPTIONAL.get(name, ())), where=where, source=source)
    kmer = motif_known = motif_discovered = proposer = feedback_rounds = None
    if MOTIF_KNOWN in settings:
        motif_known = _known_motifs(settings[MOTIF_KNOWN], source)
    if MOTIF_DISCOVERED in settings:
        motif_discovered = _discovered_motifs(settings[MOTIF_DISCOVERED], source)
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
    return SourceSettings(kmer=kmer, motif_known=motif_known, motif_discovered=motif_discovered,
                          proposer=proposer, feedback_rounds=feedback_rounds)


def _scan_settings(section: dict[str, Any], where: str, source: Path) -> dict[str, Any]:
    """The checked settings that both motif sources state about their scans."""
    summaries = _list_of(section, "summaries", str, where, source)
    unknown = sorted(set(summaries) - set(SUMMARIES))
    if unknown or len(set(summaries)) != len(summaries):
        raise ConfigError(f"{source}: `{where}.summaries` must name each of {', '.join(SUMMARIES)} "
                          f"at most once, not {summaries!r}.")
    threshold = section["p_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ConfigError(f"{source}: `{where}.p_threshold` must be a number between 0 and 1, not "
                          f"{threshold!r}.")
    share = section["min_match_share"]
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share < 1:
        raise ConfigError(f"{source}: `{where}.min_match_share` must be a number at least 0 and "
                          f"below 1, not {share!r}.")
    return {"strands": _choice(section, "strands", STRANDS, where, source),
            "summaries": tuple(summaries), "p_threshold": float(threshold),
            "normalise": _choice(section, "normalise", NORMALISATIONS, where, source),
            "min_match_share": float(share)}


def _known_motifs(section: dict[str, Any], source: Path) -> KnownMotifSettings:
    """`sources.motif_known`, with every key required. Whether the collection has been fetched is
    checked when a run starts (`seqmaestro.pipeline.motifs.collection_problems`), because a run
    scans with a copy of the collection kept in the run."""
    where = f"sources.{MOTIF_KNOWN}"
    return KnownMotifSettings(collection=_string(section, "collection", where, source),
                              **_scan_settings(section, where, source))


def _discovered_motifs(section: dict[str, Any], source: Path) -> DiscoveredMotifSettings:
    """`sources.motif_discovered`, with every key except `annotate_with` required and the
    discovery's settings checked as `DiscoverySettings` checks them."""
    where = f"sources.{MOTIF_DISCOVERED}"
    fraction = section["continuous_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 0.5:
        raise ConfigError(f"{source}: `{where}.continuous_fraction` must be a number above 0 and at "
                          f"most 0.5, not {fraction!r}.")
    settings = DiscoveredMotifSettings(
        **_scan_settings(section, where, source),
        directions=_choice(section, "directions", tuple(DIRECTION_SETTINGS), where, source),
        continuous_fraction=float(fraction),
        min_width=_positive_int(section, "min_width", where, source, floor=3),
        max_width=_positive_int(section, "max_width", where, source, floor=3),
        max_motifs=_positive_int(section, "max_motifs", where, source),
        annotate_with=(_string(section, "annotate_with", where, source)
                       if "annotate_with" in section else None))
    if not settings.min_width <= settings.max_width <= STREME_MAX_WIDTH:
        raise ConfigError(f"{source}: `{where}` needs min_width <= max_width <= {STREME_MAX_WIDTH}, "
                          f"not {settings.min_width} and {settings.max_width}.")
    return settings


def _model_settings(section: dict[str, Any], fitted: set[str], target_kind: str, source: Path
                    ) -> dict[str, dict[str, Any]]:
    """`models`: the settings of each model, which some arm must fit, by the name a cell records
    (`l1` resolved for the target's kind, as `arms` resolves it)."""
    named = {name: registry.resolve(name, target_kind) for name in section}
    known = set(VENDORED_SETTINGS) | set(models_for(target_kind))
    unknown = sorted(name for name, model in named.items() if model not in known)
    if unknown:
        raise ConfigError(f"{source}: `models` has no settings for {', '.join(unknown)}. The models "
                          f"with settings are {', '.join(sorted(known))}.")
    unused = sorted(name for name, model in named.items() if model not in fitted)
    if unused:
        raise ConfigError(f"{source}: `models` sets {', '.join(unused)}, which no arm fits. Remove "
                          f"it, or set it to null in a config that extends this one.")
    settings: dict[str, dict[str, Any]] = {}
    for name, raw in section.items():
        model, where = named[name], f"models.{name}"
        if model in settings:
            raise ConfigError(f"{source}: `models` states the settings of {model} twice.")
        entry = _mapping(raw, where, source)
        if model in VENDORED_SETTINGS:
            settings[model] = _vendored_settings(entry, model, where, source)
        else:
            settings[model] = _core_model_settings(entry, model, target_kind, where, source)
    return settings


def _vendored_settings(entry: dict[str, Any], model: str, where: str, source: Path
                       ) -> dict[str, Any]:
    """A vendored model's settings: overrides of its method's configuration, as key=value."""
    _keys(entry, required=frozenset(VENDORED_SETTINGS[model]), optional=frozenset(), where=where,
          source=source)
    overrides = _list_of(entry, "config_overrides", str, where, source)
    malformed = [item for item in overrides if "=" not in item]
    if malformed:
        raise ConfigError(f"{source}: `{where}.config_overrides` needs key=value: "
                          f"{', '.join(malformed)}.")
    return {"config_overrides": overrides}


def _core_model_settings(entry: dict[str, Any], model: str, target_kind: str, where: str,
                         source: Path) -> dict[str, Any]:
    """A Core model's settings: some of those its class lists, each a positive number of the type
    it takes. The model is built with them once, so that a value it refuses is refused here, before
    any work is done, and not in every cell."""
    allowed = core_settings(model, target_kind)
    if not allowed:
        raise ConfigError(f"{source}: `{where}`: {model} takes no settings.")
    unknown = sorted(set(entry) - set(allowed))
    if unknown:
        raise ConfigError(f"{source}: `{where}` has no setting {', '.join(unknown)}. A config may "
                          f"change {', '.join(allowed)}.")
    if not entry:
        raise ConfigError(f"{source}: `{where}` states no setting. Remove it, or state one of "
                          f"{', '.join(allowed)}.")
    values: dict[str, Any] = {}
    for key, value in entry.items():
        kind = allowed[key]
        number = isinstance(value, int if kind is int else (int, float)) and not isinstance(value, bool)
        if not number or value <= 0:
            wanted = "a positive whole number" if kind is int else "a positive number"
            raise ConfigError(f"{source}: `{where}.{key}` must be {wanted}, not {value!r}.")
        values[key] = kind(value)
    try:
        make_model(model, target_kind=target_kind, **values)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{source}: `{where}`: {error}") from error
    return values


def _apply_override(document: dict[str, Any], assignment: str, source: Path) -> None:
    """Apply one `section.key=value` override in place, parsing the value as YAML.

    Overrides let the whole pipeline be exercised cheaply (a tiny split, one seed, no bootstrap)
    without a second config file that would drift from the real one. They are applied before
    validation, so an override is checked as a value written in the file would be, and the run
    records them.

    An intermediate section must already exist: `--set grid.n_bootstrap=0` edits the grid, whereas
    the typo `--set gird.n_bootstrap=0` would otherwise create a section that nothing uses.
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
    # silently doing nothing would leave those arms running without any sign of it.
    if value is None:
        if key not in target:
            raise ConfigError(
                f"{source}: --set {assignment}: there is no `{path}` to remove. "
                f"Here: {', '.join(sorted(target)) or 'nothing'}.")
        del target[key]
    else:
        target[key] = value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` into `base`: mappings merge, `null` removes a key, and any other value
    replaces the base value.

    A list replaces the base list instead of being appended to it, which is what a reader expects
    of `models` or `config_overrides`: if lists were appended, an extending file could never
    shorten one.
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
    """Read a config and merge it into whatever it extends, innermost first.

    `extends` is a path relative to the extending file (so that a config directory can be moved
    whole), or a list of such paths. A list is merged in its order, each file resolved first, and
    the extending file is merged last, so a later file's setting replaces an earlier one's. The key
    is consumed here, so a resolved config has no `extends` and the copy recorded beside a run is a
    complete statement on its own.
    """
    here = source.resolve()
    if here in seen:
        chain = " -> ".join(path.name for path in (*seen, here))
        raise ConfigError(f"{source}: `extends` is circular: {chain}.")
    document = _read(source)
    parents = document.pop("extends", None)
    if parents is None:
        return document
    if isinstance(parents, str):
        parents = [parents]
    if (not isinstance(parents, list) or not parents
            or any(not isinstance(parent, str) or not parent for parent in parents)):
        raise ConfigError(f"{source}: `extends` must be a path relative to {source.parent}, or a "
                          f"list of them.")
    folded: dict[str, Any] = {}
    for parent in parents:
        folded = _merge(folded, resolve(source.parent / parent, (*seen, here)))
    return _merge(folded, document)


def load(path: Path | str, overrides: tuple[str, ...] = ()) -> ExperimentConfig:
    """Read, extend, override and validate a config, or raise `ConfigError` naming the bad key.

    A config may state the `version` of the shape it is written in, and a recorded one always
    does. A config that states a version other than the one this code reads is refused.
    """
    source = Path(path)
    document = resolve(source)
    for assignment in overrides:
        _apply_override(document, assignment, source)

    version = document.pop("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigError(f"{source} is written as config version {version!r}, and this code "
                          f"reads version {CONFIG_VERSION}.")
    _keys(document, required=frozenset({"name", "dataset", "grid", "language_model", "analyses",
                                        "arms"}),
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
                             {model for entry in arms.values() for model in entry.models},
                             dataset.target_kind, source)
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
        language_model=_language_model(
            _mapping(document["language_model"], "language_model", source), source),
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

    The driver records a config with its version. A file that states no version is refused,
    because it is not a record of a run and reading it as the current version would be a guess.
    """
    path = Path(run_dir) / CONFIG_FILE
    if "version" not in _read(path):
        raise ConfigError(f"{path} states no `version`, so it is not a config the driver recorded "
                          f"for a run. A stage takes its settings from a recorded config alone.")
    return load(path)


def main(argv: list[str] | None = None) -> int:
    """Validate a config and print it as it would be recorded, without starting a run."""
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
