"""Where internal names become the names a reader sees.

Every table, figure axis and sentence in a generated report takes its names from here, and so does
the spreadsheet of a run's features (`seqmaestro.report.spreadsheet`), so the report and the
spreadsheet a collaborator is handed use one vocabulary. Repository vocabulary must never
reach the reader. Making this module the only route from an internal name to a label enforces that,
where leaving each writer to remember the rename would not.

**An unknown name raises.** Adding a model or feature source means deciding how a reader sees it,
and falling back to its internal name would put repository vocabulary in front of the reader.

These names are the same for every dataset. They say what a model class or a feature source is, and
nothing here may mention a dataset, a column or a biological entity.

Inside a run, an arm is `model/pool`, as its cells record it. A table or file named after an arm
uses `model__pool` (`slug`), because `/` cannot appear in a file name.
"""

from __future__ import annotations

from dataclasses import dataclass

from seqmaestro import arms
from seqmaestro.features import (
    COVARIATES,
    LLM_GENERATED,
    LLM_ITERATIVE,
    LLM_JOINT,
    WRITTEN_MEMBERS,
    parse_pool,
)

#: The models that consume the features, by the name a cell records, in the order a report lists
#: them. Whether a model is interpreted as a statement about features, and which kinds of target it
#: fits, are declared once, in `seqmaestro.arms`, so neither is repeated here.
MODELS = {
    "ebm": {"short": "EBM", "long": "explainable boosting machine", "family": "additive"},
    "l1_logistic": {"short": "L1 Logistic Regression", "long": "L1-penalised logistic regression",
                    "family": "linear"},
    "l1_linear": {"short": "L1 Linear Regression", "long": "L1-penalised linear regression (the Lasso)",
                  "family": "linear"},
    arms.DEFT_OR_CART: {"short": "Decision Tree", "long": "decision tree", "family": "tree"},
    arms.IGSR: {"short": "IGSR", "long": "influence-guided symbolic regression, a linear model over "
                                         "terms a language model wrote", "family": "linear"},
    "xgboost": {"short": "XGBoost", "long": "gradient-boosted trees (XGBoost)", "family": "boosted trees"},
    "pysr": {"short": "PySR", "long": "symbolic regression (PySR): one equation over the features, "
                                     "found by an evolutionary search", "family": "symbolic"},
}

#: Where the features came from, one entry per member of a pool. A pool is named for a reader by
#: its members' names together, so `positional+kmer` is "positions + motifs" and no pool needs an
#: entry of its own. Which sources a language model writes is declared once, in
#: `seqmaestro.features.WRITTEN_MEMBERS`. The joint source is written inside the model's fit, so
#: what it is depends on the model, and its entry says so per model (`by_model`).
SOURCES = {
    "positional": {
        "short": "positions",
        "long": "single-base position indicators",
        "gloss": "one yes/no column per (position, base) pair over the window",
    },
    "kmer": {
        "short": "motifs",
        "long": "short-motif counts",
        "gloss": "one column for every short substring the run kept, holding how often it occurs",
    },
    LLM_GENERATED: {
        "short": "written",
        "long": "language-model-written features",
        "gloss": "each feature is a short function a language model wrote and named, computed on the sequence",
    },
    LLM_ITERATIVE: {
        "short": "written, iterative",
        "long": "language-model-written features, revised over several rounds",
        "gloss": "as written features, but the proposer was shown the fitted model's behaviour and asked again",
    },
    LLM_JOINT: {
        "short": "written jointly",
        "long": "language-model-written features, written inside the model's fit",
        "gloss": "functions a language model wrote while the model was being fitted, conditioned on "
                 "that fit",
        "by_model": {
            arms.DEFT_OR_CART: {
                "short": "written per split",
                "long": "language-model-written features, proposed separately at each split of the tree",
                "gloss": "a tree that asks a language model for candidate features at each node, "
                         "conditioned on the rows reaching it",
            },
            arms.IGSR: {
                "short": "written terms",
                "long": "language-model-written functions of the sequence, kept by their influence "
                        "inside a linear fit",
                "gloss": "each term is a short function a language model wrote over the whole "
                         "sequence; a term stays in the linear fit only if removing it raises the "
                         "error on the validation sequences",
            },
        },
    },
    COVARIATES: {
        "short": "covariates",
        "long": "measurements supplied with each sequence, used as they are",
        "gloss": "one column per measurement the dataset supplies beside each sequence, unchanged; "
                 "the language models that wrote features were shown them too",
    },
}


def source_entry(key: str, model_key: str | None = None) -> dict:
    """A source as a reader sees it, for the model it was handed to where that changes it."""
    entry = SOURCES[key]
    return {**entry, **entry.get("by_model", {}).get(model_key, {})}


#: The reference line of a report's performance figures and tables, in a reader's words. Its model
#: is the one its cells record.
BASELINE = {
    "short": "base composition",
    "measures": "the fraction of each base over the window",
}

#: Each metric a report names, in a reader's words.
METRICS = {"auroc": "AUROC", "auprc": "AUPRC", "r2": "R²", "spearman_rho": "Spearman's rho"}

#: What higher values of a feature do to a continuous target, as a report prints each direction. A
#: binary target's directions print as the class they argue for, which the dataset names.
CONTINUOUS_DIRECTIONS = {"increases_target": "raise the target", "decreases_target": "lower the target"}

#: How a report prints a feature whose effect changes sign along its range, for either kind.
BOTH_WAYS = "both ways"


def model_interpretable(model_key: str) -> bool:
    """Whether a model is interpreted as a statement about features: one that is not a black box."""
    return not arms.black_box(model_key)


@dataclass(frozen=True)
class Configuration:
    """One arm of the grid, named for a reader.

    `arm` is the internal identifier and appears nowhere in a report. It is kept so a number can be
    traced back to the file it came from.
    """

    arm: str
    model_key: str
    #: The arm's feature sources, one per member of its pool, in the pool's order.
    source_keys: tuple[str, ...]

    @property
    def model_short(self) -> str:
        return MODELS[self.model_key]["short"]

    @property
    def model_long(self) -> str:
        return MODELS[self.model_key]["long"]

    @property
    def source_key(self) -> str:
        """The sources as one key, for grouping arms by where their features came from."""
        return "+".join(self.source_keys)

    @property
    def source_short(self) -> str:
        return " + ".join(source_entry(key, self.model_key)["short"] for key in self.source_keys)

    @property
    def source_long(self) -> str:
        longs = [source_entry(key, self.model_key)["long"] for key in self.source_keys]
        return longs[0] if len(longs) == 1 else f"{', '.join(longs[:-1])} and {longs[-1]}"

    @property
    def label(self) -> str:
        """The short name a table row carries, e.g. `EBM · written`."""
        return f"{self.model_short} · {self.source_short}"

    @property
    def is_interpretable(self) -> bool:
        return model_interpretable(self.model_key)

    @property
    def writes_features(self) -> bool:
        return any(key in WRITTEN_MEMBERS for key in self.source_keys)


def slug(arm: str) -> str:
    """An arm, given as `model/pool`, as the name of a table or a file."""
    return arms.arm_slug(*arms.split_arm(arm))


def parse(arm: str) -> Configuration:
    """Turn `ebm/kmer+llm_generated_iterative` into a named configuration: the model, and the
    sources of its pool."""
    model, pool = arms.split_arm(arm)
    if model not in MODELS:
        raise KeyError(f"No reader-facing name for model {model!r}. Add one to naming.MODELS.")
    try:
        members = parse_pool(pool)
    except ValueError as error:
        raise KeyError(f"No reader-facing name for feature source {pool!r}: {error}") from error
    unknown = [key for key in members if key not in SOURCES]
    if unknown:
        raise KeyError(f"No reader-facing name for feature source(s) {unknown}. Add them to "
                       f"naming.SOURCES.")
    return Configuration(arm=arm, model_key=model, source_keys=tuple(members))


def label(arm: str) -> str:
    return parse(arm).label


def key_table(run_arms, target_kind: str) -> list[dict[str, str]]:
    """The abbreviation key every report carries once, near the front, for a run on a
    `target_kind` target. The covariates are listed only where one of the run's arms (`run_arms`)
    was given them, so a report on a dataset without any does not mention them."""
    uses_covariates = any(COVARIATES in parse(arm).source_keys for arm in run_arms)
    rows = []
    for key, entry in MODELS.items():
        if target_kind not in arms.spec(key).target_kinds:
            continue
        rows.append({"kind": "model", "short": entry["short"], "name": entry["long"],
                     "note": "" if model_interpretable(key)
                     else "reference only, not interpreted as a statement about features"})
    # A source whose meaning depends on the model is listed once for each model the run hands it.
    handed = {(key, configuration.model_key) for configuration in map(parse, run_arms)
              for key in configuration.source_keys}
    for key, entry in SOURCES.items():
        if key == COVARIATES and not uses_covariates:
            continue
        models = sorted(model for source, model in handed
                        if source == key and model in entry.get("by_model", {}))
        for model in models or [None]:
            described = source_entry(key, model)
            rows.append({"kind": "feature source", "short": described["short"],
                         "name": described["long"], "note": described["gloss"]})
    return rows
