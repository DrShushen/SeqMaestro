"""Where internal names become the names a reader sees.

Every table, figure axis and sentence in a generated report takes its names from here, and so does
the spreadsheet of a run's features (`seqmaestro.report.spreadsheet`), so the report and the
spreadsheet a collaborator is handed use one vocabulary. Repository vocabulary must never reach the
reader. Making this module the only route from an internal name to a label enforces that, whereas
leaving each writer to remember the rename would not.

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
    MOTIF_DISCOVERED,
    MOTIF_KNOWN,
    WRITTEN_MEMBERS,
    parse_pool,
)
from seqmaestro.report.markdown import series

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
#: its members' names together, so `positional+kmer` is "positions + k-mers" and no pool needs a
#: separate entry. Which sources a language model writes is declared once, in
#: `seqmaestro.features.WRITTEN_MEMBERS`. The joint source is written inside the model's fit, so
#: what it is depends on the model, and its entry describes it per model (`by_model`).
SOURCES = {
    "positional": {
        "short": "positions",
        "long": "single-base position indicators",
        "gloss": "one yes/no column per (position, base) pair over the window",
    },
    "kmer": {
        "short": "k-mers",
        "long": "k-mer counts",
        "gloss": "one column for every short substring (k-mer) the run kept, holding how often it occurs",
    },
    MOTIF_KNOWN: {
        "short": "known motifs",
        "long": "matches to the motifs of a published collection",
        "gloss": "two columns for every motif of a published collection, such as JASPAR's: the "
                 "strength of its best match in the sequence, and the number of places it matches",
    },
    MOTIF_DISCOVERED: {
        "short": "discovered motifs",
        "long": "matches to motifs discovered in the training sequences",
        "gloss": "two columns for every motif a discovery tool (STREME) found separating the "
                 "classes, or the two ends of the target, in the sequences the model was fitted "
                 "on: the strength of its best match, and the number of places it matches",
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
        """The short name that labels a table row, such as `EBM · written`."""
        return f"{self.model_short} · {self.source_short}"

    @property
    def is_interpretable(self) -> bool:
        return model_interpretable(self.model_key)

    @property
    def writes_features(self) -> bool:
        return any(key in WRITTEN_MEMBERS for key in self.source_keys)


#: Each provider a run can call (`seqmaestro.llm.PROVIDERS`), as a reader knows it.
PROVIDERS = {"azure": "Azure OpenAI", "azure_foundry": "Azure AI Foundry",
             "openai": "the OpenAI API", "claude": "the Claude API"}

#: What a part of a run used the language model for, by the part: an arm's written source (the
#: joint source by the model that writes it, `written_role`), the judge, and the grouping of written
#: features into ideas.
CALLED_FOR = {
    LLM_GENERATED: "wrote features in one round",
    LLM_ITERATIVE: "wrote features over several rounds, each shown a fit of the model",
    f"{LLM_JOINT}/{arms.DEFT_OR_CART}": "wrote candidate features at each split of the decision tree",
    f"{LLM_JOINT}/{arms.IGSR}": "wrote the terms of IGSR's linear model",
    "judge": "rated how meaningful the features look",
    "recurrence": "grouped the written features into ideas, to count those found again",
}


def written_role(arm: str) -> str | None:
    """What an arm's language model was called for, as a key of `CALLED_FOR`, or None for an arm
    on no written source. A pool that holds the iterative source is named for it, because the
    iterative source's first round is the one-round source's request."""
    configuration = parse(arm)
    members = configuration.source_keys
    if LLM_ITERATIVE in members:
        return LLM_ITERATIVE
    if LLM_GENERATED in members:
        return LLM_GENERATED
    if LLM_JOINT in members:
        return f"{LLM_JOINT}/{configuration.model_key}"
    return None


def language_model(record: dict) -> str:
    """The model a record of the calls names (`seqmaestro.llm.describe_calls`), as a reader sees
    it: its name and how it was reached. An Azure OpenAI deployment is named by the model family the
    run declared for it, with the deployment's name beside it. A record that names no provider is a
    stand-in for a model, as in a test, and is shown by its model string."""
    name = str(record["model"]).split("/", 1)[-1]
    provider = record.get("provider")
    if provider is None:
        return name
    if provider not in PROVIDERS:
        raise KeyError(f"No reader-facing name for the provider {provider!r}. Add one to "
                       f"naming.PROVIDERS.")
    if provider == "azure":
        family = str(record["base_model"]).split("/", 1)[-1]
        return f"{family} (the deployment {name}, on {PROVIDERS[provider]})"
    if provider == "azure_foundry":
        return f"{name} (a deployment on {PROVIDERS[provider]})"
    return f"{name} (through {PROVIDERS[provider]})"


def reasoning_effort(record: dict) -> str:
    """The reasoning effort a record says each call stated, or "none stated" where the calls stated
    none. Claude's calls state none, which leaves its extended thinking off."""
    effort = record.get("reasoning_effort")
    return "none stated" if effort is None else str(effort)


def agent_did(stages: list[str]) -> str:
    """What an agent did, from the stages it recorded (`seqmaestro.pipeline.agents.STAGES`)."""
    numbered = [stage for stage in stages if stage != "report"]
    done = []
    if numbered:
        done.append(f"carried out Stage{'s' if len(numbered) > 1 else ''} {series(numbered)}")
    if "report" in stages:
        done.append("wrote this report")
    return ", and ".join(done)


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
    """The abbreviation key every report includes once, near the front, for a run on a
    `target_kind` target: the models the run's arms (`run_arms`) fitted and the feature sources
    their pools held, and nothing else. A report then does not name a model or source it has no
    result for, such as the covariates of a dataset without any, or a source the run's config left
    out."""
    configurations = [parse(arm) for arm in run_arms]
    fitted = {configuration.model_key for configuration in configurations}
    rows = []
    for key, entry in MODELS.items():
        if key not in fitted or target_kind not in arms.spec(key).target_kinds:
            continue
        rows.append({"kind": "model", "short": entry["short"], "name": entry["long"],
                     "note": "" if model_interpretable(key)
                     else "reference only, not interpreted as a statement about features"})
    # A source whose meaning depends on the model is listed once for each model the run hands it.
    handed = {(key, configuration.model_key) for configuration in configurations
              for key in configuration.source_keys}
    for key, entry in SOURCES.items():
        if not any(source == key for source, _ in handed):
            continue
        models = sorted(model for source, model in handed
                        if source == key and model in entry.get("by_model", {}))
        for model in models or [None]:
            described = source_entry(key, model)
            rows.append({"kind": "feature source", "short": described["short"],
                         "name": described["long"], "note": described["gloss"]})
    return rows
