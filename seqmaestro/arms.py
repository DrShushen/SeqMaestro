"""What an arm can be: which model can be fitted on which pool of sources, and how.

An arm is one model fitted on one pool, a set of sources handed to the model together
(`seqmaestro.features`). This module declares what every model can do, once, so that the config,
the driver and the report work from the same facts, and a combination a model cannot take is
refused when the config is read, before anything is spent.

**The models.** The four in `seqmaestro.models` (`l1_logistic` or `l1_linear`, `ebm`, `xgboost` and
`pysr`) are fitted in Core, PySR's search excepted, on a pool of any sources but the joint one. Two
more are fitted in the environments of the methods they come from:
- `deft_or_cart` is DEFT's decision tree. At every node its candidates are the positional ones it
  derives from the sequence, which its pool must therefore name, and one for each other column it
  is handed, such as each k-mer count. It is plain CART on a pool without `llm_generated_joint`,
  and DEFT on one with it, where a language model writes further candidates at each node;
- `igsr` is IGSR's linear model over terms a language model writes as functions of each sequence,
  each kept by its influence on the fit. Its pool is `llm_generated_joint`, with or without the
  covariates.

**A black box** is a model whose fit cannot be interpreted feature by feature. Its arms are
reduced, since it ranks the columns it used, and the reliability panel leaves them out with
`BLACK_BOX_REASON` recorded.

**A pool holds at most one written source**, so that each arm stands for one way of writing
features.

**`l1`** in a config names the sparse linear model for the dataset's kind of target: `l1_logistic`
for a binary one, `l1_linear` for a continuous one (`resolve`).

**An arm is spelled `model/pool`**, as a config lists it, a log prints it and `--arms` takes it
(`arm_name`, `split_arm`). In the name of a file or a table it is `model__pool` (`arm_slug`), since
a file name cannot hold the slash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seqmaestro.features import (
    COVARIATES,
    LLM_GENERATED,
    LLM_ITERATIVE,
    LLM_JOINT,
    WRITTEN_MEMBERS,
    parse_pool,
    pool_name,
)
from seqmaestro.models import MODEL_NAMES, MODELS, environment, interpretable, reproducible
from seqmaestro.target import BINARY, CONTINUOUS

POSITIONAL, KMER = "positional", "kmer"

#: An arm: a model's name and a pool's name.
Arm = tuple[str, str]

#: The two models fitted in their methods' environments, and the name a config gives the sparse
#: linear model of either kind.
DEFT_OR_CART = "deft_or_cart"
IGSR = "igsr"
L1 = "l1"

#: The environment of SeqMaestro's own code, and those DEFT's tree and IGSR run in
#: (ENVIRONMENTS.md).
CORE_ENVIRONMENT = "SeqMaestro_Core"
DEFT_ENVIRONMENT = "SeqMaestro_Method_deft"
IGSR_ENVIRONMENT = "SeqMaestro_Method_igsr"

#: The kinds of arm, by how a cell of one is fitted (`kind`): a Core model on a fixed pool, whose
#: columns follow from the rows by rule; a Core model on a pool the proposer writes into; DEFT's
#: tree; and IGSR.
FIXED, PROPOSER, DEFT, IGSR_KIND = "fixed", "proposer", "deft", "igsr"

#: Why the reliability panel leaves a black-box model's arms out, as a run records it.
BLACK_BOX_REASON = "its model is read through its predictions, not its features"


@dataclass(frozen=True)
class ModelSpec:
    """What one model can do."""

    name: str
    target_kinds: tuple[str, ...]
    #: Whether it fits with per-row weights.
    weights: bool
    black_box: bool
    #: Whether refitting on the same rows, weights and seed gives the same model.
    reproducible: bool
    #: The conda environment besides Core that fitting it runs code in, or None.
    environment: str | None
    #: Whether its fitted model can be used in that environment alone, as a vendored method's can.
    #: Every later stage that predicts with it or reads it then needs the environment too. A model
    #: whose search alone runs elsewhere, as PySR's does, is saved in a form Core loads.
    vendored: bool
    #: The sources it can be handed.
    accepts: frozenset[str]
    #: The sources its pool must hold, each with why.
    requires: dict[str, str] = field(default_factory=dict)
    #: Sources it could take in principle that no code hands it yet, each with why.
    not_built: dict[str, str] = field(default_factory=dict)
    #: Sources it can be handed one at a time that no code hands it together yet, each set with why.
    not_built_together: dict[frozenset[str], str] = field(default_factory=dict)


def _core(name: str) -> ModelSpec:
    """One of the models in `seqmaestro.models`, as its classes declare it."""
    return ModelSpec(
        name, target_kinds=tuple(kind for kind in (BINARY, CONTINUOUS) if name in MODELS[kind]),
        weights=True, black_box=not interpretable(name), reproducible=reproducible(name),
        environment=environment(name), vendored=False,
        accepts=frozenset({POSITIONAL, KMER, COVARIATES, LLM_GENERATED, LLM_ITERATIVE}))


#: Every model, by the name a cell records.
MODEL_SPECS: dict[str, ModelSpec] = {
    **{name: _core(name) for name in MODEL_NAMES},
    DEFT_OR_CART: ModelSpec(
        DEFT_OR_CART, target_kinds=(BINARY, CONTINUOUS), weights=True, black_box=False,
        reproducible=True, environment=DEFT_ENVIRONMENT, vendored=True,
        accepts=frozenset({POSITIONAL, KMER, COVARIATES, LLM_JOINT}),
        requires={POSITIONAL: "DEFT derives positional candidates from the sequence, which it is "
                              "always handed"},
        not_built={
            LLM_GENERATED: "a DEFT arm has no proposer to write a proposal for the tree",
            LLM_ITERATIVE: "DEFT's tree cannot report its fit to the proposer, which the iterative "
                           "source needs",
        },
        not_built_together={
            frozenset({KMER, LLM_JOINT}): "DEFT's prompts show every column it is handed, and a "
                                          "vocabulary this wide would swamp them",
        }),
    IGSR: ModelSpec(
        IGSR, target_kinds=(CONTINUOUS,), weights=False, black_box=False,
        # The search calls a language model; the model refitted on its terms is least squares.
        reproducible=True, environment=IGSR_ENVIRONMENT, vendored=True,
        accepts=frozenset({LLM_JOINT, COVARIATES}),
        requires={LLM_JOINT: "IGSR's model is fitted on the terms its search writes"},
        not_built={
            POSITIONAL: "IGSR's prompts show every column it is handed, and a vocabulary this "
                        "wide would swamp them",
            KMER: "IGSR's prompts show every column it is handed, and a vocabulary this wide would "
                  "swamp them",
        }),
}


def arm_name(model: str, pool: str) -> str:
    """An arm as a config, a log and `--arms` spell it."""
    return f"{model}/{pool}"


def split_arm(name: str) -> Arm:
    """The model and the pool of an arm spelled `model/pool`."""
    model, found, pool = name.partition("/")
    if not found or not model or not pool or "/" in pool:
        raise ValueError(f"{name!r} is not an arm: an arm is spelled model/pool.")
    return model, pool


def arm_slug(model: str, pool: str) -> str:
    """An arm inside the name of a file or a table."""
    return f"{model}__{pool}"


def resolve(name: str, target_kind: str) -> str:
    """A model's name as a cell records it, from the name a config gives, `l1` included."""
    if name == L1:
        return "l1_logistic" if target_kind == BINARY else "l1_linear"
    return name


def spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(sorted(MODEL_SPECS))}, "
                         f"or {L1}.") from None


def black_box(name: str) -> bool:
    return spec(name).black_box


def written(pool: str) -> tuple[str, ...]:
    """The pool's sources that a language model writes."""
    return tuple(member for member in parse_pool(pool) if member in WRITTEN_MEMBERS)


def pool_problems(pool: str) -> list[str]:
    """Why no model can be fitted on `pool`, or nothing."""
    sources = written(pool)
    if len(sources) > 1:
        return [f"{pool} holds {' and '.join(sources)}; a pool holds at most one written source"]
    return []


def problems(model: str, pool: str, target_kind: str, weighted: bool) -> list[str]:
    """Why `model` cannot be fitted on `pool` for a dataset of this kind of target, or nothing.
    `weighted` says whether the dataset weights its rows."""
    declared = spec(model)
    found = []
    if target_kind not in declared.target_kinds:
        found.append(f"{model} fits a {' or '.join(declared.target_kinds)} target, and this "
                     f"dataset's is {target_kind}")
    if weighted and not declared.weights:
        found.append(f"{model} fits without per-row weights, and this dataset weights its rows")
    return found + source_problems(model, pool)


def source_problems(model: str, pool: str) -> list[str]:
    """Why `model` cannot be handed the sources of `pool`, whatever the dataset, or nothing."""
    declared = spec(model)
    members = parse_pool(pool)
    found = []
    for member in members:
        if member in declared.accepts:
            continue
        why = declared.not_built.get(member)
        if why is not None:
            found.append(f"{model} cannot be handed {member} yet: {why}")
        elif member == LLM_JOINT:
            found.append(f"{model} cannot be handed {member}, which only {DEFT_OR_CART}'s and "
                         f"{IGSR}'s fits write")
        else:
            found.append(f"{model} cannot be handed {member}")
    found += [f"{model} cannot be handed {pool_name(together)} together yet: {why}"
              for together, why in declared.not_built_together.items()
              if together <= set(members)]
    found += [f"{model} needs {member} in its pool: {why}"
              for member, why in declared.requires.items() if member not in members]
    return found


def kind(model: str, pool: str) -> str:
    """The kind of arm `model` on `pool` is, which decides how its cells are fitted."""
    if model == DEFT_OR_CART:
        return DEFT
    if model == IGSR:
        return IGSR_KIND
    return PROPOSER if {LLM_GENERATED, LLM_ITERATIVE} & set(parse_pool(pool)) else FIXED


def calls_language_model(pool: str) -> bool:
    """Whether fitting an arm on `pool` calls a language model: it holds a written source."""
    return bool(written(pool))


def retries(pool: str) -> bool:
    """Whether a failed process of an arm on `pool` is started once more. Not where its language
    model works inside the method's fit, the joint source, since a second start would pay again for
    every call the first made."""
    return LLM_JOINT not in parse_pool(pool)
