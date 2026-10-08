"""Featurisation: rows in (sequences, and any covariates), a named numeric matrix out.

```python
from seqmaestro.features import make_featuriser

featuriser = make_featuriser("positional+kmer", k_range=(1, 6), kmer_normalise="counts")
X_train = featuriser.fit_transform(dataset.inputs(train))
X_val = featuriser.transform(dataset.inputs(val))       # fitted on train only
```

A featuriser reads rows (`Inputs`: the sequences, and any covariates the dataset declares), which
`dataset.inputs` builds from a split frame using the columns `dataset.json` records. A bare
sequence column is accepted too, as rows with no covariates.

**A pool is a set of members**, which the config and the report call sources. `positional` and
`kmer` are members, and `positional+kmer` is the pool holding both. `covariates` is the dataset's
covariates as they are. Three members are features a language model writes: `llm_generated` in one
round, `llm_generated_iterative` over several rounds shown a fit of the arm's model, and
`llm_generated_joint` jointly with a model's fit, by a procedure only that model runs (DEFT's tree,
IGSR). A pool is named by joining its members with `+` in the canonical order below, so one name
always denotes one set. Which model can be handed which pool is declared in `seqmaestro.arms`.

**A pool's members are what the arm's model is handed and what its generators may read.** An arm
whose pool holds `covariates` hands them to its model and shows them to its proposer, which may
write features that use them. An arm whose pool does not is given rows without them
(`seqmaestro.dataset.Dataset.columns_for`), so neither its model nor its proposer sees them.

Anything the write-up plans next (gapped and mismatch *k*-mers, PWM scans against JASPAR or
HOCOMOCO, STREME motifs, positional concentration) would be another member: a `Featuriser` that
composes with these and emits the same `FeatureRecord`s.

**A pool holds only tabular columns.** It is the set of columns a model is handed, and the basis
for a `selected` provenance. The raw sequence is available separately to arms that *generate*, and
is never named in a pool (see `seqmaestro.features.base.PROVENANCES`). Only `llm_generated` is built
here: the iterative member is built by Stage 2 around the arm's model
(`seqmaestro.pipeline.stage2.ProposerArm.featuriser`), and the joint one by its model, on its side
of the environment boundary.

`CompositionFeaturiser` is exported beside them but is not a pool member. It is the featurisation of
the composition baseline, a reference line every run draws, and no arm is handed it.
"""

from __future__ import annotations

from collections.abc import Mapping

from seqmaestro.features.base import FeatureRecord, Featuriser
from seqmaestro.features.composition import CompositionFeaturiser
from seqmaestro.features.covariates import CovariateFeaturiser
from seqmaestro.features.kmer import KmerFeaturiser
from seqmaestro.features.positional import PositionalFeaturiser, PositionalWindow

#: The member that holds the dataset's covariates.
COVARIATES = "covariates"

#: The members a language model writes: in one round, over several rounds each shown a fit of the
#: arm's model, and jointly with a model's fit by a procedure only that model runs.
LLM_GENERATED = "llm_generated"
LLM_ITERATIVE = "llm_generated_iterative"
LLM_JOINT = "llm_generated_joint"
WRITTEN_MEMBERS = (LLM_GENERATED, LLM_ITERATIVE, LLM_JOINT)

#: The members a pool may contain, in canonical order. Pool names are built in this order and not
#: alphabetically, because alphabetical order would turn `positional+kmer` into `kmer+positional`.
#: `covariates` is last, so a pool with them is named by the pool without them and `+covariates`.
POOL_MEMBERS = ("positional", "kmer", LLM_GENERATED, LLM_ITERATIVE, LLM_JOINT, COVARIATES)

#: The pools of members enumerated from the sequence.
_ENUMERATED_POOLS = ("positional", "kmer", "positional+kmer")

#: The pools an arm can be handed without a proposer: each enumerated pool with and without the
#: covariates, and the covariates on their own.
FIXED_POOLS = (*_ENUMERATED_POOLS, *(f"{pool}+{COVARIATES}" for pool in _ENUMERATED_POOLS),
               COVARIATES)

__all__ = [
    "COVARIATES",
    "FIXED_POOLS",
    "LLM_GENERATED",
    "LLM_ITERATIVE",
    "LLM_JOINT",
    "WRITTEN_MEMBERS",
    "CompositionFeaturiser",
    "CovariateFeaturiser",
    "FeatureRecord",
    "KmerFeaturiser",
    "PositionalFeaturiser",
    "PositionalWindow",
    "make_featuriser",
    "parse_pool",
    "pool_name",
    "with_covariates",
]


def parse_pool(name: str) -> tuple[str, ...]:
    """A pool name to its members, in canonical order, refusing anything unknown or repeated."""
    members = [part.strip() for part in name.split("+") if part.strip()]
    unknown = [member for member in members if member not in POOL_MEMBERS]
    if unknown:
        raise ValueError(f"Unknown pool member(s) {unknown} in {name!r}; "
                         f"expected from {POOL_MEMBERS}.")
    if len(set(members)) != len(members):
        raise ValueError(f"Repeated member in pool {name!r}.")
    if not members:
        raise ValueError("An empty pool has no members; name at least one.")
    return tuple(member for member in POOL_MEMBERS if member in members)


def pool_name(members) -> str:
    """The canonical name for a set of members, so each set has one name and one directory."""
    return "+".join(parse_pool("+".join(members)))


def with_covariates(pool: str) -> str:
    """The pool with the covariates added: `positional` becomes `positional+covariates`."""
    return pool_name((*parse_pool(pool), COVARIATES))


def make_featuriser(pool: str, k_range: tuple[int, int] | None = None, proposer=None,
                    covariates: Mapping[str, str] | None = None,
                    window: PositionalWindow | None = None,
                    kmer_normalise: str | None = None) -> Featuriser:
    """Build the featuriser for a pool, composing its members in canonical order.

    `k_range` and `kmer_normalise` say how the `kmer` member counts, and a pool that holds it needs
    both: how k-mers are counted is a setting of a run, stated in its config, so nothing here fills
    it in. `window` (the dataset's declared positional window, `Dataset.positional_window`) is a
    setting of the `positional` member. `proposer` is required by, and only by, `llm_generated`, and
    `covariates` (each covariate's declared description, `Dataset.covariates`) by, and only by, the
    `covariates` member. Passing either to a pool that does not call for it is refused rather than
    ignored, since it means the caller expected features this pool will not contain.

    The iterative and joint members are refused: the first is built by
    `seqmaestro.pipeline.stage2.ProposerArm.featuriser` around the arm's model
    (`seqmaestro.proposer.iterative`), and the second by its model, in that model's environment.
    """
    members = parse_pool(pool)
    if LLM_ITERATIVE in members:
        raise ValueError(f"Pool {pool!r} holds {LLM_ITERATIVE}, which a proposer arm builds "
                         f"around its model; build the pool with {LLM_GENERATED} in its place "
                         f"and wrap it.")
    if LLM_JOINT in members:
        raise ValueError(f"Pool {pool!r} holds {LLM_JOINT}, which its model writes during its fit, "
                         f"on its side of the environment boundary; no featuriser builds it.")
    wanted = [member for member in members if member == LLM_GENERATED]
    if wanted and proposer is None:
        raise ValueError(f"Pool {pool!r} contains {wanted}, which needs a proposer to write its "
                         f"columns; none was given.")
    if proposer is not None and not wanted:
        raise ValueError(f"A proposer was given for pool {pool!r}, which contains no member that "
                         f"uses one. The pool would not hold the features the caller expects.")
    if COVARIATES in members and not covariates:
        raise ValueError(f"Pool {pool!r} holds the covariates, and no covariates were given: pass "
                         f"the dataset's (`Dataset.covariates`).")
    if covariates and COVARIATES not in members:
        raise ValueError(f"Covariates were given for pool {pool!r}, which does not hold them. The "
                         f"pool would not hold the features the caller expects.")
    if "kmer" in members and (k_range is None or kmer_normalise is None):
        raise ValueError(f"Pool {pool!r} holds k-mers, so `k_range` and `kmer_normalise` must say "
                         f"how they are counted.")

    from seqmaestro.proposer import GeneratedFeaturiser  # on use: the proposer imports this package

    built = {
        "positional": lambda: PositionalFeaturiser(window=window),
        "kmer": lambda: KmerFeaturiser(k_range=k_range, normalise=kmer_normalise),
        "llm_generated": lambda: GeneratedFeaturiser(proposer),
        COVARIATES: lambda: CovariateFeaturiser(covariates),
    }
    parts = [built[member]() for member in members]
    composed = parts[0]
    for part in parts[1:]:
        composed = composed + part
    return composed
