"""Featurisation: turning rows (sequences and any covariates) into a named numeric matrix.

```python
from seqmaestro.features import make_featuriser

featuriser = make_featuriser("positional+kmer", k_range=(1, 6), kmer_normalise="counts")
X_train = featuriser.fit_transform(dataset.inputs(train))
X_val = featuriser.transform(dataset.inputs(val))       # fitted on train only
```

A featuriser takes rows (`Inputs`: the sequences and any covariates the dataset declares), which
`dataset.inputs` builds from a split frame using the columns `dataset.json` records. A bare
sequence column is accepted too, as rows with no covariates.

**A pool is a set of feature sources** (the code here calls them its members). `positional`,
`kmer`, `motif_known` (every motif of a published collection) and `motif_discovered` (the motifs
that discovery finds in the fitting rows, `seqmaestro.features.motif`) are sources, and
`positional+kmer` is the pool holding the first two. `covariates` is the dataset's covariates as
they are. The three written sources are those whose features a language model writes:
`llm_generated` in one round, `llm_generated_iterative` over several rounds, each shown a fit of
the arm's model, and `llm_generated_joint` inside a model's fit, by a procedure only that model
runs (DEFT's tree, IGSR). A pool is named by joining its sources with `+` in the canonical order
below, so one name always denotes one set. `seqmaestro.arms` declares which model can be handed
which pool.

**A pool's sources are what the arm's model is handed and what its proposer may use.** An arm
whose pool holds `covariates` hands them to its model and shows them to its proposer, which may
write features that use them. An arm whose pool does not is given rows without them
(`seqmaestro.dataset.Dataset.columns_for`), so neither its model nor its proposer sees them.

A further source (gapped or mismatch *k*-mers, the concentration of motif matches around one
position) would be another `Featuriser` that composes with these and emits `FeatureRecord`s as
they do.

**A pool holds only tabular columns.** It is the set of columns a model is handed, and the basis
for a `selected` provenance. The raw sequence is available separately to arms that *generate*
features while fitting, and is never named in a pool (see `seqmaestro.features.base.PROVENANCES`).
Of the written sources, only `llm_generated` is built here. Stage 2 builds the iterative source
around the arm's model (`seqmaestro.pipeline.stage2.ProposerArm.featuriser`), and the joint source
is written inside its model's fit, in that model's conda environment.

`CompositionFeaturiser` is exported beside them, although it is not a feature source: it is the
featurisation of the composition baseline, which every run fits and every report draws as its
reference line. An arm is never handed it.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

from seqmaestro.features.base import FeatureRecord, Featuriser
from seqmaestro.features.composition import CompositionFeaturiser
from seqmaestro.features.covariates import CovariateFeaturiser
from seqmaestro.features.kmer import KmerFeaturiser
from seqmaestro.features.motif import (
    MOTIF_DISCOVERED,
    MOTIF_KNOWN,
    DiscoveredMotifFeaturiser,
    DiscoveredMotifs,
    KnownMotifFeaturiser,
    KnownMotifs,
)
from seqmaestro.features.positional import PositionalFeaturiser, PositionalWindow
from seqmaestro.motifs.store import ScanStore

#: The source that holds the dataset's covariates.
COVARIATES = "covariates"

#: The written sources: features a language model writes in one round, over several rounds each
#: shown a fit of the arm's model, or inside a model's fit by a procedure only that model runs.
LLM_GENERATED = "llm_generated"
LLM_ITERATIVE = "llm_generated_iterative"
LLM_JOINT = "llm_generated_joint"
WRITTEN_MEMBERS = (LLM_GENERATED, LLM_ITERATIVE, LLM_JOINT)

#: The sources a pool may contain, in canonical order. Pool names are built in this order instead of
#: alphabetically, because alphabetical order would turn `positional+kmer` into `kmer+positional`.
#: `covariates` is last, so the name of a pool with them is the name of the pool without them
#: followed by `+covariates`.
POOL_MEMBERS = ("positional", "kmer", MOTIF_KNOWN, MOTIF_DISCOVERED, LLM_GENERATED, LLM_ITERATIVE,
                LLM_JOINT, COVARIATES)

#: The sources whose columns follow from the rows by a rule, fixed before a model is fitted.
ENUMERATED_MEMBERS = ("positional", "kmer", MOTIF_KNOWN, MOTIF_DISCOVERED)

#: The pools of sources enumerated from the sequence: every set of them, the smaller sets first.
_ENUMERATED_POOLS = tuple("+".join(members) for size in range(1, len(ENUMERATED_MEMBERS) + 1)
                          for members in itertools.combinations(ENUMERATED_MEMBERS, size))

#: The pools an arm can be handed without a proposer: each enumerated pool with and without the
#: covariates, and the covariates on their own.
FIXED_POOLS = (*_ENUMERATED_POOLS, *(f"{pool}+{COVARIATES}" for pool in _ENUMERATED_POOLS),
               COVARIATES)

__all__ = [
    "COVARIATES",
    "ENUMERATED_MEMBERS",
    "FIXED_POOLS",
    "LLM_GENERATED",
    "LLM_ITERATIVE",
    "LLM_JOINT",
    "MOTIF_DISCOVERED",
    "MOTIF_KNOWN",
    "WRITTEN_MEMBERS",
    "CompositionFeaturiser",
    "CovariateFeaturiser",
    "DiscoveredMotifFeaturiser",
    "DiscoveredMotifs",
    "FeatureRecord",
    "KmerFeaturiser",
    "KnownMotifFeaturiser",
    "KnownMotifs",
    "PositionalFeaturiser",
    "PositionalWindow",
    "make_featuriser",
    "parse_pool",
    "pool_name",
    "with_covariates",
]


def parse_pool(name: str) -> tuple[str, ...]:
    """The sources a pool name lists, in canonical order, refusing any unknown or repeated one."""
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
    """The canonical name for a set of sources, so that each set has one name and one directory."""
    return "+".join(parse_pool("+".join(members)))


def with_covariates(pool: str) -> str:
    """The pool with the covariates added: `positional` becomes `positional+covariates`."""
    return pool_name((*parse_pool(pool), COVARIATES))


def make_featuriser(pool: str, k_range: tuple[int, int] | None = None, proposer=None,
                    covariates: Mapping[str, str] | None = None,
                    window: PositionalWindow | None = None,
                    kmer_normalise: str | None = None,
                    motif_known: KnownMotifs | None = None,
                    motif_discovered: DiscoveredMotifs | None = None,
                    motif_store: ScanStore | None = None) -> Featuriser:
    """Build the featuriser for a pool, composing its sources in canonical order.

    `k_range` and `kmer_normalise` say how the `kmer` source counts, and a pool that holds it needs
    both: how k-mers are counted is a setting of a run, stated in its config, so this function sets
    no default for it. `window` (the dataset's declared positional window,
    `Dataset.positional_window`) is a setting of the `positional` source.

    `proposer`, `covariates`, `motif_known` and `motif_discovered` each belong to one source, which
    requires it: `proposer` to `llm_generated`, `covariates` (each covariate's declared description,
    `Dataset.covariates`) to `covariates`, and `motif_known` (the collection and its scan settings)
    and `motif_discovered` (how discovery is run and how its motifs are scanned) to the source of
    the same name. Passing one to a pool that does not hold its source is refused instead of
    ignored, because it means the caller expected features this pool will not contain.

    `motif_store` is where a run keeps its scans (`seqmaestro.motifs.store`). It saves repeating
    scans and changes no value, so a pool without a motif source accepts it too and ignores it.

    The iterative and joint sources are refused. `seqmaestro.pipeline.stage2.ProposerArm.featuriser`
    builds the iterative source around the arm's model (`seqmaestro.proposer.iterative`), and the
    joint source is written inside its model's fit, in that model's environment.
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
    if MOTIF_KNOWN in members and motif_known is None:
        raise ValueError(f"Pool {pool!r} holds {MOTIF_KNOWN}, so `motif_known` must name the "
                         f"collection and how its scans are summarised.")
    if motif_known is not None and MOTIF_KNOWN not in members:
        raise ValueError(f"Motifs of a collection were given for pool {pool!r}, which does not hold "
                         f"{MOTIF_KNOWN}. The pool would not hold the features the caller expects.")
    if MOTIF_DISCOVERED in members and motif_discovered is None:
        raise ValueError(f"Pool {pool!r} holds {MOTIF_DISCOVERED}, so `motif_discovered` must say how "
                         f"discovery is asked and its motifs scanned.")
    if motif_discovered is not None and MOTIF_DISCOVERED not in members:
        raise ValueError(f"Discovery settings were given for pool {pool!r}, which does not hold "
                         f"{MOTIF_DISCOVERED}. The pool would not hold the features the caller "
                         f"expects.")

    from seqmaestro.proposer import GeneratedFeaturiser  # deferred: the proposer imports this package

    built = {
        "positional": lambda: PositionalFeaturiser(window=window),
        "kmer": lambda: KmerFeaturiser(k_range=k_range, normalise=kmer_normalise),
        MOTIF_KNOWN: lambda: KnownMotifFeaturiser(motif_known.collection, motif_known.settings,
                                                  min_match_share=motif_known.min_match_share,
                                                  store=motif_store),
        MOTIF_DISCOVERED: lambda: DiscoveredMotifFeaturiser(motif_discovered, store=motif_store),
        "llm_generated": lambda: GeneratedFeaturiser(proposer),
        COVARIATES: lambda: CovariateFeaturiser(covariates),
    }
    parts = [built[member]() for member in members]
    composed = parts[0]
    for part in parts[1:]:
        composed = composed + part
    return composed
