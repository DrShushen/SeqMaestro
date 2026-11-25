"""The featuriser of a pool, built from the run's settings for each feature source.

Every part of a run that builds the columns of a pool takes its featuriser from here: a Stage 2
cell of a Core model, the columns Core computes and hands to DEFT's tree, and the time estimate.
The k-mers are therefore counted, and the motifs scanned, in the same way for all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seqmaestro.features import MOTIF_DISCOVERED, MOTIF_KNOWN, make_featuriser, parse_pool
from seqmaestro.features.base import Featuriser
from seqmaestro.pipeline.motifs import discovered_motifs, known_motifs, store_for

if TYPE_CHECKING:
    from seqmaestro.dataset import Dataset
    from seqmaestro.pipeline.config import ExperimentConfig
    from seqmaestro.proposer import Proposer


def pool_featuriser(config: ExperimentConfig, dataset: Dataset, pool: str,
                    proposer: Proposer | None = None,
                    run_dir: str | Path | None = None) -> Featuriser:
    """The featuriser of `pool`, built from the run's settings: the k-mers as `sources.kmer`
    states, the positions over the dataset's declared window, the dataset's covariates where the
    pool holds them, the motifs of the run's collection as `sources.motif_known` states, and the
    discovered motifs as `sources.motif_discovered` states.

    `run_dir` gives a motif source the run's copies of its collections and the run's store of scans
    and discoveries (`seqmaestro.pipeline.motifs`). Without it (as for an estimate), a collection is
    read where it was fetched and nothing is stored.
    """
    kmer = config.sources.kmer
    members = parse_pool(pool)
    known = known_motifs(config, run_dir) if MOTIF_KNOWN in members else None
    discovered = (discovered_motifs(config, dataset, run_dir) if MOTIF_DISCOVERED in members
                  else None)
    holds_motifs = known is not None or discovered is not None
    store = store_for(run_dir) if holds_motifs and run_dir is not None else None
    return make_featuriser(pool, proposer=proposer, covariates=dataset.covariates_for(pool),
                           window=dataset.positional_window,
                           k_range=None if kmer is None else kmer.k_range,
                           kmer_normalise=None if kmer is None else kmer.normalise,
                           motif_known=known, motif_discovered=discovered, motif_store=store)
