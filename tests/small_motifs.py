"""Motifs for the tests that build every pool, with a stand-in for STREME and Tomtom.

The collection holds two motifs short enough to match the sequences of four bases that those tests
use, and its threshold is loose enough to count the matches, so a pool holding `motif_known` has
columns on any of their rows.

**The stand-in for discovery** takes the place of STREME and Tomtom in every test not marked
`crossenv` (`tests/conftest.py`), so the tests of everything around a discovery (the featuriser, the
store and its lock, the step before the arms, the cells, the report) need neither the MEME Suite nor
the time it takes. It ranks the words of `min_width` letters by how much more often they occur in
the sequences looked in than in those compared with, and returns each as a motif that expects that
word. On the planted dataset (`tests/planted.py`) it finds the planted site, as STREME does.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from seqmaestro.features import MOTIF_DISCOVERED, MOTIF_KNOWN, DiscoveredMotifs, KnownMotifs, parse_pool
from seqmaestro.motifs import Collection, Motif, ScanSettings
from seqmaestro.motifs.discovery import DOWN, UP, DiscoverySettings
from seqmaestro.target import BINARY

MOTIFS = (Motif("SM1", "short one", np.eye(4)[[0, 1]] * 0.97 + 0.0075, sites=20),
          Motif("SM2", None, np.eye(4)[[2, 2, 3]] * 0.97 + 0.0075, sites=20))
SETTINGS = ScanSettings(strands="both", p_threshold=0.1, summaries=("best_match", "count"),
                        normalise="counts")
#: The discovery settings of the tests that build every pool: words of three letters, which fit
#: their sequences of four bases.
DISCOVERY = DiscoverySettings(directions="both", continuous_fraction=0.333, min_width=3,
                              max_width=3, max_motifs=2, strands="both")
#: The versions of the tools that a discovery records under the stand-in.
STAND_IN_VERSIONS = {"streme": "stand-in", "tomtom": "stand-in"}


def known_motifs() -> KnownMotifs:
    return KnownMotifs(Collection("small", MOTIFS, digest="small"), SETTINGS, min_match_share=0.0)


def discovered_motifs(target_kind: str = BINARY) -> DiscoveredMotifs:
    return DiscoveredMotifs(DISCOVERY, SETTINGS, min_match_share=0.0, target_kind=target_kind,
                            described={UP: "class 1 against class 0", DOWN: "class 0 against class 1"},
                            annotate_with=None)


def pool_settings(pool: str, k_range: tuple[int, int] = (1, 2)) -> dict:
    """What `make_featuriser` needs for `pool`: how k-mers are counted, and the motif sources where
    the pool holds them."""
    settings: dict = {"k_range": k_range, "kmer_normalise": "counts"}
    if MOTIF_KNOWN in parse_pool(pool):
        settings["motif_known"] = known_motifs()
    if MOTIF_DISCOVERED in parse_pool(pool):
        settings["motif_discovered"] = discovered_motifs()
    return settings


# ------------------------------------------------------------------- the stand-in for discovery


def expecting(word: str) -> np.ndarray:
    """A matrix that expects `word`, softened so that every letter stays possible."""
    return np.eye(4)[["ACGT".index(letter) for letter in word]] * 0.94 + 0.015


def _shares(sequences: Sequence[str], width: int) -> dict[str, float]:
    """The share of `sequences` holding each word of `width` letters, with U counted as T."""
    held = Counter(word for sequence in sequences
                   for word in {sequence.upper().replace("U", "T")[start:start + width]
                                for start in range(len(sequence) - width + 1)}
                   if set(word) <= set("ACGT"))
    return {word: count / len(sequences) for word, count in held.items()}


def stand_in_streme(looked_in: Sequence[str], against: Sequence[str], out: Path,
                    settings: DiscoverySettings, rna: bool) -> list[tuple[Motif, float]]:
    """What `seqmaestro.motifs.discovery._streme` returns, without STREME: up to `max_motifs`
    words, the most enriched first, each under STREME's form of identifier (`1-GGCGCC`) and with an
    E-value that grows with its rank."""
    width = settings.min_width
    looked, compared = _shares(looked_in, width), _shares(against, width)
    ranked = sorted(looked, key=lambda word: (compared.get(word, 0.0) - looked[word], word))
    return [(Motif(f"{rank}-{word}", None, expecting(word), sites=looked[word] * len(looked_in)),
             10.0 ** (rank - 4))
            for rank, word in enumerate(ranked[: settings.max_motifs], start=1)]


def stand_in_tomtom(motifs: list[Motif], collection: Collection, out: Path,
                    rna: bool) -> dict[str, dict]:
    """What `seqmaestro.motifs.discovery._tomtom` returns, without Tomtom: the collection's first
    motif, as the closest to every motif."""
    first = collection.motifs[0]
    return {motif.identifier: {"collection": collection.name, "motif_id": first.identifier,
                               "motif_name": first.name, "q_value": 0.05, "p_value": 0.001}
            for motif in motifs}


def stand_in_versions() -> dict[str, str]:
    return dict(STAND_IN_VERSIONS)
