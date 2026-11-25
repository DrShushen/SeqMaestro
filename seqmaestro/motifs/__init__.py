"""Motifs: position frequency matrices, read from the files that hold them and matched to sequences.

A motif describes the sites that a protein, such as a transcription factor, binds: one row per
position of a site, holding how often each base occurs there. Two feature sources are built on this
package (`seqmaestro.features.motif`): the motifs of a published collection, such as JASPAR's, and
the motifs that a discovery tool (STREME) finds in the rows a cell fits on.

- `collection`: a motif and a named set of them, read from and written to MEME's text format.
- `scan`: every window of a sequence scored against a motif, a p-value for every score, and the two
  numbers a feature takes from them.
- `store`: the scans a run has made, kept so that each sequence is scanned once per run.
- `discovery`: the motifs STREME finds, and the known motif that Tomtom names as closest to each.
- `published`: where each published collection is fetched from, and how it becomes one MEME-format
  file.

**Everything here runs in Core (`SeqMaestro_Core`) and imports no MEME Suite code**, so a fitted
featuriser scans new rows (the held-out rows at the reveal, for instance) with Core alone. Only
discovery calls the MEME Suite's programs, which are installed in a separate environment
(`SeqMaestro_Method_meme`).
"""

from __future__ import annotations

from seqmaestro.motifs.collection import Collection, Motif, read_collection, read_meme, write_meme
from seqmaestro.motifs.scan import (
    NORMALISATIONS,
    STRANDS,
    SUMMARIES,
    Scanner,
    ScanSettings,
)
from seqmaestro.motifs.store import ScanStore, scan_with_store

__all__ = [
    "NORMALISATIONS",
    "STRANDS",
    "SUMMARIES",
    "Collection",
    "Motif",
    "ScanSettings",
    "ScanStore",
    "Scanner",
    "read_collection",
    "read_meme",
    "scan_with_store",
    "write_meme",
]
