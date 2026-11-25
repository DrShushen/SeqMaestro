"""The published motif collections a run can scan with: where each is fetched from, what it is
named, and how it becomes one MEME-format file.

`experiments/prepare/fetch_motif_collection.py` fetches a collection at Stage 1 into
`data/motif_collections/<name>/`: its motifs as `motifs.meme`, and `collection.json`, which says
where they came from, when, under which licence, and what was kept. This module holds what that
script needs besides the download, so it can be tested without a network.

- **JASPAR** publishes the DNA motifs of transcription factors, one file per release, collection
  and taxon, in MEME format, under CC BY 4.0. The file is kept as published, so its digest is the
  published file's.
- **CisBP-RNA** publishes the motifs of RNA-binding proteins, one archive per species, with one
  table per motif and a table of the proteins each motif belongs to. Each motif is measured for its
  protein (direct) or inferred from a homologue's. The direct motifs are kept, and the inferred ones
  too where they are asked for. The kept motifs are written as one MEME-format file in the RNA
  alphabet. A motif's name is that of the protein it belongs to, or the names of its proteins
  joined by a slash where it belongs to several. A protein the archive names only by its Ensembl
  identifier is left out of the name where another has a gene symbol. The tables state no number of
  sites, so the motifs are written with the number the MEME Suite assumes for a file that states
  none (`DEFAULT_SITES`).
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from typing import Any

import numpy as np

from seqmaestro.motifs.collection import DEFAULT_SITES, Motif

JASPAR_DOWNLOADS = "https://jaspar.elixir.no/download/data"
JASPAR_COLLECTIONS = ("CORE", "UNVALIDATED")
JASPAR_TAXA = ("vertebrates", "insects", "nematodes", "plants", "fungi", "urochordates")
JASPAR_REDUNDANCY = ("non-redundant", "redundant")
JASPAR_LICENCE = "CC BY 4.0, as JASPAR's downloads page states"

CISBP_RNA = "https://cisbp-rna.ccbr.utoronto.ca"
CISBP_RNA_BUILD = "2.00"
CISBP_RNA_LICENCE = "freely available, as CisBP-RNA states; no licence is given"
#: The table of a CisBP-RNA archive naming each motif's proteins, and its folder of motif tables.
CISBP_RNA_PROTEINS = "RBP_Information_all_motifs.txt"
CISBP_RNA_MATRICES = "pwms_all_motifs"
#: How CisBP-RNA marks a motif measured for its protein, and one inferred from a homologue's.
DIRECT, INFERRED = "D", "I"


def jaspar_name(release: str, collection: str, taxon: str | None, redundancy: str) -> str:
    """The name JASPAR gives one file, such as `JASPAR2026_CORE_vertebrates_non-redundant`. The
    name of a collection over every taxon has no taxon part."""
    if collection not in JASPAR_COLLECTIONS:
        raise ValueError(f"A JASPAR collection is one of {', '.join(JASPAR_COLLECTIONS)}, not "
                         f"{collection!r}.")
    if taxon is not None and taxon not in JASPAR_TAXA:
        raise ValueError(f"A JASPAR taxon is one of {', '.join(JASPAR_TAXA)}, not {taxon!r}.")
    if redundancy not in JASPAR_REDUNDANCY:
        raise ValueError(f"JASPAR's files are {' or '.join(JASPAR_REDUNDANCY)}, not {redundancy!r}.")
    return "_".join(part for part in (f"JASPAR{release}", collection, taxon, redundancy) if part)


def jaspar_url(release: str, collection: str, taxon: str | None, redundancy: str) -> str:
    """Where JASPAR publishes one file of position frequency matrices in MEME format."""
    name = jaspar_name(release, collection, taxon, redundancy)
    return f"{JASPAR_DOWNLOADS}/{release}/{collection}/{name}_pfms_meme.txt"


def cisbp_rna_name(species: str, inferred: bool) -> str:
    """The name a fetched CisBP-RNA collection is kept under: `CisBP-RNA_2.00_Homo_sapiens_direct`,
    or `..._all` with the inferred motifs too."""
    return f"CisBP-RNA_{CISBP_RNA_BUILD}_{species}_{'all' if inferred else 'direct'}"


def cisbp_rna_form(species: str) -> dict[str, Any]:
    """The fields of CisBP-RNA's download form that request one species' archive of motifs and
    proteins."""
    return {"selSpec": species, "Spec[]": ["RBP_Information", "PFMs"],
            "submit": "Download Species Archive!"}


def cisbp_rna_motifs(archive: bytes, inferred: bool) -> tuple[list[Motif], dict[str, int]]:
    """The motifs of a CisBP-RNA species archive, and counts of what was kept and left out.

    A motif is kept when one of its proteins has it measured directly or, with `inferred`, when any
    of its proteins has it at all. A motif whose table is empty is left out.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as opened:
        rows = list(csv.DictReader(io.TextIOWrapper(opened.open(CISBP_RNA_PROTEINS), encoding="utf-8"),
                                   delimiter="\t"))
        tables = {name.rsplit("/", 1)[-1].removesuffix(".txt"): name for name in opened.namelist()
                  if name.startswith(f"{CISBP_RNA_MATRICES}/") and name.endswith(".txt")}
        proteins: dict[str, set[str]] = defaultdict(set)
        statuses: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            motif = row.get("Motif_ID", "")
            if motif in ("", ".", "NULL"):
                continue
            proteins[motif].add(row["RBP_Name"])
            statuses[motif].add(row["RBP_Status"])
        kept, counts = [], {"listed": len(proteins), "inferred_left_out": 0, "empty_left_out": 0}
        for identifier in sorted(proteins):
            if not inferred and DIRECT not in statuses[identifier]:
                counts["inferred_left_out"] += 1
                continue
            frequencies = _matrix(opened.read(tables[identifier]).decode()) if identifier in tables else None
            if frequencies is None:
                counts["empty_left_out"] += 1
                continue
            named = sorted(proteins[identifier])
            symbols = [protein for protein in named if not protein.startswith("ENS")] or named
            kept.append(Motif(identifier, "/".join(symbols), frequencies, sites=DEFAULT_SITES))
    counts["kept"] = len(kept)
    return kept, counts


def _matrix(text: str) -> np.ndarray | None:
    """A CisBP-RNA motif table (`Pos A C G U`, one row per position) as frequencies, or None where
    it holds no row."""
    lines = [line.split("\t") for line in text.strip().splitlines()]
    if len(lines) < 2:
        return None
    header = lines[0]
    columns = [header.index(letter) for letter in ("A", "C", "G", "U")]
    return np.array([[float(row[column]) for column in columns] for row in lines[1:]])
