#!/usr/bin/env python
"""Fetch a published motif collection into data/motif_collections/<name>/, for the known-motif source.

    $CORE experiments/prepare/fetch_motif_collection.py jaspar --release 2026 --taxon vertebrates
    $CORE experiments/prepare/fetch_motif_collection.py cisbp-rna --species Homo_sapiens

Run at Stage 1, once the collection that suits the dataset is chosen with the user: JASPAR's for the
transcription factors of the dataset's taxon, where the sequences are DNA, or CisBP-RNA's for the
RNA-binding proteins of its species, where they stand for RNA. The folder it writes holds the
motifs as `motifs.meme`, and `collection.json`, which records the address, the time of the
download, the digest, the licence and what was kept. A run config names the collection by the
folder's name, which this prints (`sources.motif_known.collection`,
`experiments/configs/README.md`), and each run copies the file into the run's directory when it
first scans with it (`seqmaestro/pipeline/motifs.py`).

An existing collection is kept unless `--force` replaces it, since a run made with it may be
continued. `seqmaestro/motifs/published.py` says what each source publishes and what is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from seqmaestro.motifs import published
from seqmaestro.motifs.collection import read_collection, write_meme
from seqmaestro.pipeline.motifs import COLLECTION_FILE
from seqmaestro.pipeline.repository import MOTIF_COLLECTIONS

RECORD_FILE = "collection.json"
#: How long a download may take before it is given up, in seconds.
TIMEOUT = 300


def download(url: str, form: dict | None = None) -> bytes:
    """The bytes at `url`, posting `form` as a form where one is given."""
    data = None if form is None else urllib.parse.urlencode(form, doseq=True).encode()
    request = urllib.request.Request(url, data=data, headers={"User-Agent": "SeqMaestro"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def jaspar(args: argparse.Namespace) -> tuple[str, bytes, dict]:
    name = published.jaspar_name(args.release, args.collection, args.taxon, args.redundancy)
    url = published.jaspar_url(args.release, args.collection, args.taxon, args.redundancy)
    return name, download(url), {"source": "JASPAR", "release": args.release, "url": url,
                                 "collection": args.collection, "taxon": args.taxon or "all",
                                 "redundancy": args.redundancy, "molecule": "DNA",
                                 "licence": published.JASPAR_LICENCE}


def cisbp_rna(args: argparse.Namespace) -> tuple[str, bytes, dict]:
    name = published.cisbp_rna_name(args.species, args.inferred)
    page = download(f"{published.CISBP_RNA}/bulk_archive.php",
                    form=published.cisbp_rna_form(args.species)).decode(errors="replace")
    found = re.search(r'href="([^"]+\.zip)"', page)
    if found is None:
        raise SystemExit(f"CisBP-RNA's reply for {args.species} names no archive. Check the species "
                         f"name against {published.CISBP_RNA}/bulk.php.")
    url = f"{published.CISBP_RNA}/{found.group(1)}"
    motifs, counts = published.cisbp_rna_motifs(download(url), args.inferred)
    if not motifs:
        raise SystemExit(f"CisBP-RNA's archive for {args.species} holds no motif to keep: {counts}.")
    return name, write_meme(motifs, alphabet="ACGU").encode(), {
        "source": "CisBP-RNA", "build": published.CISBP_RNA_BUILD, "url": url,
        "species": args.species, "inferred_motifs": args.inferred, "molecule": "RNA",
        "counts": counts, "licence": published.CISBP_RNA_LICENCE}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="replace a collection that has been fetched before")
    sources = parser.add_subparsers(dest="source", required=True)
    one = sources.add_parser("jaspar", help="a file of JASPAR's DNA motifs of transcription factors")
    one.add_argument("--release", required=True, help="JASPAR's release, such as 2026")
    one.add_argument("--collection", default="CORE", choices=published.JASPAR_COLLECTIONS)
    one.add_argument("--taxon", choices=published.JASPAR_TAXA,
                     help="the taxon of the dataset's species; every taxon without it")
    one.add_argument("--redundancy", default="non-redundant", choices=published.JASPAR_REDUNDANCY,
                     help="non-redundant keeps the latest version of each matrix")
    one.set_defaults(fetch=jaspar)
    two = sources.add_parser("cisbp-rna", help="CisBP-RNA's motifs of the RNA-binding proteins of "
                                               "one species")
    two.add_argument("--species", required=True,
                     help="as CisBP-RNA spells it, such as Homo_sapiens or Mus_musculus")
    two.add_argument("--inferred", action="store_true",
                     help="keep the motifs inferred from a homologue's too, beside those measured")
    two.set_defaults(fetch=cisbp_rna)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    name, motifs, record = args.fetch(args)
    folder = MOTIF_COLLECTIONS / name
    if (folder / COLLECTION_FILE).exists() and not args.force:
        print(f"{folder} exists, and is kept. --force replaces it.", file=sys.stderr)
        return 1
    folder.mkdir(parents=True, exist_ok=True)
    (folder / COLLECTION_FILE).write_bytes(motifs)
    collection = read_collection(folder / COLLECTION_FILE, name)
    record = {"name": name, **record, "motifs": len(collection.motifs),
              "digest": hashlib.sha256(motifs).hexdigest(),
              "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (folder / RECORD_FILE).write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(collection.motifs)} motifs in {folder}\n"
          f"In a run config:  sources.motif_known.collection: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
