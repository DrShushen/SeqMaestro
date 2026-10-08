#!/usr/bin/env python3
"""Ask a sequence dataset whether its strand convention and genome build are what they claim.

Two checks, run together because each covers the other's blind spot. The composition check needs
no network but cannot detect a *global* reverse complement. The reference check detects that, and
identifies the build as well, but needs a handful of lookups.

For example, with the label column named so that each class is also checked alone:

    python experiments/prepare/check_sequence_provenance.py \\
        --csv data/<dataset>/<built>.csv --sequence-column <column> --label-column <column>

Add `--reference` to query UCSC. It is opt-in because it sends the sampled coordinates to an
external service.

`--run-dir` is optional here, unlike in the stages of a run, which require it. Without it the script
only prints and writes nothing, so there is no destination to collide with.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from seqmaestro.provenance import record_invocation
from seqmaestro.qc import (
    AS_GIVEN,
    NO_MATCH,
    REVERSE_COMPLEMENT,
    classify_against_reference,
    orientation_by_group,
)

UCSC_API = "https://api.genome.ucsc.edu/getData/sequence"
DEFAULT_BUILDS = ("hg38", "hg19")


def fetch_reference(genome: str, chrom: str, start: int, end: int, timeout: int = 30) -> str:
    query = urllib.parse.urlencode(
        {"genome": genome, "chrom": chrom, "start": start, "end": end}
    )
    with urllib.request.urlopen(f"{UCSC_API}?{query}", timeout=timeout) as response:
        return json.loads(response.read().decode())["dna"].upper()


def check_against_reference(
    df: pd.DataFrame,
    args: argparse.Namespace,
    builds: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Compare a sample of windows against each candidate build.

    The reference window is placed by *assuming* the anchor base sits at `--anchor-index` of the
    sequence, so an exact match also confirms the centring. A different centring matches nothing.
    """
    sample = pd.concat(
        [
            df[df[args.strand_column] == s].head(args.n_reference)
            for s in (args.forward, args.reverse)
        ]
    )
    results: dict[str, dict[str, int]] = {}

    for build in builds:
        tally = {AS_GIVEN: 0, REVERSE_COMPLEMENT: 0, NO_MATCH: 0, "ERROR": 0}
        print(f"\n  {build}")
        for _, row in sample.iterrows():
            sequence = row[args.sequence_column]
            start = int(row[args.position_column]) - args.anchor_index
            end = start + len(sequence)
            try:
                reference = fetch_reference(build, row[args.chrom_column], start, end)
            except Exception as exc:
                tally["ERROR"] += 1
                print(f"    {row[args.chrom_column]}:{start}-{end} [{row[args.strand_column]}]  ERROR {exc}")
                continue
            verdict = classify_against_reference(sequence, reference)
            tally[verdict] += 1
            print(
                f"    {row[args.chrom_column]}:{start}-{end} "
                f"[{row[args.strand_column]} strand]  ->  {verdict}"
            )
            time.sleep(args.delay)
        results[build] = tally
        print(f"    tally: {tally}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sequence-column", default="raw_sequence")
    parser.add_argument("--strand-column", default="strand")
    parser.add_argument(
        "--label-column",
        default=None,
        help="Run the composition check within each class as well as pooled. Strongly "
        "recommended: a class with no positional structure cannot answer, and pooling it with "
        "one that can dilutes the signal.",
    )
    parser.add_argument("--forward", default="+")
    parser.add_argument("--reverse", default="-")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also query UCSC. Sends the sampled coordinates to an external service.",
    )
    parser.add_argument("--builds", nargs="+", default=list(DEFAULT_BUILDS))
    parser.add_argument("--chrom-column", default="chrom")
    parser.add_argument("--position-column", default="start")
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=None,
        help="Index within the sequence at which the anchor base sits. Defaults to the middle.",
    )
    parser.add_argument("--n-reference", type=int, default=3, help="Loci per strand per build.")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between API calls.")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    # Only the columns the checks use. These files have hundreds of columns, about 100 of them
    # single-character, which pandas holds as objects, so reading them all costs gigabytes to use
    # five.
    needed = [args.sequence_column, args.strand_column]
    if args.label_column:
        needed.append(args.label_column)
    if args.reference:
        needed += [args.chrom_column, args.position_column]

    df = pd.read_csv(args.csv, low_memory=False, usecols=needed)
    sequences = df[args.sequence_column].tolist()
    strands = df[args.strand_column].tolist()
    if args.anchor_index is None:
        args.anchor_index = len(sequences[0]) // 2

    print(f"{args.csv.name}: {len(df)} rows")
    print("\nComposition check (no genome needed)")
    groups = df[args.label_column].tolist() if args.label_column else [None] * len(df)
    report = {}
    for key, evidence in orientation_by_group(
        sequences, strands, groups, forward=args.forward, reverse=args.reverse
    ).items():
        name = "all rows" if key is None else f"{args.label_column}={key}"
        print(
            f"  {name:<20} {evidence.verdict:<14} "
            f"oriented r={evidence.correlation_oriented:+.3f}  "
            f"flipped r={evidence.correlation_flipped:+.3f}  "
            f"self-similarity {evidence.asymmetry:+.3f}  "
            f"(n={evidence.n_forward}+{evidence.n_reverse})"
        )
        print(f"  {'':<20} {evidence.explain()}")
        report[str(key)] = {
            "verdict": evidence.verdict,
            "correlation_oriented": evidence.correlation_oriented,
            "correlation_flipped": evidence.correlation_flipped,
            "asymmetry": evidence.asymmetry,
            "n_forward": evidence.n_forward,
            "n_reverse": evidence.n_reverse,
        }

    reference_report = None
    if args.reference:
        print(f"\nReference check (anchor base assumed at index {args.anchor_index})")
        reference_report = check_against_reference(df, args, tuple(args.builds))
        matched = [b for b, t in reference_report.items() if t[NO_MATCH] == 0 and t["ERROR"] == 0]
        print(f"\n  build identified: {matched if matched else 'none matched'}")

    if args.run_dir:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        record_invocation(args.run_dir, Path(__file__).resolve().parents[2], vars(args),
                          filename="invocation_check_sequence_provenance.json")
        (args.run_dir / "sequence_provenance.json").write_text(
            json.dumps(
                {
                    "csv": str(args.csv),
                    "n_rows": len(df),
                    "anchor_index": args.anchor_index,
                    "composition": report,
                    "reference": reference_report,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nWritten to {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
