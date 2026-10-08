#!/usr/bin/env python
"""Write the group map a grouped split is checked against: `<splits>/groups.csv`.

    python experiments/prepare/write_group_map.py \
        --dataset data/<dataset>/<built>.csv \
        --splits data/<dataset>/<splits> \
        --declared experiments/prepare/declarations/<dataset>__<splits>.yaml

A grouped split promises that rows which are relatives of each other (one gene, one clonal family,
one identical window) never land on opposite sides of it. Checking that promise needs a record of
which group each row belongs to, and the record has to be joinable to the split files, which carry
only the sequence and the label. So the map is keyed on the sequence: one row per distinct sequence,
with its group under all the declared relations together.

The groups are traced in the **dataset the splits were cut from**, because that file holds the
grouping columns (the gene beside the window, the domain beside the site). The split files are read
only to find which rows they hold, and are never written. An existing split therefore gains a group
map without being redrawn, and a split already used by a finished run stays byte for byte as it was.

There is no flag for the relations. They come from the declaration, which is also where the check
reads them, so the map and the check cannot disagree about what the split was grouped on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from seqmaestro.dataset import GROUP_FILE, SPLITS, DatasetError, validate_declared
from seqmaestro.grouping import group_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="the built dataset CSV the splits were cut from, with the grouping columns")
    parser.add_argument("--splits", type=Path, required=True,
                        help="the split directory the map is written into")
    parser.add_argument("--declared", type=Path, required=True,
                        help="the declaration, which names the sequence column and what was grouped on")
    args = parser.parse_args(argv)

    try:
        declared = validate_declared(yaml.safe_load(args.declared.read_text()))
    except (DatasetError, OSError, yaml.YAMLError) as error:
        print(error, file=sys.stderr)
        return 2

    sequence = declared["columns"]["sequence"]
    relations = declared["split"]["grouped_by"]
    if not relations:
        print(f"{args.declared} declares no grouping, so there is no map to write.", file=sys.stderr)
        return 2

    target = declared["columns"]["target"]
    present: set[str] = set()
    rows_in_split: set[tuple[str, str]] = set()
    for split in SPLITS:
        path = args.splits / f"{split}.csv"
        if not path.exists():
            print(f"{path} is missing; this is not a prepared split directory.", file=sys.stderr)
            return 2
        part = pd.read_csv(path, usecols=[sequence, target], dtype=str, keep_default_na=False)
        present |= set(part[sequence])
        rows_in_split |= set(zip(part[sequence], part[target], strict=True))

    # Unlike the split files above, this file is read with missing values left missing.
    # `connected_groups` relies on a missing value joining nothing, and reading empty fields as text
    # would put every row whose gene is unknown into one group with all the others.
    frame = pd.read_csv(args.dataset, dtype={sequence: str, target: str})
    # Keep only the rows the splits hold. A build can carry rows that a split then dropped (one design
    # subsamples the abundant class, another discards a stratum with no contrast). A row in no split
    # must not attach its sequence to its group, because that would report the grouping as broken
    # over a row nobody can be shown to have seen.
    #
    # Rows are matched on the sequence and the label together. The split files carry only those two
    # columns, so no finer join exists.
    scoped = frame[[(s, t) in rows_in_split
                    for s, t in zip(frame[sequence], frame[target], strict=True)]]
    try:
        mapped = group_map(scoped, sequence, relations)
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    missing = present - set(mapped["sequence"])
    if missing:
        print(f"{len(missing)} sequence(s) in the splits are absent from {args.dataset}, so their "
              f"group is unknown. The splits were cut from a different build of this dataset; the "
              f"map would be checked against rows it does not describe.", file=sys.stderr)
        return 1
    if len(scoped) < len(frame):
        print(f"{len(frame) - len(scoped):,} of {len(frame):,} rows are not in this split and were "
              f"left out, so relations are traced only through the rows it holds")
    # A relation missing for some rows still groups the rows it covers, and the rest are grouped by
    # the other relations. Each relation's coverage is printed, because a grouping that holds for
    # four rows in five is a weaker guarantee than it looks.
    for relation in relations:
        known = int(scoped[relation].notna().sum())
        if known < len(scoped):
            print(f"{relation}: known for {known:,} of {len(scoped):,} rows; the rest are grouped "
                  f"by the other relations alone")

    out = args.splits / GROUP_FILE
    mapped.to_csv(out, index=False)
    print(f"{len(mapped):,} distinct sequences in {mapped['group'].nunique():,} group(s), over "
          f"{', '.join(relations)}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
