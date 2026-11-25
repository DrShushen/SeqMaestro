"""Rows a split must not separate.

A grouped split keeps related rows on the same side of the split. A model that has seen one row has
effectively seen its relatives, so a held-out score that includes them is not truly held out. Which
rows count as relatives is a property of the dataset (rows of one gene, one clonal family or one
identical window), and a dataset usually has more than one such relation at once.

**Grouping on one key at a time does not compose.** Two sequences can share a gene without sharing a
window, and share a window without belonging to one gene, so a split keyed on either column alone
leaves the other relation free to straddle it. The group is therefore the connected component under
*all* the relations together, which is what `connected_groups` computes.

This lives in the library, where every preparation script can use it, because more than one dataset
needs it and because it needs tests. The union-find is small, but the failure it prevents is silent:
a split that leaks reports an inflated score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def connected_groups(frame: pd.DataFrame, relations: list[str]) -> pd.Series:
    """One group per set of rows joined by any of `relations`, as contiguous integer codes.

    The groups are found by union-find over all the relations together, because the relations
    overlap and a split may separate neither kind of pair. Grouping on one column alone would let
    the other relation straddle the split, which is the leak grouping exists to prevent.

    A missing value joins nothing (`groupby` drops it), so a row whose gene is unknown is still
    linked by every other relation it participates in. A partially covered column is accepted
    because refusing it would remove the guarantee for every row when only some rows lack a value.
    The caller is expected to record the coverage.
    """
    parent = list(range(len(frame)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for column in relations:
        for _, positions in frame.reset_index(drop=True).groupby(column, sort=False).groups.items():
            members = list(positions)
            for other in members[1:]:
                root_a, root_b = find(members[0]), find(other)
                if root_a != root_b:
                    parent[root_a] = root_b

    codes = np.array([find(i) for i in range(len(frame))])
    return pd.Series(pd.factorize(codes)[0], index=frame.index)


def group_map(frame: pd.DataFrame, sequence_column: str, relations: list[str]) -> pd.DataFrame:
    """Which group each sequence belongs to under `relations`: one row per sequence and group.

    This is what `<splits>/groups.csv` holds. It is keyed on the sequence because a split file need
    hold nothing else that identifies a row, so a record keyed on a transcript or a site could not
    be joined back to it. With this key, whoever finalises the dataset can check where the rows
    went instead of trusting the builder.

    **Only the declared relations are used.** Adding the sequence as a relation is tempting, since
    two rows holding the same window are the same input to every featuriser that uses the sequence
    alone. That would hold a split to a promise it never made: a split grouped on the gene keeps
    each gene on one side, and that is all it promises. If two genes share a window, that is a fact
    about the data, which the advisory check `no_sequence_in_two_splits` in `seqmaestro.dataset`
    reports. It is not a failure of the grouping.

    One sequence can therefore belong to more than one group, and it then gets a row for each. These
    are the sequences whose split placement the declared grouping does not determine, and listing
    them is more useful than hiding them behind a merge.
    """
    if sequence_column not in frame.columns:
        raise KeyError(f"{sequence_column!r} is not a column of the dataset: {list(frame.columns)}.")
    absent = [column for column in relations if column not in frame.columns]
    if absent:
        raise KeyError(f"the split groups on {', '.join(absent)}, which the dataset does not carry. "
                       f"It has {', '.join(frame.columns)}.")
    groups = connected_groups(frame, relations)
    return (pd.DataFrame({"sequence": frame[sequence_column], "group": groups})
            .drop_duplicates().sort_values(["sequence", "group"]).reset_index(drop=True))


def shared_stretch_groups(sequences: list[str], k: int) -> np.ndarray:
    """A group per set of sequences joined by a shared exact stretch of `k` nucleotides, as integer
    codes, one per sequence.

    Two windows cut from overlapping or duplicated stretches of a genome are near copies of each
    other, so a split must keep them together like any other relatives. Every stretch of every
    sequence is hashed, and only the hashes that occur in two or more sequences join anything, so
    the components are found over those alone. Python's string hash is 64 bits, which makes a false
    join between two different stretches improbable at any size this handles. A sequence shorter
    than `k` shares no stretch, so it forms a group by itself.
    """
    rows, hashes = [], []
    for row, sequence in enumerate(sequences):
        stretches = [hash(sequence[start:start + k]) for start in range(len(sequence) - k + 1)]
        rows.append(np.full(len(stretches), row))
        hashes.append(np.array(stretches, dtype=np.int64))
    groups = np.arange(len(sequences))
    if not hashes:
        return groups

    pairs = pd.DataFrame({"row": np.concatenate(rows), "stretch": np.concatenate(hashes)}).drop_duplicates()
    shared = pairs[pairs.duplicated("stretch", keep=False)]
    if len(shared):
        components = connected_groups(shared.reset_index(drop=True), ["row", "stretch"])
        joined = pd.Series(components.to_numpy(), index=shared["row"].to_numpy())
        joined = joined[~joined.index.duplicated()]
        # Offset past every row number, so a joined component cannot take a lone row's code.
        groups[joined.index.to_numpy()] = len(sequences) + joined.to_numpy()
    return pd.factorize(groups)[0]
