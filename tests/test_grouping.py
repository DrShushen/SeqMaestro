"""Which rows a split may not separate: the connected components of a dataset's relations.

A dataset usually has more than one relation that makes two rows dependent, and the relations
overlap. A split grouped on one of them alone lets the other straddle it, and the score it reports
is then partly memory. Nothing in a finished run shows that, so the grouping is tested here.
`tests/test_dataset.py` tests the group map a split is checked against.
"""

from __future__ import annotations

import pandas as pd
import pytest

from seqmaestro.grouping import connected_groups


def rows(*records):
    return pd.DataFrame([{"sequence": sequence, "gene": gene} for sequence, gene in records])


def test_rows_of_one_gene_are_one_group():
    group = connected_groups(rows(("AAAA", "G1"), ("CCCC", "G1"), ("GGGG", "G2")),
                             ["sequence", "gene"])
    assert group.iloc[0] == group.iloc[1] != group.iloc[2]


def test_an_identical_sequence_is_one_group_whatever_gene_it_came_from():
    """Grouping by gene alone would miss this relation. Two genes can present the same window, and
    a row in one split would then be a copy of a row in another."""
    group = connected_groups(rows(("AAAA", "G1"), ("AAAA", "G2"), ("CCCC", "G3")),
                             ["sequence", "gene"])
    assert group.iloc[0] == group.iloc[1] != group.iloc[2]


def test_the_relations_chain_through_each_other():
    """The first row shares a gene with the second, and the second a sequence with the third, so
    none of the three may be separated. This is why the group is a connected component and not a
    single key."""
    frame = rows(("AAAA", "G1"), ("CCCC", "G1"), ("CCCC", "G2"))
    assert connected_groups(frame, ["sequence", "gene"]).nunique() == 1


def test_only_the_relations_named_join_rows():
    """A dataset that names the sequence alone is grouped on it alone. The gene column is there,
    and must not join anything unasked."""
    assert connected_groups(rows(("AAAA", "G1"), ("CCCC", "G1")), ["sequence"]).nunique() == 2


def test_unrelated_rows_stay_apart():
    frame = rows(("AAAA", "G1"), ("CCCC", "G2"), ("GGGG", "G3"))
    assert connected_groups(frame, ["sequence", "gene"]).nunique() == 3


def test_a_missing_value_joins_nothing():
    """Two rows whose gene is unknown are not relatives for that reason, and each is still joined
    by the other relations it takes part in."""
    frame = rows(("AAAA", None), ("CCCC", None), ("CCCC", "G1"))
    group = connected_groups(frame, ["sequence", "gene"])
    assert group.iloc[0] != group.iloc[1] and group.iloc[1] == group.iloc[2]


@pytest.mark.parametrize("relations", [["sequence"], ["sequence", "gene"]])
def test_every_row_is_in_exactly_one_group_and_keeps_its_index(relations):
    frame = rows(("AAAA", "G1"), ("CCCC", "G1"), ("AAAA", "G2"), ("TTTT", "G3"))
    frame.index = [10, 11, 12, 13]
    group = connected_groups(frame, relations)
    assert len(group) == len(frame)
    assert group.index.equals(frame.index)
