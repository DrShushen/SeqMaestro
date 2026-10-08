"""Checks on a dataset before it is modelled.

These answer questions about the data: whether the strand convention is what the column says,
whether the sequence matches the genome build it claims, whether windows cut around an exon carry
its splice sites where they should, and how much of each feature a split's grouping variable
explains (`group_dependence`, which is also a command). They are in the package because they are
logic worth testing, and because every new dataset has to be checked again: the answer for one
dataset says nothing about the next.
"""

from seqmaestro.qc.orientation import (
    AS_GIVEN,
    INCONCLUSIVE,
    NO_MATCH,
    NOT_ORIENTED,
    ORIENTED,
    REVERSE_COMPLEMENT,
    classify_against_reference,
    orientation_by_group,
    orientation_evidence,
    positional_profile,
    reverse_complement,
    reverse_complement_profile,
)
from seqmaestro.qc.splice_sites import (
    ACCEPTOR,
    DONOR,
    splice_site_check,
)

__all__ = [
    "ACCEPTOR",
    "AS_GIVEN",
    "DONOR",
    "INCONCLUSIVE",
    "NOT_ORIENTED",
    "NO_MATCH",
    "ORIENTED",
    "REVERSE_COMPLEMENT",
    "classify_against_reference",
    "orientation_by_group",
    "orientation_evidence",
    "positional_profile",
    "reverse_complement",
    "reverse_complement_profile",
    "splice_site_check",
]
