"""The splice-site check: whether windows cut from the introns beside an exon are oriented as the
transcript is.

A delivery left on the wrong strand, or cut one base off, still gives every feature a plausible
description, so a build that cuts such windows measures the consensus at the junction and refuses a
delivery without it.
"""

from __future__ import annotations

import pytest

from seqmaestro.qc import ACCEPTOR, DONOR, splice_site_check
from seqmaestro.qc.splice_sites import MIN_SPLICE_SITE_RATE, SHOWN


def test_the_check_reports_each_rate_and_the_commonest_dinucleotides():
    result = splice_site_check(["AG"] * 9 + ["TG"], ["GT"] * 8 + ["GC", "AT"])
    assert result["acceptor_rate"] == 0.9 and result["donor_rate"] == 0.9
    assert result["acceptor_dinucleotides"] == {"AG": 9, "TG": 1}
    assert result["donor_dinucleotides"] == {"GT": 8, "GC": 1, "AT": 1}


def test_the_minor_donor_counts_as_canonical():
    assert set(DONOR) == {"GT", "GC"} and ACCEPTOR == ("AG",)
    assert splice_site_check(["AG"] * 4, ["GC"] * 4)["donor_rate"] == 1.0


def test_no_more_than_the_commonest_few_are_kept():
    donors = ["GT"] * 95 + ["AA", "AC", "AT", "CA", "CC"]
    assert len(splice_site_check(["AG"] * 100, donors)["donor_dinucleotides"]) == SHOWN


def test_windows_on_the_other_strand_are_refused():
    """Reverse-complemented, the donor's `GT` reads `AC` where the acceptor should be, and the
    acceptor's `AG` reads `CT` where the donor should be."""
    with pytest.raises(ValueError, match="not exons flanked by introns in transcript orientation"):
        splice_site_check(["AC"] * 10, ["CT"] * 10)


@pytest.mark.parametrize("acceptors, donors", [(["AG"] * 8 + ["TT"] * 2, ["GT"] * 10),
                                               (["AG"] * 10, ["GT"] * 8 + ["TT"] * 2)])
def test_either_site_below_the_floor_refuses_the_delivery(acceptors, donors):
    assert MIN_SPLICE_SITE_RATE > 0.8
    with pytest.raises(ValueError, match=r"0\.800"):
        splice_site_check(acceptors, donors)


@pytest.mark.parametrize("acceptors, donors", [([], []), (["AG"] * 3, ["GT"] * 2)])
def test_every_window_needs_both_dinucleotides(acceptors, donors):
    with pytest.raises(ValueError, match="one acceptor and one donor"):
        splice_site_check(acceptors, donors)
