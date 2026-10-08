"""Whether windows cut from the introns either side of an exon are in transcript orientation.

A window made of the intron before an exon and the intron after it has a 3' splice site where the
upstream intron ends and a 5' splice site where the downstream intron begins, and both have a
consensus: `AG` ends an intron, and `GT`, or the minor `GC`, begins one. A delivery that left a
minus-strand gene on the genomic plus strand, or cut its flanks one base off, loses that consensus.
Nothing downstream could detect such a delivery, because every feature built on a mis-oriented
dataset still looks interpretable, so a build measures the consensus rates at the junction and
refuses a delivery below `MIN_SPLICE_SITE_RATE` at either site.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

#: The canonical intron boundaries, in transcript orientation. `GC` is the well-known minor donor, so
#: it counts as canonical alongside `GT`. `AT`-`AC` introns are rarer still and do not count.
ACCEPTOR = ("AG",)
DONOR = ("GT", "GC")

#: Below this the delivery is not what it claims to be, and the likeliest cause is an orientation
#: error or an off-by-one in how the flanks were cut. Set well under the true consensus rate so that
#: only a real structural failure trips it.
MIN_SPLICE_SITE_RATE = 0.90

#: How many of the commonest dinucleotides at each site the record keeps.
SHOWN = 4


def splice_site_check(acceptors: Iterable[str], donors: Iterable[str]) -> dict:
    """The consensus rates of the dinucleotides ending each upstream intron (`acceptors`) and
    beginning each downstream one (`donors`), with the commonest of each, or `ValueError` when either
    rate is below `MIN_SPLICE_SITE_RATE`."""
    acceptors, donors = Counter(acceptors), Counter(donors)
    total = sum(acceptors.values())
    if not total or total != sum(donors.values()):
        raise ValueError("Every window needs one acceptor and one donor dinucleotide.")
    acceptor_rate = sum(acceptors[pair] for pair in ACCEPTOR) / total
    donor_rate = sum(donors[pair] for pair in DONOR) / total
    if min(acceptor_rate, donor_rate) < MIN_SPLICE_SITE_RATE:
        raise ValueError(
            f"Only {acceptor_rate:.3f} of records end their upstream flank in {ACCEPTOR} and "
            f"{donor_rate:.3f} begin their downstream flank with {DONOR}. These sequences are not "
            f"exons flanked by introns in transcript orientation, and no downstream analysis could "
            f"detect that: every feature built on a mis-oriented dataset still looks interpretable."
        )
    return {
        "acceptor_rate": round(acceptor_rate, 4),
        "donor_rate": round(donor_rate, 4),
        "acceptor_dinucleotides": dict(acceptors.most_common(SHOWN)),
        "donor_dinucleotides": dict(donors.most_common(SHOWN)),
    }
