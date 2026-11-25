"""Checks that a sequence dataset's strand convention is what it is assumed to be.

There are two independent checks, because they fail in different ways and neither is sufficient
alone.

**Without a reference genome.** If minus-strand records were left as genomic plus-strand sequence,
their positional base composition would be the mirror image of the plus-strand records': reversed
in position and complemented in base. Correlating the minus profile with the plus profile, and with
the plus profile's reverse complement, therefore tells the two cases apart using only the data.

The result applies two cautions itself, so the caller does not have to. First, a profile close to
palindromic fits both hypotheses equally well. `asymmetry` reports how distinguishable the
hypotheses were before the data chose between them, and the verdict is `INCONCLUSIVE` when the
profile is too palindromic or the margin between the hypotheses is thin. Second, the check is
**target-sensitive**: on a dataset whose negatives are structureless, running it over all rows
together dilutes the signal that holds the answer. `orientation_by_group` makes running it per
class, or per tertile of a continuous target, the easy option.

This check cannot detect a global flip: if every record were reverse complemented, the two strands
would still agree perfectly.

**Against a reference.** One lookup settles what composition cannot: whether the forward-strand
records match the reference exactly. `classify_against_reference` does only the comparison,
whereas the caller fetches the reference, so that this module stays offline and testable.
Comparing against several candidate builds also identifies the build, since the wrong one matches
nothing.

The code here does not assume a length, an alphabet, or DNA, since the complement map is an
argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DNA_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

ORIENTED = "ORIENTED"
NOT_ORIENTED = "NOT_ORIENTED"
INCONCLUSIVE = "INCONCLUSIVE"

AS_GIVEN = "AS_GIVEN"
REVERSE_COMPLEMENT = "REVERSE_COMPLEMENT"
NO_MATCH = "NO_MATCH"

# A profile correlating with its own reverse complement above this is close enough to palindromic
# that the two hypotheses are barely distinguishable, whatever the comparison then returns.
PALINDROMIC_ABOVE = 0.6

# Minimum gap between the two hypotheses' correlations before the verdict is called.
DEFAULT_MARGIN = 0.2


def reverse_complement(sequence: str, complement: dict[str, str] = DNA_COMPLEMENT) -> str:
    """Reverse complement, leaving symbols absent from `complement` (N, gaps) untouched."""
    table = str.maketrans(
        {**complement, **{k.lower(): v.lower() for k, v in complement.items()}}
    )
    return sequence.translate(table)[::-1]


def positional_profile(
    sequences: list[str], alphabet: tuple[str, ...] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per-position symbol frequencies, of shape `(length, len(alphabet))`.

    The alphabet is learned from the data unless given. Sequences must be of equal length, because
    a positional profile over ragged sequences is meaningless.
    """
    if not sequences:
        raise ValueError("No sequences given.")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"Sequences must be of equal length; found {sorted(lengths)}.")

    if alphabet is None:
        alphabet = tuple(sorted(set("".join(sequences))))

    arr = np.frombuffer("".join(sequences).encode(), dtype="S1").reshape(len(sequences), -1)
    profile = np.stack([(arr == s.encode()).mean(axis=0) for s in alphabet], axis=1)
    return profile, alphabet


def reverse_complement_profile(
    profile: np.ndarray,
    alphabet: tuple[str, ...],
    complement: dict[str, str] = DNA_COMPLEMENT,
) -> np.ndarray:
    """The profile the same sequences would have if each were reverse complemented.

    A symbol without an entry in `complement` maps to itself, which is correct for the ambiguity
    codes found in real assemblies (N is its own complement), and keeps this function consistent
    with `reverse_complement`, which leaves such symbols in place. The whole check rests on the two
    functions agreeing, and a test checks that they do.

    An alphabet that cannot express the flip is refused: if a symbol's partner is absent from the
    alphabet, there is no column to map it onto. That means the data does not contain the
    complementary base, which is a degenerate case for this check.
    """
    order = []
    for symbol in alphabet:
        partner = complement.get(symbol, symbol)
        if partner not in alphabet:
            raise KeyError(
                f"Symbol {symbol!r} complements to {partner!r}, which is absent from the "
                f"alphabet {alphabet}. The profile cannot represent the reverse complement."
            )
        order.append(alphabet.index(partner))
    return profile[::-1][:, order]


@dataclass(frozen=True)
class OrientationEvidence:
    """What the composition comparison found. `verdict` is one of the module constants."""

    verdict: str
    correlation_oriented: float
    correlation_flipped: float
    asymmetry: float
    n_forward: int
    n_reverse: int

    @property
    def separation(self) -> float:
        """How decisively the data preferred one hypothesis. Negative favours a flip."""
        return self.correlation_oriented - self.correlation_flipped

    def explain(self) -> str:
        if self.verdict == ORIENTED:
            return (
                f"Reverse-strand records are already reverse-complemented into the forward "
                f"reading direction (r={self.correlation_oriented:.3f} against "
                f"{self.correlation_flipped:.3f} for the alternative)."
            )
        if self.verdict == NOT_ORIENTED:
            return (
                f"Reverse-strand records still carry forward-strand sequence "
                f"(r={self.correlation_flipped:.3f} against "
                f"{self.correlation_oriented:.3f}); they need reverse complementing."
            )
        if self.asymmetry > PALINDROMIC_ABOVE:
            return (
                f"Inconclusive: the profile is close to palindromic (self-similarity "
                f"{self.asymmetry:.3f}), so it fits both hypotheses. Try a subset with real "
                f"positional structure (the positive class, or one end of a continuous target)."
            )
        return (
            f"Inconclusive: the two hypotheses scored within {abs(self.separation):.3f} "
            f"of each other."
        )


def orientation_evidence(
    sequences: list[str],
    strands: list[str],
    forward: str = "+",
    reverse: str = "-",
    complement: dict[str, str] = DNA_COMPLEMENT,
    margin: float = DEFAULT_MARGIN,
) -> OrientationEvidence:
    """Compare the two strands' base composition to decide whether the data is oriented."""
    if len(sequences) != len(strands):
        raise ValueError(f"Got {len(sequences)} sequences but {len(strands)} strands.")

    fwd = [s for s, st in zip(sequences, strands, strict=True) if st == forward]
    rev = [s for s, st in zip(sequences, strands, strict=True) if st == reverse]
    if not fwd or not rev:
        raise ValueError(
            f"Need records on both strands; got {len(fwd)} '{forward}' and {len(rev)} '{reverse}'."
        )

    # One alphabet across both strands, so the two profiles are comparable column by column.
    alphabet = tuple(sorted(set("".join(sequences))))
    profile_fwd, _ = positional_profile(fwd, alphabet)
    profile_rev, _ = positional_profile(rev, alphabet)
    flipped_fwd = reverse_complement_profile(profile_fwd, alphabet, complement)

    def correlate(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    oriented = correlate(profile_rev, profile_fwd)
    flip = correlate(profile_rev, flipped_fwd)
    asymmetry = correlate(profile_fwd, flipped_fwd)

    separation = oriented - flip
    if not np.isfinite(separation) or abs(separation) < margin or asymmetry > PALINDROMIC_ABOVE:
        verdict = INCONCLUSIVE
    else:
        verdict = ORIENTED if separation > 0 else NOT_ORIENTED

    return OrientationEvidence(
        verdict=verdict,
        correlation_oriented=oriented,
        correlation_flipped=flip,
        asymmetry=asymmetry,
        n_forward=len(fwd),
        n_reverse=len(rev),
    )


def orientation_by_group(
    sequences: list[str],
    strands: list[str],
    groups: list,
    **kwargs,
) -> dict[object, OrientationEvidence]:
    """Run the check within each group, plus an entry under `None` for all rows pooled.

    Group by the class label, or by tertiles of a continuous target: a dataset whose negatives are
    structureless can give a clear answer within its positives and none overall, and the rows at one
    end of a continuous target can do the same. A group that cannot be evaluated (one strand
    missing, say) is omitted instead of raising, so one bad group does not hide the others.
    """
    if not (len(sequences) == len(strands) == len(groups)):
        raise ValueError("sequences, strands and groups must be the same length.")

    out: dict[object, OrientationEvidence] = {}
    for key in [*dict.fromkeys(groups), None]:
        idx = range(len(groups)) if key is None else [i for i, g in enumerate(groups) if g == key]
        try:
            out[key] = orientation_evidence(
                [sequences[i] for i in idx], [strands[i] for i in idx], **kwargs
            )
        except ValueError:
            continue
    return out


def classify_against_reference(
    sequence: str,
    reference: str,
    complement: dict[str, str] = DNA_COMPLEMENT,
    case_sensitive: bool = False,
) -> str:
    """How a record relates to the reference sequence at the same coordinates.

    `AS_GIVEN` for a forward-strand record means the data is not globally flipped, whereas
    `REVERSE_COMPLEMENT` for a reverse-strand record means that record was oriented. `NO_MATCH`
    across a whole sample usually means the wrong genome build, or a window offset by a base.
    """
    if not case_sensitive:
        sequence, reference = sequence.upper(), reference.upper()
    if sequence == reference:
        return AS_GIVEN
    if reverse_complement(sequence, complement) == reference:
        return REVERSE_COMPLEMENT
    return NO_MATCH
