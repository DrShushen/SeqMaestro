"""Matching motifs to sequences: a score for every window, a p-value for every score, and the two
numbers a motif feature takes from them.

**A window's score is the sum of log-odds over its positions.** Each frequency is first softened as
FIMO softens it: the motif's frequencies, weighted by its number of sites, are averaged with the
background frequencies, weighted by `PSEUDOCOUNT`, so a frequency of zero becomes small and finite.
The log-odds of a letter is the base-2 logarithm of its softened frequency over its background
frequency, so a window scores high where its letters are those the motif expects.

**Scores are whole numbers of `GRID` bits.** Each log-odds entry is rounded to the grid before
anything is summed, so a window's score is an integer, summed without rounding error, and a scan
stored on one machine equals a scan made afresh on another.

**A score's p-value is exact for the rounded matrix**: the probability that a window of random
sequence scores at least as high. Under a background of independent letters, the score of a window
is a sum of independent position scores, so its distribution is the convolution of the positions'
distributions, which `score_distribution` computes on the integer scale.

**The background is uniform** (a quarter for each letter), as JASPAR's files state. A scan then
depends on the sequence and the motif alone, so a run scans each sequence once and the cells of
every arm, seed and label condition use that scan (`seqmaestro.motifs.store`). On a dataset of
skewed composition, a motif rich in the commoner letters matches by chance more often than its
p-value says. The composition baseline that every run fits shows how far composition alone
predicts the target.

**Strands.** `both` scores each window on the sequence as given and on its reverse complement, and
keeps the higher score. It suits DNA, which a protein can bind on either strand. `given` scores the
sequence as given, which suits RNA, since it is single-stranded.

**Letters.** A, C, G and T are scanned in either case, and U is taken as T. A window that holds any
other symbol (N, a gap) is no match at all: it has no score, and counts neither as a match nor as a
position scanned.

**What a feature takes from the scores** (`SUMMARIES`):
- `best_match`: how strong the best match in the sequence is, as `-log10` of its p-value. A
  sequence with no window to score (shorter than the motif, or all N) takes 0, the value of the
  weakest possible match.
- `count`: the number of positions whose score has a p-value at or below the threshold. With both
  strands scanned, a position counts once when either strand passes. Matches may overlap.
  `normalise: per_length` divides each count by the positions scanned, so that a count does not
  grow with the sequence's length.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from seqmaestro.motifs.collection import Motif

#: Bits per step of the integer score scale.
GRID = 1e-3
#: The weight of the background when frequencies are softened before log-odds are taken (FIMO's
#: default pseudocount).
PSEUDOCOUNT = 0.1
#: The background frequency of A, C, G and T.
BACKGROUND = (0.25, 0.25, 0.25, 0.25)
#: Which strands are scored: `both` for DNA, or `given` (the sequence as given) for RNA.
STRANDS = ("both", "given")
#: The numbers a motif feature can take from a sequence's scores, in column order.
SUMMARIES = ("best_match", "count")
#: How a count is reported: as it is, or per position scanned. The k-mer counts use the same two
#: names.
NORMALISATIONS = ("counts", "per_length")
#: The version of this module's arithmetic. It is part of every scan's digest, so a scan stored by
#: another version is never taken for one made by this version.
SCAN_VERSION = 1

#: The code of a symbol that no motif letter stands for, and of the padding added to a shorter
#: sequence in a batch.
INVALID = 4
#: How many positions of a window are scored with one lookup, in a table that holds the summed
#: score of every group of that many letters. Groups of four need a quarter of the lookups that
#: scoring one position at a time needs.
GROUP = 4
#: The most letters, padding included, that one batch of sequences holds. Sequences are batched by
#: length, so a batch of short sequences is not padded to the length of a long one.
BATCH_LETTERS = 2_000_000

_CODES = np.full(256, INVALID, dtype=np.uint8)
for _letter, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("U", 3)):
    _CODES[ord(_letter)] = _CODES[ord(_letter.lower())] = _code

#: For each group size, the codes of the letters of every group: row i holds the codes of group i,
#: which are the digits of i in base 5.
_DIGITS = {size: np.array([[(index // 5 ** (size - 1 - place)) % 5 for place in range(size)]
                           for index in range(5 ** size)]) for size in range(1, GROUP + 1)}


def encode(sequence: str) -> np.ndarray:
    """A sequence as codes, one per symbol: 0 to 3 for A, C, G and T (or U), in either case, and
    `INVALID` for any other symbol."""
    return _CODES[np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)]


def log_odds(motif: Motif) -> np.ndarray:
    """The motif's log-odds, in bits, one row per position and one column per letter."""
    background = np.asarray(BACKGROUND)
    softened = (motif.frequencies * motif.sites + PSEUDOCOUNT * background) / (motif.sites + PSEUDOCOUNT)
    return np.log2(softened / background)


def integer_matrix(motif: Motif) -> np.ndarray:
    """The motif's log-odds rounded to whole numbers of `GRID` bits, as integers."""
    return np.rint(log_odds(motif) / GRID).astype(np.int64)


def score_distribution(matrix: np.ndarray) -> tuple[int, np.ndarray]:
    """The distribution of a window's score under the background, for an integer matrix: the lowest
    score a window can have, and the probability of each score from it upwards, one per step."""
    background = np.asarray(BACKGROUND)
    low = int(matrix.min(axis=1).sum())
    distribution = np.ones(1)
    for row in matrix - matrix.min(axis=1, keepdims=True):
        widened = np.zeros(len(distribution) + int(row.max()))
        for letter in range(len(background)):
            widened[row[letter]: row[letter] + len(distribution)] += background[letter] * distribution
        distribution = widened
    return low, distribution


def upper_tail(distribution: np.ndarray) -> np.ndarray:
    """The probability of each score or a higher one, summed from the top so that the smallest
    probabilities keep their precision."""
    return np.cumsum(distribution[::-1])[::-1]


@dataclass(frozen=True)
class _Prepared:
    """A motif made ready to scan: its integer scores in tables of groups of letters, for the given
    strand and (where both strands are scanned) the reverse complement, and the scores that bound
    and threshold them."""

    width: int
    #: The lowest and the highest score a window without an invalid symbol can have.
    low: int
    high: int
    #: The lowest score whose p-value passes the scanner's threshold, or `high + 1` if none does.
    threshold: int
    forward: tuple[tuple[int, int, np.ndarray], ...]
    reverse: tuple[tuple[int, int, np.ndarray], ...] | None


def _prepare(motif: Motif, strands: str, p_threshold: float) -> _Prepared:
    matrix = integer_matrix(motif)
    low, distribution = score_distribution(matrix)
    high = low + len(distribution) - 1
    passing = np.flatnonzero(upper_tail(distribution) <= p_threshold)
    threshold = low + int(passing[0]) if len(passing) else high + 1
    # An invalid symbol scores so low that a window holding it falls below every window without
    # one: below `low` even if every other position scored its highest.
    invalid = low - motif.width * max(int(matrix.max()), 0) - 1
    if motif.width * abs(invalid) >= 2**31:
        raise ValueError(f"Motif {motif.identifier} is too wide, or its scores too extreme, for a "
                         f"scan in 32-bit integers.")
    extended = np.column_stack([matrix, np.full(motif.width, invalid)])
    # Scoring a window's reverse complement with the matrix gives the same score as scoring the
    # window with the matrix reversed and with A and T, and C and G, exchanged.
    complemented = np.column_stack([matrix[::-1, ::-1], np.full(motif.width, invalid)])
    return _Prepared(width=motif.width, low=low, high=high, threshold=threshold,
                     forward=_tables(extended),
                     reverse=_tables(complemented) if strands == "both" else None)


def _tables(extended: np.ndarray) -> tuple[tuple[int, int, np.ndarray], ...]:
    """(first position, group size, table) for each group of up to `GROUP` positions: the table
    holds the summed score of every group of letters, invalid symbols included."""
    groups = []
    for start in range(0, len(extended), GROUP):
        size = min(GROUP, len(extended) - start)
        rows = extended[start: start + size]
        table = rows[np.arange(size), _DIGITS[size]].sum(axis=1)
        groups.append((start, size, table.astype(np.int32)))
    return tuple(groups)


def _batches(lengths: np.ndarray) -> list[np.ndarray]:
    """The rows in batches of similar length, each holding at most `BATCH_LETTERS` letters once
    padded to its longest."""
    batches: list[np.ndarray] = []
    current: list[int] = []
    for row in np.argsort(lengths, kind="stable"):
        if current and (len(current) + 1) * max(int(lengths[row]), 1) > BATCH_LETTERS:
            batches.append(np.array(current))
            current = []
        current.append(int(row))
    if current:
        batches.append(np.array(current))
    return batches


def _batch_codes(encoded: list[np.ndarray], rows: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    """The batch's codes padded to its longest sequence, by group size (for each size up to
    `GROUP`, every group of that many consecutive letters as one number in base 5), with the
    padded length."""
    length = max(len(encoded[row]) for row in rows)
    codes = np.full((len(rows), length), INVALID, dtype=np.uint8)
    for place, row in enumerate(rows):
        codes[place, : len(encoded[row])] = encoded[row]
    coded = {1: codes}
    for size in range(2, GROUP + 1):
        span = length - size + 1
        if span <= 0:
            break
        combined = np.zeros((len(rows), span), dtype=np.uint16)
        for place in range(size):
            combined = combined * 5 + codes[:, place: place + span]
        coded[size] = combined
    return coded, length


def _scan_batch(prepared: _Prepared, coded: dict[int, np.ndarray], length: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each sequence of a batch: its best score (below `low` where no window can be scored),
    the number of positions passing the threshold, and the number of positions scanned."""
    rows = len(coded[1])
    positions = length - prepared.width + 1
    if positions <= 0:
        return (np.full(rows, prepared.low - 1, dtype=np.int64), np.zeros(rows, dtype=np.int64),
                np.zeros(rows, dtype=np.int64))

    def summed(groups: tuple[tuple[int, int, np.ndarray], ...]) -> np.ndarray:
        scores = np.zeros((rows, positions), dtype=np.int32)
        for start, size, table in groups:
            scores += table[coded[size][:, start: start + positions]]
        return scores

    forward = summed(prepared.forward)
    scanned = (forward >= prepared.low).sum(axis=1)
    scores = forward if prepared.reverse is None else np.maximum(forward, summed(prepared.reverse))
    return scores.max(axis=1), (scores >= prepared.threshold).sum(axis=1), scanned


@dataclass(frozen=True)
class ScanSettings:
    """How a scan is summarised: which strands are scored, the p-value a match must reach to be
    counted, the summaries taken, and how a count is reported. The settings have no defaults,
    since each changes what a feature means."""

    strands: str
    p_threshold: float
    summaries: tuple[str, ...]
    normalise: str

    def __post_init__(self) -> None:
        if self.strands not in STRANDS:
            raise ValueError(f"strands must be one of {', '.join(STRANDS)}, not {self.strands!r}.")
        if isinstance(self.p_threshold, bool) or not 0 < self.p_threshold < 1:
            raise ValueError(f"The p-value threshold must be between 0 and 1, not "
                             f"{self.p_threshold!r}.")
        summaries = list(self.summaries)
        unknown = [summary for summary in summaries if summary not in SUMMARIES]
        if unknown or not summaries or len(set(summaries)) != len(summaries):
            raise ValueError(f"summaries must name each of {', '.join(SUMMARIES)} at most once, and "
                             f"at least one of them, not {summaries!r}.")
        if self.normalise not in NORMALISATIONS:
            raise ValueError(f"normalise must be one of {', '.join(NORMALISATIONS)}, not "
                             f"{self.normalise!r}.")
        object.__setattr__(self, "p_threshold", float(self.p_threshold))
        object.__setattr__(self, "summaries", tuple(summary for summary in SUMMARIES
                                                    if summary in summaries))

    def as_dict(self) -> dict[str, Any]:
        return {"strands": self.strands, "p_threshold": self.p_threshold,
                "summaries": list(self.summaries), "normalise": self.normalise}


class Scanner:
    """Scans sequences with a fixed list of motifs under one `ScanSettings`. The result has one
    column per motif and summary (`columns`), ordered motif by motif and, within a motif, in the
    order of `SUMMARIES`."""

    def __init__(self, motifs: Sequence[Motif], settings: ScanSettings) -> None:
        if not motifs:
            raise ValueError("A scanner needs at least one motif.")
        self.motifs = tuple(motifs)
        self.settings = settings
        self._prepared: tuple[_Prepared, ...] | None = None

    def __getstate__(self) -> dict[str, Any]:
        # The tables are rebuilt on first use, so a pickled scanner holds only what defines it.
        return {**self.__dict__, "_prepared": None}

    @property
    def columns(self) -> list[tuple[Motif, str]]:
        """The motif and the summary of each column, in order."""
        return [(motif, summary) for motif in self.motifs for summary in self.settings.summaries]

    def describe(self) -> dict[str, Any]:
        """Everything besides the motifs that decides a value: the settings, and the version and
        constants of the arithmetic."""
        return {"scan_version": SCAN_VERSION, "grid_bits": GRID, "pseudocount": PSEUDOCOUNT,
                "background": list(BACKGROUND), **self.settings.as_dict()}

    def digest(self) -> str:
        """A digest of everything that decides what this scanner computes: its settings, the
        version and constants of the arithmetic, and every motif."""
        payload = {**self.describe(), "motifs": [motif.digest() for motif in self.motifs]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def unreachable(self) -> list[Motif]:
        """The motifs whose best possible match has a p-value above the threshold, so that no
        window can pass it. A motif of four positions, for instance, has a best p-value of 1/256
        and never reaches 1e-4."""
        return [motif for motif, prepared in zip(self.motifs, self._prepare(), strict=True)
                if prepared.threshold > prepared.high]

    def scan(self, sequences: Sequence[str]) -> np.ndarray:
        """One row per sequence and one column per motif and summary (`columns`), as float64."""
        encoded = [encode(sequence) for sequence in sequences]
        shape = (len(encoded), len(self.motifs))
        best = np.empty(shape, dtype=np.int64)
        passing = np.empty(shape, dtype=np.int64)
        scanned = np.empty(shape, dtype=np.int64)
        prepared = self._prepare()
        if encoded:
            for rows in _batches(np.array([len(codes) for codes in encoded])):
                coded, length = _batch_codes(encoded, rows)
                for index, motif in enumerate(prepared):
                    best[rows, index], passing[rows, index], scanned[rows, index] = _scan_batch(
                        motif, coded, length)
        return self._summarised(best, passing, scanned)

    # ---------------------------------------------------------------------------- internals

    def _prepare(self) -> tuple[_Prepared, ...]:
        if self._prepared is None:
            self._prepared = tuple(_prepare(motif, self.settings.strands,
                                            self.settings.p_threshold) for motif in self.motifs)
        return self._prepared

    def _summarised(self, best: np.ndarray, passing: np.ndarray, scanned: np.ndarray) -> np.ndarray:
        summaries = self.settings.summaries
        values = np.zeros((len(best), len(self.motifs) * len(summaries)))
        for index, motif in enumerate(self.motifs):
            for place, summary in enumerate(summaries):
                column = index * len(summaries) + place
                if summary == "best_match":
                    low, distribution = score_distribution(integer_matrix(motif))
                    tail = upper_tail(distribution)
                    found = best[:, index] >= low
                    values[found, column] = -np.log10(tail[best[found, index] - low])
                elif self.settings.normalise == "per_length":
                    values[:, column] = np.divide(passing[:, index], scanned[:, index],
                                                  out=np.zeros(len(best)),
                                                  where=scanned[:, index] > 0)
                else:
                    values[:, column] = passing[:, index]
        return values
