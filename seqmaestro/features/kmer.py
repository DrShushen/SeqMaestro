"""Counts of overlapping substrings: *what* a sequence contains, regardless of where.

It complements `PositionalFeaturiser`, because a count is shift-invariant: a motif counts the same
wherever it sits, and where it sat is lost.

`k` is a range the caller states, inclusive at both ends. The vocabulary is **learned from the
training sequences** instead of enumerated over the alphabet, for two reasons. The complete product
space is `|alphabet|^k` summed over the range, which is 5,460 columns for four symbols up to k=6
but 19,530 for five, and soon grows past anything usable. Learning it also makes the vocabulary a
fitted property, learned from the fitting rows as the discovered motifs and the kept motif counts
are (`seqmaestro.features.motif`).

**A count grows with the sequence.** Where sequences differ in length, a longer one holds more of
every k-mer, so a count also measures the length. `normalise="per_length"` divides each count by
the number of k-mer positions in the sequence, `length - k + 1`, which gives a rate, and
`normalise="counts"` leaves the counts as they are. On sequences of one length the two differ only
by a constant factor per k. A run states which it takes (`sources.kmer.normalise`), so the class
has no default for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from seqmaestro.features.base import (
    ORIGIN_SEQMAESTRO,
    PROVENANCE_SELECTED,
    FeatureRecord,
    Featuriser,
    Inputs,
    alphabet_of,
    check_symbols,
    check_symbols_are_nameable,
    count_dtype,
)

#: How a count is reported: as it is, or per k-mer position in the sequence.
NORMALISATIONS = ("counts", "per_length")


class KmerFeaturiser(Featuriser):
    """One count column per *k*-mer seen in training, for every `k` in `k_range`.

    `k_range` is inclusive at both ends: `(1, 6)` means k = 1, 2, 3, 4, 5 and 6. `normalise` is one
    of `NORMALISATIONS` (see the module docstring). A sequence shorter than some k holds no k-mer
    of that length.
    """

    def __init__(self, k_range: tuple[int, int], normalise: str) -> None:
        super().__init__()
        low, high = k_range
        if low < 1:
            raise ValueError(f"k must be at least 1; got {low}.")
        if high < low:
            raise ValueError(f"k_range must be (low, high) with high >= low; got {k_range}.")
        if normalise not in NORMALISATIONS:
            raise ValueError(f"normalise must be one of {', '.join(NORMALISATIONS)}; got {normalise!r}.")

        self.k_range = (low, high)
        self.k_values = list(range(low, high + 1))
        self.normalise = normalise
        #: Training sequences shorter than the largest k, which hold none of its k-mers.
        self.n_shorter_than_k = 0
        self.alphabet: list[str] | None = None
        self.vocabulary: list[str] | None = None

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        # `y` is ignored: the vocabulary is a property of the sequences alone. See `Featuriser._fit`.
        sequences = inputs.sequences
        self.alphabet = alphabet_of(sequences)
        check_symbols_are_nameable(self.alphabet)

        longest = max(len(sequence) for sequence in sequences)
        if longest < self.k_range[1]:
            raise ValueError(
                f"k goes up to {self.k_range[1]} but the longest training sequence is {longest} "
                f"long, so those k-mers can never occur. Lower k_range."
            )
        self.n_shorter_than_k = sum(len(sequence) < self.k_range[1] for sequence in sequences)

        seen: set[str] = set()
        for sequence in sequences:
            for k in self.k_values:
                seen.update(sequence[start : start + k] for start in range(len(sequence) - k + 1))

        # Sorted by length then lexically, so column order is deterministic across runs and the
        # matrix is one block per k, the smallest k first.
        self.vocabulary = sorted(seen, key=lambda kmer: (len(kmer), kmer))
        self._column_of = {kmer: column for column, kmer in enumerate(self.vocabulary)}

        return [
            FeatureRecord(
                name=f"kmer_{kmer}",
                description=self._description(kmer),
                kind="kmer",
                origin=ORIGIN_SEQMAESTRO,
                # A normalised feature states it in its recipe, so the recipe differs from a count's.
                params={"kmer": kmer, "k": len(kmer),
                        **({"normalise": self.normalise} if self.normalise != "counts" else {})},
                provenance=PROVENANCE_SELECTED,
            )
            for kmer in self.vocabulary
        ]

    def _transform(self, inputs: Inputs) -> np.ndarray:
        sequences = inputs.sequences
        check_symbols(sequences, set(self.alphabet), type(self).__name__)

        # The dtype comes from *this* input's longest sequence, not from what fit happened to see.
        # A sequence of length L holds at most L - k + 1 k-mers, maximised at the smallest k, so
        # that is the bound. Choosing it per call means a longer validation sequence cannot
        # overflow a width picked on training data. The values are identical either way, and only
        # the storage differs.
        longest = max(len(sequence) for sequence in sequences)
        values = np.zeros(
            (len(sequences), len(self.vocabulary)),
            dtype=count_dtype(max(longest - self.k_range[0] + 1, 1)),
        )

        for row, sequence in enumerate(sequences):
            columns = [
                self._column_of[kmer]
                for k in self.k_values
                for start in range(len(sequence) - k + 1)
                # A k-mer of known symbols that was never seen in training is rightly left out,
                # because the model learned nothing from it. An unknown *symbol*, by contrast, is a
                # fault in the data, and `check_symbols` above has already raised on that.
                if (kmer := sequence[start : start + k]) in self._column_of
            ]
            if columns:
                values[row] = np.bincount(columns, minlength=len(self.vocabulary))
        if self.normalise == "per_length":
            return self._per_length(values, sequences)
        return values

    def describe(self) -> dict[str, Any]:
        described = {
            "kind": "kmer",
            "k_range": list(self.k_range),
            "alphabet": self.alphabet,
            "vocabulary_size": len(self.vocabulary) if self.vocabulary else None,
            "n_features": len(self._records) if self.is_fitted else None,
        }
        # Stated only where they differ from plain counts over sequences long enough for every k.
        if self.normalise != "counts":
            described["normalise"] = self.normalise
        if self.n_shorter_than_k:
            described["n_shorter_than_k"] = self.n_shorter_than_k
        return described

    # ---------------------------------------------------------------------------- internals

    def _description(self, kmer: str) -> str:
        if self.normalise == "counts":
            return f"number of times the {len(kmer)}-mer {kmer} occurs in the sequence, overlapping"
        return (f"occurrences of the {len(kmer)}-mer {kmer} per {len(kmer)}-mer position in the "
                f"sequence (the count over length - {len(kmer) - 1}), overlapping")

    def _per_length(self, counts: np.ndarray, sequences) -> np.ndarray:
        """Each count divided by the number of k-mer positions in its sequence. A sequence too short
        to hold any k-mer of a length has none, so its rate is zero."""
        lengths = np.array([len(sequence) for sequence in sequences], dtype=np.float64)
        ks = np.array([len(kmer) for kmer in self.vocabulary], dtype=np.float64)
        positions = lengths[:, None] - ks[None, :] + 1
        return np.divide(counts, positions, out=np.zeros(counts.shape), where=positions > 0)
