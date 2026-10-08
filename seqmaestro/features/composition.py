"""The fraction of the sequence each symbol makes up: *how much* of each base, and nothing more.

This is the featurisation of the composition baseline, which every run fits before its arms, so an
arm's score can be compared with what base composition alone achieves on the same rows. It is the
plainest description of a sequence there is: no position, no order, no neighbours. A dataset whose
target goes with composition is predicted by it, whatever else the target goes with, and an arm
that scores no better than it has not shown that it found anything composition does not already
give.

It is not a pool member. The baseline is a reference line and not an arm, and an arm handed these
columns would be measured against itself.

Each column is a fraction and not a count, so sequences of different lengths are comparable. The
fractions of a row sum to one, so the columns are collinear by construction. The baseline's model
is regularised and fits on standardised columns, which copes with that.
"""

from __future__ import annotations

from collections import Counter
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
)


class CompositionFeaturiser(Featuriser):
    """One column per symbol seen in training: the fraction of the sequence it makes up.

    The alphabet is learned from the training sequences, like every other featuriser's, so a symbol
    such as `N` becomes a column when the data holds it rather than being assumed away.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alphabet: list[str] | None = None

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        # `y` is ignored: the alphabet is a property of the sequences alone. See `Featuriser._fit`.
        sequences = inputs.sequences
        self.alphabet = alphabet_of(sequences)
        check_symbols_are_nameable(self.alphabet)
        return [
            FeatureRecord(
                name=f"fraction_{symbol}",
                description=f"the fraction of the sequence's positions that are {symbol}",
                kind="composition",
                origin=ORIGIN_SEQMAESTRO,
                params={"symbol": symbol},
                provenance=PROVENANCE_SELECTED,
            )
            for symbol in self.alphabet
        ]

    def _transform(self, inputs: Inputs) -> np.ndarray:
        sequences = inputs.sequences
        check_symbols(sequences, set(self.alphabet), type(self).__name__)
        values = np.zeros((len(sequences), len(self.alphabet)))
        for row, sequence in enumerate(sequences):
            counts = Counter(sequence)
            values[row] = [counts[symbol] / len(sequence) for symbol in self.alphabet]
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "composition",
            "alphabet": self.alphabet,
            "n_features": len(self._records) if self.is_fitted else None,
        }
