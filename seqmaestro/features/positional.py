"""One indicator per (position, symbol): *where* a symbol sits, exactly.

It keeps position and loses composition and any shift invariance, so the same motif two bases
along is an entirely different set of columns. `KmerFeaturiser` keeps the complement, which is why
the two are also composed (`positional+kmer`).

**An L1-regularised logistic regression on these columns is a learned position-weight matrix.** Its
coefficients form a PWM over the window, which makes that arm directly comparable to the motif
literature.

**Sequences of different lengths are read in a declared window** (`PositionalWindow`), cut and
padded by one rule, `window_symbols`. Without a window every sequence must have one length, as
position 37 means nothing across sequences of different lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
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

#: Where a positional window is read from: the first base of the sequence, or the last.
ANCHORS = ("start", "end")


@dataclass(frozen=True)
class PositionalWindow:
    """The fixed frame positional features are read in, when sequences differ in length.

    Declared at Stage 1 (`sequences.positional_window`, `docs/dataset.md`). The window is `length`
    bases read from the `anchor`: the first `length` bases with `start`, the last `length` with
    `end`. A longer sequence is cut at the far end, which costs no row. What happens to a shorter
    one is the `pad` decision, which the user confirms at Stage 1: with `pad` it is padded at the
    far end, and without it no sequence may be shorter, so Stage 1 filters the short ones out.
    """

    length: int
    anchor: str
    pad: bool

    def __post_init__(self) -> None:
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length < 1:
            raise ValueError(f"A positional window's length must be a positive whole number, "
                             f"not {self.length!r}.")
        if self.anchor not in ANCHORS:
            raise ValueError(f"A positional window's anchor must be one of {', '.join(ANCHORS)}, "
                             f"not {self.anchor!r}.")
        if not isinstance(self.pad, bool):
            raise ValueError(f"A positional window's pad must be true or false, not {self.pad!r}.")

    def as_dict(self) -> dict[str, Any]:
        return {"length": self.length, "anchor": self.anchor, "pad": self.pad}


class PositionalFeaturiser(Featuriser):
    """One binary column per position and symbol, both learned from the training data.

    The column count is `sequence_length * len(alphabet)`, and neither factor is assumed: a 101 nt
    window over ACGT gives 404 columns, and an alphabet that includes `N` widens the matrix instead
    of silently dropping the symbol. The alphabet is every symbol the training sequences hold, inside
    a window or not, so no symbol Stage 1 found in training can meet the featuriser unseen.

    With a `window`, the positions are the window's and `sequence_length` is its length. A window
    that pads adds a column per position, `pos_<i>_pad`, set where the sequence stops before that
    position. A model can read a sequence's length from those columns, which is a fact about the
    design that the columns make visible, and Stage 1's `sequence_lengths` finding puts to the user.
    """

    def __init__(self, window: PositionalWindow | None = None) -> None:
        super().__init__()
        self.alphabet: list[str] | None = None
        #: The declared window, or None where every sequence has one length.
        self.window = window
        self.sequence_length: int | None = None

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        # `y` is ignored: the layout is a property of the sequences alone. See `Featuriser._fit`.
        sequences = inputs.sequences
        if self.window is None:
            lengths = {len(sequence) for sequence in sequences}
            if len(lengths) > 1:
                raise ValueError(
                    f"Positional featurisation needs every sequence to be the same length, or a "
                    f"positional window; found {len(lengths)} distinct lengths between "
                    f"{min(lengths)} and {max(lengths)}. Position 37 means nothing across sequences "
                    f"of different length. Declare a window at Stage 1 (docs/dataset.md), where "
                    f"whether to pad is a question about the data. `KmerFeaturiser` has no such "
                    f"requirement."
                )
            self.sequence_length = lengths.pop()
        else:
            self.sequence_length = self.window.length
            for sequence in sequences:
                self._symbols(sequence)  # refuses a short sequence when the window does not pad

        self.alphabet = alphabet_of(sequences)
        check_symbols_are_nameable(self.alphabet)

        self._position_of = {symbol: i for i, symbol in enumerate(self._columns_per_position)}
        return [
            FeatureRecord(
                name=f"pos_{position}_{symbol}",
                description=self._description(position, symbol),
                kind="positional",
                origin=ORIGIN_SEQMAESTRO,
                params={"position": position, "symbol": symbol},
                provenance=PROVENANCE_SELECTED,
            )
            for position in range(self.sequence_length)
            for symbol in self._columns_per_position
        ]

    def _transform(self, inputs: Inputs) -> np.ndarray:
        sequences = inputs.sequences
        check_symbols(sequences, set(self.alphabet), type(self).__name__)

        if self.window is None:
            wrong = [len(sequence) for sequence in sequences if len(sequence) != self.sequence_length]
            if wrong:
                raise ValueError(
                    f"Fitted on length {self.sequence_length}; {len(wrong)} sequence(s) differ, "
                    f"first of length {wrong[0]}."
                )

        width = len(self._columns_per_position)
        # Indicators, so the largest value any cell can take is 1, which is the exact bound.
        values = np.zeros((len(sequences), self.sequence_length * width), dtype=count_dtype(1))
        for row, sequence in enumerate(sequences):
            columns = [position * width + self._position_of[symbol]
                       for position, symbol in enumerate(self._symbols(sequence))]
            values[row, columns] = 1
        return values

    def describe(self) -> dict[str, Any]:
        described = {
            "kind": "positional",
            "alphabet": self.alphabet,
            "sequence_length": self.sequence_length,
            "n_features": len(self._records) if self.is_fitted else None,
        }
        if self.window is not None:
            described["window"] = self.window.as_dict()
        return described

    # ---------------------------------------------------------------------------- internals

    @property
    def _columns_per_position(self) -> list[str]:
        """The symbols each position has a column for: the alphabet, and the pad marker when the
        window pads."""
        pads = self.window is not None and self.window.pad
        return [*self.alphabet, PAD] if pads else list(self.alphabet)

    def _symbols(self, sequence: str) -> list[str] | str:
        if self.window is None:
            return sequence
        return window_symbols(sequence, self.window.length, self.window.anchor, self.window.pad)

    def _description(self, position: int, symbol: str) -> str:
        if self.window is None:
            return f"1 when position {position} (0-based) of the sequence is {symbol}, else 0"
        length, anchor = self.window.length, self.window.anchor
        frame = (f"the window of the first {length} bases" if anchor == "start"
                 else f"the window of the last {length} bases")
        if symbol != PAD:
            return f"1 when position {position} (0-based) of {frame} is {symbol}, else 0"
        shorter = position + 1 if anchor == "start" else length - position
        return (f"1 when the sequence is shorter than {shorter} bases, so position {position} "
                f"(0-based) of {frame} is padding, else 0")


#: What `window_symbols` gives for a window position the sequence does not reach, when the window
#: pads. It is longer than one character, so it cannot be mistaken for a symbol of the sequence.
PAD = "pad"


# DEFT carries a verbatim copy of this function (`deft.utils.window`), so that a position means
# the same in a DEFT tree as in every other arm, and a test checks that the two agree. Change both,
# or neither.
def window_symbols(sequence: str, length: int, anchor: str, pad: bool) -> list[str]:
    """The symbol at each position of a window over `sequence`: the one cut-and-pad rule.

    The window is `length` positions read from the `anchor`. With `start`, position 0 is the
    sequence's first base; with `end`, position `length - 1` is its last. A longer sequence is cut
    at the far end. A shorter one is padded at the far end with `PAD` when `pad` is set, and
    refused otherwise.
    """
    if anchor not in ("start", "end"):
        raise ValueError(f"A window's anchor is start or end, not {anchor!r}.")
    if len(sequence) >= length:
        return list(sequence[:length] if anchor == "start" else sequence[len(sequence) - length:])
    if not pad:
        raise ValueError(f"A sequence of {len(sequence)} bases is shorter than the {length}-base "
                         f"window, which is declared without padding.")
    padding = [PAD] * (length - len(sequence))
    return [*sequence, *padding] if anchor == "start" else [*padding, *sequence]
