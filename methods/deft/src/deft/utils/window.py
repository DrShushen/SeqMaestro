"""The positional window: how sequences of different lengths are read position by position.

Positional features index into a sequence, and position 37 means nothing across sequences of
different lengths. A dataset whose sequences differ in length can declare a window
(`dataset.positional_window`): `length` positions read from the `start` or the `end` of each
sequence. A longer sequence is cut at the far end. A shorter one is padded at the far end when the
window pads, and refused when it does not.

`window_symbols` is a verbatim copy of `seqmaestro.features.positional.window_symbols`, so that a
position means the same in a DEFT tree as in every other arm SeqMaestro runs. A SeqMaestro test
checks that the two agree: change both, or neither. This module imports nothing, so that test can
load it without DEFT's environment.
"""

from __future__ import annotations

#: What `window_symbols` gives for a window position the sequence does not reach, when the window
#: pads. It is longer than one character, so it cannot be mistaken for a symbol of the sequence.
PAD = "pad"

#: Where a window is read from.
ANCHORS = ("start", "end")


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


def checked_window(window) -> dict | None:
    """A declared window as `{length, anchor, pad}`, checked, or None where none is declared.

    `pad` has no default: whether a sequence shorter than the window is padded or refused is a
    decision about the data, so a window must state it.
    """
    if window is None:
        return None
    window = dict(window)
    if set(window) != {"length", "anchor", "pad"}:
        raise ValueError(f"A positional window states length, anchor and pad, and nothing else; "
                         f"got {sorted(window)}.")
    length, anchor, pad = window["length"], window["anchor"], window["pad"]
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"A positional window's length must be a positive whole number, not "
                         f"{length!r}.")
    if anchor not in ANCHORS:
        raise ValueError(f"A positional window's anchor must be one of {', '.join(ANCHORS)}, not "
                         f"{anchor!r}.")
    if not isinstance(pad, bool):
        raise ValueError(f"A positional window's pad must be true or false, not {pad!r}.")
    return {"length": length, "anchor": anchor, "pad": pad}
