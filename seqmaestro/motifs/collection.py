"""Reading and writing motifs in MEME's text format.

JASPAR publishes its collections in this format, STREME writes the motifs it finds in it, and
Tomtom expects its input in it, so it is the only format this package reads. A collection published
in another format is converted to it when it is fetched
(`experiments/prepare/fetch_motif_collection.py`).

**A motif's columns are A, C, G and T, in that order.** A file in the RNA alphabet (A, C, G, U) is
taken with U as T, and a sequence is scanned with U taken as T too, so a motif from an RNA
collection scans a sequence written in either alphabet. A file in any other alphabet is refused.

**`sites` is the number of sites the frequencies were counted from** (MEME's `nsites`). The scan
weighs a pseudocount against it, so a zero counted from 20 sites is softened more than a zero
counted from 5,000. A motif whose file states no number is taken to have 20 sites, as the MEME
Suite assumes.

**Each row is rescaled to sum to one.** Published files round each frequency to a few decimal
places, so a row sums to one only within that rounding. A row whose sum is further from one than
`ROW_TOLERANCE` marks a damaged file, which is refused.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The letters of a motif's columns, in order.
LETTERS = "ACGT"
#: The alphabets a file may declare. An RNA file's U column is taken as T.
ALPHABETS = ("ACGT", "ACGU")
#: The number of sites taken for a motif whose file states none, as the MEME Suite assumes.
DEFAULT_SITES = 20.0
#: How far from one a row's frequencies may sum before the file is taken to be damaged.
ROW_TOLERANCE = 0.02

_FIELD = re.compile(r"(\w+)\s*=\s*(\S+)")


class MotifFileError(ValueError):
    """A motif file that cannot be read as written. The message names the file and the line."""


@dataclass(frozen=True, eq=False)
class Motif:
    """One motif: its identifier, its name, its frequencies and the number of sites they were
    counted from.

    `identifier` is unique within its file (`MA0139.2` in JASPAR, `1-GATTACA` from STREME). `name`
    is the alternate name the file gives the motif, usually the binding protein's (`CTCF`), or None
    where the file gives none. `frequencies` has one row per position and one column per letter of
    `LETTERS`.
    """

    identifier: str
    name: str | None
    frequencies: np.ndarray
    sites: float

    def __post_init__(self) -> None:
        frequencies = np.array(self.frequencies, dtype=np.float64)
        if frequencies.ndim != 2 or frequencies.shape[1] != len(LETTERS) or len(frequencies) == 0:
            raise ValueError(f"Motif {self.identifier}: the frequencies must have one row per "
                             f"position and {len(LETTERS)} columns, not shape {frequencies.shape}.")
        if not np.isfinite(frequencies).all() or (frequencies < 0).any():
            raise ValueError(f"Motif {self.identifier}: a frequency is negative or not a number.")
        sums = frequencies.sum(axis=1)
        if (np.abs(sums - 1) > ROW_TOLERANCE).any():
            row = int(np.argmax(np.abs(sums - 1)))
            raise ValueError(f"Motif {self.identifier}: row {row + 1} sums to {sums[row]:.4f}, not 1.")
        if not self.sites > 0:
            raise ValueError(f"Motif {self.identifier}: the number of sites must be positive.")
        frequencies = frequencies / sums[:, None]
        frequencies.setflags(write=False)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "sites", float(self.sites))

    @property
    def width(self) -> int:
        return len(self.frequencies)

    @property
    def consensus(self) -> str:
        """The commonest letter at each position."""
        return "".join(LETTERS[index] for index in self.frequencies.argmax(axis=1))

    def digest(self) -> str:
        """A digest of everything that defines the motif: its identifier, name, number of sites
        and every frequency."""
        payload = json.dumps([self.identifier, self.name, self.sites, self.frequencies.tolist()])
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, eq=False)
class Collection:
    """A named set of motifs read from one file, with the digest of that file's bytes."""

    name: str
    motifs: tuple[Motif, ...]
    digest: str

    def __post_init__(self) -> None:
        if not self.motifs:
            raise ValueError(f"The collection {self.name} holds no motif.")
        seen = [motif.identifier for motif in self.motifs]
        repeated = sorted({identifier for identifier in seen if seen.count(identifier) > 1})
        if repeated:
            raise ValueError(f"The collection {self.name} names motif(s) {repeated} more than once.")


def read_collection(path: str | Path, name: str) -> Collection:
    """The motifs of a MEME-format file as a collection named `name`, with the digest of the
    file's bytes."""
    path = Path(path)
    data = path.read_bytes()
    return Collection(name=name, motifs=tuple(read_meme(data.decode(), source=str(path))),
                      digest=hashlib.sha256(data).hexdigest())


def read_meme(text: str, source: str = "<text>") -> list[Motif]:
    """Every motif of a MEME-format text, in file order.

    The text must declare its alphabet on an `ALPHABET=` line, as JASPAR's and STREME's files do.
    Each motif is a `MOTIF` line followed by a `letter-probability matrix:` line and one row per
    position. Other lines that MEME's format allows around them (`strands:`, the background, `URL`)
    are skipped.
    """
    lines = text.splitlines()
    alphabet: str | None = None
    motifs: list[Motif] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("ALPHABET"):
            alphabet = _alphabet(line, source, index)
        elif line.startswith("MOTIF"):
            if alphabet is None:
                raise MotifFileError(f"{source}, line {index + 1}: a motif comes before the file "
                                     f"declares its alphabet.")
            motif, index = _motif(lines, index, source)
            motifs.append(motif)
        index += 1
    if not motifs:
        raise MotifFileError(f"{source} holds no motif.")
    return motifs


def write_meme(motifs: Iterable[Motif], alphabet: str = LETTERS) -> str:
    """MEME-format text for `motifs` in `alphabet` (`ACGT`, or `ACGU` for an RNA file), with a
    uniform background and the strands that suit the alphabet: both for DNA, and the given strand
    alone for RNA."""
    if alphabet not in ALPHABETS:
        raise ValueError(f"The alphabet must be one of {', '.join(ALPHABETS)}, not {alphabet!r}.")
    strands = "+ -" if alphabet == "ACGT" else "+"
    background = " ".join(f"{letter} 0.25" for letter in alphabet)
    out = ["MEME version 4", "", f"ALPHABET= {alphabet}", "", f"strands: {strands}", "",
           "Background letter frequencies", background, ""]
    for motif in motifs:
        out.append(f"MOTIF {motif.identifier}" + (f" {motif.name}" if motif.name else ""))
        out.append(f"letter-probability matrix: alength= 4 w= {motif.width} nsites= {motif.sites:g} E= 0")
        out.extend(" ".join(f"{value:.6f}" for value in row) for row in motif.frequencies)
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------------- internals

def _alphabet(line: str, source: str, index: int) -> str:
    _, found, letters = line.partition("=")
    letters = letters.strip().upper()
    if not found or letters not in ALPHABETS:
        raise MotifFileError(f"{source}, line {index + 1}: the alphabet must be declared as "
                             f"`ALPHABET= ACGT` or `ALPHABET= ACGU`, not {line!r}.")
    return letters


def _motif(lines: list[str], index: int, source: str) -> tuple[Motif, int]:
    """The motif whose `MOTIF` line is at `index`, and the index of its last line."""
    words = lines[index].split()
    if len(words) < 2:
        raise MotifFileError(f"{source}, line {index + 1}: a MOTIF line names its motif.")
    identifier, name = words[1], " ".join(words[2:]) or None
    start = index
    index += 1
    while index < len(lines) and not lines[index].strip().startswith("letter-probability matrix"):
        if lines[index].strip().startswith("MOTIF"):
            break
        index += 1
    if index >= len(lines) or not lines[index].strip().startswith("letter-probability matrix"):
        raise MotifFileError(f"{source}, line {start + 1}: motif {identifier} has no "
                             f"letter-probability matrix.")
    fields = dict(_FIELD.findall(lines[index].partition(":")[2]))
    if fields.get("alength", "4") != "4":
        raise MotifFileError(f"{source}, line {index + 1}: motif {identifier} has "
                             f"{fields['alength']} letters, and a DNA or RNA motif has 4.")
    width = int(fields["w"]) if "w" in fields else None
    sites = float(fields["nsites"]) if "nsites" in fields else DEFAULT_SITES
    rows: list[list[float]] = []
    while width is None or len(rows) < width:
        index += 1
        if index >= len(lines):
            break
        text = lines[index].strip()
        if not text:
            if width is None and rows:
                break
            continue
        try:
            row = [float(value) for value in text.split()]
        except ValueError:
            if width is None and rows:
                index -= 1
                break
            raise MotifFileError(f"{source}, line {index + 1}: motif {identifier} has a row that "
                                 f"is not numbers: {text!r}.") from None
        if len(row) != 4:
            raise MotifFileError(f"{source}, line {index + 1}: motif {identifier} has a row of "
                                 f"{len(row)} values, and needs 4.")
        rows.append(row)
    if width is not None and len(rows) != width:
        raise MotifFileError(f"{source}: motif {identifier} states w= {width} and has {len(rows)} rows.")
    # Both alphabets list their letters in the same order, so an RNA file's U column is the T
    # column as it stands.
    frequencies = np.array(rows)
    try:
        return Motif(identifier, name, frequencies, sites), index
    except ValueError as error:
        raise MotifFileError(f"{source}: {error}") from None
