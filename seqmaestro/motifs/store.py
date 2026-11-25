"""The scans a run has made, kept under the run so that each sequence is scanned once.

A Stage 2 cell transforms its fitting rows, the validation rows and every training row. The cells
of every arm, seed and label condition transform the same sequences again, and so do the reduction
and the reliability panel. Scanning against a collection of a thousand motifs is slow, so a run that
scanned in every process would scan each sequence dozens of times. The store keeps each scan under
`<run>/motifs/`, and a featuriser handed the store takes from it every sequence already scanned and
scans only the rest (`scan_with_store`).

**An entry is found by the scanner's digest and the sequence's.** The scanner's digest
(`Scanner.digest`) covers every motif and every setting that changes a value, so two featurisers
that would compute different numbers never share an entry. Under one scanner's digest, a sequence is
found by a digest of its letters, so its scan is found whichever split its row is in and whatever
the order of the rows.

**The store only grows.** New scans are written as a chunk: a directory written under a temporary
name and renamed into place once complete, so a reader never finds part of one. Two processes that
scan the same sequence at once each write a chunk, and their values agree, since a scan depends on
the sequence and the scanner alone. The first chunk found holding a sequence is used.

**A store whose directory does not exist is not used**, and nothing is written there. The run makes
`<run>/motifs/` before its featurisers are fitted, and a fitted featuriser keeps the store's path. A
featuriser loaded after the run has moved, or on another machine, scans every row itself.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from seqmaestro import files

if TYPE_CHECKING:
    from seqmaestro.motifs.scan import Scanner

#: Where under the store a scanner's chunks are kept, one directory per scanner digest.
SCANS_DIR = "scans"
#: The file beside a scanner's chunks that says what its digest stands for.
DESCRIPTION_FILE = "scanner.json"
#: The two files of a chunk: the digests of its sequences and their values, row for row.
DIGESTS_FILE = "digests.npy"
VALUES_FILE = "values.npy"


def sequence_digest(sequence: str) -> bytes:
    """A sequence's letters as a digest of 32 hexadecimal characters."""
    return hashlib.blake2b(sequence.encode(), digest_size=16).hexdigest().encode()


class ScanStore:
    """The scans kept under one directory, `<run>/motifs/`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        # For each scanner digest: the chunks read so far, by name, and where each sequence is.
        self._chunks: dict[str, dict[str, np.ndarray]] = {}
        self._where: dict[str, dict[bytes, tuple[str, int]]] = {}

    def __getstate__(self) -> dict[str, Any]:
        # A pickled store keeps only its path, and reads its chunks again where it is loaded.
        return {"root": self.root}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["root"])

    def usable(self) -> bool:
        """Whether the store's directory exists, which the run makes when it starts."""
        return self.root.is_dir()

    def fill(self, key: str, digests: Sequence[bytes], values: np.ndarray) -> np.ndarray:
        """Copy into `values` the row of each sequence the store holds under `key`, and return
        which rows were found. A sequence not found is looked for again in the chunks written
        since the store was last read, by this process or another."""
        found = self._fill(key, digests, values)
        if not found.all():
            self._read(key)
            found = self._fill(key, digests, values)
        return found

    def add(self, key: str, digests: Sequence[bytes], values: np.ndarray,
            description: dict[str, Any]) -> None:
        """Keep `values` (one row per digest) under `key` as one new chunk. `description` says
        what the key stands for, and is written beside the first chunk."""
        directory = self.root / SCANS_DIR / key
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / DESCRIPTION_FILE).exists():
            files.write_json(directory / DESCRIPTION_FILE, description)
        name = f"chunk-{os.getpid()}-{uuid.uuid4().hex}"
        partial = directory / f".{name}.partial"
        partial.mkdir()
        np.save(partial / DIGESTS_FILE, np.array(digests, dtype="S32"))
        np.save(partial / VALUES_FILE, np.asarray(values, dtype=np.float64))
        os.rename(partial, directory / name)

    # ---------------------------------------------------------------------------- internals

    def _fill(self, key: str, digests: Sequence[bytes], values: np.ndarray) -> np.ndarray:
        where = self._where.get(key)
        if where is None:
            self._read(key)
            where = self._where[key]
        found = np.zeros(len(digests), dtype=bool)
        for row, digest in enumerate(digests):
            place = where.get(digest)
            if place is not None:
                chunk, index = place
                values[row] = self._chunks[key][chunk][index]
                found[row] = True
        return found

    def _read(self, key: str) -> None:
        """Read the chunks of `key` not yet read, in the order of their names."""
        chunks = self._chunks.setdefault(key, {})
        where = self._where.setdefault(key, {})
        directory = self.root / SCANS_DIR / key
        if not directory.is_dir():
            return
        for chunk in sorted(path for path in directory.iterdir()
                            if path.is_dir() and not path.name.startswith(".")):
            if chunk.name in chunks:
                continue
            digests = np.load(chunk / DIGESTS_FILE)
            chunks[chunk.name] = np.load(chunk / VALUES_FILE, mmap_mode="r")
            for index, digest in enumerate(digests.tolist()):
                where.setdefault(digest, (chunk.name, index))


def scan_with_store(scanner: Scanner, sequences: Sequence[str], store: ScanStore | None) -> np.ndarray:
    """`scanner.scan(sequences)`, taking from `store` every sequence it holds and adding the rest.

    With a usable store, a sequence that appears more than once is scanned once. Without one,
    every sequence is scanned.
    """
    if store is None or not store.usable():
        return scanner.scan(sequences)
    key = scanner.digest()
    digests = [sequence_digest(sequence) for sequence in sequences]
    values = np.zeros((len(sequences), len(scanner.columns)))
    found = store.fill(key, digests, values)
    if not found.all():
        first: dict[bytes, int] = {}
        for row in np.flatnonzero(~found):
            first.setdefault(digests[row], int(row))
        scanned = scanner.scan([sequences[row] for row in first.values()])
        store.add(key, list(first), scanned,
                  description={**scanner.describe(),
                               "motifs": [motif.identifier for motif in scanner.motifs]})
        position = {digest: index for index, digest in enumerate(first)}
        for row in np.flatnonzero(~found):
            values[row] = scanned[position[digests[row]]]
    return values
