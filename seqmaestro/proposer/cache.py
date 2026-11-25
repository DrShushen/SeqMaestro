"""Propose once per arm, task, settings, rows and labels, and reuse the result wherever it is asked
for again.

**The key identifies the request.** Anything that would change the proposal is in the key, the arm
included (`Proposer.arm`), so each arm's proposal is made for that arm, and two arms never share
one. Within an arm, the iterative source asks again for the blind round that its pool already
holds, and that request is answered from the cache.

**A run that calls a language model can fail part-way through**, and when it is resumed it should
not pay again for the proposals it already has.

**Processes running side by side share the cache.** The arms of one run read and write one cache
from several processes at once. The driver gives each combination of arm, label condition and seed
to one process, but a process started by hand may ask for an entry another is making. An entry is
therefore looked up and, when absent, made while its key is held against the other processes
(`held`), so a second process asking for an entry the first is making waits and then takes it, and
is never answered differently. An entry is written whole, in one rename, so a process never takes a
part-written one.

The cache is a plain keyed store: a directory, one file per key, no expiry and no invalidation
logic. The key holds everything a proposal depends on that a run, or an edit of the prompts, can
change: the task description, the settings, the guard, the language model and the effort it
reasons at, the seed, the rows, the feedback, the arm and the wording of the prompts
(`Proposer.key_for`). It leaves out the code around them: the template `render_request` assembles a
request from, and the precision at which a feature's values are compared. A run continued after a
change to one of those is answered from the proposals made before it. A cached entry is a
`features.json`, so it is validated when it is read by the code that validates every other feature
file, and can be inspected without special tools.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from seqmaestro.features.base import FeatureRecord, as_inputs
from seqmaestro.harness.features import load_features, save_features


def proposal_key(**parts) -> str:
    """A digest of everything that would change the answer.

    `Proposer.key_for` passes the task context, the settings, the guard policy, a digest of the
    prompt's wording, the language model with its effort, the seed, digests of the rows with their
    labels and of the feedback, and, where there is one, the arm and the columns block. Two requests
    that differ only in something omitted here would silently share a cache entry, so when in doubt,
    add a part instead of removing one.
    """
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def digest_feedback(feedback: dict | None) -> str:
    """A digest of what a fitted model told the proposer, so that a later round gets its own cache
    entry.

    A blind round has no feedback, which digests to a constant. The blind round of the iterative
    source therefore has the key of the proposal its arm's pool already holds, and is answered from
    the cache. The arm is in the key as well (`Proposer.key_for`), so two arms never share an entry.
    The feedback is dumped as sorted JSON and hashed whole, so a model that adds a new signal
    changes the key without any change to this function.
    """
    if not feedback:
        return "none"
    return hashlib.sha256(
        json.dumps(feedback, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def digest_rows(inputs, y=None) -> str:
    """A digest of the rows a proposal was made from: their sequences, their covariates and their
    labels. It takes `Inputs`, or the sequences alone.

    Including the labels gives a permuted cell its own proposals, separate from those of the real
    cell. The covariates are included only where the rows have some.
    """
    inputs = as_inputs(inputs)
    hasher = hashlib.sha256()
    for sequence in inputs.sequences:
        hasher.update(str(sequence).encode())
        hasher.update(b"\x00")
    if inputs.covariate_columns:
        hasher.update(b"|covariates|")
        hasher.update(json.dumps(list(inputs.covariate_columns)).encode())
        hasher.update(np.ascontiguousarray(inputs.covariates.to_numpy(dtype=np.float64)).tobytes())
    if y is not None:
        hasher.update(b"|labels|")
        for label in y:
            hasher.update(str(label).encode())
            hasher.update(b"\x00")
    return hasher.hexdigest()[:32]


class ProposalCache:
    """One directory, one file per key. `None` for the directory disables caching entirely."""

    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory is not None else None

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def path(self, key: str) -> Path | None:
        return None if self.directory is None else self.directory / f"{key}.json"

    def get(self, key: str) -> tuple[list[FeatureRecord], dict] | None:
        path = self.path(key)
        if path is None or not path.exists():
            return None
        return load_features(path)

    def put(self, key: str, records: list[FeatureRecord], **header) -> Path | None:
        path = self.path(key)
        if path is None:
            return None
        partial = save_features(path.with_name(f"{path.name}.{os.getpid()}.partial"), records,
                                importance_kind=None, cache_key=key, **header)
        partial.replace(path)
        return path

    @contextmanager
    def held(self, key: str) -> Iterator[None]:
        """Hold `key` against every other process using this cache, until the block ends. A
        process blocks here while another holds the key."""
        if self.directory is None:
            yield
            return
        locks = self.directory / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / key).open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
