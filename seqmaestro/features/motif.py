"""Motif features: how strongly, and how often, each motif matches a sequence.

A motif is a position frequency matrix (`seqmaestro.motifs`). Each feature is one motif and one
summary of its scan over the whole sequence (`seqmaestro.motifs.scan`): the strength of its best
match, `-log10` of that match's p-value, or the number of positions where it matches at the
threshold. Both summaries give features of kind `pwm_scan`, and their `params` say which motif,
which summary and which source.

`KnownMotifFeaturiser` is the source `motif_known`: every motif of a published collection, such as
JASPAR's for the transcription factors of one taxon, or CisBP-RNA's for the RNA-binding proteins of
one species. Its features draw on what is known about the factors: each motif is a factor's
measured binding preference, named after the factor.

`DiscoveredMotifFeaturiser` is the source `motif_discovered`: the motifs STREME finds in the fitting
rows, between the two classes of a binary target or the two ends of a continuous one, in both
directions (`seqmaestro.motifs.discovery`). It is the only featuriser here that learns from the
labels, and it is handed them as every featuriser is: the cell's labels, shuffled in a permuted
cell, so a permuted cell's motifs are found on shuffled labels. Its features are named for what
was found, such as `streme_up_1_GATTACA_best`: the direction, the rank STREME gave the motif in
that direction, and its consensus. The closest known motif, where the run names a collection to
compare with, is kept in the record's `extra` and named nowhere else, so a language model is never
shown a factor's name through it.

**What is learned at `fit`.** Once the motifs are chosen, a scan depends on the sequence and the
motif alone, so a column's values are not learned from the labels. What is learned from the fitting
rows is which count columns are kept: a count is kept where at least `min_match_share` of the
fitting rows hold a match, and dropped otherwise, as a k-mer never seen in training has no column.
With `min_match_share` at 0.01, this matches the rule the reliability panel applies first, which
drops a column whose commonest value covers more than 99% of the rows
(`seqmaestro.reliability.prefilter`), so a model is never offered a count that the panel would
discard. The count of a motif too short ever to reach the threshold is dropped in the same way. The
best-match column of every motif is kept, because every row has a best match, and it measures how
close the sequence comes to a match even where its count of matches is zero.

**Scans are kept in the run's store** where the featuriser is given one (`seqmaestro.motifs.store`),
so each sequence is scanned once per run. Without a store, the featuriser keeps its last scan in
memory, so that `fit_transform` does not scan the fitting rows twice.

**Names.** A known motif's features are named `motif_<identifier>_<name>_<summary>`, with every
symbol an identifier cannot hold turned into an underscore: JASPAR's CTCF matrix `MA0139.2` gives
`motif_MA0139_2_CTCF_best` and `motif_MA0139_2_CTCF_count`. The identifier keeps two matrices for
one factor apart, and the name says what a reader would look up. A motif that belongs to several
proteins names them joined by a slash, and its features are named after the first, whereas the
description and `params` name all of them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from seqmaestro.features.base import (
    ORIGIN_SEQMAESTRO,
    PROVENANCE_SELECTED,
    FeatureRecord,
    Featuriser,
    Inputs,
)
from seqmaestro.motifs import discovery
from seqmaestro.motifs.collection import Collection, Motif
from seqmaestro.motifs.discovery import DOWN, UP, DiscoverySettings, Found
from seqmaestro.motifs.scan import INVALID as INVALID_CODE
from seqmaestro.motifs.scan import Scanner, ScanSettings, encode
from seqmaestro.motifs.store import ScanStore, scan_with_store

#: The kind of every motif feature, whichever source made it.
KIND_MOTIF = "pwm_scan"
#: The source of the motifs of a published collection, as a pool names it.
MOTIF_KNOWN = "motif_known"
#: The source of the motifs that discovery finds in the fitting rows.
MOTIF_DISCOVERED = "motif_discovered"
#: What a summary adds to a feature's name.
SUFFIXES = {"best_match": "best", "count": "count"}

_NOT_IN_A_NAME = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class KnownMotifs:
    """What the source `motif_known` scans with: a collection, how its scans are summarised, and
    the share of fitting rows that must hold a match for a count to be kept."""

    collection: Collection
    settings: ScanSettings
    min_match_share: float


def name_part(text: str) -> str:
    """`text` as part of an identifier: each run of symbols an identifier cannot hold becomes one
    underscore, and none is left at either end."""
    return _NOT_IN_A_NAME.sub("_", text).strip("_")


class _MotifFeaturiser(Featuriser):
    """What both motif sources share: a scanner over a list of motifs, the store, and the columns
    kept at `fit`. A subclass says which motifs, and how its features are named and described."""

    #: The source a pool names, recorded in each feature's `params`.
    source: str

    def __init__(self, settings: ScanSettings, store: ScanStore | None,
                 min_match_share: float) -> None:
        super().__init__()
        if isinstance(min_match_share, bool) or not 0 <= min_match_share < 1:
            raise ValueError(f"min_match_share must be at least 0 and below 1, not "
                             f"{min_match_share!r}.")
        self.settings = settings
        self.store = store
        self.min_match_share = float(min_match_share)
        self.scanner: Scanner | None = None
        self._kept: np.ndarray | None = None
        #: Facts about the fitting rows, for `describe`.
        self.fitted_facts: dict[str, Any] = {}
        self._last: tuple[str, np.ndarray] | None = None

    def __getstate__(self) -> dict[str, Any]:
        # The last scan is left out, because it serves only the process that made it and can be large.
        return {**self.__dict__, "_last": None}

    # ------------------------------------------------------------------ for the subclasses

    def _motifs(self, inputs: Inputs, y: np.ndarray | None) -> list[Motif]:
        raise NotImplementedError

    def _record(self, motif: Motif, summary: str) -> FeatureRecord:
        raise NotImplementedError

    # ---------------------------------------------------------------------------- the work

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        self.scanner = Scanner(self._motifs(inputs, y), self.settings)
        values = self._scan(list(inputs.sequences))
        columns = self.scanner.columns
        matched = (values > 0).mean(axis=0)
        kept = [column for column, (_, summary) in enumerate(columns)
                if summary != "count" or (matched[column] > 0
                                          and matched[column] >= self.min_match_share)]
        self._kept = np.array(kept, dtype=np.int64)
        self.fitted_facts = self._facts(inputs, len(columns) - len(kept))
        return [self._record(*columns[column]) for column in kept]

    def _transform(self, inputs: Inputs) -> np.ndarray:
        return self._scan(list(inputs.sequences))[:, self._kept]

    def _scan(self, sequences: list[str]) -> np.ndarray:
        """The scan of every column for these sequences, from the store where there is one."""
        if self.store is not None and self.store.usable():
            return scan_with_store(self.scanner, sequences, self.store)
        key = hashlib.blake2b("\0".join(sequences).encode(), digest_size=16).hexdigest()
        if self._last is None or self._last[0] != key:
            self._last = (key, self.scanner.scan(sequences))
        return self._last[1]

    def _facts(self, inputs: Inputs, dropped: int) -> dict[str, Any]:
        sequences = inputs.sequences
        widest = max(motif.width for motif in self.scanner.motifs)
        lengths = sequences.str.len()
        return {
            "n_motifs": len(self.scanner.motifs),
            "count_columns_dropped": dropped,
            "motifs_below_the_threshold": len(self.scanner.unreachable()),
            "widest_motif": widest,
            "fit_rows_shorter_than_the_widest_motif": int((lengths < widest).sum()),
            "fit_rows_with_other_symbols": int(sum((encode(sequence) == INVALID_CODE).any()
                                                   for sequence in sequences)),
        }

    def _strand_words(self) -> str:
        return "on either strand" if self.settings.strands == "both" else "on the strand given"

    def _summary_words(self, who: str, summary: str) -> str:
        """The description of one summary of the motif `who` names."""
        strands = self._strand_words()
        if summary == "best_match":
            return (f"the strength of the best match to {who} anywhere in the sequence, {strands}: "
                    f"-log10 of that match's p-value")
        counted = (f"the number of positions where {who} matches at p <= "
                   f"{self.settings.p_threshold:g}, {strands}")
        if self.settings.normalise == "per_length":
            return f"{counted}, divided by the number of positions scanned"
        return counted

    def _params(self, summary: str) -> dict[str, Any]:
        """The recipe's settings for one summary. The threshold and the normalisation change what a
        count is and nothing about a best match, so only a count's recipe holds them."""
        params: dict[str, Any] = {"source": self.source, "summary": summary,
                                  "strands": self.settings.strands}
        if summary == "count":
            params["p_threshold"] = self.settings.p_threshold
            if self.settings.normalise != "counts":
                params["normalise"] = self.settings.normalise
        return params

    def _described(self) -> dict[str, Any]:
        return {
            **(self.scanner.describe() if self.scanner is not None
               else self.settings.as_dict()),
            **self.fitted_facts,
            "n_features": len(self._records) if self.is_fitted else None,
        }


class KnownMotifFeaturiser(_MotifFeaturiser):
    """Two columns per motif of a collection: the strength of its best match and its count of
    matches, as `settings` specifies. The featuriser is built with the collection, so a fitted
    featuriser holds every motif it scans with."""

    source = MOTIF_KNOWN

    def __init__(self, collection: Collection, settings: ScanSettings, *, min_match_share: float,
                 store: ScanStore | None = None) -> None:
        super().__init__(settings, store, min_match_share)
        self.collection = collection

    def _motifs(self, inputs: Inputs, y: np.ndarray | None) -> list[Motif]:
        # `y` is ignored: the collection decides which motifs are scanned. See `Featuriser._fit`.
        return list(self.collection.motifs)

    def _record(self, motif: Motif, summary: str) -> FeatureRecord:
        named = f" ({motif.name})" if motif.name else ""
        who = f"the {self.collection.name} motif {motif.identifier}{named}"
        first = (motif.name or "").split("/")[0]
        parts = ["motif", name_part(motif.identifier), name_part(first), SUFFIXES[summary]]
        return FeatureRecord(
            name="_".join(part for part in parts if part),
            description=self._summary_words(who, summary),
            kind=KIND_MOTIF,
            origin=ORIGIN_SEQMAESTRO,
            params={**self._params(summary), "collection": self.collection.name,
                    "motif_id": motif.identifier, "motif_name": motif.name},
            provenance=PROVENANCE_SELECTED,
        )

    def describe(self) -> dict[str, Any]:
        return {"kind": MOTIF_KNOWN, "collection": self.collection.name,
                "collection_digest": self.collection.digest,
                "min_match_share": self.min_match_share, **self._described()}


@dataclass(frozen=True)
class DiscoveredMotifs:
    """What the source `motif_discovered` needs: how discovery is run, how its motifs' scans are
    summarised, the share of fitting rows a count must match in, the kind of target, what each
    direction is called in a description (`described`, by `UP` and `DOWN`), and the collection a
    found motif is compared with, or None."""

    discovery: DiscoverySettings
    settings: ScanSettings
    min_match_share: float
    target_kind: str
    described: dict[str, str]
    annotate_with: Collection | None


class DiscoveredMotifFeaturiser(_MotifFeaturiser):
    """Two columns per motif that discovery finds in the fitting rows, as `settings` specifies: the
    strength of its best match and its number of matches. The motifs are found at `fit`, from the
    rows and labels it is handed, and kept, so a fitted featuriser scans new rows in the Core
    environment alone, without STREME."""

    source = MOTIF_DISCOVERED

    def __init__(self, discovered: DiscoveredMotifs, store: ScanStore | None = None) -> None:
        super().__init__(discovered.settings, store, discovered.min_match_share)
        if set(discovered.described) != {UP, DOWN}:
            raise ValueError(f"`described` must say what {UP} and {DOWN} are, not "
                             f"{sorted(discovered.described)}.")
        self.discovered = discovered
        #: What discovery found, by the identifier of each motif.
        self.found: dict[str, Found] = {}
        self.tool_versions: dict[str, str] = {}
        #: The discovery's key, which names its folder under the run's `motifs/discoveries/`.
        self.discovery_key: str | None = None

    def _motifs(self, inputs: Inputs, y: np.ndarray | None) -> list[Motif]:
        if y is None:
            raise ValueError("Discovery needs the labels the cell fits against, and none were given.")
        root = self.store.root if self.store is not None and self.store.usable() else None
        asked = (list(inputs.sequences), np.asarray(y), self.discovered.target_kind,
                 self.discovered.discovery, self.discovered.annotate_with)
        found = discovery.discover(*asked, root)
        if not found:
            raise ValueError("Discovery found no motif in the fitting rows.")
        self.found = {entry.motif.identifier: entry for entry in found}
        self.tool_versions = discovery.versions()
        self.discovery_key = discovery.discovery_key(*asked)
        return [entry.motif for entry in found]

    def _record(self, motif: Motif, summary: str) -> FeatureRecord:
        entry = self.found[motif.identifier]
        tested = "" if entry.evalue is None else f", E = {entry.evalue:.2g}"
        who = (f"the motif STREME found enriched in {self.discovered.described[entry.direction]} "
               f"(consensus {entry.consensus}{tested})")
        parts = ["streme", entry.direction, str(entry.rank), name_part(entry.consensus),
                 SUFFIXES[summary]]
        return FeatureRecord(
            name="_".join(parts),
            description=self._summary_words(who, summary),
            kind=KIND_MOTIF,
            origin=ORIGIN_SEQMAESTRO,
            params={**self._params(summary), "tool": "streme", "direction": entry.direction,
                    "rank": entry.rank, "consensus": entry.consensus,
                    "motif_digest": motif.digest()[:16]},
            provenance=PROVENANCE_SELECTED,
            extra={"frequencies": motif.frequencies.tolist(), "evalue": entry.evalue,
                   "sites": entry.sites,
                   **({"closest_known": entry.closest} if entry.closest else {})},
        )

    def describe(self) -> dict[str, Any]:
        counts = {direction: sum(entry.direction == direction for entry in self.found.values())
                  for direction in (UP, DOWN)}
        return {"kind": MOTIF_DISCOVERED, "tool": "streme", "tool_versions": self.tool_versions,
                "discovery": self.discovered.discovery.as_dict(), "discovery_key": self.discovery_key,
                "found": counts,
                "annotate_with": (None if self.discovered.annotate_with is None
                                  else self.discovered.annotate_with.name),
                "min_match_share": self.min_match_share, **self._described()}
