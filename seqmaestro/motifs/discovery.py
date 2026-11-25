"""Motif discovery: STREME, run in the MEME Suite's environment, finds the motifs that separate a
cell's fitting rows by their target, and Tomtom names the known motif closest to each.

**What is compared.** For a binary target, STREME looks for motifs enriched in the rows of class 1
against those of class 0 (`up`) and, with `directions: both`, in class 0 against class 1 (`down`),
since a motif enriched in either class can predict it. For a continuous target, the rows in the top
`continuous_fraction` of the target take the place of class 1 and those in the bottom
`continuous_fraction` take the place of class 0, and the rows between take no part. STREME takes no
weights and no scores, so this is how a continuous target reaches it. The rows are those the cell
fits on, with the labels it fits against. A control cell fits against shuffled labels, so its motifs
are also found on shuffled labels.

**STREME** runs with `--dna` where both strands are scanned and with `--rna` where only the given
one is, on the sequences in upper case, with T written as U for RNA. Its motif widths are the
source's settings, it trims no control sequence (`--notrim`), and its seed is fixed at 0, since a
run's seeds already differ in the rows they fit on. STREME reports motifs until three in a row fail
its test on held-out sequences, and it reports those three too. A discovery therefore always holds a
few motifs, and a control cell on shuffled labels gets motifs just as a cell on the real labels
does. Up to `max_motifs` are kept per direction, in the order STREME reports them, each with its
E-value. STREME tests a motif on a tenth of each set, which it holds out from its search. Where that
tenth would be fewer than five sequences, as for a class of fewer than 50 fitting rows, it holds
none out, and its motifs then have no E-value. In its place STREME reports a score on the training
rows, which does not measure significance and is not kept.

**Tomtom** compares each motif found with the collection the run names for it (`annotate_with`),
and the match with the lowest q-value is kept beside the motif. The match only informs a reader of
the report: it does not appear in any feature's name or description.

**A discovery is made once.** It depends on the rows, their labels, the settings, the collection it
is compared with and the versions of the tools, which together make its key. Within a run, each
discovery is made under a lock and kept in `<run>/motifs/discoveries/<key>/` with STREME's and
Tomtom's output, so two processes that ask for one discovery make it once and get the same motifs.
A discovery is always read back from what was written, so a fresh one and a stored one are
identical.

**The tools run in `SeqMaestro_Method_meme`.** They are called by their paths beside that
environment's interpreter, which `seqmaestro.provenance.environment_python` finds. This module does
not import MEME Suite code.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seqmaestro import files
from seqmaestro.motifs.collection import Collection, Motif, read_meme, write_meme
from seqmaestro.motifs.scan import STRANDS
from seqmaestro.provenance import environment_python, missing_python
from seqmaestro.target import BINARY, CONTINUOUS

#: The environment the MEME Suite's tools are installed in (ENVIRONMENTS.md).
MEME_ENVIRONMENT = "SeqMaestro_Method_meme"
#: The two directions: what is enriched in class 1, or at the top of a continuous target, and what
#: is enriched in class 0, or at its bottom.
UP, DOWN = "up", "down"
#: The directions each value of the `directions` setting searches: both, or `up` alone.
DIRECTION_SETTINGS = {"both": (UP, DOWN), "up": (UP,)}
#: The widest motif STREME finds.
STREME_MAX_WIDTH = 30
#: Where a run keeps its discoveries, under its motifs directory.
DISCOVERIES_DIR = "discoveries"
FOUND_FILE = "found.json"
#: The version of how a discovery is made and kept. It is part of every key.
DISCOVERY_VERSION = 2


@dataclass(frozen=True)
class DiscoverySettings:
    """How STREME is run: the directions searched, the share of a continuous target's rows at each
    end, the widths of a motif, how many motifs are kept per direction, and the strands, which are
    the scan's. The settings have no defaults, since each changes the motifs found."""

    directions: str
    continuous_fraction: float
    min_width: int
    max_width: int
    max_motifs: int
    strands: str

    def __post_init__(self) -> None:
        if self.directions not in DIRECTION_SETTINGS:
            raise ValueError(f"directions must be one of {', '.join(DIRECTION_SETTINGS)}, not "
                             f"{self.directions!r}.")
        if isinstance(self.continuous_fraction, bool) or not 0 < self.continuous_fraction <= 0.5:
            raise ValueError(f"continuous_fraction must be above 0 and at most 0.5, not "
                             f"{self.continuous_fraction!r}.")
        widths = (self.min_width, self.max_width)
        if (any(isinstance(width, bool) or not isinstance(width, int) for width in widths)
                or not 3 <= self.min_width <= self.max_width <= STREME_MAX_WIDTH):
            raise ValueError(f"The widths must be whole numbers with 3 <= min_width <= max_width <= "
                             f"{STREME_MAX_WIDTH}, not {widths}.")
        if isinstance(self.max_motifs, bool) or not isinstance(self.max_motifs, int) or self.max_motifs < 1:
            raise ValueError(f"max_motifs must be a whole number of at least 1, not {self.max_motifs!r}.")
        if self.strands not in STRANDS:
            raise ValueError(f"strands must be one of {', '.join(STRANDS)}, not {self.strands!r}.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class Found:
    """One motif a discovery found, with the direction it was found in and its rank there, STREME's
    consensus, E-value (None where STREME held no sequences out to test it) and number of sites, and
    the closest known motif, where one was looked for. The motif's identifier says where it came
    from (`up_1_GATTACA`)."""

    motif: Motif
    direction: str
    rank: int
    consensus: str
    evalue: float | None
    sites: float
    closest: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {"identifier": self.motif.identifier, "direction": self.direction, "rank": self.rank,
                "consensus": self.consensus, "evalue": self.evalue, "sites": self.sites,
                "frequencies": self.motif.frequencies.tolist(), "closest": self.closest}

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> Found:
        motif = Motif(entry["identifier"], None, np.array(entry["frequencies"]), sites=entry["sites"])
        return cls(motif=motif, direction=entry["direction"], rank=entry["rank"],
                   consensus=entry["consensus"], evalue=entry["evalue"], sites=entry["sites"],
                   closest=entry["closest"])


def tool(name: str) -> Path:
    """The path of a MEME Suite program, beside the interpreter of its environment."""
    return environment_python(MEME_ENVIRONMENT).parent / name


_VERSIONS: dict[str, str] = {}


def versions() -> dict[str, str]:
    """The versions of STREME and Tomtom, asked once per process. Where the environment is
    missing, raises an error that says how to build it."""
    problem = missing_python(MEME_ENVIRONMENT)
    if problem:
        raise RuntimeError(f"Motif discovery runs the MEME Suite, and {problem} "
                           f"`./install.sh --with-motifs` builds it.")
    for name in ("streme", "tomtom"):
        if name not in _VERSIONS:
            _VERSIONS[name] = _run([str(tool(name)), "--version"]).stdout.strip()
    return dict(_VERSIONS)


def compared(y: np.ndarray, target_kind: str, settings: DiscoverySettings) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each direction searched, the rows STREME looks in and the rows it compares them with."""
    y = np.asarray(y)
    if target_kind == BINARY:
        ones, zeros = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    elif target_kind == CONTINUOUS:
        order = np.argsort(y, kind="stable")
        size = int(settings.continuous_fraction * len(y))
        zeros, ones = np.sort(order[:size]), np.sort(order[len(y) - size:])
    else:
        raise ValueError(f"No discovery for a target of kind {target_kind!r}.")
    if min(len(ones), len(zeros)) < 2:
        raise ValueError(f"Discovery needs at least two rows at each end of the target, and has "
                         f"{len(ones)} and {len(zeros)}.")
    pairs = {UP: (ones, zeros), DOWN: (zeros, ones)}
    return {direction: pairs[direction] for direction in DIRECTION_SETTINGS[settings.directions]}


def discover(sequences: Sequence[str], y: np.ndarray, target_kind: str, settings: DiscoverySettings,
             annotate_with: Collection | None, root: Path | None) -> list[Found]:
    """The motifs that separate `sequences` by their target `y`, made once and kept under `root`
    (a run's motifs directory) where it is given and exists, or made in a temporary directory
    otherwise."""
    key = discovery_key(sequences, y, target_kind, settings, annotate_with)
    if root is None or not Path(root).is_dir():
        with tempfile.TemporaryDirectory(prefix="seqmaestro-streme-") as scratch:
            folder = Path(scratch) / key
            _make(folder, sequences, y, target_kind, settings, annotate_with)
            return _read(folder)
    folder = Path(root) / DISCOVERIES_DIR / key
    with _held(folder):
        if not (folder / FOUND_FILE).exists():
            _make(folder, sequences, y, target_kind, settings, annotate_with)
    return _read(folder)


def discovery_key(sequences: Sequence[str], y: np.ndarray, target_kind: str,
                  settings: DiscoverySettings, annotate_with: Collection | None) -> str:
    """A digest of everything a discovery depends on: the rows in order, their labels and the kind
    of target, the settings, the collection it is compared with, the versions of the tools, and
    `DISCOVERY_VERSION`."""
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.encode())
        digest.update(b"\n")
    digest.update(np.asarray(y, dtype=np.float64).tobytes())
    digest.update(json.dumps({"target_kind": target_kind, "settings": settings.as_dict(),
                              "annotate_with": None if annotate_with is None else annotate_with.digest,
                              "versions": versions(), "version": DISCOVERY_VERSION},
                             sort_keys=True).encode())
    return digest.hexdigest()[:32]


# ---------------------------------------------------------------------------------- internals

def _make(folder: Path, sequences: Sequence[str], y: np.ndarray, target_kind: str,
          settings: DiscoverySettings, annotate_with: Collection | None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rna = settings.strands == "given"
    found: list[Found] = []
    for direction, (looked_in, against) in compared(y, target_kind, settings).items():
        reported = _streme([sequences[row] for row in looked_in], [sequences[row] for row in against],
                           folder / f"streme_{direction}", settings, rna)
        closest = (_tomtom([motif for motif, _ in reported], annotate_with, folder / f"tomtom_{direction}", rna)
                   if annotate_with is not None and reported else {})
        for rank, (motif, evalue) in enumerate(reported, start=1):
            consensus = motif.identifier.partition("-")[2] or motif.consensus
            found.append(Found(
                motif=Motif(f"{direction}_{rank}_{consensus}", None, motif.frequencies, motif.sites),
                direction=direction, rank=rank, consensus=consensus, evalue=evalue,
                sites=motif.sites, closest=closest.get(motif.identifier)))
    files.write_json(folder / FOUND_FILE, {"found": [entry.as_dict() for entry in found],
                                           "settings": settings.as_dict(), "versions": versions(),
                                           "rows": len(sequences)})


def _read(folder: Path) -> list[Found]:
    return [Found.from_dict(entry) for entry in json.loads((folder / FOUND_FILE).read_text())["found"]]


def _fasta(sequences: Sequence[str], path: Path, rna: bool) -> None:
    letters = [sequence.upper().replace("T", "U") if rna else sequence.upper() for sequence in sequences]
    path.write_text("".join(f">s{index}\n{sequence}\n" for index, sequence in enumerate(letters)))


def _streme(looked_in: Sequence[str], against: Sequence[str], out: Path, settings: DiscoverySettings,
            rna: bool) -> list[tuple[Motif, float | None]]:
    """The motifs STREME reports for one direction, at most `max_motifs`, each with its E-value,
    or None where it has none."""
    out.mkdir(parents=True, exist_ok=True)
    _fasta(looked_in, out / "primary.fa", rna)
    _fasta(against, out / "control.fa", rna)
    _run([str(tool("streme")), "--p", str(out / "primary.fa"), "--n", str(out / "control.fa"),
          "--rna" if rna else "--dna", "--minw", str(settings.min_width),
          "--maxw", str(settings.max_width), "--notrim", "--seed", "0", "--verbosity", "1",
          "--oc", str(out / "streme")])
    text = (out / "streme" / "streme.txt").read_text()
    motifs = read_meme(text, source=str(out / "streme" / "streme.txt"))
    evalues = _evalues(text)
    return [(motif, evalues[motif.identifier]) for motif in motifs[: settings.max_motifs]]


def _evalues(text: str) -> dict[str, float | None]:
    """Each motif's E-value, from the `E=` field of its letter-probability line, or None where
    STREME held no sequences out and wrote its score on the training rows (`S=`) instead."""
    found, current = {}, None
    for line in text.splitlines():
        if line.startswith("MOTIF"):
            current = line.split()[1]
        elif line.startswith("letter-probability") and current is not None:
            fields = line.split()
            found[current] = float(fields[fields.index("E=") + 1]) if "E=" in fields else None
    return found


def _tomtom(motifs: list[Motif], collection: Collection, out: Path, rna: bool) -> dict[str, dict]:
    """For each motif, the motif of `collection` that Tomtom finds closest, by its q-value."""
    out.mkdir(parents=True, exist_ok=True)
    alphabet = "ACGU" if rna else "ACGT"
    (out / "query.meme").write_text(write_meme(motifs, alphabet=alphabet))
    (out / "target.meme").write_text(write_meme(collection.motifs, alphabet=alphabet))
    _run([str(tool("tomtom")), "-no-ssc", "-oc", str(out / "tomtom"), "-verbosity", "1",
          "-min-overlap", "5", "-dist", "pearson", "-evalue", "-thresh", "10",
          *(["-norc"] if rna else []), str(out / "query.meme"), str(out / "target.meme")])
    return _closest((out / "tomtom" / "tomtom.tsv").read_text(), collection)


def _closest(text: str, collection: Collection) -> dict[str, dict]:
    """From Tomtom's table, the match of each query with the lowest q-value, with ties broken by
    the p-value. The table has a header row and ends with comment lines, which are skipped."""
    names = {motif.identifier: motif.name for motif in collection.motifs}
    closest: dict[str, dict] = {}
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    header = lines[0].split("\t") if lines else []
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t"), strict=False))
        match = {"collection": collection.name, "motif_id": row["Target_ID"],
                 "motif_name": names.get(row["Target_ID"]), "q_value": float(row["q-value"]),
                 "p_value": float(row["p-value"])}
        held = closest.get(row["Query_ID"])
        if held is None or (match["q_value"], match["p_value"]) < (held["q_value"], held["p_value"]):
            closest[row["Query_ID"]] = match
    return closest


def _run(command: list[str]) -> subprocess.CompletedProcess:
    """Run a MEME Suite program on one thread, with its environment's programs first on the path.
    Where the program fails, the error raised holds the end of its error output."""
    environment = {**os.environ, "PATH": f"{Path(command[0]).parent}:{os.environ.get('PATH', '')}",
                   "OMP_NUM_THREADS": "1"}
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{Path(command[0]).name} exited {completed.returncode}:\n"
                           f"{completed.stderr[-2000:]}")
    return completed


@contextmanager
def _held(folder: Path):
    """Hold the lock of one discovery against the other processes of the run."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    with open(folder.parent / f"{folder.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
