"""A run's motifs: the collections it scans with, the store of its scans and discoveries, and the
step before the arms that scans every training and validation sequence and makes the discoveries.

    python experiments/runners/run_motifs.py --run-dir experiments/runs/<date>_<name>
    python experiments/runners/run_motifs.py --run-dir ... --discover --labels real --seed 42

**A run scans with a copy of the collection kept in the run.** A config names a collection
(`sources.motif_known.collection`), which Stage 1 fetches into `data/motif_collections/<name>/`
with `experiments/prepare/fetch_motif_collection.py`. The first process of a run that needs the
collection copies its motif file into `<run>/motifs/collections/`, and every process of the run
reads that copy. Every cell therefore scans with the same file, and fetching the collection again
during a run does not affect it. A finished run also keeps the motifs it scanned with beside its
results.

**The store** (`seqmaestro.motifs.store`) is `<run>/motifs/`. A featuriser of a motif source is
built with the store, takes from it the scan of every sequence already scanned, and adds the scans
it makes.

**The step before the arms** scans every training and validation sequence once with the scanner
the known-motif source builds, and keeps the scans in the store. For the discovered motifs, it
makes one discovery per label condition and seed, on the rows and labels a cell fits on at that
seed, and scans the training and validation rows with the motifs found, as the featuriser of every
cell at that label condition and seed would. The arms whose pool holds a motif source start once
their part of the step has finished (`seqmaestro/pipeline/driver.py`), so that two of them never
make the same scan or discovery at once. Under `permutation_scope: cell`, a control's labels depend
on its arm, so the discoveries for the shuffled labels are not made in advance, and each control
cell makes its own. A featuriser makes whatever the store lacks, so the step only saves work and
does not decide any value: an arm started without it gives the same features. The held-out split is
scanned at the reveal by the featuriser each cell saved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from seqmaestro import files
from seqmaestro.dataset import Dataset
from seqmaestro.features import MOTIF_DISCOVERED, MOTIF_KNOWN, DiscoveredMotifs, KnownMotifs, parse_pool
from seqmaestro.harness.cell import LABELS, PERMUTED
from seqmaestro.harness.heldout import refuse_if_revealed
from seqmaestro.harness.runner import permutation_key, subsample_training_rows
from seqmaestro.motifs.collection import Collection, read_collection
from seqmaestro.motifs.discovery import DOWN, MEME_ENVIRONMENT, UP
from seqmaestro.motifs.scan import Scanner
from seqmaestro.motifs.store import ScanStore, scan_with_store
from seqmaestro.pipeline.config import ConfigError, load_recorded
from seqmaestro.pipeline.layout import MOTIFS_DIR, call_name
from seqmaestro.pipeline.records import snapshot_environments
from seqmaestro.pipeline.repository import MOTIF_COLLECTIONS
from seqmaestro.pipeline.rows import Rows
from seqmaestro.provenance import missing_python, record_call
from seqmaestro.target import BINARY

if TYPE_CHECKING:
    from seqmaestro.pipeline.config import ExperimentConfig

#: The stage of the driver that scans before the arms, and the name of its call for known motifs.
STAGE = "motifs"
KNOWN_CALL = call_name(STAGE, (), MOTIF_KNOWN)
#: Where a discovery made by the step is recorded, named by its label condition and seed.
DISCOVERED_RECORD = "motif_discovered__{labels}__seed_{seed}.json"
#: A fetched collection's motif file, in its folder under `data/motif_collections/`.
COLLECTION_FILE = "motifs.meme"
#: Where a run keeps its copies of the collections it scans with, under its motifs directory.
COLLECTIONS_DIR = "collections"
#: The record the step before the arms writes beside the store: what it scanned and how long it
#: took.
KNOWN_RECORD = "motif_known.json"


def motifs_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / MOTIFS_DIR


def fetched(name: str) -> Path:
    """Where a collection fetched at Stage 1 keeps its motif file."""
    return MOTIF_COLLECTIONS / name / COLLECTION_FILE


def run_copy(run_dir: str | Path, name: str) -> Path:
    return motifs_dir(run_dir) / COLLECTIONS_DIR / f"{name}.meme"


def run_collection(run_dir: str | Path, name: str) -> Collection:
    """The collection a run scans with: its copy, made from the fetched file on first use."""
    copy = run_copy(run_dir, name)
    if not copy.exists():
        source = fetched(name)
        if not source.exists():
            raise FileNotFoundError(_missing(name))
        copy.parent.mkdir(parents=True, exist_ok=True)
        files.write_bytes(copy, source.read_bytes())
    return read_collection(copy, name)


def store_for(run_dir: str | Path) -> ScanStore:
    """The run's store of scans, whose directory is created if it is missing. Its path is
    absolute, so a featuriser saved with it finds the same directory (or none) wherever it is
    loaded from."""
    directory = motifs_dir(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return ScanStore(directory)


def collection(name: str, run_dir: str | Path | None) -> Collection:
    """A collection the run scans or compares with: the run's copy, or the fetched file where there
    is no run (as for an estimate)."""
    return (run_collection(run_dir, name) if run_dir is not None
            else read_collection(fetched(name), name))


def known_motifs(config: ExperimentConfig, run_dir: str | Path | None) -> KnownMotifs:
    """The collection and the scan settings of the run's `motif_known`."""
    settings = config.sources.motif_known
    return KnownMotifs(collection(settings.collection, run_dir), settings.scan(),
                       settings.min_match_share)


def discovered_motifs(config: ExperimentConfig, dataset: Dataset,
                      run_dir: str | Path | None) -> DiscoveredMotifs:
    """How the run's `motif_discovered` discovers and scans, with what each direction is called in
    a feature's description: the classes a binary target names, or the ends of a continuous one."""
    settings = config.sources.motif_discovered
    if dataset.target_kind == BINARY:
        one, zero = dataset.classes[1].name, dataset.classes[0].name
        described = {UP: f"{one} sequences against {zero} ones",
                     DOWN: f"{zero} sequences against {one} ones"}
    else:
        name, share = dataset.record["target"]["name"], f"{settings.continuous_fraction:.0%}"
        described = {UP: f"the sequences in the highest {share} of {name} against the lowest",
                     DOWN: f"the sequences in the lowest {share} of {name} against the highest"}
    annotate = (None if settings.annotate_with is None
                else collection(settings.annotate_with, run_dir))
    return DiscoveredMotifs(settings.discovery(), settings.scan(), settings.min_match_share,
                            dataset.target_kind, described, annotate)


def collection_problems(config: ExperimentConfig, run_dir: str | Path) -> list[str]:
    """Why the run could not find a collection it names, which happens when neither its copy nor
    the fetched file exists."""
    names = [settings.collection for settings in (config.sources.motif_known,) if settings]
    names += [settings.annotate_with for settings in (config.sources.motif_discovered,)
              if settings and settings.annotate_with]
    return [_missing(name) for name in dict.fromkeys(names)
            if not run_copy(run_dir, name).exists() and not fetched(name).exists()]


def holds(config: ExperimentConfig, member: str) -> bool:
    """Whether some pool of the run holds `member`."""
    return any(member in parse_pool(pool) for pool in config.arms)


def holds_known(config: ExperimentConfig) -> bool:
    return holds(config, MOTIF_KNOWN)


def discovered_call(labels: str, seed: int) -> str:
    """The call of the step that makes the discovery of one label condition and seed."""
    return call_name(STAGE, (), MOTIF_DISCOVERED, labels, f"seed_{seed}")


def made_in_advance(labels: str, scope: str) -> bool:
    """Whether the step makes the discovery of `labels` before the arms. A control's labels under
    `permutation_scope: cell` depend on its arm, so its discovery is made by its cell."""
    return not (labels == PERMUTED and scope == "cell")


def waits_for(pool: str, labels: str, seed: int, scope: str) -> tuple[str, ...]:
    """The calls of the step before the arms that the process of an arm on `pool` at `labels` and
    `seed` waits for."""
    members = parse_pool(pool)
    calls = [KNOWN_CALL] if MOTIF_KNOWN in members else []
    if MOTIF_DISCOVERED in members and made_in_advance(labels, scope):
        calls.append(discovered_call(labels, seed))
    return tuple(calls)


def environment_problems(config: ExperimentConfig) -> list[str]:
    """Why the run could not discover motifs: the MEME Suite's environment is missing."""
    problem = missing_python(MEME_ENVIRONMENT) if holds(config, MOTIF_DISCOVERED) else None
    return [f"{problem} `./install.sh --with-motifs` builds it."] if problem else []


def scan_known(config: ExperimentConfig, rows: Rows, run_dir: str | Path) -> dict:
    """Scan every distinct training and validation sequence with the run's collection into the
    store, and return a record of what was scanned. The scanner is the one the known-motif
    featuriser builds (every motif of the collection under the run's settings), so its scans are
    the ones the featurisers look for."""
    known = known_motifs(config, run_dir)
    scanner = Scanner(known.collection.motifs, known.settings)
    column = rows.dataset.sequence_column
    sequences = pd.concat([rows.train[column], rows.val[column]]).drop_duplicates().tolist()
    started = time.perf_counter()
    scan_with_store(scanner, sequences, store_for(run_dir))
    return {"collection": known.collection.name, "collection_digest": known.collection.digest,
            "motifs": len(known.collection.motifs), "scanner": scanner.digest(),
            "sequences": len(sequences), "seconds": round(time.perf_counter() - started, 1)}


def discover_for(config: ExperimentConfig, rows: Rows, run_dir: str | Path, labels: str,
                 seed: int) -> dict:
    """Make the discovery of one label condition and seed into the store, as a cell's featuriser
    would make it, and scan the training and validation rows with its motifs. The rows and labels
    are the ones every cell at that label condition and seed fits on (`subsample_training_rows`)."""
    from seqmaestro.pipeline.sources import pool_featuriser  # deferred: `sources` imports this module

    if not made_in_advance(labels, config.grid.permutation_scope):
        raise ValueError(f"Under permutation_scope {config.grid.permutation_scope!r} the {labels} "
                         f"labels depend on the arm, so each cell makes the discovery of its labels.")
    dataset = rows.dataset
    # Under `permutation_scope: seed`, every arm at a seed shares one draw of the shuffled labels,
    # whose key `permutation_key` gives as None when no arm is passed.
    fit_rows, y_fit = subsample_training_rows(
        rows.train, seed=seed, columns=dataset.columns, train_fraction=config.grid.train_fraction,
        permute_labels=labels == PERMUTED,
        permutation_key=permutation_key(config.grid.permutation_scope, None)
        if labels == PERMUTED else None)
    featuriser = pool_featuriser(config, dataset, MOTIF_DISCOVERED, run_dir=run_dir)
    columns = dataset.columns_for(MOTIF_DISCOVERED)
    started = time.perf_counter()
    featuriser.fit(columns.inputs(fit_rows), y_fit)
    for frame in (rows.val, rows.train):
        featuriser.transform(columns.inputs(frame))
    described = featuriser.describe()
    return {"labels": labels, "seed": seed, "discovery_key": described["discovery_key"],
            "found": described["found"], "features": described["n_features"],
            "tool_versions": described["tool_versions"],
            "seconds": round(time.perf_counter() - started, 1)}


def _missing(name: str) -> str:
    return (f"The motif collection {name!r} has not been fetched: there is no {fetched(name)}. "
            f"Fetch it with experiments/prepare/fetch_motif_collection.py "
            f"(experiments/prepare/README.md).")


# ------------------------------------------------------------------------------ the command

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run, whose config.yaml names the collection and the settings")
    parser.add_argument("--discover", action="store_true",
                        help="make the discovery of one label condition and seed, in place of the "
                             "scan with the known motifs")
    parser.add_argument("--labels", choices=list(LABELS), help="with --discover: the labels")
    parser.add_argument("--seed", type=int, help="with --discover: the seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    try:
        config = load_recorded(run_dir)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    if args.discover:
        return _discover(args, config, run_dir)
    if not holds_known(config):
        print(f"No pool of the run holds {MOTIF_KNOWN}, so there is nothing to scan.", file=sys.stderr)
        return 2
    refuse_if_revealed(run_dir, "run_motifs")
    record_call(run_dir, KNOWN_CALL, vars(args))
    record = scan_known(config, Rows.of(config), run_dir)
    files.write_json(motifs_dir(run_dir) / KNOWN_RECORD, record)
    print(f"{record['sequences']} sequences scanned with the {record['motifs']} motifs of "
          f"{record['collection']} in {record['seconds']} s", flush=True)
    return 0


def _discover(args: argparse.Namespace, config: ExperimentConfig, run_dir: Path) -> int:
    if not holds(config, MOTIF_DISCOVERED):
        print(f"No pool of the run holds {MOTIF_DISCOVERED}, so there is nothing to discover.",
              file=sys.stderr)
        return 2
    if args.labels is None or args.seed is None or args.seed not in config.grid.seeds \
            or args.labels not in config.grid.labels:
        print(f"--discover needs --labels and --seed of the run's grid: {list(config.grid.labels)} "
              f"and {list(config.grid.seeds)}.", file=sys.stderr)
        return 2
    if not made_in_advance(args.labels, config.grid.permutation_scope):
        print(f"Under permutation_scope {config.grid.permutation_scope!r} each control cell makes its "
              f"own discovery, since its shuffled labels depend on its arm.", file=sys.stderr)
        return 2
    problems = environment_problems(config)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    refuse_if_revealed(run_dir, "run_motifs")
    record_call(run_dir, discovered_call(args.labels, args.seed), vars(args))
    record = discover_for(config, Rows.of(config), run_dir, args.labels, args.seed)
    files.write_json(motifs_dir(run_dir) / DISCOVERED_RECORD.format(labels=args.labels,
                                                                     seed=args.seed), record)
    snapshot_environments(run_dir, environments=(MEME_ENVIRONMENT,))
    print(f"{args.labels} labels, seed {args.seed}: {record['found']} motifs found, "
          f"{record['features']} features, in {record['seconds']} s", flush=True)
    return 0
