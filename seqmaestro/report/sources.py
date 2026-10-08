"""Read a revealed run back, in the shape a report is assembled from.

**What a run holds is decided by its config, not by what happens to be on disk.** The caller,
`seqmaestro.pipeline.report`, reads the config and knows the run's layout, so it lists every cell
and every stage's directory in an `Inventory`, and this module reads them. A listed file that is
missing raises, so a report cannot be assembled around a silent gap. The Stage 2 analyses are the
exception, because a run may not produce one (recurrence and the judge need a language model). A
missing analysis is returned as None, and the report says "not measured" in its place. The judge is
listed only where the run's config runs it.

**Test numbers are read only from a revealed run**, and only through `seqmaestro.harness.heldout`.
The reveal must be complete and must have sealed every listed cell, so every test number in a
report is one the reveal scored.

**Nothing here interprets.** A loader returns what a file says, keyed on the arm
(`model/pool`). Deciding what a number means is the report's job, and choosing which rows
to show is `facts.py`'s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from seqmaestro.dataset import Dataset
from seqmaestro.harness import cell as cells
from seqmaestro.harness.features import load_features
from seqmaestro.harness.heldout import TEST_SPLIT, reveal_record, revealed_metrics
from seqmaestro.reduction import outputs as reduction_outputs
from seqmaestro.reliability import tables as reliability_tables
from seqmaestro.reliability.files import (
    CHECKS_FILE,
    PREPARE_FILE,
    REAL,
    TIER3_FILE,
    tier1_name,
    tier2_name,
)
from seqmaestro.reliability.panel import read_record as read_panel_record


class ReportError(RuntimeError):
    """A run a report cannot be assembled from as it stands. The message says what is missing."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class Inventory:
    """Everything a run's config says it holds, as paths relative to the run directory.

    `suite` maps arm -> labels -> seed -> cell. `analyses` maps each Stage 2 analysis output a
    report reads to its file.
    """

    suite: dict[str, dict[str, dict[int, Path]]]
    baseline: dict[int, Path]
    reduced: dict[str, Path]
    profiled: dict[str, Path]
    not_profiled: dict[str, str]
    analyses: dict[str, Path]

    def cells(self, reduced_roles: tuple[str, ...]) -> list[Path]:
        """Every cell listed, which the reveal must have sealed."""
        found = [cell for labels in self.suite.values() for seeds in labels.values()
                 for cell in seeds.values()]
        found += list(self.baseline.values())
        found += [directory / role for directory in self.reduced.values() for role in reduced_roles]
        return found


@dataclass
class Cell:
    """One scored cell: what it recorded about itself, and its validation and test blocks."""

    path: Path
    run: dict[str, Any]
    val: dict[str, Any]
    test: dict[str, Any]


@dataclass
class ReducedArm:
    """One arm's Stage 3A: its record, its three revealed models, and its source cell's records."""

    directory: Path
    record: dict[str, Any]
    roles: dict[str, Cell]
    source_records: dict[str, Any]


@dataclass
class ProfiledArm:
    """One arm's Stage 3B: its record, its tiers' summaries and its assembled tables."""

    directory: Path
    record: dict[str, Any]
    prepare: dict[str, Any]
    tier1: dict[str, Any]
    tier2: dict[str, Any] | None
    tier3: dict[str, Any]
    checks: dict[str, Any]
    features: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class Run:
    """Everything one run's report is written from."""

    directory: Path
    dataset: Dataset
    reveal: dict[str, Any]
    suite: dict[str, dict[str, dict[int, Cell]]]
    baseline: dict[int, Cell]
    reduced: dict[str, ReducedArm]
    profiled: dict[str, ProfiledArm]
    not_profiled: dict[str, str]
    #: Whether the run withheld a fact from the language models, and the fact, from its config's
    #: `dataset.blinded` and its task description's `judge_only`: `{"blinded": bool, "withheld":
    #: str | None}`.
    blinding: dict[str, Any]
    analyses: dict[str, Any] = field(default_factory=dict)


def read_cell(run_dir: Path, cell: Path) -> Cell:
    """A revealed cell. Refused when its files are missing or the reveal did not score it."""
    path = run_dir / cell
    if not cells.is_finished(path):
        raise ReportError(f"{path} has no {cells.METRICS_FILE}: the cell the config lists was "
                          f"never written.")
    return Cell(path=cell, run=cells.read_run(path),
                val=cells.read_metrics(path)[cells.VALIDATION],
                test=revealed_metrics(path, run_dir))


def _analyses(run_dir: Path, files: dict[str, Path]) -> dict[str, Any]:
    """The Stage 2 analyses a report reads, each as its writer left it, or None where absent."""
    found: dict[str, Any] = {}
    for name, relative in files.items():
        path = run_dir / relative
        if not path.exists():
            found[name] = None
        elif path.suffix == ".json":
            found[name] = read_json(path)
        else:
            found[name] = pd.read_csv(path)
    return found


def _reduced(run_dir: Path, directory: Path, roles: tuple[str, ...]) -> ReducedArm:
    record = reduction_outputs.read(run_dir / directory)
    records, _ = load_features(run_dir / record["source_cell"] / cells.FEATURES_FILE)
    return ReducedArm(directory=directory, record=record,
                      roles={role: read_cell(run_dir, directory / role) for role in roles},
                      source_records={r.name: r for r in records})


def _profiled(run_dir: Path, directory: Path) -> ProfiledArm:
    arm_dir = run_dir / directory
    features_file = arm_dir / reliability_tables.FEATURES_FILE
    if not features_file.exists():
        raise ReportError(f"{arm_dir} has no {reliability_tables.FEATURES_FILE}: run "
                          f"run_reliability.py --after-reveal first.")
    record = read_panel_record(arm_dir)
    features = pd.read_csv(features_file, keep_default_na=True)
    tier2 = None if record["tier2"]["refused"] else read_json(
        arm_dir / f"{tier2_name(TEST_SPLIT, REAL)}.json")
    return ProfiledArm(
        directory=directory, record=record,
        prepare=read_json(arm_dir / PREPARE_FILE),
        tier1=read_json(arm_dir / f"{tier1_name(REAL)}.json"), tier2=tier2,
        tier3=read_json(arm_dir / TIER3_FILE), checks=read_json(arm_dir / CHECKS_FILE),
        features=features, summary=reliability_tables.summary(arm_dir, record, features))


def load(run_dir: str | Path, inventory: Inventory, dataset: Dataset,
         reduced_roles: tuple[str, ...], blinding: dict[str, Any]) -> Run:
    """Read everything `inventory` lists from a revealed run. `blinding` is `Run.blinding`, which the
    caller reads from the run's config and the task description, neither of which is read here."""
    run_dir = Path(run_dir)
    reveal = reveal_record(run_dir)
    if reveal is None:
        raise ReportError(f"{run_dir} has not been revealed; a report quotes test numbers, which exist "
                          f"only after run_reveal.py.")
    if reveal.get("completed_utc") is None:
        raise ReportError(f"{run_dir}: the reveal started at {reveal['revealed_utc']} and did not "
                          f"finish. Resume run_reveal.py first.")
    sealed = set(reveal["sealed"])
    unsealed = [str(cell) for cell in inventory.cells(reduced_roles) if str(cell) not in sealed]
    if unsealed:
        raise ReportError(f"{len(unsealed)} cell(s) the config lists were not sealed by the reveal, "
                          f"first {unsealed[:3]}; they have no test numbers to report.")

    suite = {arm: {labels: {seed: read_cell(run_dir, cell) for seed, cell in seeds.items()}
                     for labels, seeds in by_labels.items()}
               for arm, by_labels in inventory.suite.items()}
    return Run(
        directory=run_dir, dataset=dataset, reveal=reveal, suite=suite,
        baseline={seed: read_cell(run_dir, cell) for seed, cell in inventory.baseline.items()},
        reduced={arm: _reduced(run_dir, directory, reduced_roles)
                 for arm, directory in inventory.reduced.items()},
        profiled={arm: _profiled(run_dir, directory) for arm, directory in inventory.profiled.items()},
        not_profiled=dict(inventory.not_profiled),
        blinding=dict(blinding),
        analyses=_analyses(run_dir, inventory.analyses),
    )
