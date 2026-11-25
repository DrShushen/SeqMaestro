"""What a run's status page shows, gathered from the files the run writes.

Each part of the page has one function here, which reads that part's files through the modules that
name them and returns plain values for the template: `overall` for the band above the stages,
`stage1`, `stage2_`, `stage3a` and `stage3b` for the stages' columns, and `report_band` for the band
below. `gather` puts the parts together.

A run in progress lacks most files, so every function returns what exists and marks the rest as
waiting, and a file that is being written counts as absent until it can be read. What is running
comes from the scheduler's snapshot of its jobs (`seqmaestro.pipeline.schedule.snapshot`).

The page names arms, models and sources as the report does (`seqmaestro.report.naming`), so that the
page and the report use one vocabulary.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from seqmaestro.arms import arm_name, calls_language_model, written
from seqmaestro.dataset import DATASET_FILE
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import REVEAL_FILE, TEST_SPLIT
from seqmaestro.judge.files import SUMMARY as JUDGE_SUMMARY
from seqmaestro.pipeline import agents, baseline, motifs, reduction, reliability, report, reveal, stage2
from seqmaestro.pipeline.analyses import co_discovery, judge, recurrence, stability
from seqmaestro.pipeline.config import ExperimentConfig
from seqmaestro.pipeline.layout import (
    CONFIG_FILE,
    FAILURES_DIR,
    JUDGE_DIR,
    LOGS_DIR,
    REDUCED_MODEL,
    RESULTS_DIR,
    baseline_dir,
    call_name,
    cell_dir,
    reduction_dir,
    reliability_dir,
    report_dir,
)
from seqmaestro.pipeline.repository import DESCRIPTIONS
from seqmaestro.pipeline.schedule import JOBS_FILE
from seqmaestro.reduction.outputs import REDUCTION_FILE
from seqmaestro.reliability import files as panel_files
from seqmaestro.reliability.tables import FEATURES_FILE, loco_columns
from seqmaestro.report import files as report_files
from seqmaestro.report import naming
from seqmaestro.report.tables import WORDS
from seqmaestro.target import BINARY, target_for

#: The stages of the page, as `process/` and the figures number them, each with its name and what
#: it turns into what.
STAGES = {
    "1": ("Task construction", "Sequences → data and problem definition"),
    "2": ("Representation × model suite", "Gathering evidence on where the signal is"),  # noqa: RUF001
    "3A": ("Few-feature models", "Best small interpretable models"),
    "3B": ("Reliability panel", "How well is each feature supported?"),
}

#: The page's stage of each stage the driver runs, by the driver's name for it. Every other stage
#: (the baseline, the motifs, the arms and the analyses) belongs to Stage 2.
DRIVER_STAGES = {reduction.STAGE: "3A", reliability.STAGE: "3B", reveal.CALL: "3B",
                 reliability.STAGE_AFTER_REVEAL: "3B", report.CALL: "report",
                 report_files.FIGURES_CALL: "report"}

#: How many reduced arms the page names as the best.
BEST = 3


#: What Tier 3 shows for each kind of target, as the report's tables show it: the rise in log-loss
#: without the feature, or the fall in R², by its column in the panel's profile.
TIER3 = {kind: (loco_columns(kind)["mean_delta_log_loss" if kind == BINARY else "delta_r2"],
                WORDS[kind]["cost"][1]) for kind in WORDS}


# ------------------------------------------------------------------------------- reading files


def read_json(path: Path) -> Any:
    """A JSON file's content, or None while it is absent or still being written."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    """A CSV file's rows, or none while it is absent or still being written."""
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def number(value: Any) -> float | None:
    """A value as a number, or None where it is missing or not finite."""
    try:
        found = float(value)
    except (TypeError, ValueError):
        return None
    return found if np.isfinite(found) else None


def mean(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return float(np.mean(known)) if known else None


def relative(path: Path, run_dir: Path) -> str:
    """A link from the page, which sits at the run's root, to `path`."""
    return os.path.relpath(path, run_dir)


def elapsed(start: str | None, end: datetime | None = None) -> str | None:
    """The time between an ISO time and `end` (now by default), in hours and minutes."""
    if not start:
        return None
    seconds = ((end or datetime.now(UTC)) - datetime.fromisoformat(start)).total_seconds()
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


# ------------------------------------------------------------------------------- the jobs


def driver_alive(pid: int | None) -> bool:
    """Whether a process with this id is running on this machine."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def jobs_of(run_dir: Path) -> dict[str, Any]:
    """The scheduler's latest snapshot: whether the driver still runs, and each job by its call's
    name. A job the snapshot shows running under a driver that has stopped is shown as stopped."""
    snapshot = read_json(run_dir / LOGS_DIR / JOBS_FILE) or {}
    alive = bool(snapshot) and not snapshot.get("finished") and driver_alive(snapshot.get("driver_pid"))
    by_call = {}
    for job in snapshot.get("jobs", []):
        state = job["state"]
        if state == "running" and not alive:
            state = "stopped"
        by_call[job["log"]] = {**job, "state": state}
    return {"updated_utc": snapshot.get("updated_utc"), "alive": alive,
            "finished": bool(snapshot.get("finished")), "by_call": by_call}


def call_state(run_dir: Path, call: str, done: bool, jobs: dict[str, Any]) -> str:
    """One call's state: done once its files are there, otherwise as the snapshot shows it, failed
    where a failure is on file, and waiting otherwise."""
    if done:
        return "done"
    job = jobs["by_call"].get(call)
    if job is not None and job["state"] in ("running", "stopped", "failed", "held"):
        return job["state"]
    if (run_dir / FAILURES_DIR / f"{call}.json").exists():
        return "failed"
    return "waiting"


def stage_state(states: list[str]) -> str:
    """A stage's state from those of its calls."""
    if not states:
        return "waiting"
    for word in ("running", "failed", "stopped", "held"):
        if word in states:
            return word
    return "done" if all(state == "done" for state in states) else "waiting"


# ------------------------------------------------------------------------------- the overall band


def overall(run_dir: Path, config: ExperimentConfig, jobs: dict[str, Any],
            states: dict[str, str]) -> dict[str, Any]:
    """The band above the stages: what the run is, where it stands (its state, what is running and
    the failures on file), its times and estimate, the machine it runs on with its budget, the
    agents and the language model, and links to the run's records."""
    invocation = read_json(run_dir / "invocations" / "driver.json") or {}
    machines = _lines(run_dir / LOGS_DIR / "machine.jsonl")
    machine = machines[-1] if machines else None
    started = machines[0].get("started_utc") if machines else None
    revealed = (run_dir / REVEAL_FILE).exists()
    written = (report_dir(run_dir) / report_files.REPORT_FILE).exists()
    if jobs["alive"]:
        word, kind = "Running", "running"
    elif written:
        word, kind = "Finished, with its report", "done"
    elif any(state in ("failed", "stopped", "held") for state in states.values()):
        word, kind = "Stopped, with failures", "failed"
    elif revealed:
        word, kind = "Revealed; the report is to be written", "done"
    elif states.get("3B") == "done" or (states.get("2") == "done" and not config.reliability):
        word, kind = "Stopped before the reveal, for the checks", "waiting"
    elif jobs["updated_utc"]:
        word, kind = "Stopped", "waiting"
    else:
        word, kind = "Not started", "waiting"
    running = [{"label": job_label(job), "stage": DRIVER_STAGES.get(job["stage"], "2"),
                "started_utc": job["started_utc"], "for": elapsed(job["started_utc"])}
               for job in jobs["by_call"].values() if job["state"] == "running"]
    failures = sorted((run_dir / FAILURES_DIR).glob("*.json")) if (run_dir / FAILURES_DIR).is_dir() else []
    language_model = invocation.get("language_model")
    estimate = read_json(run_dir / LOGS_DIR / "estimate.json")
    return {
        "name": config.name,
        "description": config.description,
        "state": {"word": word, "kind": kind},
        "live": jobs["alive"],
        "started_utc": started,
        "invocations": len(machines),
        "updated_utc": jobs["updated_utc"],
        "elapsed": elapsed(started, None if jobs["alive"] else _last_written(run_dir)),
        "running": running,
        "machine": machine.get("machine") if machine else None,
        "budget": machine.get("budget") if machine else None,
        "estimate_hours": _estimated_hours(estimate),
        "failures": [{"name": path.stem, "href": relative(path, run_dir)} for path in failures],
        "agents": agents.recorded(run_dir) if (run_dir / CONFIG_FILE).exists() else [],
        "language_model": naming.language_model(language_model) if language_model else None,
        "links": _links(run_dir),
    }


#: The processes that are one per run, named for a reader, by their call's name.
CALLS = {baseline.CALL: "The composition baseline", stability.CALL: "Stability across seeds",
         co_discovery.CALL: "Features found by several arms", recurrence.CALL: "Ideas found again",
         judge.CALL: "The judge's ratings", reveal.CALL: "The reveal",
         reliability.STAGE_AFTER_REVEAL: "The panel after the reveal",
         report.CALL: "The report's numbers and tables",
         report_files.FIGURES_CALL: "The report's figures"}

#: The processes made once per arm, by their stage, with what each does to the arm.
PER_ARM = {reduction.STAGE: "Reducing", reliability.STAGE: "The panel of"}


def job_label(job: dict[str, Any]) -> str:
    """What a process does, as the report would name it: a cell of an arm by its arm, labels and
    seed, an arm's reduction or panel by its arm, and a process made once per run by what it does.
    A call's name is a file label, which a reader is not shown."""
    call, stage = job["log"], job["stage"]
    if call in CALLS:
        return CALLS[call]
    if stage in PER_ARM:
        model, pool = call.removeprefix(f"{stage}__").split("__", 1)
        return f"{PER_ARM[stage]} {naming.label(arm_name(model, pool))}"
    if "/" in stage:
        labels, seed = call.rsplit("__", 2)[1:]
        return (f"{naming.label(stage)}, {'real' if labels == cells.REAL else 'shuffled'} labels, "
                f"seed {seed.removeprefix('seed_')}")
    return "Finding motifs" if stage == motifs.STAGE else stage


def _lines(path: Path) -> list[dict[str, Any]]:
    """The records of a JSON-lines file, one per line, without a last line still being written."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    found = []
    for line in lines:
        try:
            found.append(json.loads(line))
        except ValueError:
            break
    return found


def _last_written(run_dir: Path) -> datetime | None:
    """When the driver last wrote to its log, which is when a run that is not running stopped."""
    try:
        return datetime.fromtimestamp((run_dir / LOGS_DIR / "driver.log").stat().st_mtime, UTC)
    except OSError:
        return None


def _estimated_hours(estimate: dict[str, Any] | None) -> float | None:
    """The estimate's total for the run: its CPU time spread over its threads."""
    if not estimate or not estimate.get("arms"):
        return None
    cpu = (sum(arm["stage2"] + arm["reduction"] + arm["panel"] for arm in estimate["arms"])
           + sum(estimate.get("before", {}).values()))
    return cpu / max(estimate["threads"], 1) / 3600


def _links(run_dir: Path) -> list[dict[str, str]]:
    found = [("The run's config", run_dir / CONFIG_FILE),
             ("The driver's log", run_dir / LOGS_DIR / "driver.log"),
             ("Every process's time and memory", run_dir / LOGS_DIR / "processes.csv"),
             ("The estimate", run_dir / LOGS_DIR / "estimate.json")]
    return [{"label": label, "href": relative(path, run_dir)} for label, path in found
            if path.exists()]


# ------------------------------------------------------------------------------- Stage 1


def stage1(run_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    """The prepared dataset: the decisions Stage 1 recorded (the target and its classes, the
    sequences, the split and the filters), what the models are told, and the checks. Everything
    here is fixed before the run starts."""
    path = config.dataset.directory / DATASET_FILE
    dataset = read_json(path) or {}
    description_path = DESCRIPTIONS / f"{config.dataset.task}.json"
    description = read_json(description_path) or {}
    target = dataset.get("target", {})
    checks = dataset.get("checks", {})
    acknowledged = dataset.get("acknowledged") or {}
    statuses = [check.get("status") for check in checks.values()]
    counted = {status.replace("_", " "): statuses.count(status) for status in dict.fromkeys(statuses)}
    return {
        "state": "done" if dataset else "waiting",
        "target_kind": target.get("kind"),
        "target_column": dataset.get("columns", {}).get("target"),
        "target_name": description.get("target_name"),
        "classes": [{"label": label, **entry} for label, entry in (dataset.get("classes") or {}).items()],
        "sequences": dataset.get("sequences", {}),
        "covariates": dataset.get("covariates") or {},
        "split": dataset.get("split", {}),
        "splits": dataset.get("splits", {}),
        "filters": dataset.get("filters", []),
        "checks": [{"name": name.replace("_", " "), **check} for name, check in checks.items()],
        "checks_counted": counted,
        "checks_failed": [name for name, check in checks.items()
                          if check.get("status") == "failed" and name not in acknowledged],
        "acknowledged": acknowledged,
        "design_power": dataset.get("design_power", {}),
        "notes": dataset.get("notes"),
        "task": description.get("task"),
        "sequence_layout": description.get("sequence_layout"),
        "blinded": config.dataset.blinded,
        "guidance": config.grid.generated_features.guidance,
        "links": [{"label": label, "href": relative(found, run_dir)} for label, found in (
            ("The dataset's record", path),
            ("What the models are told", description_path),
            ("The dataset's README", config.dataset.directory.parent / "README.md"),
        ) if found.exists()],
    }


# ------------------------------------------------------------------------------- Stage 2


def stage2_(run_dir: Path, config: ExperimentConfig, jobs: dict[str, Any]) -> dict[str, Any]:
    """The grid the run fits: every arm with its cells' states and validation scores, the baseline,
    and the analyses."""
    kind = target_for(config.dataset.target_kind)
    models = list(dict.fromkeys(model for model, _ in config.arm_cells()))
    rows, cell_states, above = [], [], 0
    for pool, entry in config.arms.items():
        row = {"pool": pool, "label": naming.parse(arm_name(entry.models[0], pool)).source_short,
               "arms": []}
        for model in models:
            if model not in entry.models:
                row["arms"].append(None)
                continue
            arm = _arm_cells(run_dir, config, model, pool, kind, jobs)
            cell_states += [cell["state"] for cell in arm["cells"]]
            above += sum(cell["above_chance"] for cell in arm["cells"])
            row["arms"].append(arm)
        rows.append(row)
    base_metrics = [read_json(baseline_dir(run_dir, seed) / cells.METRICS_FILE)
                    for seed in config.grid.seeds]
    base = [number((metrics or {}).get(cells.VALIDATION, {}).get(kind.headline))
            for metrics in base_metrics]
    base_state = call_state(run_dir, baseline.CALL, all(base_metrics), jobs)
    analyses = _analyses(run_dir, config, jobs)
    expected, done = len(cell_states), cell_states.count("done")
    failed = cell_states.count("failed")
    return {
        "state": stage_state([base_state, *cell_states, *(analysis["state"] for analysis in analyses)]),
        "headline": kind.headline,
        "chance": kind.chance,
        "models": [naming.MODELS[model]["short"] for model in models],
        "rows": rows,
        "baseline": mean(base),
        "seeds": list(config.grid.seeds),
        "labels": list(config.grid.labels),
        "analyses": analyses,
        "cells_done": done,
        "cells_expected": expected,
        "checkpoint": [
            {"text": f"Every cell has run ({done} of {expected})",
             "state": "ok" if done == expected and expected else "open"},
            {"text": "No control scores above chance" if not above else
                     f"{above} control cell{'s score' if above != 1 else ' scores'} above chance",
             "state": "no" if above else ("ok" if done == expected and expected else "open")},
            {"text": "Nothing failed" if not failed else f"{failed} cell{'s' * (failed != 1)} failed",
             "state": "no" if failed else ("ok" if done == expected and expected else "open")},
        ],
        "figure": _figure(run_dir, "stage2_grid"),
    }


def _arm_cells(run_dir: Path, config: ExperimentConfig, model: str, pool: str, kind,
               jobs: dict[str, Any]) -> dict[str, Any]:
    """One arm of the grid: each cell's state and validation headline, the mean over the real-label
    cells and that over the controls. A control is marked where its validation interval lies above
    chance, which is how a leak shows (`process/suite.md`). A model fitted to shuffled labels can
    fall below chance by picking a direction along one feature, so falling below is not marked."""
    found, real, permuted = [], [], []
    for labels in config.grid.labels:
        for seed in config.grid.seeds:
            path = cell_dir(run_dir, model, pool, labels, seed) / cells.METRICS_FILE
            metrics = read_json(path)
            block = (metrics or {}).get(cells.VALIDATION, {})
            score = number(block.get(kind.headline))
            low = number((block.get("ci", {}).get(kind.headline) or [None])[0])
            state = call_state(run_dir, stage2.cell_call(model, pool, labels, seed),
                               metrics is not None, jobs)
            found.append({"labels": labels, "seed": seed, "state": state, "score": score,
                          "above_chance": labels != cells.REAL and low is not None
                          and low > kind.chance,
                          "href": relative(path.parent, run_dir)})
            (real if labels == cells.REAL else permuted).append(score)
    return {"model": model, "pool": pool, "label": naming.label(arm_name(model, pool)),
            "writes": calls_language_model(pool), "cells": found, "score": mean(real),
            "control": mean(permuted)}


def _analyses(run_dir: Path, config: ExperimentConfig, jobs: dict[str, Any]) -> list[dict[str, Any]]:
    """The Stage 2 analyses the run makes. The recurrence analysis groups written features, so a
    run with no arm on a written source has none."""
    results = run_dir / RESULTS_DIR
    found = [("Stability across seeds", stability.CALL, results / stability.TABLE),
             ("Features found by several arms", co_discovery.CALL, results / co_discovery.TABLE)]
    if any(written(pool) for _, pool in config.arm_cells()):
        found.append(("Ideas found again", recurrence.CALL, results / recurrence.BY_CONCEPT_TABLE))
    if config.grid.judge:
        found.append(("The judge's ratings", judge.CALL, run_dir / JUDGE_DIR / JUDGE_SUMMARY))
    return [{"label": label, "state": call_state(run_dir, call, path.exists(), jobs),
             "href": relative(path, run_dir) if path.exists() else None}
            for label, call, path in found]


# ------------------------------------------------------------------------------- Stage 3A


def stage3a(run_dir: Path, config: ExperimentConfig, jobs: dict[str, Any]) -> dict[str, Any]:
    """The reduced arms: each one's features and validation scores, the best of them by the reduced
    model's validation score, their held-out scores after the reveal, and the report's figures of
    the reduced models once they are drawn."""
    if config.reduction is None:
        return {"state": "skipped", "arms": [], "best": []}
    kind = target_for(config.dataset.target_kind)
    arms, states = [], []
    for model, pool in config.reduced_arms():
        where = reduction_dir(run_dir, model, pool)
        record = read_json(where / REDUCTION_FILE)
        state = call_state(run_dir, call_name(reduction.STAGE, [(model, pool)]), record is not None,
                           jobs)
        states.append(state)
        record = record or {}
        arms.append({
            "model": model, "pool": pool, "label": naming.label(arm_name(model, pool)),
            "state": state, "href": relative(where, run_dir),
            "seed": record.get("seed"),
            "n_start": record.get("n_start"),
            "n_final": record.get("n_final"),
            "features": _ranked(record.get("final_columns", []), record.get("final_importance")),
            **{f"val_{role}": _score(where / role / cells.METRICS_FILE, cells.VALIDATION, kind)
               for role in ("full", REDUCED_MODEL, "top_n")},
            "held_out_score": _score(where / REDUCED_MODEL / cells.METRICS_FILE, TEST_SPLIT, kind),
        })
    finished = [arm for arm in arms if arm[f"val_{REDUCED_MODEL}"] is not None]
    best = sorted(finished, key=lambda arm: -arm[f"val_{REDUCED_MODEL}"])[:BEST]
    figures = [figure for name in ("stage3a_grid", "stage3a_cost", "stage3a_l1", "stage3a_ebm",
                                   "stage3a_tree", "stage3a_pysr", "stage3a_pysr_front",
                                   "stage3a_igsr")
               if (figure := _figure(run_dir, name))]
    done = states.count("done")
    return {
        "state": stage_state(states),
        "n_features": config.reduction.n_features,
        "headline": kind.headline,
        "arms": sorted(finished, key=lambda arm: -arm[f"val_{REDUCED_MODEL}"])
        + [arm for arm in arms if arm[f"val_{REDUCED_MODEL}"] is None],
        "held_out": any(arm["held_out_score"] is not None for arm in arms),
        "best": best,
        "figures": figures,
        "checkpoint": [
            {"text": f"Every arm is reduced ({done} of {len(states)})",
             "state": "ok" if done == len(states) and states else "open"},
            {"text": "Each reduced model is scored on the held-out rows",
             "state": "ok" if finished and all(arm["held_out_score"] is not None for arm in finished)
             else "open"},
        ],
    }


def _ranked(names: list[str], importance: Any) -> list[dict[str, Any]]:
    """The features of a reduced model, most important first."""
    weights = importance if isinstance(importance, dict) else {}
    ranked = sorted(names, key=lambda name: -abs(number(weights.get(name)) or 0.0))
    return [{"name": name, "importance": number(weights.get(name))} for name in ranked]


def _score(path: Path, split: str, kind) -> float | None:
    return number((read_json(path) or {}).get(split, {}).get(kind.headline))


# ------------------------------------------------------------------------------- Stage 3B


#: The parts of an arm's panel in the order they run, each with its name, the file whose presence
#: says it has finished, and whether it comes after the reveal.
PANEL_PARTS = (
    ("Candidates", panel_files.PREPARE_FILE, False),
    ("Tier 2 in sample", f"{panel_files.tier2_name(panel_files.DISCOVERY, panel_files.REAL)}.json",
     False),
    ("Tier 1", f"{panel_files.tier1_name(panel_files.REAL)}.json", False),
    ("Sealed", panel_files.RECORD_FILE, False),
    ("Tier 2 held out", f"{panel_files.tier2_name(TEST_SPLIT, panel_files.REAL)}.json", True),
    ("Tier 3", panel_files.TIER3_FILE, True),
)


def stage3b(run_dir: Path, config: ExperimentConfig, jobs: dict[str, Any]) -> dict[str, Any]:
    """Each profiled arm's panel (which parts have run, and the three tiers' evidence for each
    feature of its reduced model once the panel has finished), with the time of the reveal and the
    checks on the check arm."""
    if config.reliability is None:
        return {"state": "skipped", "arms": []}
    reveal_record = read_json(run_dir / REVEAL_FILE)
    arms, states = [], []
    for model, pool in config.profiled_arms():
        where = reliability_dir(run_dir, model, pool)
        before = call_state(run_dir, call_name(reliability.STAGE, [(model, pool)]),
                            (where / panel_files.RECORD_FILE).exists(), jobs)
        profile = read_csv(where / FEATURES_FILE)
        after = call_state(run_dir, reliability.STAGE_AFTER_REVEAL, bool(profile), jobs) \
            if reveal_record else "waiting"
        states += [before] + ([after] if reveal_record else [])
        parts = [{"name": name, "done": (where / file).exists(), "after": late}
                 for name, file, late in PANEL_PARTS]
        running = before == "running" or after == "running"
        for part in parts:
            part["state"] = "done" if part["done"] else "waiting"
        if running:
            current = next((part for part in parts if not part["done"]), None)
            if current is not None:
                current["state"] = "running"
        arms.append({
            "model": model, "pool": pool, "label": naming.label(arm_name(model, pool)),
            "state": stage_state([before] + ([after] if reveal_record else [])),
            "parts": parts,
            "selections": _selections(where, config),
            "features": _evidence(profile, config.dataset.target_kind),
            "href": relative(where, run_dir),
        })
    checks = _checks(run_dir, config)
    if not checks:
        checks = [{"text": "The checks with permuted targets and a planted signal", "state": "open"}]
    return {
        "state": stage_state(states),
        "revealed_utc": (reveal_record or {}).get("revealed_utc"),
        "tier3_label": TIER3[config.dataset.target_kind][1],
        "arms": arms,
        "checks": checks,
        "figures": [figure for name in ("stage3b_evidence", "stage3b_thresholds", "stage3b_concepts")
                    if (figure := _figure(run_dir, name))],
        "checkpoint": [
            *checks,
            {"text": "Every tier has run on every profiled arm",
             "state": "ok" if arms and all(arm["state"] == "done" for arm in arms) and reveal_record
             else "open"},
        ],
    }


def _selections(where: Path, config: ExperimentConfig) -> dict[str, int] | None:
    """How many of Tier 1's selections have been made, of those it makes."""
    path = where / panel_files.selections_file(panel_files.REAL)
    try:
        with np.load(path) as stored:
            done = int(np.count_nonzero(stored["done"]))
    except (OSError, KeyError, ValueError):
        return None
    return {"done": done, "of": 2 * config.reliability.panel.pairs}


def _evidence(profile: list[dict[str, str]], target_kind: str) -> list[dict[str, Any]]:
    """The three tiers for each feature of the reduced model, from the panel's profile."""
    column, _ = TIER3[target_kind]
    shown = []
    for row in profile:
        if row.get("in_reported_model") != "True":
            continue
        q = number(row.get("q_bh"))
        shown.append({
            "name": row["feature"],
            "description": row.get("description") or "",
            "pi_hat": number(row.get("pi_hat")),
            "q": q,
            "called": q is not None and q <= 0.05,
            "loss": number(row.get(column)),
            "verdict": (row.get("loco_verdict") or "").replace("_", " "),
        })
    return sorted(shown, key=lambda feature: -(feature["pi_hat"] or 0.0))


def _checks(run_dir: Path, config: ExperimentConfig) -> list[dict[str, str]]:
    """The checks on the check arm, as `process/reliability-panel.md` reads them: with permuted
    targets no feature is stable and none is called on the held-out rows, and with a planted signal
    every planted feature is stable."""
    model, pool = config.reliability.check_arm
    found = read_json(reliability_dir(run_dir, model, pool) / panel_files.CHECKS_FILE) or {}
    items = []
    if (null := found.get("global_null_tier1")) is not None:
        stable = null.get("n_pi_ge_0.7")
        items.append({"text": f"Permuted targets: {stable} stable feature{'s' * (stable != 1)}",
                      "state": "ok" if stable == 0 else "no"})
    if (planted := found.get("planted_tier1")) is not None:
        stable, every = planted.get("n_planted_pi_ge_0.7"), len(planted.get("planted", []))
        items.append({"text": f"Planted signal: {stable} of {every} planted features stable",
                      "state": "ok" if stable == every else "no"})
    if (null := found.get("global_null_tier2")) is not None:
        called = null.get("n_called_bh_0.05")
        items.append({"text": f"Permuted targets, held out: {called} feature{'s' * (called != 1)} "
                              f"called",
                      "state": "ok" if called == 0 else "no"})
    return items


# ------------------------------------------------------------------------------- the report


def report_band(run_dir: Path) -> dict[str, Any]:
    """The report and the files it is written from, as far as they exist."""
    where = report_dir(run_dir)
    found = [("The report", where / report_files.REPORT_FILE),
             ("The report as a PDF", where / "REPORT.pdf"),
             ("Every number of the report", where / report_files.FACTS_FILE),
             ("The report's tables", where / report_files.TABLES_DIR),
             ("The report's figures", where / report_files.FIGURES_DIR)]
    links = [{"label": label, "href": relative(path, run_dir)} for label, path in found if path.exists()]
    written = (where / report_files.REPORT_FILE).exists()
    inputs = (where / report_files.FACTS_FILE).exists()
    return {"state": "done" if written else ("running" if inputs else "waiting"),
            "written": written, "inputs": inputs, "links": links}


def _figure(run_dir: Path, name: str) -> dict[str, str] | None:
    """One of the report's figures, once it is drawn: its file and caption."""
    figures = (read_json(report_dir(run_dir) / report_files.FIGURES_FILE) or {}).get("figures", {})
    entry = figures.get(name)
    if not entry:
        return None
    path = report_dir(run_dir) / entry["path"]
    if not path.exists():
        return None
    return {"href": relative(path, run_dir), "caption": entry.get("caption", "")}


# ------------------------------------------------------------------------------- the whole page


def gather(run_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Everything the page shows, for the template."""
    run_dir = Path(run_dir)
    jobs = jobs_of(run_dir)
    stages = {"1": stage1(run_dir, config), "2": stage2_(run_dir, config, jobs),
              "3A": stage3a(run_dir, config, jobs), "3B": stage3b(run_dir, config, jobs)}
    states = {key: stage["state"] for key, stage in stages.items()}
    return {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "overall": overall(run_dir, config, jobs, states),
        "stages": [{"key": key, "name": STAGES[key][0], "subtitle": STAGES[key][1], **stage}
                   for key, stage in stages.items()],
        "report": report_band(run_dir),
    }
