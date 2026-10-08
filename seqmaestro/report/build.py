"""Assemble a run's report inputs: facts, tables, model internals and figure inputs.

`files.py` names each file, and `experiments/analysis/build_report_figures.py` draws the figures
from `figure_inputs.json`. The steps run in dependency order. `collect` builds the facts from the
run alone. Concepts need the columns the reduction kept for each profiled arm evaluated on D, and
the drawn models need to be read from their saved files. Both can mean calling a vendored method's
interpreter, so the caller (`seqmaestro.pipeline.report`) does them and passes the results to
`arm_columns` and `write`. `write` then writes everything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from seqmaestro.harness.heldout import TEST_SPLIT
from seqmaestro.harness.stability import Matcher
from seqmaestro.report import concepts as concept_module
from seqmaestro.report import facts as facts_module
from seqmaestro.report import figure_inputs, markdown, naming, tables
from seqmaestro.report.files import FACTS_FILE, FIGURE_INPUTS_FILE, INTERNALS_DIR, TABLES_DIR
from seqmaestro.report.sources import Run


def collect(run: Run, matcher: Matcher) -> dict[str, Any]:
    """Every fact the run supports, with the headline sets and the parameters."""
    dataset = run.dataset
    kind = dataset.target_kind
    names = facts_module.metrics(kind)
    headline = names["headline"]
    stage3a = facts_module.stage3a(run, names)
    found: dict[str, Any] = {
        "run": run.directory.name,
        "target_kind": kind,
        "metrics": names,
        "evaluated_on": TEST_SPLIT,
        "reveal": {"revealed_utc": run.reveal["revealed_utc"], "override": run.reveal["override"]},
        **facts_module.target(dataset),
        "directions": facts_module.directions(dataset),
        "stage1": facts_module.stage1(run),
        "stage2": facts_module.stage2(run, names),
        "baseline": facts_module.baseline(run, stage3a["feature_budget"], names),
        "stage3a": stage3a,
        "stage3b": facts_module.stage3b(run, names),
    }
    found["headline_set"] = facts_module.headline_set(
        {arm: entry[f"reduced_{headline}"] for arm, entry in found["stage3a"]["arms"].items()}, kind)
    found["headline_set_stage2"] = facts_module.headline_set(
        {arm: entry[headline] for arm, entry in found["stage2"]["arms"].items()}, kind)
    found["parameters"] = facts_module.parameters(run, found, matcher.threshold)
    found["parameters"]["shared_at_least"] = markdown.word(tables.SHARED_AT_LEAST)
    return found


def arm_columns(facts: dict[str, Any], arm: str, records: list, values: np.ndarray
                ) -> concept_module.ArmColumns:
    """The columns the reduction kept for one profiled arm, for the concept grouping, with what the
    facts say of each: its rank and marginal effect from Stage 3A, and its evidence from the panel,
    which holds only the columns the reduced model uses."""
    return concept_module.ArmColumns(
        arm=arm, records=records, values=values, features=facts["stage3a"]["arms"][arm]["features"],
        evidence={row["feature"]: row for row in facts["stage3b"]["arms"][arm]["reported"]})


def _plain(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not serialisable: {value!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_plain) + "\n")


def write(out_dir: str | Path, facts: dict[str, Any], concepts: list[concept_module.Concept],
          internals: dict[str, dict]) -> dict[str, Any]:
    """Write the report inputs. `internals` maps a headline slot to its model's exported structure.
    Returns what was written, by kind."""
    out_dir = Path(out_dir)
    facts = {**facts,
             "concepts": [concept.to_dict() for concept in concepts],
             "concepts_arms": list(facts["stage3b"]["arms"]),
             "internals": {slot: {"arm": facts["headline_set"][slot],
                                  "file": f"{INTERNALS_DIR}/{naming.slug(facts['headline_set'][slot])}.json"}
                           for slot in internals}}
    _write_json(out_dir / FACTS_FILE, facts)

    rendered = {**tables.build(facts, concepts), **tables.equation(facts, internals)}
    table_dir = out_dir / TABLES_DIR
    table_dir.mkdir(parents=True, exist_ok=True)
    for stale in table_dir.glob("*.md"):
        if stale.stem not in rendered:
            stale.unlink()
    for name, text in rendered.items():
        (table_dir / f"{name}.md").write_text(text + "\n")

    for slot, structure in internals.items():
        _write_json(out_dir / facts["internals"][slot]["file"], structure)
    figures = figure_inputs.build(facts, concepts, internals)
    _write_json(out_dir / FIGURE_INPUTS_FILE, figures)
    return {"facts": facts, "tables": sorted(rendered), "figures": sorted(figures)}
