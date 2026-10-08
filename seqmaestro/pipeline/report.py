"""Everything a run's report is written from. **Runs in `SeqMaestro_Core`, after the reveal.**

    python experiments/analysis/build_report_inputs.py --run-dir experiments/runs/<date>_<name>

Writes `<run>/report/`: `facts.json`, `tables/*.md`, `internals/*.json` and `figure_inputs.json`
(see `seqmaestro/report/build.py`). `build_report_figures.py` then draws the figures from
`figure_inputs.json` under `SeqMaestro_LocalUtils`, and the report is written against
`docs/report/TEMPLATE.md` and expanded by `assemble_report.py`. `process/reporting.md` gives the
whole sequence.

**What the run holds is read from its config**, not found on disk: every arm, seed and label of the
grid, the baseline, the reduced and the profiled arms, and why any reduced arm was not profiled. A
missing cell stops the build, and every cell must be one the reveal sealed. The run's Stage 2
analyses are read where they exist, and one the run did not produce is reported as not measured.

Two things need more than reading files, and both are asked of the side an arm's model lives on
(`seqmaestro.pipeline.sides`), because both can mean calling a vendored method's interpreter:

- **Concepts** group every profiled arm's reported features by what they compute on D (train +
  val): `Side.reported_columns`.
- **Model internals** of the models a report draws, one per interpretable class in the headline
  set: `Side.internals`. A model loaded in this process is read after a check that it reproduces
  its cell's validation scores, and a DEFT tree is read on DEFT's side.

The classes, or a continuous target's name and meaning, come from the dataset's `dataset.json`.
Nothing here refits, scores or reads the test split: every test number is one the reveal wrote.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seqmaestro.arms import arm_name, split_arm
from seqmaestro.harness import cell as cells
from seqmaestro.harness.stability import BehaviouralMatcher
from seqmaestro.llm import load_task
from seqmaestro.pipeline.analyses import co_discovery, judge, recurrence, stability
from seqmaestro.pipeline.config import ConfigError, ExperimentConfig, load_recorded
from seqmaestro.pipeline.layout import (
    JUDGE_DIR,
    REDUCED_MODEL,
    REDUCED_ROLES,
    RESULTS_DIR,
    baseline_dir,
    cell_dir,
    reduction_dir,
    reliability_dir,
    report_dir,
)
from seqmaestro.pipeline.repository import DESCRIPTIONS
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.sides import side_of
from seqmaestro.provenance import record_call
from seqmaestro.reduction.arm import Source
from seqmaestro.report import build, sources
from seqmaestro.report.concepts import build as build_concepts
from seqmaestro.report.facts import HEADLINE_FAMILIES

#: The name of this call (`layout.call_name`).
CALL = "report_inputs"

#: What a report reads of the Stage 2 analyses, and where each analysis writes it, relative to the
#: run (`seqmaestro.pipeline.analyses`).
STAGE2_ANALYSES = {
    "stability": Path(RESULTS_DIR) / stability.TABLE,
    "co_discovery_by_arm": Path(RESULTS_DIR) / co_discovery.BY_ARM_TABLE,
    "cross_arm_recurrence": Path(RESULTS_DIR) / recurrence.ACROSS_ARMS_TABLE,
    "concept_recurrence": Path(RESULTS_DIR) / recurrence.RECORD,
    "judge": Path(JUDGE_DIR) / judge.SUMMARY,
}


def drawn_slots(target_kind: str) -> list[str]:
    """The headline slots whose model a report draws for a `target_kind` run: one per interpretable
    model class of the headline set."""
    return [f"best_{family}" for family in HEADLINE_FAMILIES[target_kind]]


def inventory(config: ExperimentConfig) -> sources.Inventory:
    """Every cell and stage directory the config says the run holds, relative to it."""
    return sources.Inventory(
        suite={arm_name(*arm): {labels: {seed: cell_dir("", *arm, labels, seed)
                                           for seed in config.grid.seeds}
                                  for labels in config.grid.labels}
                 for arm in config.arm_cells()},
        baseline={seed: baseline_dir("", seed) for seed in config.grid.seeds},
        reduced={arm_name(*arm): reduction_dir("", *arm) for arm in config.reduced_arms()},
        profiled={arm_name(*arm): reliability_dir("", *arm) for arm in config.profiled_arms()},
        not_profiled=config.not_profiled(),
        # The judge is listed only where the config runs it, so a run that leaves it out has no
        # judge column, while one whose judge failed shows it as not measured.
        analyses={name: path for name, path in STAGE2_ANALYSES.items()
                  if name != "judge" or config.grid.judge},
    )


def reduced_cell(arm: str) -> Path:
    """An arm's reduced model, relative to the run."""
    return reduction_dir("", *split_arm(arm)) / REDUCED_MODEL


def concept_columns(run_dir: Path, run: sources.Run, facts: dict, rows: Rows) -> list:
    """Each profiled arm's reported columns on D, for the concept grouping."""
    found = []
    for arm in facts["stage3b"]["arms"]:
        record = run.reduced[arm].record
        source = Source.read(run_dir / record["source_cell"])
        names = record["final_columns"]
        values = side_of(cells.read_run(run_dir / reduced_cell(arm))).reported_columns(
            run_dir, split_arm(arm), names, source, rows)
        by_name = {record.name: record for record in source.records}
        found.append(build.arm_columns(facts, arm, [by_name[name] for name in names], values))
    return found


def drawn_internals(run_dir: Path, run: sources.Run, facts: dict, rows: Rows) -> dict[str, dict]:
    """What is inside each model the report draws, read from the saved model."""
    found = {}
    for slot in drawn_slots(facts["target_kind"]):
        arm = facts["headline_set"].get(slot)
        if arm is None:
            continue
        cell = reduced_cell(arm)
        marginal = {row["name"]: row["marginal_effect"]
                    for row in facts["stage3a"]["arms"][arm]["features"]}
        found[slot] = side_of(cells.read_run(run_dir / cell), cell).internals(
            run_dir, cell, run.reduced[arm].record, rows, marginal)
    return found


def blinding(config: ExperimentConfig) -> dict:
    """Whether the run withheld a fact from the language models, and the fact: the config's
    `dataset.blinded`, and its task description's `judge_only`, which the driver checked agree."""
    withheld = (load_task(DESCRIPTIONS, config.dataset.task).judge_only
                if config.dataset.blinded else None)
    return {"blinded": config.dataset.blinded, "withheld": withheld}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir

    try:
        config = load_recorded(run_dir)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    rows = Rows.of(config)
    try:
        run = sources.load(run_dir, inventory(config), rows.dataset, REDUCED_ROLES,
                           blinding=blinding(config))
    except sources.ReportError as error:
        print(error, file=sys.stderr)
        return 2
    record_call(run_dir, CALL, vars(args))

    matcher = BehaviouralMatcher()
    facts = build.collect(run, matcher)
    print(f"facts: {len(facts['stage2']['arms'])} arms, {len(facts['stage3a']['arms'])} reduced, "
          f"{len(facts['stage3b']['arms'])} profiled", flush=True)
    concepts = build_concepts(concept_columns(run_dir, run, facts, rows), matcher,
                              (facts["metrics"]["association"], facts["metrics"]["association_centre"]))
    print(f"concepts: {len(concepts)}, {sum(len(c.arms) > 1 for c in concepts)} reached by more "
          f"than one arm", flush=True)
    drawn = drawn_internals(run_dir, run, facts, rows)
    for slot in drawn_slots(facts["target_kind"]):
        if slot not in drawn:
            print(f"  no {slot}: no arm of that model class was reduced", flush=True)
    out = report_dir(run_dir)
    written = build.write(out, facts, concepts, drawn)
    print(f"wrote {out}: {len(written['tables'])} tables, {len(drawn)} model internals, "
          f"{len(written['figures'])} figures to draw", flush=True)
    for name, absent in ((name, found is None) for name, found in run.analyses.items()):
        if absent:
            print(f"  not measured: the run has no {STAGE2_ANALYSES[name]}", flush=True)
    return 0
