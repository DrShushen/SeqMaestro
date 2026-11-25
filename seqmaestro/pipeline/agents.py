"""The record of the agents that carried out a run's stages, which each agent writes itself.

A run is carried out by an AI agent that works through `process/` with the user: the agent
prepares the dataset, launches and checks the run, and writes the report. Which model that agent
runs on matters to the run as much as the model the run calls, since the agent makes and records
the decisions of Stage 1 and writes the report's interpretation. The code cannot tell which model
an agent runs on, so each agent states it. When it starts a stage of a run, it records its name,
the model it runs on and the stage, and the report names every agent recorded:

    experiments/runners/record_agent.py --run-dir <run> --agent "Claude Code" \
        --model claude-opus-5-5 --stages 1 2

The record is `<run>/agents.json`, which holds one entry for each call in the order the calls were
made. Because Stage 1 is carried out before the run exists, the agent that carried it out records
it together with Stage 2, once the launch has made the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from seqmaestro import files
from seqmaestro.pipeline.layout import AGENTS_FILE, CONFIG_FILE

#: The stages an agent carries out, named as in `process/`, which has one file for each.
STAGES = ("1", "2", "3A", "3B", "report")


def recorded(run_dir: str | Path) -> list[dict]:
    """Every agent the run records, in the order recorded, or an empty list if the run has no
    record."""
    path = Path(run_dir) / AGENTS_FILE
    return json.loads(path.read_text()) if path.exists() else []


def record(run_dir: str | Path, agent: str, model: str, stages: Iterable[str]) -> dict:
    """Add an agent to the run's record and return the entry added. The entry names the agent
    (such as `Claude Code`), the model it runs on (by the exact name its harness gives it) and the
    stages it carries out.

    The call is refused unless the directory holds a run the driver started, so that a mistyped
    path does not create a stray directory, and unless every stage is one of `STAGES`."""
    run_dir = Path(run_dir)
    if not (run_dir / CONFIG_FILE).exists():
        raise ValueError(f"{run_dir} holds no {CONFIG_FILE}, so it is no run the driver started. An "
                         f"agent records itself once the launch has made the run.")
    agent, model, stages = agent.strip(), model.strip(), list(stages)
    if not agent or not model:
        raise ValueError("Name the agent and the model it runs on, each as text.")
    unknown = [stage for stage in stages if stage not in STAGES]
    if not stages or unknown:
        raise ValueError(f"Name each stage you carry out as one of {', '.join(STAGES)}, not "
                         f"{', '.join(unknown) or 'none'}.")
    entry = {"agent": agent, "model": model,
             "stages": sorted(set(stages), key=STAGES.index),
             "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    files.write_json(run_dir / AGENTS_FILE, [*recorded(run_dir), entry])
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--agent", required=True,
                        help="the agent you are, such as Claude Code")
    parser.add_argument("--model", required=True,
                        help="the model you run on, by the exact name your harness gives you")
    parser.add_argument("--stages", nargs="+", required=True, choices=STAGES,
                        help="the stages of the run you carry out")
    args = parser.parse_args(argv)
    try:
        entry = record(args.run_dir, args.agent, args.model, args.stages)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"recorded {entry['agent']} ({entry['model']}) for stage(s) {', '.join(entry['stages'])} "
          f"in {args.run_dir / AGENTS_FILE}")
    return 0
