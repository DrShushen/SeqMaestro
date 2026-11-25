"""The run's config, the estimate, and the launch.

The config extends the shared config for the kind of target (`_shared.yaml` or
`_shared_continuous.yaml`), and states the dataset and what the preset changes:

- `quick` is the example's quick run (`experiments/configs/example_quick.yaml`) on the user's data:
  two seeds, the arms on fixed sources without PySR, and a small reliability panel. No arm calls a
  language model, so the run takes minutes, costs nothing, and finds no feature a model writes.
- `full` is the shared config as it stands: every arm, five seeds and the full panel, which takes
  hours and calls the language model `.env` names.

Where sequences differ in length, each k-mer count is divided by the number of positions in its
sequence, so that it does not grow with the length. The command starts the driver through its
command line, as a user would type it: `--estimate` first, and then the run, through the reveal and
in a session of its own (`--detach`).
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from seqmaestro import files
from seqmaestro.cli import ui
from seqmaestro.cli.prepare import Paths
from seqmaestro.pipeline import agents
from seqmaestro.pipeline.layout import CONFIG_FILE
from seqmaestro.pipeline.repository import REPO, RUNNERS, RUNS
from seqmaestro.target import BINARY

#: The presets, each with what it runs and what that takes.
PRESETS = {
    "quick": ("two seeds, the arms on positions and k-mers, and a small panel",
              "minutes, no language model"),
    "full": ("every arm of the shared config, five seeds, and the full panel",
             "hours, with the language model"),
}

#: The shared configs a config extends, by the kind of its target.
SHARED = {BINARY: "_shared.yaml", "continuous": "_shared_continuous.yaml"}
CONFIGS = REPO / "experiments" / "configs"

#: What `quick` changes in the shared config: `example_quick.yaml`'s changes to `example.yaml`.
QUICK: dict[str, Any] = {
    "grid": {"seeds": [42, 43]},
    "sources": {"llm_generated": None, "llm_generated_iterative": None},
    "arms": {"positional": {"models": ["l1", "ebm", "xgboost", "deft_or_cart"]},
             "llm_generated": None, "llm_generated_iterative": None,
             "positional+llm_generated_joint": None},
    "reliability": {"pairs": 10, "permutations": 2000},
}
#: What `quick` also changes for a continuous target, whose shared config adds IGSR's arm.
QUICK_CONTINUOUS: dict[str, Any] = {"arms": {"llm_generated_joint": None}, "models": {"igsr": None}}

#: The name the command line records itself under in a run, as the agent that carried out Stage 1,
#: launched Stage 2 and wrote the report.
AGENT = "SeqMaestro command line"
#: How long to wait for a run the driver has started to record its config, before the command
#: records itself in it.
RECORD_WAIT_SECONDS = 30


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = (_merge(merged[key], value) if isinstance(value, dict)
                       and isinstance(merged.get(key), dict) else value)
    return merged


def config_document(paths: Paths, table: Path, kind: str, preset: str,
                    variable_length: bool) -> dict[str, Any]:
    """The config of a run of the prepared dataset under a preset."""
    if preset not in PRESETS:
        raise ValueError(f"No preset {preset!r}; the presets are {', '.join(PRESETS)}.")
    shared = CONFIGS / SHARED[kind]
    splits = paths.splits
    document: dict[str, Any] = {
        "extends": shared.name if paths.config.parent == CONFIGS else str(shared),
        "name": paths.name,
        "description": f"The {preset} run of {table.name}, prepared and started by `seqmaestro run`.",
        "dataset": {"task": paths.name, "blinded": False,
                    "splits": str(splits.relative_to(REPO)) if splits.is_relative_to(REPO)
                    else str(splits)},
    }
    if preset == "quick":
        document = _merge(document, QUICK)
        if kind != BINARY:
            document = _merge(document, QUICK_CONTINUOUS)
    if variable_length:
        document = _merge(document, {"sources": {"kmer": {"normalise": "per_length"}}})
    return document


def write_config(paths: Paths, document: dict[str, Any]) -> Path:
    """Write the config where the dataset's configs go, and return its path."""
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Written by `seqmaestro run` (seqmaestro/cli/launch.py). It extends the shared config\n"
              "# and states what this run changes; experiments/configs/README.md describes every\n"
              "# section. Edit it and run the driver to change anything else (docs/user/by-hand.md).\n\n")
    return files.write_text(paths.config, header + yaml.safe_dump(document, sort_keys=False,
                                                                  allow_unicode=True, width=100))


def run_directory(name: str) -> Path:
    """Where a run started today goes, as the driver names it by default."""
    return RUNS / f"{date.today():%Y-%m-%d}_{name}_full"


def driver(config: Path, run_dir: Path, *flags: str) -> list[str]:
    """The driver's command line, as `docs/user/by-hand.md` gives it."""
    return [sys.executable, ui.shown(RUNNERS / "run_experiment.py"), "--config", ui.shown(config),
            "--run-dir", ui.shown(run_dir), *flags]


def stream(command: list[str]) -> int:
    """Run a command from the repository's folder, showing each line it prints as it prints it, and
    return its exit code."""
    with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            ui.output(line.rstrip("\n"))
    return process.returncode


def estimate(config: Path, run_dir: Path) -> int:
    """The driver's estimate of each arm's time, and of the calls, cost and waiting of the arms that
    call a language model, kept in the run's `logs/`."""
    return stream(driver(config, run_dir, "--estimate"))


def start(config: Path, run_dir: Path) -> int:
    """Start the run in a session of its own, through the reveal. The driver returns once the run is
    under way, or says why it stopped."""
    return stream(driver(config, run_dir, "--detach"))


def record(run_dir: Path, model: str, stages: list[str]) -> bool:
    """Record the command line in the run as the agent of `stages`, with the model it called, once
    the run has recorded its config. Returns whether it could."""
    waited = 0.0
    while not (run_dir / CONFIG_FILE).exists() and waited < RECORD_WAIT_SECONDS:
        time.sleep(0.5)
        waited += 0.5
    try:
        agents.record(run_dir, AGENT, model, stages)
    except ValueError:
        return False
    return True
