"""The layout of a run's directory and the names of its calls.

Every stage writes in this layout, the reveal expects the cells in it and the analyses read them
from it, so the layout is stated once here instead of being rebuilt from pieces of strings in each
module:

    <run>/
        config.yaml                                 the run's resolved config, which every stage
                                                    takes its settings from
        config_superseded_<n>.yaml                  an earlier config of the run, kept when the run
                                                    was continued under a changed one
                                                    (--allow-config-change)
        agents.json                                 the agents that carried out the run's stages,
                                                    and the model each runs on
        baseline/seed_<n>/                          the composition baseline, one cell per seed,
                                                    on the real labels
        results/<model>/<pool>/<labels>/seed_<n>/   one cell per arm, label condition and seed,
                                                    beside the tables of the Stage 2 analyses
        reduction/<model>/<pool>/                   Stage 3A: each arm's three models, laid out as
                                                    seqmaestro.reduction.outputs describes
        reliability/<model>/<pool>/                 Stage 3B: each profiled arm's panel, laid out
                                                    as seqmaestro.reliability.panel describes
        proposals/                                  what a language model proposed, one file per
                                                    proposal
        motifs/                                     where a pool includes a motif source: the
                                                    collection scanned with, and the scans and
                                                    discoveries made (seqmaestro/pipeline/motifs.py)
        judge/                                      the judge's scores, where the config runs it
        reveal.json                                 written once, when the held-out split is opened
        report/                                     after the reveal: what the report is written
                                                    from, and the report (seqmaestro.report.files)
        status.html                                 the run's status page, which the driver keeps
                                                    current (seqmaestro.status)
        logs/                                       what each call printed (<call>.log), the
                                                    driver's log, how long each process took and
                                                    the threads and memory it held (processes.csv),
                                                    the threads each panel may use now (threads/),
                                                    the machine as each invocation found it
                                                    (machine.jsonl), and what each process of the
                                                    latest invocation is doing (jobs.json)
        invocations/<call>.json                     the command, the commit and the time of each
                                                    call
        failures/<call>.json                        what failed in a call, absent if nothing failed
        environment/<name>/                         what was installed in each environment used

The baseline is kept outside `results/` so that the Stage 2 analyses, which treat every cell there
as an arm's, never mistake the reference line for an arm. The reduced models and the refits are
kept outside it for the same reason.

**A call** is one process of a run, such as one cell of an arm, one arm's reduction or the reveal.
Its log, its invocation record and its failures file share one name (`call_name`).

**A stage before the reveal reports validation numbers only.** Its log lines are printed while the
held-out split is still closed, and they are labelled `val` so that a log read months later cannot
be mistaken for held-out results. Held-out numbers exist only after the reveal, in each cell's
`metrics.json`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from seqmaestro.arms import Arm, arm_slug
from seqmaestro.harness.cell import VALIDATION
from seqmaestro.reduction.arm import ROLES
from seqmaestro.target import BINARY

RESULTS_DIR = "results"
BASELINE_DIR = "baseline"
REDUCTION_DIR = "reduction"
RELIABILITY_DIR = "reliability"
PROPOSALS_DIR = "proposals"
MOTIFS_DIR = "motifs"
JUDGE_DIR = "judge"
REPORT_DIR = "report"
LOGS_DIR = "logs"
FAILURES_DIR = "failures"
ENVIRONMENT_DIR = "environment"
#: The three models each reduced arm keeps, as cells under its directory.
REDUCED_ROLES = tuple(ROLES)
#: The model of a reduced arm that the reliability panel examines (M*).
REDUCED_MODEL = "reduced"

#: The resolved config, recorded beside the results when a run starts. Every stage reads its
#: settings from this file instead of the one passed to the driver, because the `--set` overrides
#: are part of the run's configuration and the file passed may have been edited since.
CONFIG_FILE = "config.yaml"

#: The record of the agents that carried out the run's stages and the model each runs on, which
#: each agent writes itself (`seqmaestro.pipeline.agents`).
AGENTS_FILE = "agents.json"

def call_name(stage: str | None, arms: Iterable[Arm] = (), *parts: str) -> str:
    """The name of one call of a run: its stage, the arms it works on (if any), and whatever else
    distinguishes it from the stage's other calls. A call for one cell of an arm is named after the
    cell alone (`stage` None), as `model__pool__labels__seed_<n>`."""
    arms = "-".join(arm_slug(*arm) for arm in arms)
    return "__".join(part for part in (stage, arms, *parts) if part)


def cell_dir(run_dir: Path | str, model: str, pool: str, labels: str, seed: int) -> Path:
    """Where one arm cell lives."""
    return Path(run_dir) / RESULTS_DIR / model / pool / labels / f"seed_{seed}"


def baseline_dir(run_dir: Path | str, seed: int) -> Path:
    """Where one composition-baseline cell lives."""
    return Path(run_dir) / BASELINE_DIR / f"seed_{seed}"


def reduction_dir(run_dir: Path | str, model: str, pool: str) -> Path:
    """Where one reduced arm lives: its record, and one cell per model under it."""
    return Path(run_dir) / REDUCTION_DIR / model / pool


def reliability_dir(run_dir: Path | str, model: str, pool: str) -> Path:
    """Where one arm's reliability panel lives: its record, its tiers and its refit cells."""
    return Path(run_dir) / RELIABILITY_DIR / model / pool


def report_dir(run_dir: Path | str) -> Path:
    """Where a run's report and everything it is written from live."""
    return Path(run_dir) / REPORT_DIR


def validation_line(metrics: dict, target_kind: str) -> str:
    """A cell's main validation numbers, for a log line as the cell finishes."""
    block = metrics[VALIDATION]
    if target_kind == BINARY:
        return (f"val acc {block['accuracy']:.3f}  auprc {block['auprc']:.3f}  "
                f"auroc {block['auroc']:.3f}")
    return f"val r2 {block['r2']:.3f}  rho {block['spearman_rho']:.3f}  rmse {block['rmse']:.4g}"


def target_line(y, target_kind: str) -> str:
    """The training targets in a few words, for the head of a log."""
    y = np.asarray(y, dtype=float)
    if target_kind == BINARY:
        return f"positive rate {y.mean():.4f}"
    return f"target mean {y.mean():.4g}, sd {y.std(ddof=1):.4g}"
