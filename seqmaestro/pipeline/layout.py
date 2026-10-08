"""Where a run puts things, and how a call of a run is named.

Every stage writes in this layout, the reveal expects its cells there, and the analyses read them
from there, so the layout is written down once here instead of rebuilt from string pieces in each
module:

    <run>/
        config.yaml                                 the run's resolved config
        baseline/seed_<n>/                          the composition baseline, real labels
        results/<model>/<pool>/<labels>/seed_<n>/   one cell per arm, labels and seed
        reduction/<model>/<pool>/{full,reduced,top_n}/   Stage 3A: each arm's three models
        reliability/<model>/<pool>/                 Stage 3B: each profiled arm's panel,
            loco/<label>/                           with its leave-one-out refits as cells
        proposals/                                  what a language model proposed, for each cell
        judge/                                      the judge's scores, where a run asks for them
        report/                                     after the reveal: what a report is written
                                                    from, and the report itself
        logs/<call>.log                             what each call printed
        invocations/<call>.json                     the command, the commit and the time of each call
        failures/<call>.json                        what failed in a call, absent when it ended clean
        environment/<name>/                         what was installed in each environment used

The baseline sits outside `results/` so the Stage 2 analyses, which treat every cell there as an
arm's, never mistake the reference line for an arm. The reduced models and the refits sit outside
it for the same reason.

**A call** is one process of a run: one cell of an arm, one arm's reduction, the reveal. Its log,
its invocation record and its failures file share one name (`call_name`).

**A stage before the reveal reports validation numbers only.** Its log lines are printed while the
test split is still closed, and they say `val` so a log read months later cannot be taken for test
results. Test numbers exist only after the reveal, in each cell's `metrics.json`.
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
JUDGE_DIR = "judge"
REPORT_DIR = "report"
LOGS_DIR = "logs"
FAILURES_DIR = "failures"
ENVIRONMENT_DIR = "environment"
#: The three models each reduced arm keeps, as cells under its directory.
REDUCED_ROLES = tuple(ROLES)
#: The reduced arm's model the reliability panel examines: M*.
REDUCED_MODEL = "reduced"

#: The resolved config, recorded beside the results when a run starts. Every stage reads its
#: settings from this file, and not from the one passed to the driver, because `--set` overrides
#: are part of what a run was configured with, and the file passed may have been edited since.
CONFIG_FILE = "config.yaml"

def call_name(stage: str | None, arms: Iterable[Arm] = (), *parts: str) -> str:
    """What one call of a run is named: its stage, the arms it works on where it names some, and
    whatever else tells it from the stage's other calls. A call for one cell of an arm is named for
    the cell alone (`stage` None), as `model__pool__labels__seed_<n>`."""
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
