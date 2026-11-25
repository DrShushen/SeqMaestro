"""Stage 3B: the reliability panel of each arm's reduced model, before the reveal and after it.

    python experiments/runners/run_reliability.py --run-dir experiments/runs/<date>_<name>
    python experiments/runners/run_reliability.py --run-dir ... --after-reveal

Where the config has a `reliability` section, the panel profiles every reduced arm except those of
a black-box model and those an entry of `arms` leaves out with a reason (`not_profiled`).
`docs/developer/reliability.md` says what is computed and why.

**Before the reveal** (the default, which refuses a revealed run): Step 0 and its reproduction
checks, Tier 2 in-sample on the training and validation rows, the leave-one-out refits saved as
cells, and Tier 1's selections. An arm makes `2 * pairs` selections (three times that on the check
arm), so `--workers` spreads them over parallel workers, limited to as many as fit in the memory the
panel may hold (`--memory-gb`, or what the driver gave the process), and the results depend on
neither setting. Without `--workers`, the panel runs as many workers as the driver gave it threads,
and follows that number as the driver lends it the threads that free up while it runs and takes
them back. Each arm ends with its record, which the reveal checks for and seals. The driver runs
this part once per arm (`--arms`), each call as soon as its arm is reduced.

**After the reveal** (`--after-reveal`, which refuses a run that has not been revealed): Tier 3
from the revealed predictions, the global-null and planted-signal checks, and each arm's tables,
with one row per arm in `reliability/summary.csv`.

The panel's fits are made wherever the arm's model lives (`seqmaestro.pipeline.sides`). Before the
reveal, every step of every arm whose files exist is skipped unless `--force` is given, whereas
after it, Tier 3, the checks and the tables are computed again on every call.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

from seqmaestro import files
from seqmaestro.arms import Arm, arm_name
from seqmaestro.execution import allowed_memory, allowed_threads, threads_file
from seqmaestro.harness.heldout import refuse_if_revealed, reveal_record
from seqmaestro.pipeline.config import ConfigError, ExperimentConfig, load_recorded
from seqmaestro.pipeline.layout import (
    CONFIG_FILE,
    REDUCED_MODEL,
    RELIABILITY_DIR,
    call_name,
    reduction_dir,
    reliability_dir,
)
from seqmaestro.pipeline.records import Failures, snapshot_environments
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.sides import side_of
from seqmaestro.pipeline.stage2 import selected_arms
from seqmaestro.provenance import record_call
from seqmaestro.reduction import outputs
from seqmaestro.reduction.arm import Source
from seqmaestro.reliability.panel import Panel

#: The stage's name in a call's name (`layout.call_name`), on each side of the reveal.
STAGE, STAGE_AFTER_REVEAL = "reliability", "reliability_after_reveal"
SUMMARY_FILE = "summary.csv"


def panel_for(run_dir: Path, config: ExperimentConfig, arm: Arm, rows: Rows) -> Panel:
    """One profiled arm's panel, with the fits made wherever its model lives."""
    dataset = rows.dataset
    reduced = reduction_dir(run_dir, *arm)
    if not outputs.is_reduced(reduced):
        raise FileNotFoundError(f"{reduced} has no {outputs.REDUCTION_FILE}: the arm has not been "
                                f"reduced, and the panel profiles its reduced model.")
    reduction = outputs.read(reduced)
    source = Source.read(run_dir / reduction["source_cell"])
    discovery, n_train = rows.discovery(), len(rows.train)
    fits = side_of(source.run, source.cell).fits(source, discovery, n_train, dataset.columns)
    return Panel(reliability_dir(run_dir, *arm), run_dir=run_dir, reduction=reduction,
                 reduced_cell=reduced / REDUCED_MODEL, source=source, fits=fits,
                 settings=config.reliability.panel, discovery=discovery, n_train=n_train,
                 columns=dataset.columns, is_check_arm=arm == config.reliability.check_arm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=None, metavar="MODEL/POOL",
                        help="profile only these of the arms the config profiles")
    parser.add_argument("--after-reveal", action="store_true",
                        help="Tier 3, the checks and the tables, from what the reveal wrote")
    parser.add_argument("--workers", type=int, default=None,
                        help="processes for Tier 1's selections (default: the threads this process "
                             "was given, and those the driver lends it while it runs, or every "
                             "core). The results do not depend on it")
    parser.add_argument("--memory-gb", type=float, default=None,
                        help="the gigabytes the panel may hold, its workers included; Tier 1 runs "
                             "fewer workers where fewer fit (default: what this process was given, "
                             "or no limit). The results do not depend on it")
    parser.add_argument("--force", action="store_true",
                        help="before the reveal, compute again what is already on disk, Tier "
                             "1's selections included")
    return parser


def tier1_workers(args: argparse.Namespace) -> tuple[int, Path | None]:
    """How many workers Tier 1 starts with, and the file it follows as the driver lends the panel
    threads and takes them back. A number given with `--workers` stays fixed. Otherwise Tier 1
    starts with the threads this process was given (or every core) and follows the file."""
    if args.workers is not None:
        return args.workers, None
    return allowed_threads() or os.cpu_count(), threads_file()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    try:
        config = load_recorded(run_dir)
        if config.reliability is None:
            raise ConfigError(f"{run_dir / CONFIG_FILE} has no `reliability` section, so there is "
                              f"no panel to run.")
        arms = selected_arms(args.arms, config.profiled_arms(), "arms the run's config profiles")
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    if args.after_reveal:
        if reveal_record(run_dir) is None:
            print(f"{run_dir} has not been revealed: Tier 3 and the tables need the test split's "
                  f"predictions. Run run_reveal.py first.", file=sys.stderr)
            return 2
    else:
        refuse_if_revealed(run_dir, "run_reliability")

    stage = STAGE_AFTER_REVEAL if args.after_reveal else STAGE
    call = call_name(stage, arms if args.arms else ())
    memory = allowed_memory() if args.memory_gb is None else int(args.memory_gb * 2**30)
    workers, follow = tier1_workers(args)
    record_call(run_dir, call, vars(args))

    rows = Rows.of(config)
    print(f"reliability panel, {'after' if args.after_reveal else 'before'} the reveal: "
          f"{len(arms)} arm(s); D = {len(rows.train) + len(rows.val)} rows ({len(rows.train)} "
          f"train)", flush=True)
    for name, why in config.not_profiled().items():
        print(f"  not profiled: {name} ({why})", flush=True)

    summaries, failures = [], Failures(run_dir, call)
    for arm in arms:
        started = time.time()
        print(f"\n{arm_name(*arm)}", flush=True)
        try:
            panel = panel_for(run_dir, config, arm, rows)
            if args.after_reveal:
                summaries.append(panel.after_reveal(log=print))
            else:
                panel.before_reveal(workers=workers, memory=memory, threads_file=follow,
                                    force=args.force, log=print)
        except Exception:  # a failure in one arm must not stop the others
            failures.record(arm_name(*arm))
            continue
        print(f"    done in {time.time() - started:.1f}s", flush=True)

    if summaries:
        out_dir = run_dir / RELIABILITY_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        files.write_text(out_dir / SUMMARY_FILE, pd.DataFrame(summaries).to_csv(index=False))
    # A model whose search ran in another environment (PySR's) has that environment recorded too.
    snapshot_environments(run_dir, {model for model, _ in arms})
    return failures.close()
