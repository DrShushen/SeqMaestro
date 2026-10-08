"""Run a dataset through the whole pipeline, start to finish, unattended: every arm and analysis,
the reduction and the reliability panel, the reveal, and the report's inputs.

    experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml

Launch it detached and it survives the session that started it:

    nohup $CORE experiments/runners/run_experiment.py --config <config> > /dev/null 2>&1 &

**The config file says what a run is**: which splits, which grid and which arms
(`experiments/configs/`). The driver records it, resolved, as `<run>/config.yaml`, and every stage
reads its settings from that file, so the driver passes a stage nothing but the run directory and
which part of the run it is to work on. This module holds only what is true of the pipeline
whatever dataset it runs on: the stages, their order, and which of them may run at once.

**Every stage is resumable, and the driver is idempotent.** Run again after a crash, a reboot or a
killed process, it skips the cells already done and continues. A stage cannot tell a finished cell
from one computed under different settings, so the driver refuses to continue a run whose recorded
config differs from the one it is given, since that would mix two configurations into one table
with nothing to show it. A recorded config states the `version` of its shape, and the driver also
refuses to continue a run recorded under another, since the stages could not take that run's
settings from it.

**To resume, pass the run directory.** The default directory is dated, so a resume on a later day
than the launch would start an empty run, redo hours of work, and split the results across two
directories, which nobody notices until they read both. A resume usually happens with nobody
watching, so a warning is not enough: the driver refuses to start a new run when earlier ones for
the same config exist, unless `--new-run` says that is the intent.

**A failing stage does not stop the stages after it, except the reveal** (below). Each stage
isolates its cells, and a config lists its cheap arms first, which start first, so that an abort
still leaves them done and analysable. Failures are counted and repeated at the end. `logs/` holds
what each call printed, and `failures/` what failed in it.

**The processes of a run run side by side** (`seqmaestro/pipeline/schedule.py`). An arm is one
model fitted on one pool of sources, as the config's `arms` lists them (`seqmaestro.arms` says which
model takes which pool), and every arm runs as one process per (labels, seed). The reduction and the
reliability panel run one process per arm. A process starts once what it waits for has finished and
the machine has room for it, and of two that could both start, the one listed first starts first.
The arms wait for nothing. An arm's reduction waits for its real-label cells and its panel for its
reduction, so a cheap arm is reduced and profiled while slow arms still run. The analyses wait for
every arm, and the reveal for everything before it.

**The machine is probed as the run starts**: the CPUs this process may use and the memory available
to it (`seqmaestro.execution.probe`). The run's budget is drawn from that: threads, memory, and how
many processes may call a language model at once, which the endpoint's throughput bounds and which
defaults to `LLM_SLOTS`. `--max-cpus` and `--max-memory` lower the first two, for a machine that is
also in use for other work, and `--llm-slots` sets the third. None of them changes a result, so
they are flags of the driver, and a run resumed on another machine may use other values. Each
invocation appends the probe and the budget to `logs/machine.jsonl`, and each process leaves a row
in `logs/processes.csv` with its threads, its time and the memory it held.

**The composition baseline starts first**: base fractions in a standardised sparse linear model at
every seed, the reference line every arm is compared with. **The reduction (Stage 3A)** runs when
the config has a `reduction` section. It cuts each arm to a model of a few features, using only
that arm's cells. **The reliability panel (Stage 3B)** runs when the config has a `reliability`
section: all of its work that comes before the reveal, with Tier 1's selections spread over the
threads it is given, and those lent to it as they free up, in as many workers as fit in the memory
it is given.

**Then the reveal, and what reads it.** The test split is opened once (`reveal.py`), the panel's
Tier 3 and tables are computed from what the reveal scored, and the report's inputs are built: its
facts and tables under Core, its figures under `SeqMaestro_LocalUtils`. **The reveal runs only when
every stage before it finished clean.** It opens the test split for good, so a failure before it
holds the reveal back and is reported by name. Running again with `--run-dir` continues the run.
`--stop-before-reveal` stops ahead of the reveal so the run can be checked first. Continued without
the flag, such a run starts every stage again: the cells, the reductions and the panels skip what
they have finished, stability and co-discovery are computed again and give the same tables, and the
two analyses that ask a language model keep what is on file while what they were computed from is
unchanged (`seqmaestro/pipeline/analyses/inputs.py`). **A revealed run resumes from the reveal**:
the stages before it fit or select, so they are not run again.

**Arms are offered the machine in the order the config lists them.** Of two processes that could
both start, the one listed first starts first, so a config lists its cheap arms first, and those it
wants finished early, and an early abort still leaves a result.

**`--estimate` says what each arm would cost before anything runs**: a few fits of each arm's model
on the run's data, scaled to every stage (`seqmaestro/pipeline/estimate.py`). It flags the arms
over `--slow-hours`, so whoever launches the run can decide about them first, and starts no stage.

**Nothing before the reveal reads the test split.** Every stage before it fits and scores on
validation (`seqmaestro.harness.heldout`).

**Stages run with the interpreter that runs the driver**, so no second place records which
environment the pipeline belongs to. Run it with `SeqMaestro_Core` (ENVIRONMENTS.md). The
exceptions are DEFT's and IGSR's sides, PySR's search and the report's figures, which must run
elsewhere and find their interpreter with `seqmaestro.provenance.environment_python`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from seqmaestro import arms as registry
from seqmaestro import files
from seqmaestro.arms import Arm, arm_name
from seqmaestro.execution import budget as draw_budget
from seqmaestro.execution import probe
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import reveal_record
from seqmaestro.llm import TaskContext, missing_settings
from seqmaestro.pipeline import baseline, reduction, reliability, report, reveal, stage2
from seqmaestro.pipeline.analyses import co_discovery, judge, recurrence, stability
from seqmaestro.pipeline.config import CONFIG_VERSION, ConfigError, ExperimentConfig, load
from seqmaestro.pipeline.layout import (
    CONFIG_FILE,
    FAILURES_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    call_name,
    report_dir,
)
from seqmaestro.pipeline.repository import ANALYSIS, DESCRIPTIONS, REPO, RUNNERS, RUNS
from seqmaestro.pipeline.schedule import AFTER, BEFORE, REVEAL, Command, Scheduler, Stage
from seqmaestro.provenance import environment_python, missing_python, record_call
from seqmaestro.report.files import FIGURES_CALL

#: Where a report's figures are drawn: the environment with the plotting stack (ENVIRONMENTS.md).
LOCALUTILS_ENVIRONMENT = "SeqMaestro_LocalUtils"
#: The name of the driver's call (`layout.call_name`).
CALL = "driver"



def _short(path: Path) -> str:
    """The path relative to the repository, where possible. Stages run with `cwd=REPO`, so this
    form is both what they are given and what the logs show. An absolute run directory would make
    every recorded command line hard to read and no more reproducible."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


#: How many processes may call a language model at once, unless `--llm-slots` says otherwise. The
#: endpoint's throughput bounds it, which the run cannot probe without spending calls, so it is
#: stated here and recorded with each invocation.
LLM_SLOTS = 20


def _serial(name: str, argv: tuple[str, ...], retry: bool = True, phase: str = BEFORE,
            threads: int | None = 1, after: tuple[str, ...] = (), calls_model: bool = False) -> Stage:
    """A stage that is one process, logging under its own name."""
    return Stage(name, (Command(argv, name, after=after),), retry=retry, phase=phase,
                 threads=threads, calls_model=calls_model)


# ------------------------------------------------------------------------------ building stages

def _runner(script: str) -> tuple[str, ...]:
    """A stage's script under `experiments/runners/`, started with this interpreter."""
    return (sys.executable, _short(RUNNERS / script))


def _analysis(script: str, python: Path | str = sys.executable) -> tuple[str, ...]:
    return (str(python), _short(ANALYSIS / script))


def _arm(arm: Arm, config: ExperimentConfig, run_dir: Path) -> Stage:
    """One arm as one process per (labels, seed). The arm's cells for one condition of the labels
    and one seed do not depend on its others, so the processes run side by side. The stage is named
    `model/pool`, as the arm is spelled."""
    model, pool = arm
    name = arm_name(model, pool)
    return Stage(name, tuple(
        Command((*_runner("run_arm.py"), "--run-dir", _short(run_dir), "--arms", name,
                 "--labels", labels, "--seeds", str(seed)),
                stage2.cell_call(model, pool, labels, seed),
                produces=(arm,) if labels == cells.REAL else ())
        for labels in config.grid.labels for seed in config.grid.seeds
    ), retry=registry.retries(pool), calls_model=registry.calls_language_model(pool))


def _baseline(run_dir: Path) -> Stage:
    """The composition baseline, at every seed of the grid, before any arm."""
    return _serial(baseline.CALL, (*_runner("run_baseline.py"), "--run-dir", _short(run_dir)))


def _analyses(config: ExperimentConfig, run_dir: Path, after: tuple[str, ...]) -> tuple[Stage, ...]:
    """The analyses, from the narrowest question to the widest, once every arm has finished.
    Stability comes first because it compares one arm only with itself. What each one answers, and
    how the three that cluster features differ, is in experiments/analysis/README.md. The judge runs
    when the grid says so (`grid.judge`).
    """
    run = ("--run-dir", _short(run_dir))
    judged = (_serial(judge.CALL, (*_analysis("run_judge.py"), *run), after=after,
                      calls_model=True),)
    writes = any(registry.calls_language_model(pool) for _, pool in config.arm_cells())
    return (
        _serial(stability.CALL, (*_analysis("analyse_stability.py"), *run), after=after),
        _serial(co_discovery.CALL, (*_analysis("analyse_co_discovery.py"), *run), after=after),
        # Whether an arm proposes the same *ideas* across seeds, which set-based stability cannot
        # see, because a generative arm names one idea differently each time and moves the window
        # it is evaluated over.
        _serial(recurrence.CALL, (*_analysis("analyse_recurrence.py"), *run), after=after,
                calls_model=writes),
        *(judged if config.grid.judge else ()),
    )


def _reduction(config: ExperimentConfig, run_dir: Path, arms: tuple[Stage, ...]
               ) -> tuple[Stage, ...]:
    """Stage 3A, when the config has a `reduction` section and reduces an arm: one process per
    reduced arm. An arm is reduced from its best real-label cell, so its process starts once the
    processes writing its real-label cells have finished, whatever the rest of the run is doing.
    Each takes up to four threads, which PySR's searches use."""
    if config.reduction is None or not config.reduced_arms():
        return ()
    writers: dict[Arm, list[str]] = {}
    for stage in arms:
        for command in stage.commands:
            for arm in command.produces:
                writers.setdefault(arm, []).append(command.log)
    return (Stage(reduction.STAGE, tuple(
        Command((*_runner("run_reduction.py"), "--run-dir", _short(run_dir), "--arms",
                 arm_name(*arm)),
                call_name(reduction.STAGE, [arm]), after=tuple(writers.get(arm, ())))
        for arm in config.reduced_arms()), threads=4),)


def _reliability(config: ExperimentConfig, run_dir: Path, reduced: tuple[Stage, ...]
                 ) -> tuple[Stage, ...]:
    """Stage 3B's work before the reveal, when the config has a `reliability` section: one process
    per profiled arm, started once that arm's reduction has finished. Each takes every free thread
    for Tier 1's selections, and is lent those that free up while it runs, so the panels run one at
    a time. Its work after the reveal is in `_after_reveal`."""
    if config.reliability is None:
        return ()
    reductions = set(_logs(reduced))
    return (Stage(reliability.STAGE, tuple(
        Command((*_runner("run_reliability.py"), "--run-dir", _short(run_dir), "--arms",
                 arm_name(*arm)),
                call_name(reliability.STAGE, [arm]),
                after=tuple({call_name(reduction.STAGE, [arm])} & reductions))
        for arm in config.profiled_arms()), threads=None),)


def _reveal(run_dir: Path) -> Stage:
    """The reveal: the test split opened once, every saved model scored on it, and the panel's
    test-side work. It refuses a run that is not complete, so the check that every stage before it
    finished lives in the reveal."""
    return _serial(reveal.CALL, (*_runner("run_reveal.py"), "--run-dir", _short(run_dir)),
                   phase=REVEAL, threads=None)


def _after_reveal(config: ExperimentConfig, run_dir: Path) -> tuple[Stage, ...]:
    """What reads the reveal's numbers: the panel's Tier 3 and tables, then the report's inputs
    under Core, then its figures under LocalUtils, drawn only from what the Core stage wrote. The
    report itself is written by hand or by an agent (process/reporting.md), so the driver stops at
    its inputs."""
    stages = []
    if config.reliability is not None:
        stages.append(_serial(reliability.STAGE_AFTER_REVEAL, (
            *_runner("run_reliability.py"), "--run-dir", _short(run_dir), "--after-reveal"),
            phase=AFTER, threads=None))
    stages.append(_serial(report.CALL, (
        *_analysis("build_report_inputs.py"), "--run-dir", _short(run_dir)), phase=AFTER))
    stages.append(_serial(FIGURES_CALL, (
        *_analysis("build_report_figures.py", environment_python(LOCALUTILS_ENVIRONMENT)),
        "--report-dir", _short(report_dir(run_dir))), phase=AFTER))
    return tuple(stages)


def build_stages(config: ExperimentConfig, run_dir: Path) -> tuple[Stage, ...]:
    """Everything the run will do, in order, as the exact processes it will start. A config under
    which an arm could not be started is refused here (`stage2.check`)."""
    stage2.check(config)
    arms = tuple(_arm(arm, config, run_dir) for arm in config.arm_cells())
    reduced = _reduction(config, run_dir, arms)
    return (_baseline(run_dir), *arms, *_analyses(config, run_dir, after=_logs(arms)),
            *reduced, *_reliability(config, run_dir, reduced), _reveal(run_dir),
            *_after_reveal(config, run_dir))


def _logs(stages: tuple[Stage, ...]) -> tuple[str, ...]:
    """The logs of every command of `stages`: what a command waiting for all of them names."""
    return tuple(command.log for stage in stages for command in stage.commands)


# ------------------------------------------------------------------------------------- preflight

def preflight(config: ExperimentConfig, stages: tuple[Stage, ...], run_dir: Path) -> list[str]:
    """Everything that can be checked before anything is spent.

    A run that dies three hours in for want of a credential has cost the whole night, and each of
    these is a one-line check. The checks follow the stages selected, not the whole config, so
    `--only l1_logistic/positional` needs no credentials and no vendored interpreter even when the
    config also names arms that do. The run's blinding is the exception, checked whatever is selected
    (`blinding_problems`), since it is a property of the whole run.

    The split files are not checked here. Loading the config already read the dataset's
    `dataset.json` and checked the training and validation files against it, and the reveal checks
    the test file.
    """
    selected = {stage.name for stage in stages}
    problems = []
    if reveal_record(run_dir) is not None and any(stage.phase == BEFORE for stage in stages):
        problems.append(
            f"{_short(run_dir)} has been revealed, and a stage selected fits or selects. To add to "
            "the run, start a new one.")
    description = DESCRIPTIONS / f"{config.dataset.task}.json"
    if not description.exists():
        problems.append(f"missing dataset description: {description}")
    else:
        try:
            problems += stage2.blinding_problems(config, TaskContext.from_json(description))
        except ValueError as error:
            problems.append(str(error))
    for stage in stages:
        script = REPO / stage.commands[0].argv[1]
        if not script.exists():
            problems.append(f"missing script for stage {stage.name}: {script}")
    # A model fitted in another environment needs it wherever the model is fitted: its arm's stage,
    # the reduction and the panel. A vendored method's is needed after those too, by every stage
    # that uses its models: the panel on both sides of the reveal, the reveal, and the report's
    # inputs, which read a model to draw it. PySR's equation is evaluated in Core once found.
    uses_models = {reduction.STAGE, reliability.STAGE, reveal.CALL, reliability.STAGE_AFTER_REVEAL,
                   report.CALL}
    fitting: dict[str, list[str]] = {arm_name(*arm): [arm[0]] for arm in config.arm_cells()}
    fitting[reduction.STAGE] = [model for model, _ in config.reduced_arms()]
    fitting[reliability.STAGE] = [model for model, _ in config.profiled_arms()]
    for name in selected:
        for model in fitting.get(name, ()):
            problems.append(missing_python(registry.spec(model).environment)
                            if registry.spec(model).environment else None)
    if uses_models & selected:
        for model in sorted({model for model, _ in config.arm_cells()}):
            if registry.spec(model).vendored:
                problems.append(missing_python(registry.spec(model).environment))
    if FIGURES_CALL in selected:
        problems.append(missing_python(LOCALUTILS_ENVIRONMENT))
    problems = list(dict.fromkeys(problem for problem in problems if problem))
    # A stage calls a language model when it is an arm whose pool holds a written source
    # (`seqmaestro.arms.calls_language_model`), the judge, or the recurrence analysis of a run with
    # such an arm. A run with no written source and no judge needs no provider and no credentials.
    calling = sorted(stage.name for stage in stages if stage.calls_model)
    unset = missing_settings() if calling else []
    if unset:
        problems.append(
            f"the language model is not configured: missing {', '.join(unset)}, needed by "
            f"{', '.join(calling)}. Run 'set -a && source .env && set +a' first.")
    return problems


# ------------------------------------------------------------------------------------- the run


class Journal:
    """The driver's narration: to the terminal, and to one log that outlives the terminal."""

    def __init__(self, logs: Path) -> None:
        self.logs = logs
        self.path = logs / f"{CALL}.log"

    def _write(self, line: str) -> None:
        print(line, flush=True)
        with self.path.open("a") as handle:
            handle.write(line + "\n")

    def banner(self, text: str) -> None:
        self._write(f"\n=== {time.strftime('%H:%M:%S')}  {text} ===")

    def note(self, text: str) -> None:
        self._write(text)


def notify(title: str, message: str, priority: str = "default", journal: Journal | None = None) -> None:
    """Send a push notification, if a topic is configured.

    The topic is read from `NTFY_TOPIC` in the environment and never from the config. It is how an
    agentic session watches a run it started, so it is no part of the experiment, and two runs of
    one config should not differ in their recorded settings because one was being watched. It is
    also a public channel to anyone who knows its name, so it must stay out of tracked files. When
    it is unset there are no notifications and no complaint.
    """
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    command = ("curl", "-fsS", "-m", "20", "-H", f"Title: {title}", "-H", f"Priority: {priority}",
               "-d", message, f"https://ntfy.sh/{topic}")
    try:
        subprocess.run(command, capture_output=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        if journal:
            journal.note("  (notification failed; the run is unaffected)")


def start_process(argv: tuple[str, ...], log: Path, environment: dict[str, str], cwd: Path):
    """Start one of the run's processes (`schedule.start_process`). A module attribute, so that a
    test can stand in for every process the driver starts."""
    from seqmaestro.pipeline.schedule import start_process as start
    return start(argv, log, environment, cwd)


def run_budget(args: argparse.Namespace):
    """The machine as probed now, and the budget the run takes from it under the flags given."""
    machine = probe()
    return machine, draw_budget(
        machine, model_calls=args.llm_slots, max_cpus=args.max_cpus,
        max_memory=None if args.max_memory is None else int(args.max_memory * 2**30))


def record_budget(logs: Path, machine, budget, args: argparse.Namespace) -> None:
    """Append this invocation's probe and budget to `logs/machine.jsonl`. A resumed run is a new
    invocation, possibly on another machine, so every invocation keeps a line."""
    line = {"started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": machine.as_dict(), "budget": budget.as_dict(),
            "flags": {"max_cpus": args.max_cpus, "max_memory_gb": args.max_memory,
                      "llm_slots": args.llm_slots}}
    with (logs / "machine.jsonl").open("a") as handle:
        handle.write(json.dumps(line) + "\n")


def resolve_run_dir(given: Path | None, config: ExperimentConfig, new_run: bool) -> Path:
    """Where this run's results go, refusing the mistake the dated default invites.

    Without the check, a resume on a later day than the launch would start an empty run: every
    stage "resumes" into an empty directory, redoes hours of work, and the results end up split
    across two directories. An interrupted run is the case resuming exists for, and also the case
    that meets this problem, so the driver refuses instead of relying on documentation.
    """
    if given is not None:
        return given
    default = RUNS / f"{date.today():%Y-%m-%d}_{config.name}_full"
    if default.exists():
        return default
    earlier = runs_of(config.name)
    if earlier and not new_run:
        listing = "\n".join(f"    {_short(path)}" for path in earlier)
        raise ConfigError(
            f"there are already runs for {config.name!r}:\n{listing}\n"
            f"  Starting {_short(default)} would redo their work and split the results "
            "across directories.\n"
            f"  To continue one:            --run-dir {_short(earlier[-1])}\n"
            "  To start a genuinely new one: --new-run")
    return default


def runs_of(name: str) -> list[Path]:
    """The run directories whose recorded config has this name, oldest first.

    A run is matched on the name its `config.yaml` records, whatever its directory is called. A
    directory named by hand is still a run of its config, and a directory whose name holds another
    config's name (`screen_v2` holds `screen`) is no run of that other config.
    """
    found = []
    for path in sorted(RUNS.glob(f"*/{CONFIG_FILE}")):
        try:
            recorded = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(recorded, dict) and recorded.get("name") == name:
            found.append(path.parent)
    return found


def bind_config(run_dir: Path, config: ExperimentConfig, allow_change: bool, journal: Journal) -> None:
    """Record the resolved config, or refuse to resume onto a different one.

    A stage skips the cells that already have results, and cannot tell a finished cell from one
    finished under different settings. Two configurations mixed into one table cannot be detected
    afterwards, so the difference is caught here. A deliberate change is allowed through with
    `--allow-config-change`, and the superseded config is kept beside the new one, because it is
    the only record of what the cells already in the directory were computed under.

    A run whose config was recorded under another `version` of the config's shape is refused
    whatever the flag, since the stages could not take that run's settings from it.
    """
    path = run_dir / CONFIG_FILE
    resolved = config.to_yaml()
    if path.exists():
        recorded = yaml.safe_load(path.read_text())
        version = recorded.get("version") if isinstance(recorded, dict) else None
        if version != CONFIG_VERSION:
            raise ConfigError(
                f"{_short(path)} was recorded as config version {version!r}, and this code "
                f"records version {CONFIG_VERSION}. The stages cannot take that run's settings "
                f"from it, so the run cannot be continued. Start a fresh run directory.")
    if path.exists() and path.read_text() != resolved:
        difference = "".join(difflib.unified_diff(
            path.read_text().splitlines(keepends=True), resolved.splitlines(keepends=True),
            fromfile=f"recorded ({_short(path)})", tofile=f"now ({config.source})"))
        if not allow_change:
            raise ConfigError(
                f"this run was made under a different config:\n{difference}\n"
                "  Cells already finished were computed under the recorded one, and a resumed "
                "stage cannot tell them apart.\n"
                "  Start a fresh run directory, or pass --allow-config-change if the difference "
                "does not affect them.")
        superseded = run_dir / f"config_superseded_{sum(1 for _ in run_dir.glob('config_superseded_*.yaml')) + 1}.yaml"
        path.rename(superseded)
        journal.note(f"config changed; the previous one is kept as {superseded.name}\n{difference}")
    files.write_text(path, resolved)


def _selected_by(stage: Stage) -> set[str]:
    """The names `--only` and `--skip` select a stage by: its name, and an arm's model and pool."""
    model, found, pool = stage.name.partition("/")
    return {stage.name, model, pool} if found else {stage.name}


def select(stages: tuple[Stage, ...], only: list[str], skip: list[str]) -> tuple[Stage, ...]:
    """Narrow the pipeline by stage name, or by a model or a pool, which selects every arm with it:
    `--only positional` runs each arm on the positional pool. A name that selects nothing is
    refused, since a misspelled `--only` would otherwise run nothing and report a clean finish."""
    known = set().union(*map(_selected_by, stages))
    unknown = sorted((set(only) | set(skip)) - known)
    if unknown:
        raise ConfigError(f"no stage, model or pool named {', '.join(unknown)}. This run has: "
                          f"{', '.join(stage.name for stage in stages)}.")
    chosen = tuple(stage for stage in stages
                   if (not only or _selected_by(stage) & set(only))
                   and not _selected_by(stage) & set(skip))
    if not chosen:
        raise ConfigError("--only and --skip together leave no stage to run.")
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="the run, in full: see experiments/configs/")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="where results go. Defaults to a dated directory; pass it explicitly "
                             "to continue an interrupted run")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="override one config value, repeatable. Recorded with the run, so a "
                             "smoke run is not an undocumented one. `null` removes the key")
    parser.add_argument("--only", nargs="+", default=[], metavar="STAGE",
                        help="run only these stages; a model or a pool names every arm with it")
    parser.add_argument("--skip", nargs="+", default=[], metavar="STAGE",
                        help="run every stage but these; a model or a pool names every arm with it")
    parser.add_argument("--new-run", action="store_true",
                        help="start a new dated run even though earlier ones exist for this config")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="resume onto a run recorded under a different config. The superseded "
                             "config is kept beside the new one")
    parser.add_argument("--stop-before-reveal", action="store_true",
                        help="run everything that comes before the reveal, and stop there. Running "
                             "again without it starts each of those stages again, which skips what "
                             "it has finished, and goes on through the reveal")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config, the budget and every command, run nothing")
    parser.add_argument("--estimate", action="store_true",
                        help="estimate each arm's time on this machine from a few fits on the "
                             "run's data, flag the slow ones, and run nothing; with --run-dir the "
                             "estimate is kept in the run's logs/")
    parser.add_argument("--slow-hours", type=_positive_float, default=None, metavar="HOURS",
                        help="with --estimate, the wall time over which an arm is flagged (default: "
                             "2)")
    # How much of the machine the run may take. None changes a result, and each is recorded in
    # `logs/machine.jsonl`, so a run resumed on another machine may use other values.
    parser.add_argument("--max-cpus", type=_positive_int, default=None, metavar="N",
                        help="at most this many threads at once (default: every CPU this process "
                             "may use)")
    parser.add_argument("--max-memory", type=_positive_float, default=None, metavar="GB",
                        help="at most this many gigabytes held at once (default: "
                             "a share of the memory available as the run starts)")
    parser.add_argument("--llm-slots", type=_positive_int, default=LLM_SLOTS, metavar="N",
                        help=f"at most this many processes calling a language model at once "
                             f"(default {LLM_SLOTS})")
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, not {value}")
    return number


def plan(stages: tuple[Stage, ...], revealed: bool, stop_before_reveal: bool
         ) -> tuple[tuple[Stage, ...], list[str]]:
    """The stages this call runs, and why any selected stage was left out.

    A revealed run resumes from the reveal. Everything before the reveal fits or selects, and has
    already run, because the reveal refuses to start until it has. `--stop-before-reveal` stops
    ahead of the reveal, so a run's results can be checked before the test split is opened.
    """
    notes = []
    if revealed:
        dropped = [stage.name for stage in stages if stage.phase == BEFORE]
        stages = tuple(stage for stage in stages if stage.phase != BEFORE)
        if dropped:
            notes.append(f"revealed already, so resuming from the reveal; not run: {', '.join(dropped)}")
    if stop_before_reveal:
        dropped = [stage.name for stage in stages if stage.phase != BEFORE]
        stages = tuple(stage for stage in stages if stage.phase == BEFORE)
        if dropped:
            notes.append(f"--stop-before-reveal; not run: {', '.join(dropped)}")
    if not stages:
        raise ConfigError("nothing is left to run: " + "; ".join(notes))
    return stages, notes


def run_estimate(args: argparse.Namespace, config: ExperimentConfig) -> int:
    """`--estimate`: each arm's time on this machine, printed, and kept in the run's `logs/` when a
    run directory is named. It starts no stage."""
    from seqmaestro.pipeline import estimate

    _, budget = run_budget(args)
    slow_hours = estimate.SLOW_HOURS if args.slow_hours is None else args.slow_hours
    for warning in config.warnings:
        print(f"WARNING: {warning}")
    print(f"Probing {len(config.arm_cells())} arm(s) on this machine, one thread each.", flush=True)
    try:
        probes, estimates = estimate.estimate_run(config, budget.cpus, slow_hours,
                                                  report=lambda line: print(line, flush=True))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print("\n" + estimate.table(estimates, config, budget.cpus, slow_hours), flush=True)
    if args.run_dir is not None:
        print(f"\nkept in {_short(estimate.record(args.run_dir, probes, estimates, budget.cpus, slow_hours))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.slow_hours is not None and not args.estimate:
        print("--slow-hours applies only with --estimate.", file=sys.stderr)
        return 2

    try:
        config = load(args.config, tuple(args.overrides))
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    if args.estimate:
        estimated = run_estimate(args, config)
        if estimated or not args.dry_run:
            return estimated

    try:
        run_dir = resolve_run_dir(args.run_dir, config, args.new_run)
        stages, planned = plan(select(build_stages(config, run_dir), args.only, args.skip),
                               reveal_record(run_dir) is not None, args.stop_before_reveal)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2

    machine, budget = run_budget(args)
    if args.dry_run:
        print(config.to_yaml(), end="")
        print(f"\n# run-dir: {_short(run_dir)}")
        print(f"# machine: {machine.as_dict()}")
        print(f"# budget: {budget.as_dict()}\n")
        for stage in stages:
            for command in stage.commands:
                print(f"[{stage.name}] {shlex.join(command.argv)}\n")

    sys.stdout.flush()
    problems = preflight(config, stages, run_dir)
    if problems:
        for problem in problems:
            print(f"PREFLIGHT: {problem}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("# preflight clean")
        return 0

    (run_dir / LOGS_DIR).mkdir(parents=True, exist_ok=True)
    journal = Journal(run_dir / LOGS_DIR)
    try:
        bind_config(run_dir, config, args.allow_config_change, journal)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    record_call(run_dir, CALL, vars(args), config=config.to_mapping())

    record_budget(journal.logs, machine, budget, args)

    started = time.monotonic()
    journal.banner(f"{config.name} -> {_short(run_dir)}")
    journal.note(
        f"splits={config.dataset.splits}  columns={config.dataset.sequence_column}/"
        f"{config.dataset.target_column}  seeds={list(config.grid.seeds)}  "
        f"labels={list(config.grid.labels)}  bootstrap={config.grid.n_bootstrap}  "
        f"permutation={config.grid.permutation_scope}")
    journal.note(f"stages: {', '.join(stage.name for stage in stages)}")
    for note in planned:
        journal.note(note)
    for warning in config.warnings:
        journal.note(f"WARNING: {warning}")
    journal.note(f"machine: {machine.cpus} CPUs ({machine.cores} cores), "
                 f"{machine.memory_available / 2**30:.0f} of {machine.memory_total / 2**30:.0f} GB "
                 f"available, load {machine.load}; budget: {budget.cpus} threads, "
                 f"{budget.memory / 2**30:.0f} GB, {budget.model_calls} processes calling a "
                 f"language model")

    def failed_stage(stage: Stage) -> None:
        notify(f"SeqMaestro: {config.name} stage failed",
               f"Stage '{stage.name}' failed. The run continues without it. "
               f"Logs: {_short(journal.logs)}/{stage.name}*.log", "high", journal)

    # The reveal opens the test split for good, so the scheduler holds it, and what reads it, once
    # anything before it has failed. The reveal would refuse an incomplete run anyway, but holding
    # it here reports the failed stage by name.
    outcome = Scheduler(stages, budget, journal, cwd=REPO, start=start_process,
                        on_failure=failed_stage).run()
    failed, held = outcome.failed, outcome.held

    journal.banner("finished")
    elapsed = round((time.monotonic() - started) / 60)
    finished = sum(1 for _ in (run_dir / RESULTS_DIR).rglob(cells.METRICS_FILE))
    if failed:
        journal.note(f"STAGES THAT FAILED: {' '.join(failed)}")
        journal.note(f"What failed in each call, with its traceback: {_short(run_dir / FAILURES_DIR)}")
        notify(f"SeqMaestro: {config.name} finished with gaps",
               f"{finished} cells in {elapsed} min. Failed stages: {' '.join(failed)}", "high",
               journal)
    else:
        journal.note("every stage exited clean")
        notify(f"SeqMaestro: {config.name} finished",
               f"{finished} cells in {elapsed} min, every stage clean.", journal=journal)
    journal.note(f"results: {_short(run_dir)}/{RESULTS_DIR}    logs: {_short(journal.logs)}")
    if held:
        journal.note(f"NOT RUN, because a stage before the reveal failed: {' '.join(held)}. Fix "
                     f"it and run again with --run-dir {_short(run_dir)}; finished work is kept.")
    if reveal_record(run_dir) is None:
        journal.note(f"validation only so far; the test split has not been opened. To open it and "
                     f"continue: --run-dir {_short(run_dir)} without --stop-before-reveal.")
    elif not failed and not held:
        journal.note(f"report inputs: {_short(report_dir(run_dir))}; the report is written against "
                     f"docs/report/TEMPLATE.md, as process/reporting.md says")
    return 1 if failed or held else 0

