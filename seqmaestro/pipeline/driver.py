"""Run a dataset through the whole pipeline unattended: every arm and analysis, the reduction and
the reliability panel, the reveal, and the report's inputs.

    experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml

**`--detach` starts the run in a session of its own** and returns once it is under way, so that the
run outlives the terminal, and the agent's command, that started it:

    $CORE experiments/runners/run_experiment.py --config <config> --run-dir <run> --detach

A harness that runs an agent's commands may end every process a command started as soon as the
command returns, which a job started with `nohup` alone does not survive, whereas a process in a
session of its own does. The checks before the run (the config, the run's directory, the preflight)
are made before the call returns, so a refusal is printed where the launch was typed.

**The config states the run**: which splits, which grid and which arms (`experiments/configs/`).
The driver records the resolved config as `<run>/config.yaml`, and every stage reads its settings
from that file, so the driver passes a stage only the run directory and the part of the run it is
to work on. This module holds only what is true of the pipeline on any dataset: the stages, their
order, and which of them may run at once.

**Every stage is resumable, and the driver is idempotent.** When it is run again after a crash, a
reboot or a killed process, the driver skips the cells already done and continues. A stage cannot
tell a finished cell from one computed under different settings, so the driver refuses to continue
a run whose recorded config differs from the one it is given, because that would mix two
configurations in one table without any sign of it. The driver also refuses to continue a run
whose recorded config states another `version` of the config's shape, because the stages could not
take that run's settings from it.

**To resume, pass the run directory.** The default directory is dated, so a resume on a later day
than the launch would start an empty run, redo hours of work, and split the results across two
directories, which goes unnoticed until someone reads both. A resume is usually unattended, so a
warning is not enough: the driver refuses to start a new run when earlier ones for the same config
exist, unless `--new-run` says that is the intent.

**A failing stage does not stop the stages after it, except the reveal** (below). Each stage
isolates its cells. Failures are counted and repeated at the end, `logs/` holds what each call
printed, and `failures/` what failed in it.

**A run starts its processes side by side** (`seqmaestro/pipeline/schedule.py`). An arm is one
model fitted on one pool of feature sources, as the config's `arms` lists them (`seqmaestro.arms`
says which model takes which pool), and each arm runs as one process per label condition and seed.
The reduction and the reliability panel run one process per arm. A process starts once what it
waits for has finished and the machine has room for it. The arms wait for nothing, whereas an arm's
reduction waits for its real-label cells and its panel for its reduction, so a cheap arm is reduced
and profiled while slow arms still run. The analyses wait for every arm, and the reveal for
everything before it.

**Arms are offered the machine in the order the config lists them.** Of two processes that could
both start, the one listed first starts first, so a config lists first its cheap arms and those it
wants finished early, and an early abort still leaves them finished and ready to analyse.

**The machine is probed as the run starts**: the CPUs this process may use and the memory available
to it (`seqmaestro.execution.probe`). The run's budget follows from the probe: the threads, the
memory, and how many processes may call a language model at once, which the endpoint's throughput
bounds and which defaults to `LLM_SLOTS`. `--max-cpus` and `--max-memory` lower the first two for a
machine that is also in use for other work, and `--llm-slots` sets the third. These flags do not
change any result, so they belong to the driver instead of the config, and a run resumed on another
machine may use other values. Each invocation appends the probe and the budget to
`logs/machine.jsonl`, and each process leaves a row in `logs/processes.csv` with its threads, its
time and the memory it held.

**The composition baseline starts first**: a standardised sparse linear model on the base fractions
at every seed, which gives the reference line every arm is compared with. **Where a pool holds
known motifs**, a step before the arms on that pool scans every training and validation sequence
once with the run's collection. Where a pool holds discovered motifs, the same step makes each
discovery that can be made in advance, one per label condition and seed
(`seqmaestro/pipeline/motifs.py`). **The reduction (Stage 3A)** runs when the config has a
`reduction` section. It reduces each arm to a model of a few features, using only that arm's cells.
**The reliability panel (Stage 3B)** does all of its work that comes before the reveal when the
config has a `reliability` section. Tier 1's selections are spread over the threads the panel is
given, and over those lent to it as they free up, in as many workers as fit in its memory.

**Then the reveal and the stages that use its scores.** The held-out split is opened once
(`reveal.py`), the panel's Tier 3 and tables are computed from what the reveal scored, and the
report's inputs are built: its facts and tables under Core, and its figures under
`SeqMaestro_LocalUtils`. **The reveal runs only when every stage before it finished without a
failure.** Because the reveal opens the held-out split for good, a failure before it holds the
reveal back and is reported by name. Running again with `--run-dir` continues the run.
`--stop-before-reveal` stops ahead of the reveal so that the run can be checked first. When such a
run is continued without the flag, every stage starts again: the cells, the reductions and the
panels skip what they have finished, stability and co-discovery are computed again and give the
same tables, and the two analyses that call a language model keep their results on file as long as
their inputs are unchanged (`seqmaestro/pipeline/analyses/inputs.py`). **A revealed run resumes
from the reveal**: the stages before it fit or select, so they are not run again.

**`--estimate` says how long each arm would take before anything runs**: a few fits of each arm's
model on the run's data, scaled to every stage (`seqmaestro/pipeline/estimate.py`). It starts no
stage, and it flags the arms that would take longer than `--slow-hours`, so that whoever launches
the run can decide about them first.

**The held-out split is not read before the reveal.** Every stage before it fits and scores on the
validation split (`seqmaestro.harness.heldout`).

**Stages run with the interpreter that runs the driver**, so the pipeline's environment is not
stated in a second place. Run the driver with `SeqMaestro_Core` (ENVIRONMENTS.md). The exceptions
are DEFT's and IGSR's sides, PySR's search and the report's figures, which must run elsewhere and
find their interpreter with `seqmaestro.provenance.environment_python`.
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
from seqmaestro import files, status
from seqmaestro.arms import Arm, arm_name
from seqmaestro.execution import budget as draw_budget
from seqmaestro.execution import probe
from seqmaestro.features import MOTIF_DISCOVERED, MOTIF_KNOWN, parse_pool
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import reveal_record
from seqmaestro.llm import TaskContext, describe_calls, missing_settings, provider_from_environment
from seqmaestro.pipeline import baseline, reduction, reliability, report, reveal, stage2
from seqmaestro.pipeline import motifs as run_motifs
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
    """The path relative to the repository, where possible. Stages run with `cwd=REPO`, so they
    are given this form and the logs show it. An absolute run directory would make every recorded
    command line hard to read without making it any more reproducible."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


#: How long `--detach` watches the run it started before it returns: long enough for a run that
#: stops at once, for instance on a config that differs from the one the run recorded, to say so.
DETACH_WATCH_SECONDS = 15
#: Where a detached run's printed output goes, beside the journal in `logs/driver.log`. It holds
#: what the run printed before its journal began, such as a refusal.
DETACHED_OUTPUT = f"{CALL}.out"


#: How many processes may call a language model at once, unless `--llm-slots` says otherwise. The
#: limit is set by the endpoint's throughput, which the run cannot probe without spending calls, so
#: it is stated here and recorded with each invocation.
LLM_SLOTS = 20


def _serial(name: str, argv: tuple[str, ...], retry: bool = True, phase: str = BEFORE,
            threads: int | None = 1, after: tuple[str, ...] = (), calls_model: bool = False) -> Stage:
    """A stage of one process, which logs under the stage's name."""
    return Stage(name, (Command(argv, name, after=after),), retry=retry, phase=phase,
                 threads=threads, calls_model=calls_model)


# ------------------------------------------------------------------------------ building stages

def _runner(script: str) -> tuple[str, ...]:
    """A stage's script under `experiments/runners/`, started with this interpreter."""
    return (sys.executable, _short(RUNNERS / script))


def _analysis(script: str, python: Path | str = sys.executable) -> tuple[str, ...]:
    return (str(python), _short(ANALYSIS / script))


def _arm(arm: Arm, config: ExperimentConfig, run_dir: Path) -> Stage:
    """One arm as one process per label condition and seed. An arm's cell at one label condition
    and seed does not depend on its other cells, so the processes run side by side. The stage is
    named `model/pool`, as the arm is written."""
    model, pool = arm
    name = arm_name(model, pool)
    return Stage(name, tuple(
        Command((*_runner("run_arm.py"), "--run-dir", _short(run_dir), "--arms", name,
                 "--labels", labels, "--seeds", str(seed)),
                stage2.cell_call(model, pool, labels, seed),
                after=run_motifs.waits_for(pool, labels, seed, config.grid.permutation_scope),
                produces=(arm,) if labels == cells.REAL else ())
        for labels in config.grid.labels for seed in config.grid.seeds
    ), retry=registry.retries(pool), calls_model=registry.calls_language_model(pool))


def _baseline(run_dir: Path) -> Stage:
    """The composition baseline, at every seed of the grid, before any arm."""
    return _serial(baseline.CALL, (*_runner("run_baseline.py"), "--run-dir", _short(run_dir)))


def _motifs(config: ExperimentConfig, run_dir: Path) -> tuple[Stage, ...]:
    """What the motif sources make once, before the arms on their pools, which wait for it
    (`seqmaestro/pipeline/motifs.py`): the scan with the known motifs, and one discovery for each
    label condition and seed whose discovery is made in advance. Empty where no pool holds a motif
    source."""
    run = (*_runner("run_motifs.py"), "--run-dir", _short(run_dir))
    commands = []
    if run_motifs.holds_known(config):
        commands.append(Command(run, run_motifs.KNOWN_CALL))
    if run_motifs.holds(config, MOTIF_DISCOVERED):
        commands += [Command((*run, "--discover", "--labels", labels, "--seed", str(seed)),
                             run_motifs.discovered_call(labels, seed))
                     for labels in config.grid.labels for seed in config.grid.seeds
                     if run_motifs.made_in_advance(labels, config.grid.permutation_scope)]
    return (Stage(run_motifs.STAGE, tuple(commands)),) if commands else ()


def _analyses(config: ExperimentConfig, run_dir: Path, after: tuple[str, ...]) -> tuple[Stage, ...]:
    """The analyses, run once every arm has finished, in order from the narrowest question to the
    widest. Stability comes first because it compares one arm only with itself. What each analysis
    answers, and how the three that cluster features differ, is in experiments/analysis/README.md.
    The judge runs when the grid says so (`grid.judge`).
    """
    run = ("--run-dir", _short(run_dir))
    judged = (_serial(judge.CALL, (*_analysis("run_judge.py"), *run), after=after,
                      calls_model=True),)
    writes = any(registry.calls_language_model(pool) for _, pool in config.arm_cells())
    return (
        _serial(stability.CALL, (*_analysis("analyse_stability.py"), *run), after=after),
        _serial(co_discovery.CALL, (*_analysis("analyse_co_discovery.py"), *run), after=after),
        # Whether an arm proposes the same *ideas* across seeds, which set-based stability cannot
        # detect, because an arm on a written source names one idea differently each time and
        # moves the window it is evaluated over.
        _serial(recurrence.CALL, (*_analysis("analyse_recurrence.py"), *run), after=after,
                calls_model=writes),
        *(judged if config.grid.judge else ()),
    )


def _reduction(config: ExperimentConfig, run_dir: Path, arms: tuple[Stage, ...]
               ) -> tuple[Stage, ...]:
    """Stage 3A, when the config has a `reduction` section and reduces an arm: one process per
    reduced arm. An arm is reduced from its best real-label cell, so its process starts once the
    processes writing its real-label cells have finished, whatever the rest of the run is doing.
    Each process takes up to four threads, which PySR's searches use."""
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
    per profiled arm, started once that arm's reduction has finished. Each process takes every free
    thread for Tier 1's selections and is lent the threads that free up while it runs, so the
    panels run one at a time. The panel's work after the reveal is in `_after_reveal`."""
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
    """The reveal: the held-out split opened once, every saved model scored on it, and the panel's
    Tier 2 on the held-out rows. The reveal refuses a run that is not complete, so the check that
    every stage before it has finished is made there."""
    return _serial(reveal.CALL, (*_runner("run_reveal.py"), "--run-dir", _short(run_dir)),
                   phase=REVEAL, threads=None)


def _after_reveal(config: ExperimentConfig, run_dir: Path) -> tuple[Stage, ...]:
    """The stages that use the reveal's numbers: the panel's Tier 3 and tables, then the report's
    inputs under Core, then its figures under LocalUtils, drawn only from what the Core stage
    wrote. The report itself is written by hand or by an agent (process/reporting.md), so the
    driver stops at its inputs."""
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
    """Everything the run will do, in order, as the processes it will start. A config under which
    an arm could not start is refused here (`stage2.check`)."""
    stage2.check(config)
    arms = tuple(_arm(arm, config, run_dir) for arm in config.arm_cells())
    reduced = _reduction(config, run_dir, arms)
    return (_baseline(run_dir), *_motifs(config, run_dir), *arms,
            *_analyses(config, run_dir, after=_logs(arms)),
            *reduced, *_reliability(config, run_dir, reduced), _reveal(run_dir),
            *_after_reveal(config, run_dir))


def _logs(stages: tuple[Stage, ...]) -> tuple[str, ...]:
    """The logs of every command of `stages`, which a command that waits for all of them names."""
    return tuple(command.log for stage in stages for command in stage.commands)


# ------------------------------------------------------------------------------------- preflight

def preflight(config: ExperimentConfig, stages: tuple[Stage, ...], run_dir: Path) -> list[str]:
    """Everything that can be checked before any work is done.

    A run that dies three hours in for want of a credential loses the whole night, whereas each of
    these is a one-line check. The checks follow the selected stages instead of the whole config,
    so `--only l1_logistic/positional` needs no credentials and no vendored interpreter even when
    the config also names arms that do. The run's blinding is the exception: it is checked whatever
    is selected (`blinding_problems`), because it is a property of the whole run.

    The split files are not checked here. Loading the config has already read the dataset's
    `dataset.json` and checked the training and validation files against it, and the reveal checks
    the held-out split's file.
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
    # A motif collection (the one a known-motif source scans with, or the one discovered motifs are
    # compared with) is read by the step before the arms and by the arms whose pools hold a motif
    # source. Later stages take it from the featurisers the cells saved.
    reads_collection = [stage.name for stage in stages if stage.name == run_motifs.STAGE
                        or ("/" in stage.name and set(parse_pool(stage.name.partition("/")[2]))
                            & {MOTIF_KNOWN, MOTIF_DISCOVERED})]
    if reads_collection:
        problems += run_motifs.collection_problems(config, run_dir)
        problems += run_motifs.environment_problems(config)
    # A model fitted in another environment needs that environment wherever the model is fitted: in
    # its arm's stage, the reduction and the panel. A vendored method's environment is also needed
    # afterwards, by every stage that uses its models: the panel on both sides of the reveal, the
    # reveal itself, and the report's inputs, which load a model to draw it. PySR's equation is
    # evaluated in Core once it is found.
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
    # such an arm. A run without a written source or the judge needs neither a provider nor
    # credentials.
    calling = sorted(stage.name for stage in stages if stage.calls_model)
    unset = missing_settings() if calling else []
    if unset:
        problems.append(
            f"the language model is not configured: missing {', '.join(unset)}, needed by "
            f"{', '.join(calling)}. Run 'set -a && source .env && set +a' first.")
    return problems


# ------------------------------------------------------------------------------------- the run


class Journal:
    """The driver's messages, printed to the terminal and appended to a log that outlives it."""

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

    The topic is read from `NTFY_TOPIC` in the environment and never from the config. An agent uses
    it to watch a run it started, so it is not part of the experiment, and two runs of one config
    should not differ in their recorded settings because one was being watched. The topic is also a
    public channel to anyone who knows its name, so it must stay out of tracked files. When it is
    unset, the driver sends no notifications and does not complain.
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
    """Start one of the run's processes (`schedule.start_process`). It is a module attribute so
    that a test can stand in for every process the driver starts."""
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
    """Where this run's results go, with a check against the mistake that the dated default
    directory invites.

    Without the check, a resume on a later day than the launch would start an empty run: every
    stage "resumes" into an empty directory, redoes hours of work, and the results end up split
    across two directories. Resuming exists for interrupted runs, which are also the runs that meet
    this problem, so the driver refuses instead of relying on documentation.
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
    directory named by hand is still a run of its config, and a directory whose name contains
    another config's name (`screen_v2` contains `screen`) is not a run of that other config.
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

    A stage skips the cells that already have results and cannot tell a finished cell from one
    finished under different settings. Two configurations mixed in one table cannot be detected
    afterwards, so the difference is caught here. A deliberate change is allowed with
    `--allow-config-change`, and the superseded config is kept beside the new one, because it is
    the only record of the settings the cells already in the directory were computed under.

    A run whose config was recorded under another `version` of the config's shape is refused
    whatever the flag, because the stages could not take that run's settings from it.
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
    parser.add_argument("--detach", action="store_true",
                        help="start the run in a session of its own and return once it is under "
                             "way, so that it outlives the terminal and the command that started "
                             "it. Follow it in the run's logs/driver.log and status.html")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config, the budget and every command, run nothing")
    parser.add_argument("--estimate", action="store_true",
                        help="estimate each arm's time on this machine from a few fits on the "
                             "run's data, flag the slow ones, and run nothing; with --run-dir the "
                             "estimate is kept in the run's logs/")
    parser.add_argument("--slow-hours", type=_positive_float, default=None, metavar="HOURS",
                        help="with --estimate, the wall time over which an arm is flagged (default: "
                             "2)")
    # How much of the machine the run may take. These flags do not change any result, and each is
    # recorded in `logs/machine.jsonl`, so a run resumed on another machine may use other values.
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

    A revealed run resumes from the reveal, because everything before the reveal fits or selects
    and has already run (the reveal refuses to start until it has). `--stop-before-reveal` stops
    ahead of the reveal, so that a run's results can be checked before the held-out split is
    opened.
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


def detach(argv: list[str], run_dir: Path) -> int:
    """`--detach`: start this call again without the flag, in a session of its own, and return
    once it is under way.

    The run's directory is passed on explicitly, so the run that starts is the one this call
    resolved and checked. The call watches the run for `DETACH_WATCH_SECONDS`, so that a run that
    stops at once says so where the launch was typed.
    """
    arguments = [argument for argument in argv if argument != "--detach"]
    if not any(argument == "--run-dir" or argument.startswith("--run-dir=") for argument in arguments):
        arguments += ["--run-dir", str(run_dir)]
    logs = run_dir / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / DETACHED_OUTPUT).open("ab") as output:
        process = subprocess.Popen(
            (sys.executable, str(RUNNERS / "run_experiment.py"), *arguments),
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
    try:
        code = process.wait(timeout=DETACH_WATCH_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"The run is under way in the background, as process {process.pid}. Follow it in "
              f"{_short(logs / f'{CALL}.log')} and {_short(run_dir / status.PAGE_FILE)}.")
        return 0
    if code == 0:
        print(f"The run finished within {DETACH_WATCH_SECONDS} s. Its journal is "
              f"{_short(logs / f'{CALL}.log')}.")
        return 0
    print(f"The run stopped within {DETACH_WATCH_SECONDS} s, with exit code {code}. What it printed "
          f"is in {_short(logs / DETACHED_OUTPUT)}.", file=sys.stderr)
    return code


def run_estimate(args: argparse.Namespace, config: ExperimentConfig) -> int:
    """`--estimate`: print each arm's time on this machine, and keep the estimate in the run's
    `logs/` when a run directory is named. It starts no stage."""
    from seqmaestro.pipeline import estimate

    _, budget = run_budget(args)
    slow_hours = estimate.SLOW_HOURS if args.slow_hours is None else args.slow_hours
    for warning in config.warnings:
        print(f"WARNING: {warning}")
    print(f"Probing {len(config.arm_cells())} arm(s) on this machine, one thread each.", flush=True)
    try:
        probes, estimates, before = estimate.estimate_run(
            config, budget.cpus, slow_hours, report=lambda line: print(line, flush=True))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print("\n" + estimate.table(estimates, config, budget.cpus, slow_hours, before), flush=True)
    # The language model's part: each arm's calls, their cost and waiting, and the load on the
    # provider, with DEFT's tree counted at the splits its probe made.
    from seqmaestro.pipeline import calls

    found = calls.estimate(config, budget.model_calls,
                           nodes={probe.pool: probe.nodes for probe in probes if probe.nodes})
    print("\n" + calls.table(found), flush=True)
    print("\nThe grid, each arm's time and, where it calls a language model, its cost and waiting:\n"
          + calls.grid(config, {item.arm: item.wall for item in estimates}, found), flush=True)
    if args.run_dir is not None:
        kept = estimate.record(args.run_dir, probes, estimates, budget.cpus, slow_hours, before,
                               language_model=calls.as_record(found))
        print(f"\nkept in {_short(kept)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.slow_hours is not None and not args.estimate:
        print("--slow-hours applies only with --estimate.", file=sys.stderr)
        return 2
    if args.detach and (args.dry_run or args.estimate):
        print("--detach starts a run, so it does not go with --dry-run or --estimate.",
              file=sys.stderr)
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
        print(f"# budget: {budget.as_dict()}")
        print("# the settings each Core model is fitted with (the config's, or the model's defaults):")
        for model, values in config.effective_model_settings().items():
            print(f"#   {model}: {', '.join(f'{key}={value}' for key, value in values.items())}")
        print()
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
    if args.detach:
        return detach(sys.argv[1:] if argv is None else list(argv), run_dir)

    (run_dir / LOGS_DIR).mkdir(parents=True, exist_ok=True)
    journal = Journal(run_dir / LOGS_DIR)
    try:
        bind_config(run_dir, config, args.allow_config_change, journal)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    # Record, beside the config (which names no model), the model that the environment names for
    # the stages that call one, and how the run calls it. Each part that calls the model also
    # records the model it called.
    called = ({"language_model": describe_calls(provider_from_environment(), config.language_model)}
              if any(stage.calls_model for stage in stages) else {})
    record_call(run_dir, CALL, vars(args), config=config.to_mapping(), **called)

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

    # The reveal opens the held-out split for good, so once anything before it has failed, the
    # scheduler holds back the reveal and the stages that use its scores. The reveal would refuse
    # an incomplete run anyway, but holding it back here reports the failed stage by name.
    # The run's status page, kept current from the scheduler's jobs (`seqmaestro.status`), and
    # written once before anything starts, so that a page opened then reloads itself.
    updater = status.Updater(run_dir, journal.note)
    scheduler = Scheduler(stages, budget, journal, cwd=REPO, start=start_process,
                          on_failure=failed_stage, on_step=updater)
    updater(scheduler.jobs, True)
    outcome = scheduler.run()
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
                     f"process/report-template.md, as process/reporting.md says")
    updater.finish(scheduler.jobs)
    journal.note(f"status page: {_short(run_dir / status.PAGE_FILE)}")
    return 1 if failed or held else 0

