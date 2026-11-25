"""Run a pipeline's processes side by side within the machine's budget.

The driver hands over every process of the stages it runs (`Stage`), in the order it lists the
stages, together with a budget drawn from a probe of the machine (`seqmaestro.execution.budget`):
threads, memory, and how many processes may call a language model at once. A process starts once
what it waits for has finished and the budget has room for it. Of two processes that could both
start, the earlier one starts first, but a later one may start while an earlier one waits for room.

**What a process waits for.** A process waits for every process of an earlier phase (`BEFORE`, the
`REVEAL`, then `AFTER`), for every process of an earlier stage in the `AFTER` phase (whose stages
run one after another), and for the processes its command names (`Command.after`). It waits for
them to finish, with or without a failure, so a failure does not stop the processes after it. The
reveal and the stages after it are the exception: once a process before them has failed, they are
held back and never start.

**Threads.** A process is given as many threads as its stage can use well (`Stage.threads`), or as
many as are free when it starts if that is fewer, and at least one. The process and everything it
starts run on that number (`seqmaestro.execution.thread_environment`). The threads of the running
processes stay within the budget.
- The arms fit on one thread each, so many of their processes run side by side.
- A stage that takes every free thread, such as the reliability panel, runs one such process at a
  time, and starts it only once half of the threads are free, so that it does not start its many
  selections on the few threads left free at a busy moment. While it runs, it is lent the threads
  still free once every other process whose turn has come has started, apart from processes of its
  own kind and those that start regardless of the others' threads (below). A process whose turn
  comes later (other than one of those) takes back what it needs of the lent threads, down to the
  threads the panel started with. The number of threads the panel may use is kept in a file it is
  pointed to (`seqmaestro.execution.THREADS_FILE_VARIABLE`), and its Tier 1 starts a selection only
  while fewer selections than that number are running. Threads taken back free up as the panel's
  running selections end, so for up to one selection the processes hold more threads than the
  budget.
- A process that calls a language model mostly waits on the network. Until a process of its stage
  has run for a minute or ended, it counts for its threads, and from then on for the CPUs its stage
  has been measured to use. Once its stage is measured at under half a CPU, it starts regardless of
  the others' threads, because holding it back would hold back the run's longest stages.

**Memory is measured.** The driver samples the memory of every running process and its
descendants. The first process of a stage runs alone among its stage's processes until it has run
for a minute or ended. From then on, a process of the stage is expected to hold a quarter more than
the most a finished one held, or twice the most held by one that has run for a minute and is still
running, whichever is larger. It starts only if that fits in the budget beside what the running
processes hold or are expected to hold, and only while the machine would keep its reserve free. It
is told how much it may hold (`seqmaestro.execution.MEMORY_VARIABLE`).

**At least one process always runs**, so a process that needs more than the budget still runs, on
its own.

**Retries.** A process of a stage that retries (`Stage.retry`) is started once more when it fails.
Every stage resumes, so the second start redoes only what failed.

**What the scheduler records.** It writes a line in the driver's log as each process starts and
ends and as a panel's threads change, and a row for each process in `logs/processes.csv`: the
threads it started with and the most it held, its time, its CPU time and the most memory it held.
After each step it hands its jobs to `on_step`, through which the driver keeps a snapshot of what
is running (`snapshot`) and the run's status page current (`seqmaestro.status`).
"""

from __future__ import annotations

import csv
import math
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seqmaestro.execution import (
    MEMORY_VARIABLE,
    THREADS_FILE_VARIABLE,
    Budget,
    available_memory,
    thread_environment,
    usage_of,
)

#: The three phases of a run, in order. A `BEFORE` process may not run once the reveal has begun,
#: and an `AFTER` process has nothing to read until the reveal has finished.
BEFORE, REVEAL, AFTER = "before", "reveal", "after"
PHASES = (BEFORE, REVEAL, AFTER)

#: How long a process runs before its stage's memory and CPU are taken as measured.
SETTLE_SECONDS = 60.0
#: How often the running processes are measured.
SAMPLE_SECONDS = 2.0
#: How long the scheduler waits when nothing started and nothing ended.
TICK_SECONDS = 1.0
#: The memory a process is expected to hold, as a multiple of the most a finished process of its
#: stage held, or of the most a running one has held so far (which may still grow), whichever is
#: larger.
FINISHED_MARGIN = 1.25
RUNNING_MARGIN = 2.0
#: A process that calls a language model starts regardless of the others' threads once its stage
#: is measured below this many CPUs.
LIGHT_SHARE = 0.5

PROCESSES_FILE = "processes.csv"
#: The directory under the logs where a process that takes every free thread finds the number of
#: threads it may use now.
THREADS_DIR = "threads"
#: The snapshot of the jobs of the driver's latest invocation, under the logs (`snapshot`).
JOBS_FILE = "jobs.json"


@dataclass(frozen=True)
class Command:
    """One process and the log it writes to. The log is named after the part of the grid the
    process runs, so that processes running side by side leave one readable log each."""

    argv: tuple[str, ...]
    log: str
    #: The logs of the commands this one waits for, besides what its phase waits for.
    after: tuple[str, ...] = ()
    #: The arms whose real-label cells this command writes, as `(model, pool)`: what the
    #: reduction of each of them waits for.
    produces: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline: a name to report under, and the processes that make it up.

    A stage that retries starts a failed process once more. Transient failures such as a rate
    limit or a network blip clear on a second start, and every stage resumes, so a retry redoes
    only what failed. The vendored arms that call a language model do not retry, because a failure
    part-way through a tree has already spent its tokens and a second start would spend them again.
    """

    name: str
    commands: tuple[Command, ...]
    retry: bool = True
    #: Where the stage sits relative to the reveal: `BEFORE` it (fits or selects, on validation
    #: only), the `REVEAL` itself, or `AFTER` it (reads what the reveal wrote).
    phase: str = BEFORE
    #: The most threads one of its processes can use well, or None for every free one.
    threads: int | None = 1
    #: Whether its processes call a language model, so that each takes one of the budget's places
    #: for such calls.
    calls_model: bool = False


WAITING, RUNNING, DONE, FAILED, HELD = "waiting", "running", "done", "failed", "held"


@dataclass
class Job:
    """A command as the scheduler tracks it."""

    stage: Stage
    command: Command
    order: int
    state: str = WAITING
    attempts: int = 0
    process: Any = None
    handle: Any = None
    started: float = 0.0
    started_utc: str = ""
    samples: int = 0
    #: The threads it may use (which change while it runs if its stage takes every free thread), the
    #: threads it started with, and the most it held.
    threads: int = 0
    threads_at_start: int = 0
    most_threads: int = 0
    #: The bytes it was told it may hold.
    memory: int = 0
    #: The most memory it held and the CPU seconds it had used, as of the last measurement.
    peak: int = 0
    cpu_seconds: float = 0.0
    elapsed: float = 0.0
    code: int | None = None

    @property
    def finished(self) -> bool:
        return self.state in (DONE, FAILED, HELD)


@dataclass(frozen=True)
class Outcome:
    """The stages with a process that failed, and those held back, each in the order run."""

    failed: list[str]
    held: list[str]


def start_process(argv: tuple[str, ...], log: Path, environment: dict[str, str], cwd: Path):
    """Start one process with its output appended to its log, and return the process and the open
    log."""
    handle = log.open("a")
    handle.write(f"\n$ {shlex.join(argv)}\n")
    handle.flush()
    process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, cwd=cwd,
                               env=environment)
    return process, handle


def snapshot(jobs: list[Job], finished: bool = False) -> dict[str, Any]:
    """What each process of the driver's invocation is doing, for a reader outside the driver: its
    log's name (the call's name), its stage, its state, when it started and how often. `finished`
    says that the invocation has ended, so that a reader can tell a finished run from a driver that
    stopped without saying so."""
    return {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "driver_pid": os.getpid(),
        "finished": finished,
        "jobs": [{"log": job.command.log, "stage": job.stage.name, "phase": job.stage.phase,
                  "state": job.state, "started_utc": job.started_utc or None,
                  "attempts": job.attempts, "threads": job.threads} for job in jobs],
    }


def write_threads(path: Path, threads: int) -> None:
    """Write the number of threads a process may use now through one rename, so that the process
    never finds the file half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(f"{threads}\n")
    partial.replace(path)


class Scheduler:
    """Runs every process of `stages` within `budget`, reporting to `journal`."""

    def __init__(self, stages: tuple[Stage, ...], budget: Budget, journal, *, cwd: Path,
                 start: Callable = start_process, usage: Callable = usage_of,
                 available: Callable[[], int] = available_memory,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 on_failure: Callable[[Stage], None] | None = None,
                 on_step: Callable[[list[Job], bool], None] | None = None) -> None:
        self.stages, self.budget, self.journal, self.cwd = stages, budget, journal, cwd
        self.start, self.usage, self.available = start, usage, available
        self.clock, self.sleep = clock, sleep
        self.on_failure = on_failure or (lambda stage: None)
        #: Given the jobs after every step, and whether a process started or ended in it.
        self.on_step = on_step or (lambda jobs, changed: None)
        self.jobs = [Job(stage, command, order) for order, (stage, command) in enumerate(
            (stage, command) for stage in stages for command in stage.commands)]
        self.by_log = {job.command.log: job for job in self.jobs}
        if len(self.by_log) != len(self.jobs):
            raise ValueError("Two commands share a log, so one would write over the other's.")
        self.announced: set[str] = set()
        self.measured = float("-inf")

    # ------------------------------------------------------------------------------ the loop

    def run(self) -> Outcome:
        """Run every process to the end."""
        while not self.finished:
            changed = self.step()
            self.on_step(self.jobs, changed)
            if not changed:
                self.sleep(TICK_SECONDS)
        return self.outcome()

    @property
    def finished(self) -> bool:
        return all(job.finished for job in self.jobs)

    def step(self) -> bool:
        """Measure the running processes, collect those that have ended, start those that can
        start, and lend the threads left to a process that takes every free one. The order means
        that what starts is decided on the latest measurements, and that the processes waiting to
        start take threads before any are lent. Returns whether anything ended or started."""
        self._measure()
        progressed = self._collect_what_ended()
        progressed = self._start_what_can() or progressed
        self._lend()
        return progressed

    def outcome(self) -> Outcome:
        return Outcome(
            failed=[stage.name for stage in self.stages
                    if any(job.state == FAILED for job in self._of(stage))],
            held=[stage.name for stage in self.stages
                  if (jobs := self._of(stage)) and all(job.state == HELD for job in jobs)])

    def _of(self, stage: Stage) -> list[Job]:
        return [job for job in self.jobs if job.stage is stage]

    def _start_what_can(self) -> bool:
        progressed = False
        for job in self.jobs:
            if job.state != WAITING or not self._ready(job):
                continue
            if job.stage.phase != BEFORE and self._blocked():
                job.state = HELD
                progressed = True
                continue
            grant = self._admit(job)
            if grant is not None:
                self._launch(job, *grant)
                if not self._light(job.stage):
                    self._take_back()
                progressed = True
        return progressed

    def _ready(self, job: Job) -> bool:
        phase = PHASES.index(job.stage.phase)
        for other in self.jobs:
            if other.finished or other is job:
                continue
            if PHASES.index(other.stage.phase) < phase:
                return False
            if (job.stage.phase == AFTER and other.stage.phase == AFTER
                    and other.stage is not job.stage and other.order < job.order):
                return False
        return all(self.by_log[name].finished for name in job.command.after
                   if name in self.by_log)

    def _blocked(self) -> bool:
        """Whether a process before the reveal, or the reveal itself, has failed."""
        return any(job.state == FAILED and job.stage.phase != AFTER for job in self.jobs)

    # ------------------------------------------------------------------------------ the budget

    def _share(self, stage: Stage) -> float | None:
        """The most CPU a process of the stage has been measured to use, in CPUs, or None until one
        has run for `SETTLE_SECONDS` or ended."""
        now = self.clock()
        shares = [job.cpu_seconds / (now - job.started) for job in self._of(stage)
                  if job.state == RUNNING and job.samples and now - job.started >= SETTLE_SECONDS]
        shares += [job.cpu_seconds / job.elapsed for job in self._of(stage)
                   if job.state in (DONE, FAILED) and job.samples and job.elapsed > 0]
        return max(shares) if shares else None

    def _load(self, stage: Stage, threads: int) -> float:
        """The CPUs a process of `stage` on `threads` threads counts for."""
        share = self._share(stage) if stage.calls_model else None
        return threads if share is None else min(float(threads), share)

    def _light(self, stage: Stage) -> bool:
        """Whether a process of `stage` starts regardless of the others' threads, as one that calls
        a language model does once its stage is measured at under `LIGHT_SHARE` of a CPU."""
        share = self._share(stage) if stage.calls_model else None
        return share is not None and share < LIGHT_SHARE

    def _expected_memory(self, stage: Stage) -> int | None:
        """What a process of the stage is expected to hold, or None until one has been measured."""
        now = self.clock()
        finished = [job.peak for job in self._of(stage)
                    if job.state in (DONE, FAILED) and job.samples]
        settled = [job.peak for job in self._of(stage)
                   if job.state == RUNNING and job.samples and now - job.started >= SETTLE_SECONDS]
        if not finished and not settled:
            return None
        return int(max([peak * FINISHED_MARGIN for peak in finished]
                       + [peak * RUNNING_MARGIN for peak in settled]))

    def _admit(self, job: Job) -> tuple[int, int] | None:
        """The threads and the memory `job` starts with, or None while the budget has no room."""
        stage, budget = job.stage, self.budget
        wide = stage.threads is None
        wanted = budget.cpus if wide else max(1, min(stage.threads, budget.cpus))
        running = [other for other in self.jobs if other.state == RUNNING]
        if not running:
            return wanted, budget.memory
        if stage.calls_model and sum(other.stage.calls_model for other in running) >= \
                budget.model_calls:
            return None
        used = sum(self._load(other.stage, other.threads) for other in running)
        free = budget.cpus - used
        if wide and (free < budget.cpus / 2
                     or any(other.stage.threads is None for other in running)):
            return None
        lent = 0 if wide else self._lent()
        threads = max(1, min(wanted, int(free + lent)))
        if not self._light(stage) and used - lent + self._load(stage, threads) > budget.cpus + 1e-9:
            return None
        expected = self._expected_memory(stage)
        if expected is None and any(other.stage is stage for other in running):
            return None
        held = sum(max(other.peak, self._expected_memory(other.stage) or 0) for other in running)
        need = expected or 0
        if held + need > budget.memory or self.available() - need < budget.reserve:
            return None
        return threads, max(need, budget.memory - held)

    def _waits_for_threads(self, job: Job) -> bool:
        """Whether `job` waits to start and will hold threads once it does: a process whose turn
        has come, of a stage that takes a set number of threads, unless it starts regardless of the
        others' threads (`_light`)."""
        return (job.state == WAITING and job.stage.threads is not None and self._ready(job)
                and not self._light(job.stage))

    def _lend(self) -> None:
        """Lend the threads still free to the running process that takes every free thread, once
        every process that waits for threads has started (`_waits_for_threads`)."""
        running = [job for job in self.jobs if job.state == RUNNING]
        wide = [job for job in running if job.stage.threads is None]
        if not wide or any(self._waits_for_threads(job) for job in self.jobs):
            return
        free = int(self.budget.cpus - sum(self._load(job.stage, job.threads) for job in running))
        if free >= 1:
            self._set_threads(wide[0], wide[0].threads + free)  # one runs at a time (`_admit`)

    def _lent(self) -> int:
        """The threads the running processes that take every free one were lent beyond those they
        started with: what a process whose turn comes may take back."""
        return sum(job.threads - job.threads_at_start for job in self.jobs
                   if job.state == RUNNING and job.stage.threads is None)

    def _take_back(self) -> None:
        """Take back from the process with lent threads as many threads as the running processes
        hold beyond the budget, down to the threads it started with."""
        running = [job for job in self.jobs if job.state == RUNNING]
        over = math.ceil(sum(self._load(job.stage, job.threads) for job in running)
                         - self.budget.cpus - 1e-9)
        for job in running:
            if over > 0 and job.stage.threads is None and job.threads > job.threads_at_start:
                back = min(over, job.threads - job.threads_at_start)
                self._set_threads(job, job.threads - back)
                over -= back

    def _set_threads(self, job: Job, threads: int) -> None:
        """Change the threads of a running process that takes every free one, both in the count and
        in the file the process reads them from."""
        job.threads, job.most_threads = threads, max(job.most_threads, threads)
        write_threads(self._threads_file(job), threads)
        self.journal.note(f"  {job.command.log} now on {threads} threads")

    def _threads_file(self, job: Job) -> Path:
        return self.journal.logs / THREADS_DIR / job.command.log

    # ------------------------------------------------------------------------------ processes

    def _launch(self, job: Job, threads: int, memory: int) -> None:
        stage = job.stage
        if stage.name not in self.announced:
            self.announced.add(stage.name)
            count = len(stage.commands)
            self.journal.banner(f"stage: {stage.name}" + (f"  ({count} processes)" if count > 1
                                                          else ""))
        environment = {**os.environ, **thread_environment(threads), MEMORY_VARIABLE: str(memory)}
        if stage.threads is None:
            write_threads(self._threads_file(job), threads)
            environment[THREADS_FILE_VARIABLE] = str(self._threads_file(job).resolve())
        job.process, job.handle = self.start(job.command.argv,
                                             self.journal.logs / f"{job.command.log}.log",
                                             environment, self.cwd)
        job.state, job.memory = RUNNING, memory
        job.threads = job.threads_at_start = job.most_threads = threads
        job.attempts += 1
        job.started = self.clock()
        job.started_utc = datetime.now(UTC).isoformat(timespec="seconds")
        job.samples, job.peak, job.cpu_seconds = 0, 0, 0.0
        pid = getattr(job.process, "pid", None)
        self.journal.note(f"  started {job.command.log}  ({threads} thread{'s' * (threads > 1)}"
                          + (f", pid {pid})" if pid else ")"))

    def _measure(self) -> None:
        """Measure every running process at most once every `SAMPLE_SECONDS`, or sooner when one
        has started since the last pass."""
        now = self.clock()
        running = {job.process.pid: job for job in self.jobs
                   if job.state == RUNNING and getattr(job.process, "pid", None) is not None}
        fresh = any(job.samples == 0 for job in running.values())
        if not running or (now - self.measured < SAMPLE_SECONDS and not fresh):
            return
        self.measured = now
        usage = self.usage(list(running))
        for pid, job in running.items():
            if pid in usage:
                job.samples += 1
                job.peak = max(job.peak, usage[pid].memory)
                job.cpu_seconds = max(job.cpu_seconds, usage[pid].cpu_seconds)

    def _collect_what_ended(self) -> bool:
        progressed = False
        for job in self.jobs:
            if job.state != RUNNING:
                continue
            code = job.process.poll()
            if code is None:
                continue
            progressed = True
            if job.handle is not None:
                job.handle.close()
            job.code, job.elapsed = code, self.clock() - job.started
            self._record(job)
            if code != 0 and job.stage.retry and job.attempts < 2:
                self.journal.note(f"  {job.command.log} exited {code}; starting it once more")
                job.state = WAITING
                continue
            job.state = DONE if code == 0 else FAILED
            self.journal.note(f"  {job.command.log} exited {code}  ({job.elapsed / 60:.1f} min, "
                              f"{job.cpu_seconds / 60:.1f} CPU-min, at most "
                              f"{job.peak / 2**30:.1f} GB)")
            if all(other.finished for other in self._of(job.stage)):
                self._stage_ended(job.stage)
        return progressed

    def _stage_ended(self, stage: Stage) -> None:
        jobs = self._of(stage)
        failures = sum(job.state == FAILED for job in jobs)
        if len(jobs) == 1:
            self.journal.note(f"stage {stage.name} exited {jobs[0].code}   "
                              f"(log: {self.journal.logs / jobs[0].command.log}.log)")
        else:
            self.journal.note(f"stage {stage.name} finished; {failures} of {len(jobs)} processes "
                              f"failed")
        if failures:
            self.on_failure(stage)

    def _record(self, job: Job) -> None:
        """Write a row in `logs/processes.csv` for every start of a process."""
        path = self.journal.logs / PROCESSES_FILE
        new = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["log", "stage", "attempt", "started_utc", "seconds",
                                 "cpu_seconds", "peak_mb", "threads", "most_threads",
                                 "memory_mb", "exit"])
            writer.writerow([job.command.log, job.stage.name, job.attempts, job.started_utc,
                             round(job.elapsed, 1), round(job.cpu_seconds, 1),
                             round(job.peak / 2**20), job.threads_at_start, job.most_threads,
                             round(job.memory / 2**20), job.code])
