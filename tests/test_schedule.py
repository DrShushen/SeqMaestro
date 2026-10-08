"""The scheduler that runs a pipeline's processes side by side (`seqmaestro/pipeline/schedule.py`).

Every process here is a stand-in that ends when the test says, on a clock the test moves, with the
memory and CPU time the test gives it, so each rule can be checked at the moment it applies.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from seqmaestro.execution import (
    MEMORY_VARIABLE,
    THREADS_FILE_VARIABLE,
    THREADS_VARIABLE,
    Budget,
    Usage,
)
from seqmaestro.pipeline import schedule
from seqmaestro.pipeline.schedule import AFTER, BEFORE, REVEAL, Command, Scheduler, Stage

GB = 1 << 30


class Process:
    """A process that runs until the test ends it."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code


class Journal:
    def __init__(self, logs: Path) -> None:
        self.logs = logs
        self.lines: list[str] = []

    def banner(self, text: str) -> None:
        self.lines.append(f"=== {text}")

    def note(self, text: str) -> None:
        self.lines.append(text)


class Machine:
    """The stand-ins, the clock, and what the scheduler started, with what."""

    def __init__(self, tmp_path: Path, stages, budget: Budget) -> None:
        self.now = 0.0
        self.free = 1000 * GB
        self.processes: dict[str, Process] = {}
        self.usage: dict[int, Usage] = {}
        self.started: list[tuple[str, int, int]] = []
        self.files: dict[str, str | None] = {}
        self.journal = Journal(tmp_path)
        self.scheduler = Scheduler(
            stages, budget, self.journal, cwd=tmp_path, start=self.start,
            usage=lambda pids: {pid: self.usage[pid] for pid in pids if pid in self.usage},
            available=lambda: self.free,
            clock=lambda: self.now, sleep=lambda seconds: None)

    def start(self, argv, log, environment, cwd):
        process = Process(len(self.processes) + 1)
        self.processes[log.stem] = process
        self.started.append((log.stem, int(environment[THREADS_VARIABLE]),
                             int(environment[MEMORY_VARIABLE])))
        self.files[log.stem] = environment.get(THREADS_FILE_VARIABLE)
        return process, None

    def running(self) -> list[str]:
        return [log for log, process in self.processes.items() if process.code is None]

    def end(self, log: str, code: int = 0) -> None:
        self.processes[log].code = code

    def holds(self, log: str, memory: int, cpu_seconds: float = 0.0) -> None:
        self.usage[self.processes[log].pid] = Usage(memory=memory, cpu_seconds=cpu_seconds)

    def step(self, seconds: float = 0.0) -> None:
        self.now += seconds
        self.scheduler.step()


def stage(name: str, n: int = 1, **options) -> Stage:
    commands = tuple(Command(("run", name, str(i)), name if n == 1 else f"{name}_{i}",
                             after=options.pop("after", ()) if i == n - 1 else ())
                     for i in range(n))
    return Stage(name, commands, **options)


def budget(cpus: int = 8, memory: int = 100 * GB, model_calls: int = 4) -> Budget:
    return Budget(cpus=cpus, memory=memory, model_calls=model_calls, reserve=GB)


def threads_in_use(machine: Machine) -> int:
    """The threads of the running processes, as the scheduler counts them now."""
    return sum(job.threads for job in machine.scheduler.jobs if job.state == schedule.RUNNING)


def test_the_threads_of_the_running_processes_stay_within_the_budget(tmp_path):
    """Each of a stage's processes takes one thread, as the arms fit, until the budget is spent."""
    machine = Machine(tmp_path, (stage("arm", 12),), budget(cpus=8))
    machine.step()
    assert machine.running() == ["arm_0"], "the first runs alone until its memory is measured"
    machine.holds("arm_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    assert len(machine.running()) == 8 and threads_in_use(machine) == 8
    machine.end("arm_0")
    machine.step()
    assert len(machine.running()) == 8, "a process that ends makes room for the next"


def test_a_stage_that_takes_every_thread_waits_for_half_of_them_and_runs_alone(tmp_path):
    """A reliability panel runs for hours on the threads it starts with, so it waits until half are
    free, takes all that are, and does not share the machine with another of its kind."""
    panels = Stage("panel", tuple(Command(("panel", str(i)), f"panel_{i}", after=("arm_0",))
                                  for i in range(2)), threads=None)
    machine = Machine(tmp_path, (stage("arm", 6), panels), budget(cpus=8))
    machine.step()
    machine.holds("arm_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    assert len(machine.running()) == 6
    machine.end("arm_0")
    machine.step()
    assert not [log for log in machine.running() if log.startswith("panel")], \
        "5 of 8 threads are taken, so fewer than half are free"
    machine.end("arm_1")
    machine.step()
    [panel] = [entry for entry in machine.started if entry[0].startswith("panel")]
    assert panel[:2] == ("panel_0", 4), "half are free, and it takes them all"
    for log in machine.running():
        if log.startswith("arm"):
            machine.end(log)
    machine.step()
    assert [log for log in machine.running() if log.startswith("panel")] == ["panel_0"]
    machine.end("panel_0")
    machine.step()
    assert machine.running() == ["panel_1"] and machine.started[-1][1] == 8


def test_a_process_that_takes_every_thread_is_lent_those_that_free_up(tmp_path):
    """A panel that started beside other processes is lent their threads as they end, through the
    file it is pointed to."""
    panel = Stage("panel", (Command(("panel",), "panel", after=("arm_0", "arm_1")),), threads=None)
    machine = Machine(tmp_path, (stage("arm", 6), panel), budget(cpus=8))
    machine.step()
    machine.holds("arm_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    machine.end("arm_0")
    machine.end("arm_1")
    machine.step()
    assert machine.started[-1][:2] == ("panel", 4)
    assert machine.files["arm_2"] is None, "a process that takes one thread is given no file"
    threads = Path(machine.files["panel"])
    assert threads.read_text() == "4\n"
    machine.end("arm_2")
    machine.step()
    assert threads.read_text() == "5\n" and "  panel now on 5 threads" in machine.journal.lines
    for log in ("arm_3", "arm_4", "arm_5"):
        machine.end(log)
    machine.step()
    assert threads.read_text() == "8\n" and threads_in_use(machine) == 8
    machine.end("panel")
    machine.step()
    with (tmp_path / schedule.PROCESSES_FILE).open() as handle:
        [row] = [row for row in csv.DictReader(handle) if row["log"] == "panel"]
    assert (row["threads"], row["most_threads"]) == ("4", "8")


def test_a_process_whose_turn_comes_later_takes_back_what_the_panel_was_lent(tmp_path):
    """A reduction that becomes ready while a panel holds every thread starts at once on the
    threads it needs, taken back from those the panel was lent, never from those it started with."""
    panel = Stage("panel", (Command(("panel",), "panel", after=("arm_0", "arm_1")),), threads=None)
    reduction = Stage("reduction", (Command(("reduce",), "reduction", after=("arm_5",)),),
                      threads=4)
    machine = Machine(tmp_path, (stage("arm", 6), panel, reduction), budget(cpus=8))
    machine.step()
    machine.holds("arm_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    machine.end("arm_0")
    machine.end("arm_1")
    machine.step()
    threads = Path(machine.files["panel"])
    for log in ("arm_2", "arm_3", "arm_4"):
        machine.end(log)
    machine.step()
    assert threads.read_text() == "7\n", "lent the three threads the arms left"
    machine.end("arm_5")
    machine.step()
    assert machine.started[-1][:2] == ("reduction", 4)
    assert threads.read_text() == "4\n" and threads_in_use(machine) == 8, \
        "three taken back, and the panel keeps the four it started with"


def test_threads_go_first_to_the_processes_waiting_to_start(tmp_path):
    """A process whose turn has come starts on the threads that free up, and while one still waits,
    for whatever reason, the panel is given none of them."""
    panel = Stage("panel", (Command(("panel",), "panel", after=("arm_0", "arm_1")),), threads=None)
    reduction = Stage("reduction", tuple(Command(("reduce", str(i)), f"reduction_{i}",
                                                 after=("arm_2",)) for i in range(2)))
    machine = Machine(tmp_path, (stage("arm", 6), panel, reduction), budget(cpus=8))
    machine.step()
    machine.holds("arm_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    machine.end("arm_0")
    machine.end("arm_1")
    machine.step()
    threads = Path(machine.files["panel"])
    machine.end("arm_2")
    machine.step()
    assert "reduction_0" in machine.running(), "it starts on the thread arm_2 left"
    machine.end("arm_3")
    machine.end("arm_4")
    machine.step()
    assert "reduction_1" not in machine.running(), "it waits for its stage to be measured"
    assert threads.read_text() == "4\n" and threads_in_use(machine) == 6, \
        "two threads are free, and kept for the reduction that waits"
    machine.holds("reduction_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    assert "reduction_1" in machine.running()
    assert threads.read_text() == "5\n" and threads_in_use(machine) == 8


def test_processes_calling_a_language_model_stay_within_their_places(tmp_path):
    machine = Machine(tmp_path, (stage("proposer", 6, calls_model=True),), budget(model_calls=4))
    machine.step()
    machine.holds("proposer_0", GB, cpu_seconds=1.0)
    machine.step(schedule.SETTLE_SECONDS)
    assert len(machine.running()) == 4


def test_a_measured_light_model_call_starts_whatever_the_threads_of_the_others(tmp_path):
    """Processes waiting on the network count for the CPU they were measured to use, so they run
    beside a panel that holds the threads. Holding them back would hold back the longest stages."""
    stages = (stage("panel", 1, threads=None), stage("proposer", 3, calls_model=True))
    machine = Machine(tmp_path, stages, budget(cpus=4))
    machine.step()
    assert machine.started[0][:2] == ("panel", 4)
    machine.step()
    assert "proposer_0" not in machine.running(), "unmeasured, it counts for its thread"
    machine.end("panel")
    machine.step()
    machine.processes["panel"] = Process(99)
    machine.step()
    machine.holds("proposer_0", GB, cpu_seconds=3.0)
    machine.step(schedule.SETTLE_SECONDS)
    assert {"proposer_1", "proposer_2"} <= set(machine.running()), "measured at 0.05 of a CPU"


def test_memory_is_expected_from_what_the_stage_was_measured_to_hold(tmp_path):
    """Once a process of a stage has finished, the others are expected to hold a quarter more than
    it did, and start only while that fits in the budget beside the running ones."""
    machine = Machine(tmp_path, (stage("kmer", 6),), budget(cpus=16, memory=20 * GB))
    machine.step()
    machine.holds("kmer_0", 4 * GB)
    machine.step()
    machine.end("kmer_0")
    machine.step()
    assert len(machine.running()) == 4, "5 GB each, 20 GB in all"
    assert [memory // GB for _, _, memory in machine.started[1:]] == [20, 15, 10, 5], \
        "each is told what the budget leaves beside what the others are expected to hold"


def test_a_process_waits_while_the_machine_would_drop_below_its_reserve(tmp_path):
    machine = Machine(tmp_path, (stage("kmer", 3),), budget(cpus=16))
    machine.step()
    machine.holds("kmer_0", 4 * GB)
    machine.step()
    machine.end("kmer_0")
    machine.free = 5 * GB
    machine.step()
    assert machine.running() == ["kmer_1"], \
        "one process always runs, and a second would leave the machine less than its reserve"
    machine.free = 100 * GB
    machine.step()
    assert len(machine.running()) == 2


def test_a_process_waits_for_the_commands_it_names_and_for_earlier_phases(tmp_path):
    stages = (stage("arm", 2), stage("reduction", after=("arm_0", "arm_1")),
              stage("reveal", phase=REVEAL), stage("after", phase=AFTER))
    machine = Machine(tmp_path, stages, budget())
    machine.step()
    machine.holds("arm_0", GB)
    machine.step(schedule.SETTLE_SECONDS)
    assert set(machine.running()) == {"arm_0", "arm_1"}
    machine.end("arm_0")
    machine.step()
    assert "reduction" not in machine.processes
    machine.end("arm_1")
    machine.step()
    assert machine.running() == ["reduction"]
    machine.end("reduction")
    machine.step()
    assert machine.running() == ["reveal"]
    machine.end("reveal")
    machine.step()
    assert machine.running() == ["after"]


def test_a_failure_before_the_reveal_holds_it_and_what_reads_it(tmp_path):
    stages = (stage("arm", retry=False), stage("reveal", phase=REVEAL), stage("after", phase=AFTER))
    machine = Machine(tmp_path, stages, budget())
    machine.step()
    machine.end("arm", 1)
    while not machine.scheduler.finished:
        machine.step()
    outcome = machine.scheduler.outcome()
    assert outcome.failed == ["arm"] and outcome.held == ["reveal", "after"]
    assert list(machine.processes) == ["arm"]


def test_a_failure_after_the_reveal_holds_nothing_back(tmp_path):
    stages = (stage("reveal", phase=REVEAL), stage("first", phase=AFTER, retry=False),
              stage("second", phase=AFTER))
    machine = Machine(tmp_path, stages, budget())
    machine.step()
    machine.end("reveal")
    machine.step()
    machine.end("first", 1)
    machine.step()
    machine.step()
    assert "second" in machine.running()


def test_a_stage_that_retries_starts_a_failed_process_once_more(tmp_path):
    machine = Machine(tmp_path, (stage("arm", retry=True), stage("tree", retry=False)), budget())
    machine.step()
    assert set(machine.running()) == {"arm", "tree"}
    machine.end("arm", 1)
    machine.end("tree", 1)
    machine.step()
    machine.step()
    assert [entry[0] for entry in machine.started].count("arm") == 2
    assert [entry[0] for entry in machine.started].count("tree") == 1
    with (tmp_path / schedule.PROCESSES_FILE).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["log"], row["attempt"], row["exit"]) for row in rows] == [
        ("arm", "1", "1"), ("tree", "1", "1")]


def test_two_commands_with_one_log_are_refused(tmp_path):
    clash = (Stage("a", (Command(("x",), "same"),)), Stage("b", (Command(("y",), "same"),)))
    with pytest.raises(ValueError, match="share a log"):
        Scheduler(clash, budget(), Journal(tmp_path), cwd=tmp_path)


def test_a_process_is_told_its_threads_and_memory(tmp_path):
    machine = Machine(tmp_path, (stage("reduction", threads=4),), budget(cpus=8, memory=30 * GB))
    machine.step()
    assert machine.started == [("reduction", 4, 30 * GB)]


@pytest.mark.parametrize("phase", [BEFORE, REVEAL, AFTER])
def test_the_phases_are_named_in_order(phase):
    assert schedule.PHASES.index(phase) == [BEFORE, REVEAL, AFTER].index(phase)


def test_a_stage_with_no_process_is_neither_failed_nor_held(tmp_path):
    """Held means every process of a stage was kept from starting, which a stage without any
    cannot be."""
    machine = Machine(tmp_path, (Stage("empty", ()), stage("arm")), budget())
    machine.step()
    machine.end("arm")
    machine.step()
    outcome = machine.scheduler.outcome()
    assert outcome.failed == [] and outcome.held == []
