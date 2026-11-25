"""How a process runs, as distinct from what it computes: the threads its fits may use, the memory
it may hold, and what the machine it runs on can give.

**Threads.** A process started with `SEQMAESTRO_THREADS` set fits its models on that many threads:
XGBoost's and EBM's jobs and PySR's search threads follow it, unless the model's settings name a
number, as the one-core fits of the reliability panel's Tier 1 (stability) do. The numeric
libraries' thread pools (OpenMP, and the BLAS under numpy) read other variables as they load, so a
process that starts another passes the whole set (`thread_environment`). A process started without
the variable runs as the libraries do by default, on every core.

A process that may use more threads as they free up is also handed a file that holds the number it
may use now (`SEQMAESTRO_THREADS_FILE`). The driver raises that number as other processes end, and
lowers it again when a waiting process starts and takes some of the threads back. Tier 1's
selections follow the file as they start. Everything else keeps the number the process started
with, since a library sizes its thread pool once.

The thread count changes how fast a fit runs and leaves what it fits unchanged. XGBoost's and EBM's
fits are identical at any thread count, and a linear model's predictions differ at most in their
last bit, because the order of a sum can change. PySR's search is the exception: two searches under
the same seed can give different results at any thread count, one included.

At the sizes SeqMaestro fits, one thread per fit is about as fast as many and uses a fraction of
the CPU, so running fits side by side on one thread each completes more of them. Many threads per
fit on a machine whose cores are already busy are far slower, since each fit then waits for threads
that the operating system has set aside to run other work.

**Memory.** `SEQMAESTRO_MEMORY` tells a process how many bytes it may hold together with the
processes it starts, so a process that starts workers can choose how many to start (as Tier 1 does
for its selections).

**The machine.** The driver probes the machine once as a run starts (`probe`): the CPUs this process
may run on and the memory available to it, within any limit its control group (cgroup) sets. From
that it draws a budget (`budget`), and the driver's scheduler keeps the run's processes within the
budget by measuring each as it runs (`usage_of`). `psutil` is imported only where it is used, since
this module is also imported in environments that do not have it installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

#: The environment variable that gives a process its number of threads.
THREADS_VARIABLE = "SEQMAESTRO_THREADS"

#: The environment variable that names a file holding the number of threads a process may use now,
#: which the driver changes while the process runs.
THREADS_FILE_VARIABLE = "SEQMAESTRO_THREADS_FILE"

#: The environment variable that gives the bytes a process may hold together with the processes it
#: starts.
MEMORY_VARIABLE = "SEQMAESTRO_MEMORY"

#: The variables the numeric libraries read as they load: OpenMP's, which XGBoost and scikit-learn
#: use, and those of the BLAS libraries numpy may be built on.
LIBRARY_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

#: The share of the memory available as a run starts that its processes may hold together. The rest
#: is left to the system and to whatever else the machine runs.
MEMORY_SHARE = 0.8

#: Memory kept free on the machine at every moment: a process starts only while more than this
#: would remain. It is the larger of a share of the machine's memory and a floor.
RESERVE_SHARE = 0.05
RESERVE_FLOOR = 1 << 30

#: Where cgroup v2 mounts its hierarchy, and where a process finds its place in it.
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")


def _positive_number(variable: str, what: str) -> int | None:
    value = os.environ.get(variable, "").strip()
    if not value:
        return None
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{variable} is {value!r}, and must be a whole number of {what}, at "
                         f"least 1.")
    return int(value)


def allowed_threads() -> int | None:
    """The number of threads this process's fits may use, or None for a process given no number."""
    return _positive_number(THREADS_VARIABLE, "threads")


def threads_file() -> Path | None:
    """The file that holds the number of threads this process may use now, or None for a process
    given none."""
    value = os.environ.get(THREADS_FILE_VARIABLE, "").strip()
    return Path(value) if value else None


def allowed_memory() -> int | None:
    """The bytes this process may hold together with the processes it starts, or None for a
    process given no number."""
    return _positive_number(MEMORY_VARIABLE, "bytes")


def thread_environment(threads: int) -> dict[str, str]:
    """The variables that start a process on `threads` threads: the one SeqMaestro's models read,
    and the numeric libraries'."""
    if threads < 1:
        raise ValueError(f"A process needs at least one thread, not {threads!r}.")
    return {THREADS_VARIABLE: str(threads), **{name: str(threads) for name in LIBRARY_VARIABLES}}


def jobs(given: dict) -> dict[str, int]:
    """The `n_jobs` a model with these settings passes its library, or none when the settings
    already name one or when the process was given no number."""
    threads = allowed_threads()
    return {} if threads is None or "n_jobs" in given else {"n_jobs": threads}


# ------------------------------------------------------------------------------ the machine

@dataclass(frozen=True)
class Machine:
    """What this process may use of the machine it runs on, as `probe` found it."""

    #: The logical CPUs this process may run on, within any CPU quota of its control group.
    cpus: int
    #: The machine's physical cores, where the system reports them. They are recorded for
    #: reference only.
    cores: int | None
    #: Bytes of memory, within any limit of the process's control group.
    memory_total: int
    #: Bytes of memory available when probed, within that limit.
    memory_available: int
    #: The machine's load average over the last minute: how busy it already was.
    load: float
    #: The quota and the limit the control group sets, None where it sets none.
    cgroup_cpus: float | None
    cgroup_memory: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def _cgroup_directories(root: Path, membership: Path) -> list[Path]:
    """The process's control group and every group above it, nearest first, under cgroup v2. Empty
    where the system does not use the unified hierarchy."""
    try:
        line = next(line for line in membership.read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return []
    relative = Path(line[3:].strip().lstrip("/"))
    return [root / directory for directory in (relative, *relative.parents)]


def _cgroup_cpus(directories: list[Path]) -> float | None:
    """The tightest CPU quota along the groups, in CPUs, or None where the groups set none."""
    quotas = []
    for directory in directories:
        try:
            quota, period = (directory / "cpu.max").read_text().split()
        except (OSError, ValueError):
            continue
        if quota != "max":
            quotas.append(int(quota) / int(period))
    return min(quotas) if quotas else None


def _cgroup_memory(directories: list[Path]) -> tuple[int, int] | None:
    """The memory limit along the groups that leaves the least room, with what its group holds
    now, or None where the groups set none."""
    tightest = None
    for directory in directories:
        try:
            limit = (directory / "memory.max").read_text().strip()
            current = int((directory / "memory.current").read_text())
        except (OSError, ValueError):
            continue
        if limit != "max" and (tightest is None or int(limit) - current < tightest[0] - tightest[1]):
            tightest = (int(limit), current)
    return tightest


def available_memory(root: Path = CGROUP_ROOT, membership: Path = PROC_CGROUP) -> int:
    """Bytes available now, within any limit of this process's control group."""
    import psutil

    available = psutil.virtual_memory().available
    limited = _cgroup_memory(_cgroup_directories(root, membership))
    return available if limited is None else min(available, limited[0] - limited[1])


def probe(root: Path = CGROUP_ROOT, membership: Path = PROC_CGROUP) -> Machine:
    """What this process may use of the machine at this moment."""
    import psutil

    directories = _cgroup_directories(root, membership)
    quota = _cgroup_cpus(directories)
    cpus = len(os.sched_getaffinity(0))
    if quota is not None:
        cpus = max(1, min(cpus, int(quota)))
    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    limited = _cgroup_memory(directories)
    if limited is not None:
        total, available = min(total, limited[0]), min(available, limited[0] - limited[1])
    return Machine(cpus=cpus, cores=psutil.cpu_count(logical=False), memory_total=int(total),
                   memory_available=int(available), load=round(os.getloadavg()[0], 2),
                   cgroup_cpus=quota, cgroup_memory=None if limited is None else limited[0])


@dataclass(frozen=True)
class Budget:
    """What a run's processes may use together."""

    #: Threads that may run at once.
    cpus: int
    #: Bytes the processes may hold together.
    memory: int
    #: Processes that may call a language model at once.
    model_calls: int
    #: Bytes the machine keeps free whatever the processes hold.
    reserve: int

    def as_dict(self) -> dict:
        return asdict(self)


def budget(machine: Machine, *, model_calls: int, max_cpus: int | None = None,
           max_memory: int | None = None) -> Budget:
    """The budget a run takes from the machine: every CPU it may use and a share of the memory
    available, each lowered to the caps given."""
    if model_calls < 1:
        raise ValueError(f"A run needs at least one process that may call a language model, not "
                         f"{model_calls!r}.")
    cpus = machine.cpus if max_cpus is None else max(1, min(machine.cpus, max_cpus))
    memory = int(machine.memory_available * MEMORY_SHARE)
    if max_memory is not None:
        memory = min(memory, max_memory)
    reserve = max(RESERVE_FLOOR, int(machine.memory_total * RESERVE_SHARE))
    return Budget(cpus=cpus, memory=memory, model_calls=model_calls, reserve=reserve)


@dataclass(frozen=True)
class Usage:
    """What a process and all its descendants hold and have used."""

    #: Resident bytes, summed over the processes.
    memory: int
    #: CPU seconds, summed over the processes, with those of descendants that have ended and been
    #: waited for.
    cpu_seconds: float


def usage_of(pids: Iterable[int]) -> dict[int, Usage]:
    """The usage of each process in `pids` with its descendants, from one pass over the machine's
    processes. A process that has gone is left out.

    One pass takes about as long as walking one process's tree, so every running process can be
    measured every second. A process that has ended and not yet been waited for is still counted,
    with its complete CPU time and no memory, so a short process's CPU time is not lost. Resident
    memory counts a page shared between a process and a worker it forked once in each, so the sum
    errs high, which is the safe side for deciding what else may start.
    """
    import psutil

    found, children = {}, {}
    for process in psutil.process_iter(["pid", "ppid", "memory_info", "cpu_times"]):
        found[process.info["pid"]] = process.info
        children.setdefault(process.info["ppid"], []).append(process.info["pid"])
    usage = {}
    for pid in pids:
        if pid not in found:
            continue
        memory, cpu, below = 0, 0.0, [pid]
        while below:
            current = found.get(below.pop())
            if current is None:
                continue
            if current["memory_info"] is not None:
                memory += current["memory_info"].rss
            times = current["cpu_times"]
            if times is not None:
                cpu += times.user + times.system
                if current["pid"] == pid:
                    cpu += times.children_user + times.children_system
            below.extend(children.get(current["pid"], []))
        usage[pid] = Usage(memory=int(memory), cpu_seconds=float(cpu))
    return usage
