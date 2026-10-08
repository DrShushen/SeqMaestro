"""Tier 1's selections: the arm's procedure on every half-sample, in parallel and resumably.
**numpy and the standard library only.**

One half-sample selection is one full pass of the procedure that produced the arm's reduced model
(fit on the candidates, keep what the model used, eliminate to N). Under each label set an arm
therefore costs 2B whole eliminations, which makes Tier 1 the panel's main cost. This shapes four
things.

**Resume.** The finished selections are written to disk with a `done` flag beside each, every
half minute and at the end, so an interrupted sweep restarts where it stopped. The file holds the
selection matrix itself, a (2B, p) boolean array, which is what Tier 1's statistics read, so there
is no second format. A write compresses every half-sample's rows, which is slow on a large D, so
the sweep hands the workers that finished their next half-sample before it writes.

**Fork, not pickle.** Workers are forked and inherit the candidate matrix and the fitter
copy-on-write, and each worker copies only the rows of its half-sample. Sending a pickled copy
of a large matrix to each of twenty workers would cost more than the fits. Inheriting the fitter is
also what lets the vendored tree's side pass in a closure over its tree. A worker runs one
selection after another until the sweep ends, so what it starts once, such as a PySR worker's
search process, serves every selection it runs.

**Memory decides how many workers run.** Each worker holds its half-sample's rows and its fits,
which on many candidates over many rows run to gigabytes, so as many workers as there are threads
could take more memory than the machine has. Given the memory the panel may hold, the sweep runs its
first selection alone in one worker and measures what that worker holds that no other process
shares, its descendants included (a PySR worker's search process). It then runs as many workers as
fit beside what the panel itself holds, each expected to hold a quarter more than the one measured,
and never more than the threads it may use. The memory is read from `/proc`, so on a system without
it the sweep runs as many workers as it has threads.

**The threads it may use change while it runs.** The driver starts a panel on the threads free at
the time, and may hand it a file that holds the number it may use (`threads_file`). It raises the
number as other processes end, and lowers it when a process whose turn comes takes threads back.
The sweep looks at the file before it starts each selection, and every few seconds while none ends,
and starts a selection only while fewer run than the file holds, within what fits in memory.

**Why numpy only.** The vendored tree's side runs the same sweep for its arms without importing
`seqmaestro`. It loads this file by path, as it loads the elimination loop, and hands the loop in as
`eliminate_run`. `tests/test_reduction.py` fails if this file imports anything but numpy and the
standard library.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np

_STATE: dict = {}

#: A worker is expected to hold this many times what the measured one held.
WORKER_MARGIN = 1.25

#: How often a sweep looks again at the threads it may use while no selection ends.
RECHECK_SECONDS = 5.0

#: How often a sweep writes the selections it has finished: at most once in this many seconds, and
#: at its end.
SAVE_SECONDS = 30.0


def selector(eliminate_run, names: list[str], n_target: int):
    """`select(fit_step, X, y)`: the features of `names` that the elimination keeps on these rows
    and its final model uses (the loop's `final_used`).

    A survivor the final model does not use holds its place by the tie-break alone, and counting it
    would give the first columns in record order a selection frequency near one whenever the model
    uses fewer than `n_target`. The elimination loop predicts on the validation split it is handed at
    every step. Tier 1 uses only the surviving set, so one row stands in for that split and the loop
    itself runs unchanged.
    """
    def select(fit_step, X: np.ndarray, y: np.ndarray) -> list[str]:
        result = eliminate_run(fit_step, X, y, X[:1], names, n_target)
        return [str(name) for name in result["final_used"]]
    return select


def selected_mask(names: list[str], chosen: list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(names)}
    mask = np.zeros(len(names), dtype=bool)
    mask[[lookup[name] for name in chosen]] = True
    return mask


def _run_one(index: int) -> tuple[int, np.ndarray, float]:
    started = time.time()
    rows = _STATE["halves"][index]
    weights = None if _STATE["weights"] is None else _STATE["weights"][rows]
    chosen = _STATE["select"](_STATE["make_fit_step"](weights),
                              np.ascontiguousarray(_STATE["X"][rows]), _STATE["y"][rows])
    return index, selected_mask(_STATE["names"], chosen), time.time() - started


def unshared_bytes(pid: int) -> int:
    """What process `pid` and the processes below it hold that no other process shares, from
    `/proc`; 0 where `/proc` cannot say. For a forked worker this is what it added to the panel."""
    total, pending = 0, [pid]
    while pending:
        current = pending.pop()
        try:
            rollup = Path(f"/proc/{current}/smaps_rollup").read_text()
        except OSError:
            continue
        total += sum(int(line.split()[1]) * 1024 for line in rollup.splitlines()
                     if line.startswith(("Private_Clean:", "Private_Dirty:")))
        try:
            for task in Path(f"/proc/{current}/task").iterdir():
                pending.extend(int(child) for child in (task / "children").read_text().split())
        except OSError:
            continue
    return total


def resident_bytes() -> int:
    """What this process holds now, from `/proc`; 0 where `/proc` cannot say."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def workers_for(memory: int, beside: int, held: int, workers: int) -> int:
    """How many workers fit in `memory` beside the `beside` bytes the panel holds, when one was
    measured to hold `held`: never more than `workers`, and at least one. A measurement of nothing
    leaves `workers` as it was."""
    if held <= 0:
        return workers
    return max(1, min(workers, int((memory - beside) // (held * WORKER_MARGIN))))


def threads_in(path: str | Path | None, given: int) -> int:
    """The selections that may run at once: the number in the file at `path`, which the driver
    changes as threads free up and are taken back, or `given` where there is no such file or it
    holds no number."""
    if path is None:
        return given
    try:
        return max(1, int(Path(path).read_text()))
    except (OSError, ValueError):
        return given


def _serve(connection, inherited) -> None:
    """A worker: each half-sample it is sent, selected and sent back, or the traceback where the
    selection fails, until it is sent None or the sweep has gone.

    `inherited` holds the sweep's ends of the workers' pipes, this one's included, which the fork
    copied here. They are closed, so that a worker sees its pipe end once the sweep's end closes,
    and ends with a sweep that died."""
    for other in inherited:
        other.close()
    try:
        while (index := connection.recv()) is not None:
            try:
                reply = ("done", _run_one(index))
            except BaseException:  # sent back, so the sweep raises it with its traceback
                reply = ("failed", traceback.format_exc())
            connection.send(reply)
    except (EOFError, OSError):
        pass


class _Workers:
    """Forked workers, each running the half-samples it is sent one after another until the sweep
    ends. A half-sample goes to a worker that waits, or to a new one, so there are as many workers
    as the most selections that ran at once."""

    def __init__(self) -> None:
        self.context = mp.get_context("fork")
        #: Each worker, by the sweep's end of its pipe, the half-sample each busy one runs, and the
        #: ones that wait.
        self.processes: dict = {}
        self.busy: dict = {}
        self.idle: list = []

    def start(self, index: int) -> None:
        if not self.idle:
            ours, theirs = self.context.Pipe()
            worker = self.context.Process(target=_serve,
                                          args=(theirs, [ours, *self.processes]))
            worker.start()
            theirs.close()
            self.processes[ours] = worker
            self.idle.append(ours)
        connection = self.idle.pop()
        connection.send(index)
        self.busy[connection] = index

    def finished(self, timeout: float) -> list[tuple[int, np.ndarray, float]]:
        """The selections that end within `timeout` seconds. A failed one is raised."""
        ended = []
        for connection in wait(list(self.busy), timeout):
            index = self.busy.pop(connection)
            try:
                outcome, value = connection.recv()
            except EOFError:
                worker = self.processes[connection]
                worker.join(5)
                raise RuntimeError(f"half-sample {index}: its worker ended with code "
                                   f"{worker.exitcode} and returned nothing.") from None
            if outcome == "failed":
                raise RuntimeError(f"half-sample {index} failed in its worker:\n{value}")
            self.idle.append(connection)
            ended.append(value)
        return ended

    def measured(self, index: int) -> tuple[tuple[int, np.ndarray, float], int]:
        """Half-sample `index` run alone, and the most its worker held while it ran that no other
        process shares. The first sample is taken as the worker starts, so a short selection is
        still measured."""
        self.start(index)
        [connection] = self.busy
        pid, peak = self.processes[connection].pid, 0
        while True:
            peak = max(peak, unshared_bytes(pid))
            ended = self.finished(0.2)
            if ended:
                return ended[0], peak

    def close(self) -> None:
        """Every worker ended: those that wait are told to end, and any still running a selection,
        which only a failure leaves, are stopped."""
        for connection, worker in self.processes.items():
            if connection in self.busy:
                worker.terminate()
                continue
            try:
                connection.send(None)
            except OSError:  # it has ended already
                worker.terminate()
        for connection, worker in self.processes.items():
            worker.join()
            connection.close()


def run(select, X: np.ndarray, y: np.ndarray, names: list[str], halves: np.ndarray, *,
        make_fit_step, weights: np.ndarray | None, out_path: str | Path, workers: int,
        force: bool = False, log=None, memory: int | None = None,
        threads_file: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Selections for every half-sample, as a (2B, p) boolean matrix, and how long they took.

    `make_fit_step(weights)` builds a fitter for one half-sample from that half-sample's per-row
    weights, or from None for an unweighted design. Each selection gets a separate fitter, so no
    fitted state is shared between selections.

    `workers` selections run at once, or given `threads_file`, as many as it holds as each starts
    (`threads_in`). `memory` is the bytes the panel may hold, workers included. Given it, the sweep
    measures one selection first and runs no more workers than fit (the module docstring).
    """
    # Flushed, because a sweep runs long with its output redirected to a file, and a block buffer
    # would hold the progress lines back.
    log = log or (lambda message: print(message, flush=True))
    out_path = Path(out_path)
    total, p = halves.shape[0], len(names)
    selected = np.zeros((total, p), dtype=bool)
    done = np.zeros(total, dtype=bool)
    seconds = np.zeros(total)
    if out_path.exists() and not force:
        with np.load(out_path) as saved:
            if (list(saved["names"]) != list(names) or not np.array_equal(saved["halves"], halves)):
                raise ValueError(f"{out_path} holds selections over other candidates or other "
                                 f"half-samples; rerun with force to replace it.")
            selected, done, seconds = saved["selected"], saved["done"], saved["seconds"]
        log(f"      resuming: {int(done.sum())}/{total} half-samples already done")

    todo = [int(i) for i in np.flatnonzero(~done)]
    most = 0
    if todo:
        _STATE.update({"select": select, "X": X, "y": y, "names": names, "halves": halves,
                       "weights": weights, "make_fit_step": make_fit_step})
        started = saved = time.time()
        finished = 0
        forked = workers > 1 or threads_file is not None
        every = 5 if forked else 1

        def record(index: int, mask: np.ndarray, elapsed: float) -> None:
            nonlocal finished, saved
            selected[index], done[index], seconds[index] = mask, True, elapsed
            finished += 1
            if time.time() - saved >= SAVE_SECONDS:
                _save(out_path, selected, done, seconds, names, halves)
                saved = time.time()
            if finished % every == 0 or finished == len(todo):
                log(f"      {finished}/{len(todo)}  {int(mask.sum())} selected  "
                    f"{elapsed:6.1f}s  total {time.time() - started:7.1f}s")

        pool = _Workers() if forked else None
        try:
            if pool is None:
                most = 1
                for index in todo:
                    record(*_run_one(index))
            else:
                most = _sweep(pool, todo, record, workers=workers, memory=memory,
                              threads_file=threads_file, log=log)
        finally:
            if pool is not None:
                pool.close()
            _save(out_path, selected, done, seconds, names, halves)

    if not done.all():
        raise RuntimeError(f"{out_path}: {int((~done).sum())} half-sample(s) did not finish.")
    return selected, {"core_seconds": float(seconds.sum()), "median_seconds": float(np.median(seconds)),
                      "n_half_samples": int(total), "workers": int(most)}


def _sweep(pool: _Workers, todo: list[int], record, *, workers: int, memory: int | None,
           threads_file: str | Path | None, log) -> int:
    """Every half-sample of `todo` through `pool`'s workers, each handed to `record` as it ends, as
    many at once as the threads allow and the memory fits. Returns the most that ran at once."""
    waiting = list(todo)
    measured = None
    if memory is not None and len(waiting) > 1:
        result, held = pool.measured(waiting.pop(0))
        record(*result)
        measured = resident_bytes(), held
    shown, most, ended = None, 1, []
    while waiting or pool.busy or ended:
        given = max(1, threads_in(threads_file, workers))
        allowed = given if measured is None else workers_for(memory, *measured, given)
        if shown is None and measured is not None:
            beside, held = measured
            log(f"      one selection's worker held {held / 2**30:.2f} GB beside the panel's "
                f"{beside / 2**30:.2f} GB, so {allowed} of {given} workers fit in "
                f"{memory / 2**30:.1f} GB")
        elif shown is not None and given != shown:
            log(f"      the panel may now use {given} threads, so at most {allowed} selections "
                f"run at once")
        shown = given
        while waiting and len(pool.busy) < allowed:
            pool.start(waiting.pop(0))
        most = max(most, len(pool.busy))
        # Recorded once their workers have their next half-sample, since recording writes to disk
        # now and then (`SAVE_SECONDS`).
        for result in ended:
            record(*result)
        ended = pool.finished(RECHECK_SECONDS) if pool.busy else []
    return most


def _save(path: Path, selected, done, seconds, names, halves) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, selected=selected, done=done, seconds=seconds,
                        names=np.array(names, dtype=str), halves=halves)
    partial.replace(path)
