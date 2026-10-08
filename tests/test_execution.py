"""The threads a process's fits may use (`seqmaestro.execution`).

A driver that runs processes side by side gives each a number of threads, and the fits inside
follow it. The tests check that the number reaches the libraries, that a model built with its own
number keeps it, and that the number changes nothing a fit computes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from seqmaestro.execution import (
    LIBRARY_VARIABLES,
    MEMORY_SHARE,
    MEMORY_VARIABLE,
    THREADS_FILE_VARIABLE,
    THREADS_VARIABLE,
    Machine,
    allowed_memory,
    allowed_threads,
    budget,
    jobs,
    probe,
    thread_environment,
    threads_file,
    usage_of,
)
from seqmaestro.models import make_model
from seqmaestro.provenance import record_invocation
from seqmaestro.target import BINARY, CONTINUOUS


def test_a_process_given_no_number_has_none(monkeypatch):
    monkeypatch.delenv(THREADS_VARIABLE, raising=False)
    assert allowed_threads() is None
    assert jobs({}) == {}


def test_a_process_given_a_number_has_it(monkeypatch):
    monkeypatch.setenv(THREADS_VARIABLE, "3")
    assert allowed_threads() == 3
    assert jobs({}) == {"n_jobs": 3}


def test_a_model_built_with_its_own_number_keeps_it(monkeypatch):
    """Tier 1's fits are built with one job each, whatever the process was given."""
    monkeypatch.setenv(THREADS_VARIABLE, "3")
    assert jobs({"n_jobs": 1}) == {}


@pytest.mark.parametrize("value", ["0", "-2", "two", "1.5"])
def test_a_number_that_is_not_one_is_refused(monkeypatch, value):
    monkeypatch.setenv(THREADS_VARIABLE, value)
    with pytest.raises(ValueError, match=THREADS_VARIABLE):
        allowed_threads()


def test_a_process_is_told_where_to_find_the_threads_it_may_use_now(monkeypatch, tmp_path):
    monkeypatch.delenv(THREADS_FILE_VARIABLE, raising=False)
    assert threads_file() is None
    monkeypatch.setenv(THREADS_FILE_VARIABLE, str(tmp_path / "threads"))
    assert threads_file() == tmp_path / "threads"


def test_a_process_is_told_the_memory_it_may_hold(monkeypatch):
    monkeypatch.delenv(MEMORY_VARIABLE, raising=False)
    assert allowed_memory() is None
    monkeypatch.setenv(MEMORY_VARIABLE, str(3 * 2**30))
    assert allowed_memory() == 3 * 2**30
    monkeypatch.setenv(MEMORY_VARIABLE, "3G")
    with pytest.raises(ValueError, match=MEMORY_VARIABLE):
        allowed_memory()


def test_a_process_is_started_with_every_librarys_variable():
    """OpenMP and the BLAS libraries read their own variables as they load, so a process started
    with only SeqMaestro's would run its numeric libraries on every core."""
    assert thread_environment(2) == {THREADS_VARIABLE: "2", **dict.fromkeys(LIBRARY_VARIABLES, "2")}
    with pytest.raises(ValueError, match="at least one thread"):
        thread_environment(0)


def pool(n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 4, (n, 12)).astype(float), columns=[f"f{i}" for i in range(12)])
    signal = X["f0"] - X["f3"] + rng.normal(0, 1, n)
    return X, signal.to_numpy()


@pytest.mark.parametrize(("kind", "name"), [(BINARY, "ebm"), (BINARY, "xgboost"),
                                            (CONTINUOUS, "ebm"), (CONTINUOUS, "xgboost")])
def test_the_number_reaches_the_library_and_changes_nothing_it_fits(monkeypatch, kind, name):
    X, signal = pool()
    y = (signal > 0).astype(int) if kind == BINARY else signal
    weighting = {"class_weight": "balanced"} if kind == BINARY else {}
    fits = {}
    for threads in ("1", "3"):
        monkeypatch.setenv(THREADS_VARIABLE, threads)
        model = make_model(name, target_kind=kind, seed=0, **weighting).fit(X, y)
        assert model._model.get_params()["n_jobs"] == int(threads)
        assert "n_jobs" not in model.describe(), "how a fit ran is no part of the model"
        fits[threads] = model
    np.testing.assert_array_equal(fits["1"].predict(X), fits["3"].predict(X))
    assert fits["1"].importances() == fits["3"].importances()


def test_an_invocation_records_the_threads_it_was_given(tmp_path, monkeypatch):
    monkeypatch.setenv(THREADS_VARIABLE, "4")
    record_invocation(tmp_path, filename="invocation_test.json", warn=False)
    assert json.loads((tmp_path / "invocation_test.json").read_text())["threads"] == 4
    monkeypatch.delenv(THREADS_VARIABLE)
    record_invocation(tmp_path, filename="invocation_test.json", warn=False)
    assert json.loads((tmp_path / "invocation_test.json").read_text())["threads"] is None


# ------------------------------------------------------------------------------ the machine

def control_groups(tmp_path, cpu_max: str | None, memory_max: str | None, current: int = 0):
    """A cgroup v2 tree two levels deep, with the limits set on the outer group, and the file that
    places this process in the inner one."""
    root = tmp_path / "cgroup"
    inner = root / "user.slice" / "run.scope"
    inner.mkdir(parents=True)
    for directory in (root, root / "user.slice", inner):
        (directory / "cpu.max").write_text("max 100000\n")
        (directory / "memory.max").write_text("max\n")
        (directory / "memory.current").write_text(f"{current}\n")
    if cpu_max:
        (root / "user.slice" / "cpu.max").write_text(cpu_max)
    if memory_max:
        (root / "user.slice" / "memory.max").write_text(memory_max)
    membership = tmp_path / "cgroup_of_this_process"
    membership.write_text("0::/user.slice/run.scope\n")
    return root, membership


def test_a_probe_finds_the_limits_a_control_group_above_it_sets(tmp_path):
    root, membership = control_groups(tmp_path, "200000 100000\n", f"{8 * 2**30}\n",
                                      current=2 * 2**30)
    machine = probe(root, membership)
    assert machine.cgroup_cpus == 2.0 and machine.cpus <= 2
    assert machine.cgroup_memory == 8 * 2**30
    assert machine.memory_total <= 8 * 2**30 and machine.memory_available <= 6 * 2**30


def test_a_probe_without_limits_takes_the_machine(tmp_path):
    root, membership = control_groups(tmp_path, None, None)
    machine = probe(root, membership)
    assert machine.cgroup_cpus is None and machine.cgroup_memory is None
    assert machine.cpus == len(os.sched_getaffinity(0)) and machine.memory_available > 0


def a_machine(cpus=16, total=64, available=40) -> Machine:
    return Machine(cpus=cpus, cores=cpus // 2, memory_total=total * 2**30,
                   memory_available=available * 2**30, load=0.0, cgroup_cpus=None,
                   cgroup_memory=None)


def test_a_budget_takes_every_cpu_and_a_share_of_the_memory_lowered_to_its_caps():
    taken = budget(a_machine(), model_calls=20)
    assert taken.cpus == 16 and taken.memory == int(40 * 2**30 * MEMORY_SHARE)
    assert taken.reserve == max(1 << 30, int(64 * 2**30 * 0.05))
    capped = budget(a_machine(), model_calls=4, max_cpus=6, max_memory=10 * 2**30)
    assert (capped.cpus, capped.memory, capped.model_calls) == (6, 10 * 2**30, 4)
    with pytest.raises(ValueError, match="at least one process"):
        budget(a_machine(), model_calls=0)


def test_a_process_is_measured_with_what_it_started(tmp_path):
    """The memory a process's workers hold is the process's to answer for, so a measurement covers
    everything below it."""
    child = subprocess.Popen([sys.executable, "-c", "x = bytearray(80 * 2**20); "
                              "open(__import__('sys').argv[1], 'w').close(); "
                              "__import__('time').sleep(30)", str(tmp_path / "ready")])
    try:
        while not (tmp_path / "ready").exists():
            time.sleep(0.05)
        usage = usage_of([child.pid, os.getpid(), 2**22 + 12345])
        assert usage[child.pid].memory >= 80 * 2**20
        assert usage[os.getpid()].memory >= usage[child.pid].memory, "the child is below us"
        assert 2**22 + 12345 not in usage, "a process that has gone is left out"
    finally:
        child.kill()
        child.wait()
