"""Under xdist, limit each worker process to one thread.

The suite runs in parallel, and without this limit a parallel run is slower than a sequential one.
On a 20-core machine with 353 tests:

    sequential                          166 s
    -n 8                                112 s
    -n 16                               116 s
    -n auto (20)                        170 s   <- worse than sequential
    -n 16, one BLAS thread per worker    21 s

Every worker is a separate process, and numpy, scikit-learn and XGBoost each default to one
thread per core. Twenty workers therefore ask for four hundred threads on twenty cores, and the
contention costs more than the parallelism gains. With each worker limited to one thread, the
parallelism comes from xdist's processes.

This must run before numpy is imported, because the BLAS thread pool is sized when the library
loads. A root conftest is imported before any test module, which is early enough. A fixture would
run too late.

The limit applies only under xdist. A sequential run is one process, where the libraries should use
every core, so `pytest -n 0` stays fast for the model-fitting tests when you are debugging one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Set by xdist in each worker process; absent in a sequential run.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


#: The stand-in for PySR's search process: the real protocol, a least-squares front, no Julia.
FAKE_SYMBOLIC_SEARCH = Path(__file__).parent / "fixtures" / "fake_symbolic_search.py"


@pytest.fixture(autouse=True)
def fake_symbolic_search(request, monkeypatch):
    """Every PySR fit in the suite searches with the stand-in, under this interpreter, except in a
    test marked `crossenv`, which runs the real search in PySR's environment.

    A real search compiles Julia for tens of seconds before it starts, and the tests of everything
    around the search (the choice of equation, the stages, the reveal) need only a front. Tests that
    loop over every model therefore fit PySR's model in about a second.
    """
    if request.node.get_closest_marker("crossenv"):
        return
    from seqmaestro.models import symbolic
    from seqmaestro.provenance import python_override_variable

    monkeypatch.setenv(python_override_variable(symbolic.ENVIRONMENT), sys.executable)
    monkeypatch.setattr(symbolic, "SEARCH_SCRIPT", FAKE_SYMBOLIC_SEARCH)


@pytest.fixture
def runners(monkeypatch):
    """The stages of a run in this process: for each, the module whose `main` its script calls.
    The environment snapshots shell out to conda and pip, which is slow and is not under test, so
    they are left out."""
    from types import SimpleNamespace

    from seqmaestro.pipeline import (
        baseline,
        config,
        driver,
        records,
        reduction,
        reliability,
        report,
        reveal,
        stage2,
    )

    monkeypatch.setattr(records, "write_snapshot", lambda *args, **kwargs: None)
    return SimpleNamespace(config=config, driver=driver, baseline=baseline, arm=stage2,
                           reduction=reduction, reliability=reliability, reveal=reveal,
                           report=report, records=records)
