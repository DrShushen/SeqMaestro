"""Under xdist, limit each worker process to one thread.

The suite runs in parallel, and without this limit a parallel run can be slower than a sequential
one. Every worker is a separate process, whereas numpy, scikit-learn and XGBoost each default to one
thread per core, so twenty workers on twenty cores would ask for four hundred threads, and the
contention would lose more time than the parallelism saves. With each worker limited to one thread,
the parallelism comes from xdist's processes alone.

This must run before numpy is imported, because the BLAS thread pool is sized when the library
loads. A root conftest is imported before any test module, so it runs early enough, whereas a
fixture would run too late.

The limit applies only under xdist, because a sequential run is one process, in which the libraries
should use every core. `pytest -n 0` therefore stays fast for the model-fitting tests when you are
debugging one of them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Set by xdist in each worker process, and absent in a sequential run.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


#: The stand-in for PySR's search process, which answers through the real protocol with a
#: least-squares front and needs no Julia.
FAKE_SYMBOLIC_SEARCH = Path(__file__).parent / "fixtures" / "fake_symbolic_search.py"


@pytest.fixture(autouse=True)
def fake_symbolic_search(request, monkeypatch):
    """Every PySR fit in the suite searches with the stand-in, under this interpreter, except in a
    test marked `crossenv`, which runs the real search in PySR's environment.

    A real search compiles Julia for tens of seconds before it starts, whereas the tests of
    everything around the search (the choice of equation, the stages, the reveal) need only a front
    of equations. Tests that loop over every model therefore fit PySR's model in about a second.
    """
    if request.node.get_closest_marker("crossenv"):
        return
    from seqmaestro.models import symbolic
    from seqmaestro.provenance import python_override_variable

    monkeypatch.setenv(python_override_variable(symbolic.ENVIRONMENT), sys.executable)
    monkeypatch.setattr(symbolic, "SEARCH_SCRIPT", FAKE_SYMBOLIC_SEARCH)


@pytest.fixture(autouse=True)
def stand_in_discovery(request, monkeypatch):
    """Every motif discovery in the suite runs the stand-in of `tests/small_motifs.py` in place of
    STREME and Tomtom, except in a test marked `crossenv`, which runs them in the MEME Suite's
    environment. Everything else in a discovery is the real code: the rows compared, the
    discovery's key, the store and its lock. Because the environment's interpreter is pointed at
    this one, these tests also run on a machine without the MEME Suite.
    """
    if request.node.get_closest_marker("crossenv"):
        return
    from seqmaestro.motifs import discovery
    from seqmaestro.provenance import python_override_variable
    from tests import small_motifs

    monkeypatch.setenv(python_override_variable(discovery.MEME_ENVIRONMENT), sys.executable)
    monkeypatch.setattr(discovery, "_streme", small_motifs.stand_in_streme)
    monkeypatch.setattr(discovery, "_tomtom", small_motifs.stand_in_tomtom)
    monkeypatch.setattr(discovery, "versions", small_motifs.stand_in_versions)


@pytest.fixture
def runners(monkeypatch):
    """The stages of a run, called in this process: for each stage, the module whose `main` its
    script calls. The environment snapshots are left out, because they shell out to conda and pip,
    which is slow and not under test."""
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
