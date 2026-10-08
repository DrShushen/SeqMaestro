"""`seqmaestro.models.symbolic`: PySR's search in its own process, and the model Core makes of it.

Most tests search with the stand-in (`tests/fixtures/fake_symbolic_search.py`), which speaks the
real protocol and returns a front without Julia. The `crossenv` tests at the end run the real
search in `SeqMaestro_Method_pysr`.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from seqmaestro.execution import THREADS_VARIABLE
from seqmaestro.models import make_model, symbolic
from seqmaestro.models.base import filled_rows
from seqmaestro.models.symbolic import SymbolicClassification, SymbolicRegression
from seqmaestro.provenance import environment_python, python_override_variable
from seqmaestro.reduction.arm import rebuild_model
from seqmaestro.target import BINARY, CONTINUOUS

FAKE = Path(__file__).parent / "fixtures" / "fake_symbolic_search.py"
PYSR_PYTHON = environment_python(symbolic.ENVIRONMENT)
#: A placeholder name of the search's, anywhere in a text a reader could see.
PLACEHOLDER = re.compile(r"\bx\d+\b")


def features(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"gc_content": rng.uniform(0.2, 0.8, n),
                         "motif_count": rng.integers(0, 6, n).astype(float),
                         "loop_length": rng.uniform(1.0, 9.0, n)})


def planted(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X = features(n, seed)
    noise = np.random.default_rng(seed + 1).normal(0, 0.05, n)
    return X, 3.0 * X["gc_content"].to_numpy() - 0.5 * X["motif_count"].to_numpy() + noise


def const(c):
    return ["const", c]


def col(i):
    return ["col", i]


@pytest.fixture
def searches(tmp_path, monkeypatch):
    """A copy of the stand-in that can be given a front, and that keeps the last request."""
    home = tmp_path / "search"
    home.mkdir()
    shutil.copy(FAKE, home / FAKE.name)
    (home / "keep_requests").touch()
    monkeypatch.setattr(symbolic, "SEARCH_SCRIPT", home / FAKE.name)

    def give(trees=None, losses=None, perturb=0.0):
        if trees is not None:
            (home / "front.json").write_text(json.dumps({"trees": trees, "losses": losses,
                                                         "perturb": perturb}))
        return home

    yield give
    symbolic.close_search_processes()


def last_request(home: Path) -> dict:
    with np.load(home / "last_request.npz") as saved:
        return {key: saved[key] for key in saved.files}


# ------------------------------------------------------------------------------ the choice

def test_a_fit_searches_most_rows_and_keeps_an_equation_over_the_planted_features():
    X, y = planted()
    model = SymbolicRegression(seed=3).fit(X, y)
    fitted = model.describe()["fitted"]

    assert fitted["n_searched"] + fitted["n_set_aside"] == len(y)
    assert fitted["n_set_aside"] == round(0.15 * len(y))
    assert set(model.importances()) == {"gc_content", "motif_count"}
    assert fitted["versions"] == {"pysr": "fake", "symbolic_regression_jl": "fake", "julia": "fake"}


def test_the_equation_is_chosen_on_the_rows_set_aside_not_by_the_search_loss(searches):
    X = features()
    y = 3.0 * X["gc_content"].to_numpy()
    trees = [const(0.0), ["mul", const(3.0), col(0)],
             ["add", ["mul", const(3.0), col(0)], ["mul", const(2.0), col(2)]]]
    # The search's losses favour the longest equation, which is wrong on every row.
    searches(trees, losses=[1.0, 0.5, 0.01])
    model = SymbolicRegression(seed=0).fit(X, y)

    assert model.describe()["fitted"]["complexity"] == 5
    assert model.equation_text() == "3 * gc_content"
    losses = [entry["set_aside_loss"] for entry in model.front()]
    assert losses[1] == min(losses)


def test_a_tie_goes_to_the_simpler_equation(searches):
    X = features()
    searches([col(1), ["mul", const(1.0), col(1)]])
    model = SymbolicRegression(seed=0).fit(X, X["motif_count"].to_numpy())

    chosen = [entry for entry in model.front() if entry["chosen"]]
    assert [entry["complexity"] for entry in chosen] == [1]
    assert model.front()[0]["set_aside_loss"] == model.front()[1]["set_aside_loss"]


def test_an_equation_undefined_on_a_row_set_aside_is_never_chosen(searches):
    X = features()
    X.loc[::5, "loop_length"] = 0.0
    y = np.where(X["loop_length"] > 0, X["gc_content"] / X["loop_length"].replace(0, 1), 0.0)
    searches([const(float(y.mean())), ["mul", col(0), ["pow", col(2), -1]]])
    model = SymbolicRegression(seed=0).fit(X, y)

    assert model.front()[1]["set_aside_loss"] is None
    assert model.describe()["fitted"]["complexity"] == 1
    assert model.importances() == {}


def test_rows_where_the_chosen_equation_is_undefined_get_the_constant(searches):
    X = features()
    y = (X["gc_content"] / X["loop_length"]).to_numpy()
    searches([const(0.0), ["mul", col(0), ["pow", col(2), -1]]])
    weights = np.linspace(0.5, 1.5, len(y))
    model = SymbolicRegression(seed=0).fit(X, y, sample_weight=weights)

    new = features(10, seed=9)
    new.loc[[2, 7], "loop_length"] = 0.0
    predicted = model.predict(new)
    assert model.filled(new).tolist() == [i in (2, 7) for i in range(10)]
    assert filled_rows(model, new) == 2
    assert predicted[2] == predicted[7] == pytest.approx(np.average(y, weights=weights))
    assert np.all(np.isfinite(predicted))


def test_a_fit_stops_when_core_and_pysr_evaluate_an_equation_differently(searches):
    X, y = planted()
    searches([const(0.0), ["mul", const(3.0), col(0)]], perturb=1e-6)
    with pytest.raises(RuntimeError, match="export is wrong"):
        SymbolicRegression(seed=0).fit(X, y)


# ------------------------------------------------------------------------------ what reaches the search

def test_weights_and_the_rows_searched_reach_the_search(searches):
    home = searches()
    X, y = planted()
    weights = np.random.default_rng(4).uniform(0.2, 2.0, len(y))
    model = SymbolicRegression(seed=0).fit(X, y, sample_weight=weights)
    sent = last_request(home)

    assert len(sent["y"]) == model.describe()["fitted"]["n_searched"]
    rows = [int(np.flatnonzero(np.all(X.to_numpy() == row, axis=1))[0]) for row in sent["X"]]
    normalised = weights * len(weights) / weights.sum()
    np.testing.assert_allclose(sent["weights"], normalised[rows])
    assert json.loads(str(sent["options"]))["niterations"] == 200


def test_a_binary_target_is_searched_as_minus_one_and_one_and_predicted_as_a_probability(searches):
    home = searches()
    X, _ = planted()
    y = (X["gc_content"] > 0.45).to_numpy().astype(int)
    model = SymbolicClassification(seed=0).fit(X, y)
    sent = last_request(home)

    assert set(np.unique(sent["y"])) == {-1.0, 1.0}
    assert len(np.unique(sent["weights"])) == 2, "balanced class weights, as for every model"
    assert json.loads(str(sent["options"]))["elementwise_loss"] == "LogitMarginLoss()"
    predicted = model.predict(X)
    assert np.all((predicted > 0) & (predicted < 1))
    np.testing.assert_allclose(predicted, expit(model.output(X)))


def test_threads_reach_the_search_process(searches):
    home = searches()
    X, y = planted()
    SymbolicRegression(seed=0, threads=1).fit(X, y)
    assert str(last_request(home)["threads"]) == "1"
    SymbolicRegression(seed=0).fit(X, y)
    assert str(last_request(home)["threads"]) == "auto"


def test_a_search_takes_the_threads_its_process_was_given(searches, monkeypatch):
    """A model built with no thread count searches on the threads the process was given, and one
    built with a count keeps it, as Tier 1's one-core fits do."""
    home = searches()
    X, y = planted()
    monkeypatch.setenv(THREADS_VARIABLE, "2")
    SymbolicRegression(seed=0).fit(X, y)
    assert str(last_request(home)["threads"]) == "2"
    SymbolicRegression(seed=0, threads=1).fit(X, y)
    assert str(last_request(home)["threads"]) == "1"


def _fit_in_child(X: pd.DataFrame, y: np.ndarray) -> tuple[list[int], str]:
    SymbolicRegression(seed=0).fit(X, y)
    return ([key[0] for key in symbolic._PROCESSES],
            str(symbolic.search_process(None).workdir))


def test_a_forked_child_starts_its_own_search_process_which_cleans_up_after_it():
    """A forked worker exits without running exit handlers, so its search process removes its own
    working directory when the worker's end closes the request pipe."""
    X, y = planted()
    SymbolicRegression(seed=0).fit(X, y)
    parent = [key[0] for key in symbolic._PROCESSES]

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork")) as pool:
        in_child, workdir = pool.submit(_fit_in_child, X, y).result()

    assert set(parent) < set(in_child), "the child kept its parent's entry and added its own"
    assert [key[0] for key in symbolic._PROCESSES] == parent
    SymbolicRegression(seed=1).fit(X, y)  # the parent's process still serves
    for _ in range(100):
        if not Path(workdir).exists():
            break
        time.sleep(0.1)
    assert not Path(workdir).exists()


def test_a_search_process_keeps_its_temporary_files_in_its_own_directory(searches):
    home = searches()
    X, y = planted()
    SymbolicRegression(seed=0).fit(X, y)
    assert str(last_request(home)["tmpdir"]) == str(symbolic.search_process(None).workdir)
    assert load_search_module().WORKDIR_PREFIX == symbolic.WORKDIR_PREFIX


@pytest.mark.parametrize(("name", "removed"), [("seqmaestro-pysr-abc", True), ("elsewhere", False)])
def test_only_a_working_directory_the_caller_made_is_removed(tmp_path, name, removed):
    workdir = tmp_path / name
    workdir.mkdir()
    requests, closed = os.pipe()
    os.close(closed)
    replies_read, replies = os.pipe()
    load_search_module().run([str(requests), str(replies), str(workdir)], search=None)
    os.close(replies_read)
    assert workdir.exists() != removed


# ------------------------------------------------------------------------------ the fitted model

def test_importance_and_direction_come_from_the_equation():
    X, y = planted()
    model = SymbolicRegression(seed=0).fit(X, y)

    importances = model.importances()
    assert importances["gc_content"] > 0 and importances["motif_count"] > 0
    assert model.directions(X) == {"gc_content": "increases_target",
                                   "motif_count": "decreases_target"}


def test_a_saved_model_predicts_without_any_search_process(monkeypatch):
    X, y = planted()
    model = SymbolicRegression(seed=0).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    symbolic.close_search_processes()
    monkeypatch.setenv(python_override_variable(symbolic.ENVIRONMENT), "/nonexistent/python")

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.importances() == model.importances()


@pytest.mark.parametrize(("kind", "options"), [(CONTINUOUS, {}), (BINARY, {"class_weight": "balanced"})])
def test_the_description_rebuilds_the_same_model(kind, options):
    X, y = planted()
    if kind == BINARY:
        y = (y > np.median(y)).astype(int)
    model = make_model("pysr", target_kind=kind, seed=7, **options).fit(X, y)

    rebuilt = rebuild_model(model.describe(), 7, kind)
    assert rebuilt.describe()["fitted"] is None
    assert rebuild_model(model.describe(), 7, kind, {"threads": 1}).threads == 1


def test_no_placeholder_name_reaches_what_a_reader_sees():
    X, y = planted()
    model = SymbolicRegression(seed=0).fit(X, y)
    shown = json.dumps([model.describe(), model.feedback(X, y), model.equation_text(),
                        model.front()])

    assert not PLACEHOLDER.search(shown), PLACEHOLDER.findall(shown)
    assert "gc_content" in model.equation_text()


def test_feedback_shows_the_equation_and_the_front():
    X, y = planted()
    model = SymbolicRegression(seed=0).fit(X, y)
    feedback = model.feedback(X, y)

    assert feedback["equation"] == model.equation_text()
    assert [set(entry) for entry in feedback["front"]] == [
        {"complexity", "set_aside_loss", "equation"}] * len(model.front())
    assert feedback["importance_kind"] == "mean_abs_shapley"


# ------------------------------------------------------------------------------ the search process

def load_search_module():
    spec = importlib.util.spec_from_file_location(
        "symbolic_search", Path(symbolic.__file__).with_name("symbolic_search.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_search_process_imports_numpy_sympy_pysr_and_the_standard_library_only():
    """It runs in PySR's environment, which has no `seqmaestro`."""
    tree = ast.parse(Path(symbolic.__file__).with_name("symbolic_search.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"numpy", "sympy", "pysr", "__future__"} | set(sys.stdlib_module_names)


def test_julia_is_confined_to_the_environments_own_prefix(tmp_path):
    search = load_search_module()
    environ = {}
    search.confine_julia(tmp_path, environ)
    assert all(value == "yes" or value.startswith(str(tmp_path)) for value in environ.values())
    assert set(environ) == {"JULIA_DEPOT_PATH", "PYTHON_JULIAPKG_PROJECT", "PYTHON_JULIAPKG_EXE",
                            "PYTHON_JULIAPKG_OFFLINE"}

    with pytest.raises(RuntimeError, match="keeps Julia inside"):
        search.confine_julia(tmp_path, {"JULIA_DEPOT_PATH": str(Path.home() / ".julia")})


def test_a_failed_search_is_raised_with_its_error(searches):
    searches([["sin", col(0)]])
    X, y = planted()
    with pytest.raises(RuntimeError, match="PySR search failed"):
        SymbolicRegression(seed=0).fit(X, y)


# ------------------------------------------------------------------------------ the real search

def julia_homes() -> dict[str, object]:
    """What lies under the home directory's Julia locations, to show that a search adds nothing."""
    home = Path.home()
    state = {}
    for name in (".julia", ".juliaup"):
        root = home / name
        state[name] = (sorted((str(path.relative_to(root)), path.stat().st_mtime_ns)
                              for path in root.rglob("*")) if root.exists() else None)
    state[".pyjuliapkg.json"] = (home / ".pyjuliapkg.json").exists()
    return state


@pytest.mark.crossenv
@pytest.mark.skipif(not PYSR_PYTHON.exists(), reason="the PySR environment is not installed here")
def test_a_real_search_recovers_a_planted_equation_and_writes_nothing_home():
    before = julia_homes()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0.5, 2.0, (400, 4)), columns=["alpha", "beta", "gamma_", "delta"])
    y = 2.5 * X["alpha"] - 1.5 * X["beta"] + rng.normal(0, 0.01, 400)
    try:
        model = SymbolicRegression(seed=0, niterations=20).fit(X, y)
        # A second search in the same process, which Julia's hold on the standard streams once
        # stopped.
        again = SymbolicRegression(seed=1, niterations=20).fit(X, y)
    finally:
        symbolic.close_search_processes()

    for fitted in (model, again):
        importances = fitted.importances()
        assert set(sorted(importances, key=importances.get)[-2:]) == {"alpha", "beta"}
    new = pd.DataFrame(rng.uniform(0.5, 2.0, (200, 4)), columns=X.columns)
    truth = 2.5 * new["alpha"] - 1.5 * new["beta"]
    assert np.corrcoef(model.predict(new), truth)[0, 1] > 0.999
    versions = model.describe()["fitted"]["versions"]
    assert versions["pysr"] and versions["julia"] and versions["symbolic_regression_jl"]
    assert julia_homes() == before


@pytest.mark.crossenv
@pytest.mark.skipif(not PYSR_PYTHON.exists(), reason="the PySR environment is not installed here")
def test_the_search_process_refuses_julia_variables_pointing_elsewhere(tmp_path):
    result = subprocess.run(
        [str(PYSR_PYTHON), str(Path(symbolic.__file__).with_name("symbolic_search.py")), "0", "1"],
        env={"PATH": "/usr/bin:/bin", "JULIA_DEPOT_PATH": str(tmp_path / "elsewhere")},
        input="", capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode != 0
    assert "keeps Julia inside" in result.stderr
    assert not (tmp_path / "elsewhere").exists()


EXPORT_CHECK = """
import json, sys, importlib.util
import sympy
spec = importlib.util.spec_from_file_location("search", sys.argv[1])
search = importlib.util.module_from_spec(spec); spec.loader.exec_module(search)
out = []
for text in json.loads(sys.argv[2]):
    expression = sympy.sympify(text, evaluate=False)
    try:
        out.append(search.tree_of(expression, 3))
    except search.OutsideVocabulary as error:
        out.append(str(error))
print(json.dumps(out))
"""


@pytest.mark.crossenv
@pytest.mark.skipif(not PYSR_PYTHON.exists(), reason="the PySR environment is not installed here")
def test_sympy_forms_are_exported_as_they_come():
    """PySR parses its equations unevaluated, so a subtraction is a sum with a product of -1 and a
    division a product with a power of -1; the tree keeps both as they are."""
    texts = ["x0 - 0.6221292", "x1 / (x0 + 2.5)", "exp(x2) * log(x0)", "sin(x0)", "x5"]
    result = subprocess.run(
        [str(PYSR_PYTHON), "-c", EXPORT_CHECK, str(Path(symbolic.__file__).with_name(
            "symbolic_search.py")), json.dumps(texts)],
        capture_output=True, text=True, timeout=120, check=True)
    trees = json.loads(result.stdout)

    assert trees[0] == ["add", ["col", 0], ["mul", ["const", -1.0], ["const", 0.6221292]]]
    assert trees[1] == ["mul", ["col", 1], ["pow", ["add", ["col", 0], ["const", 2.5]], -1.0]]
    assert trees[2] == ["mul", ["exp", ["col", 2]], ["log", ["col", 0]]]
    assert trees[3] == "a node of kind sin"
    assert trees[4] == "a symbol that is not one of the searched columns"
