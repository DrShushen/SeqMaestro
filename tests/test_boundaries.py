"""Checks of the two architectural rules that the repository's structure depends on.

Either rule could break without anything else failing:

1. SeqMaestro must not import a vendored method, and a vendored method must not import SeqMaestro.
   Each method is meant to become a standalone library (its changes go back upstream) and lives in
   a separate conda environment, so the two sides exchange files on disk (a written feature
   matrix, a serialised tree, a saved model). An `import` across that line would couple two
   independent environments, and the coupling would go unnoticed on a machine where both happen
   to be installed.

2. A test run from here must not collect tests under `methods/`. With `testpaths` set but
   `norecursedirs` unset, pytest can fall back to a recursive search and collect DEFT's suite
   under this interpreter, where most of it passes and a few tests fail or error. A mostly
   passing result from the wrong environment is worse than a failing one, because nobody
   investigates it.

The checks fail if they find nothing to check, so they cannot pass vacuously.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQMAESTRO = REPO_ROOT / "seqmaestro"
METHODS = REPO_ROOT / "methods"


def vendored_method_import_names() -> dict[str, str]:
    """Every name by which a vendored method could be imported, mapped to the method it belongs to.

    A method may contribute more than one name, because the distribution name need not match the
    importable package (a distribution `deft` could ship a top-level package called `src`). Both
    the flat and the src layout are searched, so this works whichever layout a method uses.
    """
    names: dict[str, str] = {}
    for method_dir in sorted(p for p in METHODS.iterdir() if p.is_dir()):
        method = method_dir.name

        # Packages sitting directly in the method directory (flat layout), and inside a `src/`
        # container (src layout).
        for parent in (method_dir, method_dir / "src"):
            if not parent.is_dir():
                continue
            for child in parent.iterdir():
                if child.is_dir() and (child / "__init__.py").exists() and child.name != "tests":
                    names[child.name] = method

        pyproject = method_dir / "pyproject.toml"
        if pyproject.exists():
            declared = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
            if declared:
                names[declared] = method

    return names


def imported_names(path: Path):
    """Yield `(lineno, top_level_name)` for every absolute import in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]


# --------------------------------------------------------------- rule 1: the import direction


def test_the_discovery_itself_works():
    """Keeps the tests below from passing merely because they found nothing to forbid.

    The assertion is made per method and does not name any package, so a method that changes its
    layout does not fail here. Only a method that contributes no name to forbid fails, because that
    is the case that would make the tests below vacuous.
    """
    found = vendored_method_import_names()
    methods = {path.name for path in METHODS.iterdir() if path.is_dir()}

    assert methods, "no vendored methods found under methods/"
    assert set(found.values()) == methods, (
        f"every vendored method must contribute at least one importable name to forbid; "
        f"found {found} for methods {sorted(methods)}"
    )
    assert list(SEQMAESTRO.rglob("*.py")), "no modules found under seqmaestro/"


def test_seqmaestro_does_not_import_a_vendored_method():
    forbidden = vendored_method_import_names()
    offenders = [
        f"  {path.relative_to(REPO_ROOT)}:{lineno} imports `{name}` (from methods/{forbidden[name]})"
        for path in sorted(SEQMAESTRO.rglob("*.py"))
        for lineno, name in imported_names(path)
        if name in forbidden
    ]

    assert not offenders, (
        "seqmaestro must not import a vendored method; the two sides live in separate conda "
        "environments and exchange files, not objects:\n" + "\n".join(offenders)
    )


def test_no_vendored_method_imports_seqmaestro():
    """The other direction: a method's code and tests stand alone, so that the copy can go back
    upstream and its suite can run in its environment, where `seqmaestro` is not installed."""
    modules = sorted(path for path in METHODS.rglob("*.py") if "__pycache__" not in path.parts)
    assert modules, "no modules found under methods/"
    offenders = [
        f"  {path.relative_to(REPO_ROOT)}:{lineno} imports `{name}`"
        for path in modules
        for lineno, name in imported_names(path)
        if name == SEQMAESTRO.name
    ]

    assert not offenders, "a vendored method must not import seqmaestro:\n" + "\n".join(offenders)


def test_tests_do_not_import_a_vendored_method_either():
    """A test across the boundary exercises the file contract instead, by running the interpreter
    of the method's environment in a subprocess under the `crossenv` marker. Importing the method
    here would work only on a machine where both environments happen to be installed, and would
    hide the coupling."""
    forbidden = vendored_method_import_names()
    offenders = [
        f"  {path.relative_to(REPO_ROOT)}:{lineno} imports `{name}`"
        for path in sorted(Path(__file__).parent.rglob("*.py"))
        for lineno, name in imported_names(path)
        if name in forbidden
    ]

    assert not offenders, "\n".join(offenders)


# ------------------------------------------------------- rule 2: the collection stays scoped


@pytest.fixture(scope="module")
def pytest_config() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]


def test_collection_does_not_recurse_into_methods(pytest_config):
    """`testpaths` alone is not enough, because when it matches nothing, pytest warns and searches
    recursively from the current directory, which would run DEFT's suite under the wrong
    interpreter. `norecursedirs` prevents that."""
    assert "methods" in pytest_config["norecursedirs"]


def test_the_run_is_pointed_at_this_directory(pytest_config):
    assert pytest_config["testpaths"] == ["tests"]


def test_pytest_defaults_were_not_dropped_when_norecursedirs_was_set(pytest_config):
    """Setting `norecursedirs` replaces pytest's defaults instead of extending them, so the
    defaults must be listed again."""
    assert {"*.egg", ".*", "build", "dist", "node_modules", "venv"} <= set(pytest_config["norecursedirs"])
