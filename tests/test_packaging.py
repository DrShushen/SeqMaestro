"""The package is installed, under the name it claims.

The main guard is against a distribution whose name differs from its importable top-level package,
for example a distribution named `deft` whose top-level package is `src`, so `import deft` fails.
`pyproject.toml` accepts that without complaint, and the mistake stays invisible until another
project tries to depend on the package.
"""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import seqmaestro

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def declared() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]


def test_the_distribution_name_is_the_import_name():
    top_level = {
        (file.parts[0] if len(file.parts) > 1 else file.stem)
        for file in metadata.files("seqmaestro") or []
        if file.suffix == ".py" and not str(file).startswith("..")
    }

    assert "seqmaestro" in top_level, (
        f"`pip install seqmaestro` must give you `import seqmaestro`; top-level names are {top_level}"
    )


def test_the_installed_package_is_this_working_tree():
    """The package must be an editable install of this working tree. Otherwise the suite would
    silently test a stale copy in site-packages."""
    assert Path(seqmaestro.__file__).resolve().parent == REPO_ROOT / "seqmaestro"


def test_the_version_is_not_two_versions(declared):
    assert seqmaestro.__version__ == declared["version"]


def test_the_base_install_stays_light(declared):
    """Base dependencies must stay installable into a method's own environment, so they include
    no boosting library and no LLM client. Those live behind the `models` and `llm` extras.

    `scipy` is listed because the harness imports it directly for the Hungarian assignment and
    rank transforms, and `threadpoolctl` because the reliability panel imports it to hold a worker
    to one thread. Neither adds weight, since scikit-learn already depends on both, but every
    direct import is declared. Anything heavier belongs in an extra, so changing this set is a
    decision to argue for and never a quiet widening of the assertion."""
    base = {requirement.split(">")[0].split("=")[0].split("[")[0].strip() for requirement in declared["dependencies"]}

    assert base == {"numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl"}
