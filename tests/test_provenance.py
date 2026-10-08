"""`seqmaestro.provenance`: the snapshot layer of the environment record.

The tests snapshot the environment that runs them, so a snapshot that would be empty or unreadable
fails here and not months later, when someone tries to reproduce a run.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

from seqmaestro.provenance import (
    INVOCATIONS_DIR,
    JULIA_PROJECT,
    environment_python,
    git_is_dirty,
    record_call,
    record_invocation,
    snapshot_directory,
    snapshot_julia,
    write_snapshot,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    out = tmp_path_factory.mktemp("snapshot")
    return write_snapshot("SeqMaestro_Core", out), out


def test_the_snapshot_records_the_interpreter_running_it(snapshot):
    manifest, out = snapshot

    assert manifest["python"] == (out / "python-version.txt").read_text().strip()
    assert manifest["python"].startswith("3.12"), "SeqMaestro_Core is a 3.12 environment"


def test_the_freeze_is_not_empty_and_is_hashed(snapshot):
    manifest, out = snapshot
    freeze = (out / "pip-freeze.txt").read_text()

    assert "seqmaestro" in freeze
    assert manifest["n_packages"] > 10
    assert len(manifest["pip_freeze_sha256"]) == 64


def test_the_manifest_is_valid_json_on_disk(snapshot):
    manifest, out = snapshot

    assert json.loads((out / "environment.json").read_text()) == manifest


def test_a_missing_conda_is_recorded_rather_than_fatal(snapshot):
    """The snapshot must still be usable on a machine without conda on the path."""
    manifest, out = snapshot

    assert manifest["conda_export_available"] == (out / "conda-environment.yml").exists()


def test_an_environment_without_julia_says_so(snapshot):
    manifest, out = snapshot

    assert manifest["julia"] is None
    assert not (out / "julia").exists()


def test_a_julia_project_is_copied_with_the_julia_that_resolved_it(tmp_path):
    """pip does not see Julia's packages, so an environment with a Julia runtime is recorded by its
    project as well: what was asked for, what was resolved, and by which Julia."""
    project = tmp_path / "prefix" / JULIA_PROJECT
    project.mkdir(parents=True)
    (project / "Project.toml").write_text('[deps]\nExample = "7876af07-990d-54b4-ab0e-23690620f79a"\n')
    (project / "Manifest.toml").write_text('julia_version = "1.2.3"\nmanifest_format = "2.0"\n')

    found = snapshot_julia(tmp_path / "prefix", tmp_path / "out")

    assert found["julia_version"] == "1.2.3"
    for name in ("Project.toml", "Manifest.toml"):
        assert (tmp_path / "out" / "julia" / name).read_text() == (project / name).read_text()
    assert len(found["manifest_sha256"]) == 64


PYSR_PYTHON = environment_python("SeqMaestro_Method_pysr")


def test_a_snapshot_directory_is_named_for_its_environment():
    assert snapshot_directory("SeqMaestro_Core") == "core"
    assert snapshot_directory("SeqMaestro_Method_pysr") == "pysr"


@pytest.mark.crossenv
@pytest.mark.skipif(not PYSR_PYTHON.exists(), reason="the PySR environment is not installed here")
def test_the_pysr_environment_is_snapshotted_by_its_own_interpreter_with_its_julia_project(tmp_path):
    """Called from Core, the snapshot of another environment runs this module under that
    environment's interpreter from the repository root, since it imports only the standard library,
    and there it finds the Julia project the recipe placed."""
    manifest = write_snapshot("SeqMaestro_Method_pysr", tmp_path, repo_root=REPO)

    assert Path(manifest["executable"]).parent.parent.resolve() == PYSR_PYTHON.parent.parent.resolve()
    assert manifest == json.loads((tmp_path / "environment.json").read_text())
    resolved = tomllib.loads((tmp_path / "julia" / "Manifest.toml").read_text())
    assert manifest["julia"]["julia_version"] == resolved["julia_version"]
    assert "pysr" in (tmp_path / "pip-freeze.txt").read_text()


# ---------------------------------------------------------------- the invocation record


def test_the_invocation_records_the_command_and_the_code(tmp_path):
    record = record_invocation(tmp_path, arguments={"seeds": [42, 43], "pools": ["kmer"]},
                               filename="a_call.json", warn=False, call="a_call")

    assert record["command"].endswith(tuple(sys.argv[-1:]) or ("",))
    assert record["arguments"]["seeds"] == "[42, 43]", "resolved, not the raw command line"
    assert record["call"] == "a_call"
    assert json.loads((tmp_path / "a_call.json").read_text()) == record


def test_a_dirty_tree_is_recorded_not_only_warned(tmp_path):
    """A commit SHA taken against modified files does not identify the code that ran. Whether the
    tree was dirty must be saved in the record, because a terminal warning scrolls away."""
    record = record_invocation(tmp_path, filename="a_call.json", warn=False)

    assert record["git_dirty"] in (True, False, None)
    assert "git_sha" in record


def test_two_calls_writing_to_one_directory_keep_a_record_each(tmp_path):
    """Every command names its record, so two that write into one directory cannot overwrite each
    other's by leaving the name out."""
    record_invocation(tmp_path, filename="fit.json", warn=False, call="fit")
    record_invocation(tmp_path, filename="analyse.json", warn=False, call="analyse")

    assert json.loads((tmp_path / "fit.json").read_text())["call"] == "fit"
    assert json.loads((tmp_path / "analyse.json").read_text())["call"] == "analyse"
    with pytest.raises(TypeError, match="filename"):
        record_invocation(tmp_path, warn=False)


def test_the_dirty_check_answers_for_this_repository():
    assert git_is_dirty(Path(__file__).resolve().parents[1]) in (True, False)


# --------------------------------------------------------------------------------------------
# The record of one call of a run
# --------------------------------------------------------------------------------------------


def test_a_call_of_a_run_keeps_its_record_under_its_name(tmp_path):
    """A call's invocation record is named for the call, as its log and its failures file are, and
    holds the parsed arguments, so a flag left out of the command shows the value it took."""
    record_call(tmp_path, "reduction__ebm__positional", {"arms": ["ebm/positional"], "force": False})
    record = json.loads((tmp_path / INVOCATIONS_DIR / "reduction__ebm__positional.json").read_text())
    assert record["call"] == "reduction__ebm__positional"
    assert record["arguments"] == {"arms": "['ebm/positional']", "force": "False"}
