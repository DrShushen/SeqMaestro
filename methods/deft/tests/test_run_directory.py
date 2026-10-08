"""Run directories must not be silently reused (`check_run_directory`).

`intermediate_tree.pkl` is rewritten at the start of every node, and the tree, the provenance
record and the token report all land in the same directory. Two runs sharing one therefore
interleave and corrupt each other's record — and Hydra's timestamped directories resolve only to
the second, so a scripted multi-seed sweep really can collide.

All offline.
"""

from __future__ import annotations

import logging

import pytest

from deft.cli.run_deft_tree import check_run_directory


def test_an_empty_directory_is_fine(tmp_path):
    check_run_directory(tmp_path)  # must not raise


def test_hydra_own_files_do_not_count_as_a_previous_run(tmp_path):
    """Hydra writes these *before* `run_experiment` is reached, so they are there on a first run
    too. Counting them would make every run look like a reuse."""
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text("a: 1\n")
    (tmp_path / "deft_tree.log").write_text("starting\n")

    check_run_directory(tmp_path)  # must not raise


@pytest.mark.parametrize(
    "artifact",
    ["tree_polymerase.dill", "intermediate_tree.pkl", "provenance.json", "token_usage.json"],
)
def test_each_run_artifact_blocks_a_second_run(tmp_path, artifact):
    (tmp_path / artifact).write_text("{}")

    with pytest.raises(FileExistsError) as excinfo:
        check_run_directory(tmp_path)

    message = str(excinfo.value)
    assert artifact in message, "the message must name what it found"
    assert "hydra.run.dir" in message, "and how to give this run its own directory"
    assert "allow_run_dir_reuse" in message, "and how to override deliberately"


def test_the_refusal_names_everything_it_found(tmp_path):
    for name in ("tree_x.dill", "provenance.json", "token_usage.json"):
        (tmp_path / name).write_text("{}")

    with pytest.raises(FileExistsError) as excinfo:
        check_run_directory(tmp_path)

    for name in ("tree_x.dill", "provenance.json", "token_usage.json"):
        assert name in str(excinfo.value)


def test_reuse_can_be_allowed_deliberately_and_says_so(tmp_path, caplog):
    (tmp_path / "provenance.json").write_text("{}")

    with caplog.at_level(logging.WARNING):
        check_run_directory(tmp_path, allow_reuse=True)  # must not raise

    assert "allow_run_dir_reuse" in caplog.text
    assert "overwriting" in caplog.text


def test_a_directory_holding_only_a_subdirectory_is_fine(tmp_path):
    """Only files count; a stray directory named like an artifact must not block a run."""
    (tmp_path / "provenance.json").mkdir()

    check_run_directory(tmp_path)  # must not raise
