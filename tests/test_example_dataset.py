"""The example dataset: its builder, its declaration, its description and its config agree.

The example is the one dataset a clone holds, and it is what shows how a dataset is prepared. Its
files are written by hand in four places, and `data/` is not in git, so nothing but these tests
would notice the builder being changed without the declaration, or the description stating a
layout the builder no longer cuts.
"""

from __future__ import annotations

import json
import shutil
import sys

import pandas as pd
import pytest
import yaml

from seqmaestro import arms
from seqmaestro.dataset import GROUP_FILE, SPLITS, failed_checks, finalise, flagged_checks
from seqmaestro.llm import load_task
from seqmaestro.pipeline import config as experiment_config
from seqmaestro.pipeline.repository import DESCRIPTIONS, REPO

# The builder and the group-map writer are scripts, so they are loaded by path.
sys.path.insert(0, str(REPO / "experiments" / "prepare"))
import build_example_dataset as example
import write_group_map

PREPARE = REPO / "experiments" / "prepare"
DECLARATION = PREPARE / "declarations" / "example__splits.yaml"
CONFIG = REPO / "experiments" / "configs" / "example.yaml"
QUICK = REPO / "experiments" / "configs" / "example_quick.yaml"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The example, built in a temporary folder and finalised with its declaration: the folder and
    the record `dataset.json` holds."""
    directory = tmp_path_factory.mktemp("data") / "example"
    assert example.main(["--out", str(directory)]) == 0
    record = finalise(directory / example.SPLITS, yaml.safe_load(DECLARATION.read_text()))
    return directory, record


def test_the_example_passes_every_check_with_nothing_left_to_answer(built):
    _, record = built
    assert not failed_checks(record) and not flagged_checks(record)
    assert record["checks"]["groups_disjoint"]["status"] == "passed"
    assert record["checks"]["readme_present"]["status"] == "passed"


def test_the_declaration_states_what_the_builder_did(built):
    """The declaration is written by hand, so each fact it shares with the builder is compared."""
    directory, record = built
    declared = yaml.safe_load(DECLARATION.read_text())
    step = json.loads((directory / example.MANIFEST).read_text())["filters"][0]

    assert record["filters"] == [{"step": step["step"], "why": declared["filters"][0]["why"].strip(),
                                  "rows_in": step["rows_in"], "rows_out": step["rows_out"]}]
    assert declared["columns"] == {"sequence": example.SEQUENCE_COLUMN,
                                   "target": example.TARGET_COLUMN, "weight": None}
    assert declared["sequences"]["length"] == example.WINDOW
    assert declared["split"]["grouped_by"] == [example.GROUP_COLUMN]
    assert declared["split"]["stratified_on"] == [example.TARGET_COLUMN]
    assert declared["split"]["seed"] == example.SPLIT_SEED
    assert {label: entry["name"] for label, entry in declared["classes"].items()} == example.CLASS_NAMES
    assert declared["builders"] == [str((PREPARE / "build_example_dataset.py").relative_to(REPO))]
    assert declared["sources"][0]["path"] == example.DELIVERY


def test_the_split_files_hold_the_sequence_and_the_label_alone(built):
    directory, record = built
    for split in SPLITS:
        frame = pd.read_csv(directory / example.SPLITS / f"{split}.csv")
        assert list(frame.columns) == [example.SEQUENCE_COLUMN, example.TARGET_COLUMN]
    assert record["columns"]["unused"] == []


def test_the_motif_is_where_the_description_says_the_site_is(built):
    """A feature's positions are interpreted against the description, which states the window's
    length and where the site's centre is."""
    directory, _ = built
    centre, half = example.WINDOW // 2, len(example.MOTIF) // 2
    rows = pd.read_csv(directory / example.SPLITS / "train.csv")
    carries = rows[example.SEQUENCE_COLUMN].str[centre - half:centre + half + 1] == example.MOTIF
    bound = rows[example.TARGET_COLUMN] == 1

    assert carries[bound].mean() == pytest.approx(example.MOTIF_SHARE, abs=0.1)
    assert not carries[~bound].any()
    layout = load_task(DESCRIPTIONS, "example").sequence_layout
    assert f"{example.WINDOW} nucleotides" in layout
    assert f"position {centre} is the centre of the site" in layout


def test_the_group_map_script_writes_the_map_the_builder_wrote(built, tmp_path):
    """`write_group_map.py` gives an existing split its map. On the example it must reproduce the
    builder's, since both trace the declared grouping in the built file."""
    directory, _ = built
    copy = tmp_path / "splits"
    shutil.copytree(directory / example.SPLITS, copy)
    (copy / GROUP_FILE).unlink()

    assert write_group_map.main(["--dataset", str(directory / example.BUILT), "--splits", str(copy),
                                 "--declared", str(DECLARATION)]) == 0
    assert (copy / GROUP_FILE).read_bytes() == (directory / example.SPLITS / GROUP_FILE).read_bytes()


def test_the_config_names_the_folder_the_builder_writes_and_loads_on_it(built):
    directory, _ = built
    stated = yaml.safe_load(CONFIG.read_text())["dataset"]
    assert REPO / stated["splits"] == example.DATASET / example.SPLITS

    config = experiment_config.load(CONFIG, (f"dataset.splits={directory / example.SPLITS}",))
    assert (config.name, config.dataset.task, config.dataset.target_kind) == (
        "example", "example", "binary")
    assert config.dataset.blinded is bool(load_task(DESCRIPTIONS, "example").judge_only)
    assert config.reduction is not None and config.reliability is not None


def test_the_quick_config_is_the_example_cut_down_to_what_needs_no_language_model(built):
    """The pages offer the quick config as a run that needs no credentials, so no arm of it may be
    on a pool with a written source, and the judge must be off. Everything it runs, the full
    config runs too."""
    directory, _ = built
    splits = (f"dataset.splits={directory / example.SPLITS}",)
    full, quick = experiment_config.load(CONFIG, splits), experiment_config.load(QUICK, splits)

    assert quick.dataset == full.dataset
    assert set(quick.arm_cells()) < set(full.arm_cells())
    assert not any(arms.calls_language_model(pool) for _, pool in quick.arm_cells())
    assert not quick.grid.judge
    assert len(quick.grid.seeds) == 2
    assert set(quick.reduced_arms()) == set(quick.arm_cells()), "every arm goes through Stage 3A"


def test_building_again_writes_the_same_files(built, tmp_path):
    """A second build must leave a finalised example usable: `dataset.json` records the digest of
    each split file, and a file that changed is refused."""
    directory, _ = built
    assert example.main(["--out", str(tmp_path / "again")]) == 0
    for name in (example.DELIVERY, example.BUILT, *(f"{example.SPLITS}/{split}.csv" for split in SPLITS)):
        assert (tmp_path / "again" / name).read_bytes() == (directory / name).read_bytes(), name


def test_a_window_that_drops_much_of_a_class_is_refused(built, monkeypatch):
    """A filter reports what it dropped, and refuses to drop many: a wider window here would cost
    one class most of its rows, and the build stops before writing a dataset with another task."""
    directory, _ = built
    monkeypatch.setattr(example, "WINDOW", 100)
    with pytest.raises(ValueError, match="drops 60% of the bound sites"):
        example.build(directory / example.DELIVERY)
