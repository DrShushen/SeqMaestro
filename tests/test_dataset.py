"""The Stage 1 standard: what `finalise` writes, what each check catches, and what `load_dataset` refuses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqmaestro.dataset import (
    ACKNOWLEDGED,
    ADVISORY,
    CHECKS,
    DATASET_FILE,
    FAILED,
    FLAGGED,
    GROUP_FILE,
    NOT_APPLICABLE,
    PASSED,
    Columns,
    DatasetError,
    finalise,
    load_dataset,
    validate_declared,
)
from seqmaestro.features.positional import PositionalWindow
from seqmaestro.grouping import group_map, shared_stretch_groups
from seqmaestro.harness.power import smallest_detectable_auroc, smallest_detectable_correlation
from seqmaestro.target import BINARY, CONTINUOUS, STRATA, target_for
from tests.planted import ANALYSES

PREPARE = Path(__file__).resolve().parents[1] / "experiments" / "prepare"

SEQUENCES = {
    "train": ["ACGTAC", "CCGTAA", "TTGACA", "GGGTCA", "ACCTGA", "TAGCAT", "CATGCA", "GTACGT"],
    "val": ["AAGTCC", "CTGACG", "GTTACA", "TCAGGA"],
    "test": ["ACTTGA", "GCAGTC", "TTCAGG", "CAGGTA"],
}
LABELS = {"train": [1, 0, 1, 0, 1, 0, 1, 0], "val": [1, 0, 1, 0], "test": [1, 0, 1, 0]}


def a_declaration(**changes) -> dict:
    declared = {
        "columns": {"sequence": "sequence", "target": "is_positive", "weight": None},
        "target": {"kind": "binary"},
        "classes": {1: {"name": "bound", "meaning": "sites the factor binds"},
                    0: {"name": "unbound", "meaning": "matched sites it does not"}},
        "sequences": {"window": "6 nt around the site", "anchor": "the site centre",
                      "orientation": "the strand the site sits on", "length": 6},
        "split": {"method": "grouped, stratified", "grouped_by": ["locus"],
                  "stratified_on": ["is_positive"], "seed": 7},
        "filters": [{"step": "delivered", "why": "as received",
                     "rows_in": {1: 8, 0: 8}, "rows_out": {1: 8, 0: 8}}],
        "builders": ["experiments/prepare/build_example.py"],
    }
    declared.update(changes)
    return declared


def a_split(tmp_path: Path, weights: dict | None = None, readme: bool = True,
            groups: dict | None = None, sequences: dict | None = None,
            extra: dict | None = None, targets: dict | None = None,
            target_column: str = "is_positive") -> Path:
    """A small prepared dataset: three splits, a group map, and a README one level up.

    Each pair of sequences in a split shares a locus, so the group map is coarser than one group
    per row and the grouping check has something to test. `sequences` replaces the default
    sequences, split by split, `targets` the default labels, and `extra` adds columns, as
    `{column: {split: values}}`.
    """
    directory = tmp_path / "dataset" / "splits"
    directory.mkdir(parents=True)
    rows = []
    for split, texts in (sequences or SEQUENCES).items():
        frame = pd.DataFrame({"sequence": texts, target_column: (targets or LABELS)[split]})
        if weights is not None:
            frame["fit_weight"] = weights[split]
        for column, values in (extra or {}).items():
            frame[column] = values[split]
        frame.to_csv(directory / f"{split}.csv", index=False)
        rows += [{"sequence": sequence, "group": f"{split}_locus_{i // 2}"}
                 for i, sequence in enumerate(texts)]
    pd.DataFrame(rows if groups is None else groups).to_csv(directory / GROUP_FILE, index=False)
    if readme:
        (directory.parent / "README.md").write_text("# An example dataset\n")
    return directory


def statuses(record: dict) -> dict[str, str]:
    return {name: result["status"] for name, result in record["checks"].items()}


# ------------------------------------------------------------------------------ what is written


def test_a_clean_split_passes_every_check_that_applies(tmp_path):
    record = finalise(a_split(tmp_path), a_declaration())
    found = statuses(record)
    assert list(found) == list(CHECKS)
    # A fixed-length dataset with no weights, no covariates and no positional window.
    absent = {"weights_valid", "sequence_lengths", "positional_window_fits", "covariates_valid",
              "target_distribution"}
    assert {found[name] for name in absent} == {NOT_APPLICABLE}
    assert {status for name, status in found.items() if name not in absent} == {PASSED}


def test_the_record_counts_what_the_files_hold(tmp_path):
    record = finalise(a_split(tmp_path), a_declaration())
    assert record["splits"]["train"] == {**record["splits"]["train"], "rows": 8, "positive": 4,
                                         "negative": 4, "groups": 4}
    assert record["splits"]["test"]["rows"] == 4
    assert sum(split["share"] for split in record["splits"].values()) == pytest.approx(1.0, abs=1e-3)
    assert record["sequences"]["alphabet"] == ["A", "C", "G", "T"]
    assert record["sequences"]["observed_length"] == {"min": 6, "max": 6}
    assert record["classes"]["1"]["name"] == "bound"


def test_the_record_is_written_beside_the_splits(tmp_path):
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    written = json.loads((directory / DATASET_FILE).read_text())
    assert written["columns"]["target"] == "is_positive"


def test_the_design_power_is_the_harness_arithmetic_on_the_test_counts():
    test = np.array([1] * 120 + [0] * 80)
    target = target_for(BINARY)
    test_side, train_side = target.held_out_power(test), target.training_information([1, 0] * 50)
    assert test_side["smallest_detectable_auroc"] == round(smallest_detectable_auroc(120, 80), 3)
    assert test_side["n_positive"] == 120 and train_side["weighted"] is False


def test_a_test_split_too_small_to_detect_anything_says_so_rather_than_writing_nan(tmp_path):
    """One positive has no sampling distribution, so there is no floor to report."""
    directory = a_split(tmp_path)
    _relabel(directory, "test", [1, 0, 0, 0])
    finalise(directory, a_declaration(filters=[]))
    written = json.loads((directory / DATASET_FILE).read_text())
    assert written["design_power"]["test"]["smallest_detectable_auroc"] is None


def test_unequal_weights_lower_the_training_information():
    labels = np.array([1] * 100 + [0] * 100)
    weights = np.concatenate([np.ones(50), np.full(50, 1e-3), np.ones(100)])
    weighted = target_for(BINARY).training_information(labels, weights)
    assert weighted["weighted"] is True
    assert weighted["effective_n"] < weighted["effective_n_counting_rows"]


# ------------------------------------------------------------------------------ what each check catches


def _relabel(directory: Path, split: str, labels: list):
    frame = pd.read_csv(directory / f"{split}.csv")
    frame["is_positive"] = labels
    frame.to_csv(directory / f"{split}.csv", index=False)


def _resequence(directory: Path, split: str, index: int, sequence: str):
    frame = pd.read_csv(directory / f"{split}.csv")
    frame.loc[index, "sequence"] = sequence
    frame.to_csv(directory / f"{split}.csv", index=False)


def test_a_label_that_is_not_zero_or_one_fails(tmp_path):
    directory = a_split(tmp_path)
    _relabel(directory, "val", [1, 0, 2, 0])
    assert statuses(finalise(directory, a_declaration()))["target_valid"] == FAILED


def test_a_split_holding_one_class_fails(tmp_path):
    directory = a_split(tmp_path)
    _relabel(directory, "test", [0, 0, 0, 0])
    record = finalise(directory, a_declaration(filters=[]))
    assert statuses(record)["target_varies_in_every_split"] == FAILED


def test_a_symbol_training_never_saw_fails(tmp_path):
    directory = a_split(tmp_path)
    _resequence(directory, "test", 0, "ACTNGA")
    assert statuses(finalise(directory, a_declaration()))["alphabet_seen_in_training"] == FAILED


def test_a_length_other_than_the_declared_one_fails(tmp_path):
    directory = a_split(tmp_path)
    _resequence(directory, "val", 1, "CTGACGT")
    assert statuses(finalise(directory, a_declaration()))["lengths_as_declared"] == FAILED


def test_a_declared_variable_length_is_not_checked(tmp_path):
    declared = a_declaration(sequences={**a_declaration()["sequences"], "length": "variable"})
    assert statuses(finalise(a_split(tmp_path), declared))["lengths_as_declared"] == NOT_APPLICABLE


def test_one_sequence_in_two_splits_is_flagged_rather_than_failed(tmp_path):
    """The check is advisory. Whether a repeated sequence is a leak or a property of the data
    depends on the experiment, so the check measures the overlap and Stage 1 decides."""
    directory = a_split(tmp_path)
    _resequence(directory, "test", 2, SEQUENCES["train"][0])
    record = finalise(directory, a_declaration())
    assert statuses(record)["no_sequence_in_two_splits"] == FLAGGED
    assert load_dataset(directory).target_column == "is_positive"


def test_a_flagged_sequence_overlap_reports_the_share_of_the_held_out_rows(tmp_path):
    """A count of sequences does not say how much of a held-out score could come from memorised
    sequences, but the share of held-out rows does."""
    directory = a_split(tmp_path)
    _resequence(directory, "test", 2, SEQUENCES["train"][0])
    measured = finalise(directory, a_declaration())["checks"]["no_sequence_in_two_splits"]["measured"]
    assert measured["test"] == {"rows": 1, "of": 4, "share": 0.25,
                                "share_by_class": {"1": 0.5, "0": 0.0}}
    assert measured["val"]["rows"] == 0
    assert measured["test_in_val"]["rows"] == 0


def test_a_test_sequence_that_repeats_a_validation_one_is_flagged_too(tmp_path):
    """A run makes its choices on validation, so a test row whose sequence sits there has been
    seen by those choices. It is no training sequence, so it is counted apart, and the grouping
    check leaves it to this one."""
    directory = a_split(tmp_path)
    _resequence(directory, "test", 2, SEQUENCES["val"][0])
    result = finalise(directory, a_declaration())["checks"]["no_sequence_in_two_splits"]
    assert result["status"] == FLAGGED
    assert "repeat a validation sequence" in result["detail"]
    assert result["measured"]["test_in_val"] == {"rows": 1, "of": 4, "share": 0.25}
    assert result["measured"]["test"]["rows"] == 0, "it repeats no training sequence"


def test_two_different_sequences_of_one_group_in_two_splits_fails(tmp_path):
    """This is the leak that only grouping can prevent, and the check exists to catch it."""
    directory = a_split(tmp_path)
    table = pd.read_csv(directory / GROUP_FILE)
    table.loc[table["sequence"] == SEQUENCES["test"][0], "group"] = "train_locus_0"
    table.to_csv(directory / GROUP_FILE, index=False)
    assert statuses(finalise(directory, a_declaration()))["groups_disjoint"] == FAILED


def test_a_sequence_in_two_groups_is_set_aside_rather_than_blamed_on_the_grouping(tmp_path):
    """The split files do not identify source rows, so for a sequence carried by two groups there is
    no record of which group's row went to which split. Treating it as evidence would fail a
    grouping over a placement that was never observed."""
    directory = a_split(tmp_path)
    table = pd.read_csv(directory / GROUP_FILE)
    doubled = pd.concat([table, pd.DataFrame([{"sequence": SEQUENCES["test"][0],
                                               "group": "train_locus_0"}])])
    doubled.to_csv(directory / GROUP_FILE, index=False)
    result = finalise(directory, a_declaration())["checks"]["groups_disjoint"]
    assert result["status"] == PASSED
    assert "more than one group" in result["detail"]


def test_a_missing_group_map_fails_rather_than_passing(tmp_path):
    directory = a_split(tmp_path)
    (directory / GROUP_FILE).unlink()
    assert statuses(finalise(directory, a_declaration()))["groups_disjoint"] == FAILED


def test_a_group_map_that_does_not_cover_the_splits_fails(tmp_path):
    """A group map can miss sequences in the splits when the dataset was rebuilt after its split
    was cut."""
    directory = a_split(tmp_path)
    table = pd.read_csv(directory / GROUP_FILE)
    table[table["sequence"] != SEQUENCES["test"][0]].to_csv(directory / GROUP_FILE, index=False)
    result = finalise(directory, a_declaration())["checks"]["groups_disjoint"]
    assert result["status"] == FAILED
    assert "absent from" in result["detail"]


def test_negative_weights_fail(tmp_path):
    weights = {split: [1.0] * len(sequences) for split, sequences in SEQUENCES.items()}
    weights["train"][0] = -1.0
    declared = a_declaration(columns={"sequence": "sequence", "target": "is_positive",
                                      "weight": "fit_weight"})
    record = finalise(a_split(tmp_path, weights=weights), declared)
    assert statuses(record)["weights_valid"] == FAILED


def test_valid_weights_pass_and_mark_the_design_weighted(tmp_path):
    weights = {split: [1.0, 2.0] * (len(sequences) // 2) for split, sequences in SEQUENCES.items()}
    declared = a_declaration(columns={"sequence": "sequence", "target": "is_positive",
                                      "weight": "fit_weight"})
    record = finalise(a_split(tmp_path, weights=weights), declared)
    assert statuses(record)["weights_valid"] == PASSED
    assert record["design_power"]["train"]["weighted"] is True


def test_filters_that_do_not_reach_the_row_count_fail(tmp_path):
    declared = a_declaration(filters=[{"step": "delivered", "why": "as received",
                                       "rows_in": {1: 9, 0: 8}, "rows_out": {1: 9, 0: 8}}])
    assert statuses(finalise(a_split(tmp_path), declared))["filters_account_for_rows"] == FAILED


def test_a_dataset_without_a_readme_fails(tmp_path):
    assert statuses(finalise(a_split(tmp_path, readme=False), a_declaration()))["readme_present"] == FAILED


def test_an_acknowledgement_records_the_answer_and_what_it_answered(tmp_path):
    directory = a_split(tmp_path)
    _resequence(directory, "test", 2, SEQUENCES["train"][0])
    record = finalise(directory, a_declaration(
        acknowledged={"no_sequence_in_two_splits": "one site measured twice, kept deliberately"}))
    result = record["checks"]["no_sequence_in_two_splits"]
    assert result["status"] == ACKNOWLEDGED
    assert "measured twice" in result["acknowledged"]
    assert result["was"] == FLAGGED
    assert load_dataset(directory).target_column == "is_positive"


def test_an_acknowledgement_also_accepts_a_blocking_failure(tmp_path):
    """`acknowledged` also overrides a blocking failure, and the record keeps the original status
    under `was`."""
    directory = a_split(tmp_path)
    (directory / "README.md").unlink() if (directory / "README.md").exists() else None
    (directory.parent / "README.md").unlink()
    record = finalise(directory, a_declaration(
        acknowledged={"readme_present": "written up in the shared drive instead"}))
    assert record["checks"]["readme_present"]["status"] == ACKNOWLEDGED
    assert record["checks"]["readme_present"]["was"] == FAILED
    assert load_dataset(directory).target_column == "is_positive"


# ------------------------------------------------------------------------------ the declaration


@pytest.mark.parametrize("change, message", [
    ({"colour": "blue"}, "no field colour"),
    ({"classes": {1: {"name": "a", "meaning": "x"}}}, "both labels"),
    ({"classes": {1: {"name": "a", "meaning": "x"}, 0: {"name": "a", "meaning": "y"}}}, "same name"),
    ({"classes": {1: {"name": "a", "meaning": "x"}, 2: {"name": "b", "meaning": "y"}}}, "labels are 1 and 0"),
    ({"acknowledged": {"not_a_check": "because"}}, "not a check"),
    ({"acknowledged": {"groups_disjoint": ""}}, "reason"),
])
def test_a_bad_declaration_is_refused_by_name(change, message):
    with pytest.raises(DatasetError, match=message):
        validate_declared(a_declaration(**change))


def test_a_split_that_declares_no_grouping_reports_it_rather_than_passing(tmp_path):
    """The check reports `not_applicable` and not a pass, because a check that did not run has
    shown nothing."""
    split = {**a_declaration()["split"], "grouped_by": []}
    directory = a_split(tmp_path)
    assert statuses(finalise(directory, a_declaration(split=split)))["groups_disjoint"] == NOT_APPLICABLE


def test_nothing_about_the_columns_is_defaulted():
    declared = a_declaration()
    del declared["columns"]
    with pytest.raises(DatasetError, match="missing columns"):
        validate_declared(declared)


@pytest.mark.parametrize(("section", "key"), [("columns", "weight"), ("split", "grouped_by"),
                                              ("split", "stratified_on"), ("split", "seed")])
def test_an_answer_of_none_is_stated_and_never_read_from_a_missing_key(section, key):
    """A split with no `grouped_by` is not checked for a group on both sides, so a declaration
    that only forgot the key must not pass for an ungrouped one. The same holds for the weights,
    the stratification and the seed: each is stated, with null or an empty list for none."""
    declared = a_declaration()
    declared[section] = {name: value for name, value in declared[section].items() if name != key}
    with pytest.raises(DatasetError, match=f"`{section}` is missing {key}"):
        validate_declared(declared)
    none = [] if key in ("grouped_by", "stratified_on") else None
    validate_declared(a_declaration(**{section: {**a_declaration()[section], key: none}}))


# ------------------------------------------------------------------------------ what loading refuses


def test_loading_refuses_a_dataset_that_was_never_finalised(tmp_path):
    with pytest.raises(DatasetError, match=r"finalise_dataset\.py"):
        load_dataset(a_split(tmp_path))


def test_loading_refuses_a_failed_check(tmp_path):
    """A blocking failure means the dataset is broken, so it stops Stage 2."""
    directory = a_split(tmp_path)
    (directory.parent / "README.md").unlink()
    finalise(directory, a_declaration())
    with pytest.raises(DatasetError, match="readme_present"):
        load_dataset(directory)


def test_loading_accepts_an_advisory_finding_and_keeps_it_readable(tmp_path):
    """An advisory finding is a question for Stage 1 and does not stop a run. It stays visible in
    the loaded record, because a reader of the result needs it."""
    directory = a_split(tmp_path)
    _resequence(directory, "test", 2, SEQUENCES["train"][0])
    finalise(directory, a_declaration())
    dataset = load_dataset(directory)
    assert dataset.record["checks"]["no_sequence_in_two_splits"]["status"] == FLAGGED


def test_loading_refuses_a_training_file_changed_after_finalising(tmp_path):
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    _resequence(directory, "train", 0, "ACGTAA")
    with pytest.raises(DatasetError, match="has changed"):
        load_dataset(directory)


def test_loading_does_not_read_the_test_split(tmp_path):
    """The test split's digest is checked at the reveal, which is the only point after Stage 1
    that opens the file."""
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    (directory / "test.csv").write_text("not a split at all\n")
    assert load_dataset(directory).sequence_column == "sequence"


def test_loading_refuses_another_schema_version(tmp_path):
    directory = a_split(tmp_path)
    record = finalise(directory, a_declaration())
    (directory / DATASET_FILE).write_text(json.dumps({**record, "schema_version": 0}))
    with pytest.raises(DatasetError, match="schema version"):
        load_dataset(directory)


def test_a_loaded_dataset_carries_its_classes(tmp_path):
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    loaded = load_dataset(directory)
    assert loaded.classes[1].name == "bound" and loaded.classes[0].name == "unbound"
    assert loaded.weight_column is None


# ------------------------------------------------------------------------------ covariates

COVARIATES = {
    "signal": {"train": [0.5, 1.5, 2.0, 0.1, 3.3, 0.0, 1.1, 2.2], "val": [0.4, 1.0, 2.5, 0.3],
               "test": [1.2, 0.2, 0.9, 1.7]},
    "peaks": {"train": [0, 1, 2, 0, 1, 1, 3, 0], "val": [1, 0, 2, 1], "test": [0, 0, 1, 2]},
}
DESCRIBED = {"signal": {"description": "mean signal over the window"},
             "peaks": {"description": "peaks overlapping the window"}}


def covariate_declaration(**changes) -> dict:
    return a_declaration(**{"columns": {"sequence": "sequence", "target": "is_positive",
                                        "weight": None, "covariates": ["signal", "peaks"]},
                            "covariates": DESCRIBED, **changes})


def with_values(column: str, split: str, values: list) -> dict:
    """COVARIATES with one covariate's values in one split replaced."""
    return {**COVARIATES, column: {**COVARIATES[column], split: values}}


def test_declared_covariates_are_recorded_with_their_spread_on_the_training_rows(tmp_path):
    record = finalise(a_split(tmp_path, extra=COVARIATES), covariate_declaration())
    assert record["columns"]["covariates"] == ["signal", "peaks"]
    assert record["covariates"]["signal"]["description"] == "mean signal over the window"
    peaks = record["covariates"]["peaks"]["training"]
    assert {key: peaks[key] for key in ("min", "max", "mean", "distinct")} == \
           {"min": 0.0, "max": 3.0, "mean": 1.0, "distinct": 4}
    assert peaks["sd"] == pytest.approx(float(np.std([0, 1, 2, 0, 1, 1, 3, 0], ddof=1)), rel=1e-5)
    assert statuses(record)["covariates_valid"] == PASSED


def test_a_dataset_without_covariates_says_so(tmp_path):
    directory = a_split(tmp_path)
    record = finalise(directory, a_declaration())
    assert record["columns"]["covariates"] == [] and record["covariates"] == {}
    assert statuses(record)["covariates_valid"] == NOT_APPLICABLE
    loaded = load_dataset(directory)
    assert loaded.covariate_columns == () and loaded.covariates == {}


def test_a_column_that_nothing_declares_is_listed_as_unused(tmp_path):
    """A builder may keep provenance columns in the split files, so an undeclared column is not an
    error. It is listed, so it cannot sit in the files unnoticed."""
    record = finalise(a_split(tmp_path, extra={"signal": COVARIATES["signal"]}), a_declaration())
    assert record["columns"]["unused"] == ["signal"]


def test_the_dataset_hands_over_its_covariates_as_inputs_without_the_label(tmp_path):
    directory = a_split(tmp_path, extra=COVARIATES)
    finalise(directory, covariate_declaration())
    dataset = load_dataset(directory)
    assert dataset.covariate_columns == ("signal", "peaks")
    assert dataset.covariates == {"signal": "mean signal over the window",
                                  "peaks": "peaks overlapping the window"}
    assert dataset.columns == Columns("sequence", "is_positive", None, ("signal", "peaks"), BINARY)

    inputs = dataset.inputs(pd.read_csv(dataset.train_path))
    assert inputs.sequences.tolist() == SEQUENCES["train"]
    assert inputs.covariate_columns == ("signal", "peaks")
    assert inputs.covariates["peaks"].tolist() == [float(v) for v in COVARIATES["peaks"]["train"]]
    assert list(inputs.frame("sequence").columns) == ["sequence", "signal", "peaks"]


def test_columns_refuse_rows_that_lack_one_of_them():
    columns = Columns("sequence", "label", None, ("signal",), BINARY)
    with pytest.raises(DatasetError, match="no column signal"):
        columns.inputs(pd.DataFrame({"sequence": ["ACGT"], "label": [1]}))


@pytest.mark.parametrize("covariates, message", [
    (with_values("signal", "val", [0.4, "", 2.5, 0.3]), "signal has 1 missing or non-numeric"),
    (with_values("peaks", "test", [0, "many", 1, 2]), "peaks has 1 missing or non-numeric"),
    (with_values("signal", "val", [0.4, "inf", 2.5, 0.3]), "signal is infinite in val"),
    (with_values("peaks", "train", [1] * 8), "peaks is constant on the training rows"),
])
def test_a_covariate_no_model_can_use_as_it_stands_fails(tmp_path, covariates, message):
    """Missing values are refused and not filled: imputation is a Stage 1 decision, recorded as a
    preparation step, so no stage after it has to guess one."""
    result = finalise(a_split(tmp_path, extra=covariates), covariate_declaration())["checks"]
    assert result["covariates_valid"]["status"] == FAILED
    assert message in result["covariates_valid"]["detail"]


def test_a_covariate_missing_from_a_split_fails_and_the_record_is_still_written(tmp_path):
    directory = a_split(tmp_path, extra=COVARIATES)
    frame = pd.read_csv(directory / "test.csv")
    frame.drop(columns="peaks").to_csv(directory / "test.csv", index=False)
    result = finalise(directory, covariate_declaration())["checks"]["covariates_valid"]
    assert result["status"] == FAILED and "peaks is absent from test" in result["detail"]
    assert (directory / DATASET_FILE).exists()


def test_true_and_false_are_read_as_one_and_zero(tmp_path):
    flags = {"train": [True, False, True, False, True, True, False, False],
             "val": [True, False, False, True], "test": [False, True, True, False]}
    declared = a_declaration(columns={"sequence": "sequence", "target": "is_positive",
                                      "weight": None, "covariates": ["open"]},
                             covariates={"open": {"description": "whether the site is open"}})
    record = finalise(a_split(tmp_path, extra={"open": flags}), declared)
    assert statuses(record)["covariates_valid"] == PASSED
    assert record["covariates"]["open"]["training"]["mean"] == 0.5


@pytest.mark.parametrize("change, message", [
    ({"covariates": {"signal": DESCRIBED["signal"]}}, "missing peaks"),
    ({"covariates": {**DESCRIBED, "depth": {"description": "d"}}}, "does not declare"),
    ({"covariates": {**DESCRIBED, "peaks": {"description": ""}}}, "non-empty text"),
    ({"covariates": {**DESCRIBED, "peaks": {"description": "d", "unit": "count"}}}, "no key unit"),
    ({"columns": {"sequence": "sequence", "target": "is_positive", "weight": None,
                  "covariates": ["signal", "is_positive"]}}, "already the sequence"),
    ({"columns": {"sequence": "sequence", "target": "is_positive", "weight": None,
                  "covariates": ["signal", "2nd_peak"]}}, "legal Python identifiers"),
    ({"columns": {"sequence": "sequence", "target": "is_positive", "weight": None,
                  "covariates": ["signal", "signal"]}}, "more than once"),
])
def test_a_bad_covariate_declaration_is_refused_by_name(change, message):
    with pytest.raises(DatasetError, match=message):
        validate_declared(covariate_declaration(**change))


def test_a_repeated_sequence_whose_covariates_differ_is_counted_apart(tmp_path):
    """Two rows with one sequence and different measurements are different samples, so the overlap
    finding says how many of the repeated rows those are."""
    directory = a_split(tmp_path, extra=COVARIATES)
    _resequence(directory, "val", 0, SEQUENCES["train"][0])
    _resequence(directory, "val", 1, SEQUENCES["train"][1])
    frame = pd.read_csv(directory / "val.csv")
    frame.loc[1, ["signal", "peaks"]] = [COVARIATES["signal"]["train"][1],
                                         COVARIATES["peaks"]["train"][1]]
    frame.to_csv(directory / "val.csv", index=False)
    result = finalise(directory, covariate_declaration())["checks"]["no_sequence_in_two_splits"]
    assert result["status"] == FLAGGED
    assert result["measured"]["val"]["rows"] == 2
    assert result["measured"]["val"]["differ_in_covariates"] == 1


# ------------------------------------------------------------------------------ variable lengths

#: Class 1 is longer than class 0, which is the finding the length check exists to put to Stage 1.
RAGGED = {"train": ["ACGTACGTAC", "CCGTAA", "TTGACAGGAT", "GGGTCA", "ACCTGAACGT", "TAGCA",
                    "CATGCATGCA", "GTACG"],
          "val": ["AAGTCCATGC", "CTGAC", "GTTACAGGTA", "TCAGG"],
          "test": ["ACTTGACCAT", "GCAGT", "TTCAGGATCC", "CAGGT"]}


def ragged_declaration(window: dict | None = None) -> dict:
    sequences = {**a_declaration()["sequences"], "length": "variable"}
    if window is not None:
        sequences["positional_window"] = window
    return a_declaration(sequences=sequences)


def test_lengths_that_vary_are_flagged_with_their_spread_by_class(tmp_path):
    result = finalise(a_split(tmp_path, sequences=RAGGED), ragged_declaration())["checks"]
    lengths = result["sequence_lengths"]
    assert lengths["status"] == FLAGGED
    assert "10 long in class 1 and 5.5 in class 0" in lengths["detail"]
    assert lengths["measured"]["by_split"]["train"]["1"]["median"] == 10.0
    assert result["lengths_as_declared"]["status"] == NOT_APPLICABLE


def test_a_declared_variable_length_that_does_not_vary_passes(tmp_path):
    result = finalise(a_split(tmp_path), ragged_declaration())["checks"]["sequence_lengths"]
    assert result["status"] == PASSED and "every sequence is 6 long" in result["detail"]


def test_the_window_reports_what_it_pads_and_cuts_by_class(tmp_path):
    record = finalise(a_split(tmp_path, sequences=RAGGED),
                      ragged_declaration({"length": 6, "anchor": "start", "pad": True}))
    measured = record["checks"]["sequence_lengths"]["measured"]["window"]
    assert measured["shorter_by_class"] == {"1": 0.0, "0": 0.75}
    assert measured["longer_by_class"] == {"1": 1.0, "0": 0.0}
    assert record["checks"]["positional_window_fits"]["status"] == NOT_APPLICABLE
    assert record["sequences"]["positional_window"] == {"length": 6, "anchor": "start", "pad": True}


def test_a_window_without_padding_refuses_a_shorter_sequence(tmp_path):
    """Without padding, Stage 1 filters the short sequences out as a recorded step, so every arm
    reads the same rows. One left in the splits is a defect."""
    record = finalise(a_split(tmp_path, sequences=RAGGED),
                      ragged_declaration({"length": 6, "anchor": "end", "pad": False}))
    result = record["checks"]["positional_window_fits"]
    assert result["status"] == FAILED and "declared without padding" in result["detail"]


def test_every_sequence_filling_the_window_passes(tmp_path):
    record = finalise(a_split(tmp_path, sequences=RAGGED),
                      ragged_declaration({"length": 5, "anchor": "end", "pad": False}))
    assert statuses(record)["positional_window_fits"] == PASSED
    assert load_dataset(record_directory(tmp_path)).positional_window == \
           PositionalWindow(length=5, anchor="end", pad=False)


def record_directory(tmp_path: Path) -> Path:
    return tmp_path / "dataset" / "splits"


def test_a_dataset_without_a_window_says_so(tmp_path):
    directory = a_split(tmp_path)
    record = finalise(directory, a_declaration())
    assert record["sequences"]["positional_window"] is None
    assert load_dataset(directory).positional_window is None


@pytest.mark.parametrize("window, message", [
    ({"length": 6, "anchor": "start"}, "must state length, anchor and pad"),
    ({"length": 6, "anchor": "middle", "pad": True}, "anchor must be one of start, end"),
    ({"length": 0, "anchor": "start", "pad": True}, "positive whole number"),
    ({"length": 6, "anchor": "start", "pad": "yes"}, "pad must be true or false"),
])
def test_a_bad_window_is_refused_by_name(window, message):
    with pytest.raises(DatasetError, match=message):
        validate_declared(ragged_declaration(window))


def test_a_window_longer_than_a_fixed_length_is_refused():
    sequences = {**a_declaration()["sequences"],
                 "positional_window": {"length": 7, "anchor": "start", "pad": False}}
    with pytest.raises(DatasetError, match="cannot be longer than the sequences"):
        validate_declared(a_declaration(sequences=sequences))


def test_a_misspelled_sequences_key_is_refused_rather_than_dropped():
    sequences = {**a_declaration()["sequences"],
                 "positional_windw": {"length": 6, "anchor": "start", "pad": True}}
    with pytest.raises(DatasetError, match="no key positional_windw"):
        validate_declared(a_declaration(sequences=sequences))


# ------------------------------------------------------------------------------ the finaliser


# ------------------------------------------------------------------------------ a continuous target


#: A number per row, spread without heavy tails, so that every check passes on it.
CHANGES = {"train": [0.12, -0.3, 0.05, 0.8, -0.02, 0.4, -0.6, 0.25],
           "val": [0.2, -0.1, 0.33, -0.4], "test": [0.1, -0.2, 0.6, -0.05]}


def continuous_split(tmp_path: Path, targets: dict | None = None, **kwargs) -> Path:
    return a_split(tmp_path, targets=targets or CHANGES, target_column="change", **kwargs)


def continuous_declaration(**changes) -> dict:
    declared = a_declaration(
        columns={"sequence": "sequence", "target": "change", "weight": None},
        target={"kind": "continuous", "name": "change in binding",
                "meaning": "the change in the factor's binding at the site, as a log ratio; "
                           "positive means binding rises"},
        split={"method": "grouped, stratified", "grouped_by": ["locus"],
               "stratified_on": ["deciles of change"], "seed": 7},
        filters=[{"step": "delivered", "why": "as received", "rows_in": 16, "rows_out": 16}])
    del declared["classes"]
    declared.update(changes)
    return declared


def test_a_clean_continuous_split_passes_every_check_that_applies(tmp_path):
    found = statuses(finalise(continuous_split(tmp_path), continuous_declaration()))
    assert list(found) == list(CHECKS)
    absent = {"weights_valid", "sequence_lengths", "positional_window_fits", "covariates_valid"}
    assert {found[name] for name in absent} == {NOT_APPLICABLE}
    assert {status for name, status in found.items() if name not in absent} == {PASSED}


def test_a_continuous_record_describes_the_target_and_summarises_each_split(tmp_path):
    record = finalise(continuous_split(tmp_path), continuous_declaration())
    assert record["target"] == {"kind": CONTINUOUS, "name": "change in binding",
                                "meaning": continuous_declaration()["target"]["meaning"]}
    assert record["classes"] is None
    train = record["splits"]["train"]
    assert "positive" not in train
    assert train["target"]["min"] == -0.6 and train["target"]["max"] == 0.8
    assert train["target"]["median"] == pytest.approx(np.median(CHANGES["train"]))
    assert record["checks"]["target_valid"]["detail"] == "every value is a finite number"


@pytest.mark.parametrize("value", ["inf", "", "a lot"])
def test_a_continuous_value_that_is_not_a_finite_number_fails(tmp_path, value):
    targets = {**CHANGES, "val": [0.2, value, 0.33, -0.4]}
    record = finalise(continuous_split(tmp_path, targets), continuous_declaration())
    assert statuses(record)["target_valid"] == FAILED


def test_a_split_whose_continuous_target_is_constant_fails(tmp_path):
    targets = {**CHANGES, "test": [0.5, 0.5, 0.5, 0.5]}
    record = finalise(continuous_split(tmp_path, targets), continuous_declaration())
    result = record["checks"]["target_varies_in_every_split"]
    assert result["status"] == FAILED and "test (every value 0.5)" in result["detail"]


def test_a_heavy_tailed_target_is_flagged_with_its_distribution(tmp_path):
    """Most rows near zero and a few far out: the tails are for Stage 1 to explain."""
    train = [0.01, -0.01, 0.02, 0.0, -0.02, 0.01, 0.9, -0.8]
    record = finalise(continuous_split(tmp_path, {**CHANGES, "train": train}),
                      continuous_declaration())
    result = record["checks"]["target_distribution"]
    assert result["status"] == FLAGGED
    assert result["measured"]["train"]["share_beyond_3_robust_sd"] == 0.25
    assert "beyond three robust standard deviations" in result["detail"]


def test_a_target_piled_on_one_value_is_flagged(tmp_path):
    train = [0.0, 0.0, 0.0, 0.8, -0.3, 0.4, -0.6, 0.25]
    record = finalise(continuous_split(tmp_path, {**CHANGES, "train": train}),
                      continuous_declaration())
    result = record["checks"]["target_distribution"]
    assert result["status"] == FLAGGED and result["measured"]["train"]["commonest_share"] == 0.375


def test_the_distribution_check_does_not_apply_to_a_binary_target(tmp_path):
    record = finalise(a_split(tmp_path), a_declaration())
    assert record["checks"]["target_distribution"]["status"] == NOT_APPLICABLE


def test_a_continuous_dataset_counts_its_filters_in_rows(tmp_path):
    steps = [{"step": "delivered", "why": "as received", "rows_in": 20, "rows_out": 17}]
    record = finalise(continuous_split(tmp_path), continuous_declaration(filters=steps))
    result = record["checks"]["filters_account_for_rows"]
    assert result["status"] == FAILED
    assert "leaves 17 rows, but the splits hold 16" in result["detail"]


@pytest.mark.parametrize("change, message", [
    ({"target": {"kind": "ordinal"}}, "binary or continuous"),
    ({"target": {"kind": "continuous", "name": "x"}}, "target.meaning"),
    ({"target": {"kind": "continuous", "name": "x", "meaning": "y", "units": "z"}}, "no key units"),
    ({"classes": {1: {"name": "a", "meaning": "x"}, 0: {"name": "b", "meaning": "y"}}},
     "continuous target is described by"),
    ({"filters": [{"step": "s", "why": "w", "rows_in": {1: 8, 0: 8}, "rows_out": 16}]},
     "whole number of rows"),
])
def test_a_bad_continuous_declaration_is_refused_by_name(change, message):
    with pytest.raises(DatasetError, match=message):
        validate_declared(continuous_declaration(**change))


def test_a_binary_target_needs_its_classes():
    declared = a_declaration()
    del declared["classes"]
    with pytest.raises(DatasetError, match="needs `classes`"):
        validate_declared(declared)
    with pytest.raises(DatasetError, match="no key name"):
        validate_declared(a_declaration(target={"kind": "binary", "name": "bound"}))


def test_the_kind_of_target_is_never_defaulted():
    declared = a_declaration()
    del declared["target"]
    with pytest.raises(DatasetError, match="missing target"):
        validate_declared(declared)


def test_continuous_weights_need_only_carry_some_weight(tmp_path):
    weights = {"train": [1.0] * 8, "val": [1.0] * 4, "test": [1.0] * 4}
    directory = continuous_split(tmp_path, weights=weights)
    record = finalise(directory, continuous_declaration(
        columns={"sequence": "sequence", "target": "change", "weight": "fit_weight"}))
    assert record["checks"]["weights_valid"] == {
        "status": PASSED,
        "detail": "weights are finite and non-negative, and the training rows carry weight"}


def test_continuous_lengths_are_reported_with_their_rank_correlation_with_the_target(tmp_path):
    """Longer sequences carry larger changes here, so the correlation is high and positive."""
    targets = {split: [len(text) / 10 + 0.01 * index for index, text in enumerate(texts)]
               for split, texts in RAGGED.items()}
    sequences = {**continuous_declaration()["sequences"], "length": "variable",
                 "positional_window": {"length": 6, "anchor": "start", "pad": True}}
    record = finalise(continuous_split(tmp_path, targets, sequences=RAGGED),
                      continuous_declaration(sequences=sequences))
    result = record["checks"]["sequence_lengths"]
    assert result["status"] == FLAGGED
    assert result["measured"]["spearman_with_target"] > 0.8
    assert result["measured"]["by_split"]["train"]["median"] == 8.0
    assert result["measured"]["window"]["shorter"] == 0.375
    assert "rank correlation" in result["detail"] and "of the sequences are padded" in result["detail"]


def test_a_repeated_sequence_is_counted_without_classes_for_a_continuous_target(tmp_path):
    sequences = {**SEQUENCES, "test": ["ACGTAC", *SEQUENCES["test"][1:]]}
    record = finalise(continuous_split(tmp_path, sequences=sequences), continuous_declaration())
    measured = record["checks"]["no_sequence_in_two_splits"]["measured"]["test"]
    assert measured == {"rows": 1, "of": 4, "share": 0.25}


def test_the_continuous_design_power_is_the_correlation_arithmetic_on_the_test_rows():
    target = target_for(CONTINUOUS)
    held_out = target.held_out_power(np.zeros(403))
    assert held_out["smallest_detectable_r"] == round(smallest_detectable_correlation(403), 3)
    assert held_out["smallest_detectable_r2"] == pytest.approx(
        smallest_detectable_correlation(403) ** 2, abs=1e-4)
    assert target.training_information(np.zeros(1000)) == {
        "rows": 1000, "weighted": False, "effective_n": 1000.0, "effective_n_counting_rows": 1000}


def test_a_loaded_continuous_dataset_carries_its_kind_and_no_classes(tmp_path):
    directory = continuous_split(tmp_path)
    finalise(directory, continuous_declaration())
    loaded = load_dataset(directory)
    assert loaded.target_kind == CONTINUOUS and loaded.classes == {}
    assert loaded.columns.target_kind == CONTINUOUS and loaded.target.kind == CONTINUOUS


def test_columns_refuse_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown target kind"):
        Columns("sequence", "label", None, (), "ordinal")


def test_a_continuous_target_is_stratified_on_quantile_bins():
    values = np.random.default_rng(0).normal(size=200)
    bins = target_for(CONTINUOUS).strata(values)
    assert sorted(np.unique(bins)) == list(range(STRATA))
    assert np.bincount(bins).min() == 200 // STRATA
    assert list(target_for(BINARY).strata([1, 0, 1])) == [1, 0, 1]


@pytest.fixture
def finaliser():
    sys.path.insert(0, str(PREPARE))
    import finalise_dataset

    yield finalise_dataset
    sys.path.remove(str(PREPARE))


def test_every_advisory_check_puts_its_question(finaliser):
    """The finaliser ends each advisory finding with the question it raises, so a finding without
    one would stop the finaliser at the moment it matters."""
    assert set(finaliser.QUESTIONS) == set(ADVISORY)


def test_the_summary_shows_the_covariates_the_window_and_the_unused_columns(tmp_path, finaliser):
    batches = {"train": ["a"] * 8, "val": ["b"] * 4, "test": ["c"] * 4}
    directory = a_split(tmp_path, sequences=RAGGED, extra={**COVARIATES, "batch": batches})
    sequences = {**ragged_declaration()["sequences"],
                 "positional_window": {"length": 6, "anchor": "start", "pad": True}}
    text = finaliser.summary(finalise(directory, covariate_declaration(sequences=sequences)))
    assert "covariate  signal: mean signal over the window; training 0 to 3.3, 8 distinct" in text
    assert "window     positional features read 6 bases from the start; shorter sequences padded" in text
    assert "unused     columns in the split files that nothing reads: batch" in text


def test_the_summary_describes_a_continuous_target_by_its_distribution(tmp_path, finaliser):
    text = finaliser.summary(finalise(continuous_split(tmp_path), continuous_declaration()))
    assert "target     change in binding, continuous: the change in the factor's binding" in text
    assert "mean" in text and "negative" not in text
    assert "filters, rows before and after each step:" in text
    assert "delivered: 16 -> 16  (lost 0)" in text
    assert "smallest detectable correlation" in text


# ------------------------------------------------------------------------------ the run config


def _config(tmp_path: Path, splits: Path, continuous: bool = False, **dataset) -> Path:
    """A run config on `splits`. A continuous target has no class weighting and its own linear
    model."""
    document = {
        "name": "example",
        "dataset": {"task": "example", "splits": str(splits), "blinded": False, **dataset},
        "grid": {"seeds": [1, 2], "labels": ["real"], "train_fraction": 0.8,
                 "n_bootstrap": 0, "permutation_scope": "seed",
                 **({} if continuous else {"class_weight": "balanced"}),
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "analyses": ANALYSES,
        "arms": {"positional": {"models": ["l1_linear" if continuous else "l1_logistic"]}},
    }
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture
def experiment_config():
    from seqmaestro.pipeline import config as experiment_config

    return experiment_config


def test_a_config_takes_its_columns_from_the_dataset(tmp_path, experiment_config):
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    config = experiment_config.load(_config(tmp_path, directory))
    assert (config.dataset.sequence_column, config.dataset.target_column) == ("sequence", "is_positive")
    assert config.dataset.weight_column is None
    assert config.to_mapping()["dataset"]["target_column"] == "is_positive"


def test_a_config_that_contradicts_its_dataset_is_refused(tmp_path, experiment_config):
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    with pytest.raises(experiment_config.ConfigError, match="records 'is_positive'"):
        experiment_config.load(_config(tmp_path, directory, target_column="label"))


def test_a_config_restating_its_columns_correctly_is_accepted(tmp_path, experiment_config):
    """A run records its config with the columns stated, so a config that restates them correctly
    must load when the run is read back."""
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    config = experiment_config.load(_config(tmp_path, directory, sequence_column="sequence",
                                            target_column="is_positive"))
    assert config.dataset.target_column == "is_positive"


def test_a_config_records_the_covariates_its_dataset_declares(tmp_path, experiment_config):
    directory = a_split(tmp_path, extra=COVARIATES)
    finalise(directory, covariate_declaration())
    config = experiment_config.load(_config(tmp_path, directory))
    assert config.dataset.covariate_columns == ("signal", "peaks")
    assert config.dataset.columns.covariates == ("signal", "peaks")
    assert config.to_mapping()["dataset"]["covariate_columns"] == ["signal", "peaks"]
    # The recorded form restates them, and must load back.
    restated = experiment_config.load(_config(tmp_path, directory,
                                              covariate_columns=["signal", "peaks"]))
    assert restated.to_yaml() == config.to_yaml()
    with pytest.raises(experiment_config.ConfigError, match="covariate_columns"):
        experiment_config.load(_config(tmp_path, directory, covariate_columns=["signal"]))


def test_a_config_on_a_dataset_without_covariates_records_none(tmp_path, experiment_config):
    """A recorded config restates the covariates only where the dataset has some."""
    directory = a_split(tmp_path)
    finalise(directory, a_declaration())
    config = experiment_config.load(_config(tmp_path, directory))
    assert config.dataset.covariate_columns == ()
    assert "covariate_columns" not in config.to_mapping()["dataset"]


def test_a_recorded_config_states_the_kind_of_its_target(tmp_path, experiment_config):
    """The copy of a config recorded beside a run states the kind of target, so the run describes
    itself, and a config that states another kind than its dataset's is refused."""
    binary = a_split(tmp_path / "binary")
    finalise(binary, a_declaration())
    config = experiment_config.load(_config(tmp_path / "binary", binary))
    assert config.to_mapping()["dataset"]["target_kind"] == BINARY
    assert config.dataset.columns.target_kind == BINARY

    continuous = continuous_split(tmp_path / "continuous")
    finalise(continuous, continuous_declaration())
    config = experiment_config.load(_config(tmp_path / "continuous", continuous, continuous=True))
    assert config.to_mapping()["dataset"]["target_kind"] == CONTINUOUS
    assert config.grid.class_weight is None and "class_weight" not in config.to_mapping()["grid"]
    with pytest.raises(experiment_config.ConfigError, match="records 'continuous'"):
        experiment_config.load(_config(tmp_path / "continuous", continuous, continuous=True,
                                       target_kind="binary"))


def test_a_continuous_config_refuses_a_class_weighting_and_a_binary_model(tmp_path, experiment_config):
    import yaml

    directory = continuous_split(tmp_path)
    finalise(directory, continuous_declaration())
    path = _config(tmp_path, directory, continuous=True)
    document = yaml.safe_load(path.read_text())
    document["grid"]["class_weight"] = "balanced"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(experiment_config.ConfigError, match="class_weight"):
        experiment_config.load(path)
    path = _config(tmp_path, directory, continuous=True)
    path.write_text(path.read_text().replace("l1_linear", "l1_logistic"))
    with pytest.raises(experiment_config.ConfigError, match="fits a binary target"):
        experiment_config.load(path)


def test_a_config_naming_an_unfinalised_dataset_is_refused(tmp_path, experiment_config):
    with pytest.raises(experiment_config.ConfigError, match="not a usable dataset"):
        experiment_config.load(_config(tmp_path, a_split(tmp_path)))


# ------------------------------------------------------------------------------ the group map


def test_the_group_map_is_one_row_per_sequence_and_group():
    frame = pd.DataFrame({"sequence": ["AA", "CC", "GG"], "gene": ["x", "x", "y"]})
    mapped = group_map(frame, "sequence", ["gene"])
    assert dict(zip(mapped["sequence"], mapped["group"], strict=True)) == {"AA": 0, "CC": 0, "GG": 1}


def test_the_group_map_joins_relations_that_overlap():
    """Two rows can share a gene without sharing a window, or share a window without sharing a
    gene. A split keyed on only one relation would let rows linked by the other straddle splits."""
    frame = pd.DataFrame({"sequence": ["AA", "AA", "CC", "GG"], "gene": ["x", "y", "y", "z"]})
    mapped = group_map(frame, "sequence", ["sequence", "gene"])
    groups = dict(zip(mapped["sequence"], mapped["group"], strict=True))
    assert groups["AA"] == groups["CC"] and groups["GG"] != groups["AA"]


def test_a_sequence_in_two_groups_gets_a_row_for_each():
    """Only the declared relations are used, so a window carried by two genes is not merged into
    one group. The map's two rows record that the declared grouping does not decide where the
    window goes."""
    frame = pd.DataFrame({"sequence": ["AA", "AA", "CC"], "gene": ["x", "y", "y"]})
    mapped = group_map(frame, "sequence", ["gene"])
    assert sorted(mapped.loc[mapped["sequence"] == "AA", "group"]) == [0, 1]


def test_a_missing_value_joins_nothing_in_the_group_map():
    """Rows whose gene is unknown must not all land in one group, which is what reading the file
    with empty fields as text would do."""
    frame = pd.DataFrame({"sequence": ["AA", "CC", "GG"], "gene": [None, None, "y"]})
    mapped = group_map(frame, "sequence", ["gene"])
    assert mapped["group"].nunique() == 3


def test_the_group_map_refuses_a_relation_the_dataset_does_not_carry():
    frame = pd.DataFrame({"sequence": ["AA"], "gene": ["x"]})
    with pytest.raises(KeyError, match="domain"):
        group_map(frame, "sequence", ["domain"])


def test_sequences_sharing_a_stretch_are_grouped_whatever_their_lengths():
    """A stretch shared at different offsets, by sequences of different lengths, joins them, and
    joining is transitive. A sequence shorter than the stretch shares nothing."""
    stretch = "ACGTTGCAAC"
    sequences = ["GG" + stretch, "TTTTT" + stretch + "CCCC", "AAAA" + stretch[:4], "ACG",
                 stretch[:6] + "GATTACA", "GATTACA" + "TT"]
    groups = shared_stretch_groups(sequences, len(stretch))
    assert groups[0] == groups[1]
    assert len(set(groups)) == 5
    joined = shared_stretch_groups(sequences, 7)
    assert joined[4] == joined[5] and joined[0] == joined[1] != joined[4]


# ---------------------------------------------------------------- the declarations in the repository
#
# A clone holds the example's. A machine with datasets of its own also holds theirs, which git
# ignores, and the tests below check those too.

DECLARATIONS = sorted((Path(__file__).resolve().parents[1] / "experiments" / "prepare"
                       / "declarations").glob("*.yaml"))


def test_there_is_a_declaration_to_check():
    """A glob that matched nothing would make the two tests below pass without reading anything."""
    assert DECLARATIONS


@pytest.mark.parametrize("path", DECLARATIONS, ids=lambda path: path.stem)
def test_every_declaration_here_still_validates(path):
    """A declaration is the only record of what a dataset's builder decided. If a declaration field
    is renamed in `seqmaestro.dataset`, only this test would notice before the dataset was
    finalised again."""
    import yaml

    validate_declared(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("path", DECLARATIONS, ids=lambda path: path.stem)
def test_a_declaration_is_named_for_the_split_it_describes(path):
    """Declarations are named `<dataset>__<splits>.yaml`, so the file that describes a split
    directory can be found from its path, and two declarations cannot collide."""
    dataset, separator, splits = path.stem.partition("__")
    assert separator and dataset and splits, f"{path.name} is not <dataset>__<splits>.yaml"
