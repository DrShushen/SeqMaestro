"""Covariates through the in-process arms: the featuriser, the pool member, the config key, and one
run on the planted dataset's covariate variant, fitted, reduced, profiled and revealed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import FIXED_POOLS, CovariateFeaturiser, make_featuriser, with_covariates
from seqmaestro.features.base import Inputs
from seqmaestro.harness.features import load_features
from seqmaestro.harness.fitted import load_fitted
from seqmaestro.harness.heldout import TEST_SPLIT, reveal_record
from seqmaestro.report import facts
from tests.planted import ANALYSES, COVARIATES, planted_dataset

DESCRIBED = {"signal": "a signal over the window", "peaks": "peaks overlapping the window"}


def rows(**covariates) -> Inputs:
    sequences = pd.Series(["ACGT", "TTGA", "GGCA", "CATG"])
    return Inputs(sequences, pd.DataFrame(covariates, index=sequences.index))


# --------------------------------------------------------------------------------------------
# The featuriser and the pool member
# --------------------------------------------------------------------------------------------


def test_each_covariate_becomes_one_feature_carrying_its_values_and_its_description():
    featuriser = CovariateFeaturiser(DESCRIBED)
    X = featuriser.fit_transform(rows(signal=[0.5, 1.5, 2.5, 3.5], peaks=[1, 0, 2, 1]))
    assert list(X.columns) == ["covariate_signal", "covariate_peaks"]
    assert X["covariate_peaks"].tolist() == [1.0, 0.0, 2.0, 1.0]
    record = featuriser.records[0]
    assert (record.kind, record.params, record.description, record.provenance) == \
           ("covariate", {"column": "signal"}, DESCRIBED["signal"], "selected")


def test_a_covariates_pool_needs_rows_that_carry_covariates():
    with pytest.raises(ValueError, match="carry no covariates"):
        CovariateFeaturiser(DESCRIBED).fit(pd.Series(["ACGT", "TTGA"]))


def test_a_covariate_with_no_description_is_refused():
    with pytest.raises(ValueError, match="No description"):
        CovariateFeaturiser({"signal": "d"}).fit(rows(signal=[1.0, 2.0, 3.0, 4.0], depth=[1, 2, 3, 4]))


def test_a_fitted_covariate_featuriser_reads_its_own_covariates_and_refuses_rows_without():
    featuriser = CovariateFeaturiser(DESCRIBED).fit(rows(signal=[0.5, 1.5, 2.5, 3.5]))
    wider = rows(peaks=[9, 9, 9, 9], signal=[4.0, 3.0, 2.0, 1.0])
    assert featuriser.transform(wider)["covariate_signal"].tolist() == [4.0, 3.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="no covariate signal"):
        featuriser.transform(pd.Series(["ACGT"]))


def test_the_covariates_compose_with_a_sequence_pool_and_come_last():
    featuriser = make_featuriser("positional+covariates", covariates=DESCRIBED)
    X = featuriser.fit_transform(rows(signal=[0.5, 1.5, 2.5, 3.5]))
    assert X.shape == (4, 4 * 4 + 1)
    assert X.columns[-1] == "covariate_signal" and X.columns[0] == "pos_0_A"


def test_make_featuriser_asks_for_the_covariates_exactly_when_the_pool_holds_them():
    with pytest.raises(ValueError, match="no covariates were given"):
        make_featuriser("positional+covariates")
    with pytest.raises(ValueError, match="does not hold them"):
        make_featuriser("positional", covariates=DESCRIBED)


@pytest.mark.parametrize("pool", [pool for pool in FIXED_POOLS if "covariates" in pool])
def test_every_covariate_pool_names_its_features_legally_and_keeps_the_rows_index(pool):
    sequences = pd.Series(["ACGT", "TGCA", "GGCC"], index=[17, 42, 5])
    rows = Inputs(sequences, pd.DataFrame({"signal": [0.1, 0.9, 0.4]}, index=sequences.index))
    X = make_featuriser(pool, k_range=(1, 2), kmer_normalise="counts", covariates=DESCRIBED).fit_transform(rows)
    assert all(name.isidentifier() for name in X.columns)
    assert list(X.index) == [17, 42, 5] and X.columns[-1] == "covariate_signal"


def test_every_pool_keeps_its_name_and_gains_one_with_the_covariates():
    assert with_covariates("positional+kmer") == "positional+kmer+covariates"
    assert with_covariates("llm_generated") == "llm_generated+covariates"
    assert {"positional", "positional+kmer", "positional+covariates", "covariates"} <= set(FIXED_POOLS)


# --------------------------------------------------------------------------------------------
# The config: what each arm does with the covariates
# --------------------------------------------------------------------------------------------


def a_config(tmp_path: Path, dataset, arms: dict, **sections) -> Path:
    held = {member for pool in arms for member in pool.split("+")}
    sources = {**({"kmer": {"k_range": [1, 3], "normalise": "counts"}} if "kmer" in held else {}),
               **({"llm_generated": {"proposal_rounds": 3, "n_per_round": 10, "n_examples": 20}}
                  if "llm_generated" in held else {})}
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10,
                 "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "sources": sources,
        "analyses": ANALYSES,
        "arms": arms,
        **sections,
    }, sort_keys=False))
    return source


def test_the_covariates_are_a_source_and_a_run_that_hands_them_to_no_pool_is_warned(runners, tmp_path):
    """A pool holds the covariates by naming them, so the config says which arms are handed them.
    A dataset that declares covariates no pool holds is warned about, not refused."""
    dataset = planted_dataset(tmp_path, covariates=True)
    config = runners.config.load(a_config(tmp_path, dataset, {
        "positional+covariates": {"models": ["l1_logistic"]}}))
    assert config.arm_cells() == [("l1_logistic", "positional+covariates")]
    assert config.warnings == ()
    config = runners.config.load(a_config(tmp_path, dataset, {"positional": {"models": ["l1_logistic"]}}))
    assert config.arm_cells() == [("l1_logistic", "positional")]
    assert len(config.warnings) == 1 and "no pool in `arms` holds them" in config.warnings[0]


def test_a_pool_with_the_covariates_and_one_without_run_a_model_each_way(runners, tmp_path):
    dataset = planted_dataset(tmp_path, covariates=True)
    config = runners.config.load(a_config(tmp_path, dataset, {
        "positional": {"models": ["l1_logistic"]},
        "positional+covariates": {"models": ["l1_logistic"]},
        "llm_generated": {"models": ["ebm"]},
    }))
    assert config.arm_cells() == [("l1_logistic", "positional"),
                                  ("l1_logistic", "positional+covariates"),
                                  ("ebm", "llm_generated")]


def test_a_pool_with_the_covariates_is_refused_on_a_dataset_without_them(runners, tmp_path):
    dataset = planted_dataset(tmp_path)
    with pytest.raises(runners.config.ConfigError, match="declares none"):
        runners.config.load(a_config(tmp_path, dataset, {
            "positional+covariates": {"models": ["l1_logistic"]}}))
    config = runners.config.load(a_config(tmp_path, dataset, {"positional": {"models": ["ebm"]}}))
    assert config.warnings == (), "nothing to warn about where the dataset declares no covariates"


def test_the_tree_takes_the_covariates_like_every_other_model(runners, tmp_path):
    dataset = planted_dataset(tmp_path, covariates=True)
    config = runners.config.load(a_config(tmp_path, dataset, {
        "positional": {"models": ["deft_or_cart"]},
        "positional+covariates": {"models": ["deft_or_cart"]}}))
    assert config.arm_cells() == [("deft_or_cart", "positional"),
                                  ("deft_or_cart", "positional+covariates")]
    stages = {stage.name: stage for stage in runners.driver.build_stages(config, tmp_path / "run")}
    argv = stages["deft_or_cart/positional+covariates"].commands[0].argv
    assert argv[argv.index("--arms") + 1] == "deft_or_cart/positional+covariates"


def test_the_covariates_come_last_in_a_pools_name(runners, tmp_path):
    dataset = planted_dataset(tmp_path, covariates=True)
    with pytest.raises(runners.config.ConfigError, match="Write it `positional\\+covariates`"):
        runners.config.load(a_config(tmp_path, dataset, {
            "covariates+positional": {"models": ["l1_logistic"]}}))


def test_the_driver_hands_each_arm_its_pool(runners, tmp_path):
    dataset = planted_dataset(tmp_path, covariates=True)
    config = runners.config.load(a_config(tmp_path, dataset, {
        "positional": {"models": ["l1_logistic"]},
        "positional+covariates": {"models": ["l1_logistic"]},
        "llm_generated+covariates": {"models": ["ebm"]},
        "positional+kmer": {"models": ["xgboost"]},
    }))
    commands = {stage.name: stage.commands[0].argv
                for stage in runners.driver.build_stages(config, tmp_path / "run")}
    for model, pool in config.arm_cells():
        argv = commands[f"{model}/{pool}"]
        assert argv[argv.index("--arms") + 1] == f"{model}/{pool}"
        assert argv[argv.index("--arms") + 2].startswith("--"), "one arm per process"
    assert not any("--sequence-column" in argv for argv in commands.values())


# --------------------------------------------------------------------------------------------
# One run, end to end
# --------------------------------------------------------------------------------------------

RELIABILITY = {"pairs": 3, "permutations": 200, "cluster_rho": 0.8, "max_modal_share": 0.99,
               "planted_count": 3, "planted_signal_sd": 1.5,
               "check_arm": "l1_logistic/positional+covariates", "seed": 0}


def test_covariate_arms_are_fitted_reduced_profiled_and_revealed(runners, tmp_path):
    dataset = planted_dataset(tmp_path, covariates=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = runners.config.load(a_config(
        tmp_path, dataset, {"positional": {"models": ["l1_logistic"]},
                            "positional+covariates": {"models": ["l1_logistic"]}},
        reduction={"n_features": 3}, reliability=RELIABILITY))
    (run_dir / "config.yaml").write_text(config.to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main(flags) == 0

    cells = run_dir / "results" / "l1_logistic"
    with_them, without = cells / "positional+covariates" / "real" / "seed_1", \
        cells / "positional" / "real" / "seed_1"
    kinds = {record.kind for record in load_features(with_them / "features.json")[0]}
    assert "covariate" in kinds, "the signal covariate carries the class, so the model uses it"
    assert "covariate" not in {record.kind for record in load_features(without / "features.json")[0]}
    assert json.loads((with_them / "run.json").read_text())["sources"] == ["positional", "covariates"]
    auroc = {cell: json.loads((cell / "metrics.json").read_text())["val"]["auroc"]
             for cell in (with_them, without)}
    assert auroc[with_them] > auroc[without], "the covariate adds what the motif does not carry"

    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "2"]) == 0
    prepared = json.loads((run_dir / "reliability" / "l1_logistic" / "positional+covariates" /
                           "prepare.json").read_text())
    assert prepared["candidates"]["p"] == 80 + len(COVARIATES), \
        "the covariates are among the arm's candidates"

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert reveal_record(run_dir)["completed_utc"] is not None
    reduced = run_dir / "reduction" / "l1_logistic" / "positional+covariates" / "reduced"
    assert TEST_SPLIT in json.loads((reduced / "metrics.json").read_text())

    arm = load_fitted(reduced)
    test = pd.read_csv(dataset.directory / f"{TEST_SPLIT}.csv")
    with np.load(reduced / "predictions.npz") as predictions:
        assert np.allclose(arm.predict(dataset.inputs(test)), predictions[f"{TEST_SPLIT}_score"])
    if any(name.startswith("covariate_") for name in arm.featuriser.feature_names):
        with pytest.raises(ValueError, match="no covariate"):
            arm.predict(test[dataset.sequence_column])



def test_a_model_is_shown_the_covariates_by_name_only_where_its_pool_holds_them(tmp_path):
    """What a blinded run's column names are checked over (`seqmaestro.llm.blinding_leaks`)."""
    dataset = planted_dataset(tmp_path, covariates=True)
    plain = {dataset.sequence_column: None, dataset.target_column: None}
    assert dataset.shown_columns(["llm_generated"]) == plain
    assert dataset.shown_columns(["llm_generated", "llm_generated+covariates"]) == {
        **plain, **dataset.covariates}

def test_the_report_states_each_covariate_at_stage_1(tmp_path):
    dataset = planted_dataset(tmp_path, covariates=True)
    unblinded = {"blinded": False, "withheld": None}
    stage1 = facts.stage1(SimpleNamespace(dataset=dataset, blinding=unblinded))
    assert list(stage1["covariates"]) == ["signal", "noise"]
    assert stage1["covariates"]["signal"]["description"] == dataset.covariates["signal"]
    assert set(stage1["covariates"]["signal"]["training"]) >= {"min", "max", "mean", "sd"}
    assert facts.stage1(SimpleNamespace(dataset=planted_dataset(tmp_path / "plain"),
                                        blinding=unblinded))["covariates"] == {}
