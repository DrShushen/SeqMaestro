"""A continuous target through a run: the models, the metrics, the target's arithmetic, what the
proposer is shown, the bundle contract, and the stages on a planted continuous dataset, the panel
and the reveal among them.

The binary path is covered by every other test module. These check what differs for a continuous
target, and that nothing written for labels runs on one unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import make_featuriser
from seqmaestro.harness.bundle import read_bundle
from seqmaestro.harness.direction import DECREASES_TARGET, INCREASES_TARGET
from seqmaestro.harness.features import load_features
from seqmaestro.harness.heldout import score_cell
from seqmaestro.harness.metrics import regression_metrics
from seqmaestro.harness.runner import score_validation
from seqmaestro.models import make_model, models_for
from seqmaestro.proposer.prompt import Example, render_request, sample_examples
from seqmaestro.reduction.arm import NoFeatures, rebuild_model
from seqmaestro.target import BINARY, CONTINUOUS, target_for
from tests.planted import ANALYSES, SEQUENCE, TARGET, planted_dataset, run_cell


def planted_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Two informative columns, one that raises the target and one that lowers it, and a third
    that does nothing."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["up", "down", "idle"])
    return X, 2.0 * X["up"].to_numpy() - 1.0 * X["down"].to_numpy() + rng.normal(0, 0.5, n)


# ------------------------------------------------------------------------------------ the models


@pytest.mark.parametrize("name", ["l1_linear", "ebm", "xgboost"])
def test_each_continuous_model_predicts_values_and_points_the_right_way(name):
    X, y = planted_frame()
    model = make_model(name, target_kind=CONTINUOUS, seed=0).fit(X, y)
    predicted = model.predict(X)
    assert predicted.min() < 0 < predicted.max(), "values, not probabilities"
    assert np.corrcoef(predicted, y)[0, 1] > 0.8
    directions = model.directions(X)
    assert directions["up"] == INCREASES_TARGET and directions["down"] == DECREASES_TARGET
    assert model.describe()["target_kind"] == CONTINUOUS


def test_the_lasso_penalty_scales_with_the_target():
    """A fraction of the penalty that zeroes everything keeps the same features whatever the
    target's units, where a fixed alpha would not."""
    X, y = planted_frame()
    small = make_model("l1_linear", target_kind=CONTINUOUS).fit(X, y / 1000)
    large = make_model("l1_linear", target_kind=CONTINUOUS).fit(X, y * 1000)
    assert set(small.importances()) == set(large.importances()) >= {"up", "down"}


def test_a_model_of_the_wrong_kind_is_refused_by_name():
    with pytest.raises(ValueError, match="the models that do are ebm, l1_linear, pysr, xgboost"):
        make_model("l1_logistic", target_kind=CONTINUOUS)
    with pytest.raises(ValueError, match="no classes to weight"):
        make_model("ebm", target_kind=CONTINUOUS, class_weight="balanced")
    assert models_for(CONTINUOUS) == ("ebm", "l1_linear", "pysr", "xgboost")


def test_a_continuous_model_rebuilds_from_its_description_and_the_kind_is_checked():
    X, y = planted_frame()
    fitted = make_model("l1_linear", target_kind=CONTINUOUS, seed=2).fit(X, y)
    description = fitted.describe()
    # The fitted alpha is not a setting: a rebuild derives it again from its own rows.
    assert rebuild_model(description, 2, CONTINUOUS).describe()["strength"] == description["strength"]
    with pytest.raises(ValueError):
        rebuild_model(description, 2, BINARY)


def test_xgboost_stops_early_on_its_own_rows_and_keeps_only_the_trees_up_to_the_best_round():
    """On a heavy-tailed target that nothing predicts, the trees stop long before the ceiling; on a
    planted one they grow further. The contributions TreeSHAP reads add up to the predictions, so
    they come from the trees kept. The fit is fixed by its seed, and equal weights reproduce the
    unweighted fit, the part set aside included."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["up", "down", "idle"])
    noise = make_model("xgboost", target_kind=CONTINUOUS, seed=1).fit(X, rng.standard_t(2, 400))
    X_signal, y_signal = planted_frame()
    signal = make_model("xgboost", target_kind=CONTINUOUS, seed=1).fit(X_signal, y_signal)
    assert noise.describe()["trees_kept"] < signal.describe()["trees_kept"] <= 300

    import xgboost as xgb
    contributions = signal._model.get_booster().predict(xgb.DMatrix(X_signal), pred_contribs=True)
    assert np.allclose(contributions.sum(axis=1), signal.predict(X_signal), atol=1e-4)

    again = make_model("xgboost", target_kind=CONTINUOUS, seed=1).fit(X_signal, y_signal)
    weighted = make_model("xgboost", target_kind=CONTINUOUS, seed=1).fit(
        X_signal, y_signal, sample_weight=np.full(len(y_signal), 3.0))
    assert np.array_equal(again.predict(X_signal), signal.predict(X_signal))
    assert np.allclose(weighted.predict(X_signal), signal.predict(X_signal), atol=1e-5)


def test_xgboost_rebuilds_from_its_description_but_not_from_one_that_predates_stopping_early():
    X, y = planted_frame()
    description = make_model("xgboost", target_kind=CONTINUOUS, seed=2).fit(X, y).describe()
    # The number of trees kept is found by the fit, so a rebuild finds its own.
    assert rebuild_model(description, 2, CONTINUOUS).describe()["early_stopping_rounds"] == 20
    older = {key: value for key, value in description.items()
             if key not in {"validation_fraction", "early_stopping_rounds", "trees_kept"}}
    with pytest.raises(ValueError, match="early_stopping_rounds=absent"):
        rebuild_model(older, 2, CONTINUOUS)
    with pytest.raises(ValueError, match="validation_fraction"):
        make_model("xgboost", target_kind=CONTINUOUS, validation_fraction=1.0)


def test_no_features_predicts_the_weighted_mean_of_a_continuous_target():
    model = NoFeatures(CONTINUOUS).fit(pd.DataFrame(index=range(4)), [1.0, 2.0, 3.0, 6.0],
                                       sample_weight=np.array([1.0, 1.0, 1.0, 3.0]))
    assert model.predict(pd.DataFrame(index=range(2))).tolist() == [4.0, 4.0]
    assert model.describe()["prediction"] == 4.0


# ------------------------------------------------------------------------------------ the metrics


def test_regression_metrics_are_right_on_known_cases():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = regression_metrics(y, y)
    assert perfect["r2"] == 1.0 and perfect["rmse"] == 0.0 and perfect["spearman_rho"] == 1.0
    assert regression_metrics(y, np.full(4, y.mean()))["r2"] == 0.0
    assert np.isnan(regression_metrics(y, np.full(4, 2.5))["pearson_r"])
    reversed_order = regression_metrics(y, y[::-1])
    assert reversed_order["spearman_rho"] == -1.0 and reversed_order["r2"] < 0


def test_regression_intervals_are_seeded_and_bracket_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    prediction = y + rng.normal(0, 0.5, 200)
    first = regression_metrics(y, prediction, n_bootstrap=50, seed=3)
    assert first == regression_metrics(y, prediction, n_bootstrap=50, seed=3)
    low, high = first["ci"]["r2"]
    assert low < first["r2"] < high


def test_a_continuous_validation_block_has_no_threshold():
    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    scored = score_validation(y, y + rng.normal(0, 0.1, 50), target_kind=CONTINUOUS, seed=0,
                              n_bootstrap=0)
    assert set(scored) == {"val"} and "threshold" not in scored["val"] and scored["val"]["r2"] > 0.9


def test_the_target_objects_agree_on_their_own_arithmetic():
    continuous, binary = target_for(CONTINUOUS), target_for(BINARY)
    assert (continuous.headline, continuous.chance, binary.headline, binary.chance) == \
        ("r2", 0.0, "auroc", 0.5)
    y = np.arange(20, dtype=float)
    values = np.column_stack([y, -y, np.ones(20)])
    assert continuous.association(values, y)[:2].tolist() == [1.0, -1.0]
    assert np.isnan(continuous.association(values, y)[2])
    with pytest.raises(ValueError):
        binary.values([0, 1, 0.5])


# ------------------------------------------------------------------------------------ the proposer


def test_examples_of_a_continuous_target_cover_its_range_and_show_its_values():
    y = np.linspace(-1, 1, 200)
    rows = pd.Series(["ACGT"] * 200)
    examples = sample_examples(rows, y, 10, seed=0, target_kind=CONTINUOUS)
    shown = sorted(example.label for example in examples)
    assert len(examples) == 10 and shown[0] < -0.8 and shown[-1] > 0.8
    assert all(isinstance(example.label, float) for example in examples)


def test_the_request_asks_for_the_target_of_its_kind():
    from seqmaestro.llm.context import TaskContext

    common = {"n_features": 2, "already_proposed": [], "sequence_column": "window",
              "target_column": "change"}
    continuous = render_request(TaskContext(task="t"), examples=[Example("ACGT", -0.0421)],
                                target_kind=CONTINUOUS, **common)
    binary = render_request(TaskContext(task="t"), examples=[Example("ACGT", 1)],
                            target_kind=BINARY, **common)
    assert "might predict the target's value" in continuous and "-> change=-0.0421" in continuous
    assert "might distinguish the classes" in binary and "-> change=1\n" in binary + "\n"


# ------------------------------------------------------------------------------------ the bundle


def test_a_continuous_bundle_carries_values_and_float_targets(tmp_path):
    from tests.test_bundle import write_bundle

    y = np.array([0.5, -1.25, 2.0])
    case = SimpleNamespace(y_val=y, p_val=np.array([0.4, -1.0, 3.5]))
    path = write_bundle(tmp_path / "b", case.y_val, case.p_val)
    bundle = read_bundle(path, CONTINUOUS, y_val=y)
    assert bundle.y_val.dtype == float and bundle.p_val.max() == 3.5
    with pytest.raises(ValueError, match=r"not in \[0, 1\]"):
        read_bundle(path, BINARY)


# ------------------------------------------------------------------------------------ the stages


def test_a_continuous_cell_is_scored_recorded_and_revealed_by_its_kind(tmp_path):
    dataset = planted_dataset(tmp_path, continuous=True)
    train, val = pd.read_csv(dataset.train_path), pd.read_csv(dataset.val_path)
    cell = tmp_path / "cell"
    run = run_cell(make_model("l1_linear", target_kind=CONTINUOUS, seed=1),
                    make_featuriser("positional"), train, val, columns=dataset.columns, seed=1,
                    out_dir=cell, n_bootstrap=5, permutation_repeats=1)
    assert run["target_kind"] == CONTINUOUS
    assert run["marginal_effect"]["statistic"] == "spearman"
    assert run["permutation_importance"]["metric"] == "r2"
    assert set(json.loads((cell / "metrics.json").read_text())) == {"val"}
    records, header = load_features(cell / "features.json")
    assert header["target_kind"] == CONTINUOUS
    assert {record.direction for record in records} <= {INCREASES_TARGET, DECREASES_TARGET, None}

    block = score_cell(cell, val[TARGET].to_numpy(), val[TARGET].to_numpy(),
                       target_kind=CONTINUOUS)
    assert block["r2"] == 1.0
    (cell / "metrics.json").write_text(json.dumps({"val": {}}))
    with pytest.raises(ValueError, match="continuous target, and the dataset's target is binary"):
        score_cell(cell, [0, 1], [0.2, 0.8], target_kind=BINARY)


def a_continuous_config(tmp_path, dataset, arms: dict, **sections) -> Path:
    """A config for a planted dataset with a continuous target, as a file."""
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10, "permutation_scope": "seed", "permutation_repeats": 0,
                 "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "analyses": ANALYSES,
        "arms": arms, **sections}))
    return source


def test_the_stages_fit_a_continuous_target_and_refuse_a_class_weighting(runners, tmp_path):
    dataset = planted_dataset(tmp_path, continuous=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = a_continuous_config(tmp_path, dataset, {"positional": {"models": ["l1", "ebm"]}})
    (run_dir / "config.yaml").write_text(runners.config.load(source).to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main([*flags, "--seeds", "1"]) == 0
    cells = run_dir / "results"
    real = json.loads((cells / "l1_linear/positional/real/seed_1/metrics.json").read_text())
    permuted = json.loads((cells / "l1_linear/positional/permuted/seed_1/metrics.json").read_text())
    # The motif moves the target by 1.5 noise SDs in half the rows, so a real fit explains much of
    # it and a fit to shuffled targets explains nothing.
    assert real["val"]["r2"] > 0.2 > permuted["val"]["r2"]
    baseline = json.loads((run_dir / "baseline/seed_1/run.json").read_text())
    assert baseline["model_settings"]["model"] == "l1_linear"

    # A continuous target has no classes to weight, and no logistic model, so a config with
    # either is refused when it is read.
    document = yaml.safe_load(source.read_text())
    document["grid"]["class_weight"] = "balanced"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(runners.config.ConfigError, match="class_weight"):
        runners.config.load(source)
    with pytest.raises(runners.config.ConfigError, match="fits a binary target"):
        runners.config.load(a_continuous_config(tmp_path, dataset,
                                                {"positional": {"models": ["l1_logistic"]}}))


def test_the_planted_continuous_dataset_describes_its_target(tmp_path):
    dataset = planted_dataset(tmp_path, continuous=True)
    assert dataset.target_kind == CONTINUOUS and dataset.classes == {}
    train = pd.read_csv(dataset.train_path)
    assert train[TARGET].dtype == float and train[SEQUENCE].str.len().nunique() == 1
    assert yaml.safe_dump(dataset.record["target"])


# ------------------------------------------------------------------------ the panel and the reveal


def test_tier2_measures_spearmans_rho_exactly_and_calls_what_goes_with_the_target():
    from scipy.stats import spearmanr

    from seqmaestro.reliability import tier2

    rng = np.random.default_rng(4)
    n = 150
    X = np.column_stack([rng.integers(0, 3, n), rng.normal(size=n), rng.integers(0, 2, n)]).astype(float)
    y = np.round(0.8 * X[:, 0] + rng.normal(0, 1, n), 1)  # rounded, so the target has ties
    out = tier2.profile(X, y, ["counts", "noise", "flag"], target_kind=CONTINUOUS, seed=1,
                        n_permutations=999)
    expected = [spearmanr(X[:, j], y).statistic for j in range(3)]
    assert np.allclose(out["rho"], expected)
    assert out["association"] == "rho" and "auc" not in out and "n_positive" not in out
    assert out["p_value"][0] == 1 / 1000 and out["p_value"][1] > 0.05
    assert out["direction"][0] == INCREASES_TARGET


def test_tier3_tests_necessity_on_squared_error():
    from seqmaestro.reliability import tier3

    rng = np.random.default_rng(2)
    y = rng.normal(size=300)
    good = y + rng.normal(0, 0.3, 300)
    worse = y + rng.normal(0, 1.0, 300)
    hurt = tier3.compare(y, good, worse, target_kind=CONTINUOUS, seed=0, n_sign_flips=2000)
    assert hurt["mean_delta_squared_error"] > 0 and hurt["p_t_one_sided"] < 0.001
    assert hurt["delta_r2"] > 0 and "n_at_clip" not in hurt
    same = tier3.compare(y, good, good.copy(), target_kind=CONTINUOUS, seed=0, n_sign_flips=2000)
    assert same["p_t_one_sided"] == 1.0


def test_the_planted_signal_draws_a_continuous_target_of_the_stated_strength():
    from seqmaestro.reliability import checks

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 10))
    recipe = checks.planted_recipe(X, [f"f{j}" for j in range(10)], np.full(10, 0.1), seed=1,
                                   n_planted=4, signal_sd=1.5, target_kind=CONTINUOUS)
    y = checks.draw_planted(X, recipe, seed=2)
    assert y.dtype == float and np.array_equal(y, checks.draw_planted(X, recipe, seed=2))
    block = X[:, recipe["columns"]]
    predictor = ((block - recipe["mean"]) / recipe["sd"]) @ np.asarray(recipe["beta"]) * recipe["scale"]
    explained = 1 - np.var(y - predictor) / np.var(y)
    assert explained == pytest.approx(1.5 ** 2 / (1.5 ** 2 + 1), abs=0.02)


def test_a_continuous_run_is_reduced_profiled_and_revealed(runners, tmp_path):
    from seqmaestro.harness.heldout import TEST_SPLIT, reveal_record

    dataset = planted_dataset(tmp_path, covariates=True, continuous=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = a_continuous_config(
        tmp_path, dataset, {"positional+covariates": {"models": ["l1_linear"]}},
        reduction={"n_features": 3},
        reliability={"pairs": 3, "permutations": 200, "cluster_rho": 0.8,
                     "max_modal_share": 0.99, "planted_count": 3, "planted_signal_sd": 1.5,
                     "check_arm": "l1_linear/positional+covariates", "seed": 0})
    config = runners.config.load(source)
    (run_dir / "config.yaml").write_text(config.to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main(flags) == 0
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    reduced = run_dir / "reduction" / "l1_linear" / "positional+covariates"
    record = json.loads((reduced / "reduction.json").read_text())
    assert "val_r2" in record["steps"][0] and "val_auroc" not in record["steps"][0]

    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "2"]) == 0
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert reveal_record(run_dir)["completed_utc"] is not None
    assert "r2" in json.loads((reduced / "reduced" / "metrics.json").read_text())[TEST_SPLIT]
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0

    panel = run_dir / "reliability" / "l1_linear" / "positional+covariates"
    features = pd.read_csv(panel / "profile_features.csv")
    assert {"rho", "abs_rho", "loco_mean_delta_squared_error", "loco_delta_r2"} <= set(features.columns)
    assert json.loads((panel / "tier3.json").read_text())["target_kind"] == CONTINUOUS
    checks_found = json.loads((panel / "checks.json").read_text())
    assert checks_found["global_null_tier2"]["p"] > 0
