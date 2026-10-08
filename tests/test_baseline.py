"""The composition baseline: its featuriser, the standardised L1 fit it uses, and its stage.

The baseline is a reference line, so these tests check that it is always built the same way: from
the base fractions of the window, with a standardised L1 fit whose answer does not depend on the
units of its columns, as one cell per seed on the rows the arms fit on, and written where no arm
analysis would count it as an arm.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import CompositionFeaturiser
from seqmaestro.harness.features import load_features
from seqmaestro.models import L1Logistic
from tests.planted import ANALYSES, SEQUENCE, TARGET, planted_dataset, planted_frame

# --------------------------------------------------------------------------------------------
# The featuriser
# --------------------------------------------------------------------------------------------


def test_each_column_is_the_fraction_of_the_sequence_one_symbol_makes_up():
    featuriser = CompositionFeaturiser().fit(pd.Series(["AACG", "TTTT"]))
    values = featuriser.transform(pd.Series(["AACG", "GGTA"]))

    assert list(values.columns) == ["fraction_A", "fraction_C", "fraction_G", "fraction_T"]
    assert values.iloc[0].tolist() == [0.5, 0.25, 0.25, 0.0]
    assert values.iloc[1].tolist() == [0.25, 0.0, 0.5, 0.25]
    assert np.allclose(values.sum(axis=1), 1.0)


def test_the_alphabet_is_the_training_rows_and_nothing_is_assumed():
    """Each symbol in the training rows becomes a column. A symbol the training rows lack raises
    at transform time, so it cannot be silently left out of the fractions."""
    featuriser = CompositionFeaturiser().fit(pd.Series(["ACGN", "ACGT"]))
    assert featuriser.alphabet == ["A", "C", "G", "N", "T"]

    narrow = CompositionFeaturiser().fit(pd.Series(["ACGT"]))
    with pytest.raises(ValueError, match="absent when it was fitted"):
        narrow.transform(pd.Series(["ACGN"]))


def test_the_records_describe_a_composition_feature():
    records = CompositionFeaturiser().fit(pd.Series(["ACGT"])).records
    assert {record.kind for record in records} == {"composition"}
    assert [record.params for record in records] == [{"symbol": s} for s in "ACGT"]
    assert all(record.provenance == "selected" for record in records)


# --------------------------------------------------------------------------------------------
# The L1 fit, on standardised columns
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def columns():
    """Base fractions of a planted-motif sample, and its labels."""
    frame = planted_frame(300, np.random.default_rng(11))
    X = CompositionFeaturiser().fit_transform(frame[SEQUENCE])
    return X, frame[TARGET].to_numpy()


def test_standardising_makes_the_fit_blind_to_the_units_of_a_column(columns):
    """On raw columns, a column's scale decides how hard the penalty presses on it, so the same data
    in other units would select differently. On standardised columns, rescaling a column leaves the
    selection, the coefficients and the predictions unchanged."""
    X, y = columns
    rescaled = X.assign(fraction_G=X["fraction_G"] * 1000.0)
    one = L1Logistic(C=0.05).fit(X, y)
    other = L1Logistic(C=0.05).fit(rescaled, y)

    assert one.importances().keys() == other.importances().keys()
    assert one.importances() == pytest.approx(other.importances(), rel=1e-5)
    assert one.predict(X) == pytest.approx(other.predict(rescaled), abs=1e-7)


def test_the_importance_kind_names_the_scale():
    assert L1Logistic().importance_kind == "l1_coefficient_per_sd"


# --------------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------------


def a_config(tmp_path, seeds, arms: dict, **sections):
    """The config of a planted run at these seeds, as a file."""
    dataset = planted_dataset(tmp_path)
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": list(seeds), "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 5, "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "analyses": ANALYSES,
        "arms": arms, **sections}))
    return source


def test_the_baseline_writes_one_standard_cell_per_seed(runners, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = runners.config.load(a_config(tmp_path, (1, 2), {"positional": {"models": ["l1"]}}))
    (run_dir / "config.yaml").write_text(config.to_yaml())
    assert runners.baseline.main(["--run-dir", str(run_dir)]) == 0
    assert (run_dir / "invocations" / "baseline.json").exists()
    assert not (run_dir / "failures").exists(), "a call that ends clean leaves no failures file"

    cells = sorted((run_dir / "baseline").glob("seed_*"))
    assert [cell.name for cell in cells] == ["seed_1", "seed_2"]
    assert not (run_dir / "results").exists(), "outside results/, so no analysis reads it as an arm"
    for cell in cells:
        assert {path.name for path in cell.iterdir()} == {
            "run.json", "metrics.json", "predictions.npz", "features.json", "feature_values.npz",
            "fitted.pkl"}
        run = json.loads((cell / "run.json").read_text())
        assert (run["model"], run["pool"]) == ("composition_baseline", "composition")
        assert run["sources"] is None and run["labels"] == "real"
        assert {key: run["model_settings"][key] for key in ("C", "class_weight")} == \
               {"C": 1.0, "class_weight": "balanced"}
        assert run["importance_kind"] == "l1_coefficient_per_sd"
        records, header = load_features(cell / "features.json")
        assert header["importance_kind"] == "l1_coefficient_per_sd"
        assert {record.kind for record in records} <= {"composition"}

    first, second = (json.loads((cell / "run.json").read_text()) for cell in cells)
    assert first["seed"] != second["seed"]
    assert first["n_fit_rows"] == second["n_fit_rows"] == int(0.8 * 160), \
        "the arms' subsample, one draw per seed"


def test_the_driver_runs_the_baseline_first_and_expects_its_cells(runners, tmp_path):
    config = runners.config.load(a_config(
        tmp_path, (4, 5),
        {"positional+kmer": {"models": ["xgboost"]}, "positional": {"models": ["deft_or_cart"]}},
        sources={"kmer": {"k_range": [1, 3], "normalise": "counts"}}))

    stages = runners.driver.build_stages(config, tmp_path / "run")
    assert stages[0].name == "baseline"
    assert "run_baseline.py" in stages[0].commands[0].argv[1]
    assert sorted(str(cell) for cell in config.expected_cells()) == sorted([
        "baseline/seed_4", "baseline/seed_5",
        *(f"results/xgboost/positional+kmer/{labels}/seed_{seed}"
          for labels in ("real", "permuted") for seed in (4, 5)),
        *(f"results/deft_or_cart/positional/{labels}/seed_{seed}"
          for labels in ("real", "permuted") for seed in (4, 5)),
    ])


def test_the_baseline_is_fixed_whatever_the_run(runners):
    """The seed and class weight come from the run, because the baseline shares them with the arms.
    Its own settings, such as `C`, are fixed."""
    model = runners.baseline.baseline_model(seed=7, target_kind="binary", class_weight="balanced")
    assert (model.C, model.seed) == (1.0, 7)
    assert model.describe()["model"] == "l1_logistic"
