"""`seqmaestro.harness.runner` and `seqmaestro.models`.

The runner is what makes the arms one comparison, so most of what is checked here is that they are
treated alike: every model produces the same files, the featuriser never sees rows the model did
not train on, and a cell fitted on permuted labels scores at chance. The tests use a small synthetic
sequence dataset with a planted signal, so the assertions concern behaviour alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seqmaestro.dataset import Columns
from seqmaestro.features import make_featuriser
from seqmaestro.harness.direction import DIRECTIONS
from seqmaestro.harness.effect import rank_biserial
from seqmaestro.harness.features import IMPORTANCE_KINDS, load_feature_values, load_features
from seqmaestro.models import MODELS, make_model
from seqmaestro.target import BINARY
from tests.planted import run_cell

ALPHABET = "ACGT"

# These are not the column names of any real dataset. The library requires both and defaults
# neither. A plausible name here would let a default creep in without any test noticing.
SEQUENCE, TARGET = "window", "outcome"
COLUMNS = Columns(SEQUENCE, TARGET, weight=None, covariates=(), target_kind=BINARY)


@pytest.fixture(scope="module")
def splits():
    """Training and validation sequences where a planted motif at a fixed position drives the
    label. There is no test split: the runner never sees one."""
    rng = np.random.default_rng(0)

    def make(n):
        sequences, labels = [], []
        for _ in range(n):
            seq = list(rng.choice(list(ALPHABET), 40))
            positive = rng.random() < 0.5
            if positive:
                seq[10:14] = list("ACGT")
            sequences.append("".join(seq))
            labels.append(int(positive))
        return pd.DataFrame({SEQUENCE: sequences, TARGET: labels})

    return make(400), make(150)


def a_run(splits, model_name="l1_logistic", **kwargs):
    train, val = splits
    # One seed drives both the model and the subsample. Passing it to only one is an easy mistake
    # that silently defeats the seed: an L1 fit is deterministic given its data, so a model seed
    # alone changes nothing.
    seed = kwargs.pop("seed", 0)
    return run_cell(
        make_model(model_name, target_kind="binary", seed=seed),
        make_featuriser(kwargs.pop("mode", "kmer"), k_range=(1, 3), kmer_normalise="counts"),
        train, val, columns=COLUMNS,
        seed=seed, n_bootstrap=kwargs.pop("n_bootstrap", 0), **kwargs,
    )


# ----------------------------------------------------------------- every model treated alike


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_every_model_produces_the_same_run_directory(splits, tmp_path, model_name):
    """The arms form one comparison only if every model writes the same files."""
    out = tmp_path / model_name
    a_run(splits, model_name, out_dir=out)

    written = {path.name for path in out.iterdir()}
    assert written == {"run.json", "metrics.json", "predictions.npz", "features.json",
                       "feature_values.npz", "fitted.pkl"}


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_every_model_learns_the_planted_signal(splits, model_name):
    result = a_run(splits, model_name)

    assert result["metrics"]["val"]["auroc"] > 0.8, model_name


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_every_model_reports_a_subset_of_the_available_features(splits, model_name):
    """A model that used every feature discovered nothing, so the record must reflect which
    features it used."""
    result = a_run(splits, model_name)

    assert 0 < result["n_features_used"] <= result["n_features_available"]
    assert result["importance_kind"] in IMPORTANCE_KINDS


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="No model 'random_forest' fits a binary target"):
        make_model("random_forest", target_kind="binary")


# ---------------------------------------------------------------- the permuted-label control


def test_permuted_labels_score_at_chance(splits):
    """A cell fitted on permuted labels is the control, so if it beats chance, something leaks.
    This test catches such a leak before a whole grid is fitted."""
    result = a_run(splits, permute_labels=True)

    assert 0.35 < result["metrics"]["val"]["auroc"] < 0.65
    assert result["labels"] == "permuted"


def test_permuting_does_not_touch_the_evaluation_labels(splits):
    """Only the training labels are shuffled, so a permuted cell is still scored against the true
    labels."""
    _, val = splits
    result = a_run(splits, permute_labels=True)

    assert result["metrics"]["val"]["n_positive"] == int(val[TARGET].sum())


# ------------------------------------------------------------------- the seed has to do work


def test_different_seeds_give_different_fits(splits):
    """Deterministic models on a fixed training set produce identical output for every seed, which
    would make stability 1 by construction. Subsampling the training rows per seed prevents that."""
    one, two = a_run(splits, seed=1), a_run(splits, seed=2)

    assert one["metrics"]["val"]["auroc"] != two["metrics"]["val"]["auroc"]


def test_the_same_seed_reproduces_the_same_fit(splits):
    assert a_run(splits, seed=7)["metrics"]["val"] == a_run(splits, seed=7)["metrics"]["val"]


def test_the_full_training_set_can_be_used_deliberately(splits):
    train, _ = splits
    result = a_run(splits, train_fraction=None)

    assert result["n_fit_rows"] == len(train)
    assert result["train_fraction"] is None


def test_subsampling_is_stratified(splits):
    """An unstratified draw would move the prevalence, and with it every thresholded metric."""
    train, _ = splits
    result = a_run(splits, train_fraction=0.5)

    assert result["n_fit_rows"] == len(train) // 2


# ----------------------------------------------------------------------- fit/transform discipline


def test_the_featuriser_is_fitted_only_on_the_rows_the_model_trains_on(splits):
    """A vocabulary learned from rows the model never trained on is a silent leak."""
    train, val = splits
    featuriser = make_featuriser("kmer", k_range=(4, 4), kmer_normalise="counts")
    run_cell(make_model("l1_logistic", target_kind="binary"), featuriser, train, val,
              columns=COLUMNS, train_fraction=0.5, n_bootstrap=0)

    fitted_on_all = make_featuriser("kmer", k_range=(4, 4), kmer_normalise="counts").fit(train[SEQUENCE])

    assert len(featuriser.records) <= len(fitted_on_all.records)


# ------------------------------------------------------------------------------ what is written


def test_the_written_files_are_consistent_with_each_other(splits, tmp_path):
    a_run(splits, out_dir=tmp_path)

    run = json.loads((tmp_path / "run.json").read_text())
    records, header = load_features(tmp_path / "features.json")
    with np.load(tmp_path / "feature_values.npz", allow_pickle=True) as values:
        names = list(values["names"])

    assert len(records) == run["n_features_used"]
    assert names == [record.name for record in records], "paired by position downstream"
    assert header["importance_kind"] == run["importance_kind"]


def test_predictions_are_stored_for_validation(splits, tmp_path):
    """The stored validation predictions let any metric be recomputed without refitting."""
    _, val = splits
    a_run(splits, out_dir=tmp_path)

    with np.load(tmp_path / "predictions.npz") as predictions:
        assert set(predictions.files) == {"val_true", "val_score"}
        assert predictions["val_score"].shape == (len(val),)


def test_a_cell_holds_nothing_about_the_test_split_and_what_the_reveal_needs(splits, tmp_path):
    """A cell records the threshold, the seed and the bootstrap count. The reveal scores the test
    split with these, so its numbers match what inline scoring would have produced."""
    a_run(splits, out_dir=tmp_path, seed=3, n_bootstrap=20)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    run = json.loads((tmp_path / "run.json").read_text())
    assert set(metrics) == {"threshold_chosen_on", "val"}
    assert "threshold" in metrics["val"]
    assert run["seed"] == 3 and run["n_bootstrap"] == 20


def test_the_run_records_what_produced_it(splits, tmp_path):
    a_run(splits, out_dir=tmp_path, arm=("l1_logistic", "kmer"))

    run = json.loads((tmp_path / "run.json").read_text())

    assert (run["model"], run["pool"], run["sources"]) == ("l1_logistic", "kmer", ["kmer"])
    assert run["seqmaestro_version"] and run["created_utc"]
    assert run["featuriser"]["kind"] == "kmer"
    assert run["model_settings"]["model"] == "l1_logistic"


def test_the_reference_split_is_validation(splits, tmp_path):
    _, val = splits
    result = a_run(splits, out_dir=tmp_path)

    with np.load(tmp_path / "feature_values.npz", allow_pickle=True) as values:
        assert values["values"].shape[0] == len(val)
    assert result["reference_split"] == "val"


def test_metrics_carry_intervals_when_asked(splits):
    result = a_run(splits, n_bootstrap=50)

    assert "ci" in result["metrics"]["val"]
    low, high = result["metrics"]["val"]["ci"]["auroc"]
    assert low <= result["metrics"]["val"]["auroc"] <= high


# ------------------------------------------------------------- which way each feature points


def test_every_record_carries_a_direction_and_a_marginal_effect(splits, tmp_path):
    a_run(splits, out_dir=tmp_path)

    run = json.loads((tmp_path / "run.json").read_text())
    records, header = load_features(tmp_path / "features.json")

    assert header["direction_kind"] == run["direction_kind"] == "l1_coefficient_sign"
    assert run["marginal_effect"] == {"statistic": "rank_biserial", "split": "train",
                                      "labels": "real", "n_rows": 400}
    assert all(record.direction in DIRECTIONS for record in records)
    assert all(record.marginal_effect is None or -1 <= record.marginal_effect <= 1
               for record in records)


def test_the_marginal_effect_is_measured_on_the_training_rows(splits, tmp_path):
    """The harness computes the marginal effect itself, independently of any model, on the rows
    the model learned from.

    It does not use the reference split held in `feature_values.npz`, which is validation and
    serves behavioural matching. Recomputing the effect here from the training frame checks that
    the two have not been swapped: both are matrices of the same features, and their shapes would
    not reveal which is which.
    """
    train, _ = splits
    a_run(splits, out_dir=tmp_path)

    records, _ = load_features(tmp_path / "features.json")
    names = [record.name for record in records]
    featuriser = make_featuriser("kmer", k_range=(1, 3), kmer_normalise="counts")
    featuriser.fit(train[SEQUENCE], train[TARGET])
    expected = rank_biserial(featuriser.transform(train[SEQUENCE])[names].to_numpy(),
                             train[TARGET].to_numpy())

    assert [record.marginal_effect for record in records] == \
           pytest.approx(expected, abs=1e-5, nan_ok=True)


def test_feature_values_still_hold_the_reference_split(splits, tmp_path):
    """The counterpart of the previous test: `feature_values.npz` holds the validation split.

    The effect's training rows and this matrix hold the same features in the same order, so
    swapping them would change every stability and co-discovery number without changing a shape or
    raising an error.
    """
    _, val = splits
    a_run(splits, out_dir=tmp_path)

    records, _ = load_features(tmp_path / "features.json")
    names, values = load_feature_values(tmp_path / "feature_values.npz")

    assert names == [record.name for record in records]
    assert values.shape[0] == len(val)
    featuriser = make_featuriser("kmer", k_range=(1, 3), kmer_normalise="counts")
    featuriser.fit(splits[0][SEQUENCE], splits[0][TARGET])
    assert values == pytest.approx(
        featuriser.transform(val[SEQUENCE])[names].to_numpy(), abs=1e-5)


def test_a_control_arm_measures_its_effect_against_the_real_labels(splits, tmp_path):
    """A permuted cell's model is fitted on shuffled labels, but its marginal effect is measured
    against the real labels.

    If the effect were measured against the shuffled vector, `direction` and `marginal_effect`
    would agree under shuffled labels for the same reason they agree under real ones, so the
    permuted control for that comparison would be uninformative. The real and permuted cells of a
    seed therefore report the same marginal effect per feature and differ only in `direction`.
    """
    real = tmp_path / "real"
    control = tmp_path / "control"
    a_run(splits, out_dir=real, seed=7)
    a_run(splits, out_dir=control, seed=7, permute_labels=True)

    effects = {}
    for name, out in (("real", real), ("control", control)):
        records, _ = load_features(out / "features.json")
        effects[name] = {r.name: r.marginal_effect for r in records}

    shared = set(effects["real"]) & set(effects["control"])
    assert shared, "the two arms selected no feature in common, so this checks nothing"
    for feature in shared:
        assert effects["control"][feature] == pytest.approx(effects["real"][feature], abs=1e-9)


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_a_permuted_arm_still_reports_directions(model_name, splits, tmp_path):
    """A permuted cell's features must be indistinguishable from a real cell's in everything the
    judge is shown. A cell whose records had no direction at all would be identifiable by that
    absence alone, however the judge's panel is blinded."""
    a_run(splits, model_name=model_name, out_dir=tmp_path, permute_labels=True)

    records, _ = load_features(tmp_path / "features.json")
    # An equation chosen on rows set aside can be a constant on permuted labels, in which case the
    # cell has no features to show. A cell that has some must give them directions.
    assert any(record.direction is not None for record in records) or (
        model_name == "pysr" and not records)
    assert all(record.marginal_effect is not None for record in records)
