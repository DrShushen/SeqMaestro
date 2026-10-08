"""A continuous target: the squared-error criterion, the tree's leaves and predictions, the loader,
the prompts, and a run on each shape of input.

The two shapes are a table of fixed width alone, and sequences of different lengths beside such a
table, with a generated feature reading each sequence whole. The language model is scripted, as
`test_run_experiment` scripts it, so nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

from deft.data.generic_csv_plugin import GenericCsvDatasetPlugin
from deft.features.feature_finder import CartFeatureFinder
from deft.targets import BINARY, CONTINUOUS, check_target
from deft.trees.split_criterion import GiniCriterion, SquaredErrorCriterion
from deft.trees.tree import AdaptiveDecisionTree
from tests.test_run_experiment import build_cfg, run

REGRESSION = ["dataset.target_kind=continuous", "splitting_criterion=squared_error"]


def prepared(y, weights=None, offset=1.0) -> SquaredErrorCriterion:
    criterion = SquaredErrorCriterion(offset=offset)
    criterion.prepare(y, weights)
    return criterion


def a_target(n=200, seed=0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(3.0, 2.0, n))


# ------------------------------------------------------------------------------- the criterion


def test_the_root_holds_one_unit_of_impurity():
    y = a_target()
    criterion = prepared(y, offset=1.0)
    assert criterion(y, y.iloc[:0]) == pytest.approx(0.0, abs=1e-12)


def test_a_split_scores_the_offset_less_the_share_of_variance_left_within_its_sides():
    y = a_target()
    left, right = y.iloc[:80], y.iloc[80:]
    within = (len(left) * left.var(ddof=0) + len(right) * right.var(ddof=0)) / len(y)
    assert prepared(y)(left, right) == pytest.approx(1.0 - within / y.var(ddof=0), abs=1e-12)


def test_an_empty_side_leaves_the_offset_less_the_nodes_own_impurity():
    """The identity the adapter's node arithmetic relies on, at a node below the root."""
    y = a_target()
    node = y.iloc[:50]
    expected = 1.0 - node.var(ddof=0) / y.var(ddof=0)
    assert prepared(y)(node, node.iloc[:0]) == pytest.approx(expected, abs=1e-12)


def test_scores_do_not_depend_on_the_targets_scale():
    y = a_target()
    scaled = y * 1000.0 + 7.0
    assert prepared(scaled)(scaled.iloc[:70], scaled.iloc[70:]) == pytest.approx(
        prepared(y)(y.iloc[:70], y.iloc[70:]), abs=1e-12)


def test_whole_row_weights_count_as_repeated_rows():
    y = a_target(n=30)
    weights = pd.Series(np.random.default_rng(1).integers(1, 4, len(y)).astype(float), index=y.index)
    repeated = pd.Series(np.repeat(y.to_numpy(), weights.astype(int)))
    cut = int(weights.iloc[:12].sum())
    assert prepared(y, weights)(y.iloc[:12], y.iloc[12:], weights) == pytest.approx(
        prepared(repeated)(repeated.iloc[:cut], repeated.iloc[cut:]), abs=1e-12)


def test_an_unprepared_criterion_says_so():
    y = a_target()
    with pytest.raises(RuntimeError, match="prepare"):
        SquaredErrorCriterion()(y, y)


def test_a_class_weighting_has_no_classes_to_weight():
    y = a_target()
    with pytest.raises(TypeError, match="row by row"):
        prepared(y, {0: 1.0, 1: 2.0})


# ------------------------------------------------------------------------------- the tree


def a_tree(kind, criterion, **settings) -> AdaptiveDecisionTree:
    return AdaptiveDecisionTree(feature_finder=CartFeatureFinder(), splitting_criterion=criterion,
                                max_depth=2, min_samples_leaf=5, save_locally=False,
                                target_kind=kind, **settings)


def a_table(n=120, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = pd.DataFrame({"target": 3.0 * X["signal"] + rng.normal(0, 0.3, n)})
    return X, y


def test_a_continuous_tree_splits_on_the_column_that_explains_the_variance():
    X, y = a_table()
    tree = a_tree(CONTINUOUS, SquaredErrorCriterion())
    tree.fit(X, y, sequence_column=None)
    assert tree.root.feature.name == "value_of_signal"


def test_a_continuous_tree_predicts_its_leaves_means_with_nothing_to_threshold():
    X, y = a_table()
    tree = a_tree(CONTINUOUS, SquaredErrorCriterion())
    tree.fit(X, y, sequence_column=None)
    predicted = np.asarray(tree.predict(X), dtype=float)
    assert np.array_equal(predicted, np.asarray(tree.predict(X, return_proba=True), dtype=float))
    for leaf_value in np.unique(predicted):
        rows = predicted == leaf_value
        assert leaf_value == pytest.approx(float(y["target"][rows].mean()), abs=1e-12)
    assert len(np.unique(predicted)) > 2


def test_a_leaf_predicts_the_weighted_mean_of_its_rows():
    X, y = a_table(n=60)
    weights = np.random.default_rng(3).uniform(0.5, 2.0, len(y))
    tree = a_tree(CONTINUOUS, SquaredErrorCriterion(), sample_weight=weights)
    tree.fit(X, y, sequence_column=None)
    assert tree._get_prediction(y) == pytest.approx(np.average(y["target"], weights=weights),
                                                    abs=1e-12)


@pytest.mark.parametrize("kind, criterion", [(CONTINUOUS, GiniCriterion()),
                                             (BINARY, SquaredErrorCriterion())])
def test_a_criterion_written_for_the_other_kind_is_refused(kind, criterion):
    X, y = a_table()
    if kind == BINARY:
        y = (y > 0).astype(int)
    with pytest.raises(ValueError, match="written for"):
        a_tree(kind, criterion).fit(X, y, sequence_column=None)


def test_a_class_weighting_is_refused_for_a_continuous_target():
    X, y = a_table()
    with pytest.raises(ValueError, match=r"no classes|has none"):
        a_tree(CONTINUOUS, SquaredErrorCriterion(), class_weight="balanced").fit(
            X, y, sequence_column=None)


def test_a_binary_tree_refuses_a_target_holding_other_values():
    X, y = a_table()
    with pytest.raises(ValueError, match="0 and 1"):
        a_tree(BINARY, GiniCriterion()).fit(X, y, sequence_column=None)


def test_a_continuous_target_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        check_target([1.0, np.nan, 2.0], CONTINUOUS)


def test_a_tree_saved_before_trees_had_a_kind_is_binary():
    tree = a_tree(BINARY, GiniCriterion())
    del tree.__dict__["target_kind"]
    assert tree.target_kind == BINARY


# ------------------------------------------------------------------------------- the loader


def a_csv(tmp_path, frame: pd.DataFrame) -> str:
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_the_loader_reads_a_continuous_target_as_numbers(tmp_path):
    path = a_csv(tmp_path, pd.DataFrame({"signal": [1.0, 2.0, 3.0], "response": [-0.25, 0.5, 0.125]}))
    bundle = GenericCsvDatasetPlugin(path, target_column="response", sequence_column=None,
                                     target_kind=CONTINUOUS).load_for_deft()
    assert bundle["y_train"]["response"].tolist() == [-0.25, 0.5, 0.125]
    assert bundle["target_kind"] == CONTINUOUS


def test_the_loader_refuses_a_missing_continuous_value(tmp_path):
    path = a_csv(tmp_path, pd.DataFrame({"signal": [1.0, 2.0, 3.0], "response": [0.1, None, 0.2]}))
    with pytest.raises(ValueError, match=r"`response`.*finite"):
        GenericCsvDatasetPlugin(path, target_column="response", sequence_column=None,
                                target_kind=CONTINUOUS).load_for_deft()


def test_the_loader_refuses_a_binary_target_that_is_not_0_and_1(tmp_path):
    """Read as whole numbers, 0.4 and 0.6 used to become 0 and 0 without a word."""
    path = a_csv(tmp_path, pd.DataFrame({"signal": [1.0, 2.0, 3.0], "label": [0.4, 0.6, 1.0]}))
    with pytest.raises(ValueError, match="0 and 1"):
        GenericCsvDatasetPlugin(path, target_column="label", sequence_column=None).load_for_deft()


# ------------------------------------------------------------------------------- a whole run


def test_cart_grows_a_regression_tree_from_a_table_alone(tmp_path):
    """The first shape: a table of fixed width and no sequence."""
    X, y = a_table()
    path = a_csv(tmp_path, X.assign(label=y["target"]))
    cfg = build_cfg(tmp_path, path, engine=None,
                    extra=["feature_finder=cart", "max_depth=2", "dataset.sequence_column=null",
                           *REGRESSION])
    result = run(cfg)
    tree = dill.loads(Path(result["tree_path"]).read_bytes())
    assert result["status"] == "fit_complete"
    assert tree.target_kind == CONTINUOUS
    assert tree.root.feature.name == "value_of_signal"


def sequences_of_different_lengths(n=48, seed=0) -> pd.DataFrame:
    """Sequences of 8 to 15 nucleotides beside a covariate, the target following their GC share."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        length = int(rng.integers(8, 16))
        gc_share = (i % 4) / 3
        sequence = "".join(rng.choice(list("GC")) if rng.random() < gc_share
                           else rng.choice(list("AT")) for _ in range(length))
        rows.append({"raw_sequence": sequence, "depth": float(rng.normal()),
                     "label": 10.0 * gc_share + float(rng.normal(0, 0.1))})
    return pd.DataFrame(rows)


def test_a_generated_feature_reads_sequences_of_every_length_beside_a_table(tmp_path):
    """The second shape: sequences of different lengths beside a covariate. The scripted model's
    feature reads each sequence whole, at its own length, and the tree splits on it."""
    frame = sequences_of_different_lengths()
    assert frame["raw_sequence"].str.len().nunique() > 1
    path = a_csv(tmp_path, frame)
    cfg = build_cfg(tmp_path, path, extra=[
        *REGRESSION, "params_generation=regression",
        "dataset.positional_window={length: 8, anchor: start, pad: false}"])
    result = run(cfg)
    tree = dill.loads(Path(result["tree_path"]).read_bytes())
    assert result["status"] == "fit_complete"
    assert tree.root.feature.name == "gc_fraction"
    predicted = np.asarray(tree.predict(frame.drop(columns=["label"])), dtype=float)
    assert predicted.min() < 2.0 < 8.0 < predicted.max()


def test_prompts_written_for_a_binary_target_are_refused_for_a_continuous_one(tmp_path):
    path = a_csv(tmp_path, sequences_of_different_lengths())
    cfg = build_cfg(tmp_path, path, extra=[
        *REGRESSION, "dataset.positional_window={length: 8, anchor: start, pad: false}"])
    with pytest.raises(ValueError, match="params_generation=regression"):
        run(cfg)


def test_the_regression_prompts_ask_to_separate_higher_values_from_lower(tmp_path):
    path = a_csv(tmp_path, sequences_of_different_lengths())
    prompts = build_cfg(tmp_path, path, extra=["params_generation=regression"]).params_generation
    assert prompts.target_kind == CONTINUOUS
    assert "higher {target_name}" in prompts.context_template_name_desc
    for text in [prompts.context_template_name_desc, prompts.dataset_info,
                 *prompts.list_instructions_reflection]:
        assert "classif" not in text
