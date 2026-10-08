"""Stage 3A: the elimination loop, the arm it is run on, and what it leaves behind.

The loop is checked on stand-in fits whose importances are set by hand, because for the loop the
rule is what matters: the schedule, the unused block, the tie-break, and never overshooting the
target. The arm is checked on real fits of a planted signal: the seed chosen, the model rebuilt,
the columns, the weights, and the saved models predicting what the reduction measured. The vendored
tree's side is checked with stand-ins here, and against DEFT's own interpreter when that is
installed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import make_featuriser
from seqmaestro.harness.fitted import load_fitted
from seqmaestro.harness.heldout import TEST_SPLIT, reveal_record
from seqmaestro.models import L1Logistic
from seqmaestro.models.base import FILLED_ROWS
from seqmaestro.provenance import environment_python
from seqmaestro.reduction import check, eliminate
from seqmaestro.reduction.arm import (
    ROLES,
    SelectedColumns,
    Source,
    best_seed,
    fit_step_for,
    rebuild_model,
)
from tests.planted import ANALYSES, SEQUENCE, TARGET, planted_dataset

REPO = Path(__file__).resolve().parents[1]
ADAPTERS = REPO / "experiments" / "adapters"

# --------------------------------------------------------------------------------------------
# The loop, on stand-in fits
# --------------------------------------------------------------------------------------------


class Scripted:
    """A fit whose importance for each column is looked up by name, so a test states it."""

    def __init__(self, importance: dict[str, float], names: list[str]):
        self.values = np.array([importance.get(name, 0.0) for name in names])

    def importances(self) -> np.ndarray:
        return self.values

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 0.5)


def scripted(importance: dict[str, float]):
    return lambda X, y, names: Scripted(importance, names)


def a_run(importance: dict[str, float], n: int, n_target: int) -> dict:
    names = [f"f{i}" for i in range(n)]
    X = np.zeros((4, n))
    return eliminate.run(scripted(importance), X, np.array([0, 1, 0, 1]), X, names, n_target)


def test_a_quarter_goes_at_a_time_above_the_fine_threshold_and_one_below():
    assert eliminate.schedule(100, 10) == 25
    assert eliminate.schedule(31, 10) == 31 - int(np.ceil(31 * eliminate.KEEP_FRACTION))
    assert eliminate.schedule(eliminate.FINE_BELOW, 10) == 1
    assert eliminate.schedule(11, 10) == 1
    assert eliminate.schedule(10, 10) == 0


def test_the_schedule_never_overshoots_the_target():
    assert eliminate.schedule(40, 38) == 2
    for n in range(11, 400):
        assert n - eliminate.schedule(n, 10) >= 10


def test_features_the_model_does_not_use_go_together():
    """Six of forty features carry importance. The thirty-four unused ones leave in a single step,
    instead of a quarter at a time, and that step stops at the target."""
    importance = {f"f{i}": float(i + 1) for i in range(6)}
    result = a_run(importance, n=40, n_target=10)

    assert result["steps"][1]["n_features"] == 10, "the unused block went at once, stopping at N"
    assert set(result["final_columns"]) >= {f"f{i}" for i in range(6)}


def test_the_target_is_reached_exactly_even_when_the_model_uses_fewer():
    result = a_run({"f0": 1.0, "f1": 2.0}, n=50, n_target=10)
    assert len(result["final_columns"]) == 10
    assert result["final_importance"].tolist().count(0.0) == 8
    assert result["final_used"] == ["f0", "f1"], "the eight others hold their places by the tie-break"


def test_a_model_that_uses_nothing_has_used_nothing():
    result = a_run({}, n=50, n_target=10)
    assert len(result["final_columns"]) == 10
    assert result["final_used"] == []


def test_a_pool_already_within_the_target_is_fitted_once_and_kept():
    result = a_run({"f0": 1.0}, n=6, n_target=10)
    assert len(result["steps"]) == 1
    assert result["final_columns"] == [f"f{i}" for i in range(6)]
    assert result["first_fitted"] is result["fitted"]


def test_ties_at_zero_go_to_what_a_feature_last_carried_then_to_record_order():
    """Three features stop being used at the same step and two of them must go. Record order
    alone would keep f0, the earliest. The tie-break instead keeps f2, which carried the most
    importance when it was last used."""
    names = [f"f{i}" for i in range(13)]
    everything = dict.fromkeys(names, 9.0)
    fits = [{**everything, "f0": 1.0, "f1": 2.0, "f2": 3.0, "f12": 0.1},
            {**everything, "f0": 0.0, "f1": 0.0, "f2": 0.0}]
    calls = []

    def fit_step(X, y, active):
        calls.append(active)
        return Scripted(fits[min(len(calls), 2) - 1], active)

    X = np.zeros((4, 13))
    result = eliminate.run(fit_step, X, np.array([0, 1, 0, 1]), X, names, n_target=10)

    assert result["steps"][1]["features"] == names[:12], "the weakest went first, one at a time"
    assert result["final_columns"] == [f"f{i}" for i in range(2, 12)]
    worst = eliminate.order_worst_first(np.arange(4), np.zeros(4), np.zeros(4), np.zeros(4))
    assert worst.tolist() == [3, 2, 1, 0], "with nothing else to go on, the latest record goes first"


def test_the_elimination_is_deterministic():
    importance = {f"f{i}": float((i * 7) % 5) for i in range(60)}
    one, two = a_run(importance, 60, 10), a_run(importance, 60, 10)
    assert one["final_columns"] == two["final_columns"]
    assert one["removed_at"].tolist() == two["removed_at"].tolist()


def test_the_top_n_control_is_the_first_fits_favourites_in_record_order():
    assert eliminate.top_n([0.1, 0.9, 0.0, 0.5, 0.9], 3).tolist() == [1, 3, 4]


# --------------------------------------------------------------------------------------------
# The shared modules stay numpy-only, because the other environment loads them by path
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["eliminate", "check", "selections"])
def test_the_shared_modules_import_numpy_and_the_standard_library_only(name):
    tree = ast.parse(load_cart_adapter().SHARED[name].read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"numpy", "__future__"} | set(sys.stdlib_module_names), imported


def test_the_adapter_names_the_files_it_loads():
    adapter = load_cart_adapter()
    assert set(adapter.SHARED) == {"eliminate", "check", "selections"}
    assert all(path.exists() for path in adapter.SHARED.values())


# --------------------------------------------------------------------------------------------
# The column check
# --------------------------------------------------------------------------------------------


def stored_cell(tmp_path: Path, names: list[str], values: np.ndarray) -> Path:
    np.savez_compressed(tmp_path / "feature_values.npz", names=np.array(names, dtype=object),
                        values=values.astype(np.float32))
    return tmp_path


def test_identical_columns_pass_and_a_differing_one_is_named(tmp_path):
    values = np.array([[0.0, 1.0], [1.0, 3.0], [np.nan, 2.0]])
    cell = stored_cell(tmp_path, ["a", "b"], values)

    assert check.compare(values, ["a", "b"], cell)["n_columns_over_tolerance"] == 0
    changed = values.copy()
    changed[1, 1] = 4.0
    with pytest.raises(AssertionError, match="worst b"):
        check.compare(changed, ["a", "b"], cell)
    with pytest.raises(AssertionError, match="name"):
        check.compare(values, ["a", "c"], cell)


# --------------------------------------------------------------------------------------------
# The arm: seed, model, columns and weights, rebuilt from its cell
# --------------------------------------------------------------------------------------------


def test_the_best_seed_is_the_highest_validation_auroc_and_ties_go_low():
    assert best_seed({42: {"auroc": 0.7}, 43: {"auroc": 0.8}, 44: {"auroc": 0.75}}, "auroc") == 43
    assert best_seed({45: {"auroc": 0.8}, 43: {"auroc": 0.8}}, "auroc") == 43
    assert best_seed({42: {"r2": 0.1}, 43: {"r2": -0.2}}, "r2") == 42


def test_a_model_is_rebuilt_from_its_description_and_refused_when_it_would_differ():
    description = L1Logistic(seed=3).describe()
    assert rebuild_model(description, seed=3, target_kind="binary").describe() == description

    with pytest.raises(ValueError, match="standardise=False"):
        # A rebuild always standardises.
        rebuild_model({**description, "standardise": False}, seed=3, target_kind="binary")
    with pytest.raises(ValueError, match=r"C=0\.5"):
        rebuild_model({**description, "C": 0.5}, seed=3, target_kind="binary")
    with pytest.raises(ValueError, match=r"C=absent \(rebuilt 0\.1\)"):
        # A setting the model has and the cell did not record is refused too.
        rebuild_model({key: value for key, value in description.items() if key != "C"}, seed=3,
                      target_kind="binary")


@pytest.mark.parametrize("model_name", ["l1_logistic", "xgboost"])
def test_the_elimination_keeps_the_features_that_carry_the_signal(model_name):
    """Three of forty columns drive the label. Eliminating to three with a real model keeps those
    three."""
    from seqmaestro.models import make_model

    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 40))
    y = ((X[:, 3] + X[:, 17] - X[:, 29] + rng.normal(scale=0.5, size=400)) > 0).astype(int)
    names = [f"c{i}" for i in range(40)]
    description = make_model(model_name, target_kind="binary", seed=0).describe()

    result = eliminate.run(fit_step_for(description, 0, None, target_kind="binary"),
                           X[:300], y[:300], X[300:], names, 3)
    assert sorted(result["final_columns"]) == ["c17", "c29", "c3"]


def test_every_fit_is_handed_the_fitting_weights():
    X = np.random.default_rng(0).random((40, 3))
    y = np.tile([0, 1], 20)
    weighted = fit_step_for(L1Logistic().describe(), 0, np.linspace(0.5, 2.0, 40),
                            target_kind="binary")(X, y, list("abc"))
    plain = fit_step_for(L1Logistic().describe(), 0, None, target_kind="binary")(X, y, list("abc"))
    assert weighted.model.describe()["per_row_weights"] is True
    assert plain.model.describe()["per_row_weights"] is False


def test_selected_columns_are_the_featurisers_own():
    sequences = pd.Series(["ACGTAC", "CCGTAA", "TTGACA"])
    featuriser = make_featuriser("kmer", k_range=(1, 2), kmer_normalise="counts").fit(sequences)
    selected = SelectedColumns(featuriser, ["kmer_A", "kmer_GT"])

    frame = selected.transform(sequences)
    assert list(frame.columns) == ["kmer_A", "kmer_GT"]
    assert np.array_equal(frame.to_numpy(), featuriser.transform(sequences)[["kmer_A", "kmer_GT"]])
    with pytest.raises(ValueError, match="no column"):
        SelectedColumns(featuriser, ["kmer_ZZ"])


def test_generated_columns_keep_the_precision_the_stage_2_model_saw(tmp_path):
    """The reduction fits on the values the arm's model was fitted on, not on a rounding of them."""
    from seqmaestro.llm import ScriptedClient, TaskContext
    from seqmaestro.proposer import GeneratedFeaturiser, ProposalSettings, Proposer

    code = ("def gc_share(X):\n    s = X['window'].astype(str)\n"
            "    return (s.str.count('G') + s.str.count('C')) / (s.str.len() + 7)")
    proposer = Proposer(ScriptedClient(json.dumps([{"name": "gc_share", "description": "d",
                                                    "rationale": "r", "code": code}])),
                        TaskContext(task="Predicting a thing."),
                        settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=2),
                        sequence_column="window", target_column="outcome", target_kind="binary")
    sequences = pd.Series(["ACGTACGTAC", "GGGCCCATAT", "TTTTAAAAGC", "CGCGCGATAT"])
    featuriser = GeneratedFeaturiser(proposer).fit(sequences, np.array([0, 1, 0, 1]))

    selected = SelectedColumns(featuriser, ["gc_share"]).transform(sequences)
    assert np.array_equal(selected["gc_share"].to_numpy(),
                          featuriser.transform(sequences)["gc_share"].to_numpy(dtype=float))


# --------------------------------------------------------------------------------------------
# A reduced arm, end to end: fitted, written, revealed
# --------------------------------------------------------------------------------------------


def a_reduced_run(tmp_path, runners, n_features: int = 3, model: str = "l1_logistic"):
    dataset = planted_dataset(tmp_path)
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
        "analyses": ANALYSES,
        "arms": {"positional": {"models": [model]}},
        "reduction": {"n_features": n_features},
    }))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(runners.config.load(source).to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main(flags) == 0
    for call in ("baseline", "arms__real-permuted__seed_1-2"):
        assert (run_dir / "invocations" / f"{call}.json").exists(), call
    return dataset, run_dir


@pytest.mark.parametrize("model", ["l1_logistic", "pysr"])
def test_a_reduced_arm_leaves_three_standard_cells_that_the_reveal_scores(runners, tmp_path, model):
    dataset, run_dir = a_reduced_run(tmp_path, runners, model=model)
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0

    arm_dir = run_dir / "reduction" / model / "positional"
    record = json.loads((arm_dir / "reduction.json").read_text())
    seeds = {seed: json.loads((run_dir / "results" / model / "positional" / "real" /
                               f"seed_{seed}" / "metrics.json").read_text())["val"]["auroc"]
             for seed in (1, 2)}
    assert record["seed"] == max(seeds, key=seeds.get)
    assert record["n_final"] == 3 and len(record["top_n_columns"]) == 3
    assert record["featurisation_check"]["n_columns_over_tolerance"] == 0

    val = pd.read_csv(dataset.val_path)
    for role in ROLES:
        cell = arm_dir / role
        assert {"run.json", "metrics.json", "predictions.npz", "fitted.pkl"} <= \
               {path.name for path in cell.iterdir()}
        run = json.loads((cell / "run.json").read_text())
        restored = load_fitted(cell)
        assert restored.featuriser.feature_names == run["columns"]
        with np.load(cell / "predictions.npz") as predictions:
            assert np.array_equal(restored.predict(val[SEQUENCE]), predictions["val_score"]), \
                "the saved model is the one the reduction scored"

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    record = reveal_record(run_dir)
    assert record["scored"] == {"baseline": 2, "results": 4, "reduction": 3}
    for role in ROLES:
        metrics = json.loads((arm_dir / role / "metrics.json").read_text())
        assert TEST_SPLIT in metrics
        # An equation can be undefined on a row, so its cells count the rows it filled.
        assert (FILLED_ROWS in metrics) == (model == "pysr")
        if model == "pysr":
            assert set(metrics[FILLED_ROWS]) == {"val", TEST_SPLIT}


def test_a_run_that_fits_pysr_records_its_environment_beside_cores(runners, tmp_path, monkeypatch):
    written = set()
    monkeypatch.setattr(runners.records, "write_snapshot",
                        lambda environment, out, **_: written.add((environment, out.name)))
    _, run_dir = a_reduced_run(tmp_path, runners, model="pysr")
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    assert written == {("SeqMaestro_Core", "core"), ("SeqMaestro_Method_pysr", "pysr")}


def test_a_reduced_run_is_not_ready_to_reveal_until_the_reduction_has_run(runners, tmp_path):
    _, run_dir = a_reduced_run(tmp_path, runners)
    assert runners.reveal.main(["--run-dir", str(run_dir), "--check"]) == 2
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reveal.main(["--run-dir", str(run_dir), "--check"]) == 0


def test_a_cell_fitted_against_another_kind_of_target_is_not_ready(runners, tmp_path, capsys):
    """Scoring refuses such a cell, which it would do after the seal was written, so the check
    before the seal asks first."""
    _, run_dir = a_reduced_run(tmp_path, runners)
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    record = sorted((run_dir / "results").glob("*/*/real/seed_*"))[0] / "run.json"
    record.write_text(json.dumps({**json.loads(record.read_text()), "target_kind": "continuous"}))

    assert runners.reveal.main(["--run-dir", str(run_dir), "--check"]) == 2
    said = capsys.readouterr()
    assert "fitted against a continuous target, and the dataset's is binary" in said.out + said.err


def test_the_reduction_resumes_and_refuses_a_revealed_run(runners, tmp_path, capsys):
    _, run_dir = a_reduced_run(tmp_path, runners)
    runners.reduction.main(["--run-dir", str(run_dir)])
    written = (run_dir / "reduction" / "l1_logistic" / "positional" / "reduction.json").read_text()

    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    assert "already reduced" in capsys.readouterr().out
    assert (run_dir / "reduction" / "l1_logistic" / "positional" / "reduction.json").read_text() \
        == written

    runners.reveal.main(["--run-dir", str(run_dir)])
    from seqmaestro.harness.heldout import SealError

    with pytest.raises(SealError):
        runners.reduction.main(["--run-dir", str(run_dir), "--force"])


def test_the_fitting_rows_are_the_cells_own(tmp_path):
    dataset = planted_dataset(tmp_path)
    train = pd.read_csv(dataset.train_path)
    source = Source(tmp_path, {"labels": "real", "seed": 5, "train_fraction": 0.5,
                               "weight_column": None}, [])
    rows, y, weights = source.fitting_rows(train, dataset.columns)
    assert len(rows) == len(train) // 2 and weights is None
    assert np.array_equal(rows[TARGET].to_numpy(), y), "real labels"

    permuted = Source(tmp_path, {**source.run, "labels": "permuted"}, [])
    with pytest.raises(ValueError, match="only real-label cells"):
        permuted.fitting_rows(train, dataset.columns)


# --------------------------------------------------------------------------------------------
# The vendored tree's side, with stand-ins
# --------------------------------------------------------------------------------------------


def load_cart_adapter():
    spec = importlib.util.spec_from_file_location("deft_cart_under_test", ADAPTERS / "deft_cart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deft_python() -> Path:
    return environment_python("SeqMaestro_Method_deft")


#: Run under DEFT's interpreter: compose a configuration as a fit would, record it as hydra records
#: a fit's, then build the tree a later stage builds from that record, and print what it holds.
TREE_FROM_A_RECORD = """
import importlib.util, json, sys
from pathlib import Path
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location("deft_cart", sys.argv[1])
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
bundle = adapter.stage2_adapter()
kind, cell, overrides = sys.argv[2], Path(sys.argv[3]), sys.argv[4:]

composed = adapter.composed_configuration(overrides, kind, bundle)
record = cell / "bundle" / bundle.WORK_DIR / adapter.HYDRA_RECORD
record.parent.mkdir(parents=True)
OmegaConf.save(composed, record)

configuration = adapter.recorded_configuration(cell, bundle)
tree = adapter.cart_tree(configuration, 1000, None, kind)
print(json.dumps({
    "max_depth": tree.max_depth, "min_samples_split": tree.min_samples_split,
    "min_samples_leaf": tree.min_samples_leaf, "max_leaf_nodes": tree.max_leaf_nodes,
    "class_weight": tree.class_weight, "criterion": type(tree.splitting_criterion).__name__,
    "finder": type(tree.feature_finder).__name__, "settings": adapter.tree_settings(configuration)}))
"""


def tree_from_a_record(tmp_path, kind: str, *overrides: str) -> dict:
    completed = subprocess.run(
        [str(deft_python()), "-c", TREE_FROM_A_RECORD, str(ADAPTERS / "deft_cart.py"), kind,
         str(tmp_path / f"cell_{len(list(tmp_path.iterdir()))}"), *overrides],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.crossenv
@pytest.mark.skipif(not deft_python().exists(), reason="DEFT's interpreter is not installed here")
def test_a_later_tree_has_every_setting_the_cells_fit_recorded(tmp_path):
    """The reduction and the panel grow their trees from the configuration hydra recorded for the
    cell's fit, so a setting a run changed is in force whichever setting it is, and the settings
    that configure the language model change nothing about the tree."""
    conf = yaml.safe_load((REPO / "methods" / "deft" / "src" / "deft" / "conf" / "config.yaml")
                          .read_text())
    plain = tree_from_a_record(tmp_path, "binary")
    assert plain["max_depth"] == conf["max_depth"]
    assert plain["min_samples_leaf"] == int(conf["min_samples_leaf_fraction"] * 1000)
    assert plain["class_weight"] is None
    assert plain["criterion"] == "GiniCriterion" and plain["finder"] == "CartFeatureFinder"

    changed = tree_from_a_record(
        tmp_path, "binary", "tree.class_weight=balanced", "max_depth=3", "tree.min_samples_split=7",
        "tree.max_leaf_nodes=11", "min_samples_leaf_fraction=0.05", "splitting_criterion.offset=2.0",
        "params_generation.n_reflections=5", "random_state=45")
    assert (changed["max_depth"], changed["min_samples_split"], changed["max_leaf_nodes"]) == (3, 7, 11)
    assert changed["min_samples_leaf"] == 50 and changed["class_weight"] == "balanced"
    assert changed["settings"]["criterion"]["offset"] == 2.0
    assert changed["settings"]["min_samples_leaf_fraction"] == 0.05


@pytest.mark.crossenv
@pytest.mark.skipif(not deft_python().exists(), reason="DEFT's interpreter is not installed here")
def test_a_later_tree_is_grown_by_the_criterion_the_cell_was(tmp_path):
    assert tree_from_a_record(tmp_path, "continuous")["criterion"] == "SquaredErrorCriterion"
    entropy = tree_from_a_record(tmp_path, "binary", "splitting_criterion=information_gain")
    assert entropy["criterion"] == "InformationGainCriterion"


def test_a_trees_importance_is_its_share_of_the_impurity_reduction_by_column():
    adapter = load_cart_adapter()
    leaf = SimpleNamespace(feature=None, left=None, right=None)
    root = SimpleNamespace(feature=SimpleNamespace(name="value_of_a"),
                           left=SimpleNamespace(feature=SimpleNamespace(name="value_of_b"),
                                                left=leaf, right=leaf),
                           right=leaf)
    tree = SimpleNamespace(root=root)

    def statistics(tree, X, y):
        return {"": {"n_samples": 100, "impurity_decrease": 0.3},
                "L": {"n_samples": 50, "impurity_decrease": 0.2}}

    shares = adapter.impurity_importances(tree, None, None, statistics)
    assert shares == pytest.approx({"value_of_a": 0.75, "value_of_b": 0.25})
    fitted = adapter.FittedTree(tree, ["a", "b", "c"], {"a": 0.75, "b": 0.25})
    assert fitted.importances().tolist() == [0.75, 0.25, 0.0]


def test_both_sides_agree_where_the_reduced_tree_is_kept():
    from seqmaestro.pipeline.vendored import deft_cart as deft_cart_calls
    assert deft_cart_calls.MODEL_FILE == load_cart_adapter().MODEL_FILE


# --------------------------------------------------------------------------------------------
# The vendored tree's side, for real, when its interpreter is installed
# --------------------------------------------------------------------------------------------


def deft_python() -> Path:
    from seqmaestro.pipeline.vendored import deft as run_deft
    return run_deft.DEFT_PYTHON


@pytest.mark.skipif(not deft_python().exists(), reason="DEFT's interpreter is not installed here")
def test_a_deft_arm_is_reduced_and_predicted_on_defts_side(tmp_path, monkeypatch):
    """Crosses the environment boundary with DEFT's real interpreter. A CART cell is fitted by the
    Stage 2 adapter and reduced by `deft_cart.py`, and the reduced tree must predict the validation
    rows exactly as the reduction scored them. The tree is then read back for a report, with every
    fitting row routed to a leaf."""
    import subprocess

    from seqmaestro.harness.runner import ingest_bundle, subsample_training_rows
    from seqmaestro.pipeline.vendored import deft as run_deft
    from seqmaestro.pipeline.vendored import deft_cart as deft_cart_calls
    from seqmaestro.reduction import outputs

    dataset = planted_dataset(tmp_path)
    train, val = pd.read_csv(dataset.train_path), pd.read_csv(dataset.val_path)
    run_dir = tmp_path / "run"
    cell = run_dir / "results" / "deft_or_cart" / "positional" / "real" / "seed_1"
    rows, y_fit = subsample_training_rows(train, seed=1, columns=dataset.columns,
                                          train_fraction=0.8, permute_labels=False,
                                          permutation_key=None)
    paths = run_deft.write_inputs("positional", rows, y_fit, train, val, cell / "inputs",
                                  columns=dataset.columns)
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    (descriptions / "planted.json").write_text(json.dumps({"task": "Predicting a planted motif."}))
    completed = subprocess.run(
        [str(run_deft.DEFT_PYTHON), str(run_deft.ADAPTER), "fit", "--train", str(paths["train"]),
         "--val", str(paths["val"]), "--effect", str(paths["effect"]),
         "--out-bundle", str(cell / "bundle"), "--feature-finder", "cart",
         "--target-column", TARGET, "--target-kind", "binary", "--sequence-column", SEQUENCE,
         "--task", "planted",
         "--dataset-dir", str(descriptions), "--permutation-repeats", "0",
         "--permutation-seed", "0", "--config-override", "tree.class_weight=balanced"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    ingest_bundle(cell / "bundle", val, target_column=TARGET, target_kind="binary", out_dir=cell, seed=1, n_bootstrap=0,
                  labels="real", arm=("deft_or_cart", "positional"),
                  permutation_scope="seed", train_fraction=0.8, n_fit_rows=len(rows),
                  weight_column=None, predict_with=run_deft.PREDICT_WITH)

    source = Source.read(cell)
    assert len(source.names) >= 2, "the tree has to use more features than it is reduced to"
    arm_dir = run_dir / "reduction" / "deft_or_cart" / "positional"
    reduction = deft_cart_calls.reduce_arm(
        cell, rows, y_fit, dataset.inputs(val), columns=dataset.columns,
        weight_column=None, n_target=1, out_dir=arm_dir)
    assert len(reduction.steps) > 1 and len(reduction.final_columns) == 1
    assert reduction.featurisation_check["n_columns_over_tolerance"] == 0
    assert reduction.settings["tree"]["class_weight"] == "balanced"

    outputs.write(arm_dir, reduction, source=source, run_dir=run_dir, n_target=1,
                  y_val=val[TARGET].to_numpy(), target_kind="binary",
                  save_model=lambda role, _: {"predict_with": deft_cart_calls.PREDICT_WITH})
    scores = deft_cart_calls.predict_cell(run_dir, arm_dir / "reduced", dataset.inputs(val), SEQUENCE)
    with np.load(arm_dir / "reduced" / "predictions.npz") as predictions:
        assert np.array_equal(scores, predictions["val_score"])

    # The report reads the same saved tree, and checks it reproduces the cell's validation scores.
    structure = deft_cart_calls.export_tree(run_dir, arm_dir.relative_to(run_dir) / "reduced", train,
                                            val, columns=dataset.columns)
    nodes = structure["nodes"]
    assert nodes[0]["n_samples"] == structure["n_fit_rows"] == len(rows)
    assert sum(node["n_samples"] for node in nodes if node["is_leaf"]) == len(rows)
    assert {node["feature"] for node in nodes if not node["is_leaf"]} <= set(reduction.final_columns)
    assert structure["routing_max_difference"] == 0.0
