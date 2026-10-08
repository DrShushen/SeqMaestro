"""The IGSR arm from this side: its adapter's reading of a result, tested without IGSR, the driver's
wiring of the arm, and a saved model predicting across the environment boundary.

The adapter runs in IGSR's environment and imports `igsr` only inside its three modes, so what it
builds from a result (the records, the directions, the permutation importance, what IGSR is told
about the data) can be exercised here with stand-in results.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features.base import Inputs
from seqmaestro.harness.direction import DECREASES_TARGET, INCREASES_TARGET
from seqmaestro.provenance import environment_python
from tests.planted import ANALYSES, planted_dataset

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "experiments" / "adapters" / "igsr_bundle.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("igsr_bundle_under_test", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_adapter()


def test_the_kinds_an_igsr_cell_records_are_listed_ones():
    from seqmaestro.harness.direction import DIRECTION_KINDS
    from seqmaestro.harness.features import IMPORTANCE_KINDS

    assert adapter.IMPORTANCE_KIND in IMPORTANCE_KINDS
    assert adapter.DIRECTION_KIND in DIRECTION_KINDS


def a_term(name, code, **fields):
    """One term as IGSR's saved model records it (`FunctionTerm.record`)."""
    return {"name": name, "description": f"{name} measured", "rationale": "because", "code": code,
            "fill": 0.5, "undefined_share": 0.0, "first_seen": "node_0_1", "repaired": False,
            "renamed_from": None, **fields}


def a_model(terms, weights, intercept=0.1):
    return {"term_language": "function", "terms": [term["name"] for term in terms],
            "target_columns": ["y"], "weights": [[weight] for weight in weights],
            "intercept": [intercept], "functions": terms, "code_guard": None, "timeout_seconds": 5}


# --------------------------------------------------------------------------------------------
# The adapter's contract with this side
# --------------------------------------------------------------------------------------------


def test_the_adapter_restates_this_sides_contract_and_vocabulary():
    """The adapter restates these to avoid importing across the environment boundary, so the copies
    are checked here."""
    from seqmaestro.features.base import PROVENANCE_SELECTED
    from seqmaestro.harness.bundle import BUNDLE_VERSION
    from seqmaestro.harness.features import SCHEMA_VERSION
    from seqmaestro.proposer.proposer import KIND_LLM_GENERATED
    from seqmaestro.target import CONTINUOUS

    assert (adapter.BUNDLE_VERSION, adapter.FEATURE_SCHEMA_VERSION) == (BUNDLE_VERSION, SCHEMA_VERSION)
    assert (adapter.INCREASES_TARGET, adapter.DECREASES_TARGET) == (INCREASES_TARGET, DECREASES_TARGET)
    assert adapter.KIND_GENERATED == KIND_LLM_GENERATED
    assert adapter.PROVENANCE_SELECTED == PROVENANCE_SELECTED and adapter.CONTINUOUS == CONTINUOUS


def test_the_adapter_imports_igsr_only_inside_its_modes():
    """Loaded above in this environment, where `igsr` is not installed."""
    assert importlib.util.find_spec("igsr") is None
    assert callable(adapter.records_for)


# --------------------------------------------------------------------------------------------
# What a term becomes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("code", "column"), [
    ('def depth(X):\n    return X["depth"]', "depth"),
    ('def depth(X):\n    """The depth."""\n    return X["depth"].astype(float)', "depth"),
    ('def depth(X):\n    return X["depth"] * 2', None),
    ('def depth(X):\n    return X["seq"].str.len()', None),
    ('def depth(X):\n    x = X["depth"]\n    return x', None),
    ('import numpy as np\ndef depth(X):\n    return X["depth"]', None),
])
def test_a_bare_read_of_one_column_is_recognised_on_the_parsed_source(code, column):
    assert adapter.bare_column(code) == column


def test_a_term_becomes_a_record_with_its_influence_weight_and_origin():
    terms = [a_term("gc_count", 'def gc_count(X):\n    return X["seq"].str.count("GC")', repaired=True,
                    renamed_from="gc"),
             a_term("depth_itself", 'def depth_itself(X):\n    return X["depth"]')]
    handed = {"depth": {"kind": "covariate", "params": {"column": "depth"}, "description": "read depth"}}
    records = adapter.records_for(a_model(terms, [0.8, -0.2]), {"gc_count": 0.3, "depth_itself": 0.01},
                                  {"gc_count": 0.2}, handed)

    written, handed_back = records
    assert written["kind"] == "llm_generated" and written["provenance"] == "generated"
    assert written["source"].startswith("def gc_count") and written["origin"] == "igsr"
    assert (written["importance"], written["permutation_importance"]) == (0.3, 0.2)
    assert written["direction"] == INCREASES_TARGET
    assert written["extra"]["repaired"] and written["extra"]["renamed_from"] == "gc"
    assert written["extra"]["weight"] == 0.8 and written["extra"]["first_seen"] == "node_0_1"
    # A term that reads a handed column unchanged is that column, as the dataset describes it.
    assert (handed_back["kind"], handed_back["params"]) == ("covariate", {"column": "depth"})
    assert handed_back["source"] is None and handed_back["provenance"] == "selected"
    assert handed_back["description"] == "read depth" and handed_back["direction"] == DECREASES_TARGET
    assert handed_back["permutation_importance"] is None and "repaired" not in handed_back["extra"]


def test_a_zero_weight_moves_the_model_nowhere():
    assert adapter.direction_of(0.0) is None


def test_the_permutation_importance_is_the_fall_in_r_squared_when_a_term_is_shuffled():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(400, 2))
    y = 2.0 * values[:, 0] + rng.normal(0, 0.1, 400)
    found = adapter.permutation_importances(values, np.array([[2.0], [0.0]]), 0.0, y, ["strong", "unused"],
                                            repeats=3, rng=np.random.default_rng(1))
    assert found["strong"] > 1.0 and found["unused"] == pytest.approx(0.0)
    again = adapter.permutation_importances(values, np.array([2.0, 0.0]), 0.0, y, ["strong", "unused"],
                                            repeats=3, rng=np.random.default_rng(1))
    assert again == found
    assert adapter.permutation_importances(values, np.ones(2), 0.0, y, ["a", "b"], repeats=0,
                                           rng=np.random.default_rng(1)) == {}


def test_igsr_is_told_the_task_the_target_the_layout_and_each_covariate():
    description = {"task": "Predict the thing.", "target_name": "the measured thing",
                   "sequence_layout": "40 bases, 5' to 3'."}
    manifest = {"sequence": {"column": "seq", "window": None},
                "columns": {"depth": {"kind": "covariate", "params": {"column": "depth"},
                                      "description": "read depth"}}}
    assert adapter.data_dictionary(description, "y") == (
        "Predict the thing.\n\nThe target column `y` is the measured thing.")
    assert adapter.column_descriptions(manifest, description) == {
        "depth": "read depth", "seq": "40 bases, 5' to 3'."}


def test_the_search_is_given_the_labels_to_choose_on_and_val_keeps_the_real_ones(tmp_path,
                                                                                   monkeypatch):
    """What the handover writes is what the adapter reads: the search's validation rows carry the
    labels to choose on, and `val.csv` keeps the real ones, which the cell is scored against."""
    from seqmaestro.dataset import Columns
    from seqmaestro.pipeline.vendored import handover
    from seqmaestro.target import CONTINUOUS

    rows = pd.DataFrame({"seq": ["ACGT", "AGGT", "ACCT", "TTGA"], "y": [0.5, 1.5, 2.5, 3.5]})
    paths = handover.write_inputs(rows, rows["y"].to_numpy(), rows, rows, tmp_path,
                                  columns=Columns("seq", "y", None, (), CONTINUOUS),
                                  choose_on=np.array([2.5, 0.5, 3.5, 1.5]))
    val = pd.read_csv(paths["val"])
    searched = adapter.search_validation(val, pd.read_csv(paths["choose_on"]), ["seq"], "y")
    assert searched.to_dict("list") == {"seq": ["ACGT", "AGGT", "ACCT", "TTGA"],
                                        "y": [2.5, 0.5, 3.5, 1.5]}
    assert val["y"].tolist() == [0.5, 1.5, 2.5, 3.5]


def test_labels_to_choose_on_that_do_not_fit_the_validation_rows_are_refused():
    """They are matched to the rows by position, so another column, a second one, or another number
    of rows means they belong to something else."""
    val = pd.DataFrame({"seq": ["AC", "GG", "TT"], "y": [0.1, 0.2, 0.3]})
    for wrong in (pd.DataFrame({"y": [0.3, 0.1]}), pd.DataFrame({"z": [0.3, 0.1, 0.2]}),
                  pd.DataFrame({"y": [0.3, 0.1, 0.2], "seq": ["AC", "GG", "TT"]})):
        with pytest.raises(SystemExit, match="choose-on"):
            adapter.search_validation(val, wrong, ["seq"], "y")


# --------------------------------------------------------------------------------------------
# The arm in the driver
# --------------------------------------------------------------------------------------------


@pytest.fixture
def driver():
    from seqmaestro.pipeline import config as experiment_config
    from seqmaestro.pipeline import driver as run_experiment
    return experiment_config, run_experiment


def a_config(tmp_path, experiment_config, continuous=True, overrides=None):
    dataset = planted_dataset(tmp_path, covariates=True, variable=True, continuous=continuous)
    grid = {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
            "n_bootstrap": 10,
            "permutation_scope": "seed", "permutation_repeats": 0, "judge": False, "judge_direction": False,
            "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                   "max_example_length": 300, "max_seconds": 60},
            **({} if continuous else {"class_weight": "balanced"})}
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": grid,
        "models": {"igsr": {"config_overrides": overrides or ["experiment.n_successors=2"]}},
        "analyses": ANALYSES,
        "arms": {"llm_generated_joint": {"models": ["igsr"]},
                 "llm_generated_joint+covariates": {"models": ["igsr"]}},
        "reduction": {"n_features": 3},
        "reliability": {"pairs": 3, "permutations": 200, "cluster_rho": 0.8, "max_modal_share": 0.99,
                        "planted_count": 3, "planted_signal_sd": 1.5,
                        "check_arm": "igsr/llm_generated_joint", "seed": 0},
    }, sort_keys=False))
    return experiment_config.load(source)


def test_the_arm_runs_one_process_per_labels_and_seed_under_the_grids_policy(driver, tmp_path,
                                                                             monkeypatch):
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    assert config.arm_cells() == [("igsr", "llm_generated_joint"),
                                  ("igsr", "llm_generated_joint+covariates")]
    stages = {stage.name: stage for stage in run_experiment.build_stages(config, tmp_path / "run")}
    stage = stages["igsr/llm_generated_joint"]
    assert len(stage.commands) == 4 and not stage.retry and stage.calls_model
    argv = stage.commands[0].argv
    assert argv[1] == "experiments/runners/run_arm.py"
    assert argv[argv.index("--arms") + 1] == "igsr/llm_generated_joint"
    covariates = stages["igsr/llm_generated_joint+covariates"].commands[0].argv
    assert covariates[covariates.index("--arms") + 1] == "igsr/llm_generated_joint+covariates"

    from seqmaestro.pipeline.vendored import igsr

    from_the_run = ["experiment.function_terms.repair_rounds=1",
                    "experiment.function_terms.max_undefined_share=0.2",
                    "experiment.function_terms.max_example_length=300",
                    "experiment.function_terms.timeout_seconds=60",
                    "experiment.n_successors=2"]
    assert igsr.run_overrides(config) == from_the_run
    # The language model is the one the environment names, passed as a cell starts.
    for name, value in {"SEQMAESTRO_LLM_PROVIDER": "claude", "SEQMAESTRO_CLAUDE_KEY": "a-key",
                        "SEQMAESTRO_CLAUDE_MODEL": "a-model"}.items():
        monkeypatch.setenv(name, value)
    assert igsr.overrides(config) == ["llm=seqmaestro_claude", *from_the_run]
    monkeypatch.delenv("SEQMAESTRO_LLM_PROVIDER")
    with pytest.raises(RuntimeError, match="SEQMAESTRO_LLM_PROVIDER"):
        igsr.overrides(config)


def test_the_arm_is_refused_on_a_binary_target_and_may_not_restate_the_grids_policy(driver, tmp_path):
    experiment_config, run_experiment = driver
    with pytest.raises(experiment_config.ConfigError, match="fits a continuous target"):
        a_config(tmp_path / "binary", experiment_config, continuous=False)
    config = a_config(tmp_path / "restated", experiment_config,
                      overrides=["experiment.function_terms.repair_rounds=3"])
    with pytest.raises(experiment_config.ConfigError, match="which the grid already states"):
        run_experiment.build_stages(config, tmp_path / "run")


def test_preflight_names_igsrs_interpreter_when_it_is_missing(driver, tmp_path, monkeypatch):
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    monkeypatch.setenv("SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_IGSR", str(tmp_path / "no_python"))
    stages = [stage for stage in run_experiment.build_stages(config, tmp_path / "run")
              if stage.name.startswith("igsr/")]
    problems = run_experiment.preflight(config, tuple(stages), tmp_path / "run")
    assert any("SeqMaestro_Method_igsr" in problem for problem in problems)


# --------------------------------------------------------------------------------------------
# Across the boundary
# --------------------------------------------------------------------------------------------


def igsr_python() -> Path:
    return environment_python("SeqMaestro_Method_igsr")


@pytest.mark.crossenv
@pytest.mark.skipif(not igsr_python().exists(), reason="IGSR's interpreter is not installed here")
def test_a_saved_model_predicts_test_rows_on_igsrs_side_in_their_order(tmp_path, monkeypatch):
    """The reveal's path for an IGSR cell: the rows cross numbered and without their target, the
    saved terms run under IGSR's interpreter, and the scores come back in the rows' order."""
    from seqmaestro.pipeline.vendored import igsr as run_igsr

    cell = tmp_path / "cell"
    (cell / "inputs").mkdir(parents=True)
    (cell / "inputs" / "columns.json").write_text(json.dumps({
        "sequence": {"column": "seq", "window": None},
        "columns": {"depth": {"kind": "covariate", "params": {"column": "depth"},
                              "description": "read depth"}}}))
    terms = [a_term("gc_count", 'def gc_count(X):\n    return X["seq"].str.count("GC")'),
             a_term("depth_itself", 'def depth_itself(X):\n    return X["depth"]')]
    (cell / "bundle").mkdir()
    (cell / "bundle" / "model.json").write_text(json.dumps(a_model(terms, [1.5, 0.5], intercept=0.25)))
    (cell / "bundle" / "arm.json").write_text(json.dumps({"bundle_version": 3, "fitted_model": "model.json"}))

    rows = pd.DataFrame({"seq": ["GCGC", "AAAAGCAAAA", "TTT"], "depth": [2.0, 0.0, -4.0],
                         "unrelated": [9.0, 9.0, 9.0]})
    inputs = Inputs(sequences=rows["seq"], covariates=rows[["depth", "unrelated"]])
    scores = run_igsr.predict_cell(cell, inputs, "seq")
    np.testing.assert_allclose(scores, [0.25 + 1.5 * 2 + 0.5 * 2.0, 0.25 + 1.5 + 0.0, 0.25 - 2.0])


#: The adapter under IGSR's interpreter with the language model scripted at the call IGSR's tests
#: script, so the rest runs as it does live. `script.json` beside it names the adapter, the replies
#: in turn, and arguments to add to those the stage starts it with.
SCRIPTED_ADAPTER = """\
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from igsr.agent.agent import LiteLLMAgent

script = json.loads((Path(__file__).parent / "script.json").read_text())
replies = iter(script["replies"])


def complete(self, messages, completion_kwargs):
    message = SimpleNamespace(content=next(replies))
    usage = SimpleNamespace(to_dict=lambda: {"prompt_tokens": 3, "completion_tokens": 2})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


LiteLLMAgent._create_chat_completion = complete
sys.argv = [script["adapter"], *sys.argv[1:], *script["arguments"]]
runpy.run_path(script["adapter"], run_name="__main__")
"""

#: IGSR's configuration for one plain iteration with no language model behind it, as IGSR's tests
#: compose it. The pruning keeps every term, so the model the search scores predicts from its term.
SCRIPTED_SEARCH = ["experiment.tree=false",
                   "experiment.n_iters=1", "experiment.algorithmic_pruning=true",
                   "experiment.early_stop_enabled=false", "experiment.retry_attempts=1"]


@pytest.mark.crossenv
@pytest.mark.skipif(not igsr_python().exists(), reason="IGSR's interpreter is not installed here")
def test_a_control_cells_search_chooses_on_shuffled_labels_and_is_scored_on_the_real_ones(
        tmp_path, monkeypatch):
    """IGSR's search chooses on the validation rows (the iteration it keeps, the terms it prunes, the
    error its prompts show), so a control arm's must choose against shuffled labels there, or the
    control carries the real target. A control cell runs end to end through Stage 2: the validation
    error the search records is the one against the labels it was handed to choose on, and the cell
    is scored against the real ones."""
    from seqmaestro.harness.runner import permutation_rng
    from seqmaestro.pipeline import config as experiment_config
    from seqmaestro.pipeline import records, stage2
    from seqmaestro.pipeline.vendored import igsr as run_igsr

    dataset = planted_dataset(tmp_path, continuous=True)
    scripted = tmp_path / "scripted"
    (scripted / "datasets").mkdir(parents=True)
    (scripted / "datasets" / "planted.json").write_text(json.dumps({"task": "Predict the response."}))
    term = {"name": "gc_steps", "description": "GC steps in the window", "rationale": "GC-rich motif",
            "code": 'def gc_steps(X):\n    return X["window"].str.count("GC")'}
    (scripted / "script.json").write_text(json.dumps({
        "adapter": str(ADAPTER), "replies": ["TERMS:\n```json\n" + json.dumps([term]) + "\n```"],
        "arguments": ["--dataset-dir", str(scripted / "datasets")]}))
    (scripted / "adapter.py").write_text(SCRIPTED_ADAPTER)
    monkeypatch.setattr(run_igsr, "ADAPTER", scripted / "adapter.py")
    # The stage reads the description too, for its blinding check, from where the adapter does.
    monkeypatch.setattr(stage2, "DESCRIPTIONS", scripted / "datasets")
    # The stage passes IGSR the provider the environment names. No call is made to it: the
    # scripted adapter answers in the model's place.
    for name, value in {"SEQMAESTRO_LLM_PROVIDER": "azure", "SEQMAESTRO_AZURE_OPENAI_KEY": "dummy-key",
                        "SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT": "dummy",
                        "SEQMAESTRO_AZURE_OPENAI_ENDPOINT": "https://dummy.invalid"}.items():
        monkeypatch.setenv(name, value)
    snapshots = []
    monkeypatch.setattr(records, "write_snapshot",
                        lambda environment, out, **kwargs: snapshots.append((environment, out.name)) or {})

    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [42], "labels": ["permuted"], "train_fraction": 0.8, "n_bootstrap": 10,
                 "permutation_scope": "seed", "permutation_repeats": 2, "judge": False,
                 "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "models": {"igsr": {"config_overrides": SCRIPTED_SEARCH}},
        "analyses": ANALYSES,
        "arms": {"llm_generated_joint": {"models": ["igsr"]}}}, sort_keys=False))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(experiment_config.load(source).to_yaml())
    status = stage2.main(["--run-dir", str(run_dir)])
    assert status == 0, [path.read_text() for path in (run_dir / "failures").glob("*.json")]
    assert snapshots == [("SeqMaestro_Core", "core"), ("SeqMaestro_Method_igsr", "igsr")], \
        "the run records the environment its method ran in, beside Core's"

    cell = run_dir / "results" / "igsr" / "llm_generated_joint" / "permuted" / "seed_42"
    target = dataset.target_column
    real = pd.read_csv(dataset.val_path)[target].to_numpy()
    chosen_on = pd.read_csv(cell / "inputs" / "choose_on.csv")[target].to_numpy()
    assert np.array_equal(pd.read_csv(cell / "inputs" / "val.csv")[target], real)
    assert np.array_equal(chosen_on, permutation_rng(42).permutation(real)), \
        "shuffled by the draw the fitting rows' labels take"
    with np.load(cell / "predictions.npz") as scored:
        assert np.array_equal(scored["val_true"], real), "scored against the real labels"
        predicted = scored["val_score"]
    searched = json.loads((cell / "bundle" / "arm.json").read_text())["search"]["best_validation_mse"]
    assert searched == pytest.approx(np.mean((chosen_on - predicted) ** 2), rel=1e-9)
    assert searched != pytest.approx(np.mean((real - predicted) ** 2), rel=1e-6)


# --------------------------------------------------------------------------------------------
# After Stage 2: least squares here, on terms IGSR evaluates
# --------------------------------------------------------------------------------------------


def test_the_fit_the_search_made_is_recorded_for_the_later_stages():
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    assert adapter.fit_record(LinearRegression, {}) == {"penalty": None, "alpha": None,
                                                        "fit_intercept": True}
    assert adapter.fit_record(Ridge, {"fit_intercept": False}) == {"penalty": "l2", "alpha": 1.0,
                                                                   "fit_intercept": False}
    assert adapter.fit_record(Ridge, {"fit_intercept": False, "alpha": 1e-8})["alpha"] == 1e-8
    assert adapter.fit_record(Lasso, {"fit_intercept": False})["penalty"] == "l1"

    class NoOptimization:
        def __init__(self, *args, **kwargs):
            pass

    with pytest.raises(SystemExit, match="not a least-squares fit"):
        adapter.fit_record(NoOptimization, {"fit_intercept": False})


@pytest.mark.parametrize(("penalty", "alpha", "estimator"),
                         [("l2", 5.0, "Ridge"), ("l1", 0.05, "Lasso")])
def test_least_squares_refits_igsrs_penalised_fits_on_the_values_as_they_are(penalty, alpha,
                                                                             estimator):
    """A penalty depends on each column's scale, so the refit is made on the raw values, as IGSR
    makes it, and its predictions are scikit-learn's; the weights are then per standard deviation."""
    import sklearn.linear_model

    from seqmaestro.models.sparse import LeastSquares

    rng = np.random.default_rng(5)
    X = pd.DataFrame({"up": rng.normal(0, 10, 200), "down": rng.normal(0, 0.1, 200)})
    y = 0.2 * X["up"] - 5.0 * X["down"] + rng.normal(0, 0.5, 200)
    model = LeastSquares(penalty=penalty, alpha=alpha, fit_intercept=False).fit(X, y)
    reference = getattr(sklearn.linear_model, estimator)(alpha=alpha, fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-9)
    expected = dict(zip(X.columns, reference.coef_ * X.std(ddof=0).to_numpy(), strict=True))
    assert model.importances() == pytest.approx({k: v for k, v in expected.items() if v != 0})
    assert model.describe()["penalty"] == penalty and not model.describe()["fit_intercept"]
    with pytest.raises(ValueError, match="positive alpha"):
        LeastSquares(penalty=penalty)
    with pytest.raises(ValueError, match="positive alpha"):
        LeastSquares(alpha=1.0)


def test_an_igsr_arm_is_refitted_as_its_bundle_records_or_refused_without_the_record(tmp_path,
                                                                                     monkeypatch):
    from seqmaestro.pipeline.vendored import igsr_terms as igsr_calls

    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "arm.json").write_text(json.dumps({"bundle_version": 3}))
    with pytest.raises(ValueError, match="does not record the fit"):
        igsr_calls.igsr_fit(tmp_path)
    recorded = {"penalty": "l2", "alpha": 1e-8, "fit_intercept": False}
    (tmp_path / "bundle" / "arm.json").write_text(json.dumps({"bundle_version": 3, "fit": recorded}))
    rng = np.random.default_rng(6)
    X = rng.normal(size=(50, 2))
    fitted = igsr_calls.fit_step(igsr_calls.igsr_fit(tmp_path), None)(X, X[:, 0], ["a", "b"])
    assert {key: fitted.model.describe()[key] for key in recorded} == recorded


def test_least_squares_weighs_each_column_per_standard_deviation_and_predicts_as_ols():
    from sklearn.linear_model import LinearRegression

    from seqmaestro.models.sparse import LeastSquares

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"up": rng.normal(0, 10, 300), "down": rng.normal(0, 0.1, 300)})
    y = 0.2 * X["up"] - 5.0 * X["down"] + rng.normal(0, 0.01, 300)
    model = LeastSquares().fit(X, y)
    weights = model.importances()
    assert weights["up"] == pytest.approx(0.2 * X["up"].std(ddof=0), rel=1e-2)
    assert weights["down"] == pytest.approx(-5.0 * X["down"].std(ddof=0), rel=1e-2)
    assert model.directions() == {"up": INCREASES_TARGET, "down": DECREASES_TARGET}
    np.testing.assert_allclose(model.predict(X), LinearRegression().fit(X, y).predict(X), atol=1e-9)


def test_a_pool_saved_in_one_environment_is_read_back_whole_and_in_order(tmp_path):
    from seqmaestro.reliability.pool import SEARCH_HISTORY, Pool

    pool = Pool(names=["a", "b"], kinds=["llm_generated", "covariate"],
                params=[{}, {"column": "depth"}], descriptions=["one", "two"],
                values=np.arange(6.0).reshape(3, 2), source=SEARCH_HISTORY)
    pool.save(tmp_path / "pool.npz")
    back = Pool.load(tmp_path / "pool.npz", 3)
    assert (back.names, back.kinds, back.params, back.source) == (pool.names, pool.kinds,
                                                                  pool.params, SEARCH_HISTORY)
    np.testing.assert_array_equal(back.values, pool.values)
    with pytest.raises(RuntimeError, match="other rows"):
        Pool.load(tmp_path / "pool.npz", 4)


def a_stage2_cell(tmp_path: Path):
    """An IGSR cell as Stage 2 leaves one, on a planted continuous dataset: its inputs,
    a bundle holding the saved model and every accepted term, and the records and validation values
    ingest writes. The final terms read the planted covariate and a sequence count; the search also
    accepted a third term."""
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import save_feature_values, save_features

    dataset = planted_dataset(tmp_path, covariates=True, variable=True, continuous=True)
    cell = tmp_path / "run" / "results" / "igsr" / "llm_generated_joint+covariates" / "real" / "seed_1"
    (cell / "inputs").mkdir(parents=True)
    (cell / "bundle").mkdir()
    (cell / "inputs" / "columns.json").write_text(json.dumps({
        "sequence": {"column": "window", "window": None},
        "columns": {"signal": {"kind": "covariate", "params": {"column": "signal"},
                               "description": "the planted signal"}}}))
    accepted = [a_term("signal_itself", 'def signal_itself(X):\n    return X["signal"]'),
                a_term("gc_count", 'def gc_count(X):\n    return X["window"].str.count("GC")', fill=0.0),
                a_term("length", 'def length(X):\n    return X["window"].str.len()', fill=20.0)]
    (cell / "bundle" / "terms.json").write_text(json.dumps(accepted))
    final = accepted[:2]
    (cell / "bundle" / "model.json").write_text(json.dumps(a_model(final, [1.2, 0.1])))
    (cell / "bundle" / "arm.json").write_text(json.dumps({
        "bundle_version": 3, "fitted_model": "model.json",
        "fit": {"penalty": None, "alpha": None, "fit_intercept": True}}))
    (cell / "run.json").write_text(json.dumps({
        "labels": "real", "seed": 1, "train_fraction": 0.8, "weight_column": None,
        "predict_with": "igsr_bundle", "model": "igsr", "pool": "llm_generated_joint+covariates",
        "target_kind": "continuous", "model_settings": None}))
    val = pd.read_csv(dataset.val_path)
    values = np.column_stack([val["signal"].to_numpy(dtype=float),
                              val["window"].str.count("GC").to_numpy(dtype=float)])
    save_features(cell / "features.json", [FeatureRecord(name=term["name"], description="d",
                                                         kind="llm_generated") for term in final])
    save_feature_values(cell / "feature_values.npz", [term["name"] for term in final], values)
    return dataset, cell


@pytest.mark.crossenv
@pytest.mark.skipif(not igsr_python().exists(), reason="IGSR's interpreter is not installed here")
def test_an_igsr_arm_is_reduced_profiled_and_predicted_on_terms_igsr_evaluates(tmp_path, monkeypatch):
    from seqmaestro.pipeline.vendored import igsr_terms as igsr_calls
    from seqmaestro.reduction.arm import Source

    dataset, cell = a_stage2_cell(tmp_path)
    train, val = pd.read_csv(dataset.train_path), pd.read_csv(dataset.val_path)
    reduction = igsr_calls.reduce_arm(Source.read(cell), train, val, columns=dataset.columns,
                                      n_target=1)
    assert reduction.featurisation_check["n_columns_over_tolerance"] == 0
    assert reduction.final_columns == ["signal_itself"]
    model, columns = reduction.models["reduced"]

    reduced = tmp_path / "run" / "reduction" / "igsr" / "llm_generated_joint+covariates" / "reduced"
    about = igsr_calls.save_model(reduced, model)
    assert about["predict_with"] == igsr_calls.PREDICT_WITH
    (reduced / "run.json").write_text(json.dumps({"source_cell": str(cell.relative_to(tmp_path / "run")),
                                                  "columns": columns}))
    predicted = igsr_calls.predict_cell(tmp_path / "run", reduced, dataset.inputs(val), "window")
    np.testing.assert_allclose(predicted, reduction.val_scores["reduced"], atol=1e-9)

    side = igsr_calls.IgsrFits(Source.read(cell), pd.concat([train, val], ignore_index=True),
                               columns=dataset.columns)
    pool = side.pool(tmp_path / "panel")
    assert pool.names == ["signal_itself", "gc_count", "length"] and pool.source == "search_history"
    assert pool.kinds[0] == "covariate" and pool.params[0] == {"column": "signal"}
    np.testing.assert_array_equal(pool.values[:, 2], pd.concat([train, val])["window"].str.len())
    assert (tmp_path / "panel" / "pool.npz").exists()
