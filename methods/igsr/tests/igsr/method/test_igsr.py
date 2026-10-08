"""Quick-running regression tests for the main IGSR method (`igsr`).

Goal: a fast, deterministic safety net to catch *particularly bad* regressions during
repo cleanup -- NOT high coverage.

Design:
  * The whole pipeline is driven with the LLM **stubbed** (patch `LiteLLMAgent.run`),
    so these tests need no network and no credentials, and are fully deterministic.
  * `algorithmic_pruning=True` -> only the *propose* agent is ever called (pruning is
    pure influence-based math), so a single stub suffices.
  * Group A: pure-math units (no stub) -- influence, design matrix, fit/eval, parsing.
  * Group B: end-to-end smoke with the propose-agent stubbed -- result-dict contract,
    MCTS wiring, determinism, multi-target, compute-profiler.
  * Group C: one opt-in `@pytest.mark.slow` real-LLM end-to-end (skipped by default;
    run with `-m slow`, needs a valid Azure key).
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from igsr.agent.agent import AgentResponse, LiteLLMAgent, PostprocessorOutput
from igsr.compose import compose_config
from igsr.dataset.data_bundle import DataBundle
from igsr.method.fitted_model import FittedModel
from igsr.method.igsr import (
    igsr,
    prompt_generate_terms,
    train_and_evaluate,
)
from igsr.method.igsr_utils import (
    DesignMatrix,
    ols_influence,
    postprocessor_generate_terms,
    prepare_dataset,
    safe_eval,
)
from igsr.tree_search.session import SearchSession

RESULT_KEYS = {
    "seed",
    "best_iter_id",
    "best_formula",
    "best_metric",
    "best_terms",
    "best_test_metrics",
    "best_model",
    "term_records",
    "term_failures",
    "proposal_stats",
    "history_validation",
    "history_test",
    "history_ood_test",
    "compute_profiler",
}
METRIC_KEYS = {"mse", "rmse", "r2", "nrmse", "nmse", "accuracy_tol", "accuracy_tol_max"}


# ---------------------------------------------------------------------------
# Helpers: config + toy dataset + LLM stub
# ---------------------------------------------------------------------------


def _base_cfg_dict():
    return {
        "logging": {"logger_name": "test_igsr", "mode": "w", "log_file": ""},
        "llm": {
            "kind": "azure",
            "model_type": "gpt-4o",
            "model_version": "2024-11-20",
            "deployment": "gpt-4o-es",
            "endpoint": "https://dummy.openai.azure.com/",
            "api_version": "2025-03-01-preview",
            "api_key": "dummy-key",
        },
        "dataset": {
            "kind": "synthetic",
            "name": "toy",
            "task": "regression",
            "target_columns": ["y"],
            "multiple_output_targets": False,
            "seed": 0,
        },
        "experiment": {
            "kind": "igsr",
            "term_language": "expression",
            # search / loop
            "tree": False,
            "n_iters": 1,
            "total_budget": 2,
            "depth_limit": 3,
            "n_successors": 2,
            "c": 1.41421356,
            "rollout_is_just_node_reward": True,
            "rollout_depth": 1,
            # prompting
            "terms_per_round": 5,
            "first_round_n_candidates": 10,
            "keep_n_terms": 6,
            "print_prompts": False,
            "preview_N_elements": 10,
            "history_enabled": True,
            "simplified_prompts": False,
            "max_agent_steps": 1,
            "retry_attempts": 1,
            # feedback / pruning
            "influence_feedback": True,
            "algorithmic_pruning": True,
            # early stop
            "early_stop_enabled": False,
            "early_stop_patience": 10,
            # optimization / influence
            "optimization_method": "linear",
            "refit_aware": False,
            "refit_aware_efficient": False,
            "fallback_ridge_small_alpha_for_stability": False,
            "error_on_all_weights_zero": False,
            # budgets (off)
            "break_on_tokens": False,
            "break_above_N_tokens": 260000,
            "break_on_wallclock": False,
            "break_above_wallclock": 300,
            # misc
            "use_xgb_predictor": False,
            "keep_n_columns": None,
            "additional_validation_of_terms": True,
            "allow_error_history_length": 999999,  # -> any pipeline error raises (visible in tests)
            "show_feature_importance": False,
            "feature_importance_max_features": 20,
            "collapse_seq_feats": False,
            "focus_on_seq_feats": False,
        },
    }


def _make_igsr_cfg(**overrides):
    """Full MainConfig-shaped OmegaConf cfg. Accepts dotted overrides, e.g.
    `_make_igsr_cfg(**{"experiment.tree": True})`."""
    cfg = OmegaConf.create(_base_cfg_dict())
    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val, merge=False)
    return cfg


def _toy_bundle(multi=False, n_train=120, n_val=40, n_test=40, seed=0):
    """Toy dataset with a known linear-in-basis relationship.

    single: y  = 2*x1 + 3*sin(x2)
    multi : y1 = 2*x1 + 3*sin(x2), y2 = x1*x2
    """
    rng = np.random.default_rng(seed)

    def make_df(n):
        x1 = rng.uniform(-2.0, 2.0, n)
        x2 = rng.uniform(-2.0, 2.0, n)
        cols = {"x1": x1, "x2": x2}
        cols["y"] = 2.0 * x1 + 3.0 * np.sin(x2)
        if multi:
            cols["y1"] = 2.0 * x1 + 3.0 * np.sin(x2)
            cols["y2"] = x1 * x2
            del cols["y"]
        return pd.DataFrame(cols)

    target_columns = ["y1", "y2"] if multi else ["y"]
    return DataBundle(
        name="toy_igsr",
        data_settings=None,
        dataset_train=make_df(n_train),
        dataset_validation=make_df(n_val),
        dataset_test=make_df(n_test),
        target_columns=target_columns,
        equation=None,
        data_dictionary="Toy synthetic dataset. Features x1, x2. Targets are linear in {x1, sin(x2), x1*x2}.",
    )


def _stub_propose(monkeypatch, terms, tokens=(10, 5)):
    """Patch the LLM propose-agent to return fixed `terms`.

    Bypasses the network + postprocessor (parsing is covered separately by
    `test_postprocessor_*`). Sets the usage attrs the caller reads afterwards.
    """
    def fake_run(self):
        self.usage_history = [{"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}]
        self.num_llm_calls = 1
        return AgentResponse(
            success=True,
            message="",
            postprocessor_output=PostprocessorOutput(
                success=True, inform_agent=False, parsed_output=list(terms)
            ),
        )

    monkeypatch.setattr(LiteLLMAgent, "run", fake_run)


# ===========================================================================
# Group A -- pure-math units (no stub)
# ===========================================================================


def test_ols_influence_separates_relevant_from_irrelevant_terms():
    """Core IGSR signal: Delta~0 for a true-zero-weight term, Delta>0 for real terms."""
    rng = np.random.default_rng(1)
    n = 200
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 2.0 * a + 3.0 * b  # 'c' is irrelevant (true weight 0)
    phi = np.column_stack([a, b, c])  # columns: [a, b, c]

    cfg = _make_igsr_cfg()
    res = ols_influence(cfg, phi, y, phi, y)  # reuse same data as "validation"

    # OLS recovers weights [2, 3, 0]
    np.testing.assert_allclose(res.weights.ravel(), [2.0, 3.0, 0.0], atol=1e-8)
    delta = res.delta.ravel()
    # Irrelevant term's influence ~ 0; relevant terms strictly larger.
    assert abs(delta[2]) < 1e-8
    assert delta[0] > 1e-6 and delta[1] > 1e-6
    # Deterministic top-K-by-influence ranking keeps the two real terms.
    top2 = set(np.argsort(delta)[::-1][:2].tolist())
    assert top2 == {0, 1}


def test_ols_influence_multi_output_shapes():
    rng = np.random.default_rng(2)
    n = 150
    a, b = rng.normal(size=n), rng.normal(size=n)
    Y = np.column_stack([2 * a + b, a - 3 * b])  # 2 outputs
    phi = np.column_stack([a, b])
    res = ols_influence(_make_igsr_cfg(), phi, Y, phi, Y)
    assert res.weights.shape == (2, 2)  # (n_terms, n_outputs)
    assert res.delta.shape == (2, 2)
    assert res.delta_validation.shape == (2, 2)


def test_design_matrix_evaluates_and_drops_bad_terms():
    data = {"x1": np.array([1.0, 2.0, 3.0]), "x2": np.array([0.0, np.pi / 2, np.pi])}
    dm = DesignMatrix.from_terms(["x1", "x1*x2", "np.sin(x2)"], data)
    assert dm.term_names == ["x1", "x1*x2", "np.sin(x2)"]
    assert dm.phi.shape == (3, 3)
    np.testing.assert_allclose(dm.phi[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dm.phi[:, 2], np.sin([0.0, np.pi / 2, np.pi]), atol=1e-9)

    # An unevaluable term (undefined name) is dropped, not fatal, and recorded in errors.
    dm2 = DesignMatrix.from_terms(["x1", "undefined_var"], data)
    assert dm2.term_names == ["x1"]
    assert dm2.errors


def test_safe_eval_rejects_dunder_access():
    with pytest.raises(ValueError):
        safe_eval("__import__", {})


def test_train_and_evaluate_recovers_low_mse():
    bundle = _toy_bundle()
    pd_ds = prepare_dataset(bundle, exp_logger=None)
    terms = ["x1", "np.sin(x2)"]
    reg, mse_per_out, mse_total, metrics = train_and_evaluate(
        _make_igsr_cfg(), terms, pd_ds.data, pd_ds.y, pd_ds.data_test, pd_ds.y_test
    )
    assert mse_total < 1e-6  # perfect basis -> near-zero error
    assert METRIC_KEYS.issubset(metrics.keys())
    assert np.isfinite(metrics["mse"]) and metrics["r2"] > 0.99


def test_postprocessor_generate_terms_parses_and_rejects():
    """IGSR LLM-output parser: extracts terms from a well-formed response, rejects a bad one."""
    from types import SimpleNamespace

    bundle = _toy_bundle()
    pd_ds = prepare_dataset(bundle, exp_logger=None)
    cfg = _make_igsr_cfg()
    kwargs = dict(
        cfg=cfg,
        data_bundle=bundle,
        seed=0,
        iter_num=0,
        node_id=None,
        stage_name="propose",
        terms_sentinel="TERMS:",
        prepared_dataset=pd_ds,
        correct_format_generate_terms="",
        current_terms=[],
    )

    good = SimpleNamespace(conversation_history=[{"role": "assistant", "content": "TERMS:\n```\nx1\nnp.sin(x2)\n```"}])
    out = postprocessor_generate_terms(good, **kwargs)
    assert out.success is True
    assert out.parsed_output == ["x1", "np.sin(x2)"]

    bad = SimpleNamespace(conversation_history=[{"role": "assistant", "content": "no sentinel here"}])
    out_bad = postprocessor_generate_terms(bad, **kwargs)
    assert out_bad.success is False


def test_prompt_generate_terms_builds_string():
    """Prompt construction shouldn't crash and should embed the data dictionary + sentinel."""
    bundle = _toy_bundle()
    pd_ds = prepare_dataset(bundle, exp_logger=None)
    prompt = prompt_generate_terms(
        _make_igsr_cfg(), bundle, data=pd_ds.data, y=pd_ds.y, history=[], current_terms=[], terms_sentinel="TERMS:"
    )
    assert isinstance(prompt, str)
    assert "TERMS:" in prompt
    assert bundle.data_dictionary in prompt


# ===========================================================================
# Group B -- end-to-end smoke, propose-agent stubbed
# ===========================================================================


def test_iterative_end_to_end_recovers_true_basis(monkeypatch):
    bundle = _toy_bundle()
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    result = igsr(_make_igsr_cfg(**{"experiment.tree": False, "experiment.n_iters": 2}), seed=7, data_bundle=bundle)

    assert set(result.keys()) == RESULT_KEYS
    assert result["seed"] == 7
    assert isinstance(result["best_formula"], str) and result["best_formula"].strip()
    assert np.isfinite(result["best_metric"]) and result["best_metric"] < 1e-6
    assert "x1" in result["best_formula"] and "np.sin(x2)" in result["best_formula"]
    assert len(result["history_validation"]) >= 1


def test_mcts_end_to_end_runs_and_returns_contract(monkeypatch):
    bundle = _toy_bundle()
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    cfg = _make_igsr_cfg(
        **{"experiment.tree": True, "experiment.total_budget": 2, "experiment.n_successors": 2, "experiment.depth_limit": 3}
    )
    result = igsr(cfg, seed=3, data_bundle=bundle)
    assert set(result.keys()) == RESULT_KEYS
    assert np.isfinite(result["best_metric"]) and result["best_metric"] < 1e-6
    assert isinstance(result["best_formula"], str) and result["best_formula"].strip()


def test_deterministic_across_repeated_runs(monkeypatch):
    bundle = _toy_bundle()
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    r1 = igsr(_make_igsr_cfg(**{"experiment.n_iters": 2}), seed=42, data_bundle=bundle)
    r2 = igsr(_make_igsr_cfg(**{"experiment.n_iters": 2}), seed=42, data_bundle=_toy_bundle())
    assert r1["best_formula"] == r2["best_formula"]
    assert r1["best_metric"] == pytest.approx(r2["best_metric"], rel=1e-12, abs=1e-12)


def test_multi_target_end_to_end(monkeypatch):
    bundle = _toy_bundle(multi=True)
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)", "x1*x2"])
    cfg = _make_igsr_cfg(
        **{"dataset.target_columns": ["y1", "y2"], "dataset.multiple_output_targets": True, "experiment.n_iters": 1}
    )
    result = igsr(cfg, seed=5, data_bundle=bundle)
    assert set(result.keys()) == RESULT_KEYS
    assert np.isfinite(result["best_metric"])
    assert result["best_metric"] < 1e-6  # both targets are in the injected basis


def test_compute_profiler_contract(monkeypatch):
    bundle = _toy_bundle()
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    n_iters = 3
    result = igsr(_make_igsr_cfg(**{"experiment.n_iters": n_iters}), seed=1, data_bundle=bundle)
    cp = result["compute_profiler"]
    assert set(cp.keys()) == {
        "wall_clock_total",
        "wall_clock_llm_calls",
        "input_tokens",
        "output_tokens",
        "number_llm_calls",
    }
    # One stubbed propose call per iteration.
    assert cp["number_llm_calls"] == n_iters
    assert cp["input_tokens"] == 10 * n_iters
    assert cp["output_tokens"] == 5 * n_iters
    assert cp["wall_clock_total"] > 0


# ===========================================================================
# Group D -- the copy's patches, in the expression form
# ===========================================================================


def _stub_propose_in_turn(monkeypatch, replies):
    """Patch the propose-agent to return each of `replies` in turn, one per call."""
    replies = iter(replies)

    def fake_run(self):
        self.usage_history = [{"prompt_tokens": 1, "completion_tokens": 1}]
        self.num_llm_calls = 1
        return AgentResponse(success=True, message="", postprocessor_output=PostprocessorOutput(
            success=True, inform_agent=False, parsed_output=list(next(replies))))

    monkeypatch.setattr(LiteLLMAgent, "run", fake_run)


def test_the_test_split_is_optional_and_chooses_nothing(monkeypatch):
    """Without a test split the search runs and reports no test metrics; with one it reports them,
    and chooses the same iteration."""
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    cfg = _make_igsr_cfg(**{"experiment.n_iters": 2})
    with_test = igsr(cfg, seed=4, data_bundle=_toy_bundle())
    bundle = _toy_bundle()
    bundle.dataset_test = None
    without = igsr(cfg, seed=4, data_bundle=bundle)

    assert without["history_test"] is None and without["best_test_metrics"] is None
    assert with_test["best_test_metrics"]["mse"] < 1e-6
    assert (with_test["best_iter_id"], with_test["best_formula"]) == (without["best_iter_id"], without["best_formula"])


def test_the_iteration_chosen_is_the_best_on_validation_not_on_test(monkeypatch):
    """On a test split where sin(x2) plays no part, the first iteration (x1 alone) is best on test and
    the second (x1 and sin(x2)) on validation. The search reports the second."""
    _stub_propose_in_turn(monkeypatch, [["x1"], ["np.sin(x2)"]])
    bundle = _toy_bundle()
    bundle.dataset_test = bundle.dataset_test.assign(y=2.0 * bundle.dataset_test["x1"])
    result = igsr(_make_igsr_cfg(**{"experiment.n_iters": 2}), seed=0, data_bundle=bundle)

    assert result["best_iter_id"] == "1" and set(result["best_terms"]) == {"x1", "np.sin(x2)"}
    assert min(result["history_test"], key=lambda entry: entry["metrics"]["mse"])["iter_id"] == "0"


def test_the_tree_search_is_seeded_by_the_run(monkeypatch):
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    seen = {}
    search = SearchSession.mcts

    def recording(self, start, **kwargs):
        seen.update(kwargs)
        return search(self, start, **kwargs)

    monkeypatch.setattr(SearchSession, "mcts", recording)
    cfg = _make_igsr_cfg(**{"experiment.tree": True, "experiment.total_budget": 2, "experiment.n_successors": 2})
    igsr(cfg, seed=11, data_bundle=_toy_bundle())
    assert seen["seed"] == 11


def test_the_chosen_model_is_saved_and_predicts_new_rows(monkeypatch):
    _stub_propose(monkeypatch, ["x1", "np.sin(x2)"])
    bundle = _toy_bundle()
    model = igsr(_make_igsr_cfg(**{"experiment.n_iters": 1}), seed=2, data_bundle=bundle)["best_model"]

    np.testing.assert_allclose(model.weights.ravel(), [2.0, 3.0], atol=1e-8)
    restored = FittedModel.from_dict(json.loads(json.dumps(model.to_dict())))
    rows = bundle.dataset_test[["x1", "x2"]]
    np.testing.assert_allclose(restored.predict(rows).ravel(), bundle.dataset_test["y"], atol=1e-6)


def test_a_round_that_keeps_no_term_is_recorded_and_its_model_predicts_the_mean(monkeypatch):
    """With `keep_n_terms` 0 the pruner keeps nothing, a case the loop records as a round with no
    equation; the saved model then predicts the training mean."""
    _stub_propose(monkeypatch, ["x1"])
    bundle = _toy_bundle()
    result = igsr(_make_igsr_cfg(**{"experiment.n_iters": 1, "experiment.keep_n_terms": 0}), seed=0,
                  data_bundle=bundle)

    assert result["best_terms"] == []
    predicted = result["best_model"].predict(bundle.dataset_validation[["x1", "x2"]]).ravel()
    np.testing.assert_allclose(predicted, bundle.dataset_train["y"].mean())


def test_importing_the_method_writes_nothing(tmp_path):
    subprocess.run([sys.executable, "-c", "import igsr.logging_setup, igsr.method.igsr"], cwd=tmp_path, check=True)
    assert list(tmp_path.iterdir()) == []


def test_the_configuration_composes_and_the_language_model_is_named():
    with pytest.raises(Exception, match="llm"):
        compose_config([])
    cfg = compose_config(["llm=seqmaestro_azure"])
    assert cfg.experiment.term_language == "expression"
    assert cfg.llm.kind == "azure" and cfg.logging.log_file is None
    # One config for each provider SeqMaestro's .env can name. A kind is LiteLLM's route.
    assert compose_config(["llm=seqmaestro_openai"]).llm.kind == "openai"
    assert compose_config(["llm=seqmaestro_claude"]).llm.kind == "anthropic"
    assert set(cfg.experiment.function_terms) >= {"max_undefined_share", "repair_rounds", "n_examples",
                                                  "max_example_length", "timeout_seconds"}


# ===========================================================================
# Group C -- opt-in real-LLM end-to-end (skipped by default; run with `-m slow`)
# ===========================================================================


@pytest.mark.slow
def test_real_llm_end_to_end(live_llm_config):
    """Genuine 1-iteration IGSR run against a live model: the provider SeqMaestro's root .env names,
    with the settings the copy's config for it reads there. Needs a valid key."""
    from omegaconf import OmegaConf

    bundle = _toy_bundle()
    llm = OmegaConf.to_container(compose_config([live_llm_config]).llm, resolve=True)
    cfg = _make_igsr_cfg(
        **{
            "experiment.tree": False,
            "experiment.n_iters": 1,
            **{f"llm.{key}": value for key, value in llm.items()},
        }
    )
    result = igsr(cfg, seed=0, data_bundle=bundle)
    assert set(result.keys()) == RESULT_KEYS
    assert isinstance(result["best_formula"], str)
    assert np.isfinite(result["best_metric"])
    assert result["compute_profiler"]["number_llm_calls"] >= 1
    assert result["compute_profiler"]["input_tokens"] > 0
