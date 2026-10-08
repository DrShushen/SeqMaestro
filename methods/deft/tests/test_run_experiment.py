"""`run_experiment` — the wiring, offline.

Every part it assembles is tested on its own; the assembly was not. It is where provenance,
the budget, the code guard, the dataset plugin, the run-directory check and the integrity
policy are attached to each other, and where each of their failure paths is handled — and the
only thing exercising it end to end was the two `llm`-marked tests, which skip without
credentials.

The engine is replaced by a scripted one through Hydra's own `_target_`, so this composes the
*shipped* config and instantiates the real objects for everything else. Nothing here touches
the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from deft.llm.engines.litellm_engine import LiteLLMEngine
from deft.provenance import DegradedRunError

DEFT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = str(DEFT_ROOT / "src" / "deft" / "conf")

PROPOSAL = (
    '{"rationale": "GC content separates the classes", '
    '"description": "Fraction of G and C in the sequence", "name": "gc_fraction"}'
)
CODE = (
    "def gc_fraction(X):\n"
    "    seqs = X['raw_sequence'].astype(str)\n"
    "    return (seqs.str.count('G') + seqs.str.count('C')) / seqs.str.len()"
)


def _response(content: str, n: int) -> dict:
    return {
        "choices": [{"message": {"content": content}} for _ in range(n)],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class ScriptedEngine(LiteLLMEngine):
    """Answers as a working model would, without a network call.

    Subclasses the real engine so the config group instantiates unchanged and everything
    below `query_llm` -- recording, redaction -- is the real code.
    """

    async def query_llm(self, list_prompts: list, n_generations_per_prompt: int):
        if self.budget is not None:
            self.budget.check()
        responses = []
        for prompt in list_prompts:
            content = CODE if "python code" in prompt else PROPOSAL
            response = _response(content, n_generations_per_prompt)
            self._record_request(response, n_generations_per_prompt)
            responses.append(response)
        return responses


class DeadEngine(LiteLLMEngine):
    """Every request fails after its retries — the silent-degradation case."""

    async def query_llm(self, list_prompts: list, n_generations_per_prompt: int):
        if self.budget is not None:
            self.budget.check()
        for _ in list_prompts:
            self._record_request(None, n_generations_per_prompt)
        return [None for _ in list_prompts]


class ExplodingEngine(LiteLLMEngine):
    """A fault that is not one of the handled ones."""

    async def query_llm(self, list_prompts: list, n_generations_per_prompt: int):
        raise RuntimeError("the provider melted")


@pytest.fixture
def csv_path(tmp_path):
    """A tiny dataset the scripted feature actually separates."""
    high = ["GCGCGCGCGC", "CGCGCGCGCG", "GGCCGGCCGG", "CCGGCCGGCC"] * 3
    low = ["ATATATATAT", "TATATATATA", "AATTAATTAA", "TTAATTAATT"] * 3
    frame = pd.DataFrame(
        {"raw_sequence": high + low, "label": [1] * len(high) + [0] * len(low)}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return str(path)


def build_cfg(tmp_path, csv_path, engine: str | None = "ScriptedEngine", extra=()):
    """Compose the shipped config exactly as the CLI does, and publish HydraConfig.

    `run_experiment` reads its output directory from `HydraConfig`, which `hydra.main` would
    normally set; `return_hydra_config=True` plus `set_config` is the supported way to do that
    from a test.

    `engine=None` leaves the configured engine in place, which is how the `llm`-marked test in
    `test_llm_regression.py` reuses this against a real model.
    """
    overrides = [
        *(
            [f"llm._target_=tests.test_run_experiment.{engine}",
             "llm_models.api_key=not-a-real-key"]
            if engine is not None else []
        ),
        f"dataset.csv_path={csv_path}",
        "dataset.target_column=label",
        "max_depth=1",
        "min_samples_leaf_fraction=0.1",
        "params_generation.use_reflection=false",
        "params_generation.n_reflections=0",
        "params_generation.population_size=2",
        "params_generation.n_samples_per_prompt=2",
        "params_generation.initialize_with_naive_features=false",
        f"hydra.run.dir={tmp_path}",
        *extra,
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        # `hydra.main` fills this in from `hydra.run.dir`; `compose` leaves it mandatory-missing.
        with open_dict(cfg):
            cfg.hydra.runtime.output_dir = str(tmp_path)
        HydraConfig.instance().set_config(cfg)
        return cfg


def run(cfg):
    from deft.cli.run_deft_tree import run_experiment

    return run_experiment(cfg)


# ------------------------------------------------------------------------------- happy path


def test_a_run_writes_its_three_artifacts(tmp_path, csv_path):
    result = run(build_cfg(tmp_path, csv_path))

    for key in ("tree_path", "provenance_path", "token_usage_path"):
        assert Path(result[key]).is_file(), f"{key} was not written"
    assert result["status"] == "fit_complete"


def test_the_result_reports_what_the_run_did(tmp_path, csv_path):
    result = run(build_cfg(tmp_path, csv_path))

    assert result["integrity"]["nodes_expanded"] >= 1
    assert result["integrity"]["llm_split_share"] == 1.0, (
        "naive features are disabled here, so every split must be the generated one"
    )
    assert result["total_tokens"] > 0, "token accounting is not reaching the report"
    assert result["budget_stopped"] is False
    assert result["budget_stopped_reason"] is None


def test_provenance_records_the_whole_chain(tmp_path, csv_path):
    result = run(build_cfg(tmp_path, csv_path))
    record = json.loads(Path(result["provenance_path"]).read_text())

    node = record["nodes"][0]
    for field in ("semantic_prompts", "semantic_proposals_parsed", "code_prompts",
                  "code_features_parsed", "llm_features_usable"):
        assert node[field] > 0, f"{field} was not recorded"
    assert node["split_feature_origin"] == "llm"


def test_the_fitted_tree_is_written_to_the_log_not_only_stdout(tmp_path, csv_path, caplog):
    """A detached sweep keeps the log and loses the terminal, so the tree has to be in it."""
    import logging

    with caplog.at_level(logging.INFO):
        run(build_cfg(tmp_path, csv_path))

    assert "Decision Tree Structure:" in caplog.text
    assert "gc_fraction" in caplog.text


def test_the_saved_tree_holds_no_credential(tmp_path, csv_path):
    """The engine reaches the artifact through the feature finder."""
    result = run(build_cfg(tmp_path, csv_path))

    assert b"not-a-real-key" not in Path(result["tree_path"]).read_bytes()


def test_the_tree_can_be_loaded_back_and_used(tmp_path, csv_path):
    import dill

    result = run(build_cfg(tmp_path, csv_path))
    tree = dill.loads(Path(result["tree_path"]).read_bytes())

    predictions = tree.predict(pd.read_csv(csv_path)[["raw_sequence"]])
    assert len(predictions) == 24


# ---------------------------------------------------------------------------- failure paths


def test_a_degraded_run_fails_but_still_writes_its_provenance(tmp_path, csv_path):
    """The point of the integrity policy: stop, and leave behind the evidence of why."""
    cfg = build_cfg(tmp_path, csv_path, engine="DeadEngine")

    with pytest.raises(DegradedRunError):
        run(cfg)

    record = json.loads((tmp_path / "provenance.json").read_text())
    assert record["nodes"][0]["llm_features_usable"] == 0
    assert (tmp_path / "token_usage.json").is_file()


def test_warn_only_lets_a_degraded_run_finish(tmp_path, csv_path):
    """Same run, different policy — so the numbers can be seen without failing."""
    cfg = build_cfg(
        tmp_path, csv_path, engine="DeadEngine",
        extra=["integrity=warn_only", "params_generation.initialize_with_naive_features=true"],
    )

    result = run(cfg)

    assert result["integrity"]["llm_split_share"] == 0.0
    assert result["integrity"]["nodes_with_no_usable_llm_feature"] > 0


def test_an_unexpected_error_during_fit_is_re_raised(tmp_path, csv_path):
    """Upstream logged and exited 0 here, which made a failed run look like a finished one."""
    cfg = build_cfg(tmp_path, csv_path, engine="ExplodingEngine")

    with pytest.raises(RuntimeError, match="the provider melted"):
        run(cfg)

    assert (tmp_path / "provenance.json").is_file(), "provenance must survive the failure"


def test_a_run_refuses_a_directory_that_already_holds_a_run(tmp_path, csv_path):
    """And refuses it *before* the engine is reached, so nothing is spent."""
    (tmp_path / "provenance.json").write_text("{}")
    cfg = build_cfg(tmp_path, csv_path)

    with pytest.raises(FileExistsError):
        run(cfg)


def test_reuse_can_be_allowed(tmp_path, csv_path):
    (tmp_path / "provenance.json").write_text("{}")
    cfg = build_cfg(tmp_path, csv_path, extra=["allow_run_dir_reuse=true"])

    assert run(cfg)["status"] == "fit_complete"


# ------------------------------------------------------------------------- config validation


@pytest.mark.parametrize(
    "fraction", [0, -0.1, 1.5],
)
def test_an_out_of_range_min_samples_leaf_fraction_is_refused(tmp_path, csv_path, fraction):
    cfg = build_cfg(tmp_path, csv_path, extra=[f"min_samples_leaf_fraction={fraction}"])

    with pytest.raises(ValueError, match="min_samples_leaf_fraction"):
        run(cfg)


@pytest.mark.parametrize(
    "block", ["dataset", "llm", "feature_finder", "splitting_criterion", "tree"],
)
def test_a_missing_config_block_names_itself(tmp_path, csv_path, block):
    cfg = build_cfg(tmp_path, csv_path)
    with open_dict(cfg):
        del cfg[block]

    with pytest.raises((ValueError, KeyError), match=block):
        run(cfg)


def test_a_sequence_column_set_on_the_prompts_block_is_refused(tmp_path, csv_path):
    """It belongs to the dataset; accepting it in both places would let them disagree."""
    cfg = build_cfg(tmp_path, csv_path, extra=["+params_generation.sequence_column=raw_sequence"])

    with pytest.raises(ValueError, match=r"dataset\.sequence_column"):
        run(cfg)


# ---------------------------------------------------------------------------------- budget


def test_a_budget_stops_the_tree_instead_of_failing_the_run(tmp_path, csv_path):
    """A reached limit produces a complete but shallower tree, and says so."""
    cfg = build_cfg(
        tmp_path, csv_path,
        extra=["max_depth=4", "budget.max_api_calls=1",
               "params_generation.initialize_with_naive_features=true"],
    )

    result = run(cfg)

    assert result["budget_stopped"] is True
    assert result["budget_stopped_reason"] == "max_api_calls"
    assert Path(result["tree_path"]).is_file(), "a budget stop still produces a tree"


# --------------------------------------------------------------- the CART baseline finder


@pytest.fixture
def tabular_csv_path(tmp_path):
    """CART thresholds columns directly, so it needs one it can compare numerically.

    A raw sequence string is not that -- see `test_cart_cannot_split_a_raw_sequence_column`.
    """
    # The sequence is constant, so no positional feature can split on it and only `signal` can.
    frame = pd.DataFrame(
        {
            "raw_sequence": ["GCGCGCGCGC"] * 24,
            "signal": [9.0, 1.0] * 12,
            "label": [1, 0] * 12,
        }
    )
    path = tmp_path / "tabular.csv"
    frame.to_csv(path, index=False)
    return str(path)


def cart_cfg(tmp_path, csv_path, extra=()):
    """`feature_finder=cart` with no credentials of any kind in the config."""
    return build_cfg(
        tmp_path, csv_path, engine=None,
        extra=["feature_finder=cart", "max_depth=2", *extra],
    )


def test_cart_runs_with_no_language_model_at_all(tmp_path, tabular_csv_path):
    """The point of the baseline: same tree, same criterion, same threshold search, no API."""
    result = run(cart_cfg(tmp_path, tabular_csv_path))

    assert result["status"] == "fit_complete"
    assert result["total_tokens"] == 0
    assert result["integrity"]["nodes_expanded"] >= 1
    assert Path(result["tree_path"]).is_file()


def test_cart_needs_no_credentials(tmp_path, tabular_csv_path, monkeypatch):
    """No engine is built, so an unresolvable api_key in the config must not matter — a
    baseline that demands an API key it never calls defeats the purpose."""
    monkeypatch.delenv("DEFT_API_KEY", raising=False)

    result = run(cart_cfg(tmp_path, tabular_csv_path))

    assert result["status"] == "fit_complete"


def test_the_integrity_policy_is_skipped_rather_than_failing_the_run(tmp_path, tabular_csv_path, caplog):
    """"Did the language model contribute?" has no meaning here, and the answer would be no at
    every node — so `integrity=strict` must not turn a working baseline into a failure."""
    import logging

    with caplog.at_level(logging.INFO):
        result = run(cart_cfg(tmp_path, tabular_csv_path, extra=["integrity=strict"]))

    assert result["status"] == "fit_complete"
    assert "Integrity checks do not apply" in caplog.text


def test_cart_splits_on_the_column_that_separates_the_classes(tmp_path, tabular_csv_path):
    import dill

    result = run(cart_cfg(tmp_path, tabular_csv_path))
    tree = dill.loads(Path(result["tree_path"]).read_bytes())

    assert tree.root.feature.name == "value_of_signal", (
        "the numeric covariate is the only column that can separate these classes"
    )
    assert not tree.root._is_leaf()


def test_cart_splits_a_raw_sequence_column_by_position(tmp_path, csv_path):
    """The sequence is decoded into per-position indicators, so CART works on sequence data.

    Upstream's finder made one identity feature per column, which on a DNA string cannot be
    cast to a number and so never split -- a baseline that silently did nothing on exactly the
    data this method is for.
    """
    import dill

    result = run(cart_cfg(tmp_path, csv_path))
    tree = dill.loads(Path(result["tree_path"]).read_bytes())

    assert result["integrity"]["nodes_split"] >= 1
    assert tree.root.feature.name.startswith("pos_"), (
        f"expected a positional split, got `{tree.root.feature.name}`"
    )
    assert "raw_sequence" in tree.root.feature.string


def test_cart_uses_the_same_candidate_population_as_a_deft_run(tmp_path, csv_path):
    """Not an accident: it is `get_naive_features`, so `llm_split_share` and this baseline are
    measured against the same yardstick."""
    from deft.features.feature_finder import CartFeatureFinder
    from deft.utils.feature import get_naive_features

    X = pd.read_csv(csv_path)[["raw_sequence"]]

    cart = CartFeatureFinder().generate_features(X, sequence_column="raw_sequence")
    naive = get_naive_features(X, sequence_column="raw_sequence")

    assert [f.name for f in cart] == [f.name for f in naive]
