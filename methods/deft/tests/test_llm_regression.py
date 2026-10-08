"""Regression tests that exercise the real LLM path end to end.

Why these exist: DEFT degrades *silently* when its LLM backend is misconfigured. The feature
population is seeded with naive single-position features (`get_naive_features`,
src/deft/utils/feature.py:42), so a dead endpoint, a rejected request parameter or a parsing
regression does not raise -- it just yields a tree built entirely from `pos_N_is_X` splits
that still fits, still serializes and still prints. Upstream commit ee31e77 fixed one instance
of this; these tests are the standing guard against the rest.

The tests are marked `llm`, cost a few thousand tokens, and skip cleanly when the SeqMaestro
root `.env` has no Azure credentials.

    pytest tests/test_llm_regression.py            # runs them if credentials are present
    pytest -m "not llm"                            # everything else, offline
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra.utils import instantiate

from deft.provenance import IntegrityPolicy, RunProvenance

# DEFT's own naive features are exactly single-position nucleotide indicators.
NAIVE_FEATURE_NAME = re.compile(r"^pos_(neg)?\d+_is_[ACGTN]$")

SEQUENCE_LENGTH = 101
UPSTREAM_LENGTH = 50


def make_planted_dataset(
    n: int = 240,
    gc_high: float = 0.75,
    gc_low: float = 0.25,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sequences whose label is set by upstream G+C fraction and by nothing else.

    Each upstream position is independently G or C with probability `gc_high` (label 1) or
    `gc_low` (label 0), so the best possible *single-position* rule is right about 75% of the
    time. Averaged over 50 positions the two classes separate almost perfectly.

    That gap is the point. A run whose LLM has silently failed falls back to naive
    single-position features and cannot clear it.
    """
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    downstream_length = SEQUENCE_LENGTH - UPSTREAM_LENGTH

    sequences = []
    for label in labels:
        p_gc = gc_high if label == 1 else gc_low
        weights = [p_gc / 2, p_gc / 2, (1 - p_gc) / 2, (1 - p_gc) / 2]
        upstream = rng.choice(list("GCAT"), size=UPSTREAM_LENGTH, p=weights)
        downstream = rng.choice(list("ACGT"), size=downstream_length)
        sequences.append("".join(upstream) + "".join(downstream))

    X = pd.DataFrame({"raw_sequence": sequences})
    y = pd.DataFrame({"label": labels})
    return X, y


def best_single_position_accuracy(X: pd.DataFrame, y: pd.DataFrame) -> float:
    """Accuracy of the best `pos_N_is_X` rule, i.e. the naive-fallback ceiling on this data."""
    labels = y["label"].to_numpy()
    chars = np.array([list(s) for s in X["raw_sequence"]])
    best = 0.0
    for position in range(SEQUENCE_LENGTH):
        for nucleotide in "ACGT":
            hit = chars[:, position] == nucleotide
            accuracy = max((hit == labels).mean(), (~hit == labels).mean())
            best = max(best, accuracy)
    return float(best)



def fit_depth_one(config, X, y, tmp_path):
    """Fit one node with the shipped objects, wired exactly as `run_experiment` wires them.

    So these cover its healthy path against a live model; `tests/test_run_experiment.py` covers
    the same wiring offline, and `tests/test_integrity.py` the degraded paths.
    """
    engine = instantiate(config.llm)
    feature_finder = instantiate(config.feature_finder, llm_engine=engine)
    splitting_criterion = instantiate(config.splitting_criterion)
    tree = instantiate(
        config.tree,
        feature_finder=feature_finder,
        splitting_criterion=splitting_criterion,
        min_samples_leaf=max(1, int(0.05 * len(y))),
        folder_path=str(tmp_path),
    )
    selector = instantiate(config.selector)

    provenance = RunProvenance(policy=IntegrityPolicy.from_config(config.get("integrity")))
    engine.provenance = provenance
    feature_finder.provenance = provenance
    tree.provenance = provenance

    tree.fit(X, y, **config.params_generation, selector=selector)
    return tree, provenance


@pytest.mark.llm
def test_engine_returns_usable_completions(deft_config):
    """The narrowest check: credentials, endpoint, deployment and adapter all line up.

    Isolated from DEFT's logic so that a credential problem reports as a credential problem
    rather than as a mysterious tree-quality failure.
    """
    engine = instantiate(deft_config.llm)

    responses = asyncio.run(
        engine.query_llm(["Reply with exactly the four letters: ACGT"], 2)
    )

    assert len(responses) == 1
    assert responses[0] is not None, (
        "Engine returned None. The request failed every retry -- check the deployment name, "
        "endpoint and api_version in the SeqMaestro .env."
    )

    response = responses[0]
    choices = response["choices"]
    assert len(choices) >= 1, f"No completions returned: {response}"
    assert any(
        choice["message"]["content"].strip() for choice in choices
    ), "Every completion was empty."

    # Guards the token accounting that the budget cap is built on: `usage` must be present
    # and populated on this provider, or the cap silently counts nothing.
    usage = response["usage"]
    assert usage["total_tokens"] > 0, f"Provider reported no token usage: {usage}"
    assert usage["prompt_tokens"] > 0


@pytest.mark.llm
def test_root_split_uses_an_llm_generated_feature(deft_config, tmp_path):
    """A depth-1 fit must reach for an aggregate feature the naive population cannot express.

    The prompt is pointed at the synthetic task rather than at RNA polymerase pausing, so that
    this stays a test of the machinery rather than a test of whether the model guesses the
    right biology. Everything else is the shipped configuration.
    """
    X, y = make_planted_dataset()
    naive_ceiling = best_single_position_accuracy(X, y)
    assert naive_ceiling < 0.85, (
        f"Planted dataset is too easy for single-position rules ({naive_ceiling:.3f}); "
        "the assertion below would no longer distinguish a real feature from a fallback."
    )

    config = deft_config.copy()
    config.params_generation.target_name = "a synthetic binary label"
    config.params_generation.dataset_info = (
        "This is a synthetic benchmark dataset, not real biology. The label is determined "
        "entirely by the overall G+C nucleotide content of the upstream region, positions 0 "
        "to 49 inclusive. No single position is decisive; only an aggregate over the "
        "upstream window separates the classes."
    )

    tree, provenance = fit_depth_one(config, X, y, tmp_path)

    summary = provenance.summary()
    assert summary["request_success_rate"] == 1.0, f"some API requests failed: {summary}"
    assert summary["total_usable_llm_features"] > 0
    assert summary["nodes_with_no_usable_llm_feature"] == 0
    assert summary["llm_split_share"] == 1.0, (
        f"the split should have been won by a generated feature: {summary}"
    )

    root = tree.root
    assert root is not None, "fit produced no tree"
    assert root.feature is not None, "root was never assigned a feature"

    feature = root.feature
    assert feature.name, "root split has no feature name -- no split was found at all"
    assert feature.string, f"root feature `{feature.name}` carries no generated source"
    assert feature.threshold is not None
    assert callable(feature.fn)

    # The generated function must be well behaved on fresh data of the same shape.
    values = np.asarray(feature.fn(X)).astype(float)
    assert values.shape[0] == len(X)
    assert np.isfinite(values).all(), "feature function produced non-finite values"

    assert not NAIVE_FEATURE_NAME.match(feature.name), (
        f"Root split fell back to a naive single-position feature (`{feature.name}`). "
        "That is what DEFT does when the LLM path is broken but not raising."
    )

    predictions = np.asarray(tree.predict(X)).astype(int)
    accuracy = float((predictions == y["label"].to_numpy()).mean())
    assert accuracy >= 0.85, (
        f"Depth-1 training accuracy {accuracy:.3f} did not beat the naive single-position "
        f"ceiling of {naive_ceiling:.3f} by a clear margin, so the split is not using an "
        f"aggregate feature. Root feature was `{feature.name}`."
    )


@pytest.mark.llm
def test_a_generated_feature_can_use_a_covariate(deft_config, tmp_path):
    """DEFT is no longer sequence-only, and only a live model can show that end to end.

    The covariate decides the label outright here and the sequence is pure noise, so a tree
    that separates the classes has used the covariate. That the model *can* is not obvious: it
    learns which columns exist from `serialize_features` and the in-context rows and never from
    the prompt, so this covers the column-description path as well as the generation.
    """
    rng = np.random.default_rng(3)
    n = 120
    labels = np.tile([0, 1], n // 2)
    X = pd.DataFrame(
        {
            "raw_sequence": ["".join(rng.choice(list("ACGT"), size=20)) for _ in range(n)],
            "expression_level": np.where(labels == 1, 90.0, 10.0) + rng.normal(0, 2.0, n),
        }
    )
    y = pd.DataFrame({"label": labels})

    config = deft_config.copy()
    config.params_generation.target_name = "a synthetic binary label"
    config.params_generation.dataset_info = (
        "This is a synthetic benchmark dataset, not real biology. The label is determined "
        "entirely by the numeric column `expression_level`: high values are label 1 and low "
        "values are label 0. The DNA sequence is random noise and carries no signal at all."
    )
    # Naive features off, so the split can only come from the model.
    config.params_generation.initialize_with_naive_features = False

    tree, _ = fit_depth_one(config, X, y, tmp_path)

    feature = tree.root.feature
    assert feature is not None and feature.string, "no generated feature reached the split"
    assert "expression_level" in feature.string, (
        f"the generated feature ignored the covariate it was told decides the label: "
        f"{feature.string}"
    )

    accuracy = float((np.asarray(tree.predict(X)).astype(int) == labels).mean())
    assert accuracy >= 0.95, (
        f"a feature over `expression_level` should separate these classes almost perfectly; "
        f"got {accuracy:.3f} from `{feature.name}`"
    )


@pytest.mark.llm
def test_the_shipped_cli_path_runs_against_a_real_model(language_model, tmp_path):
    """`run_experiment` end to end with a real model, which the offline version cannot cover.

    The offline test scripts the engine, so it cannot catch what only real output exercises:
    code the guard's allowlist rejects, a response shape changed by a LiteLLM upgrade, or a
    completion count the provider silently capped. It reuses the same harness, with the engine
    left as configured.
    """
    from omegaconf import open_dict

    from deft.cli.run_deft_tree import run_experiment
    from tests.test_run_experiment import build_cfg

    frame = pd.DataFrame(
        {
            "raw_sequence": ["GCGCGCGCGC", "CGCGCGCGCG", "ATATATATAT", "TATATATATA"] * 8,
            "label": [1, 1, 0, 0] * 8,
        }
    )
    csv_path = tmp_path / "train.csv"
    frame.to_csv(csv_path, index=False)

    cfg = build_cfg(
        tmp_path,
        csv_path,
        engine=None,
        extra=[
            *language_model["overrides"],
            "params_generation.n_parents_per_prompt=2",
        ],
    )
    # Set through the config object rather than as overrides: Hydra's override grammar reads a
    # comma as a list separator, and these are prose.
    with open_dict(cfg):
        cfg.params_generation.target_name = "a synthetic binary label"
        cfg.params_generation.dataset_info = (
            "This is a synthetic benchmark dataset. The label is determined by the G+C content "
            "of the sequence: G/C-rich sequences are label 1 and A/T-rich sequences are label 0."
        )

    result = run_experiment(cfg)

    assert result["status"] == "fit_complete"
    for key in ("tree_path", "provenance_path", "token_usage_path"):
        assert Path(result[key]).is_file(), f"{key} was not written"

    assert result["integrity"]["request_success_rate"] == 1.0
    assert result["integrity"]["completion_yield"] == 1.0, (
        "the provider returned fewer completions than were asked for"
    )
    assert result["integrity"]["total_usable_llm_features"] > 0, (
        "no generated feature survived parsing, the code guard and scoring"
    )
    assert result["total_tokens"] > 0

    key = os.environ[language_model["key_variable"]]
    assert key.encode() not in Path(result["tree_path"]).read_bytes(), (
        "the real API key reached the saved tree"
    )
