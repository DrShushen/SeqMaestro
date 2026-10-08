"""The repair round: a generated candidate that fails on a node's rows goes back to the model.

A scripted engine stands in for the model and answers by the kind of prompt it is sent: a
proposal, the code for it, or a repair. Everything below the engine is the real code: the
prompts, the parsing, the guard, the screening and the provenance counters. All offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from deft.features.feature_finder import LLMFeatureFinder
from deft.provenance import IntegrityPolicy, RunProvenance

#: The shipped generation settings, so the repair wording tested is the wording a run sends.
PARAMS = OmegaConf.load(Path(__file__).resolve().parents[1] / "src" / "deft" / "conf"
                        / "params_generation" / "default.yaml")

PROPOSAL = json.dumps({"rationale": "G marks the class", "description": "G per T",
                       "name": "g_per_t"})
RAISES = "def g_per_t(X):\n    return X['raw_sequence'].str.count('G') / X['missing']"
FIXED = ("def g_per_t(X):\n    s = X['raw_sequence']\n"
         "    return s.str.count('G') / (s.str.count('T') + 1)")
CONSTANT = "def g_per_t(X):\n    return X['raw_sequence'].str.len() * 1.0"
G_COUNT = "def g_count(X):\n    return X['raw_sequence'].str.count('G').astype(float)"


def repair(code: str, name: str = "g_per_t", description: str = "") -> str:
    return json.dumps({"name": name, "description": description, "code": code})


class Scripted:
    """Answers a proposal prompt with one proposal, a code prompt with `code`, and each repair
    prompt with the next of `repairs`. Keeps every prompt it was sent."""

    def __init__(self, code: str, *repairs: str):
        self.code, self.repairs, self.prompts = code, list(repairs), []

    async def query_llm(self, list_prompts, n_generations_per_prompt):
        replies = []
        for prompt in list_prompts:
            self.prompts.append(prompt)
            if "failed when it was run" in prompt:
                content = self.repairs.pop(0)
            elif "python code" in prompt:
                content = self.code
            else:
                content = PROPOSAL
            replies.append({"choices": [{"message": {"content": content}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                      "total_tokens": 2}})
        return replies


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"raw_sequence": ["".join(rng.choice(list("ACGT"), 12))
                                          for _ in range(40)]})


def generate(engine, X, **settings):
    finder = LLMFeatureFinder(llm_engine=engine, random_state=0)
    finder.provenance = RunProvenance(policy=IntegrityPolicy(enabled=False))
    finder.provenance.start_node(depth=0, n_samples=len(X))
    features = finder.generate_features(
        X, prompts_first_step=["propose a feature"], context_code=PARAMS.context_code,
        instructions_code=PARAMS.instructions_code,
        instructions_code_repair=PARAMS.instructions_code_repair,
        repair_guidance=PARAMS.repair_guidance, **settings,
    )
    return features, finder.provenance.nodes[0]


def test_code_that_raises_goes_back_once_and_the_correction_is_kept(X):
    engine = Scripted(RAISES, repair(FIXED))
    [feature], node = generate(engine, X, repair_rounds=1)

    assert feature.string == FIXED and feature.repaired and feature.renamed_from is None
    assert feature.name == "g_per_t" and feature.description == "G per T", "kept, as it says"
    assert (node.repair_prompts, node.repaired, node.unrepaired) == (1, 1, 0)
    assert node.code_features_parsed == 0, "first attempts only"
    sent = engine.prompts[-1]
    assert RAISES in sent and "What went wrong: it raised KeyError" in sent
    assert PARAMS.repair_guidance.code_fault in sent
    assert PARAMS.repair_guidance.constant not in sent


def test_a_constant_candidate_comes_back_under_a_name_for_what_it_now_computes(X):
    engine = Scripted(CONSTANT, repair(G_COUNT, "g_count", "the number of G"))
    [feature], node = generate(engine, X, repair_rounds=1)

    assert (feature.name, feature.description) == ("g_count", "the number of G")
    assert feature.repaired and feature.renamed_from == "g_per_t"
    assert PARAMS.repair_guidance.constant in engine.prompts[-1]


def test_nothing_goes_back_when_repair_rounds_is_zero(X):
    engine = Scripted(RAISES, repair(FIXED))
    features, node = generate(engine, X, repair_rounds=0)
    assert features == [] and node.repair_prompts == 0
    assert not any("failed when it was run" in prompt for prompt in engine.prompts)


def test_a_correction_that_fails_again_goes_back_as_written_with_its_own_failure(X):
    engine = Scripted(RAISES, repair(CONSTANT), repair(FIXED))
    [feature], node = generate(engine, X, repair_rounds=2)

    assert feature.string == FIXED
    assert (node.repair_prompts, node.repaired, node.unrepaired) == (2, 1, 0)
    second = engine.prompts[-1]
    assert CONSTANT in second and RAISES not in second
    assert "the same value on every row" in second


def test_a_candidate_still_failing_when_the_rounds_run_out_is_counted(X):
    engine = Scripted(RAISES, repair(RAISES), "not json at all")
    features, node = generate(engine, X, repair_rounds=2)

    assert features == []
    assert (node.repair_prompts, node.repaired, node.unrepaired) == (2, 0, 1)


def test_a_new_name_without_a_description_is_not_kept(X):
    engine = Scripted(CONSTANT, repair(G_COUNT, "g_count"), repair(G_COUNT, "g_count", "G count"))
    [feature], node = generate(engine, X, repair_rounds=2)

    assert "with no description of what it now computes" in engine.prompts[-1]
    assert (feature.name, feature.renamed_from) == ("g_count", "g_per_t")


def test_a_repair_round_needs_its_wording():
    finder = LLMFeatureFinder(llm_engine=Scripted(RAISES), random_state=0)
    with pytest.raises(ValueError, match="instructions_code_repair"):
        finder._repair(pd.DataFrame({"raw_sequence": ["ACGT"]}), [(None, "raised", "raised")],
                       rounds=1, context="", instructions=None, guidance=None,
                       column_descriptions=None, max_undefined_share=0.0)
