"""A fitted tree must never carry credentials.

A fitted tree holds the feature finder, which holds the LLM engine, which holds the resolved
model config -- including the API key. So `dill.dump(tree)` wrote the key into the tree
artifact in plaintext, and tree artifacts are the primary output of a run: they get saved,
committed, attached to issues and shared.

`GenericEngine.__getstate__` redacts on the way out. These tests keep it that way.
"""

from __future__ import annotations

import dill
import pytest
from omegaconf import OmegaConf

from deft.features.feature_finder import LLMFeatureFinder
from deft.llm.engines.generic_engine import REDACTED, redact_engine_config
from deft.llm.engines.litellm_engine import LiteLLMEngine
from deft.trees.split_criterion import GiniCriterion
from deft.trees.tree import AdaptiveDecisionTree

FAKE_KEY = "sk-test-THIS-MUST-NEVER-BE-SERIALIZED-0123456789"

MODEL_CONFIG = {
    "model": "azure/some-deployment",
    "base_model": "azure/gpt-5.4",
    "api_base": "https://example.openai.azure.com/",
    "api_version": "2025-04-01-preview",
    "api_key": FAKE_KEY,
}


def make_engine(config):
    return LiteLLMEngine(content="system prompt", config=config)


@pytest.mark.parametrize("as_omegaconf", [False, True], ids=["dict", "DictConfig"])
def test_dilled_tree_does_not_contain_the_api_key(as_omegaconf):
    config = OmegaConf.create(MODEL_CONFIG) if as_omegaconf else dict(MODEL_CONFIG)
    tree = AdaptiveDecisionTree(
        feature_finder=LLMFeatureFinder(llm_engine=make_engine(config)),
        splitting_criterion=GiniCriterion(),
    )

    blob = dill.dumps(tree)

    assert FAKE_KEY.encode() not in blob, (
        "The API key was serialized into the tree artifact. Check "
        "GenericEngine.__getstate__."
    )


def test_the_live_engine_keeps_its_key():
    """Redaction happens on serialization only; the running engine must still work."""
    engine = make_engine(dict(MODEL_CONFIG))
    assert engine.config["api_key"] == FAKE_KEY

    dill.dumps(engine)  # serializing must not mutate the live object
    assert engine.config["api_key"] == FAKE_KEY


def test_round_tripped_engine_keeps_non_secret_provenance():
    """Redaction must not cost us the record of which model produced the tree."""
    restored = dill.loads(dill.dumps(make_engine(dict(MODEL_CONFIG))))

    assert restored.config["api_key"] == REDACTED
    assert restored.config["model"] == "azure/some-deployment"
    assert restored.config["api_version"] == "2025-04-01-preview"


@pytest.mark.parametrize(
    "secret_key", ["api_key", "API_KEY", "key", "token", "password", "secret"]
)
def test_secret_looking_keys_are_redacted_whatever_they_are_called(secret_key):
    redacted = redact_engine_config({secret_key: FAKE_KEY, "model": "m"})
    assert redacted[secret_key] == REDACTED
    assert redacted["model"] == "m"


def test_nested_secrets_are_redacted():
    """Engines accept both a flat model config and one nested under `llm_models`."""
    redacted = redact_engine_config({"llm_models": dict(MODEL_CONFIG), "other": [1, 2]})
    assert redacted["llm_models"]["api_key"] == REDACTED
    assert redacted["llm_models"]["model"] == "azure/some-deployment"


def test_unconvertible_config_is_dropped_rather_than_risked():
    class Hostile:
        def __iter__(self):
            raise RuntimeError("cannot be converted")

    assert redact_engine_config(Hostile()) is not None
    assert FAKE_KEY not in str(redact_engine_config(Hostile()))


def test_none_config_survives():
    assert redact_engine_config(None) is None
