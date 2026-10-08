"""Shared fixtures.

The LLM-backed tests call the provider the SeqMaestro root `.env` names, with its settings. They
skip cleanly when it is unfilled, so `pytest` is always safe to run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFT_ROOT = Path(__file__).resolve().parents[1]

#: Names the provider the tests call: a key of `PROVIDERS`.
PROVIDER_ENV = "SEQMAESTRO_LLM_PROVIDER"

#: For each provider SeqMaestro's `.env` can name: the variable holding its key, the others a
#: call needs, and the config overrides that select it.
PROVIDERS = {
    "azure": ("SEQMAESTRO_AZURE_OPENAI_KEY",
              ("SEQMAESTRO_AZURE_OPENAI_ENDPOINT", "SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT"),
              ("llm_models=seqmaestro_azure", "llm=gpt5")),
    "openai": ("SEQMAESTRO_OPENAI_KEY", ("SEQMAESTRO_OPENAI_MODEL",),
               ("llm_models=seqmaestro_openai", "llm=gpt5")),
    "claude": ("SEQMAESTRO_CLAUDE_KEY", ("SEQMAESTRO_CLAUDE_MODEL",),
               ("llm_models=seqmaestro_claude", "llm=claude")),
}


def _find_seqmaestro_root() -> Path | None:
    """Walk up from this file looking for the SeqMaestro repo root."""
    for candidate in [DEFT_ROOT, *DEFT_ROOT.parents]:
        if (candidate / "methods").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    return None


@pytest.fixture(scope="session", autouse=True)
def load_seqmaestro_env() -> None:
    """Load the SeqMaestro root `.env` without clobbering anything already exported."""
    root = _find_seqmaestro_root()
    if root is None:
        return
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - only when dev extras are absent
        return
    load_dotenv(env_file, override=False)


@pytest.fixture(scope="session")
def language_model(load_seqmaestro_env) -> dict:
    """The provider the environment names, as the variable holding its key and the config
    overrides that select it, or skip the test."""
    name = os.environ.get(PROVIDER_ENV, "").strip()
    if name not in PROVIDERS:
        pytest.skip(f"No language model configured; set {PROVIDER_ENV} to one of "
                    f"{', '.join(PROVIDERS)} in the SeqMaestro root .env to run the LLM tests.")
    key, further, overrides = PROVIDERS[name]
    missing = [variable for variable in (key, *further) if not os.environ.get(variable, "").strip()]
    if missing:
        pytest.skip(f"The {name} provider is not configured; set " + ", ".join(missing)
                    + " in the SeqMaestro root .env to run the LLM tests.")
    return {"key_variable": key, "overrides": list(overrides)}


@pytest.fixture(scope="session")
def deft_config(language_model):
    """Compose the real Hydra config, exactly as the CLI would, with test overrides.

    Composing the shipped `config.yaml` rather than hand-building objects means this test
    also covers config resolution: the defaults tree, the `${oc.env:...}` interpolations
    into the model's block, and every `_target_` on the DEFT path.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(DEFT_ROOT / "src" / "deft" / "conf"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                *language_model["overrides"],
                "max_depth=1",
                "params_generation.n_reflections=0",
                "params_generation.use_reflection=false",
                "params_generation.population_size=4",
                "params_generation.n_samples_per_prompt=4",
                "params_generation.n_parents_per_prompt=4",
            ],
        )
