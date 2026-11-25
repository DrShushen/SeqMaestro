"""The language model the command line calls itself: to draft the task description at Stage 1, and
to write the report.

It is the model `.env` names for the whole run (`seqmaestro.llm.client`), called with the shared
config's `language_model` settings, as every run's calls are. `load_credentials` reads `.env` into
the environment once, so the driver the command starts inherits the credentials too. A variable
already set in the shell keeps its value.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from seqmaestro.llm import CallSettings, LiteLLMClient, missing_settings
from seqmaestro.pipeline.config import resolve
from seqmaestro.pipeline.repository import REPO

#: The credentials, which the command reads and never prints.
CREDENTIALS = REPO / ".env"

#: The config whose `language_model` settings the command's own calls are made with.
SHARED_CONFIG = REPO / "experiments" / "configs" / "_shared.yaml"


def load_credentials(path: Path = CREDENTIALS) -> None:
    """Read `.env` into the environment, where it exists, without replacing a variable already
    set."""
    if path.exists():
        load_dotenv(path, override=False)


def missing() -> list[str]:
    """What the environment lacks for a call to be made, in words; empty when a call can be made."""
    return missing_settings()


def client() -> LiteLLMClient:
    """A client of the model the environment names, with the shared config's call settings."""
    return LiteLLMClient(CallSettings(**resolve(SHARED_CONFIG)["language_model"]))


def name(of: LiteLLMClient) -> str:
    """The model as a run records it, such as `azure/gpt-5.4`."""
    return str(of.describe()["model"])
