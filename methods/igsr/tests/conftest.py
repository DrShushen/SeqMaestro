import pytest


@pytest.fixture(autouse=True)
def configure_logging_fixture():
    """Configure logging for tests: DEBUG to the console, and no log file."""
    try:
        from igsr.logging_setup import configure_logging

        configure_logging(log_level="DEBUG", log_file_path=None)
    except Exception as e:  # pylint: disable=W0703
        # NOTE: Do not fail if this configuration fails.
        print(f"Error configuring logging: {e}")


@pytest.fixture
def live_llm_config() -> str:
    """The `llm` config of the provider SeqMaestro's root .env names, for a test that calls a model.
    The test is skipped where no provider is named."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    provider = os.environ.get("SEQMAESTRO_LLM_PROVIDER")
    if not provider:
        pytest.skip("SEQMAESTRO_LLM_PROVIDER is unset: SeqMaestro's root .env names the provider to call.")
    return f"llm=seqmaestro_{provider}"
