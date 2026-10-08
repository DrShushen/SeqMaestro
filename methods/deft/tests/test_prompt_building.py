"""The prompts, as assembled (`src/deft/llm/prompt_builder.py`).

This layer had no tests, and it is where a defect is most expensive: a prompt fault is not a
crash, it is a worse tree. Two of the three things pinned here were real bugs found by running
a covariate dataset through it, not by reading.

`serialize_features` is covered in `test_dataset_loading.py`, next to the column handling it
describes. All offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.llm.prompt_builder import (
    construct_prompt_code_generation,
    construct_prompt_code_repair,
    construct_prompt_name_description,
    construct_prompt_reflection,
    serialize_history,
    serialize_in_context,
    serialize_list_features,
)
from deft.utils.feature import FeatureInfo
from deft.utils.selection import TopKSelector

HISTORY_START = "<Beginning Splitting conditions from root to current node>"
HISTORY_END = "<End of Splitting conditions from root to current node>"


@pytest.fixture
def X():
    return pd.DataFrame({"raw_sequence": ["ACGT", "TGCA", "GGCC"], "signal": [0.1, 0.5, 0.9]})


def a_feature(name="a_feature_name", score=0.5):
    return FeatureInfo(
        name=name, description=f"{name}_description", rationale=f"{name}_rationale",
        string=f"def {name}(X): ...", score=score, threshold=0.5,
    )


def history_section(prompt: str) -> str:
    return prompt[prompt.index(HISTORY_START) : prompt.index(HISTORY_END)]


SHARED = {
    "target_name": "t",
    "dataset_info": "d",
    "context_template": "{target_name}: {dataset_info}",
    "instructions": "INSTRUCTIONS",
    "include_ic_samples": False,
    "include_history": True,
    "include_feature_description": False,
    "n_ic_samples": 1,
}


def build_initial(X, **kwargs):
    return construct_prompt_name_description(
        X=X, rng=np.random.default_rng(0), **{**SHARED, **kwargs}
    )


def build_reflection(X, **kwargs):
    kwargs.setdefault("list_features", [a_feature()])
    kwargs.setdefault("n_parents_per_prompt", 1)
    return construct_prompt_reflection(
        X=X, rng=np.random.default_rng(0), selector=TopKSelector(), **{**SHARED, **kwargs}
    )


# ------------------------------------------------------------------ the history in both prompts


@pytest.mark.parametrize("build", [build_initial, build_reflection])
def test_the_split_history_describes_the_features_by_default(X, build):
    prompt = build(X, history=[(a_feature(), "smaller")], use_description_in_history=True)

    assert "a_feature_name_description" in history_section(prompt)


@pytest.mark.parametrize("build", [build_initial, build_reflection])
def test_use_description_in_history_is_honoured_by_both_prompts(X, build):
    """It used to stop at the initial prompt: the reflection builder passed two positional
    arguments to `serialize_history` and did not accept the third, so setting the config key
    to false silently changed one prompt of the two."""
    prompt = build(X, history=[(a_feature(), "smaller")], use_description_in_history=False)

    section = history_section(prompt)
    assert "a_feature_name_description" not in section
    assert "a_feature_name" in section, "the feature is still named, just not described"


@pytest.mark.parametrize("build", [build_initial, build_reflection])
def test_a_root_node_says_so(X, build):
    assert "root node" in build(X, history=[], use_description_in_history=True)


# --------------------------------------------------------------- the reflection population


def population():
    return [a_feature("middling", 0.5), a_feature("worst", 0.1), a_feature("best", 0.9)]


def order(rendered: str) -> list[str]:
    return [
        line.split("Feature name: ")[1].split(" \n")[0].strip()
        for line in rendered.split("\n")
        if "Feature name:" in line
    ]


def test_the_population_is_shown_worst_first_by_default():
    """Long-standing behaviour, kept so the pilot's prompts stay the paper's. It reads as a
    mistake next to every other sort in the codebase, which is why it is now a setting rather
    than a silent convention."""
    assert order(serialize_list_features(population(), sort_by_score=True)) == [
        "worst", "middling", "best"
    ]


def test_the_population_can_be_shown_best_first():
    rendered = serialize_list_features(population(), sort_by_score=True, sort_best_first=True)

    assert order(rendered) == ["best", "middling", "worst"]


def test_generation_order_is_kept_when_sorting_is_off():
    assert order(serialize_list_features(population(), sort_by_score=False)) == [
        "middling", "worst", "best"
    ]


def test_the_direction_reaches_the_reflection_prompt(X):
    prompt = build_reflection(
        X, history=[], list_features=population(), sort_by_score=True,
        sort_best_first=True, n_parents_per_prompt=3,
    )

    assert order(prompt) == ["best", "middling", "worst"]


def test_every_parent_carries_its_score(X):
    """The order matters less because the score is always shown; that must stay true."""
    rendered = serialize_list_features(population(), sort_by_score=True)

    assert rendered.count("Score:") == 3


@pytest.mark.parametrize("flag,needle", [("include_code", "def "), ("include_rationale", "_rationale")])
def test_code_and_rationale_are_opt_in(flag, needle):
    without = serialize_list_features(population(), **{flag: False})
    with_it = serialize_list_features(population(), **{flag: True})

    assert needle not in without
    assert needle in with_it


# ------------------------------------------------------------------------- in-context rows


def test_in_context_rows_show_every_column(X):
    rendered = serialize_in_context(X, n_samples=1, rng=np.random.default_rng(0))

    assert "raw_sequence:" in rendered
    assert "signal:" in rendered, "a covariate must be visible to the model too"
    assert f"has {len(X)} samples" in rendered


def test_asking_for_more_rows_than_exist_is_fine(X):
    rendered = serialize_in_context(X, n_samples=99, rng=np.random.default_rng(0))

    assert rendered.count("Sample ") == len(X)


def test_the_same_generator_state_gives_the_same_examples(X):
    """Reproducibility of a run rests on this — see `test_reproducibility.py`."""
    first = serialize_in_context(X, n_samples=2, rng=np.random.default_rng(7))
    second = serialize_in_context(X, n_samples=2, rng=np.random.default_rng(7))

    assert first == second


# ------------------------------------------------------------------------ code generation


def test_the_code_prompt_names_the_feature_it_must_implement(X):
    """This is why the model copies the proposed name into `def <name>(X):`."""
    prompt = construct_prompt_code_generation(
        X=X, context="CONTEXT", proposed_feature=a_feature(), instructions="INSTRUCTIONS"
    )

    assert "Feature name: a_feature_name" in prompt
    assert "a_feature_name_description" in prompt
    assert prompt.startswith("CONTEXT")
    assert prompt.endswith("INSTRUCTIONS")


def test_the_code_prompt_describes_the_available_columns(X):
    """The model learns column names from here, not from the instructions."""
    prompt = construct_prompt_code_generation(
        X=X, context="c", proposed_feature=a_feature(), instructions="i"
    )

    assert "raw_sequence" in prompt and "signal" in prompt


# ------------------------------------------------------------------------------- assembly


def test_the_optional_blocks_are_optional(X):
    minimal = build_initial(
        X, history=[], include_ic_samples=False, include_feature_description=False
    )
    full = build_initial(
        X, history=[], include_ic_samples=True, include_feature_description=True
    )

    assert "randomly sampled examples" not in minimal
    assert "randomly sampled examples" in full
    assert len(full) > len(minimal)


def test_the_task_context_is_interpolated(X):
    prompt = build_initial(X, history=[])

    assert prompt.startswith("t: d")


def test_history_is_suppressed_when_asked(X):
    prompt = serialize_history([(a_feature(), "smaller")], include_history=False)

    assert "a_feature_name" not in prompt


# ------------------------------------------------------------------ long values, and repairs


def test_a_long_text_value_is_shown_by_its_ends_with_its_length():
    X = pd.DataFrame({"raw_sequence": ["A" * 30 + "C" * 940 + "G" * 30]})
    shown = serialize_in_context(X, 1, rng=np.random.default_rng(0), max_chars=60)
    assert "A" * 30 + "..." + "G" * 30 + "  (1,000 characters, the middle not shown)" in shown
    whole = serialize_in_context(X, 1, rng=np.random.default_rng(0))
    assert "C" * 940 in whole, "without a cap every value is shown whole"


def test_a_repair_prompt_shows_the_feature_its_failed_code_and_how_to_correct_it(X):
    feature = a_feature("g_ratio")
    prompt = construct_prompt_code_repair(
        X, "CONTEXT", feature, "raised KeyError: 'missing'", "GUIDANCE", "INSTRUCTIONS",
        column_descriptions={"signal": "a measured signal"},
    )
    assert prompt.startswith("CONTEXT")
    assert "signal: " in prompt and ": a measured signal" in prompt
    assert "Feature name: g_ratio" in prompt and feature.string in prompt
    assert "What went wrong: it raised KeyError: 'missing'." in prompt
    assert prompt.index("GUIDANCE") < prompt.index("INSTRUCTIONS")

