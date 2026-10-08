"""Everything we control must vary because the seed varied, and for no other reason.

`random_state` used to reach exactly one consumer -- the dataset plugin's train/test split.
Three other places drew from unseeded global generators, so two runs with the same
`random_state` still differed:

    * the in-context examples shown in every prompt (`X.sample` with no `random_state`)
    * `TournamentSelector` (the global `random` module)
    * `RankSelector` (the global numpy RNG)

That matters for the pilot's stability axis, which is meant to measure run-to-run variation in
the *method*, not in unpinned plumbing.

What remains non-deterministic is the model's own sampling. See
`test_documented_limit_of_reproducibility` at the bottom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.features.feature_finder import LLMFeatureFinder
from deft.llm.prompt_builder import construct_prompt_name_description, serialize_in_context
from deft.utils.feature import FeatureInfo
from deft.utils.selection import RankSelector, TopKSelector, TournamentSelector


def make_frame(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"raw_sequence": ["".join(rng.choice(list("ACGT"), 12)) for _ in range(n)]}
    )


def population(n=10):
    return [FeatureInfo(name=f"f{i}", score=i / n) for i in range(n)]


# --------------------------------------------------------------------------------------------
# In-context examples
# --------------------------------------------------------------------------------------------


def test_same_seed_shows_the_same_in_context_examples():
    X = make_frame()

    first = serialize_in_context(X, 3, rng=np.random.default_rng(7))
    second = serialize_in_context(X, 3, rng=np.random.default_rng(7))

    assert first == second


def test_different_seeds_show_different_in_context_examples():
    X = make_frame()

    first = serialize_in_context(X, 3, rng=np.random.default_rng(7))
    second = serialize_in_context(X, 3, rng=np.random.default_rng(8))

    assert first != second, "the seed must actually change what the model is shown"


def test_successive_prompts_still_vary_within_a_run():
    """A shared generator, not a fixed integer: reproducible overall, varied prompt to prompt.

    Fixing the sample would show the model the same handful of sequences at every node, which
    would cost real diversity in what it proposes.
    """
    X = make_frame()
    rng = np.random.default_rng(7)

    draws = [serialize_in_context(X, 3, rng=rng) for _ in range(4)]

    assert len(set(draws)) > 1


def test_the_whole_sequence_of_prompts_is_reproducible():
    """The property that actually matters: rerun with the same seed, get the same prompts."""
    X = make_frame()

    def run(seed):
        rng = np.random.default_rng(seed)
        return [serialize_in_context(X, 3, rng=rng) for _ in range(5)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_unseeded_still_works():
    """`rng=None` must remain valid, for callers outside a configured run."""
    assert serialize_in_context(make_frame(), 3, rng=None)


# --------------------------------------------------------------------------------------------
# The feature finder wires it up
# --------------------------------------------------------------------------------------------


def test_two_finders_with_the_same_seed_build_the_same_prompt():
    X = make_frame()

    def build(seed):
        finder = LLMFeatureFinder(llm_engine=None, random_state=seed)
        return construct_prompt_name_description(
            X=X,
            target_name="t",
            dataset_info="d",
            history=[],
            context_template="{target_name} {dataset_info}",
            instructions="go",
            include_ic_samples=True,
            include_history=False,
            include_feature_description=False,
            n_ic_samples=3,
            rng=finder.rng,
        )

    assert build(11) == build(11)
    assert build(11) != build(12)


# --------------------------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("selector_cls", [TournamentSelector, RankSelector])
def test_selectors_are_reproducible_under_a_seed(selector_cls):
    names = lambda sel: [f.name for f in sel.select(population(), k=5)]  # noqa: E731

    assert names(selector_cls(random_state=3)) == names(selector_cls(random_state=3))


@pytest.mark.parametrize("selector_cls", [TournamentSelector, RankSelector])
def test_selectors_respond_to_the_seed(selector_cls):
    names = lambda sel: [f.name for f in sel.select(population(20), k=8)]  # noqa: E731

    assert names(selector_cls(random_state=3)) != names(selector_cls(random_state=4))


@pytest.mark.parametrize("selector_cls", [TournamentSelector, RankSelector])
def test_selectors_do_not_touch_the_global_generators(selector_cls):
    """They used to draw from `random` and `np.random`, so a selection perturbed everything else."""
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)

    selector_cls(random_state=3).select(population(), k=5)

    assert np.random.random() == before


def test_top_k_is_deterministic_without_any_seed():
    """The default selector. Included so a future change that adds randomness here is caught."""
    selected = [f.name for f in TopKSelector().select(population(), k=3)]

    assert selected == [f.name for f in TopKSelector().select(population(), k=3)]
    assert selected == ["f9", "f8", "f7"]


def test_tournament_handles_a_population_smaller_than_the_tournament():
    """Sampling without replacement from a short population must not raise."""
    selected = TournamentSelector(tournament_size=5, random_state=1).select(population(2), k=3)

    assert len(selected) == 3


# --------------------------------------------------------------------------------------------
# What is still not reproducible
# --------------------------------------------------------------------------------------------


def test_documented_limit_of_reproducibility():
    """No `seed` is sent to the provider, deliberately.

    The Azure GPT-5 deployment accepts a `seed` parameter but does not honour it: two calls with
    the same seed return different text, and no `system_fingerprint` is returned, which is the
    field a provider uses to promise determinism. Sending it would imply a reproducibility
    guarantee that does not exist, which is worse than not sending it.

    So a DEFT run is reproducible in everything under our control -- the split, the in-context
    examples, the selectors -- and not in the model's own sampling. This test exists to record
    that, and to fail if someone adds `seed` without also demonstrating it works.
    """
    from deft.llm.engines.litellm_engine import LiteLLMEngine

    engine = LiteLLMEngine(content="s", config={"model": "azure/d", "api_key": "k"})
    assert "seed" not in engine._build_request_kwargs("a prompt", 1)
