"""The estimate of a run's calls to the language model (`seqmaestro/pipeline/calls.py`): how many
each part makes, how two small calls time a reply, and what the load on the provider is."""

from __future__ import annotations

import yaml

from seqmaestro.llm import TaskContext
from seqmaestro.llm.client import Completion
from seqmaestro.pipeline import calls
from seqmaestro.pipeline.vendored import deft, igsr
from tests.planted import ANALYSES, LANGUAGE_MODEL, planted_dataset

SIZE = calls.Sizes(proposal=3000, instructions=1500, example=75.0, item=70)
SPEED = calls.Speed(latency=1.0, tokens_per_second=100.0, feature_tokens=300, cost=0.001)


def a_config(tmp_path, arms, models=None, judge=False):
    from seqmaestro.pipeline import config as experiment_config

    dataset = planted_dataset(tmp_path, continuous=any("igsr" in entry["models"]
                                                       for entry in arms.values()))
    source = tmp_path / "planted.yaml"
    held = {member for pool in arms for member in pool.split("+")}
    sources = {}
    if held & {"llm_generated", "llm_generated_iterative"}:
        sources["llm_generated"] = {"proposal_rounds": 3, "n_per_round": 10, "n_examples": 20}
    if "llm_generated_iterative" in held:
        sources["llm_generated_iterative"] = {"feedback_rounds": 4}
    grid = {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
            "n_bootstrap": 10, "permutation_scope": "seed", "permutation_repeats": 0,
            "judge": judge, "judge_direction": False,
            "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                   "max_example_length": 300, "max_seconds": 60, "guidance": None}}
    if not any("igsr" in entry["models"] for entry in arms.values()):
        grid["class_weight"] = "balanced"
    source.write_text(yaml.safe_dump({
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory),
                                       "blinded": False},
        "grid": grid, "sources": sources, "language_model": LANGUAGE_MODEL, "analyses": ANALYSES,
        "arms": arms, **({"models": models} if models else {}),
    }, sort_keys=False))
    return experiment_config.load(source)


def test_a_written_source_makes_its_rounds_and_at_most_one_repair_for_each(tmp_path):
    config = a_config(tmp_path, {"llm_generated": {"models": ["l1_logistic"]},
                                 "llm_generated_iterative": {"models": ["l1_logistic"]}})
    cells = 2 * 2
    one = calls.arm_part("l1_logistic", "llm_generated", config, SIZE, "azure")
    assert one.calls() == (cells * 3, cells * 3 * 2) and one.processes == cells
    iterative = calls.arm_part("l1_logistic", "llm_generated_iterative", config, SIZE, "azure")
    assert iterative.calls() == (cells * 5 * 3, cells * 5 * 3 * 2), "five proposals in each cell"
    assert calls.arm_part("l1_logistic", "positional", config, SIZE, "azure") is None


def test_deft_is_counted_at_its_probes_nodes_and_at_most_its_depth_allows(tmp_path):
    """At each node, 1 + 2R prompts for S candidates each, sent as ceil(S / per request) requests,
    and a request for each candidate's code, with a repair for each at the high end."""
    config = a_config(tmp_path, {"positional+llm_generated_joint": {"models": ["deft_or_cart"]}},
                      models={"deft_or_cart": {"config_overrides": [
                          "params_generation.n_reflections=5",
                          "params_generation.n_samples_per_prompt=10"]}})
    assert deft.call_settings(config, "azure") == {"reflections": 5, "candidates": 10,
                                                   "per_request": 8, "max_depth": 6}
    assert deft.call_settings(config, "claude")["per_request"] == 1
    assert deft.call_settings(config, None)["per_request"] == 1, "unknown: the most requests"
    part = calls.arm_part("deft_or_cart", "positional+llm_generated_joint", config, SIZE, "azure",
                          nodes=20)
    cells = 4
    assert part.calls() == (cells * 20 * (11 * 2 + 11 * 10),
                            cells * 63 * (11 * 2 + 11 * 10) + cells * 63 * 11 * 10)
    assert part.in_flight == 10, "the code of a prompt's ten candidates is asked for at once"
    claude = calls.arm_part("deft_or_cart", "positional+llm_generated_joint", config, SIZE,
                            "claude", nodes=20)
    assert claude.calls()[0] == cells * 20 * 220, "Claude returns one completion per request"


def test_igsr_runs_its_agent_for_each_successor_at_the_nodes_its_budget_allows(tmp_path):
    config = a_config(tmp_path, {"llm_generated_joint": {"models": ["igsr"]}},
                      models={"igsr": {"config_overrides": ["experiment.total_budget=10",
                                                            "experiment.n_successors=3"]}})
    assert igsr.call_settings(config) == {"budget": 10, "successors": 3}
    part = calls.arm_part("igsr", "llm_generated_joint", config, SIZE, "azure")
    assert part.calls() == (4 * 12, 4 * 54), "12 to 54 calls in each of the four cells"


def test_the_judge_and_the_recurrence_analysis_are_counted_where_they_run(tmp_path):
    arms = {"llm_generated": {"models": ["l1_logistic", "ebm"]}}
    assert [part.name for part in calls.analysis_parts(a_config(tmp_path, arms), SIZE)] == [
        "recurrence"]
    (tmp_path / "judged").mkdir()
    judge, recurrence = calls.analysis_parts(a_config(tmp_path / "judged", arms, judge=True), SIZE)
    rated = 2 * 2 * 1  # two arms, two seeds, the real labels alone
    assert judge.calls() == (rated * 20 * 3,) * 2
    assert recurrence.calls() == (3, 3), "one for each written arm, and one across them"


class Client:
    """A stand-in for the run's client: the first call, the feature, replies with 300 tokens after
    four seconds, and the second with one token after a second."""

    def __init__(self, clock):
        self.clock, self.prompt_tokens, self.completion_tokens = clock, 0, 0

    def complete(self, system, user, max_completion_tokens):
        first = self.completion_tokens == 0
        self.clock.now += 4.0 if first else 1.0
        self.prompt_tokens += 10
        self.completion_tokens += 300 if first else 1
        return Completion("{}" if first else "ready")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_two_small_calls_give_the_latency_the_speed_and_the_size_of_a_feature():
    clock = Clock()
    speed = calls.measure(Client(clock), "azure/gpt-5.4", clock=clock)
    assert speed.latency == 1.0 and speed.feature_tokens == 300
    assert speed.tokens_per_second == 100.0, "300 tokens in the three seconds after the latency"
    assert speed.cost is not None and speed.cost > 0, "LiteLLM's list prices the model"
    assert calls.price("no-such-model", 1000, 100) is None


def test_fewer_slots_spread_the_same_calls_over_a_longer_time(tmp_path):
    config = a_config(tmp_path, {"llm_generated": {"models": ["l1_logistic", "ebm"]}})
    parts = [calls.arm_part(model, "llm_generated", config, SIZE, "azure")
             for model in ("l1_logistic", "ebm")]
    one, many = calls.load(parts, SPEED, 1), calls.load(parts, SPEED, 8)
    assert one["in_flight"] == 1 and many["in_flight"] == 8
    assert one["hours"] > many["hours"]
    assert many["requests_per_minute"] > one["requests_per_minute"]


def test_without_credentials_the_calls_are_counted_and_nothing_is_measured(tmp_path, monkeypatch):
    config = a_config(tmp_path, {"llm_generated": {"models": ["l1_logistic"]},
                                 "positional": {"models": ["l1_logistic"]}})
    monkeypatch.setattr(calls, "missing_settings", lambda: ["missing SEQMAESTRO_LLM_PROVIDER"])
    monkeypatch.setattr(calls, "load_task", lambda directory, name: TaskContext(
        task="Predict binding.", sequence_layout="Sixty bases."))
    found = calls.estimate(config, slots=20)
    assert found.speed is None and "SEQMAESTRO_LLM_PROVIDER" in found.unmeasured
    part, recurrence = found.parts
    assert part.calls() == (12, 24) and part.sent()[0] > 0
    assert recurrence.name == "recurrence"
    sizes = found.sizes["llm_generated"]
    assert sizes.proposal > sizes.instructions > 0, "the examples add to the request"
    text = calls.table(found)
    assert "not estimated" in text and "l1_logistic/llm_generated" in text
    grid = calls.grid(config, {"l1_logistic/positional": 60.0, "l1_logistic/llm_generated": 120.0},
                      found)
    assert "| Written (`llm_generated`) | 2 min, 12 to 24 calls |" in grid
    assert "| Positions (`positional`) | 1 min |" in grid
    assert "L1 Logistic Regression (`l1_logistic`)" in grid


def test_deft_waits_for_its_requests_for_candidates_one_after_another(tmp_path):
    """A node's prompts for candidates go one after another, and their code at once, so a tree waits
    for each request for candidates and for one code request per prompt."""
    config = a_config(tmp_path, {"positional+llm_generated_joint": {"models": ["deft_or_cart"]}},
                      models={"deft_or_cart": {"config_overrides": [
                          "params_generation.n_reflections=5",
                          "params_generation.n_samples_per_prompt=10"]}})
    part = calls.arm_part("deft_or_cart", "positional+llm_generated_joint", config, SIZE, "azure",
                          nodes=20)
    semantic, code = SPEED.seconds(0.5 * 10 / 2 * 300), SPEED.seconds(0.5 * 300)
    cells, prompts = 4, 11
    assert part.waiting(SPEED)[0] == cells * 20 * prompts * (2 * semantic + code)


def test_a_call_is_priced_on_its_own_tokens():
    """A provider may price a call's tokens by its size, so a part is priced call by call."""
    part = calls.Part("arm", (calls.Kind(100, 100, 1000, features=1.0),), processes=1)
    found = calls.Estimate(parts=[part], slots=1, speed=SPEED, model="azure/gpt-5.4")
    assert found.cost(part)[0] == 100 * calls.price("azure/gpt-5.4", 1000, 300)


def test_a_measured_estimate_prices_each_part_and_shows_the_load(tmp_path):
    config = a_config(tmp_path, {"llm_generated": {"models": ["l1_logistic"]}})
    part = calls.arm_part("l1_logistic", "llm_generated", config, SIZE, "azure")
    found = calls.Estimate(parts=[part], slots=4, speed=SPEED, model="azure/gpt-5.4")
    low, high = found.cost(part)
    assert 0 < low < high
    text = calls.table(found)
    assert "All calls:" in text and "(this run)" in text and "cached" in text
    record = calls.as_record(found)
    assert record["parts"][0]["calls"] == (12, 24) and record["load"]
