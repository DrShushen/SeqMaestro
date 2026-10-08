"""The proposer arms handed a fixed vocabulary beside the proposal: Stage 2 builds the pool, fits
its models on the proposal and the vocabulary together, and makes each arm a proposal of its own
draw, the arm handed the proposal alone included. An entry of a config's `arms` offers such an arm
to the pipeline, whose later stages take it as any other."""

from __future__ import annotations

import json

import pytest
import yaml

from seqmaestro.llm import ScriptedClient
from tests.planted import ANALYSES, LENGTH, MOTIF, SEQUENCE, planted_dataset

CODE = f"def motif_count(X):\n    return X['{SEQUENCE}'].str.count('{MOTIF}')"
REPLY = json.dumps([{"name": "motif_count", "description": "copies of the planted motif",
                     "rationale": "the planted class carries it", "code": CODE}])
RELIABILITY = {"pairs": 3, "permutations": 200, "cluster_rho": 0.8, "max_modal_share": 0.99,
               "planted_count": 3, "planted_signal_sd": 1.5, "seed": 0}


@pytest.fixture
def scripted(runners, monkeypatch, tmp_path):
    """The stages, with the language model scripted to propose one feature and every request it
    was sent kept in `calls`."""
    calls = {"n": 0, "prompts": []}

    def client():
        def answer(system, user):
            calls["n"] += 1
            calls["prompts"].append(user)
            return REPLY
        return ScriptedClient(answer)

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "planted.json").write_text(json.dumps({
        "target_name": "whether a window carries the planted motif",
        "task": "Predicting whether a window carries a planted motif.",
        "sequence_layout": f"{LENGTH} bases."}))
    monkeypatch.setattr(runners.arm, "LiteLLMClient", client)
    monkeypatch.setattr(runners.arm, "DESCRIPTIONS", datasets)
    runners.calls = calls
    return runners


def a_run(tmp_path, runners, arms: dict, seeds=(1,), labels=("real",), feedback_rounds=None,
          **sections):
    """A run directory holding the resolved config of a planted run on these arms. Returns it."""
    dataset = planted_dataset(tmp_path)
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory),
                                       "blinded": False},
        "grid": {"seeds": list(seeds), "labels": list(labels), "train_fraction": 0.8,
                 "n_bootstrap": 10, "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "sources": {
            **({"kmer": {"k_range": [1, 2], "normalise": "counts"}}
               if any("kmer" in pool for pool in arms) else {}),
            "llm_generated": {"proposal_rounds": 1, "n_per_round": 1, "n_examples": 20},
            **({} if feedback_rounds is None
               else {"llm_generated_iterative": {"feedback_rounds": feedback_rounds}})},
        "analyses": ANALYSES,
        "arms": arms, **sections}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(runners.config.load(source).to_yaml())
    return run_dir


def test_a_pool_with_the_vocabulary_fits_it_beside_a_proposal_made_for_that_arm(scripted, tmp_path):
    run_dir = a_run(tmp_path, scripted, {"llm_generated": {"models": ["l1_logistic"]},
                                         "positional+kmer+llm_generated": {"models": ["l1_logistic"]}})
    assert scripted.arm.main(["--run-dir", str(run_dir)]) == 0

    alone = run_dir / "results" / "l1_logistic" / "llm_generated" / "real" / "seed_1"
    beside = run_dir / "results" / "l1_logistic" / "positional+kmer+llm_generated" / "real" / "seed_1"
    run_alone = json.loads((alone / "run.json").read_text())
    run_beside = json.loads((beside / "run.json").read_text())
    assert run_beside["sources"] == ["positional", "kmer", "llm_generated"]
    assert (run_beside["model"], run_beside["pool"]) == ("l1_logistic", "positional+kmer+llm_generated")
    positions, kmers = 4 * LENGTH, 4 + 16
    assert run_alone["n_features_available"] == 1
    assert run_beside["n_features_available"] == positions + kmers + 1
    assert scripted.calls["n"] == 2, "each arm is made a proposal of its own draw"
    assert len(list((run_dir / "proposals").glob("*.json"))) == 2
    usage = json.loads((alone / "token_usage.json").read_text())
    assert usage["fit_report"]["n_proposed"] == 1, "what the proposal cost is kept with its cell"


def test_the_iterative_arms_first_round_makes_the_one_round_arms_request(scripted, tmp_path):
    """The two arms share no proposal, and the iterative arm's blind first round asks what the
    one-round arm asks, so one round of the iterative arm is the one-round arm's procedure."""
    run_dir = a_run(tmp_path, scripted, {"llm_generated": {"models": ["l1_logistic"]},
                                         "llm_generated_iterative": {"models": ["l1_logistic"]}},
                    feedback_rounds=1)
    flags = ["--run-dir", str(run_dir)]
    assert scripted.arm.main([*flags, "--arms", "l1_logistic/llm_generated"]) == 0
    one_round = list(scripted.calls["prompts"])
    assert scripted.arm.main([*flags, "--arms", "l1_logistic/llm_generated_iterative"]) == 0
    iterative = scripted.calls["prompts"][len(one_round):]
    assert len(one_round) == 1 and len(iterative) == 2
    assert iterative[0] == one_round[0], "the blind round asks the one-round arm's question"
    assert iterative[1] != one_round[0], "the informed round is shown the fit"
    assert len(list((run_dir / "proposals").glob("*.json"))) == 3


def test_a_stage_fits_only_what_the_runs_config_names(scripted, tmp_path, capsys):
    """An arm, a seed or a label condition the config does not name is refused, since a cell
    fitted under a setting the run does not record could not be told from the run's."""
    run_dir = a_run(tmp_path, scripted, {"llm_generated": {"models": ["l1_logistic"]}})
    flags = ["--run-dir", str(run_dir)]
    for narrowing, message in ((["--arms", "ebm/llm_generated"], "arms the run's config names"),
                               (["--seeds", "7"], "seeds of the run's grid"),
                               (["--labels", "permuted"], "label conditions of the run's grid")):
        assert scripted.arm.main([*flags, *narrowing]) == 2
        assert message in capsys.readouterr().err
    assert scripted.calls["n"] == 0 and not (run_dir / "results").exists()


def test_an_iterative_arm_over_the_kmers_is_reduced_profiled_revealed_and_reported(
        scripted, tmp_path):
    """From an entry of the config's `arms` to the report's inputs, with the language model
    scripted."""
    arm = "l1_logistic/kmer+llm_generated_iterative"
    run_dir = a_run(tmp_path, scripted, {"kmer+llm_generated_iterative": {"models": ["l1_logistic"]}},
                    seeds=(1, 2), labels=("real", "permuted"), feedback_rounds=1,
                    reduction={"n_features": 3}, reliability={**RELIABILITY, "check_arm": arm})
    flags = ["--run-dir", str(run_dir)]

    assert scripted.baseline.main(flags) == 0
    assert scripted.arm.main(flags) == 0
    assert scripted.reduction.main(flags) == 0
    assert scripted.reliability.main([*flags, "--workers", "2"]) == 0
    assert scripted.reveal.main(flags) == 0
    assert scripted.reliability.main([*flags, "--after-reveal"]) == 0
    assert scripted.report.main(flags) == 0

    facts = json.loads((run_dir / "report" / "facts.json").read_text())
    assert list(facts["stage3a"]["arms"]) == [arm]
    assert list(facts["stage3b"]["arms"]) == [arm]
    assert facts["stage2"]["arms"][arm]["label"] == "L1 Logistic Regression · motifs + written, iterative"
