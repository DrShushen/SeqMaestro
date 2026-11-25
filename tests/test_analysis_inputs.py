"""An analysis that calls a language model makes its calls once.

The recurrence analysis and the judge keep what is on file while the features, the settings, the
task description and the language model they were computed from are unchanged, so that continuing a
run does not replace tables that were already checked. A cell fitted again, another reasoning
effort or `--force` makes them call the model again. A call that failed leaves nothing to keep, so
the next attempt makes it again.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import yaml

from seqmaestro.features.base import PROVENANCE_SELECTED, FeatureRecord
from seqmaestro.harness.features import save_features
from seqmaestro.judge import ScriptedEngine
from seqmaestro.llm import ScriptedClient
from seqmaestro.pipeline import config as configs
from seqmaestro.pipeline.analyses import inputs, judge, recurrence
from tests.planted import ANALYSES, LANGUAGE_MODEL, planted_dataset

ARM = ("l1_logistic", "llm_generated")
#: The run's settings for the analyses, with the counts reduced so that the analyses run quickly.
SMALL = {**ANALYSES, "judge": {**ANALYSES["judge"], "repeats": 1},
         "recurrence": {**ANALYSES["recurrence"], "draws": 50, "across_arms": False}}
IDEAS = ("gc_fraction", "cpg_count", "at_run", "purine_share")


def write_cell(run_dir, labels: str, seed: int, names, arm=ARM, kind: str = "llm_generated") -> None:
    """One cell's feature file. A one-round proposal is fixed before the fit, so its features are
    `selected`, as the columns of an enumerated source are."""
    cell = run_dir / "results" / arm[0] / arm[1] / labels / f"seed_{seed}"
    records = [FeatureRecord(name=name, description=f"what {name} measures", kind=kind,
                             importance=1.0 / (rank + 1), provenance=PROVENANCE_SELECTED)
               for rank, name in enumerate(names)]
    save_features(cell / "features.json", records, target_kind="binary")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run with its recorded config and the cells of one arm on the one-round source."""
    dataset = planted_dataset(tmp_path / "data")
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 0, "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": True, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60, "guidance": None}},
        "sources": {"llm_generated": {"proposal_rounds": 1, "n_per_round": 4, "n_examples": 20}},
        "language_model": LANGUAGE_MODEL,
        "analyses": SMALL,
        "arms": {"llm_generated": {"models": ["l1_logistic"]}}}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(configs.load(source).to_yaml())
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    (descriptions / "planted.json").write_text(json.dumps({"task": "Predicting a planted class."}))
    for module in (recurrence, judge):
        monkeypatch.setattr(module, "DESCRIPTIONS", descriptions)
    for labels in ("real", "permuted"):
        for seed in (1, 2):
            write_cell(run, labels, seed, IDEAS)
    return run


def restate(run_dir, section: str, part: str | None, key: str, value) -> None:
    """Change one setting of the run's recorded config, as a changed config would record it."""
    recorded = run_dir / "config.yaml"
    document = yaml.safe_load(recorded.read_text())
    (document[section][part] if part else document[section])[key] = value
    recorded.write_text(yaml.safe_dump(document, sort_keys=False))


def one_group(system: str, listing: str) -> str:
    """A grouping reply that puts every listed feature in one idea."""
    items = list(range(1, len(listing.splitlines()) + 1))
    return json.dumps({"groups": [{"name": "everything", "items": items}]})


def described(settings) -> dict:
    """What a stand-in reports about the model it replaces: one model, called with the run's
    `language_model` settings."""
    return {"model": "scripted", **asdict(settings)}


def test_the_recurrence_analysis_groups_once_and_again_only_when_something_changed(
        run_dir, monkeypatch, capsys):
    clients = []
    monkeypatch.setattr(recurrence, "LiteLLMClient", lambda settings: clients.append(
        ScriptedClient(one_group, described=described(settings))) or clients[-1])
    given = ["--run-dir", str(run_dir)]
    record = run_dir / "results" / recurrence.RECORD

    def asked() -> int:
        return sum(len(client.seen) for client in clients)

    assert recurrence.main(given) == 0
    assert asked() == 1
    first = record.read_text()
    assert inputs.on_file(record) is not None
    assert json.loads(first)["language_model"] == described(configs.load_recorded(run_dir).language_model)

    assert recurrence.main(given) == 0
    assert asked() == 1, "nothing changed, so nothing was asked"
    assert record.read_text() == first
    assert "the tables on file are kept" in capsys.readouterr().out

    assert recurrence.main([*given, "--force"]) == 0
    assert asked() == 2

    write_cell(run_dir, "real", 2, (*IDEAS[:3], "a_new_idea"))
    assert recurrence.main(given) == 0
    assert asked() == 3, "a cell fitted again reported other features"

    restate(run_dir, "analyses", "recurrence", "items_per_group", 3)
    assert recurrence.main(given) == 0
    assert asked() == 4, "a changed setting is another analysis"

    restate(run_dir, "language_model", None, "retries", 0)
    assert recurrence.main(given) == 0
    assert asked() == 4, "how failures are retried changes no answer"

    restate(run_dir, "language_model", None, "reasoning_effort", "high")
    assert recurrence.main(given) == 0
    assert asked() == 5, "a model asked to reason at another effort answers differently"


def test_the_arms_grouped_are_those_on_a_written_source(run_dir, monkeypatch):
    """Given no arms to group, the analysis groups the arm on the one-round source (whose features
    are `selected`) and leaves out the arm on an enumerated pool, whose feature names recur by
    construction."""
    for labels in ("real", "permuted"):
        for seed in (1, 2):
            write_cell(run_dir, labels, seed, ("pos_3_A", "pos_7_C"), arm=("l1_logistic", "positional"),
                       kind="positional")
    monkeypatch.setattr(recurrence, "LiteLLMClient", lambda settings: ScriptedClient(one_group))

    assert recurrence.main(["--run-dir", str(run_dir)]) == 0

    record = json.loads((run_dir / "results" / recurrence.RECORD).read_text())
    assert [arm["arm"] for arm in record["arms"]] == ["l1_logistic/llm_generated"]
    assert record["arms"][0]["n_grouped"] == record["arms"][0]["n_features"] == 16


def test_a_grouping_call_that_fails_fails_the_stage_and_is_made_again(run_dir, monkeypatch, capsys):
    replies = iter([ConnectionError("the endpoint is down"), None])
    monkeypatch.setattr(
        recurrence, "LiteLLMClient",
        lambda settings: ScriptedClient(lambda system, listing: next(replies) or one_group(system, listing)))
    given = ["--run-dir", str(run_dir)]
    record = run_dir / "results" / recurrence.RECORD

    assert recurrence.main(given) == 1, "the one arm's grouping failed"
    assert "the grouping call for l1_logistic/llm_generated failed" in capsys.readouterr().err
    assert not record.exists()

    assert recurrence.main(given) == 0
    assert inputs.on_file(record) is not None


def test_the_judge_rates_once_and_again_only_when_something_changed(run_dir, monkeypatch, capsys):
    engines = []
    monkeypatch.setattr(judge, "LiteLLMEngine", lambda settings: engines.append(
        ScriptedEngine('{"score": 3}', described=described(settings))) or engines[-1])
    given = ["--run-dir", str(run_dir)]
    summary = run_dir / "judge" / judge.SUMMARY

    def rated() -> int:
        """How many of the calls to the judge rated anything."""
        return sum(bool(engine.seen) for engine in engines)

    assert judge.main(given) == 0
    assert rated() == 1
    first = summary.read_text()
    assert inputs.on_file(summary) is not None
    assert json.loads(first)["language_model"] == described(configs.load_recorded(run_dir).language_model)

    assert judge.main(given) == 0
    assert rated() == 1 and summary.read_text() == first
    assert "the scores on file are kept" in capsys.readouterr().out

    assert judge.main([*given, "--force"]) == 0
    assert rated() == 2

    restate(run_dir, "analyses", "judge", "min_answered", 0.5)
    assert judge.main(given) == 0
    assert rated() == 2, "the share of answers a usable run needs changes no score"

    write_cell(run_dir, "real", 1, (*IDEAS[:3], "a_new_idea"))
    assert judge.main(given) == 0
    assert rated() == 3

    restate(run_dir, "language_model", None, "timeout_seconds", 600)
    assert judge.main(given) == 0
    assert rated() == 3, "how long a reply is waited for changes no score"

    restate(run_dir, "language_model", None, "reasoning_effort", "low")
    assert judge.main(given) == 0
    assert rated() == 4, "a judge asked to reason at another effort is another instrument"


def test_a_judge_that_answered_too_little_is_asked_again(run_dir, monkeypatch):
    replies = [ScriptedEngine("I cannot answer."), ScriptedEngine('{"score": 4}')]
    monkeypatch.setattr(judge, "LiteLLMEngine", lambda settings: replies.pop(0))
    given = ["--run-dir", str(run_dir)]
    summary = run_dir / "judge" / judge.SUMMARY

    assert judge.main(given) == 1
    assert summary.exists() and inputs.on_file(summary) is None, "kept as evidence, with no digest"
    assert judge.main(given) == 0
    assert not replies and inputs.on_file(summary) is not None


def test_the_judge_takes_the_direction_setting_from_the_runs_config(run_dir, monkeypatch):
    """Whether the judge is told which way a feature runs changes what a score means, so it is a
    setting of the run and has no command-line flag. A judge started by hand therefore rates as the
    driver's does."""
    monkeypatch.setattr(judge, "LiteLLMEngine", lambda settings: ScriptedEngine('{"score": 3}'))
    given = ["--run-dir", str(run_dir)]
    summary = run_dir / "judge" / judge.SUMMARY

    assert judge.main(given) == 0
    assert json.loads(summary.read_text())["settings"]["include_direction"] is False

    restate(run_dir, "grid", None, "judge_direction", True)
    assert judge.main(given) == 0, "another setting is another rating, so the judge is asked again"
    assert json.loads(summary.read_text())["settings"]["include_direction"] is True
    for flag in (["--include-direction"], ["--repeats", "2"], ["--top-n", "5"]):
        with pytest.raises(SystemExit):
            judge.main([*given, *flag])
