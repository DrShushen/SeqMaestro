"""The record of the agents that carried out a run: each agent records itself, and a record that
the report could not place is refused.

The report names every agent a run records and the model it runs on, which only the agent can
state. A record made outside a run, or one naming a stage the process does not have, would name an
agent that the report cannot place."""

from __future__ import annotations

import json

import pytest

from seqmaestro.pipeline import agents
from seqmaestro.pipeline.layout import AGENTS_FILE, CONFIG_FILE


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / CONFIG_FILE).write_text("version: 4\n")
    return run


def test_each_agent_is_recorded_in_the_order_it_recorded_itself(run_dir):
    assert agents.recorded(run_dir) == []
    first = agents.record(run_dir, "Claude Code", "claude-opus-5-5", ["2", "1", "2"])
    agents.record(run_dir, " Claude Code ", "claude-sonnet-5-5", ["report"])
    found = agents.recorded(run_dir)
    assert [(entry["agent"], entry["model"], entry["stages"]) for entry in found] == [
        ("Claude Code", "claude-opus-5-5", ["1", "2"]),
        ("Claude Code", "claude-sonnet-5-5", ["report"])]
    assert first == found[0] and found[0]["recorded_utc"]
    assert json.loads((run_dir / AGENTS_FILE).read_text()) == found


@pytest.mark.parametrize(("agent", "model", "stages", "refusal"), [
    ("Claude Code", "claude-opus-5-5", ["4"], "as one of 1, 2, 3A, 3B, report, not 4"),
    ("Claude Code", "claude-opus-5-5", [], "not none"),
    ("", "claude-opus-5-5", ["1"], "Name the agent and the model"),
    ("Claude Code", " ", ["1"], "Name the agent and the model"),
])
def test_a_record_that_could_not_be_placed_is_refused(run_dir, agent, model, stages, refusal):
    with pytest.raises(ValueError, match=refusal):
        agents.record(run_dir, agent, model, stages)
    assert not (run_dir / AGENTS_FILE).exists()


def test_an_agent_records_itself_only_in_a_run_the_driver_started(tmp_path):
    """Otherwise a mistyped path would create a separate directory, which the report never reads."""
    with pytest.raises(ValueError, match="no run the driver started"):
        agents.record(tmp_path / "typo", "Claude Code", "claude-opus-5-5", ["1"])
    assert not (tmp_path / "typo").exists()


def test_the_command_records_the_agent_and_refuses_a_stage_the_process_does_not_have(run_dir, capsys):
    given = ["--run-dir", str(run_dir), "--agent", "Claude Code", "--model", "claude-opus-5-5"]
    assert agents.main([*given, "--stages", "1", "2"]) == 0
    assert "recorded Claude Code (claude-opus-5-5) for stage(s) 1, 2" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        agents.main([*given, "--stages", "stage2"])
    assert len(agents.recorded(run_dir)) == 1
