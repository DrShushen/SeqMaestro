"""The run's status page (`seqmaestro/status/`): what it shows before, during and after a run, and
that building it never stops one."""

from __future__ import annotations

import json
import os
import subprocess

import yaml

from seqmaestro import status
from seqmaestro.harness import cell as cells
from seqmaestro.pipeline import config as experiment_config
from seqmaestro.pipeline.layout import FAILURES_DIR, LOGS_DIR, cell_dir
from seqmaestro.pipeline.schedule import JOBS_FILE
from seqmaestro.status import gather
from tests.planted import ANALYSES, LANGUAGE_MODEL, planted_dataset

ARMS = {"positional": {"models": ["l1_logistic", "ebm"]},
        "positional+kmer": {"models": ["l1_logistic"]}}


def a_run(tmp_path):
    """A run that the driver has just started: its recorded config, and nothing else."""
    dataset = planted_dataset(tmp_path)
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory),
                                       "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10, "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60,
                                        "guidance": None}},
        "sources": {"kmer": {"k_range": [1, 3], "normalise": "counts"}},
        "language_model": LANGUAGE_MODEL, "analyses": ANALYSES, "arms": ARMS,
        "reduction": {"n_features": 3},
    }, sort_keys=False))
    config = experiment_config.load(source)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(config.to_yaml())
    return run_dir, config


def a_cell(run_dir, model, pool, labels, seed, auroc, interval):
    path = cell_dir(run_dir, model, pool, labels, seed) / cells.METRICS_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"val": {"auroc": auroc, "ci": {"auroc": interval}}}))


def a_snapshot(run_dir, jobs, pid=None, finished=False):
    (run_dir / LOGS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / LOGS_DIR / JOBS_FILE).write_text(json.dumps({
        "updated_utc": "2026-11-22T10:00:00+00:00", "driver_pid": pid or os.getpid(),
        "finished": finished, "jobs": jobs}))


def a_job(call, stage, state):
    return {"log": call, "stage": stage, "phase": "before", "state": state,
            "started_utc": "2026-11-22T09:50:00+00:00", "attempts": 1, "threads": 1}


def test_each_process_is_named_as_the_report_names_what_it_works_on():
    assert gather.job_label(a_job("reduction__ebm__positional+kmer", "reduction", "running")) == (
        "Reducing EBM · positions + k-mers")
    assert gather.job_label(a_job("reliability__l1_logistic__positional", "reliability",
                                  "running")) == "The panel of L1 Logistic Regression · positions"
    assert gather.job_label(a_job("ebm__positional__permuted__seed_43", "ebm/positional",
                                  "running")) == "EBM · positions, shuffled labels, seed 43"
    assert gather.job_label(a_job("reveal", "reveal", "running")) == "The reveal"


def a_dead_pid():
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def arm_cells(gathered, model, pool):
    stage = next(stage for stage in gathered["stages"] if stage["key"] == "2")
    arms = [arm for row in stage["rows"] for arm in row["arms"] if arm]
    arm = next(arm for arm in arms if (arm["model"], arm["pool"]) == (model, pool))
    return {(cell["labels"], cell["seed"]): cell for cell in arm["cells"]}, stage


def test_a_run_not_yet_started_shows_its_dataset_and_its_grid(tmp_path):
    run_dir, _ = a_run(tmp_path)
    html = status.write_page(run_dir).read_text()
    assert "Not started" in html and "binding of the factor" not in html
    assert "0 of 12 cells done" in html, "three arms, two label conditions, two seeds"
    assert "Positions + k-mers" in html, "pools are named as the report names them"
    assert 'http-equiv="refresh"' not in html, "only a running run's page reloads itself"


def test_a_running_run_shows_what_runs_and_reloads_itself(tmp_path):
    run_dir, config = a_run(tmp_path)
    a_cell(run_dir, "l1_logistic", "positional", "real", 1, 0.8, [0.7, 0.9])
    running = "l1_logistic__positional__real__seed_2"
    a_snapshot(run_dir, [a_job(running, "l1_logistic/positional", "running")])
    gathered = gather.gather(run_dir, config)
    assert gathered["overall"]["state"]["kind"] == "running"
    assert [job["label"] for job in gathered["overall"]["running"]] == [
        "L1 Logistic Regression · positions, real labels, seed 2"]
    found, stage = arm_cells(gathered, "l1_logistic", "positional")
    assert found["real", 1]["state"] == "done" and found["real", 1]["score"] == 0.8
    assert found["real", 2]["state"] == "running" and found["permuted", 1]["state"] == "waiting"
    assert stage["state"] == "running"
    html = status.write_page(run_dir).read_text()
    assert 'http-equiv="refresh"' in html and running not in html, "a call's name is a file label"


def test_a_job_whose_driver_stopped_shows_as_stopped_and_a_failure_on_file_as_failed(tmp_path):
    run_dir, config = a_run(tmp_path)
    a_snapshot(run_dir, [a_job("ebm__positional__real__seed_1", "ebm/positional", "running")],
               pid=a_dead_pid())
    (run_dir / FAILURES_DIR).mkdir()
    (run_dir / FAILURES_DIR / "ebm__positional__real__seed_2.json").write_text("{}")
    gathered = gather.gather(run_dir, config)
    found, _ = arm_cells(gathered, "ebm", "positional")
    assert found["real", 1]["state"] == "stopped"
    assert found["real", 2]["state"] == "failed"
    assert gathered["overall"]["state"]["word"] == "Stopped, with failures"
    assert gathered["overall"]["running"] == []


def test_a_control_whose_interval_lies_above_chance_is_flagged_and_one_below_it_is_not(tmp_path):
    """A leak shows as a control above chance, whereas a model fitted to shuffled labels can fall
    below chance by picking a direction along one feature."""
    run_dir, config = a_run(tmp_path)
    a_cell(run_dir, "ebm", "positional", "permuted", 1, 0.62, [0.55, 0.7])
    a_cell(run_dir, "ebm", "positional", "permuted", 2, 0.33, [0.2, 0.45])
    found, stage = arm_cells(gather.gather(run_dir, config), "ebm", "positional")
    assert found["permuted", 1]["above_chance"] and not found["permuted", 2]["above_chance"]
    (controls,) = [item for item in stage["checkpoint"] if "chance" in item["text"]]
    assert controls == {"text": "1 control cell scores above chance", "state": "no"}


def test_the_updater_writes_the_snapshot_on_each_change_and_the_page_at_most_so_often(
        tmp_path, monkeypatch):
    run_dir, _ = a_run(tmp_path)
    pages, now = [], [0.0]
    monkeypatch.setattr(status, "write_page", lambda run: pages.append(now[0]))
    updater = status.Updater(run_dir, lambda text: None, clock=lambda: now[0])
    snapshot = run_dir / LOGS_DIR / JOBS_FILE

    updater([], True)
    assert pages == [0.0] and snapshot.exists()
    now[0] = status.MIN_SECONDS / 2
    updater([], True)
    assert pages == [0.0], "a change soon after the last page waits for the next"
    now[0] = status.QUIET_SECONDS / 2
    updater([], False)
    assert pages == [0.0]
    now[0] = status.QUIET_SECONDS + 1
    updater([], False)
    assert pages == [0.0, status.QUIET_SECONDS + 1], "a quiet run's page is still renewed"
    updater.finish([])
    assert len(pages) == 3 and json.loads(snapshot.read_text())["finished"]


def test_a_page_that_cannot_be_built_is_noted_once_and_stops_nothing(tmp_path, monkeypatch):
    run_dir, _ = a_run(tmp_path)

    def broken(run):
        raise ValueError("no page")

    monkeypatch.setattr(status, "write_page", broken)
    notes, now = [], [0.0]
    updater = status.Updater(run_dir, notes.append, clock=lambda: now[0])
    updater([], True)
    now[0] = status.QUIET_SECONDS + 1
    updater([], True)
    updater.finish([])
    assert len(notes) == 1 and "no page" in notes[0]
