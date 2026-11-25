"""The time estimate a run is checked against before it starts (`seqmaestro/pipeline/estimate.py`):
its arithmetic on given timings, and real probes on the planted dataset.

A wrong estimate either hides a slow arm, which then takes a whole night, or flags a fast one,
which teaches whoever launches runs to ignore the flag.
"""

from __future__ import annotations

import json

import pytest

from tests.test_driver import a_config


@pytest.fixture
def runners():
    from seqmaestro.pipeline import config as experiment_config
    from seqmaestro.pipeline import driver as run_experiment
    from seqmaestro.pipeline import estimate
    return experiment_config, estimate, run_experiment


ARMS = {"positional": {"models": ["l1", "xgboost"]},
        "llm_generated_iterative": {"models": ["l1"]}}


def probe(estimate, model="l1_logistic", pool="positional", **given):
    return estimate.Probe(**{"model": model, "pool": pool, "width": 80, "used": 20,
                             "fit_seconds": 2.0, "featurise_seconds": 1.0, "probed_rows": 100,
                             "fitting_rows": 100, "language_model": False, **given})


def test_an_elimination_takes_the_reductions_steps(runners):
    from seqmaestro.reduction.eliminate import FINE_BELOW

    _, estimate, _ = runners
    widths = estimate.elimination_widths(100, 10)
    assert widths[:6] == [100, 75, 57, 43, 33, 25], "a quarter at a time above the fine range"
    assert widths[5:] == list(range(25, 9, -1)) and FINE_BELOW >= 25, "one at a time below it"
    assert estimate.elimination_widths(8, 10) == [8], "a pool already small enough is fitted once"


def test_an_arms_cost_follows_from_one_fit(runners, tmp_path):
    """Two label conditions and two seeds, an N of 3 for the reduction and three pairs for the
    panel, on four threads. The check arm runs three sweeps of the panel's selections, a black box
    none, and the iterative source fits once per round."""
    experiment_config, estimate, _ = runners
    config = a_config(tmp_path, experiment_config, arms=ARMS)

    checked = estimate.estimate_arm(probe(estimate), config, threads=4, panel_rows=150)
    assert checked.stage2 == pytest.approx(4 * (1.0 + 2.0))
    # From the 20 columns used down to 3, one at a time, and the top-N control: 2 s per 80 columns.
    assert checked.reduction == pytest.approx(2.0 * (sum(range(3, 21)) + 3) / 80)
    selection = 1.5 * (2.0 + checked.reduction)
    assert checked.panel == pytest.approx(2 * 3 * 3 * selection)
    assert checked.wall == pytest.approx(3.0 + checked.reduction + checked.panel / 4)

    boxed = estimate.estimate_arm(probe(estimate, model="xgboost"), config, threads=4, panel_rows=150)
    assert boxed.panel == 0 and boxed.reduction == checked.reduction

    rounds = estimate.estimate_arm(probe(estimate, pool="llm_generated_iterative"), config,
                                   threads=4, panel_rows=150)
    assert rounds.stage2 == pytest.approx(4 * (1.0 + 3 * 2.0)), "1 + feedback_rounds fits a cell"
    assert rounds.panel == pytest.approx(2 * 3 * selection), "one sweep off the check arm"


def test_a_pysr_search_costs_the_same_at_any_width_and_past_its_batches(runners):
    _, estimate, _ = runners
    searched = probe(estimate, model="pysr", fitting_rows=5000, probed_rows=1000, fit_seconds=60.0)
    assert estimate.fit_at(searched, 3, 5000) == 60.0
    assert estimate.fit_at(searched, 80, 500) == pytest.approx(30.0)
    assert estimate.fit_at(probe(estimate), 40, 50) == pytest.approx(0.5), "others by width and rows"


def test_a_fit_is_scaled_to_every_row_along_its_last_two_samples(runners):
    """The time is extended along the line through the last two fits, so the part of a fit's time
    that does not depend on its rows is not multiplied with them: fits of 10 s on 250 rows and 11 s
    on 500 give 25 s on 4,000, where scaling in proportion to the rows would give 88 s."""
    _, estimate, _ = runners
    assert estimate.scale_rows([(250, 10.0, 5), (500, 11.0, 5)], 4000) == pytest.approx(25.0)
    assert estimate.scale_rows([(250, 10.0, 5)], 1000) == pytest.approx(40.0), \
        "in proportion to the rows from one fit"
    assert estimate.scale_rows([(250, 10.0, 5), (500, 9.0, 5)], 4000) == 9.0, "never a falling line"
    assert estimate.scale_rows([(250, 1.0, 5), (500, 2.0, 5)], 500) == 2.0


def test_a_slow_arm_is_flagged_with_what_can_be_done(runners, tmp_path):
    experiment_config, estimate, _ = runners
    config = a_config(tmp_path, experiment_config, arms=ARMS)
    slow = estimate.estimate_arm(probe(estimate, fit_seconds=900.0), config, threads=4,
                                 panel_rows=150, slow_hours=1)
    quick = estimate.estimate_arm(probe(estimate), config, threads=4, panel_rows=150, slow_hours=1)
    assert slow.slow and not quick.slow
    text = estimate.table([slow, quick], config, 4, 1)
    assert "1 arm(s) over 1 h" in text and "SLOW" in text
    assert "not_profiled: {l1_logistic: <reason>}" in text and "reliability.pairs" in text


def test_the_written_sources_are_stood_in_for_at_a_proposals_width(runners, tmp_path):
    experiment_config, estimate, _ = runners
    config = a_config(tmp_path, experiment_config, arms=ARMS)
    assert estimate.stand_in_width("llm_generated", config) == 3 * 10
    assert estimate.stand_in_width("llm_generated_iterative", config) == (1 + 2) * 3 * 10
    assert estimate.stand_in_width("llm_generated_joint", config) == 0


def test_a_probe_fits_the_arms_model_on_its_pool(runners, tmp_path):
    """One real probe of a Core model, on enumerated columns and on stand-ins, and one of the tree
    on DEFT's side."""
    experiment_config, estimate, _ = runners
    config = a_config(tmp_path, experiment_config)
    rows = estimate.Rows(config)
    positional = estimate.probe_arm("l1_logistic", "positional", rows)
    assert positional.width == 20 * 4 and 0 < positional.used <= 80
    assert positional.fitting_rows == len(rows.fit_rows) == 128
    assert positional.fit_seconds > 0 and not positional.language_model

    written = estimate.probe_arm("l1_logistic", "llm_generated", rows)
    assert written.width == 30 and written.language_model and written.featurise_seconds == 0

    tree = estimate.probe_arm("deft_or_cart", "positional", rows)
    assert tree.width == 80, "DEFT's naive candidates: one per position and base"
    assert tree.probed_rows == 128 and tree.fit_seconds > 0 and tree.used > 0


def test_the_driver_estimates_and_starts_nothing(runners, tmp_path, monkeypatch):
    experiment_config, estimate, run_experiment = runners
    config = a_config(tmp_path, experiment_config, arms={"positional": {"models": ["l1"]}})
    source = tmp_path / "resolved.yaml"
    source.write_text(config.to_yaml())
    measured = probe(estimate)
    monkeypatch.setattr(estimate, "estimate_run", lambda config, threads, slow_hours, report: (
        [measured], [estimate.estimate_arm(measured, config, threads, 150, slow_hours)], {}))
    monkeypatch.setattr(run_experiment, "start_process",
                        lambda *args: pytest.fail("the estimate started a process"))

    run_dir = tmp_path / "run"
    assert run_experiment.main(["--config", str(source), "--estimate", "--slow-hours", "0.5",
                                "--run-dir", str(run_dir)]) == 0
    kept = json.loads((run_dir / "logs" / "estimate.json").read_text())
    assert kept["slow_hours"] == 0.5 and kept["arms"][0]["arm"] == "l1_logistic/positional"
    assert not (run_dir / "config.yaml").exists(), "no run was started"
    assert run_experiment.main(["--config", str(source), "--slow-hours", "1"]) == 2
