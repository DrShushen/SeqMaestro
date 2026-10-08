"""The driver (`seqmaestro/pipeline/driver.py`): what it runs and in what order around the reveal,
and the names and interpreters that every call it makes relies on.

The stages themselves are tested where they live, and the scheduler in `tests/test_schedule.py`.
These tests cover the driver's own logic, with every process it starts replaced so that nothing is
fitted.
"""

from __future__ import annotations

import json

import pytest
import yaml

from seqmaestro.dataset import load_dataset
from seqmaestro.pipeline import stage2
from seqmaestro.pipeline.repository import REPO
from seqmaestro.pipeline.vendored import deft, igsr
from seqmaestro.provenance import (
    CONDA_ROOT,
    environment_python,
    missing_python,
    python_override_variable,
)
from tests.planted import ANALYSES, planted_dataset

EVERY_ARM = {
    "positional": {"models": ["l1_logistic", "ebm", "xgboost", "deft_or_cart"]},
    "llm_generated": {"models": ["l1_logistic", "ebm"]},
    "llm_generated_iterative": {"models": ["l1_logistic"]},
    "positional+llm_generated_joint": {"models": ["deft_or_cart"]},
    "positional+kmer": {"models": ["l1_logistic"]},
}


def sources_for(arms) -> dict:
    """The `sources` a config with these arms states: the settings of each source a pool holds."""
    held = {member for pool in arms for member in pool.split("+")}
    sources = {}
    if "kmer" in held:
        sources["kmer"] = {"k_range": [1, 3], "normalise": "counts"}
    if held & {"llm_generated", "llm_generated_iterative"}:
        sources["llm_generated"] = {"proposal_rounds": 3, "n_per_round": 10, "n_examples": 20}
    if "llm_generated_iterative" in held:
        sources["llm_generated_iterative"] = {"feedback_rounds": 2}
    return sources


@pytest.fixture
def driver():
    from seqmaestro.pipeline import config as experiment_config
    from seqmaestro.pipeline import driver as run_experiment
    return experiment_config, run_experiment


def a_config(tmp_path, experiment_config, arms=None, blinded=False, sources=None):
    dataset = planted_dataset(tmp_path)
    source = tmp_path / "planted.yaml"
    arms = arms or EVERY_ARM
    source.write_text(yaml.safe_dump({
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": blinded},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10,
                 "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "sources": sources_for(arms) if sources is None else sources,
        "analyses": ANALYSES,
        "arms": arms,
        "reduction": {"n_features": 3},
        "reliability": {"pairs": 3, "permutations": 200, "cluster_rho": 0.8, "max_modal_share": 0.99,
                        "planted_count": 3, "planted_signal_sd": 1.5,
                        "check_arm": "l1/positional", "seed": 0},
    }, sort_keys=False))
    return experiment_config.load(source)


# --------------------------------------------------------------------------------------------
# The stages, and the reveal among them
# --------------------------------------------------------------------------------------------


def test_the_reveal_follows_everything_that_fits_and_precedes_what_reads_it(driver, tmp_path):
    experiment_config, run_experiment = driver
    stages = run_experiment.build_stages(a_config(tmp_path, experiment_config), tmp_path / "run")
    names = [stage.name for stage in stages]
    assert names[0] == "baseline"
    assert names[-4:] == ["reveal", "reliability_after_reveal", "report_inputs", "report_figures"]
    phases = [stage.phase for stage in stages]
    assert phases == sorted(phases, key=["before", "reveal", "after"].index), "one phase at a time"
    assert phases.count("reveal") == 1


def test_a_reduction_that_reduces_no_arm_is_no_stage(driver, tmp_path):
    """A config may leave every arm out of the reduction. A stage with no process would have no
    script for preflight to look for and nothing for the scheduler to report."""
    experiment_config, run_experiment = driver
    a_config(tmp_path, experiment_config)
    source = tmp_path / "planted.yaml"
    mapping = yaml.safe_load(source.read_text())
    del mapping["reliability"]
    mapping["arms"] = {"positional": {"models": ["xgboost"],
                                      "not_reduced": {"xgboost": "left out of this run"}}}
    mapping["sources"] = sources_for(mapping["arms"])
    source.write_text(yaml.safe_dump(mapping, sort_keys=False))
    config = experiment_config.load(source)

    stages = run_experiment.build_stages(config, tmp_path / "run")
    assert "reduction" not in [stage.name for stage in stages]
    assert all(stage.commands for stage in stages)


def test_the_judge_runs_only_where_the_grid_says_so(driver, tmp_path):
    """`grid.judge` is required, so a run's config says whether the judge was meant to run, and a
    run that leaves it out has no judge stage to fail or to ask for credentials."""
    import yaml

    experiment_config, run_experiment = driver
    without = run_experiment.build_stages(a_config(tmp_path, experiment_config), tmp_path / "run")
    assert "judge" not in [stage.name for stage in without]

    source = tmp_path / "planted.yaml"
    mapping = yaml.safe_load(source.read_text())
    mapping["grid"]["judge"] = True
    source.write_text(yaml.safe_dump(mapping))
    config = experiment_config.load(source)
    names = [stage.name for stage in run_experiment.build_stages(config, tmp_path / "run")]
    assert names.index("judge") == names.index("analyse_recurrence") + 1
    assert config.to_mapping()["grid"]["judge"] is True

    del mapping["grid"]["judge"]
    source.write_text(yaml.safe_dump(mapping))
    with pytest.raises(experiment_config.ConfigError, match="missing judge"):
        experiment_config.load(source)


def test_a_run_with_no_written_source_and_no_judge_calls_no_language_model(driver, tmp_path):
    """The recurrence analysis groups features a language model wrote, so it asks one only in a
    run with an arm on a written source. Without one, and without the judge, preflight asks for no
    credentials."""
    experiment_config, run_experiment = driver
    fixed = {"positional": {"models": ["l1_logistic", "xgboost"]}}
    config = a_config(tmp_path / "fixed", experiment_config, arms=fixed)
    calling = [stage.name for stage in run_experiment.build_stages(config, tmp_path / "run")
               if stage.calls_model]
    assert calling == []

    written = {**fixed, "llm_generated": {"models": ["l1_logistic"]}}
    config = a_config(tmp_path / "written", experiment_config, arms=written)
    calling = [stage.name for stage in run_experiment.build_stages(config, tmp_path / "run")
               if stage.calls_model]
    assert calling == ["l1_logistic/llm_generated", "analyse_recurrence"]


def test_every_arm_that_writes_takes_the_one_policy_for_failed_features(driver, tmp_path):
    """The proposer, DEFT's generative finder and IGSR each take `grid.generated_features` from the
    run's config, so all of them write under one policy per run."""
    experiment_config, _ = driver
    config = a_config(tmp_path, experiment_config)
    settings = stage2.proposal_settings(config)
    assert (settings.repair_rounds, settings.max_undefined_share, settings.max_example_length,
            settings.max_seconds) == (1, 0.2, 300, 60)
    assert (settings.n_rounds, settings.n_per_round, settings.n_examples) == (3, 10, 20)
    assert deft.stated(config, "positional+llm_generated_joint") == {
        "tree.class_weight": "balanced", "params_generation.repair_rounds": 1,
        "params_generation.max_undefined_share": 0.2,
        "params_generation.max_sequence_chars_in_prompt": 300}
    assert deft.stated(config, "positional") == {"tree.class_weight": "balanced"}, \
        "the CART finder writes nothing, so it is told nothing about writing"
    assert igsr.stated(config)["experiment.function_terms.timeout_seconds"] == 60
    assert config.to_mapping()["grid"]["generated_features"] == {
        "repair_rounds": 1, "max_undefined_share": 0.2, "max_example_length": 300, "max_seconds": 60}


def test_the_language_model_is_the_environments_and_no_config_names_one(driver, tmp_path, monkeypatch):
    """One provider serves every part of a run. An arm's stage passes it to the vendored method as
    a cell starts, a CART arm is told of none, and a config that names a model for one method is
    refused before anything starts."""
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    joint = "positional+llm_generated_joint"
    for name, value in {"SEQMAESTRO_LLM_PROVIDER": "claude", "SEQMAESTRO_CLAUDE_KEY": "a-key",
                        "SEQMAESTRO_CLAUDE_MODEL": "a-model"}.items():
        monkeypatch.setenv(name, value)
    assert deft.overrides(config, joint) == ["llm_models=seqmaestro_claude", "llm=claude",
                                             *deft.run_overrides(config, joint)]
    monkeypatch.setenv("SEQMAESTRO_LLM_PROVIDER", "openai")
    monkeypatch.delenv("SEQMAESTRO_OPENAI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing SEQMAESTRO_OPENAI_KEY"):
        deft.overrides(config, joint)
    assert deft.overrides(config, "positional") == deft.run_overrides(config, "positional")

    source = tmp_path / "planted.yaml"
    document = yaml.safe_load(source.read_text())
    for model, naming in (("deft_or_cart", "llm_models=seqmaestro_azure"), ("deft_or_cart", "llm=gpt5")):
        document["models"] = {model: {"config_overrides": [naming]}}
        source.write_text(yaml.safe_dump(document))
        with pytest.raises(experiment_config.ConfigError, match="SEQMAESTRO_LLM_PROVIDER"):
            run_experiment.build_stages(experiment_config.load(source), tmp_path / "run")


def test_preflight_asks_for_the_providers_settings_only_where_a_stage_calls_a_model(
        driver, tmp_path, monkeypatch):
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    for name in ("SEQMAESTRO_LLM_PROVIDER", "SEQMAESTRO_CLAUDE_KEY", "SEQMAESTRO_CLAUDE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    stages = run_experiment.build_stages(config, tmp_path / "run")
    calling = tuple(stage for stage in stages if stage.calls_model)
    quiet = tuple(stage for stage in stages if not stage.calls_model and stage.name == "baseline")
    assert calling and quiet

    def unset(selected):
        return [problem for problem in run_experiment.preflight(config, selected, tmp_path / "run")
                if "language model is not configured" in problem]

    assert unset(quiet) == []
    (problem,) = unset(calling)
    assert "missing SEQMAESTRO_LLM_PROVIDER (one of azure, openai, claude)" in problem
    monkeypatch.setenv("SEQMAESTRO_LLM_PROVIDER", "claude")
    (problem,) = unset(calling)
    assert "missing SEQMAESTRO_CLAUDE_KEY, SEQMAESTRO_CLAUDE_MODEL" in problem
    monkeypatch.setenv("SEQMAESTRO_CLAUDE_KEY", "a-key")
    monkeypatch.setenv("SEQMAESTRO_CLAUDE_MODEL", "a-model")
    assert unset(calling) == []


def test_a_vendored_models_settings_may_not_restate_what_the_grid_states(driver, tmp_path):
    """Two places that set one value can come to disagree, so the driver refuses the config before
    it starts anything."""
    experiment_config, run_experiment = driver
    a_config(tmp_path, experiment_config)
    source = tmp_path / "planted.yaml"
    document = yaml.safe_load(source.read_text())
    document["models"] = {"deft_or_cart": {"config_overrides": ["tree.max_depth=3"]}}
    source.write_text(yaml.safe_dump(document))
    config = experiment_config.load(source)
    assert deft.overrides(config, "positional") == ["tree.class_weight=balanced", "tree.max_depth=3"]
    run_experiment.build_stages(config, tmp_path / "run")

    document["models"]["deft_or_cart"]["config_overrides"].append("tree.class_weight=none")
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(experiment_config.ConfigError, match="which the grid already states"):
        run_experiment.build_stages(experiment_config.load(source), tmp_path / "run")


@pytest.mark.parametrize(("change", "message"), [
    (None, "missing generated_features"),
    ({"repair_rounds": 1, "max_undefined_share": 1.5, "max_example_length": 300, "max_seconds": 60},
     r"in \[0, 1\]"),
    ({"repair_rounds": -1, "max_undefined_share": 0.2, "max_example_length": 300, "max_seconds": 60},
     ">= 0"),
    ({"repair_rounds": 1, "max_undefined_share": 0.2, "max_seconds": 60}, "missing max_example_length"),
    ({"repair_rounds": 1, "max_undefined_share": 0.2, "max_example_length": 300}, "missing max_seconds"),
    ({"repair_rounds": 1, "max_undefined_share": 0.2, "max_example_length": 300, "max_seconds": 0},
     ">= 1"),
])
def test_the_policy_for_failed_features_is_stated_in_full(driver, tmp_path, change, message):
    import yaml

    experiment_config, _ = driver
    a_config(tmp_path, experiment_config)
    source = tmp_path / "planted.yaml"
    document = yaml.safe_load(source.read_text())
    if change is None:
        del document["grid"]["generated_features"]
    else:
        document["grid"]["generated_features"] = change
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(experiment_config.ConfigError, match=message):
        experiment_config.load(source)




def test_a_stage_is_told_which_part_of_the_run_it_fits_and_reads_its_settings_from_the_run(
        driver, tmp_path):
    """The driver passes a stage the run directory and its arm, labels and seed. How each source is
    computed is stated once, in `sources`, and every arm on a pool holding it takes it from the
    run's config: the k-mers' counting, the proposer's settings and an iterative pool's rounds."""
    experiment_config, run_experiment = driver
    arms = {"positional": {"models": ["l1"]},
            "kmer+llm_generated_iterative": {"models": ["l1", "xgboost"]},
            "positional+kmer": {"models": ["deft_or_cart"]}}
    sources = {**sources_for(arms), "kmer": {"k_range": [1, 2], "normalise": "per_length"}}
    config = a_config(tmp_path, experiment_config, arms=arms, sources=sources)
    assert config.arm_cells() == [("l1_logistic", "positional"),
                                  ("l1_logistic", "kmer+llm_generated_iterative"),
                                  ("xgboost", "kmer+llm_generated_iterative"),
                                  ("deft_or_cart", "positional+kmer")]
    run_dir = tmp_path / "run"
    stages = {stage.name: stage for stage in run_experiment.build_stages(config, run_dir)}
    stage = stages["xgboost/kmer+llm_generated_iterative"]
    assert len(stage.commands) == 2 * 2, "one process per labels and seed"
    assert stage.commands[0].argv[1:] == (
        "experiments/runners/run_arm.py", "--run-dir", str(run_dir), "--arms",
        "xgboost/kmer+llm_generated_iterative", "--labels", "real", "--seeds", "1")
    for script in {command.argv[1] for stage in stages.values() for command in stage.commands}:
        assert (REPO / script).exists(), script

    dataset = load_dataset(config.dataset.directory)
    described = stage2.pool_featuriser(config, dataset, "positional+kmer").describe()
    assert {"k_range": [1, 2], "normalise": "per_length"}.items() <= {
        key: list(value) if isinstance(value, tuple) else value
        for part in described["parts"] for key, value in part.items()}.items()
    assert config.sources.feedback_rounds == 2
    assert config.to_mapping()["arms"]["positional"] == {"models": ["l1_logistic"]}, \
        "`l1` is recorded as the model the target called for"


@pytest.mark.parametrize(("arms", "message"), [
    ({"kmer+positional": {"models": ["l1"]}}, "out of order. Write it `positional\\+kmer`"),
    ({"llm_generated+llm_generated_iterative": {"models": ["l1"]}}, "at most one written source"),
    ({"llm_generated+motifs": {"models": ["l1"]}}, "not a pool"),
    ({"positional+covariates": {"models": ["l1"]}}, "declares none"),
    ({"llm_generated_joint": {"models": ["l1"]}}, "cannot be handed llm_generated_joint"),
    ({"llm_generated_joint": {"models": ["deft_or_cart"]}}, "needs positional in its pool"),
    ({"llm_generated_joint": {"models": ["igsr"]}}, "fits a continuous target"),
    ({"positional+kmer+llm_generated_joint": {"models": ["deft_or_cart"]}},
     "cannot be handed kmer\\+llm_generated_joint together yet"),
    ({"positional": {"models": ["l1", "l1_logistic"]}}, "names l1_logistic twice"),
    ({"positional": {"models": ["cart"]}}, "Unknown model 'cart'"),
])
def test_an_arm_no_model_can_run_is_refused_when_the_config_is_read(driver, tmp_path, arms, message):
    experiment_config, _ = driver
    with pytest.raises(experiment_config.ConfigError, match=message):
        a_config(tmp_path, experiment_config, arms={"positional": {"models": ["l1"]}, **arms})


@pytest.mark.parametrize(("sources", "message"), [
    ({}, "needs kmer, llm_generated"),
    ({"kmer": {"k_range": [1, 3], "normalise": "counts"},
      "llm_generated": {"proposal_rounds": 3, "n_per_round": 10, "n_examples": 20},
      "llm_generated_iterative": {"feedback_rounds": 2}}, "no pool in `arms` uses"),
    ({"kmer": {"k_range": [3, 1], "normalise": "counts"},
      "llm_generated": {"proposal_rounds": 3, "n_per_round": 10, "n_examples": 20}}, "1 <= min <= max"),
    ({"kmer": {"k_range": [1, 3], "normalise": "counts"},
      "llm_generated": {"proposal_rounds": 3, "n_per_round": 10}}, "missing n_examples"),
])
def test_a_source_some_pool_holds_is_stated_in_full_and_no_other(driver, tmp_path, sources, message):
    experiment_config, _ = driver
    with pytest.raises(experiment_config.ConfigError, match=message):
        a_config(tmp_path, experiment_config, sources=sources,
                 arms={"positional": {"models": ["l1"]}, "kmer+llm_generated": {"models": ["l1"]}})


def test_an_entry_leaves_its_models_out_of_stage_3_only_with_a_reason(driver, tmp_path):
    """Every arm is reduced and profiled unless its entry says otherwise, with the reason the run
    records. A black-box model is left out of the panel by its kind."""
    experiment_config, _ = driver
    config = a_config(tmp_path, experiment_config, arms={
        "positional": {"models": ["l1", "ebm", "xgboost"],
                       "not_reduced": {"ebm": "too slow here"}},
        "positional+kmer": {"models": ["l1"], "not_profiled": {"l1": "checked elsewhere"}}})
    assert ("ebm", "positional") not in config.reduced_arms()
    assert config.profiled_arms() == [("l1_logistic", "positional")]
    assert config.not_reduced() == {"ebm/positional": "too slow here"}
    assert config.not_profiled() == {
        "ebm/positional": "not reduced: too slow here",
        "xgboost/positional": "its model is read through its predictions, not its features",
        "l1_logistic/positional+kmer": "checked elsewhere"}
    for case, (entry, message) in enumerate((
            ({"not_profiled": {"xgboost": "a reason"}}, "leaves out anyway"),
            ({"not_profiled": {"ebm": "a reason"}}, "does not list"),
            ({"not_reduced": {"l1": ""}}, "needs a reason"),
            ({"not_reduced": {"l1": "a"}, "not_profiled": {"l1": "b"}}, "under `not_reduced` only"))):
        (tmp_path / str(case)).mkdir()
        with pytest.raises(experiment_config.ConfigError, match=message):
            a_config(tmp_path / str(case), experiment_config, arms={
                "positional": {"models": ["l1", "xgboost"], **entry}})


def test_the_share_of_rows_a_seed_fits_on_is_stated_and_below_one(driver, tmp_path):
    """No stage has a value of its own for it, so a config without one is refused, and so is one
    under which every seed would fit on the same rows."""
    experiment_config, _ = driver
    config = a_config(tmp_path, experiment_config)
    assert config.grid.train_fraction == 0.8
    assert config.to_mapping()["grid"]["train_fraction"] == 0.8
    assert stage2.cell_settings(config, "permuted", 2) == {
        "seed": 2, "train_fraction": 0.8, "permute_labels": True, "permutation_scope": "seed",
        "n_bootstrap": 10, "permutation_repeats": 0}
    source = tmp_path / "planted.yaml"
    document = yaml.safe_load(source.read_text())
    for value, message in ((None, "missing train_fraction"), (1, "must be below 1"),
                           (0, r"in \(0, 1\]"), ("most", r"in \(0, 1\]")):
        if value is None:
            del document["grid"]["train_fraction"]
        else:
            document["grid"]["train_fraction"] = value
        source.write_text(yaml.safe_dump(document))
        with pytest.raises(experiment_config.ConfigError, match=message):
            experiment_config.load(source)


def test_how_the_analyses_take_the_cells_is_stated_in_full(driver, tmp_path):
    """The cut an analysis takes, the judge's ratings and the grouping's settings decide what a
    table of the run holds, so a config states each and a run records them."""
    experiment_config, _ = driver
    config = a_config(tmp_path, experiment_config)
    assert config.analyses.top_n == 20 and config.analyses.judge.repeats == 3
    assert config.analyses.recurrence.kinds == ("llm_generated",)
    assert config.to_mapping()["analyses"] == ANALYSES
    source = tmp_path / "planted.yaml"
    stated = yaml.safe_load(source.read_text())
    for change, message in (
            (lambda d: d.pop("analyses"), "missing analyses"),
            (lambda d: d["analyses"].pop("top_n"), "`analyses` is missing top_n"),
            (lambda d: d["analyses"]["judge"].pop("repeats"), "`analyses.judge` is missing repeats"),
            (lambda d: d["analyses"]["judge"].update(labels=["real", "shuffled"]), "at most once"),
            (lambda d: d["analyses"]["recurrence"].update(draws=0), "integer >= 1"),
            (lambda d: d["analyses"]["recurrence"].update(passes=2), "has no key passes")):
        document = yaml.safe_load(yaml.safe_dump(stated))
        change(document)
        source.write_text(yaml.safe_dump(document))
        with pytest.raises(experiment_config.ConfigError, match=message):
            experiment_config.load(source)


def test_a_recorded_config_states_its_version_and_a_stage_reads_no_other(driver, tmp_path):
    """A stage takes its settings from the config the driver recorded, whose shape the version
    names. A file without one is no record of a run, and one of another version is refused."""
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    journal = run_experiment.Journal(run_dir / "logs")
    recorded = run_dir / "config.yaml"

    run_experiment.bind_config(run_dir, config, allow_change=False, journal=journal)
    document = yaml.safe_load(recorded.read_text())
    assert document["version"] == experiment_config.CONFIG_VERSION
    assert experiment_config.load_recorded(run_dir).to_yaml() == config.to_yaml()
    run_experiment.bind_config(run_dir, config, allow_change=False, journal=journal)

    document["grid"]["seeds"] = [1, 2, 3]
    recorded.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(experiment_config.ConfigError, match="made under a different config"):
        run_experiment.bind_config(run_dir, config, allow_change=False, journal=journal)
    run_experiment.bind_config(run_dir, config, allow_change=True, journal=journal)
    assert recorded.read_text() == config.to_yaml()
    assert yaml.safe_load((run_dir / "config_superseded_1.yaml").read_text())["grid"]["seeds"] == [1, 2, 3]

    document["version"] = experiment_config.CONFIG_VERSION + 1
    recorded.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(experiment_config.ConfigError, match="reads version"):
        experiment_config.load_recorded(run_dir)
    with pytest.raises(experiment_config.ConfigError, match="cannot be continued"):
        run_experiment.bind_config(run_dir, config, allow_change=True, journal=journal)

    del document["version"]
    recorded.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(experiment_config.ConfigError, match="states no `version`"):
        experiment_config.load_recorded(run_dir)
    with pytest.raises(experiment_config.ConfigError, match="cannot be continued"):
        run_experiment.bind_config(run_dir, config, allow_change=True, journal=journal)


def test_a_call_is_named_for_its_stage_its_arms_and_what_else_tells_it_apart():
    from seqmaestro.pipeline.layout import call_name

    assert call_name("reduction") == "reduction"
    assert call_name("reduction", [("ebm", "positional+kmer")]) == "reduction__ebm__positional+kmer"
    assert call_name("reliability", [("ebm", "kmer"), ("l1_logistic", "kmer")]) == \
        "reliability__ebm__kmer-l1_logistic__kmer"
    assert stage2.cell_call("deft_or_cart", "positional", "real", 42) == \
        "deft_or_cart__positional__real__seed_42"


def test_a_config_states_whether_its_run_is_blinded(driver, tmp_path):
    experiment_config, _ = driver
    a_config(tmp_path, experiment_config)
    source = tmp_path / "planted.yaml"
    document = yaml.safe_load(source.read_text())
    assert experiment_config.load(source).to_mapping()["dataset"]["blinded"] is False
    del document["dataset"]["blinded"]
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(experiment_config.ConfigError, match="missing blinded"):
        experiment_config.load(source)
    document["dataset"]["blinded"] = "yes"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(experiment_config.ConfigError, match="blinded"):
        experiment_config.load(source)


#: Withholds what the planted target column is called, among other words.
NAMES_THE_TARGET = "The outcome column marks the windows that carry the planted motif."


@pytest.mark.parametrize(("blinded", "judge_only", "refusal"), [
    (True, None, "withholds nothing"),
    (False, NAMES_THE_TARGET, "withholds a fact from the models"),
    (True, NAMES_THE_TARGET, "the column name 'outcome' holds the withheld word 'outcome'"),
    (True, "The planted motif is GGCGCC.", None),
    (False, None, None),
], ids=["blinded withholding nothing", "not blinded withholding a fact", "a leak", "clean",
        "not blinded"])
def test_the_preflight_refuses_a_blinding_that_cannot_stand(driver, tmp_path, monkeypatch, blinded,
                                                             judge_only, refusal):
    """A property of the whole run, so checked whatever stages are selected."""
    experiment_config, run_experiment = driver
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "planted.json").write_text(json.dumps(
        {"task": "Predicting a class from random windows.",
         **({"judge_only": judge_only} if judge_only else {})}))
    monkeypatch.setattr(run_experiment, "DESCRIPTIONS", datasets)
    config = a_config(tmp_path, experiment_config, arms={"positional": {"models": ["l1_logistic"]}},
                      blinded=blinded)
    stages = [stage for stage in run_experiment.build_stages(config, tmp_path / "run")
              if stage.name == "l1_logistic/positional"]
    problems = [problem for problem in run_experiment.preflight(config, tuple(stages), tmp_path / "run")
                if "blinded" in problem]
    if refusal is None:
        assert problems == []
    else:
        assert len(problems) == 1 and refusal in problems[0]

def test_the_figures_are_drawn_where_the_plotting_stack_is(driver, tmp_path):
    experiment_config, run_experiment = driver
    stages = run_experiment.build_stages(a_config(tmp_path, experiment_config), tmp_path / "run")
    figures = next(stage for stage in stages if stage.name == "report_figures")
    assert figures.commands[0].argv[0] == str(environment_python("SeqMaestro_LocalUtils"))
    assert all(stage.commands[0].argv[0] == run_experiment.sys.executable
               for stage in stages if stage.name != "report_figures")


def test_every_call_the_driver_makes_has_a_name_no_other_call_has(driver, tmp_path):
    """A call's log, its invocation record and its failures file share its name, so no two calls
    of a run may share one. That covers every cell of every arm, each arm's reduction and panel,
    and the panel before and after the reveal."""
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    logs = [command.log for stage in run_experiment.build_stages(config, tmp_path / "run")
            for command in stage.commands]
    assert len(set(logs)) == len(logs)
    assert not [log for log in logs if "/" in log], "a call's name is a file name"


def test_every_arm_runs_as_one_process_per_labels_and_seed(driver, tmp_path):
    """The cells of one labels and seed depend on no other cells of their arm, so each arm runs as
    one process per pair, and its real-label processes name the arm whose cells they write."""
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    stages = {stage.name: stage for stage in run_experiment.build_stages(config, tmp_path / "run")}
    for model, pool in config.arm_cells():
        stage = stages[f"{model}/{pool}"]
        logs = [command.log for command in stage.commands]
        assert logs == [f"{model}__{pool}__{labels}__seed_{seed}"
                        for labels in ("real", "permuted") for seed in (1, 2)]
        for command in stage.commands:
            argv = command.argv
            assert argv[argv.index("--seeds") + 1] == command.log[-1]
            assert command.produces == (((model, pool),) if "__real__" in command.log else ())
    cart, deft = stages["deft_or_cart/positional"], stages["deft_or_cart/positional+llm_generated_joint"]
    assert not cart.calls_model and deft.calls_model
    assert cart.retry and not deft.retry, "a language model writing inside the fit is not run twice"
    assert stages["l1_logistic/llm_generated"].calls_model and stages["l1_logistic/llm_generated"].retry


def test_an_arm_is_reduced_and_profiled_as_soon_as_its_cells_are_there(driver, tmp_path):
    """Each arm's reduction waits for the processes that write its real-label cells and nothing
    else, and its panel for its reduction, so a cheap arm is reduced and profiled while the slow
    ones run. The analyses compare arms, so they wait for every arm."""
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config)
    stages = {stage.name: stage
              for stage in run_experiment.build_stages(config, tmp_path / "run")}
    every_arm = {command.log for model, pool in config.arm_cells()
                 for command in stages[f"{model}/{pool}"].commands}
    for name in ("analyse_stability", "analyse_co_discovery", "analyse_recurrence"):
        assert set(stages[name].commands[0].after) == every_arm, name

    reductions = {command.log: command for command in stages["reduction"].commands}
    assert len(reductions) == len(config.reduced_arms())
    l1 = reductions["reduction__l1_logistic__positional"]
    assert l1.argv[l1.argv.index("--arms") + 1] == "l1_logistic/positional"
    assert set(l1.after) == {"l1_logistic__positional__real__seed_1",
                             "l1_logistic__positional__real__seed_2"}
    tree = "deft_or_cart__positional+llm_generated_joint"
    assert set(reductions[f"reduction__{tree}"].after) == {
        f"{tree}__real__seed_1", f"{tree}__real__seed_2"}
    assert stages["reduction"].threads == 4

    panels = {command.log: command for command in stages["reliability"].commands}
    assert len(panels) == len(config.profiled_arms())
    assert panels["reliability__l1_logistic__positional"].after == (
        "reduction__l1_logistic__positional",)
    assert stages["reliability"].threads is None and stages["reveal"].threads is None
    assert stages["l1_logistic/positional"].threads == 1


def test_a_revealed_run_resumes_from_the_reveal_and_a_stop_stops_before_it(driver, tmp_path):
    experiment_config, run_experiment = driver
    stages = run_experiment.build_stages(a_config(tmp_path, experiment_config), tmp_path / "run")
    resumed, notes = run_experiment.plan(stages, revealed=True, stop_before_reveal=False)
    assert resumed[0].name == "reveal" and "resuming from the reveal" in notes[0]
    stopped, _ = run_experiment.plan(stages, revealed=False, stop_before_reveal=True)
    assert {stage.phase for stage in stopped} == {"before"} and stopped[-1].name == "reliability"
    with pytest.raises(experiment_config.ConfigError, match="nothing is left to run"):
        run_experiment.plan(stages, revealed=True, stop_before_reveal=True)


# --------------------------------------------------------------------------------------------
# A run, with the stages stubbed out
# --------------------------------------------------------------------------------------------


@pytest.fixture
def stubbed(driver, tmp_path, monkeypatch):
    """The driver on a small config, with every process it starts recorded by its log's name, and
    those of the stages named in `fail` failing. A stage of one process logs under the stage's name;
    one of several logs each under the stage's name and its slice."""
    experiment_config, run_experiment = driver
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "planted.json").write_text(json.dumps({"task": "Predicting a planted motif."}))
    monkeypatch.setattr(run_experiment, "DESCRIPTIONS", datasets)
    config = a_config(tmp_path, experiment_config, arms={"positional": {"models": ["l1_logistic"]}})
    source = tmp_path / "resolved.yaml"
    source.write_text(config.to_yaml())
    ran, fail = [], set()

    class Ended:
        pid = None

        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    def start_process(argv, log, environment, cwd):
        ran.append(log.stem)
        failing = any(log.stem == name or log.stem.startswith(f"{name}_") for name in fail)
        return Ended(1 if failing else 0), None

    monkeypatch.setattr(run_experiment, "start_process", start_process)
    run_dir = tmp_path / "run"
    argv = ["--config", str(source), "--run-dir", str(run_dir),
            "--skip", "analyse_recurrence", "report_figures"]
    return run_experiment, argv, run_dir, ran, fail


def test_the_reveal_waits_for_every_stage_before_it(stubbed):
    run_experiment, argv, _, ran, fail = stubbed
    fail.add("l1_logistic__positional")
    assert run_experiment.main(argv) == 1
    assert "reliability__l1_logistic__positional" in ran, \
        "a failing stage does not stop the ones after it"
    assert not {"reveal", "reliability_after_reveal", "report_inputs"} & set(ran)


def test_a_clean_run_goes_on_through_the_reveal(stubbed):
    run_experiment, argv, _, ran, _ = stubbed
    assert run_experiment.main(argv) == 0
    assert ran[-3:] == ["reveal", "reliability_after_reveal", "report_inputs"]
    assert ran.index("reliability__l1_logistic__positional") < ran.index("analyse_co_discovery"), \
        "the arm was profiled once its real-label cells were there, before the analyses, which " \
        "wait for every cell"


def test_a_failure_after_the_reveal_holds_nothing_back(stubbed):
    run_experiment, argv, _, ran, fail = stubbed
    fail.add("reliability_after_reveal")
    assert run_experiment.main(argv) == 1
    assert ran[-2:] == ["reliability_after_reveal", "report_inputs"]


def test_a_revealed_run_runs_only_what_comes_after(stubbed):
    run_experiment, argv, run_dir, ran, _ = stubbed
    run_dir.mkdir()
    (run_dir / "reveal.json").write_text(json.dumps({"revealed_utc": "then", "sealed": []}))
    assert run_experiment.main(argv) == 0
    assert ran == ["reveal", "reliability_after_reveal", "report_inputs"]
    assert run_experiment.main([*argv, "--only", "l1_logistic/positional"]) == 2, \
        "a fitting stage is refused"


def test_a_run_is_matched_to_its_config_by_the_name_it_recorded(driver, tmp_path, monkeypatch):
    """Starting a run for a config that already has one is refused, and the earlier run is offered.
    A run of another config whose name holds this one's is no run of this config."""
    from types import SimpleNamespace

    _, run_experiment = driver
    monkeypatch.setattr(run_experiment, "RUNS", tmp_path)
    for directory, name in (("2026-01-01_promoters_wide_full", "promoters_wide"),
                            ("named_by_hand", "promoters")):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "config.yaml").write_text(yaml.safe_dump({"name": name}))

    assert run_experiment.runs_of("promoters") == [tmp_path / "named_by_hand"]
    with pytest.raises(run_experiment.ConfigError, match="named_by_hand"):
        run_experiment.resolve_run_dir(None, SimpleNamespace(name="promoters"), new_run=False)
    fresh = run_experiment.resolve_run_dir(None, SimpleNamespace(name="enhancers"), new_run=False)
    assert fresh.parent == tmp_path and fresh.name.endswith("_enhancers_full")


def test_a_model_or_a_pool_selects_every_arm_with_it(driver):
    _, run_experiment = driver
    stages = tuple(run_experiment.Stage(name, ()) for name in (
        "baseline", "l1_logistic/positional", "ebm/positional", "ebm/positional+kmer", "reduction"))

    def names(only=(), skip=()):
        return [stage.name for stage in run_experiment.select(stages, list(only), list(skip))]

    assert names(only=["positional"]) == ["l1_logistic/positional", "ebm/positional"]
    assert names(only=["ebm", "baseline"]) == ["baseline", "ebm/positional", "ebm/positional+kmer"]
    assert names(skip=["ebm", "reduction"]) == ["baseline", "l1_logistic/positional"]
    with pytest.raises(run_experiment.ConfigError, match="no stage, model or pool named kmer"):
        names(only=["kmer"])


# --------------------------------------------------------------------------------------------
# Interpreters and records
# --------------------------------------------------------------------------------------------


def test_an_interpreter_is_found_under_this_conda_root_unless_overridden(monkeypatch, tmp_path):
    variable = python_override_variable("SeqMaestro_Method_deft")
    assert variable == "SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_DEFT"
    monkeypatch.delenv(variable, raising=False)
    assert environment_python("SeqMaestro_Method_deft") == \
        CONDA_ROOT / "envs" / "SeqMaestro_Method_deft" / "bin" / "python"
    elsewhere = tmp_path / "python"
    monkeypatch.setenv(variable, str(elsewhere))
    assert environment_python("SeqMaestro_Method_deft") == elsewhere
    message = missing_python("SeqMaestro_Method_deft")
    assert str(elsewhere) in message and variable in message
    elsewhere.write_text("")
    assert missing_python("SeqMaestro_Method_deft") is None


def test_preflight_names_pysrs_interpreter_for_the_stages_that_fit_it(driver, tmp_path, monkeypatch):
    """PySR's search runs in its own environment, so its arm's stage, the reduction and the panel
    need that interpreter; a stage that fits no PySR model does not."""
    experiment_config, run_experiment = driver
    config = a_config(tmp_path, experiment_config, arms={
        "positional": {"models": ["l1_logistic", "pysr"]},
        "llm_generated_iterative": {"models": ["pysr"]},
        "positional+kmer": {"models": ["l1_logistic"]}})
    monkeypatch.setenv(python_override_variable("SeqMaestro_Method_pysr"), str(tmp_path / "none"))
    stages = {stage.name: stage for stage in run_experiment.build_stages(config, tmp_path / "run")}

    for name in ("pysr/positional", "pysr/llm_generated_iterative", "reduction", "reliability"):
        problems = run_experiment.preflight(config, (stages[name],), tmp_path / "run")
        assert [problem for problem in problems if "SeqMaestro_Method_pysr" in problem], name
    for name in ("l1_logistic/positional+kmer", "reveal"):
        problems = run_experiment.preflight(config, (stages[name],), tmp_path / "run")
        assert not [problem for problem in problems if "SeqMaestro_Method_pysr" in problem], name
