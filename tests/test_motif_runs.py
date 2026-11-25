"""The motif sources in a run: the config that turns them on, the run's copy of its collection and
its store of scans and discoveries, the step before the arms, the arms that wait for it, the columns
handed to DEFT, the time estimate, and how a published collection is converted when it is fetched.

The dataset is the planted one (`tests/planted.py`), whose positives contain GGCGCC, and the
collection holds two motifs written for it: one for the planted site and one that matches nothing
in particular. Discovery runs the stand-in for STREME (`tests/small_motifs.py`), which finds the
planted site as STREME does.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import FIXED_POOLS, MOTIF_DISCOVERED, MOTIF_KNOWN, KnownMotifs, make_featuriser
from seqmaestro.harness.features import load_features
from seqmaestro.harness.fitted import load_fitted
from seqmaestro.motifs import Collection, Motif, ScanSettings, discovery, published, read_meme, write_meme
from seqmaestro.motifs.discovery import MEME_ENVIRONMENT
from seqmaestro.motifs.scan import Scanner
from seqmaestro.pipeline import config as experiment_config
from seqmaestro.pipeline import driver as run_experiment
from seqmaestro.pipeline import estimate, records, stage2
from seqmaestro.pipeline import motifs as run_motifs
from seqmaestro.pipeline.config import ConfigError
from seqmaestro.pipeline.layout import CONFIG_FILE, cell_dir
from seqmaestro.pipeline.repository import REPO
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.vendored import deft
from seqmaestro.provenance import python_override_variable
from seqmaestro.qc import group_dependence
from tests.planted import ANALYSES, LANGUAGE_MODEL, planted_dataset

COLLECTION = "small_collection"


def site(letters: str) -> np.ndarray:
    """A motif that expects `letters`, softened so that every letter stays possible."""
    return np.eye(4)[["ACGT".index(letter) for letter in letters]] * 0.94 + 0.015


MOTIFS = [Motif("PL1", "PLANTED", site("GGCGCC"), sites=50),
          Motif("TB1", "TATA", site("TATAAA"), sites=50)]


@pytest.fixture
def collections(tmp_path, monkeypatch):
    """A fetched collection in a folder of collections where the run looks for it."""
    folder = tmp_path / "collections" / COLLECTION
    folder.mkdir(parents=True)
    (folder / run_motifs.COLLECTION_FILE).write_text(write_meme(MOTIFS))
    monkeypatch.setattr(run_motifs, "MOTIF_COLLECTIONS", tmp_path / "collections")
    return tmp_path / "collections"


def base_config(tmp_path, arms: dict, blinded: bool = False) -> dict:
    dataset = planted_dataset(tmp_path)
    return {
        "name": "planted", "dataset": {"task": "planted", "splits": str(dataset.directory),
                                       "blinded": blinded},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10, "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60, "guidance": None}},
        "language_model": LANGUAGE_MODEL,
        "analyses": ANALYSES,
        "arms": arms,
    }


def with_fragment(tmp_path, arms=None, extra: dict | None = None):
    """A dataset's config extending a base config and the repository's `_motifs.yaml`."""
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    shutil.copy(REPO / "experiments" / "configs" / "_motifs.yaml", configs / "_motifs.yaml")
    (configs / "base.yaml").write_text(yaml.safe_dump(
        base_config(tmp_path, arms or {"positional": {"models": ["l1"]}}), sort_keys=False))
    stated = {"extends": ["base.yaml", "_motifs.yaml"],
              "sources": {MOTIF_KNOWN: {"collection": COLLECTION, "strands": "both"},
                          MOTIF_DISCOVERED: {"strands": "both"}}}
    for key, value in (extra or {}).items():
        stated[key] = value
    (configs / "planted.yaml").write_text(yaml.safe_dump(stated, sort_keys=False))
    return configs / "planted.yaml"


def a_run(tmp_path, collections, arms=None, threshold=1e-3, scope="seed"):
    """Make a run directory with its recorded config: both motif sources at a threshold the planted
    six-letter site can reach, discovered motifs compared with the collection, the fragment's arms
    plus any that `arms` names, and the permutation scope `scope`."""
    scan = {"strands": "both", "p_threshold": threshold, "min_match_share": 0.0}
    source = with_fragment(tmp_path, arms, extra={
        "sources": {MOTIF_KNOWN: {"collection": COLLECTION, **scan},
                    MOTIF_DISCOVERED: {**scan, "annotate_with": COLLECTION}},
        "grid": {"permutation_scope": scope}})
    config = experiment_config.load(source)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / CONFIG_FILE).write_text(config.to_yaml())
    return config, run_dir


# ------------------------------------------------------------------------------------ the config


def test_the_fragment_turns_both_motif_sources_on_with_their_settings(tmp_path, collections):
    config = experiment_config.load(with_fragment(tmp_path))

    assert list(config.arms) == ["positional", MOTIF_KNOWN, MOTIF_DISCOVERED,
                                 f"positional+{MOTIF_DISCOVERED}"]
    assert config.arms[MOTIF_KNOWN].models == ("l1_logistic", "xgboost")
    assert config.arms[MOTIF_DISCOVERED].models == ("l1_logistic", "ebm", "xgboost", "pysr")
    assert config.arms[f"positional+{MOTIF_DISCOVERED}"].models == ("deft_or_cart",)
    known = config.sources.motif_known
    assert (known.collection, known.strands, known.summaries) == (COLLECTION, "both",
                                                                  ("best_match", "count"))
    assert (known.p_threshold, known.normalise, known.min_match_share) == (1e-4, "counts", 0.01)
    discovered = config.sources.motif_discovered
    assert discovered.scan() == known.scan()
    assert (discovered.directions, discovered.continuous_fraction, discovered.min_width,
            discovered.max_width, discovered.max_motifs, discovered.annotate_with) == (
        "both", 0.333, 6, 15, 10, None)


def test_a_dataset_takes_one_motif_source_by_setting_the_other_to_null(tmp_path, collections):
    source = with_fragment(tmp_path, extra={
        "sources": {MOTIF_KNOWN: {"collection": COLLECTION, "strands": "both"},
                    MOTIF_DISCOVERED: None},
        "arms": {MOTIF_DISCOVERED: None, f"positional+{MOTIF_DISCOVERED}": None}})

    config = experiment_config.load(source)

    assert list(config.arms) == ["positional", MOTIF_KNOWN]
    assert config.sources.motif_discovered is None and config.sources.motif_known is not None


def test_the_recorded_config_states_the_known_motifs_and_loads_again(tmp_path, collections):
    config = experiment_config.load(with_fragment(tmp_path))
    recorded = tmp_path / "recorded.yaml"
    recorded.write_text(config.to_yaml())

    again = experiment_config.load(recorded)

    assert again.to_mapping() == config.to_mapping()
    assert again.to_mapping()["sources"][MOTIF_KNOWN]["min_match_share"] == 0.01


def test_a_dataset_that_states_no_collection_or_strands_is_refused(tmp_path, collections):
    source = with_fragment(tmp_path)
    stated = yaml.safe_load(source.read_text())
    del stated["sources"]
    source.write_text(yaml.safe_dump(stated))

    with pytest.raises(ConfigError, match="missing collection, strands"):
        experiment_config.load(source)


def stating(member: str, setting: str, value) -> dict:
    """The sources a dataset's config states, with `setting` of `member` set to `value`."""
    sources = {MOTIF_KNOWN: {"collection": COLLECTION, "strands": "both"},
               MOTIF_DISCOVERED: {"strands": "both"}}
    sources[member][setting] = value
    return {"sources": sources}


@pytest.mark.parametrize("member", [MOTIF_KNOWN, MOTIF_DISCOVERED])
@pytest.mark.parametrize("setting, value", [
    ("strands", "forward"),
    ("p_threshold", 1.5),
    ("min_match_share", 1.0),
    ("summaries", ["best_match", "best_match"]),
])
def test_a_scan_setting_that_means_nothing_is_refused(tmp_path, collections, member, setting, value):
    source = with_fragment(tmp_path, extra=stating(member, setting, value))

    with pytest.raises(ConfigError, match=rf"sources\.{member}\.{setting}"):
        experiment_config.load(source)


@pytest.mark.parametrize("setting, value, message", [
    ("directions", "sideways", "directions"),
    ("continuous_fraction", 0.6, "continuous_fraction"),
    ("min_width", 2, "min_width"),
    ("max_width", 31, "max_width"),
    ("min_width", 16, "min_width <= max_width"),
    ("max_motifs", 0, "max_motifs"),
    ("annotate_with", ["two", "collections"], "annotate_with"),
])
def test_a_discovery_setting_that_means_nothing_is_refused(tmp_path, collections, setting, value,
                                                           message):
    source = with_fragment(tmp_path, extra=stating(MOTIF_DISCOVERED, setting, value))

    with pytest.raises(ConfigError, match=message):
        experiment_config.load(source)


def test_a_discovered_source_states_its_strands_and_may_name_a_collection(tmp_path, collections):
    """The strands must be stated because whether a molecule is single- or double-stranded is a fact
    about the dataset, whereas comparing a found motif with a collection is a choice."""
    source = with_fragment(tmp_path)
    stated = yaml.safe_load(source.read_text())
    assert "annotate_with" not in experiment_config.load(source).to_mapping()["sources"][
        MOTIF_DISCOVERED], "a source that names no collection records none"

    stated["sources"][MOTIF_DISCOVERED]["annotate_with"] = COLLECTION
    source.write_text(yaml.safe_dump(stated))
    config = experiment_config.load(source)
    assert config.sources.motif_discovered.annotate_with == COLLECTION
    recorded = tmp_path / "recorded.yaml"
    recorded.write_text(config.to_yaml())
    assert experiment_config.load(recorded).to_mapping() == config.to_mapping()

    del stated["sources"][MOTIF_DISCOVERED]["strands"]
    source.write_text(yaml.safe_dump(stated))
    with pytest.raises(ConfigError, match=f"sources.{MOTIF_DISCOVERED}` is missing strands"):
        experiment_config.load(source)


def test_extends_folds_a_list_in_order_and_the_extending_file_last(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(yaml.safe_dump(
        base_config(tmp_path, {"positional": {"models": ["l1"]}})))
    (configs / "a.yaml").write_text(yaml.safe_dump({"grid": {"n_bootstrap": 20}}))
    (configs / "b.yaml").write_text(yaml.safe_dump({"grid": {"n_bootstrap": 30}}))
    source = configs / "run.yaml"

    source.write_text(yaml.safe_dump({"extends": ["base.yaml", "a.yaml", "b.yaml"]}))
    assert experiment_config.load(source).grid.n_bootstrap == 30
    source.write_text(yaml.safe_dump({"extends": ["base.yaml", "b.yaml", "a.yaml"]}))
    assert experiment_config.load(source).grid.n_bootstrap == 20
    source.write_text(yaml.safe_dump({"extends": ["base.yaml", "a.yaml"], "grid": {"n_bootstrap": 5}}))
    assert experiment_config.load(source).grid.n_bootstrap == 5
    source.write_text(yaml.safe_dump({"extends": []}))
    with pytest.raises(ConfigError, match="list of them"):
        experiment_config.load(source)


def test_a_blinded_run_keeps_known_motifs_away_from_a_language_model(tmp_path, collections):
    """A known motif is named after its factor, and a language model writing features beside it is
    shown the names the fit relied on, which could reveal what blinding withholds."""
    arms = {"motif_known+llm_generated_iterative": {"models": ["l1"]}}
    configs = tmp_path / "configs"
    with_fragment(tmp_path, arms)
    base = yaml.safe_load((configs / "base.yaml").read_text())
    base["sources"] = {"llm_generated": {"proposal_rounds": 1, "n_per_round": 2, "n_examples": 4},
                       "llm_generated_iterative": {"feedback_rounds": 1}}
    base["dataset"]["blinded"] = True
    (configs / "base.yaml").write_text(yaml.safe_dump(base))

    with pytest.raises(ConfigError, match="blinded"):
        experiment_config.load(configs / "planted.yaml")
    base["dataset"]["blinded"] = False
    (configs / "base.yaml").write_text(yaml.safe_dump(base))
    assert experiment_config.load(configs / "planted.yaml").arms


# --------------------------------------------------------------------------------- the pools


def test_every_set_of_the_enumerated_sources_is_a_fixed_pool():
    assert {"motif_known", "positional+motif_known", "kmer+motif_known",
            "positional+kmer+motif_known", "motif_known+covariates"} <= set(FIXED_POOLS)


def test_the_stage_1_check_offers_only_the_pools_it_can_build():
    """The group-dependence check of Stage 1 has neither a run's motif settings nor its labels."""
    assert group_dependence.POOLS == ("positional", "kmer", "positional+kmer")


def test_a_pool_takes_the_motifs_exactly_when_it_holds_them():
    known = KnownMotifs(Collection("c", tuple(MOTIFS), "d"),
                        ScanSettings(strands="both", p_threshold=1e-3, summaries=("count",),
                                     normalise="counts"), min_match_share=0.0)
    with pytest.raises(ValueError, match="must name the collection"):
        make_featuriser("positional+motif_known")
    with pytest.raises(ValueError, match="does not hold"):
        make_featuriser("positional", motif_known=known)
    assert make_featuriser("motif_known", motif_known=known).fit(pd.Series(["GGCGCCAA"])).feature_names


# ------------------------------------------------------------------------------- the run's cells


def test_a_cell_scans_with_the_runs_copy_and_the_next_one_reads_the_store(tmp_path, collections,
                                                                          monkeypatch):
    config, run_dir = a_run(tmp_path, collections, {"positional": {"models": ["l1"]}})
    rows = Rows.of(config)
    arm = stage2.arm_of(config, rows, run_dir, "l1_logistic", MOTIF_KNOWN)

    arm.fit(cell_dir(run_dir, "l1_logistic", MOTIF_KNOWN, "real", 1), "real", 1)

    copy = run_motifs.run_copy(run_dir, COLLECTION)
    assert read_meme(copy.read_text())[0].identifier == "PL1"
    records, _ = load_features(cell_dir(run_dir, "l1_logistic", MOTIF_KNOWN, "real", 1) / "features.json")
    assert {record.kind for record in records} == {"pwm_scan"}
    assert any(record.params["motif_id"] == "PL1" for record in records), \
        "the planted motif is what separates the classes"

    # The control at the same seed fits on the same rows, so it finds every scan in the store.
    monkeypatch.setattr(Scanner, "scan", lambda self, sequences: pytest.fail("scanned again"))
    arm.fit(cell_dir(run_dir, "l1_logistic", MOTIF_KNOWN, "permuted", 1), "permuted", 1)


def test_the_step_before_the_arms_scans_every_training_and_validation_sequence(
        tmp_path, collections, monkeypatch):
    config, run_dir = a_run(tmp_path, collections)
    rows = Rows.of(config)

    assert run_motifs.main(["--run-dir", str(run_dir)]) == 0

    record = json.loads((run_motifs.motifs_dir(run_dir) / run_motifs.KNOWN_RECORD).read_text())
    column = rows.dataset.sequence_column
    assert record["sequences"] == len(set(rows.train[column]) | set(rows.val[column]))
    assert record["collection"] == COLLECTION
    monkeypatch.setattr(Scanner, "scan", lambda self, sequences: pytest.fail("scanned again"))
    stage2.arm_of(config, rows, run_dir, "xgboost", MOTIF_KNOWN).fit(
        cell_dir(run_dir, "xgboost", MOTIF_KNOWN, "real", 2), "real", 2)


def test_the_step_is_refused_where_no_pool_holds_the_known_motifs(tmp_path, collections):
    config = experiment_config.load(with_fragment(tmp_path))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mapping = config.to_mapping()
    del mapping["arms"][MOTIF_KNOWN]
    del mapping["sources"][MOTIF_KNOWN]
    (run_dir / CONFIG_FILE).write_text(yaml.safe_dump(mapping, sort_keys=False))

    assert run_motifs.main(["--run-dir", str(run_dir)]) == 2


def test_the_arms_on_known_motifs_wait_for_the_step_and_the_others_do_not(tmp_path, collections):
    config, run_dir = a_run(tmp_path, collections)

    stages = run_experiment.build_stages(config, run_dir)

    names = [stage.name for stage in stages]
    assert names[:2] == ["baseline", run_motifs.STAGE]
    waits = {stage.name: {command.after for command in stage.commands} for stage in stages}
    assert waits[f"l1_logistic/{MOTIF_KNOWN}"] == {(run_motifs.KNOWN_CALL,)}
    assert waits["l1_logistic/positional"] == {()}


def test_the_preflight_names_a_collection_that_was_never_fetched(tmp_path, collections):
    config, run_dir = a_run(tmp_path, collections)
    shutil.rmtree(collections / COLLECTION)

    problems = run_experiment.preflight(config, run_experiment.build_stages(config, run_dir), run_dir)

    assert any("fetch_motif_collection.py" in problem for problem in problems)


def test_deft_is_handed_the_motif_columns_with_their_recipes(tmp_path, collections):
    config, run_dir = a_run(tmp_path, collections)
    rows = Rows.of(config)
    columns = rows.dataset.columns
    supplied = deft.supplied_featuriser(config, rows.dataset, f"positional+{MOTIF_KNOWN}", run_dir)

    deft.write_inputs(f"positional+{MOTIF_KNOWN}", rows.train, rows.train[columns.target].to_numpy(),
                      rows.train, rows.val, tmp_path / "inputs", columns=columns, supplied=supplied)

    handed = json.loads((tmp_path / "inputs" / "columns.json").read_text())["columns"]
    assert {entry["kind"] for entry in handed.values()} == {"pwm_scan"}
    assert handed["motif_PL1_PLANTED_best"]["params"]["collection"] == COLLECTION
    with pytest.raises(ValueError, match="no featuriser was given"):
        deft.write_inputs(f"positional+{MOTIF_KNOWN}", rows.train, rows.train[columns.target].to_numpy(),
                          rows.train, rows.val, tmp_path / "again", columns=columns)


def test_the_estimate_charges_the_scan_once_and_not_to_each_cell(tmp_path, collections):
    config, _ = a_run(tmp_path, collections)
    rows = estimate.Rows(config)

    X, seconds = rows.columns(MOTIF_KNOWN)

    assert X.shape[1] == 4 and seconds == 0.0
    assert list(rows.before) == [estimate.SCANNED_ONCE[MOTIF_KNOWN]]
    assert "Once, before the arms" in estimate.table([], config, 2, 1.0, rows.before)


# ----------------------------------------------------------------------------- discovered motifs


def discovery_calls(labels=("real", "permuted"), seeds=(1, 2)) -> list[str]:
    return [run_motifs.discovered_call(label, seed) for label in labels for seed in seeds]


def test_the_step_makes_each_discovery_once_and_each_arm_waits_for_its_own(tmp_path, collections):
    config, run_dir = a_run(tmp_path, collections)

    stages = run_experiment.build_stages(config, run_dir)

    [step] = [stage for stage in stages if stage.name == run_motifs.STAGE]
    assert [command.log for command in step.commands] == [run_motifs.KNOWN_CALL, *discovery_calls()]
    waits = {stage.name: {command.after for command in stage.commands} for stage in stages}
    each = {(call,) for call in discovery_calls()}
    assert waits[f"l1_logistic/{MOTIF_DISCOVERED}"] == each
    assert waits[f"deft_or_cart/positional+{MOTIF_DISCOVERED}"] == each


def test_under_the_cell_scope_a_control_makes_its_own_discovery(tmp_path, collections):
    """Under `permutation_scope: cell`, a control's shuffled labels depend on its arm, so no
    discovery can be made for them before the arms."""
    config, run_dir = a_run(tmp_path, collections, scope="cell")

    stages = run_experiment.build_stages(config, run_dir)

    [step] = [stage for stage in stages if stage.name == run_motifs.STAGE]
    assert [command.log for command in step.commands] == [run_motifs.KNOWN_CALL,
                                                         *discovery_calls(labels=("real",))]
    waits = {stage.name: {command.after for command in stage.commands} for stage in stages}
    assert waits[f"l1_logistic/{MOTIF_DISCOVERED}"] == {
        *((call,) for call in discovery_calls(labels=("real",))), ()}
    assert run_motifs.main(["--run-dir", str(run_dir), "--discover", "--labels", "permuted",
                            "--seed", "1"]) == 2


def test_a_cell_finds_its_motifs_in_its_fitting_rows_with_its_labels(tmp_path, collections):
    config, run_dir = a_run(tmp_path, collections)
    rows = Rows.of(config)
    arm = stage2.arm_of(config, rows, run_dir, "l1_logistic", MOTIF_DISCOVERED)

    arm.fit(cell_dir(run_dir, "l1_logistic", MOTIF_DISCOVERED, "real", 1), "real", 1)

    used, _ = load_features(cell_dir(run_dir, "l1_logistic", MOTIF_DISCOVERED, "real", 1)
                            / "features.json")
    assert {record.kind for record in used} == {"pwm_scan"}
    planted = [record for record in used if record.params["consensus"] == "GGCGCC"]
    assert planted, "the planted site is what separates the classes"
    assert {(record.params["direction"], record.params["rank"]) for record in planted} == {("up", 1)}
    assert planted[0].extra["closest_known"]["motif_id"] == "PL1", "compared with the collection"
    assert "GGCGCC" in planted[0].description and "PL1" not in planted[0].description

    # The control at the same seed fits on the same rows with shuffled labels, so it makes a
    # discovery of its own with them.
    arm.fit(cell_dir(run_dir, "l1_logistic", MOTIF_DISCOVERED, "permuted", 1), "permuted", 1)
    made = sorted(path.name for path in (run_motifs.motifs_dir(run_dir) / "discoveries").iterdir()
                  if path.is_dir())
    assert len(made) == 2


def test_the_step_before_the_arms_makes_the_discovery_the_cells_find(tmp_path, collections,
                                                                    monkeypatch):
    """The step draws the rows and labels that every cell at a seed fits on (a control's shuffled
    labels included), so the cells find its discoveries and scans in the store."""
    config, run_dir = a_run(tmp_path, collections)
    monkeypatch.setattr(records, "write_snapshot", lambda *args, **kwargs: None)

    for labels in ("real", "permuted"):
        assert run_motifs.main(["--run-dir", str(run_dir), "--discover", "--labels", labels,
                                "--seed", "2"]) == 0

    record = json.loads((run_motifs.motifs_dir(run_dir)
                         / run_motifs.DISCOVERED_RECORD.format(labels="real", seed=2)).read_text())
    assert record["found"] == {"up": 10, "down": 10} and record["features"] > 0
    assert (run_motifs.motifs_dir(run_dir) / discovery.DISCOVERIES_DIR / record["discovery_key"]
            / discovery.FOUND_FILE).exists()
    monkeypatch.setattr(discovery, "_streme", lambda *args: pytest.fail("discovered again"))
    monkeypatch.setattr(Scanner, "scan", lambda self, sequences: pytest.fail("scanned again"))
    for model in ("xgboost", "l1_logistic"):
        for labels in ("real", "permuted"):
            stage2.arm_of(config, Rows.of(config), run_dir, model, MOTIF_DISCOVERED).fit(
                cell_dir(run_dir, model, MOTIF_DISCOVERED, labels, 2), labels, 2)


def test_the_discovery_step_is_refused_outside_the_runs_grid(tmp_path, collections):
    _, run_dir = a_run(tmp_path, collections)
    discover = ["--run-dir", str(run_dir), "--discover"]

    assert run_motifs.main([*discover, "--labels", "real"]) == 2
    assert run_motifs.main([*discover, "--labels", "real", "--seed", "7"]) == 2


def test_a_saved_cell_scans_new_rows_without_discovering_again(tmp_path, collections, monkeypatch):
    """The motifs found are kept in the featuriser the cell saves, so the reveal scans the test rows
    with them in the Core environment."""
    config, run_dir = a_run(tmp_path, collections)
    rows = Rows.of(config)
    cell = cell_dir(run_dir, "l1_logistic", MOTIF_DISCOVERED, "real", 1)
    stage2.arm_of(config, rows, run_dir, "l1_logistic", MOTIF_DISCOVERED).fit(cell, "real", 1)
    monkeypatch.setattr(discovery, "_streme", lambda *args: pytest.fail("discovered again"))

    fitted = load_fitted(cell)
    test = pd.read_csv(rows.dataset.directory / "test.csv")
    X = fitted.transform(rows.dataset.columns_for(MOTIF_DISCOVERED).inputs(test))

    assert list(X.columns) == fitted.featuriser.feature_names and len(X) == len(test)


def test_the_preflight_asks_for_the_meme_suite_where_a_pool_discovers(tmp_path, collections,
                                                                      monkeypatch):
    config, run_dir = a_run(tmp_path, collections)
    monkeypatch.setenv(python_override_variable(MEME_ENVIRONMENT), str(tmp_path / "no" / "python"))

    problems = run_experiment.preflight(config, run_experiment.build_stages(config, run_dir), run_dir)

    assert any("./install.sh --with-motifs" in problem for problem in problems)


def test_deft_is_handed_the_discovered_motifs_with_what_their_records_hold(tmp_path, collections):
    config, run_dir = a_run(tmp_path, collections)
    rows = Rows.of(config)
    columns = rows.dataset.columns
    pool = f"positional+{MOTIF_DISCOVERED}"
    supplied = deft.supplied_featuriser(config, rows.dataset, pool, run_dir)

    deft.write_inputs(pool, rows.train, rows.train[columns.target].to_numpy(), rows.train, rows.val,
                      tmp_path / "inputs", columns=columns, supplied=supplied)

    handed = json.loads((tmp_path / "inputs" / "columns.json").read_text())["columns"]
    best = handed["streme_up_1_GGCGCC_best"]
    assert best["params"]["source"] == MOTIF_DISCOVERED
    assert best["extra"]["closest_known"]["motif_id"] == "PL1"


def test_the_estimate_charges_the_discoveries_to_the_step_before_the_arms(tmp_path, collections):
    config, _ = a_run(tmp_path, collections)
    rows = estimate.Rows(config)

    X, seconds = rows.columns(MOTIF_DISCOVERED)

    assert X.shape[1] > 0 and seconds == 0.0
    assert list(rows.before) == [estimate.SCANNED_ONCE[MOTIF_DISCOVERED]]


# ------------------------------------------------------------------------- published collections


def test_a_jaspar_file_is_named_and_found_as_jaspar_publishes_it():
    assert published.jaspar_name("2026", "CORE", "vertebrates", "non-redundant") == \
        "JASPAR2026_CORE_vertebrates_non-redundant"
    assert published.jaspar_url("2026", "CORE", None, "non-redundant").endswith(
        "/2026/CORE/JASPAR2026_CORE_non-redundant_pfms_meme.txt")
    with pytest.raises(ValueError, match="taxon"):
        published.jaspar_name("2026", "CORE", "mammals", "non-redundant")


def cisbp_archive() -> bytes:
    """A species archive as CisBP-RNA lays one out: a table of each motif's proteins, and a table
    per motif, one of them empty."""
    proteins = ("RBP_ID\tMotif_ID\tRBP_Name\tRBP_Status\n"
                "T1\tM1_2.00\tPTB\tD\n"
                "T2\tM2_2.00\tENSG000001\tI\n"
                "T3\tM2_2.00\tHOMOLOG\tI\n"
                "T4\tM3_2.00\tEMPTY\tD\n"
                "T5\t.\tNOMOTIF\tD\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(published.CISBP_RNA_PROTEINS, proteins)
        archive.writestr(f"{published.CISBP_RNA_MATRICES}/M1_2.00.txt",
                         "Pos\tA\tC\tG\tU\n1\t0.1\t0.7\t0.1\t0.1\n2\t0.1\t0.1\t0.1\t0.7\n")
        archive.writestr(f"{published.CISBP_RNA_MATRICES}/M2_2.00.txt",
                         "Pos\tA\tC\tG\tU\n1\t0.7\t0.1\t0.1\t0.1\n")
        archive.writestr(f"{published.CISBP_RNA_MATRICES}/M3_2.00.txt", "")
    return buffer.getvalue()


def test_a_cisbp_rna_archive_keeps_the_motifs_measured_for_their_proteins():
    motifs, counts = published.cisbp_rna_motifs(cisbp_archive(), inferred=False)

    assert [(motif.identifier, motif.name, motif.consensus) for motif in motifs] == [
        ("M1_2.00", "PTB", "CT")]
    assert counts == {"listed": 3, "inferred_left_out": 1, "empty_left_out": 1, "kept": 1}


def test_an_inferred_motif_is_kept_when_asked_for_and_named_by_its_symbols():
    motifs, counts = published.cisbp_rna_motifs(cisbp_archive(), inferred=True)

    assert [(motif.identifier, motif.name) for motif in motifs] == [("M1_2.00", "PTB"),
                                                                    ("M2_2.00", "HOMOLOG")]
    assert counts["kept"] == 2
    assert published.cisbp_rna_name("Homo_sapiens", inferred=True) == "CisBP-RNA_2.00_Homo_sapiens_all"
