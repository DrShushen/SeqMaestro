"""One description per dataset, read by everything that has to tell a model about the data.

The proposer, the judge and the vendored methods all need the same facts, and without one shared
description each would carry its own copy. Copies of a fact drift silently: a model told the wrong
thing answers confidently and the run completes. So most of these tests check that the description
loads from one place, that a typo in it is refused and not dropped, and that the descriptions in
the repository's folder are usable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seqmaestro.llm import TaskContext, blinding_leaks, load_task
from tests.planted import ANALYSES, TARGET, planted_dataset

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "experiments" / "datasets"
CONFIGS = sorted(path for path in (REPO / "experiments" / "configs").glob("*.yaml")
                 if not path.name.startswith("_"))
DECLARATIONS = REPO / "experiments" / "prepare" / "declarations"


def write(tmp_path, name="thing", **fields):
    payload = {"task": "Predicting a thing.", **fields}
    (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    return tmp_path


def test_a_description_loads_from_one_file(tmp_path):
    context = load_task(write(tmp_path, sequence_layout="12 nt, position 6 is the site."), "thing")

    assert context.task == "Predicting a thing."
    assert context.sequence_layout == "12 nt, position 6 is the site."


def test_a_misspelled_field_is_refused_rather_than_dropped(tmp_path):
    """Ignoring a misspelled field would leave a fact silently missing from every prompt, which is
    the failure a single shared description exists to prevent."""
    write(tmp_path, sequence_lyaout="12 nt")

    with pytest.raises(ValueError, match="sequence_lyaout"):
        load_task(tmp_path, "thing")


def test_a_description_without_a_task_is_refused(tmp_path):
    (tmp_path / "empty.json").write_text(json.dumps({"sequence_layout": "12 nt"}))

    with pytest.raises(ValueError, match="task"):
        load_task(tmp_path, "empty")


def test_an_unknown_dataset_says_what_is_available(tmp_path):
    write(tmp_path, name="promoters")

    with pytest.raises(ValueError, match="promoters"):
        load_task(tmp_path, "no_such_dataset")


def test_target_name_is_not_rendered(tmp_path):
    """`target_name` exists so a method's prompt config can be generated from this description.
    Rendering it would change prompts that have already been measured, and no model needs it."""
    context = load_task(write(tmp_path, target_name="thing-ness"), "thing")

    assert context.target_name == "thing-ness"
    assert "thing-ness" not in context.render()


def test_render_carries_the_layout_because_positions_are_meaningless_without_it():
    context = TaskContext(task="Predicting.", sequence_layout="101 nt, position 50 is the site.")
    rendered = context.render()

    assert "Predicting." in rendered
    assert "position 50 is the site" in rendered


def test_as_dict_records_every_field_so_a_result_names_how_it_was_asked():
    context = TaskContext(task="t", sequence_layout="s", notes="n", target_name="tn", judge_only="j")
    assert context.as_dict() == {"task": "t", "sequence_layout": "s", "notes": "n",
                                 "target_name": "tn", "judge_only": "j"}


# --------------------------------------------------------------------------------------------
# Blinding: one fact, told to the judge and to nobody else
# --------------------------------------------------------------------------------------------
#
# The failure these guard against is silent in both directions. A leak makes a proposer's discovery
# uninterpretable, because it may have looked the answer up, and nothing in the output would say
# so. A judge left blind scores features against a phenotype it cannot name, which looks like a
# flat panel and not like a missing fact.

WITHHELD = "The factor is Widgetin-4, a cap-binding initiation factor."


def blinded(**fields):
    return TaskContext(task="Predicting a thing.", sequence_layout="100 nt, 0-based.",
                       judge_only=WITHHELD, **fields)


def test_a_withheld_fact_loads_like_any_other_field(tmp_path):
    assert load_task(write(tmp_path, judge_only=WITHHELD), "thing").judge_only == WITHHELD


def test_the_blind_rendering_is_the_plain_one():
    """The plain `render` is the blind one, as a safety property. Every model consumer calls
    `render`, so a caller who reaches for it by habit under-discloses and does not leak."""
    assert WITHHELD not in blinded().render()
    assert WITHHELD in blinded().render_for_judge()


def test_the_unblinded_rendering_carries_everything_the_blind_one_does():
    """`render_for_judge` is built on top of `render`, so a field added to one cannot go missing
    from the other. Two independent renderings would invite that drift."""
    context = blinded(notes="A note.")
    for line in context.render().splitlines():
        assert line in context.render_for_judge()


def test_a_dataset_that_withholds_nothing_renders_identically():
    context = TaskContext(task="Predicting a thing.")
    assert context.render_for_judge() == context.render()


def test_the_judge_is_told_the_withheld_fact():
    """Checked end to end through the judge's renderer and not through `render_for_judge`
    directly, because what matters is that the prompt the judge receives carries the fact."""
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.judge import PanelSource, build_panel, render

    entry = build_panel([PanelSource(
        arm="an_arm", labels="real", seed=42, target_kind="binary",
        records=[FeatureRecord(name="f", description="d", kind="test", importance=1.0)],
    )], top_n=1)[0]

    assert WITHHELD in render(entry, blinded()).user


def test_the_proposer_is_not():
    """The proposer's prompt, checked end to end, does not carry the withheld fact. A proposer
    shown the phenotype's name can retrieve what is known about it instead of finding it in the
    data, and the run would still complete."""
    from seqmaestro.proposer import Example, render_request

    prompt = render_request(blinded(), n_features=3,
                            examples=[Example("ACGT", 1), Example("TTTT", 0)],
                            already_proposed=[], sequence_column="window", target_column="y", target_kind="binary")

    assert WITHHELD not in prompt
    assert "Widgetin" not in prompt


def test_a_proposal_key_is_unchanged_by_the_withheld_fact():
    """The withheld fact never reaches the proposer, so it cannot change a reply. Keying on it
    would discard cached proposals every time the judge's part of a description was edited, and
    pay again for identical answers."""
    import pandas as pd

    from seqmaestro.llm import ScriptedClient
    from seqmaestro.proposer import ProposalSettings, Proposer

    columns = {"sequence_column": "window", "target_column": "y", "target_kind": "binary"}
    settings = ProposalSettings(n_rounds=1, n_per_round=1, n_examples=2)
    rows = pd.Series(["ACGT", "TTTT", "GGCC", "AATT"])
    labels = np.array([0, 1, 0, 1])

    def key(context):
        return Proposer(ScriptedClient("[]"), context, settings=settings,
                        **columns).key_for(rows, labels)

    assert key(blinded()) == key(TaskContext(task="Predicting a thing.",
                                             sequence_layout="100 nt, 0-based."))
    assert key(blinded()) != key(TaskContext(task="Predicting something else.",
                                             sequence_layout="100 nt, 0-based."))


def test_each_consumer_records_what_it_was_told():
    """A blinded run can only be read afterwards if the record says which side of the blind each
    number came from. The proposer's manifest omits the fact, and the judge's output carries it."""
    from seqmaestro.judge import summarise
    from seqmaestro.llm import ScriptedClient
    from seqmaestro.proposer import ProposalSettings, Proposer

    proposer = Proposer(ScriptedClient("[]"), blinded(), settings=ProposalSettings(n_rounds=1),
                        sequence_column="window", target_column="y", target_kind="binary")

    assert "judge_only" not in proposer.describe()["context"]
    assert summarise([], blinded())["task_context"]["judge_only"] == WITHHELD


# --------------------------------------------------------------------------------------------
# The descriptions in the repository's folder
# --------------------------------------------------------------------------------------------
#
# A clone holds the example's. A machine with datasets of its own also holds theirs, which git
# ignores, and the tests below check those too.


def test_every_description_here_loads():
    files = sorted(DATASETS.glob("*.json"))
    assert files, f"no dataset descriptions in {DATASETS}"

    for path in files:
        context = load_task(DATASETS, path.stem)
        assert context.task, path.name
        assert context.sequence_layout, (
            f"{path.name} states no sequence layout, so a feature over positions 41-49 cannot be "
            f"judged or proposed against anything."
        )


# --------------------------------------------------------------------------------------------
# What the column names give away
# --------------------------------------------------------------------------------------------

BLINDED = TaskContext(task="Predicting whether an element drives expression in cell type A or B.",
                      judge_only="Cell type A is HepG2, a liver line, and cell type B is K562.")


def test_the_withheld_words_are_those_of_judge_only_that_the_models_are_not_told():
    # "cell" and "type" are in the task, so telling them gives nothing away. Words under three
    # letters are left out.
    assert BLINDED.withheld_words() == {"hepg2", "liver", "line", "and", "k562"}
    assert TaskContext(task="Predicting a thing.").withheld_words() == frozenset()


@pytest.mark.parametrize("name", ["is_hepg2", "isHepG2", "HEPG2_ACTIVE"])
def test_a_column_name_holds_a_withheld_word_however_it_is_written(name):
    assert blinding_leaks(BLINDED, {name: None}) == [
        f"the column name {name!r} holds the withheld word 'hepg2'"]


def test_neutral_column_names_give_nothing_away():
    assert blinding_leaks(BLINDED, {"sequence": None, "in_cell_type_a": None}) == []


def test_a_description_holds_a_withheld_word_only_as_a_whole_word():
    assert blinding_leaks(BLINDED, {"signal": "accessibility in liver"}) == [
        "the description of 'signal' holds the withheld word 'liver'"]
    assert blinding_leaks(BLINDED, {"signal": "counted over deliveries"}) == []


def test_a_description_that_withholds_nothing_finds_nothing():
    assert blinding_leaks(TaskContext(task="Predicting a thing."), {"is_hepg2": "in liver"}) == []


def test_every_config_here_is_blinded_as_its_description_is_and_its_columns_give_nothing_away():
    """Read from each dataset's declaration, which states its columns and describes its covariates,
    since `data/` is not in git. A declaration is named for its data folder and split directory.
    The configs are those in the folder: the example's in a clone, and beside it those of the
    datasets a machine holds. A config that extends another may state no dataset of its own, so
    each is read with what it extends folded in."""
    from seqmaestro.pipeline.config import resolve

    assert CONFIGS, "a glob that matched nothing would pass without reading anything"
    for path in CONFIGS:
        dataset = resolve(path)["dataset"]
        context = load_task(DATASETS, dataset["task"])
        assert dataset["blinded"] is bool(context.judge_only), path.name
        splits = Path(dataset["splits"])
        declared = yaml.safe_load(
            (DECLARATIONS / f"{splits.parent.name}__{splits.name}.yaml").read_text())
        shown = {declared["columns"]["sequence"]: None, declared["columns"]["target"]: None,
                 **{name: entry["description"]
                    for name, entry in (declared.get("covariates") or {}).items()}}
        assert blinding_leaks(context, shown) == [], path.name


def test_a_proposer_arm_refuses_a_blinded_run_whose_target_column_names_what_it_withholds(
        runners, tmp_path, monkeypatch, capsys):
    """Before anything is sent or written: `--show-prompt`, which prints the prompt, refuses too."""
    import yaml

    datasets = write(tmp_path, name="planted", task="Predicting a class from random windows.",
                     sequence_layout="Random bases.",
                     judge_only=f"The {TARGET} column marks the windows that carry the motif.")
    monkeypatch.setattr(runners.arm, "DESCRIPTIONS", datasets)
    dataset = planted_dataset(tmp_path / "data")
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": True},
        "grid": {"seeds": [1], "labels": ["real"], "train_fraction": 0.8, "n_bootstrap": 0,
                 "permutation_scope": "seed", "class_weight": "balanced", "permutation_repeats": 0,
                 "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "sources": {"llm_generated": {"proposal_rounds": 1, "n_per_round": 1, "n_examples": 20}},
        "analyses": ANALYSES,
        "arms": {"llm_generated": {"models": ["l1_logistic"]}}}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(runners.config.load(source).to_yaml())

    assert runners.arm.main(["--run-dir", str(run_dir), "--show-prompt"]) == 2
    assert f"the column name {TARGET!r} holds the withheld word {TARGET!r}" in capsys.readouterr().err
    assert [path.name for path in run_dir.iterdir()] == ["config.yaml"]
