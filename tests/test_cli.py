"""The command line (`seqmaestro/cli/`): Stage 1 from one table, the task description, the config of
each preset, the options that answer every question, and the summary of a run.

Nothing here starts the driver or calls a language model: the estimate, the launch and the live view
are replaced, and a stand-in client drafts the description.
"""

from __future__ import annotations

import json
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from seqmaestro import arms as registry
from seqmaestro.cli import app, describe, launch, model, prepare, ui, watch
from seqmaestro.cli import report as report_writer
from seqmaestro.cli import run as steps
from seqmaestro.dataset import FAILED, FLAGGED, PASSED, load_dataset
from seqmaestro.llm import ScriptedClient
from seqmaestro.pipeline import config as experiment_config
from tests.test_status import a_cell, a_job, a_run, a_snapshot

MOTIF = "GGCGCC"
WORDS = "Made-up binding sites; each row is one candidate site, labelled by whether a factor bound it."


def a_table(path, n=240, kind="binary", variable=False, rna=False, seed=0):
    """A table as a user brings one: a sequence column (some rows in lower case), a target, a gene
    shared by three rows at a time, a column nothing reads, one row whose sequence holds a symbol
    outside the alphabet, and one row without a target."""
    rng = np.random.default_rng(seed)
    letters = list("ACGU" if rna else "ACGT")
    rows = []
    for index in range(n):
        sequence = "".join(rng.choice(letters, int(rng.integers(30, 61)) if variable else 40))
        positive = bool(rng.random() < 0.5)
        if positive:
            sequence = sequence[:5] + MOTIF + sequence[5 + len(MOTIF):]
        target = (("bound" if positive else "unbound") if kind == "binary"
                  else round(1.5 * positive + float(rng.normal()), 4))
        rows.append({"seq": sequence.lower() if index % 7 == 0 else sequence, "outcome": target,
                     "gene": f"g{index // 3}", "note": "kept by nobody"})
    rows[3]["seq"] = rows[3]["seq"][:10] + "X" + rows[3]["seq"][11:]
    rows[4]["outcome"] = ""
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def prepared(tmp_path, kind="binary", variable=False, rna=False, groups=("gene",), anchor="start"):
    """A table taken through Stage 1 as `seqmaestro run` takes it, with the plain description."""
    table = a_table(tmp_path / "table.csv", kind=kind, variable=variable, rna=rna)
    frame = prepare.read_table(table)
    target = prepare.inspect_target(frame, "outcome")
    rows = prepare.keep_rows(frame, "seq", target, "bound" if kind == "binary" else None,
                             list(groups))
    window = prepare.positional_window(rows, anchor)
    split = prepare.assign_splits(rows)
    paths = prepare.Paths("table", tmp_path / "root")
    record = prepare.write(table, paths, rows, split, window,
                           describe.plain(describe.Facts.of(table, rows), ""))
    return table, rows, split, paths, record


def failed(record):
    return {name: check for name, check in record["checks"].items() if check["status"] == FAILED}


# ------------------------------------------------------------------------------ Stage 1


def test_a_binary_table_becomes_a_dataset_that_passes_every_check(tmp_path):
    table, rows, split, paths, record = prepared(tmp_path)
    assert not failed(record)
    assert record["classes"]["1"]["name"] == "bound", "the class is named by its value"
    assert rows.values == ("bound", "unbound") and rows.without_target == 1
    assert rows.step["rows_in"] == {1: rows.step["rows_in"][1], 0: rows.step["rows_in"][0]}
    assert sum(rows.step["rows_in"].values()) - sum(rows.step["rows_out"].values()) == 1, (
        "the row with a symbol outside the alphabet is dropped, and counted")
    train = pd.read_csv(paths.splits / "train.csv")
    assert list(train.columns) == ["seq", "outcome"], "the split files hold the two columns alone"
    assert train["seq"].str.fullmatch("[ACGT]+").all(), "the sequences are in upper case"
    assert set(train["outcome"]) == {0, 1}
    genes = rows.frame.assign(split=split).groupby("gene")["split"].nunique()
    assert (genes == 1).all(), "the rows of one gene are on one side of the split"
    assert load_dataset(paths.splits).target_kind == "binary"
    assert (paths.data / "raw" / table.name).read_bytes() == table.read_bytes()
    assert "## The grouping" in (paths.data / "README.md").read_text()


def test_identical_sequences_are_kept_on_one_side_of_the_split(tmp_path):
    table = a_table(tmp_path / "table.csv")
    frame = prepare.read_table(table)
    frame.loc[100:139, "seq"] = frame.loc[0:39, "seq"].to_numpy()
    frame.loc[100:139, "outcome"] = frame.loc[0:39, "outcome"].to_numpy()
    target = prepare.inspect_target(frame, "outcome")
    rows = prepare.keep_rows(frame, "seq", target, "bound", [])
    split = prepare.assign_splits(rows)
    paths = prepare.Paths("table", tmp_path / "root")
    record = prepare.write(table, paths, rows, split, None,
                           describe.plain(describe.Facts.of(table, rows), ""))
    assert record["checks"]["no_sequence_in_two_splits"]["status"] == PASSED
    assert record["split"]["grouped_by"] == ["seq"]


def test_a_continuous_target_is_balanced_across_the_folds_in_bins(tmp_path):
    _, rows, split, _, record = prepared(tmp_path, kind="continuous")
    assert not failed(record)
    assert rows.kind == "continuous" and record["target"]["kind"] == "continuous"
    assert record["split"]["stratified_on"] == [f"{prepare.TARGET_BINS} bins of outcome"]
    means = rows.frame.assign(split=split).groupby("split")["outcome"].mean()
    assert means.max() - means.min() < 0.5, "each split covers the range of the target"


def test_sequences_of_different_lengths_get_a_window_as_long_as_the_shortest(tmp_path):
    _, rows, _, _, record = prepared(tmp_path, variable=True, anchor="end")
    shortest, _ = rows.lengths
    assert record["sequences"]["length"] == "variable"
    assert record["sequences"]["positional_window"] == {"length": shortest, "anchor": "end",
                                                        "pad": False}
    assert record["checks"]["positional_window_fits"]["status"] == PASSED, "no row is dropped"
    assert record["checks"]["sequence_lengths"]["status"] == FLAGGED
    assert not failed(record)


def test_sequences_with_u_and_no_t_are_read_as_rna(tmp_path):
    _, rows, _, _, record = prepared(tmp_path, rna=True)
    assert rows.alphabet == "RNA" and rows.symbols == "ACGU"
    assert record["sequences"]["alphabet"] == ["A", "C", "G", "U"]


def test_a_target_of_neither_two_classes_nor_numbers_is_refused():
    frame = pd.DataFrame({"seq": ["ACGT"] * 4, "three": ["a", "b", "c", "a"], "one": ["x"] * 4})
    with pytest.raises(prepare.PrepareError, match="two classes, or a number"):
        prepare.inspect_target(frame, "three")
    with pytest.raises(prepare.PrepareError, match="nothing to predict"):
        prepare.inspect_target(frame, "one")


def test_a_number_is_one_value_however_it_is_written():
    frame = pd.DataFrame({"y": ["1", "1.0", "0", "0.00", "NA"]})
    target = prepare.inspect_target(frame, "y")
    assert (target.kind, target.values, target.unusable) == ("binary", ("0", "1"), 1)
    assert target.value_of("1.00") == "1" and target.value_of("2") is None


def test_a_split_with_fewer_groups_than_folds_is_refused(tmp_path):
    table = a_table(tmp_path / "table.csv", n=60)
    frame = prepare.read_table(table)
    frame["gene"] = [f"g{index % 5}" for index in range(len(frame))]
    target = prepare.inspect_target(frame, "outcome")
    rows = prepare.keep_rows(frame, "seq", target, "bound", ["gene"])
    with pytest.raises(prepare.PrepareError, match="5 groups"):
        prepare.assign_splits(rows)


def test_names_come_from_the_table_and_a_table_must_be_csv_or_tsv(tmp_path):
    assert prepare.name_for(tmp_path / "Sites 60nt.csv") == "sites_60nt"
    assert prepare.name_for(tmp_path / "rows.tsv.gz") == "rows"
    assert prepare.name_problem("sites_60nt") is None and prepare.name_problem("Sites") is not None
    (tmp_path / "rows.xlsx").write_text("")
    with pytest.raises(prepare.PrepareError, match=r"\.csv or a \.tsv"):
        prepare.read_table(tmp_path / "rows.xlsx")


def test_the_files_of_a_dataset_are_never_replaced(tmp_path):
    table, rows, split, paths, _ = prepared(tmp_path)
    with pytest.raises(prepare.PrepareError, match="exist already"):
        prepare.write(table, paths, rows, split, None,
                      describe.plain(describe.Facts.of(table, rows), ""))


# ------------------------------------------------------------------------------ the presets


@pytest.mark.parametrize("kind", ["binary", "continuous"])
@pytest.mark.parametrize("preset", list(launch.PRESETS))
def test_each_preset_gives_a_config_that_loads(tmp_path, kind, preset):
    table, rows, _, paths, _ = prepared(tmp_path, kind=kind, variable=True)
    path = launch.write_config(paths, launch.config_document(paths, table, rows.kind, preset, True))
    config = experiment_config.load(path)
    assert config.dataset.target_kind == kind
    assert config.sources.kmer.normalise == "per_length", "counts do not grow with the length"
    pools = [pool for _, pool in config.arm_cells()]
    if preset == "quick":
        assert list(config.grid.seeds) == [42, 43]
        assert not any(registry.calls_language_model(pool) for pool in pools)
        assert "igsr" not in {model for model, _ in config.arm_cells()}
    else:
        assert any(registry.calls_language_model(pool) for pool in pools)


# ------------------------------------------------------------------------------ the description


def a_reply(task="Predicting binding from the sequence.", **changes):
    found = {"task": task, "sequence_layout": "40 nucleotides of DNA.",
             "target_name": "binding of the factor",
             "classes": {"1": {"name": "bound", "meaning": "sites the factor bound"},
                         "0": {"name": "not bound", "meaning": "sites it did not"}}} | changes
    return f"Here it is:\n```json\n{json.dumps(found)}\n```"


def test_the_draft_is_read_from_the_reply_and_is_sent_no_row_of_the_table(tmp_path):
    table, rows, _, _, _ = prepared(tmp_path)
    facts = describe.Facts.of(table, rows)
    client = ScriptedClient(a_reply())
    drafted = describe.draft(client, facts, WORDS)
    assert drafted.classes == {1: ("bound", "sites the factor bound"),
                               0: ("not bound", "sites it did not")}
    system, user = client.seen[0]
    assert WORDS in user and "`outcome`" in user and '"classes"' in system
    assert not any(sequence in user for sequence in rows.frame["seq"])
    with pytest.raises(describe.DescriptionError, match="class 0"):
        describe.draft(ScriptedClient(a_reply(classes={"1": {"name": "a", "meaning": "b"}})),
                       facts, WORDS)
    with pytest.raises(describe.DescriptionError, match="did not answer"):
        describe.draft(ScriptedClient(TimeoutError("slow")), facts, WORDS)


def test_an_edited_description_reads_back(tmp_path):
    table, rows, _, _, _ = prepared(tmp_path, kind="continuous")
    facts = describe.Facts.of(table, rows)
    plain = describe.plain(facts, WORDS)
    assert plain.task == WORDS and plain.target_meaning
    assert describe.checked(yaml.safe_load(describe.as_text(plain)), facts) == plain
    with pytest.raises(describe.DescriptionError, match="target_meaning"):
        describe.checked({"task": "a", "sequence_layout": "b", "target_name": "c"}, facts)


# ------------------------------------------------------------------------------ the command


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """`seqmaestro run` with its files written under `tmp_path`, no credentials read, and the
    estimate, the launch, the record and the live view replaced. Returns what each was given."""
    given = {}

    def estimate(config, run_dir):
        given["estimate"] = config
        return 0

    def start(config, run_dir):
        given["start"] = config
        return 0

    def record(run_dir, name, stages):
        given["record"] = (name, stages)
        return True

    monkeypatch.setattr(steps, "ROOT", tmp_path / "root")
    monkeypatch.setattr(model, "load_credentials", lambda: None)
    monkeypatch.setattr(model, "missing", lambda: ["SEQMAESTRO_LLM_PROVIDER"])
    monkeypatch.setattr(launch, "estimate", estimate)
    monkeypatch.setattr(launch, "start", start)
    monkeypatch.setattr(launch, "record", record)
    monkeypatch.setattr(watch, "follow", lambda run_dir: None)
    return given


def invoke(*arguments):
    return CliRunner().invoke(app, list(arguments), env={"COLUMNS": "200"})


OPTIONS = ("--name", "mine", "--sequence", "seq", "--target", "outcome", "--positive", "bound",
           "--group-by", "gene", "--yes")


def test_the_options_answer_every_question(tmp_path, stand_ins):
    table = a_table(tmp_path / "table.csv")
    result = invoke("run", str(table), *OPTIONS, "--describe", WORDS)
    assert result.exit_code == 0, result.output
    paths = prepare.Paths("mine", tmp_path / "root")
    assert json.loads(paths.description.read_text())["task"] == WORDS, "no model drafts it"
    assert stand_ins["estimate"] == stand_ins["start"] == paths.config
    assert stand_ins["record"] == (steps.NO_MODEL, ["1", "2"])
    config = experiment_config.load(paths.config)
    assert list(config.grid.seeds) == [42, 43], "the quick preset is the default"
    assert "the run goes on" in result.output


def test_under_yes_a_question_about_the_data_is_never_guessed(tmp_path, stand_ins):
    table = a_table(tmp_path / "table.csv")
    result = invoke("run", str(table), "--name", "mine", "--target", "outcome", "--yes")
    assert result.exit_code == 2
    assert "--sequence" in result.output
    assert not (tmp_path / "root").exists(), "nothing is written before every answer is in"


def test_the_models_draft_is_what_the_models_are_told(tmp_path, stand_ins, monkeypatch):
    monkeypatch.setattr(model, "missing", list)
    monkeypatch.setattr(model, "client", lambda: ScriptedClient(a_reply("The drafted task.")))
    table = a_table(tmp_path / "table.csv")
    result = invoke("run", str(table), *OPTIONS, "--describe", WORDS, "--preset", "full")
    assert result.exit_code == 0, result.output
    paths = prepare.Paths("mine", tmp_path / "root")
    assert json.loads(paths.description.read_text())["task"] == "The drafted task."
    assert load_dataset(paths.splits).classes[0].name == "not bound"
    assert stand_ins["record"] == ("scripted", ["1", "2"])


# ------------------------------------------------------------------------------ where a run stands


def rendered(renderable) -> str:
    console = Console(file=StringIO(), width=200, theme=ui.THEME)
    console.print(renderable)
    return console.file.getvalue()


def test_the_summary_of_a_run_shows_each_stage_and_what_runs(tmp_path):
    run_dir, _ = a_run(tmp_path)
    a_cell(run_dir, "l1_logistic", "positional", "real", 1, 0.8, [0.7, 0.9])
    a_snapshot(run_dir, [a_job("l1_logistic__positional__real__seed_2", "l1_logistic/positional",
                               "running")])
    text = rendered(watch.summary(run_dir, watch.look(run_dir)))
    assert "planted  Running" in text
    assert "1 of 12 cells" in text and "Task construction" in text
    assert "L1 Logistic Regression · positions, real labels, seed 2" in text
    assert watch.find(str(run_dir)) == run_dir.resolve()
    with pytest.raises(FileNotFoundError):
        watch.find(str(tmp_path / "nowhere"))
    result = invoke("status", str(run_dir))
    assert result.exit_code == 0 and "Task construction" in result.output


# ------------------------------------------------------------------------------ the report


GOOD_SOURCE = ("# SeqMaestro Report: planted sites\n\nThe models kept {{param:n_features}} features.\n\n"
               "{{table:key}}\n\n{{figure:stage2_grid}}\n")


@pytest.fixture
def finished_run(tmp_path, monkeypatch):
    """A run whose report's inputs are built, with a stand-in model configured. Returns the run's
    folder."""
    run_dir, _ = a_run(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("The planted dataset's README.")
    report = run_dir / "report"
    (report / "tables").mkdir(parents=True)
    (report / "tables" / "key.md").write_text("| Short | Meaning |\n|---|---|\n| POS | positions |\n")
    (report / "facts.json").write_text(json.dumps({
        "run": run_dir.name, "target_kind": "binary", "parameters": {"n_features": "three"},
        "stage1": {"readme": str(readme)}, "concepts": [{"idea": "left out of the request"}]}))
    (report / "figures.json").write_text(json.dumps({"figures": {"stage2_grid": {
        "path": "figures/stage2_grid.png", "caption": "Every model on held-out sequences",
        "width": "100%"}}}))
    monkeypatch.setattr(model, "missing", list)
    monkeypatch.setattr(report_writer.shutil, "which", lambda program: None)
    return run_dir


def with_model(monkeypatch, reply):
    client = ScriptedClient(reply)
    monkeypatch.setattr(model, "client", lambda: client)
    return client


def test_the_report_is_written_expanded_and_names_its_writer(finished_run, monkeypatch):
    client = with_model(monkeypatch, f"```markdown\n{GOOD_SOURCE}```")
    assert report_writer.write(finished_run, ui.Questions(yes=True), again=False) == 0
    report = finished_run / "report"
    assert (report / "REPORT.src.md").read_text() == GOOD_SOURCE, "the fence around it is removed"
    expanded = (report / "REPORT.md").read_text()
    assert "kept three features" in expanded and "| POS | positions |" in expanded
    assert json.loads((finished_run / "agents.json").read_text())[-1]["stages"] == ["report"]
    system, user = client.seen[0]
    assert "- tables: key" in system and "Consolidated report template" in system
    assert "The planted dataset's README." in user and "Every model on held-out sequences" in user
    assert "left out of the request" not in user, "the ideas reach the model as their tables"


def test_a_misused_placeholder_goes_back_to_the_model_once(finished_run, monkeypatch):
    drafts = iter(["{{table:nowhere}}\n\nAs the key shows, from {{table:key}}.\n", GOOD_SOURCE])
    client = with_model(monkeypatch, lambda system, user: next(drafts))
    assert report_writer.write(finished_run, ui.Questions(yes=True), again=False) == 0
    retry = client.seen[1][1]
    assert len(client.seen) == 2 and "{{table:nowhere}} names no table" in retry
    assert "{{table:key}} shares its line with other text" in retry, (
        "a table placeholder inside a sentence would put the table there")
    assert report_writer.problems("Kept `{{param:n}}` features.", {"param": {"n"}}) == [
        "{{param:n}} is in backticks, which would set its words as code"]


def test_a_source_still_wrong_after_the_retry_is_kept_for_the_user(finished_run, monkeypatch):
    client = with_model(monkeypatch, "{{param:nowhere}} and {{stray}}\n")
    assert report_writer.write(finished_run, ui.Questions(yes=True), again=False) == 1
    assert len(client.seen) == 2
    assert (finished_run / "report" / "REPORT.src.md").exists()
    assert not (finished_run / "report" / "REPORT.md").exists()
    assert report_writer.problems("{{param:nowhere}} and {{stray}}", {"param": set()}) == [
        "{{param:nowhere}} names no param this report has",
        "{{stray}} is none of the four kinds of placeholder"]


def test_a_report_written_before_is_kept_unless_asked_again(finished_run, monkeypatch):
    client = with_model(monkeypatch, GOOD_SOURCE)
    (finished_run / "report" / "REPORT.src.md").write_text("Written by an agent.\n")
    assert report_writer.write(finished_run, ui.Questions(yes=True), again=False) == 0
    assert not client.seen
    assert (finished_run / "report" / "REPORT.src.md").read_text() == "Written by an agent.\n"


def test_a_report_needs_its_inputs_and_a_model(finished_run, monkeypatch, tmp_path):
    monkeypatch.setattr(model, "missing", lambda: ["SEQMAESTRO_LLM_PROVIDER"])
    with pytest.raises(report_writer.ReportError, match="An agent can write the report"):
        report_writer.write(finished_run, ui.Questions(yes=True), again=False)
    unfinished, _ = a_run(tmp_path / "other")
    with pytest.raises(report_writer.ReportError, match="has not finished"):
        report_writer.write(unfinished, ui.Questions(yes=True), again=False)


def test_a_run_that_finishes_clean_has_its_report_written(finished_run, monkeypatch):
    written = []
    monkeypatch.setattr(report_writer, "write",
                        lambda run_dir, questions, again: written.append(run_dir) or 0)
    seen = watch.look(finished_run)
    assert steps.finished(finished_run, seen, ui.Questions(yes=True)) == 0
    assert written == [finished_run]
    monkeypatch.setattr(model, "missing", lambda: ["SEQMAESTRO_LLM_PROVIDER"])
    assert steps.finished(finished_run, seen, ui.Questions(yes=True)) == 0
    assert written == [finished_run], "without a model the command says how, and writes nothing"


def test_the_next_step_of_a_run_follows_where_it_stands(finished_run):
    assert "seqmaestro report" in watch.next_step(finished_run, watch.look(finished_run))
    a_snapshot(finished_run, [a_job("reveal", "reveal", "running")])
    assert watch.next_step(finished_run, watch.look(finished_run)) is None, "it runs"
