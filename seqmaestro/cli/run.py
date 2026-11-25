"""`seqmaestro run`: from a table of sequences to a finished run, as a short conversation.

The steps, each under a numbered heading: the table and the dataset's name; the columns (the
sequences, the target, and the columns that group rows); the rows kept and the split; what the
models are told; the dataset's files and checks; the run's preset, its estimate and its launch; and
the run followed to its end. Each question can be answered with an option, and `--yes` takes the
default of every question that has one, so the same steps can run from a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.table import Table

from seqmaestro.cli import describe, launch, model, prepare, report, ui, watch
from seqmaestro.cli.prepare import Paths
from seqmaestro.dataset import DATASET_FILE, FAILED, FLAGGED, NOT_APPLICABLE, PASSED
from seqmaestro.pipeline.config import load
from seqmaestro.pipeline.layout import FAILURES_DIR, report_dir
from seqmaestro.pipeline.repository import REPO
from seqmaestro.target import BINARY

#: Where the dataset's files go: under the repository, as an agent writes them.
ROOT = REPO

#: How the command line records the model it ran on, where no model drafted anything.
NO_MODEL = "none"
#: A cell of the table's preview shows at most this many characters.
PREVIEW_WIDTH = 24
#: Below this share of the longest sequence's length, the shortest sequence's length makes a short
#: positional window, which the command says.
SHORT_WINDOW_SHARE = 0.5


class LaunchError(RuntimeError):
    """The estimate or the launch failed. The driver has said why, above."""


@dataclass(frozen=True)
class Options:
    """The answers given as options, each None where it was not given."""

    table: Path | None
    name: str | None
    sequence: str | None
    target: str | None
    positive: str | None
    group_by: list[str] | None
    anchor: str | None
    describe: str | None
    preset: str | None
    yes: bool


def _clip(value: str, width: int = PREVIEW_WIDTH) -> str:
    return value if len(value) <= width else value[:width - 1] + "…"


def _table_problem(answer: str) -> str | None:
    path = Path(answer).expanduser()
    if not path.is_file():
        return f"there is no file {answer}"
    return None


def _name_problem(name: str) -> str | None:
    problem = prepare.name_problem(name)
    if problem:
        return problem
    paths = Paths(name, ROOT)
    taken = paths.taken()
    if taken:
        return (f"{', '.join(paths.shown(path) for path in taken)} exist already, so choose another "
                f"name")
    return None


def _preview(frame) -> Table:
    table = Table(header_style="name", border_style="cyan")
    for column in frame.columns:
        table.add_column(escape(_clip(column)), no_wrap=True, overflow="ellipsis")
    for _, row in frame.head(3).iterrows():
        table.add_row(*(escape(_clip(str(value))) for value in row))
    return table


def _column_choices(frame, columns: list[str]) -> list[ui.Choice]:
    return [ui.Choice(column, f"{column}   [{_clip(str(frame[column].iloc[0]), 40)}]")
            for column in columns]


def _default_positive(values: tuple[str, ...]) -> str | None:
    """The positive class where the values say which it is: `1` where they are 0 and 1, and `true`
    where they are false and true."""
    lowered = tuple(value.lower() for value in values)
    for pair, positive in ((("0", "1"), "1"), (("false", "true"), "true")):
        if lowered == pair:
            return values[lowered.index(positive)]
    return None


def _split_table(rows: prepare.Rows, split) -> Table:
    table = Table(header_style="name", border_style="cyan")
    table.add_column("Split")
    table.add_column("Rows", justify="right")
    if rows.kind == BINARY:
        positive, negative = rows.values
        table.add_column(f"Positive ({escape(_clip(positive, 12))})", justify="right")
        table.add_column(f"Negative ({escape(_clip(negative, 12))})", justify="right")
    else:
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
    for name, count, *middle in prepare.split_summary(rows, split):
        table.add_row(name, f"{count:,}", *middle)
    return table


def _checks(record: dict[str, Any]) -> bool:
    """Show what the dataset's checks found, and return whether every blocking check passed."""
    checks = record["checks"]
    passed = [name for name, check in checks.items() if check["status"] in (PASSED, NOT_APPLICABLE)]
    ui.ok(f"{len(passed)} of {len(checks)} checks passed or did not apply.")
    for name, check in checks.items():
        if check["status"] == FLAGGED:
            ui.warn(f"[name]{name.replace('_', ' ')}[/]: {escape(check['detail'])}.")
            ui.hint("  This finding is for you to judge. It does not stop the run, and the report "
                    "states it.")
        elif check["status"] == FAILED:
            ui.fail(f"[name]{name.replace('_', ' ')}[/]: {escape(check['detail'])}.")
    ui.say(prepare.detection_floor(record))
    return not any(check["status"] == FAILED for check in checks.values())


def run(options: Options) -> int:
    """The steps of `seqmaestro run`. Returns the command's exit code."""
    questions = ui.Questions(options.yes)
    model.load_credentials()
    ui.console.print("[accent]SeqMaestro[/] [hint]from a table of sequences to a report. Each "
                     "question has an option: seqmaestro run --help[/]")
    table, frame, paths = the_table(questions, options)
    sequence, target, positive, groups = the_columns(questions, options, frame)
    rows, window, split = the_rows(questions, options, frame, sequence, target, positive, groups)

    ui.heading(4, "What the models are told")
    described, drafted_by = describe.settle(questions, describe.Facts.of(table, rows),
                                            options.describe)

    ui.heading(5, "The dataset")
    record = prepare.write(table, paths, rows, split, window, described)
    for path in (paths.data, paths.declaration, paths.description):
        ui.hint(f"  wrote {paths.shown(path)}")
    if not _checks(record):
        ui.fail(f"The dataset cannot be run as it stands. Its record is "
                f"{paths.shown(paths.splits / DATASET_FILE)}.")
        return 1

    run_dir = the_run(questions, options, table, paths, rows, window)
    if run_dir is None:
        return 0
    if not launch.record(run_dir, drafted_by or NO_MODEL, ["1", "2"]):
        ui.warn("The run did not record its config in time, so the command line is not named in it "
                "as the agent of Stages 1 and 2.")

    ui.heading(7, "Monitoring the run")
    seen = watch.follow(run_dir)
    if seen is None:
        ui.say(f"You stopped watching, and the run goes on. [name]seqmaestro status "
               f"{escape(run_dir.name)}[/] shows where it stands.")
        return 0
    return finished(run_dir, seen, questions)


def the_table(questions: ui.Questions, options: Options):
    """Step 1: the table, read, and the dataset's name."""
    ui.heading(1, "The table")
    table = Path(questions.text("Where is the table of sequences (a .csv or .tsv file)?",
                                option="TABLE", given=None if options.table is None
                                else str(options.table), check=_table_problem)).expanduser()
    frame = prepare.read_table(table)
    ui.say(f"[name]{escape(table.name)}[/]: {len(frame):,} rows, {len(frame.columns)} columns.")
    ui.console.print(_preview(frame))
    name = questions.text("What should the dataset be called?", option="--name", given=options.name,
                          default=prepare.name_for(table), check=_name_problem)
    return table, frame, Paths(name, ROOT)


def the_columns(questions: ui.Questions, options: Options, frame):
    """Step 2: the column of sequences, the target and its positive class, and the columns that
    group rows."""
    ui.heading(2, "The columns")
    ranked = prepare.sequence_columns(frame)
    sequence = questions.choose(
        "Which column holds the sequences?", _column_choices(frame, ranked), option="--sequence",
        given=options.sequence,
        suggested=ranked[0] if prepare.sequence_share(frame[ranked[0]]) > 0.9 else None)
    others = [column for column in frame.columns if column != sequence]
    column = questions.choose("Which column holds what was measured, the target?",
                              _column_choices(frame, others), option="--target", given=options.target)
    target = prepare.inspect_target(frame, column)
    positive = None
    if target.kind == BINARY:
        ui.say(f"`{escape(column)}` holds two values, so the target has two classes (binary).")
        given = None if options.positive is None else (target.value_of(options.positive)
                                                       or options.positive)
        positive = questions.choose(
            "Which value marks the positive class?",
            [ui.Choice(value, value) for value in target.values], option="--positive",
            given=given, default=_default_positive(target.values))
    else:
        ui.say(f"`{escape(column)}` holds numbers, so the target is a number (continuous).")
    groups = questions.choose_many(
        "Which columns group rows that the split must keep together, such as a gene or a region? "
        "Rows with identical sequences are kept together anyway.",
        _column_choices(frame, [other for other in others if other != column]),
        option="--group-by", given=options.group_by)
    return sequence, target, positive, groups


def the_rows(questions: ui.Questions, options: Options, frame, sequence: str,
             target: prepare.TargetColumn, positive: str | None, groups: list[str]):
    """Step 3: the rows kept, the positional window where the lengths differ, and the split."""
    ui.heading(3, "The rows and the split")
    rows = prepare.keep_rows(frame, sequence, target, positive, groups)
    before, after = rows.step["rows_in"], rows.step["rows_out"]
    dropped = (sum(before.values()) - sum(after.values()) if rows.kind == BINARY
               else before - after)
    ui.ok(f"{len(rows.frame):,} rows kept, with sequences of {rows.alphabet}.")
    if dropped:
        ui.warn(f"{dropped:,} rows dropped, since their sequence was missing or held a symbol other "
                f"than {prepare.spelled(prepare.ALPHABETS[rows.alphabet])}.")
    if rows.without_target:
        ui.warn(f"{rows.without_target:,} rows left out, since they have no usable value in "
                f"`{escape(target.column)}`.")
    shortest, longest = rows.lengths
    window = None
    if shortest == longest:
        ui.ok(f"Every sequence is {shortest} nucleotides long.")
    else:
        anchor = questions.choose(
            f"The sequences are {shortest} to {longest} nucleotides long. Positional features use "
            f"the {shortest} bases at one end of each. Which end?",
            [ui.Choice("start", f"The start: the first {shortest} bases"),
             ui.Choice("end", f"The end: the last {shortest} bases")],
            option="--anchor", given=options.anchor, default="start")
        window = prepare.positional_window(rows, anchor)
        if shortest < SHORT_WINDOW_SHARE * longest:
            ui.warn("The shortest sequence is less than half as long as the longest, so positional "
                    "features see little of most sequences. An agent can prepare such data with "
                    "you in other ways (a window around a landmark, or padding).")
        ui.say("Each k-mer count is divided by the number of positions in its sequence, so that it "
               "does not grow with the length.")
    split = prepare.assign_splits(rows)
    ui.console.print(_split_table(rows, split))
    return rows, window, split


def the_run(questions: ui.Questions, options: Options, table: Path, paths: Paths,
            rows: prepare.Rows, window) -> Path | None:
    """Step 6: the preset, the config, the estimate and the launch. Returns the run's folder, or
    None where the user chose not to start it."""
    ui.heading(6, "The run")
    preset = questions.choose(
        "Which run?", [ui.Choice(key, f"{key}: {contents} ({cost})")
                       for key, (contents, cost) in launch.PRESETS.items()],
        option="--preset", given=options.preset, default="quick")
    config = launch.write_config(paths, launch.config_document(paths, table, rows.kind, preset,
                                                               window is not None))
    ui.hint(f"  wrote {paths.shown(config)}")
    load(config)
    run_dir = launch.run_directory(paths.name)
    ui.say("The estimate fits each arm a few times on your data, which takes a minute or more.")
    if launch.estimate(config, run_dir):
        raise LaunchError("The estimate failed, as it says above.")
    if not questions.confirm("Start the run?", default=True):
        later = " ".join(launch.driver(config, run_dir, "--detach"))
        ui.say(f"Not started. To start it later: [name]{escape(later)}[/]")
        return None
    if launch.start(config, run_dir):
        raise LaunchError("The run did not start, as the driver says above.")
    return run_dir


def finished(run_dir: Path, seen: dict[str, Any], questions: ui.Questions) -> int:
    """What the command says and does when the run has ended: the report, where the run finished
    clean and a language model is configured. Returns the command's exit code."""
    overall = seen["overall"]
    if overall["failures"] or overall["state"]["kind"] == "failed":
        ui.fail(f"The run stopped with failures. What failed, with its traceback, is in "
                f"{ui.shown(run_dir / FAILURES_DIR)}.")
        return 1
    if not seen["report"]["inputs"]:
        ui.warn("The run stopped before its report's tables and figures were built.")
        return 1
    ui.ok(f"The run has finished. Its report's tables and figures are in "
          f"{ui.shown(report_dir(run_dir))}.")
    if model.missing():
        ui.say(f"With a language model configured in .env, [name]seqmaestro report "
               f"{escape(run_dir.name)}[/] writes its report. An agent can write it too.")
        return 0
    ui.heading(8, "The report")
    return report.write(run_dir, questions, again=False)
