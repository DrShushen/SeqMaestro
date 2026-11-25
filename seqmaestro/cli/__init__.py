"""SeqMaestro from the command line: `seqmaestro run`, `seqmaestro status` and `seqmaestro report`.

The command line is a shorter way through a run than the agent's. It prepares a dataset from one
table with fixed defaults, asks the user only what it cannot decide (which column is which, the
positive class, what the data is), starts the run and follows it, and calls the language model only
to draft the task description and to write the report. The agent's way (`process/`) decides each
step with the user and takes more kinds of data.

**It stands apart from the rest of the package**, which never imports it. It writes the files that
Stage 1 writes, where an agent writes them (`seqmaestro.cli.prepare`), reads a run through the
status page's gatherer (`seqmaestro.cli.watch`), and starts the driver and the report's scripts
through their command lines, as a user would type them.

    seqmaestro run [TABLE]     prepare a table, start its run and follow it
    seqmaestro status [RUN]    where a run stands
    seqmaestro report RUN      write the report of a finished run
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from seqmaestro.cli import ui

app = typer.Typer(
    name="seqmaestro", add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
    help="SeqMaestro finds interpretable features of DNA and RNA sequence that predict what was "
         "measured for each sequence. These commands take a table of sequences through a run to "
         "its report. With an agent instead, start ./agent.sh in the repository's folder.")


class Anchor(StrEnum):
    start = "start"
    end = "end"


class Preset(StrEnum):
    quick = "quick"
    full = "full"


def _finish(work) -> None:
    """Run a command's work, show what stopped it, and exit with its code."""
    from seqmaestro.cli.describe import DescriptionError
    from seqmaestro.cli.prepare import PrepareError
    from seqmaestro.cli.report import ReportError
    from seqmaestro.cli.run import LaunchError
    from seqmaestro.pipeline.config import ConfigError

    try:
        code = work()
    except ui.Unanswered as error:
        ui.fail(str(error))
        raise typer.Exit(2) from None
    except (PrepareError, DescriptionError, ConfigError, LaunchError, ReportError,
            FileNotFoundError) as error:
        ui.fail(str(error))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        ui.console.print()
        ui.warn("Stopped. The files written so far are kept.")
        raise typer.Exit(130) from None
    raise typer.Exit(code)


@app.command("run")
def run_command(
    table: Annotated[Path | None, typer.Argument(
        metavar="TABLE", help="The table of sequences, as a .csv or .tsv file: a row for each "
                              "sequence, with what was measured for it.", show_default=False)] = None,
    name: Annotated[str | None, typer.Option(
        metavar="NAME", help="The dataset's name, which names its folder under data/ and its run. Default: the "
             "table's file name.", show_default=False)] = None,
    sequence: Annotated[str | None, typer.Option(
        metavar="COLUMN", help="The column of sequences.", show_default=False)] = None,
    target: Annotated[str | None, typer.Option(
        metavar="COLUMN", help="The column of what was measured: two values (two classes), or a number for each "
             "row.", show_default=False)] = None,
    positive: Annotated[str | None, typer.Option(
        metavar="VALUE", help="The value of the target that marks the positive class, for a target of two classes. "
             "Default with --yes: 1 where the values are 0 and 1.", show_default=False)] = None,
    group_by: Annotated[list[str] | None, typer.Option(
        "--group-by", metavar="COLUMN", help="A column whose rows belong together, such as a gene, kept on one side "
                           "of the split. Repeat it for more columns. Default: none.",
        show_default=False)] = None,
    anchor: Annotated[Anchor | None, typer.Option(
        help="The end of each sequence that positional features count from, where the sequences "
             "differ in length. Default: start.", show_default=False)] = None,
    describe: Annotated[str | None, typer.Option(
        metavar="TEXT", help="A few sentences about the data: what was measured, what one row is, what the "
             "sequences are. The language model drafts the task description from them.",
        show_default=False)] = None,
    preset: Annotated[Preset | None, typer.Option(
        help="quick: minutes, no language model in the run. full: every arm, hours, with the "
             "language model. Default: quick.", show_default=False)] = None,
    yes: Annotated[bool, typer.Option(
        "--yes", "-y", help="Take the default of every question that has one, and start the run "
                            "without asking.")] = False,
) -> None:
    """Prepare a table of sequences, start its run, and follow the run to its end.

    Asks for what it cannot decide, and takes each answer from its option where one is given.
    """
    from seqmaestro.cli import run

    options = run.Options(table=table, name=name, sequence=sequence, target=target,
                            positive=positive, group_by=group_by,
                            anchor=None if anchor is None else anchor.value, describe=describe,
                            preset=None if preset is None else preset.value, yes=yes)
    _finish(lambda: run.run(options))


@app.command("status")
def status_command(
    which: Annotated[str | None, typer.Argument(
        metavar="RUN", help="The run: its folder, or the folder's name under experiments/runs/. "
                            "Default: the run that changed last.", show_default=False)] = None,
) -> None:
    """Show where a run stands: each stage, what is running, and what failed."""
    from seqmaestro.cli import watch

    def work() -> int:
        run_dir = watch.find(which)
        step = watch.next_step(run_dir, watch.show(run_dir))
        if step:
            ui.say(step)
        return 0

    _finish(work)


@app.command("report")
def report_command(
    which: Annotated[str, typer.Argument(
        metavar="RUN", help="The run: its folder, or the folder's name under experiments/runs/.",
        show_default=False)],
    again: Annotated[bool, typer.Option(
        "--again", help="Write the report again, replacing the one written before.")] = False,
) -> None:
    """Write the report of a finished run, with the language model that .env names.

    The model writes the report's source against the template an agent follows, from the run's
    tables, figures and numbers. The source is then expanded into REPORT.md, and rendered as a PDF
    where pandoc and xelatex are installed.
    """
    from seqmaestro.cli import model, report, watch

    def work() -> int:
        model.load_credentials()
        return report.write(watch.find(which), ui.Questions(yes=False), again)

    _finish(work)
