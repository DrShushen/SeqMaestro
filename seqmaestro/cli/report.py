"""`seqmaestro report`: the report of a finished run, written by the language model.

An agent writes a run's report against `process/report-template.md`, from the run's facts, tables
and figures (`process/reporting.md`). Here one call to the language model that `.env` names writes
the report's source from the same template and inputs, and `experiments/analysis/assemble_report.py`
expands it into `REPORT.md`, putting in the table, figure, component or setting each placeholder
names. Every placeholder is checked first, and a source with one that names nothing, or with a
table, figure or component that would land inside a sentence, goes back to the model once, with the
list. The PDF is rendered where pandoc and xelatex are installed, and the run's status page is built
again.

The model is given the template; the names each kind of placeholder can take; the dataset's README;
the task description the run's language models were told; the run's facts without the ideas
(`concepts`), whose tables it has; every table as the report shows it; and each figure's caption.
The command records itself in the run as the agent of the report, with the model, so the report's
table of agents names the model that wrote it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from rich.markup import escape

from seqmaestro.cli import launch, model, ui
from seqmaestro.pipeline import agents
from seqmaestro.pipeline.config import load_recorded
from seqmaestro.pipeline.layout import report_dir
from seqmaestro.pipeline.repository import ANALYSIS, DESCRIPTIONS, REPO
from seqmaestro.report.files import (
    COMPONENTS_DIR,
    FACTS_FILE,
    FIGURES_FILE,
    REPORT_FILE,
    SOURCE_FILE,
    TABLES_DIR,
)
from seqmaestro.report.placeholders import LEFTOVER, PLACEHOLDER

#: The tokens the report's source may take, its reasoning included, and the seconds its reply is
#: waited for, which a reply of that length needs.
REPLY_TOKENS = 32000
REPLY_SECONDS = 900

#: The template an agent writes a report against, which the model follows too.
TEMPLATE = REPO / "process" / "report-template.md"

#: The parts of the facts the model is not given: the ideas, which reach it as their tables.
LEFT_OUT = ("concepts",)

#: The report's PDF, beside its markdown.
PDF_FILE = "REPORT.pdf"

SYSTEM = """\
You write the source of the report of one finished run of SeqMaestro, REPORT.src.md, which is \
expanded into the report its readers see. Follow the template below: its sections, their order, \
their headings and what each must contain, its style rules, and its rule that every number comes \
from an included table or figure.

You cannot read files, run commands or look at the PDF, so the template's steps that need them are \
done for you: its inputs are in the message that follows, and the source is expanded and rendered \
after you reply.

The title is the only first-level heading (#). Each section the template lists is a second-level \
heading (##), and anything within a section is third-level (###). Quote numbers in sentences where \
the template asks for them (the best model's held-out score, the base composition's, the detection \
floor, the counts of each tier), each copied from the facts or from a table you include and written \
as plain text, without quotation marks or backticks, and name in words the table or figure where a \
reader can check it.

How to write a placeholder:
- Each placeholder is replaced by the whole of what it names: {{table:NAME}} by the table, \
{{figure:NAME}} by the figure with its caption, and {{component:NAME}} by several paragraphs under \
their own heading. So write each of these on a line of its own, where the template says to include \
that table, figure or component, and never inside a sentence, a list item or a table.
- To point to a table or a figure from a sentence, name it in words, such as "the headline table" \
or "the figure of every configuration".
- {{param:NAME}} is replaced by a few words, such as "ten", so it stands inside a sentence as plain \
text: never in backticks, and never at the start of a sentence.
- Use only these names:

"""

CLOSING = """
Reply with the source alone, in markdown, starting with its title.

The template:

"""


class ReportError(RuntimeError):
    """A report that cannot be written. The message says why."""


def names(report: Path, facts: dict[str, Any]) -> dict[str, set[str]]:
    """The names each kind of placeholder can take in this run's report: the components (a
    component's twins share one name), the tables, the figures and the settings."""
    return {
        "component": {path.name.split(".")[0] for path in COMPONENTS_DIR.glob("*.md")},
        "table": {path.stem for path in (report / TABLES_DIR).glob("*.md")},
        "figure": set(json.loads((report / FIGURES_FILE).read_text())["figures"]),
        "param": set(facts["parameters"]),
    }


def problems(source: str, known: dict[str, set[str]]) -> list[str]:
    """What is wrong with the placeholders of `source`, a line for each, for the model: one that
    names nothing, a brace pair that is no placeholder, a table, figure or component that shares
    its line with text (it would land inside a sentence or a table), and a setting in backticks."""
    found = []
    for line in source.splitlines():
        for match in PLACEHOLDER.finditer(line):
            kind, name, text = match.group(1), match.group(2), match.group(0)
            if name not in known.get(kind, set()):
                found.append(f"{text} names no {kind} this report has")
            elif kind != "param" and line.strip() != text:
                found.append(f"{text} shares its line with other text, and it brings in the whole "
                             f"{kind}; put it on a line of its own, and name the {kind} in words "
                             f"where a sentence points to it")
            elif kind == "param" and f"`{text}`" in line:
                found.append(f"{text} is in backticks, which would set its words as code")
    found += [f"{text} is none of the four kinds of placeholder"
              for text in LEFTOVER.findall(PLACEHOLDER.sub("", source))]
    return list(dict.fromkeys(found))


def unfenced(text: str) -> str:
    """The reply without the code fence a model may put around a whole document."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip() + "\n"


def system_prompt(known: dict[str, set[str]]) -> str:
    listed = "\n".join(f"- {kind}s: {', '.join(sorted(found))}" for kind, found in known.items())
    return SYSTEM + listed + "\n" + CLOSING + TEMPLATE.read_text()


def request(run_dir: Path, report: Path, facts: dict[str, Any]) -> str:
    """The inputs, as the message to the model states them."""
    config = load_recorded(run_dir)
    readme = Path(facts["stage1"]["readme"])
    description = DESCRIPTIONS / f"{config.dataset.task}.json"
    figures = json.loads((report / FIGURES_FILE).read_text())["figures"]
    tables = sorted((report / TABLES_DIR).glob("*.md"))
    kept = {key: value for key, value in facts.items() if key not in LEFT_OUT}
    parts = [
        "The dataset's README:", readme.read_text() if readme.exists() else "(none)",
        "The task description the run's language models were told:",
        description.read_text() if description.exists() else "(none)",
        f"The note in the run's config on what the run is: {config.description or '(none)'}",
        "The run's facts, without the ideas (`concepts`), whose tables follow:",
        json.dumps(kept, separators=(",", ":")),
        "The tables, each as {{table:NAME}} brings it in:",
        *(f"### {path.stem}\n\n{path.read_text().strip()}" for path in tables),
        "The figures, each as {{figure:NAME}} brings it in, with its caption:",
        "\n".join(f"- {name}: {entry.get('caption', '')}" for name, entry in figures.items()),
    ]
    return "\n\n".join(parts)


def write(run_dir: Path, questions: ui.Questions, again: bool) -> int:
    """Write the report of the finished run in `run_dir`, expand it and render it. Returns the
    command's exit code."""
    report = report_dir(run_dir)
    if not (report / FACTS_FILE).exists() or not (report / FIGURES_FILE).exists():
        raise ReportError(f"{ui.shown(run_dir)} has no report's tables and figures yet, so the "
                          f"run has not finished.")
    source_path = report / SOURCE_FILE
    if source_path.exists() and not again and not questions.confirm(
            f"{ui.shown(source_path)} exists already. Write the report again, replacing it?",
            default=False, option="--again"):
        return 0
    unconfigured = model.missing()
    if unconfigured:
        raise ReportError("The report is written by the language model that .env names, and .env "
                          "configures none (./install.sh --check says what it lacks). An agent can "
                          "write the report instead (process/reporting.md).")
    facts = json.loads((report / FACTS_FILE).read_text())
    known = names(report, facts)
    system, user = system_prompt(known), request(run_dir, report, facts)
    client = model.client()
    name = model.name(client)
    source = _drafted(client, name, system, user, "The language model ({}) is writing the report")
    found = problems(source, known)
    if found:
        ui.warn(f"The draft has {len(found)} misused placeholder(s), so it goes back to the model "
                f"once.")
        retry = (f"{user}\n\nYour source, below, misuses these placeholders:\n"
                 + "\n".join(f"- {line}" for line in found)
                 + f"\n\nReply with the whole source again, corrected.\n\n{source}")
        source = _drafted(client, name, system, retry, "The language model ({}) is correcting it")
        found = problems(source, known)
    usage = client.usage()
    tokens = (f": {usage['prompt_tokens']:,} tokens in, {usage['completion_tokens']:,} out"
              if "prompt_tokens" in usage else "")
    ui.hint(f"  {usage['calls']} call(s) to the language model{tokens}.")
    report.joinpath(SOURCE_FILE).write_text(source)
    ui.hint(f"  wrote {ui.shown(source_path)}")
    if found:
        for line in found:
            ui.fail(escape(line))
        ui.fail(f"The source still misuses its placeholders. Correct it, then expand it with "
                f"{ui.shown(ANALYSIS / 'assemble_report.py')} --report-dir {ui.shown(report)}.")
        return 1
    agents.record(run_dir, launch.AGENT, name, ["report"])
    if launch.stream([sys.executable, ui.shown(ANALYSIS / "assemble_report.py"),
                      "--report-dir", ui.shown(report)]):
        raise ReportError("The source could not be expanded, as it says above.")
    written = [report / REPORT_FILE]
    if shutil.which("pandoc") and shutil.which("xelatex"):
        if launch.stream(["bash", ui.shown(ANALYSIS / "render_pdf.sh"),
                          ui.shown(report / REPORT_FILE)]):
            ui.warn("The PDF could not be rendered, as it says above. The report is in markdown.")
        else:
            written.append(report / PDF_FILE)
    else:
        ui.hint("  pandoc and xelatex are not both installed, so there is no PDF "
                "(docs/user/installation.md, the PDF tools).")
    launch.stream([sys.executable, ui.shown(ANALYSIS / "build_status_page.py"),
                   "--run-dir", ui.shown(run_dir)])
    ui.ok(f"The report is written: {' and '.join(ui.shown(path) for path in written)}.")
    return 0


def _drafted(client, name: str, system: str, user: str, saying: str) -> str:
    with ui.console.status(saying.format(escape(name))):
        reply = client.complete(system, user, max_completion_tokens=REPLY_TOKENS,
                                timeout=REPLY_SECONDS)
    if not reply.ok:
        raise ReportError(f"The language model did not answer: {reply.error}")
    return unfenced(reply.text)
