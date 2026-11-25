"""The task description and what each class means, drafted by the language model from the user's own
words, or written from the table where no model is configured.

Every model of the run is told the description (`experiments/datasets/README.md`), and the report
prints what each class means beside every direction, so the user sees the draft, and accepts it,
edits it or describes the data again. One call drafts all of it, from the user's few sentences and
what the command line knows of the table: its columns, the kind of target, the classes or the
target's range, the lengths and the alphabet. No row of the table is sent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import yaml
from rich.markup import escape
from rich.table import Table

from seqmaestro.cli import model, ui
from seqmaestro.cli.prepare import Described, Rows, spelled
from seqmaestro.target import BINARY

#: The tokens the draft may take, its reasoning included.
REPLY_TOKENS = 4000

SYSTEM = """\
You write the task description of a dataset for SeqMaestro, which looks for interpretable \
features of DNA or RNA sequence that predict a measurement. Every language model in the run is \
told this description, and the report prints it, so it states what was measured and what the \
sequences are, in the words of a domain expert, and nothing else: no hypothesis about which \
features matter, no instruction to a model, and nothing that the user's words and the facts below \
do not say. Where they leave something out, leave it out too.

Reply with one JSON object and nothing else, with these keys:
- "task": two to four sentences: what is predicted, from which sequence, what one row is, and \
what each class means or what the number measures.
- "sequence_layout": one to three sentences: the alphabet, the length, and what the positions \
mean, as far as the facts say. Positions are numbered from 0.
- "target_name": a short noun phrase for what is predicted, such as "binding of the factor".
{target_keys}"""

BINARY_KEYS = """\
- "classes": {"1": {"name": ..., "meaning": ...}, "0": {"name": ..., "meaning": ...}}: for each \
class, a name of one to three words and a meaning of one line."""

CONTINUOUS_KEYS = """\
- "target_meaning": what the number measures, in which units where they are known, and what a \
higher value means."""


class DescriptionError(ValueError):
    """A description that is incomplete, or a draft that could not be read."""


@dataclass(frozen=True)
class Facts:
    """What the command line knows of the table, from which the description is drafted or written."""

    table: str
    sequence: str
    target: str
    kind: str
    alphabet: str
    #: The symbols the sequences hold.
    symbols: str
    lengths: tuple[int, int]
    rows: int
    #: For a binary target: the positive class's value and the other's, as the table writes them.
    values: tuple[str, str] | None
    #: For a continuous target: its lowest, median and highest value.
    spread: tuple[float, float, float] | None

    @classmethod
    def of(cls, table: Path, rows: Rows) -> Facts:
        target = rows.frame[rows.target]
        spread = (None if rows.kind == BINARY else
                  (float(target.min()), float(target.median()), float(target.max())))
        return cls(table.name, rows.sequence, rows.target, rows.kind, rows.alphabet, rows.symbols,
                   rows.lengths, len(rows.frame), rows.values, spread)

    @property
    def length_text(self) -> str:
        shortest, longest = self.lengths
        return (f"{shortest} nucleotides" if shortest == longest
                else f"{shortest} to {longest} nucleotides")

    def as_prompt(self) -> str:
        """The facts as the draft's request states them."""
        lines = [f"- {self.rows:,} rows from {self.table}, each with a sequence of {self.alphabet} "
                 f"of {self.length_text}, in upper case {spelled(self.symbols)}, in the column "
                 f"`{self.sequence}`."]
        if self.kind == BINARY:
            positive, negative = self.values
            lines.append(f"- The target is the column `{self.target}`, which holds two values: the "
                         f"rows with `{positive}` are class 1, the positive class, and the rows "
                         f"with `{negative}` are class 0.")
        else:
            low, median, high = self.spread
            lines.append(f"- The target is the column `{self.target}`, a number for each row, from "
                         f"{low:g} to {high:g}, with a median of {median:g}.")
        return "\n".join(lines)


def class_name(facts: Facts, value: str) -> str:
    """A class's name where nobody gave one: its value, after the column's name where the value is
    a number, which on its own would say nothing."""
    try:
        float(value)
    except ValueError:
        return value
    return f"{facts.target} = {value}"


def plain(facts: Facts, words: str) -> Described:
    """The description written from the user's words, where there are some, and from the table."""
    shortest, longest = facts.lengths
    letters = spelled(facts.symbols)
    layout = (f"{shortest} nucleotides of {facts.alphabet} in upper case {letters}, read left to "
              f"right, with positions numbered from 0 to {shortest - 1}." if shortest == longest else
              f"Sequences of {facts.alphabet} of {shortest} to {longest} nucleotides, in upper case "
              f"{letters}, read left to right, with positions numbered from 0.")
    if facts.kind == BINARY:
        positive, negative = facts.values
        task = (f"Predicting from the sequence of {facts.alphabet} in each row whether its "
                f"`{facts.target}` is `{positive}` (the positive class) or `{negative}`.")
        classes = {label: (class_name(facts, value), f"rows whose {facts.target} is {value}")
                   for label, value in ((1, positive), (0, negative))}
        return Described(words or task, layout, facts.target, classes, None)
    task = f"Predicting the number in `{facts.target}` from the sequence of {facts.alphabet} in each row."
    return Described(words or task, layout, facts.target, None,
                     f"the value in the column {facts.target} of {facts.table}")


def checked(found: Any, facts: Facts) -> Described:
    """A description from a mapping (the model's reply, or the user's edit), with every part it
    needs, or `DescriptionError` naming what is missing."""
    if not isinstance(found, dict):
        raise DescriptionError("the description is not a set of named parts")

    def text(where: dict, key: str, label: str) -> str:
        value = where.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DescriptionError(f"{label} is missing")
        return " ".join(value.split())

    task = text(found, "task", "task")
    layout = text(found, "sequence_layout", "sequence_layout")
    target_name = text(found, "target_name", "target_name")
    if facts.kind != BINARY:
        return Described(task, layout, target_name, None,
                         text(found, "target_meaning", "target_meaning"))
    given = found.get("classes")
    if not isinstance(given, dict):
        raise DescriptionError("classes is missing")
    entries = {str(key).strip(): entry for key, entry in given.items()}
    classes = {}
    for label in (1, 0):
        entry = entries.get(str(label))
        if not isinstance(entry, dict):
            raise DescriptionError(f"class {label} is missing")
        classes[label] = (text(entry, "name", f"the name of class {label}"),
                          text(entry, "meaning", f"the meaning of class {label}"))
    if classes[1][0] == classes[0][0]:
        raise DescriptionError("the two classes have one name")
    return Described(task, layout, target_name, classes, None)


def draft(client, facts: Facts, words: str) -> Described:
    """The model's draft, from the user's words and the facts of the table."""
    system = SYSTEM.format(target_keys=BINARY_KEYS if facts.kind == BINARY else CONTINUOUS_KEYS)
    user = (f"The user's words about the data:\n{words}\n\nWhat the table holds:\n"
            f"{facts.as_prompt()}\n\nReply with the JSON object.")
    reply = client.complete(system, user, max_completion_tokens=REPLY_TOKENS)
    if not reply.ok:
        raise DescriptionError(f"the language model did not answer: {reply.error}")
    start, end = reply.text.find("{"), reply.text.rfind("}")
    try:
        found = json.loads(reply.text[start:end + 1]) if start >= 0 else None
    except ValueError as error:
        raise DescriptionError(f"the draft is not JSON: {error}") from error
    return checked(found, facts)


def as_text(described: Described) -> str:
    """The description as the user edits it: YAML, with a note at the top."""
    found: dict[str, Any] = {"task": described.task, "sequence_layout": described.sequence_layout,
                             "target_name": described.target_name}
    if described.classes:
        found["classes"] = {label: {"name": name, "meaning": meaning}
                            for label, (name, meaning) in described.classes.items()}
    else:
        found["target_meaning"] = described.target_meaning
    return ("# What every model of the run is told about the task, and what the report prints.\n"
            "# Edit it, save it and close the editor to go on.\n\n"
            + yaml.safe_dump(found, sort_keys=False, allow_unicode=True, width=88))


def shown(described: Described, facts: Facts) -> Table:
    """The description as a table of its parts."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="name", no_wrap=True)
    table.add_column()
    table.add_row("Task", escape(described.task))
    table.add_row("Sequences", escape(described.sequence_layout))
    table.add_row("Target", escape(described.target_name))
    if described.classes:
        positive, negative = facts.values
        for label, value in ((1, positive), (0, negative)):
            name, meaning = described.classes[label]
            heading = f"Class {label}" if value == str(label) else f"Class {label} ({escape(value)})"
            table.add_row(heading, f"{escape(name)}: {escape(meaning)}")
    else:
        table.add_row("It measures", escape(described.target_meaning))
    return table


def settle(questions: ui.Questions, facts: Facts, words: str | None) -> tuple[Described, str | None]:
    """The description the user accepts, and the model that drafted it, or None where none did.

    The model drafts where `.env` configures one and the user said something about the data.
    Without a terminal or under `--yes`, the first description is taken: the draft, or the plain
    one where no model is configured. A draft that fails stops the command there, since the
    description would otherwise be one the user did not ask for.
    """
    unconfigured = model.missing()
    if unconfigured:
        ui.hint("No language model is configured in .env, so the description is written from your "
                "words and the table. ./install.sh --check says what .env lacks.")
    while True:
        words = questions.text(
            "Describe the data in a few sentences: what was measured, what one row is, and what "
            "the sequences are.", option="--describe", given=words, default="")
        drafted_by = None
        if words and not unconfigured:
            client = model.client()
            with ui.console.status(f"The language model ({escape(model.name(client))}) is drafting "
                                   f"the description"):
                try:
                    described = draft(client, facts, words)
                    drafted_by = model.name(client)
                except DescriptionError as error:
                    if questions.yes or not questions.interactive:
                        raise
                    ui.warn(f"The draft failed ({escape(str(error))}), so the description is "
                            f"written from your words and the table. Edit it as you need.")
                    described = plain(facts, words)
        else:
            described = plain(facts, words)
        while True:
            ui.panel(shown(described, facts), "What the models are told, and the report says")
            if questions.yes or not questions.interactive:
                return described, drafted_by
            action = questions.choose("Use this description?", [
                ui.Choice("use", "Use it"),
                ui.Choice("edit", "Edit it in your editor"),
                ui.Choice("again", "Describe the data again"),
            ], option=None, default="use")
            if action == "use":
                return described, drafted_by
            if action == "again":
                words = None
                break
            described = edited(described, facts)


def edited(described: Described, facts: Facts) -> Described:
    """The description after the user edited it in their editor, opened again until it reads."""
    text = as_text(described)
    while True:
        changed = click.edit(text, extension=".yaml")
        if changed is None:
            return described
        try:
            return checked(yaml.safe_load(changed), facts)
        except (DescriptionError, yaml.YAMLError) as error:
            ui.warn(f"The edited description cannot be used ({escape(str(error))}). It opens again.")
            text = changed
