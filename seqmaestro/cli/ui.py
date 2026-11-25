"""The look of the command line, and the questions it asks.

Rich draws the headings, panels and tables, and Questionary the questions answered from a list.
**Every question goes through `Questions`**, which takes the answer from the question's option where
the option was given, takes the default under `--yes`, and asks otherwise, so a run can be scripted
as well as answered. Under `--yes`, a question without a default is refused, with the option that
answers it named, since such a question is about the data (which column holds the sequences, which
value is the positive class), and a guess would be one table's answer applied to the next.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import questionary
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.theme import Theme

from seqmaestro.pipeline.repository import REPO

#: The colours of the command line, by what they mark.
THEME = Theme({
    "accent": "bold cyan",
    "ok": "green",
    "warn": "yellow",
    "bad": "bold red",
    "hint": "dim",
    "name": "bold",
})

#: The same colours in Questionary's questions.
QUESTION_STYLE = questionary.Style([
    ("qmark", "fg:ansicyan bold"),
    ("question", "bold"),
    ("answer", "fg:ansigreen"),
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("selected", "fg:ansigreen"),
    ("instruction", "fg:ansibrightblack"),
])

console = Console(theme=THEME, highlight=False)


class Unanswered(Exception):
    """A question that cannot be asked here and was not answered by an option."""


def shown(path: Path) -> str:
    """A path as the command shows it, and as the commands it starts are given it: relative to the
    repository's folder, which they run in, where it is inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def heading(number: int | str, title: str) -> None:
    """The heading of one step of the command."""
    console.print()
    console.rule(f"[accent]{number}[/]  [name]{escape(title)}[/]", align="left", style="cyan")


def say(text: str) -> None:
    console.print(text)


def ok(text: str) -> None:
    console.print(f"[ok]✓[/] {text}")


def warn(text: str) -> None:
    console.print(f"[warn]![/] {text}")


def fail(text: str) -> None:
    console.print(f"[bad]✗[/] {text}")


def hint(text: str) -> None:
    console.print(f"[hint]{text}[/]")


def boxed(body, title: str, style: str = "cyan", subtitle: str | None = None) -> Panel:
    """Text or a table in a box with a title, and a note under it."""
    return Panel(body, title=f"[name]{escape(title)}[/]", title_align="left", border_style=style,
                 padding=(0, 1), subtitle=f"[hint]{escape(subtitle)}[/]" if subtitle else None,
                 subtitle_align="left")


def panel(body, title: str, style: str = "cyan") -> None:
    """Print text or a table in a box with a title."""
    console.print(boxed(body, title, style))


def output(line: str) -> None:
    """A line another command printed, shown as it was printed: neither markup nor wrapping, so that
    its tables keep their columns."""
    console.print(line, markup=False, highlight=False, soft_wrap=True)


@dataclass
class Choice:
    """One answer to a question with a list: the value returned, and the title shown."""

    value: str
    title: str


@dataclass
class Questions:
    """Asks the command's questions, or takes their answers from the options and defaults.

    `yes` is `--yes`: every question with a default takes it. `interactive` is whether the command
    runs in a terminal it can ask in; without one, a question no option answered is refused, as
    under `--yes`.
    """

    yes: bool
    interactive: bool = field(default_factory=lambda: sys.stdin.isatty() and sys.stdout.isatty())

    def _cannot_ask(self, question: str, option: str | None) -> Unanswered:
        reason = "--yes takes the defaults, and this question has none" if self.yes else \
            "there is no terminal to ask in"
        remedy = f"Give {option}." if option else "Run the command in a terminal."
        return Unanswered(f"{question} ({reason}). {remedy}")

    def choose(self, question: str, choices: Sequence[Choice], *, option: str | None,
               given: str | None = None, default: str | None = None,
               suggested: str | None = None) -> str:
        """One value from `choices`: `given` where an option gave it, else the default under
        `--yes`, else the user's choice. `suggested` is where the choice starts when the user is
        asked, which `--yes` never takes, since it is a guess about the data."""
        values = [choice.value for choice in choices]
        if given is not None:
            if given not in values:
                raise Unanswered(f"{option} is {given!r}, which is none of {', '.join(values)}.")
            return given
        if self.yes and default is not None:
            return default
        if self.yes or not self.interactive:
            raise self._cannot_ask(question, option)
        return questionary.select(
            question, choices=[questionary.Choice(choice.title, choice.value) for choice in choices],
            default=default or suggested, style=QUESTION_STYLE).unsafe_ask()

    def choose_many(self, question: str, choices: Sequence[Choice], *, option: str | None,
                    given: Sequence[str] | None = None, default: Sequence[str] = ()) -> list[str]:
        """Any number of values from `choices`, taken as `choose` takes one."""
        values = [choice.value for choice in choices]
        if given is not None:
            unknown = [value for value in given if value not in values]
            if unknown:
                raise Unanswered(f"{option} names {', '.join(unknown)}, which is none of "
                                 f"{', '.join(values)}.")
            return list(given)
        if self.yes:
            return list(default)
        if not self.interactive:
            raise self._cannot_ask(question, option)
        return questionary.checkbox(
            question, choices=[questionary.Choice(choice.title, choice.value,
                                                  checked=choice.value in default)
                               for choice in choices],
            instruction="(space to tick, enter when done)", style=QUESTION_STYLE).unsafe_ask()

    def confirm(self, question: str, *, default: bool, option: str | None = "--yes") -> bool:
        """Yes or no: the default under `--yes`, else the user's answer."""
        if self.yes:
            return default
        if not self.interactive:
            raise self._cannot_ask(question, option)
        return questionary.confirm(question, default=default, style=QUESTION_STYLE).unsafe_ask()

    def text(self, question: str, *, option: str | None, given: str | None = None,
             default: str | None = None, check: Callable[[str], str | None] | None = None) -> str:
        """A line of text. `check` returns what is wrong with an answer, or None for a good one."""
        if given is not None:
            problem = check(given) if check else None
            if problem:
                raise Unanswered(f"{option}: {problem}")
            return given
        if self.yes and default is not None:
            return default
        if self.yes or not self.interactive:
            raise self._cannot_ask(question, option)
        return questionary.text(
            question, default=default or "",
            validate=(lambda answer: check(answer.strip()) or True) if check else None,
            style=QUESTION_STYLE).unsafe_ask().strip()
