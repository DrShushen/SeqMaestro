"""Expand a report source into the finished markdown.

A report source, `REPORT.src.md`, is prose with four kinds of placeholder:

    {{component:50_reliability_overview}}   a run-independent explanation, from docs/report/components/
    {{table:stage3b_summary}}                a rendered table, from the report's tables/
    {{figure:stage3a_tree}}                  a figure, with the caption and width its build recorded
    {{param:half_samples}}                   a setting the run recorded, from facts.json

**A component can have a twin per kind of target**, `NAME.binary.md` and `NAME.continuous.md`, where
what it explains differs between the kinds (a classification metric, the classes). The run's kind
chooses the twin, so a component's prose never switches on the kind inside itself. A component that
serves both kinds is one file, `NAME.md`.

Components and tables are substituted verbatim, so a number is never retyped. The writer of a report
chooses which table goes where and says what it means, and the numbers come from the file that
computed them. A parameter lets a run-independent explanation quote a number the run chose without
stating it as a constant.

Expansion runs twice over the text, so the placeholders a component brings in (usually parameters)
are filled too. **An unresolved placeholder raises** instead of reaching the reader as literal
braces, and so does any text still shaped like a placeholder after expansion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{(component|table|figure|param):([A-Za-z0-9_.+-]+)\}\}")

#: Anything that still looks like a placeholder once expansion is done: a misspelt kind, a stray
#: brace pair.
LEFTOVER = re.compile(r"\{\{[^{}]*\}\}")

#: HTML comments a component carries for its maintainers, stripped before the text reaches a reader.
COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)


class PlaceholderError(KeyError):
    """A placeholder with nothing to put in its place."""

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass
class Sources:
    """Where each kind of placeholder is resolved from."""

    components: Path
    tables: Path
    figures: dict[str, dict]
    parameters: dict[str, str]
    #: The run's kind of target, which chooses between a component's twins.
    target_kind: str
    used: list[str] = field(default_factory=list)


def figure_markdown(entry: dict) -> str:
    """One figure, as plain markdown with the caption and width its build recorded.

    Whether a figure needs a rotated page is a property of the page it is printed on, so the
    markdown says nothing about it. The figure build decides from the aspect it measured and
    records the answer in `figures.json`, which the PDF render follows.
    """
    return f"![{entry['caption']}]({entry['path']}){{width={entry['width']}}}"


def component_path(directory: Path, name: str, target_kind: str) -> Path:
    """A component's file: its twin for `target_kind` where it has twins, `NAME.md` otherwise."""
    twin = directory / f"{name}.{target_kind}.md"
    return twin if twin.exists() else directory / f"{name}.md"


def _resolve(kind: str, name: str, sources: Sources) -> str:
    if kind == "param":
        if name not in sources.parameters:
            raise PlaceholderError(f"parameter {name!r} unknown; facts.json has "
                                   f"{sorted(sources.parameters)}")
        return str(sources.parameters[name])
    if kind == "figure":
        if name not in sources.figures:
            raise PlaceholderError(f"figure {name!r} was not built; figures.json has "
                                   f"{sorted(sources.figures)}")
        return figure_markdown(sources.figures[name])
    path = (component_path(sources.components, name, sources.target_kind) if kind == "component"
            else sources.tables / f"{name}.md")
    if not path.exists():
        raise PlaceholderError(f"{kind} {name!r} not found at {path}")
    text = path.read_text()
    return COMMENT.sub("", text).strip() if kind == "component" else text.strip()


def expand(text: str, sources: Sources) -> str:
    """`text` with every placeholder filled, in two passes so a component's own are filled too."""
    def replace(match: re.Match) -> str:
        sources.used.append(f"{match.group(1)}:{match.group(2)}")
        return _resolve(match.group(1), match.group(2), sources)

    expanded = PLACEHOLDER.sub(replace, PLACEHOLDER.sub(replace, text))
    leftover = LEFTOVER.findall(expanded)
    if leftover:
        raise PlaceholderError(f"{len(leftover)} placeholder(s) could not be read, first "
                               f"{leftover[:3]}; the kinds are component, table, figure and param.")
    return expanded


def unused(sources: Sources) -> dict[str, list[str]]:
    """Tables and figures that exist and that nothing included, for the writer to check."""
    included = {kind: {entry.split(":", 1)[1] for entry in sources.used if entry.startswith(kind + ":")}
                for kind in ("table", "figure")}
    tables = {path.stem for path in sources.tables.glob("*.md")}
    return {"table": sorted(tables - included["table"]),
            "figure": sorted(set(sources.figures) - included["figure"])}
