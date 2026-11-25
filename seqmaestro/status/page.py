"""Render a run's status page from what `seqmaestro.status.gather` found, and write it.

The page is one HTML file with its stylesheet inlined, so that it opens from the run's folder in any
browser and still works when the folder is copied elsewhere. Its links are relative to the run's
root, where the page sits. The layout is the template `templates/page.html`, its look is
`templates/page.css`, and the filters below format the numbers it shows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

#: Where the template and the stylesheet are.
TEMPLATES = Path(__file__).resolve().parent / "templates"

#: How often an open page reloads itself while the driver runs, in seconds.
RELOAD_SECONDS = 30

#: What a number the run has not produced yet is shown as.
MISSING = "\N{EN DASH}"


def _score(value: float | None, digits: int = 3) -> str:
    return MISSING if value is None else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int = 3) -> str:
    return MISSING if value is None else f"{value:+.{digits}f}"


def _share(value: float | None) -> str:
    return MISSING if value is None else f"{100 * value:.0f}%"


def _hours(value: float | None) -> str:
    if value is None:
        return MISSING
    return f"{value:.1f} h" if value >= 1 else f"{60 * value:.0f} min"


def _clock(value: str | None) -> str:
    """An ISO time as the day and the time of day, in the zone it was recorded in."""
    return MISSING if not value else datetime.fromisoformat(value).strftime("%d %b %H:%M")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _environment() -> Environment:
    environment = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(),
                              undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    environment.filters.update(score=_score, signed=_signed, share=_share, hours=_hours,
                               clock=_clock, count=_count)
    return environment


def render(gathered: dict[str, Any]) -> str:
    """The page's HTML."""
    css = (TEMPLATES / "page.css").read_text()
    return _environment().get_template("page.html").render(
        css=css, reload_seconds=RELOAD_SECONDS, **gathered)


def write(path: Path, html: str) -> Path:
    """Write the page through one rename, so that a browser reloading it never finds it half
    written."""
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(html)
    partial.replace(path)
    return path
