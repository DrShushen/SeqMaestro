"""Markdown tables, and the printed precision of every number a report shows.

Every table a report carries is rendered here and included verbatim. The writer of the report
chooses which tables to show and what to say about them, and never transcribes a number, because
transcription is where a report and its evidence come apart.

A cell holds one line. Markdown has no lists inside cells, and the PDF path silently drops the
workarounds that render on a web page. A number is written at the precision it is worth: three
decimals for a score, two for a frequency, and a q-value below a thousandth as `<0.001` instead of
an exponent a reader has to decode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: What a cell holds where the source recorded no number.
MISSING = "—"


def render(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    """A markdown pipe table. Cells are stringified as they arrive; format numbers first."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        row = list(row)
        if len(row) != len(headers):
            raise ValueError(f"Row has {len(row)} cells for {len(headers)} headers: {row}")
        cells = ["" if cell is None else str(cell) for cell in row]
        if any("\n" in cell for cell in cells):
            raise ValueError(f"A table cell holds one line: {row}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def score(value, places: int = 3) -> str:
    """A performance number."""
    return MISSING if value is None else f"{float(value):.{places}f}"


def signed(value, places: int = 3) -> str:
    """A change, always with its sign. One that rounds to zero is written `+0.000`, not `-0.000`."""
    if value is None:
        return MISSING
    rounded = round(float(value), places)
    return f"{rounded + 0.0:+.{places}f}"


def frequency(value) -> str:
    return MISSING if value is None else f"{float(value):.2f}"


def q_value(value) -> str:
    """A multiplicity-corrected p-value, written the way a reader compares two of them."""
    if value is None:
        return MISSING
    value = float(value)
    if value < 0.001:
        return "<0.001"
    if value >= 0.9995:
        return "1.00"
    return f"{value:.3f}".lstrip("0")


def count(value) -> str:
    return MISSING if value is None else f"{float(value):,.0f}"


def interval(low, high, places: int = 3) -> str:
    if low is None or high is None:
        return MISSING
    return f"{float(low):.{places}f} to {float(high):.{places}f}"


def percentage(value, places: int = 1) -> str:
    return MISSING if value is None else f"{float(value) * 100:.{places}f}%"


def clip(text: str, width: int = 190) -> str:
    """A description cut to something a table cell can hold.

    Some feature records carry a full implementation specification where a description was asked
    for. The first sentence is almost always the summary, so it is preferred, and a word boundary
    is the fallback.
    """
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    first = text.split(". ")[0].rstrip(".")
    if len(first) <= width:
        return first + "."
    return first[:width].rsplit(" ", 1)[0] + " ..."


#: Numbers up to twenty, spelled out as prose writes them. Above twenty a numeral is clearer.
WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


def word(value) -> str:
    """A count as prose writes it: spelled out while it is small, a numeral once it is not."""
    if value is None:
        return MISSING
    value = int(value)
    return WORDS.get(value, f"{value:,}")
