"""Markdown tables, and the printed precision of every number a report shows.

Every table a report includes is rendered here and included verbatim. The writer of the report
chooses which tables to show and what to say about them, and never transcribes a number, because
transcription is where a report and its evidence come apart.

**A cell holds one line**, because markdown has no lists inside cells, and the conversion to PDF
silently drops the workarounds that render on a web page. A number is written at the precision it
merits: three decimals for a score, two for a frequency, and a q-value below a thousandth as
`<0.001` instead of an exponent a reader has to decode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: What a cell holds where the source recorded no number.
MISSING = "—"


def render(headers: Sequence[str], rows: Iterable[Sequence],
           widths: Sequence[int] | None = None) -> str:
    """A markdown pipe table. Cells are converted to strings as they arrive, so format numbers
    first.

    `widths` gives the columns' relative widths, as the numbers of dashes under the headers. Pandoc
    reads them when a table is wider than the page and wraps its cells, so a column of prose can be
    given most of the width. Without `widths`, every column has the same width."""
    if widths is None:
        widths = [3] * len(headers)
    if len(widths) != len(headers) or any(width < 1 for width in widths):
        raise ValueError(f"A table of {len(headers)} columns needs as many positive widths, "
                         f"not {list(widths)}.")
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("-" * width for width in widths) + "|"]
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


def prose(text: str) -> str:
    """Text from a record, such as a feature's description, in full as one table cell.

    A description is kept whole, since the part that says what a feature computes can come in any
    of its sentences: a description DEFT writes is a specification, which often opens with the
    stretch of sequence it measures and only then says what it counts there. Its whitespace is
    collapsed, since a cell holds one line, and a pipe is escaped, since an unescaped one would end
    the cell."""
    return " ".join((text or "").split()).replace("|", "\\|")


#: Numbers up to twenty, spelled out as prose writes them. Above twenty a numeral is clearer.
WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


def series(items: Sequence[str]) -> str:
    """Items as prose: `a`, `a and b`, `a, b and c`."""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def word(value) -> str:
    """A count as prose writes it: spelled out while it is small, a numeral once it is not."""
    if value is None:
        return MISSING
    value = int(value)
    return WORDS.get(value, f"{value:,}")
