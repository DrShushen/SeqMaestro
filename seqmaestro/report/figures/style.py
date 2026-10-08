"""One palette and one set of defaults, so the figures of a report look like one system.

The colours are the validated light-mode reference palette from the project's data-visualisation
guidance, checked with its validator. Three of the choices encode meaning:

**Two categorical hues carry the project's central contrast**: features a language model wrote
against features enumerated in advance. A third hue marks the simple reference baselines. A palette
of three clears every colour-vision gate on the all-pairs test, which one hue for each feature
source would not. How the features arose is also the split that carries meaning.

**A diverging blue-red pair carries polarity**: blue for the positive class of a binary target, or a
higher value of a continuous one, and red for the other, with a grey midpoint at zero. Both poles
are labelled wherever they appear, with the class names or with what a higher value does, so no
reader has to remember which is which.

**A single blue ramp carries magnitude.** Light means near nothing, dark means strong.

Figures are rendered for print at one column width, so nothing here reads a viewer theme.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# -- surfaces and ink ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# -- categorical: how the features arose --------------------------------------------------------
WRITTEN = "#2a78d6"
ENUMERATED = "#eb6834"
BASELINE = "#1baf7a"

# -- diverging: which class a value argues for, or which way it moves the target ----------------
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
MIDPOINT = "#f0efec"

# -- sequential: magnitude ----------------------------------------------------------------------
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: Neutral fill for a mark that carries no polarity, e.g. a density strip under a curve.
NEUTRAL = "#c3c2b7"


def apply() -> None:
    """Set the defaults every figure in a report is drawn with."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "text.color": INK,
        }
    )


def ramp(value: float) -> str:
    """A step of the sequential ramp for a value in [0, 1]."""
    if value != value:  # NaN
        return MIDPOINT
    index = min(len(SEQUENTIAL) - 1, max(0, round(value * (len(SEQUENTIAL) - 1))))
    return SEQUENTIAL[index]


def polarity(value: float, floor: float = 0.0) -> str:
    """Blue where a value points to the positive class or a higher target, red where it points the
    other way."""
    if value is None or value != value or abs(value) <= floor:
        return NEUTRAL
    return POSITIVE if value > 0 else NEGATIVE


def source_colour(writes_features: bool) -> str:
    return WRITTEN if writes_features else ENUMERATED


def caption(heading: str, body: str, width: int = 110) -> str:
    """A figure's heading and its explanatory line, wrapped.

    Saving with a tight bounding box grows the canvas to hold whatever is widest, so an unwrapped
    explanatory line sets the figure's width and leaves the panels stranded in the middle of it.
    Wrapping it keeps the canvas the size of the panels.
    """
    import textwrap

    lines = [heading, *textwrap.wrap(" ".join(body.split()), width)]
    return "\n".join(line for line in lines if line)


def shorten(name: str, width: int = 34) -> str:
    """A feature name that fits an axis, cut from the middle so both ends survive.

    Feature names in this project are long and their informative parts are at both ends, so
    truncating the tail alone leaves rows that all read alike.
    """
    if len(name) <= width:
        return name
    keep = (width - 1) // 2
    return f"{name[:keep]}…{name[-keep:]}"


def wrap_name(name: str, width: int, max_lines: int | None = None) -> str:
    """Wrap a feature name onto several lines, breaking after an underscore where one is available.

    `textwrap` treats an underscore as an ordinary character inside a word, so a long feature name
    is broken wherever the width happens to fall, leaving fragments like `..._windo` / `w`.
    Breaking after the underscore keeps every line a readable piece of the name. A single token
    longer than the width is still broken on a character, since there is nothing else to break on.
    """
    import textwrap

    pieces: list[str] = []
    for token in name.split("_"):
        pieces.append(token + "_")
    if pieces:
        pieces[-1] = pieces[-1][:-1]

    lines: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > width:
            lines.append(current)
            current = piece
        elif len(piece) > width and not current:
            wrapped = textwrap.wrap(piece, width)
            lines.extend(wrapped[:-1])
            current = wrapped[-1] if wrapped else ""
        else:
            current += piece
    if current:
        lines.append(current)
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(lines)


def save(figure, path) -> Path:
    """Write a figure and release it. Every figure is saved with the defaults `apply` set."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
