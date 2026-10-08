"""A symbolic model's equation, drawn: each feature's Shapley values, and the front it was chosen
from.

**The Shapley figure** has one panel per feature the equation names, strongest first. Each dot is
one training sequence the fit explained: the feature's value on the horizontal axis, and on the
vertical axis the feature's Shapley value there, its share of that sequence's prediction in the
model's scale (`scale`: log-odds for a binary target, the target's units for a continuous one).
The dots of one panel sum, with the other panels' dots for the same sequence, to that sequence's
prediction less the average prediction. A feature the equation uses alone draws a curve; a
feature that interacts with another draws a band, since its share depends on the other feature's
value too.

**The front** shows the equations the search found, one per complexity, with each one's loss on the
rows it searched and on the rows set aside, and marks the one chosen.

Drawn from `seqmaestro.report.internals.pysr_equation`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seqmaestro.report.figures import style


def draw_shapley(internals: dict, path: Path, note: str, columns: int, title: str) -> Path:
    """Every feature's Shapley values against its values, one panel each, strongest first. `note`
    says which way above and below the line point."""
    import matplotlib.pyplot as plt

    style.apply()
    features = internals["features"]
    columns = max(1, min(columns, len(features)))
    rows = max(1, int(np.ceil(len(features) / columns)))
    # At least two panels' width, so that an equation of one feature still has room for its title.
    wide = max(columns, 2)
    figure, axes = plt.subplots(rows, columns, figsize=(2.95 * wide, 2.7 * rows + 0.6),
                                squeeze=False)
    for panel, axis in enumerate(axes.ravel()):
        if panel >= len(features):
            axis.set_visible(False)
            continue
        feature = features[panel]
        values = np.asarray(feature["values"], dtype=float)
        shapley = np.asarray(feature["shapley"], dtype=float)
        colours = np.where(shapley >= 0, style.POSITIVE, style.NEGATIVE)
        axis.axhline(0.0, color=style.AXIS, linewidth=0.8, zorder=1)
        axis.scatter(values, shapley, s=5, c=colours, alpha=0.45, linewidths=0, zorder=2)
        axis.set_title(f"{panel + 1}.  " + style.wrap_name(feature["name"], 30), fontsize=8,
                       color=style.INK, pad=4, loc="left")
        axis.tick_params(length=2, labelsize=7)
        if panel % columns == 0:
            axis.set_ylabel(internals["scale"], fontsize=8)

    figure.suptitle(
        style.caption(title, f"{note} Panels are ordered by the mean size of the values.",
                      width=32 * wide),
        fontsize=10, color=style.INK, y=0.995, va="top")
    lines = figure._suptitle.get_text().count("\n") + 1
    figure.subplots_adjust(top=1 - (0.55 + 0.2 * lines) / figure.get_figheight(), bottom=0.08,
                           left=0.07, right=0.985, hspace=0.55, wspace=0.34)
    return style.save(figure, path)


def draw_front(internals: dict, path: Path, title: str, note: str) -> Path:
    """Each front equation's loss on the rows searched and on the rows set aside, by complexity,
    with the chosen equation, the one lowest on the rows set aside, filled."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    style.apply()
    front = internals["front"]
    complexity = np.array([entry["complexity"] for entry in front], dtype=float)
    searched = np.array([entry["searched_loss"] for entry in front], dtype=float)
    aside = np.array([np.nan if entry["set_aside_loss"] is None else entry["set_aside_loss"]
                      for entry in front], dtype=float)
    chosen = [index for index, entry in enumerate(front) if entry["chosen"]]

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(complexity, searched, color=style.NEUTRAL, marker="o", markersize=3.5, linewidth=1.2,
              label="rows searched", zorder=2)
    defined = np.isfinite(aside)
    axis.plot(complexity[defined], aside[defined], color=style.POSITIVE, marker="o", markersize=3.5,
              linewidth=1.4, label="rows set aside", zorder=3)
    for index in chosen:
        axis.scatter([complexity[index]], [aside[index]], s=70, facecolors=style.POSITIVE,
                     edgecolors=style.INK, linewidths=1.0, zorder=4, label="chosen")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("complexity (nodes in the equation)", fontsize=8)
    axis.set_ylabel(internals["loss"], fontsize=8)
    axis.tick_params(length=2, labelsize=7)
    axis.legend(fontsize=7, frameon=False, loc="upper right")
    figure.suptitle(style.caption(title, note, width=95), fontsize=10, color=style.INK, y=0.995,
                    va="top")
    lines = figure._suptitle.get_text().count("\n") + 1
    figure.subplots_adjust(top=1 - (0.45 + 0.2 * lines) / figure.get_figheight(), bottom=0.14,
                           left=0.11, right=0.98)
    return style.save(figure, path)
