"""The shape functions of an additive model, drawn.

An additive model with no interaction terms is the sum of one curve per feature, so together these
panels are the whole model. A reader can add up the heights and get the score the model gives a
sequence, which makes this the figure to lead an interpretability section with.

Each panel shows one curve against the value of one feature, in the model's scale (`scale`: log-odds
for a binary target, the target's units for a continuous one), with the horizontal line at zero
meaning the feature moves the score neither way. A narrow strip under each panel shows how many of
the fitted sequences sit at each value, so a reader can tell a steep section the data supports from
one resting on a handful of rows.

Drawn from `seqmaestro.report.internals.ebm_shapes`: the terms, strongest first, each with its bin
boundaries, its score per bin and the fitted rows' values of its feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seqmaestro.report.figures import style

#: Bars in the density strip under each curve.
DENSITY_BINS = 26


def draw(internals: dict, path: Path, note: str, columns: int, title: str) -> Path:
    """Every shape function, one panel each, ordered by how much the curve moves. `note` says
    which way above and below the line point."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    style.apply()
    terms = internals["terms"]
    rows = int(np.ceil(len(terms) / columns))

    # One outer cell per feature, spaced so a two-line title never lands on the panel above it.
    # Inside each cell, the curve and its density strip are a tightly spaced nested pair.
    figure = plt.figure(figsize=(2.95 * columns, 3.05 * rows))
    outer = GridSpec(rows, columns, figure=figure, hspace=0.62, wspace=0.34)

    for panel, term in enumerate(terms):
        row, column = panel // columns, panel % columns
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, column], height_ratios=[5, 1],
                                        hspace=0.08)
        curve_axis = figure.add_subplot(inner[0])
        strip_axis = figure.add_subplot(inner[1], sharex=curve_axis)

        boundaries = np.asarray(term["boundaries"], dtype=float)
        steps = np.repeat(boundaries, 2)[1:-1]
        heights = np.repeat(np.asarray(term["scores"], dtype=float), 2)

        curve_axis.axhline(0.0, color=style.AXIS, linewidth=0.8, zorder=1)
        curve_axis.fill_between(steps, heights, 0.0, where=heights >= 0, color=style.POSITIVE,
                                alpha=0.26, linewidth=0, zorder=2)
        curve_axis.fill_between(steps, heights, 0.0, where=heights <= 0, color=style.NEGATIVE,
                                alpha=0.26, linewidth=0, zorder=2)
        curve_axis.plot(steps, heights, color=style.INK, linewidth=1.5, zorder=3)
        curve_axis.set_xlim(boundaries[0], boundaries[-1])
        curve_axis.tick_params(labelbottom=False, length=2)
        curve_axis.set_title(f"{panel + 1}.  " + style.wrap_name(term["name"], 30), fontsize=8,
                             color=style.INK, pad=4, loc="left")
        if column == 0:
            curve_axis.set_ylabel(internals["scale"], fontsize=8)

        counts, edges = np.histogram(np.asarray(term["fit_values"], dtype=float), bins=DENSITY_BINS,
                                     range=(boundaries[0], boundaries[-1]))
        if counts.max() > 0:
            strip_axis.bar(edges[:-1], counts / counts.max(), width=np.diff(edges), align="edge",
                           color=style.NEUTRAL, linewidth=0)
        strip_axis.set_ylim(0, 1.05)
        strip_axis.set_yticks([])
        strip_axis.spines["left"].set_visible(False)
        strip_axis.tick_params(length=2)
        if column == 0:
            strip_axis.set_ylabel("rows", fontsize=7, color=style.INK_MUTED, rotation=0, ha="right",
                                  va="center", labelpad=8)

    figure.suptitle(
        style.caption(title, f"{note} Panels are ordered by how much each curve moves.",
                      width=32 * columns),
        fontsize=10, color=style.INK, y=0.995, va="top")
    # The suptitle can run to several lines and each panel carries a two-line title. The space for
    # them is reserved in inches and converted to a fraction of the figure height, because a fixed
    # fraction would be right only at the figure height it was tuned at.
    lines = figure._suptitle.get_text().count("\n") + 1
    figure.subplots_adjust(top=1 - (0.55 + 0.2 * lines) / figure.get_figheight(), bottom=0.06,
                           left=0.055, right=0.985)
    return style.save(figure, path)
