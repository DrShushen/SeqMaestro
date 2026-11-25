"""Draw how well each model configuration did, beside the two comparisons that give its score
meaning.

Two comparisons decide whether a score means anything, and both are drawn beside the bar so a reader
does not have to hold them in mind. The first is the shuffled-label control: the same configuration
fitted on the same rows with the training labels shuffled, which shows what the bar would look like
if there were nothing to find. The second is the score the base composition of the sequence already
achieves, which a model has to exceed to have added anything.

Bars are coloured by where the features came from, since that is the contrast the design exists to
make. A configuration whose model is present only as a black-box reference is drawn as an outline:
it shows how much signal the data holds, and it is not interpreted as a statement about features.

Each row arrives with its `label`, its `value`, whether it is `interpretable` and whether its
features were written (`writes_features`). Nothing here knows what an arm is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seqmaestro.report.figures import style


def _bar_style(row: dict) -> dict:
    if not row["interpretable"]:
        return {"facecolor": "none", "edgecolor": style.INK_MUTED, "linewidth": 1.1}
    return {"facecolor": style.source_colour(row["writes_features"]), "edgecolor": "none",
            "linewidth": 0}


def draw_grid(rows: list[dict], path: Path, baselines: list[dict], title: str, subtitle: str,
              metric_label: str, chance: float) -> Path:
    """One bar per configuration, with its shuffled-label control and any interval it has.

    A row's `control`, `low` and `high` are drawn when it has them. Bars start at `chance`, the
    metric's value for a model that knows nothing, and run left of it for a score below chance, as
    an R² can be.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    style.apply()
    rows = sorted(rows, key=lambda row: row["value"])
    positions = np.arange(len(rows))
    floor = chance

    figure, axis = plt.subplots(figsize=(9.2, 0.40 * len(rows) + 2.5))
    drew_control = drew_interval = False
    for position, row in zip(positions, rows, strict=True):
        axis.barh(position, row["value"] - floor, left=floor, height=0.62, zorder=3, **_bar_style(row))
        label_at = row["value"]
        if row.get("low") is not None and row.get("high") is not None:
            axis.plot([row["low"], row["high"]], [position, position], color=style.INK_SECONDARY,
                      linewidth=1.0, zorder=4, solid_capstyle="butt")
            for end in (row["low"], row["high"]):
                axis.plot([end, end], [position - 0.13, position + 0.13],
                          color=style.INK_SECONDARY, linewidth=1.0, zorder=4)
            label_at = max(label_at, row["high"])
            drew_interval = True
        if row.get("control") is not None:
            axis.plot([row["control"]], [position], marker="|", markersize=11, color=style.INK,
                      markeredgewidth=1.6, zorder=5)
            drew_control = True
        # The label goes past whatever the row drew furthest out, so that it never overlaps a
        # whisker.
        if row["value"] >= floor:
            axis.text(label_at + 0.007, position, f"{row['value']:.3f}", va="center", ha="left",
                      fontsize=7.8, color=style.INK_SECONDARY, zorder=6)
        else:
            leftmost = min(value for value in (row["value"], row.get("low"), row.get("control"))
                           if value is not None)
            axis.text(leftmost - 0.007, position, f"{row['value']:.3f}", va="center", ha="right",
                      fontsize=7.8, color=style.INK_SECONDARY, zorder=6)

    # One line per reference, with the labels stacked above the top bar so that two close values do
    # not print over each other or over the data.
    for index, mark in enumerate(baselines):
        axis.axvline(mark["value"], color=style.BASELINE, linewidth=1.6, zorder=2,
                     linestyle=mark.get("linestyle", "-"))
        axis.text(mark["value"], len(rows) - 0.4 + (len(baselines) - 1 - index) * 0.8,
                  f"  {mark['label']} {mark['value']:.3f}", fontsize=8, color=style.BASELINE,
                  va="bottom", ha="left")
    if baselines:
        axis.set_ylim(-0.7, len(rows) - 0.4 + 0.8 * (len(baselines) - 1) + 0.55)

    axis.axvline(floor, color=style.AXIS, linewidth=1.0, zorder=2)
    axis.set_yticks(positions)
    axis.set_yticklabels([row["label"] for row in rows], fontsize=8.4, color=style.INK)
    axis.set_xlabel(metric_label, fontsize=8.5)
    # The axis runs from chance, or from the lowest value drawn where one falls below it, to at
    # least 0.45 above chance, so a small score is shown as small.
    furthest = max(max(row["value"], row.get("high") or row["value"]) for row in rows)
    lowest = min([floor, *(value for row in rows for value in
                           (row["value"], row.get("low"), row.get("control")) if value is not None),
                  *(mark["value"] for mark in baselines)])
    right = max(floor + 0.45, furthest + 0.055)
    left = lowest - (0.012 if lowest == floor else 0.06)
    # A score below chance prints its value left of its leftmost mark, so the axis also reaches past
    # that label (about a tenth of the axis wide), or the label would run into the row names.
    labelled = [min(value for value in (row["value"], row.get("low"), row.get("control"))
                    if value is not None) - 0.007 for row in rows if row["value"] < floor]
    if labelled:
        left = min(left, min(labelled) - 0.1 * (right - left))
    axis.set_xlim(left, right)
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)

    # The legend lists only the marks that some row has.
    legend = []
    if any(row["interpretable"] and row["writes_features"] for row in rows):
        legend.append(Patch(facecolor=style.WRITTEN, label="features a language model wrote"))
    if any(row["interpretable"] and not row["writes_features"] for row in rows):
        legend.append(Patch(facecolor=style.ENUMERATED, label="features enumerated in advance"))
    if any(not row["interpretable"] for row in rows):
        legend.append(Patch(facecolor="none", edgecolor=style.INK_MUTED,
                            label="black-box reference model"))
    if drew_control:
        legend.append(Line2D([0], [0], marker="|", color=style.INK, linestyle="none",
                             markersize=11, markeredgewidth=1.6, label="same model, labels shuffled"))
    if drew_interval:
        legend.append(Line2D([0], [0], color=style.INK_SECONDARY, linewidth=1.0,
                             label="95% interval on the held-out sequences"))
    for mark in baselines:
        legend.append(Line2D([0], [0], color=style.BASELINE, linewidth=1.6,
                             linestyle=mark.get("linestyle", "-"), label=mark["legend"]))
    # The legend sits below the axes. How far the bars run depends on the dataset, so every corner
    # inside the axes holds data on some dataset.
    legend_columns = 2 if len(legend) <= 4 else 3
    legend_rows = int(np.ceil(len(legend) / legend_columns))
    figure.legend(handles=legend, loc="lower center", fontsize=7.8, ncols=legend_columns,
                  frameon=False)

    figure.suptitle(style.caption(title, subtitle, width=86), fontsize=10, color=style.INK,
                    y=0.995, va="top")
    figure.subplots_adjust(top=1 - (1.05 if subtitle else 0.75) / figure.get_figheight(),
                           bottom=(0.95 + 0.24 * legend_rows) / figure.get_figheight(),
                           left=0.27, right=0.985)
    return style.save(figure, path)


def draw_cost(rows: list[dict], path: Path, reduced_label: str, title: str, subtitle: str,
              metric_label: str, chance: float) -> Path:
    """What each configuration gave up when it was reduced to a few features.

    Each row has `full` and `reduced`. A short connector between the two makes the size of the drop
    what the eye measures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style.apply()
    rows = sorted(rows, key=lambda row: row["reduced"])
    positions = np.arange(len(rows))

    figure, axis = plt.subplots(figsize=(8.4, 0.38 * len(rows) + 2.2))
    for position, row in zip(positions, rows, strict=True):
        colour = (style.source_colour(row["writes_features"]) if row["interpretable"]
                  else style.INK_MUTED)
        axis.plot([row["full"], row["reduced"]], [position, position], color=style.AXIS,
                  linewidth=1.1, zorder=2)
        axis.plot([row["full"]], [position], marker="o", markersize=5.5,
                  markerfacecolor=style.SURFACE, markeredgecolor=colour, markeredgewidth=1.4,
                  linestyle="none", zorder=3)
        axis.plot([row["reduced"]], [position], marker="o", markersize=6.5, color=colour,
                  linestyle="none", zorder=4)
        axis.text(min(row["full"], row["reduced"]) - 0.008, position,
                  f"{row['reduced'] - row['full']:+.3f}", va="center", ha="right", fontsize=7.6,
                  color=style.INK_SECONDARY)

    # The quantity shown is the distance between the two circles, so the axis spans the values
    # instead of starting at chance. An axis anchored at chance would crowd every pair into one end
    # on a dataset whose models score well.
    values = [value for row in rows for value in (row["full"], row["reduced"])]
    lowest, highest = min(values), max(values)
    span = highest - lowest
    margin = max(0.02, span * 0.30)
    axis.set_xlim(lowest - margin, highest + max(0.006, span * 0.06))
    if lowest - margin <= chance:
        axis.axvline(chance, color=style.AXIS, linewidth=1.0, zorder=1)

    axis.set_yticks(positions)
    axis.set_yticklabels([row["label"] for row in rows], fontsize=8.4, color=style.INK)
    axis.set_xlabel(metric_label, fontsize=8.5)
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    figure.legend(
        handles=[Line2D([0], [0], marker="o", color=style.INK_MUTED, linestyle="none",
                        markerfacecolor=style.SURFACE, markersize=5.5,
                        label="every feature it selected"),
                 Line2D([0], [0], marker="o", color=style.INK_MUTED, linestyle="none",
                        markersize=6.5, label=reduced_label)],
        loc="lower center", fontsize=7.8, ncols=2, frameon=False)
    figure.suptitle(style.caption(title, subtitle, width=86), fontsize=10, color=style.INK,
                    y=0.995, va="top")
    figure.subplots_adjust(top=1 - (1.0 if subtitle else 0.72) / figure.get_figheight(),
                           bottom=1.14 / figure.get_figheight(), left=0.29, right=0.985)
    return style.save(figure, path)

