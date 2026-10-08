"""The reliability panel, drawn.

There are three figures.

**The evidence grid** puts the three tiers side by side for every feature a model reported, so the
common cases are visible at a glance: strong on all three, strong on selection but absent from the
held-out rows, or reached only once.

**The threshold sweep** addresses the most common misunderstanding of the panel. Calling a feature
"reproducible" needs a cut-off, and no single cut-off is the right one, so the count is shown at
every cut-off from a bare majority up to unanimity. A finding that survives the whole sweep is
stronger than one that survives only at the loosest cut-off.

**The concept map** shows ideas instead of individual columns. This pipeline is built to surface
ideas that several configurations reached under different names, and the map shows which
configuration reached which idea, beside the evidence attached to each idea.

Each draws from what `seqmaestro.report.figure_inputs` wrote for it; nothing here reads a run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seqmaestro.report.figures import style


#: How a q-value or a selection frequency is turned into a shade. In both, stronger evidence is
#: darker, so each is mapped onto [0, 1] before the ramp sees it.
def _shade_frequency(value: float | None) -> str:
    return style.MIDPOINT if value is None else style.ramp(float(value))


def _shade_q(value: float | None) -> str:
    if value is None:
        return style.MIDPOINT
    value = float(value)
    if value <= 0:
        return style.SEQUENTIAL[-1]
    # Four decades of q, so 1e-4 and below is shaded at full strength and 1 at none.
    return style.ramp(min(1.0, max(0.0, -np.log10(value) / 4.0)))


def _shade_association(value: float | None, centre: float, full: float) -> str:
    """How far a feature goes with the target on its own, shaded on its distance from `centre`, the
    value that means no association, reaching full strength `full` away from it.

    A value below the centre belongs to a feature that runs the other way, and it is as strong as
    its mirror image above, so the shade follows the distance alone.
    """
    if value is None:
        return style.MIDPOINT
    return style.ramp(min(1.0, max(0.0, abs(float(value) - centre) / full)))


def _association_text(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _text_colour(shade: str) -> str:
    """Ink that stays legible on the ramp, which runs from a light blue to a dark one."""
    return "#ffffff" if shade in set(style.SEQUENTIAL[4:]) else style.INK


def draw_evidence_grid(
    panels: list[dict],
    path: Path,
    columns: int,
    title: str,
) -> Path:
    """One panel per configuration, one row per reported feature and three columns of evidence.

    Each panel carries `label`, `subtitle` and `features`, where a feature carries `feature`,
    `cluster_pi_hat`, `q_bh` and `p_holm`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    style.apply()
    columns = min(columns, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    figure = plt.figure(figsize=(5.5 * columns, 3.3 * rows))
    grid = GridSpec(rows, columns, figure=figure, hspace=0.55, wspace=0.85)

    headings = ["would it be\nchosen again", "does it hold on\nheld-out rows", "does the model\nneed it"]
    for index, panel in enumerate(panels):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        features = panel["features"]
        axis.set_xlim(0, 3)
        axis.set_ylim(0, len(features))
        axis.set_xticks([0.5, 1.5, 2.5])
        axis.set_xticklabels(headings, fontsize=7.2, color=style.INK_SECONDARY)
        axis.set_yticks([len(features) - i - 0.5 for i in range(len(features))])
        axis.set_yticklabels(
            [style.shorten(f["feature"], 30) for f in features], fontsize=7.2, color=style.INK
        )
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.xaxis.set_ticks_position("top")
        axis.xaxis.set_label_position("top")

        for row, entry in enumerate(features):
            y = len(features) - row - 1
            cells = [
                (_shade_frequency(entry.get("cluster_pi_hat")), _frequency_text(entry)),
                (_shade_q(entry.get("q_bh")), _q_text(entry.get("q_bh"))),
                (_shade_q(entry.get("p_holm")), _q_text(entry.get("p_holm"))),
            ]
            for column, (shade, text) in enumerate(cells):
                axis.add_patch(
                    plt.Rectangle(
                        (column + 0.03, y + 0.08), 0.94, 0.84,
                        facecolor=shade, edgecolor=style.SURFACE, linewidth=1.2,
                    )
                )
                axis.text(
                    column + 0.5, y + 0.5, text, ha="center", va="center",
                    fontsize=7.2, color=_text_colour(shade),
                )
        axis.set_title(f"{panel['label']}\n{panel['subtitle']}", fontsize=8.6,
                       color=style.INK, pad=26, loc="left")

    for index in range(len(panels), rows * columns):
        figure.add_subplot(grid[index // columns, index % columns]).axis("off")

    figure.suptitle(
        style.caption(
            title,
            "Darker is stronger evidence. The left column is how often the feature was picked again "
            "on a different half of the data, the middle is whether it goes with the target on its "
            "own on held-out sequences, and the right is whether the model predicts worse without "
            "it.",
            width=48 * columns,
        ),
        fontsize=10, color=style.INK, y=0.995, va="top",
    )
    lines = figure._suptitle.get_text().count("\n") + 1
    figure.subplots_adjust(top=1 - (0.75 + 0.2 * lines) / figure.get_figheight(), bottom=0.03,
                           left=0.11 * 3 / max(columns, 1), right=0.99)
    return style.save(figure, path)


def _frequency_text(entry: dict) -> str:
    value = entry.get("cluster_pi_hat")
    return "-" if value is None else f"{float(value):.2f}"


def _q_text(value) -> str:
    """A q-value to the three decimals the tables give it. At two, every value from 0.001 to 0.005
    would be printed as .00, which looks stronger than <.001."""
    if value is None:
        return "-"
    value = float(value)
    if value < 0.001:
        return "<.001"
    if value >= 0.9995:
        return "1.00"
    return f"{value:.3f}".lstrip("0")


def draw_threshold_sweep(
    rows: list[dict],
    path: Path,
    n_reported: int,
    n_word: str,
    title: str,
) -> Path:
    """How many features survive at every cut-off, drawn as one grid.

    One row per configuration, one column per cut-off. The large number in a cell is how many of
    the features the configuration reported clear that cut-off. `n_reported` is the most a
    configuration reports, the run's budget, and sets the scale of the shade. The small number
    beneath it is how many of all the candidates it could have used clear it, near-duplicates
    counted as one, which shows whether the pool held anything stable for the selection to find. A
    cell is ringed when that pool count is above the bound on how many candidates the cut-off would
    admit by chance.

    The three quantities share one grid, instead of three panels side by side, because a reader
    compares them cell by cell and separate panels would spread that comparison across a page.

    Each row needs `label`, `taus`, `reported`, `pool` and `bound`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style.apply()
    taus = rows[0]["taus"]
    figure, axis = plt.subplots(figsize=(0.62 * len(taus) + 3.6, 0.66 * len(rows) + 2.9))
    axis.set_xlim(0, len(taus))
    axis.set_ylim(0, len(rows))

    for index, row in enumerate(rows):
        y = len(rows) - index - 1
        for column in range(len(taus)):
            reported = row["reported"][column]
            pool = row["pool"][column]
            bound = row["bound"][column]
            shade = style.ramp(reported / n_reported)
            axis.add_patch(
                plt.Rectangle(
                    (column + 0.03, y + 0.07), 0.94, 0.86,
                    facecolor=shade, edgecolor=style.SURFACE, linewidth=1.2, zorder=2,
                )
            )
            # A bound at or above the number the cut-off selected constrains nothing. That happens
            # whenever the candidate pool is only a small multiple of the number selected, and those
            # cells get no ring, since a ring there would mislead.
            if pool > 0 and bound < pool:
                axis.add_patch(
                    plt.Rectangle(
                        (column + 0.03, y + 0.07), 0.94, 0.86, facecolor="none",
                        edgecolor=style.BASELINE, linewidth=1.6, zorder=4,
                    )
                )
            axis.text(column + 0.5, y + 0.58, f"{reported:.0f}", ha="center", va="center",
                      fontsize=8.4, color=_text_colour(shade), zorder=3)
            axis.text(column + 0.5, y + 0.25, f"{pool:,.0f}", ha="center", va="center",
                      fontsize=6.4, color=_text_colour(shade), alpha=0.85, zorder=3)

    axis.set_xticks([i + 0.5 for i in range(len(taus))])
    axis.set_xticklabels([f"{t:g}" for t in taus], fontsize=7.6)
    axis.set_yticks([len(rows) - i - 0.5 for i in range(len(rows))])
    axis.set_yticklabels([row["label"] for row in rows], fontsize=8.2, color=style.INK)
    axis.tick_params(length=0)
    axis.xaxis.set_ticks_position("top")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel("cut-off: the fraction of half-samples that chose the feature", fontsize=8.2)

    axis.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=style.SEQUENTIAL[3],
                   markersize=9, label=f"large number: of the features it reported, at most {n_word}"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=style.SEQUENTIAL[1],
                   markersize=9, label="small number: of every candidate it could have used, "
                                       "near-duplicates as one"),
            Line2D([0], [0], marker="s", color=style.BASELINE, markerfacecolor="none",
                   markersize=9, markeredgewidth=1.6, linestyle="none",
                   label="ringed: more than the cut-off admits by chance"),
        ],
        loc="upper left", bbox_to_anchor=(0.0, -0.06), fontsize=7.6, ncols=1,
    )

    figure.suptitle(
        style.caption(
            title,
            "Read a row left to right: the count falls as the cut-off tightens, and how fast it "
            "falls is the result.",
            width=78,
        ),
        fontsize=10, color=style.INK, y=0.995, va="top",
    )
    figure.subplots_adjust(top=1 - 1.05 / figure.get_figheight(),
                           bottom=1.5 / figure.get_figheight(), left=0.28, right=0.985)
    return style.save(figure, path)


def draw_concept_map(
    concepts: list[dict],
    columns: list[str],
    path: Path,
    legend: list[str],
    association: dict,
    n_word: str,
    title: str,
) -> Path:
    """Which configurations reached each idea, and what evidence the idea carries.

    `columns` names the configurations; each concept carries `label`, the indices of the columns it
    `reached`, its `direction`, and its `best_stability`, `strongest_association` and
    `best_association_q`. `legend` says what a dot of each direction means, positive first, and
    `association` names the association's column and gives its `centre` and the distance from it
    shaded at `full` strength.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style.apply()
    grid_columns = len(columns) + 4

    figure, axis = plt.subplots(figsize=(1.05 * grid_columns + 4.6, 0.44 * len(concepts) + 3.0))
    axis.set_xlim(0, grid_columns)
    axis.set_ylim(0, len(concepts))

    colours = {"positive": style.POSITIVE, "negative": style.NEGATIVE}
    for index, concept in enumerate(concepts):
        y = len(concepts) - index - 1
        if index % 2 == 0:
            axis.add_patch(plt.Rectangle((0, y), grid_columns, 1.0, facecolor=style.GRID, alpha=0.45,
                                         edgecolor="none", zorder=0))
        reached = set(concept["reached"])
        for column in range(len(columns)):
            if column in reached:
                axis.plot([column + 0.5], [y + 0.5], marker="o", markersize=8.5,
                          color=colours.get(concept["direction"], style.NEUTRAL), linestyle="none",
                          zorder=3)
            else:
                axis.plot([column + 0.5], [y + 0.5], marker=".", markersize=2.5, color=style.AXIS,
                          linestyle="none", zorder=2)

        # The association sits beside the q-value. On a large held-out set every q reaches the
        # floor the permutations allow, and the association is then the only one of the two that
        # still orders the ideas.
        cells = [
            (_shade_frequency(concept["best_stability"]),
             _frequency_text({"cluster_pi_hat": concept["best_stability"]})),
            (_shade_association(concept["strongest_association"], association["centre"],
                                association["full"]),
             _association_text(concept["strongest_association"])),
            (_shade_q(concept["best_association_q"]), _q_text(concept["best_association_q"])),
        ]
        for offset, (shade, text) in enumerate(cells):
            column = len(columns) + offset
            axis.add_patch(plt.Rectangle((column + 0.06, y + 0.12), 0.88, 0.76, facecolor=shade,
                                         edgecolor=style.SURFACE, linewidth=1.2, zorder=3))
            axis.text(column + 0.5, y + 0.5, text, ha="center", va="center", fontsize=7.2,
                      color=_text_colour(shade), zorder=4)
        axis.text(len(columns) + 3.1, y + 0.5, f"{len(reached)}", ha="center", va="center",
                  fontsize=8, color=style.INK, zorder=4)

    axis.set_yticks([len(concepts) - i - 0.5 for i in range(len(concepts))])
    axis.set_yticklabels([style.shorten(c["label"], 40) for c in concepts], fontsize=8, color=style.INK)
    axis.set_xticks([i + 0.5 for i in range(grid_columns)])
    axis.set_xticklabels(
        [*columns, "chosen\nagain", association["label"], "holds on\nheld-out",
         "how many\nreached it"],
        fontsize=7.2, color=style.INK_SECONDARY, rotation=42, ha="left")
    axis.xaxis.set_ticks_position("top")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color=style.POSITIVE, linestyle="none", markersize=8.5,
                   label=legend[0]),
            Line2D([0], [0], marker="o", color=style.NEGATIVE, linestyle="none", markersize=8.5,
                   label=legend[1]),
            Line2D([0], [0], marker="o", color=style.NEUTRAL, linestyle="none", markersize=8.5,
                   label="reached it; no agreed direction"),
        ],
        loc="upper left", bbox_to_anchor=(0.0, -0.02 / max(1, len(concepts)) - 0.01),
        fontsize=7.6, ncols=3,
    )

    figure.suptitle(
        style.caption(
            title,
            "One row per idea and one column per model configuration. A filled dot means that "
            f"configuration kept a feature computing this idea when it was cut to at most "
            f"{n_word}, whatever it called it.",
            width=104,
        ),
        fontsize=10, color=style.INK, y=0.995, va="top",
    )
    figure.subplots_adjust(top=1 - 2.0 / figure.get_figheight(), bottom=1.0 / figure.get_figheight(),
                           left=0.30, right=0.99)
    return style.save(figure, path)
