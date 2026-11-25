"""Draw a fitted decision tree in the shape of a tree.

A tree is a set of yes/no questions asked in order, and a reader follows one sequence down it to see
what the tree predicts for it. The figure keeps that shape: one box per question, the two answers
below it, and the number of rows that reach each branch.

**Deep branches are folded instead of shrunk**, because a large tree drawn in full makes every label
too small to read. Below the drawn depth, a branch becomes one wedge showing how many rows reach it
and what they hold, so the figure shows how much of the model is off the page.

Each box is shaded on the diverging scale the rest of the report uses, so a branch that separates
the rows is visible before any number is read. For a binary target the shade follows the rate of
the positive class among the rows reaching the box, and for a continuous one how far their mean
sits from the mean of every fitting row, in those rows' standard deviations.

The figure is drawn from what `experiments/adapters/deft_cart.py tree` exports: the saved tree's
nodes, with the fitting rows routed down it, and the kind of target it was grown for
(`target_kind`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seqmaestro.report.figures import style


def _blend(colour: str, weight: float) -> tuple[float, float, float]:
    """`colour` mixed toward the neutral midpoint, where a weight of 0 gives the midpoint and 1 the
    pole."""
    import matplotlib.colors as mcolors

    pole = np.array(mcolors.to_rgb(colour))
    neutral = np.array(mcolors.to_rgb(style.MIDPOINT))
    return tuple(neutral + (pole - neutral) * float(np.clip(weight, 0.0, 1.0)))


#: How far a box's mean may sit from the mean of every fitting row, in their standard deviations,
#: before its shade is at full strength.
FULL_SHADE_SDS = 1.0


def _shade(rate: float | None) -> tuple[float, float, float]:
    if rate is None:
        return _blend(style.POSITIVE, 0.0)
    away = abs(rate - 0.5) * 2.0
    return _blend(style.POSITIVE if rate >= 0.5 else style.NEGATIVE, away)


def _shade_mean(mean: float | None, centre: float, spread: float) -> tuple[float, float, float]:
    if mean is None or spread <= 0:
        return _blend(style.POSITIVE, 0.0)
    away = abs(mean - centre) / (FULL_SHADE_SDS * spread)
    return _blend(style.POSITIVE if mean >= centre else style.NEGATIVE, away)


def _number(value: float | None) -> str:
    return "?" if value is None else f"{value:.3g}"


def _boxes(structure: dict, nodes: dict, poles: list[str]):
    """How a box is drawn, by the kind of target: its shade, the number it shows for its rows, and
    what a leaf predicts. A binary tree's leaf shows the class the tree predicts there, named by
    `poles`: the positive one where the leaf's score is at least a half. The score is the share
    of the leaf's fitting rows in the positive class, weighted as the run weights the classes, so
    under a weighting a leaf can predict a class to which fewer than half of its rows belong. A
    continuous tree's leaf shows its predicted value."""
    if structure["target_kind"] == "continuous":
        root = nodes[""]
        return (lambda node: _shade_mean(node["mean"], root["mean"], root["sd"]),
                lambda node: f"mean {_number(node['mean'])}",
                lambda node: _number(node["prediction"]))
    return (lambda node: _shade(node["positive_rate"]),
            lambda node: f"{node['positive_rate']:.0%}",
            lambda node: poles[0] if node["prediction"] >= 0.5 else poles[1])


def _layout(nodes: dict, root: str, max_depth: int) -> dict[str, tuple[float, float]]:
    """A tidy layout: leaves evenly spaced left to right, parents centred over their children."""
    x_cursor = [0.0]
    position: dict[str, tuple[float, float]] = {}

    def place(path: str, depth: int) -> float:
        node = nodes[path]
        folded = depth >= max_depth or node["is_leaf"]
        if folded:
            x = x_cursor[0]
            x_cursor[0] += 1.0
        else:
            left = place(node["left"], depth + 1)
            right = place(node["right"], depth + 1)
            x = (left + right) / 2.0
        position[path] = (x, -float(depth))
        return x

    place(root, 0)
    return position


def draw(
    structure: dict,
    path: Path,
    poles: list[str],
    note: str,
    max_depth: int,
    title: str,
) -> Path:
    """Draw a serialised tree down to `max_depth`, folding what lies below. `poles` names the two
    classes of a binary target, positive first, and `note` says what the shading shows."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon

    style.apply()
    nodes = {node["path"]: node for node in structure["nodes"]}
    shade, figure_of, prediction = _boxes(structure, nodes, poles)
    root = ""
    position = _layout(nodes, root, max_depth)
    drawn = sorted(position, key=lambda p: (len(p), p))
    width = max(x for x, _ in position.values()) + 1.0

    figure, axis = plt.subplots(figsize=(max(8.0, width * 1.55), 1.0 + (max_depth + 1) * 1.05))
    axis.set_axis_off()

    box_w, box_h = 0.92, 0.60
    for node_path in drawn:
        node = nodes[node_path]
        x, y = position[node_path]
        hidden = -y >= max_depth and not node["is_leaf"]

        if hidden:
            # A wedge stands for the branch that continues below the drawn depth.
            axis.add_patch(
                Polygon(
                    [(x - box_w / 2, y + box_h / 2), (x + box_w / 2, y + box_h / 2), (x, y - box_h / 2)],
                    closed=True, facecolor=shade(node),
                    edgecolor=style.AXIS, linewidth=0.8, zorder=2,
                )
            )
            axis.text(
                x, y + box_h * 0.16, f"{node['n_samples']} rows",
                ha="center", va="center", fontsize=7.5, color=style.INK, zorder=3,
            )
            axis.text(
                x, y - box_h * 0.10, figure_of(node),
                ha="center", va="center", fontsize=7.5, color=style.INK_SECONDARY, zorder=3,
            )
            continue

        axis.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.012,rounding_size=0.06",
                facecolor=shade(node),
                edgecolor=style.AXIS, linewidth=0.8, zorder=2,
            )
        )
        if node["is_leaf"]:
            headline = "predicts\n" + prediction(node)
            axis.text(x, y + box_h * 0.10, headline, ha="center", va="center",
                      fontsize=7, color=style.INK, zorder=3)
        else:
            axis.text(
                x, y + box_h * 0.16,
                _wrap(node["feature"], 22) + f"\n$\\leq$ {_threshold(node['threshold'])}",
                ha="center", va="center", fontsize=7, color=style.INK, zorder=3,
            )
        axis.text(
            x, y - box_h * 0.30,
            f"{node['n_samples']} rows · {figure_of(node)}",
            ha="center", va="center", fontsize=6.8, color=style.INK_SECONDARY, zorder=3,
        )

        if not node["is_leaf"] and -y < max_depth:
            for child, label, offset in ((node["left"], "yes", -1), (node["right"], "no", 1)):
                cx, cy = position[child]
                axis.plot([x, cx], [y - box_h / 2, cy + box_h / 2],
                          color=style.AXIS, linewidth=0.9, zorder=1)
                axis.text(
                    (x + cx) / 2 + offset * 0.06, (y - box_h / 2 + cy + box_h / 2) / 2,
                    label, ha="center", va="center", fontsize=6.8, color=style.INK_MUTED,
                    zorder=3, bbox={"facecolor": style.SURFACE, "edgecolor": "none", "pad": 0.8},
                )

    axis.set_xlim(-0.7, width - 0.3)
    axis.set_ylim(-max_depth - 0.62, 0.48)

    folded_count = sum(1 for p in drawn if -position[p][1] >= max_depth and not nodes[p]["is_leaf"])
    total_leaves = sum(1 for node in structure["nodes"] if node["is_leaf"])
    tail = (
        f" The {folded_count} triangles are branches that continue below this depth; the whole tree "
        f"has {total_leaves} end points."
        if folded_count
        else ""
    )
    figure.suptitle(
        style.caption(
            title,
            f"Each box asks one question, and \"yes\" goes left. {note}{tail}",
            width=118,
        ),
        fontsize=10, color=style.INK, y=0.995, va="top",
    )
    figure.subplots_adjust(top=1 - 1.15 / figure.get_figheight(), bottom=0.02, left=0.01, right=0.99)
    return style.save(figure, path)


def _threshold(value: float) -> str:
    """A split threshold as a reader would say it, with no trailing zeros to read past."""
    if value is None:
        return "?"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.4g}"


def _wrap(name: str, width: int) -> str:
    return style.wrap_name(name, width, max_lines=3)
