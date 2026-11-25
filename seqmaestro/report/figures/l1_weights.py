"""Draw a linear model's weights beside what each feature does on its own.

A linear model adds up one weighted term per feature, so the whole model is a list of weights. Each
weight is exported per standard deviation of its feature on the fitting rows, for a sparse linear
model and for the least-squares model of an IGSR arm (`seqmaestro.report.internals.l1_weights`), so
the weights are comparable across features (a count in the hundreds and a yes/no indicator are each
measured in units of their own spread). The left panel draws the weights as they are, per standard
deviation, in the model's scale (`scale`: log-odds for a binary target, the target's units for a
continuous one).

The right panel answers a different question in the same row order: how far each feature goes with
the target on its own, with no model involved (its marginal effect, a rank correlation). The two
panels can point opposite ways for a feature. That is a finding rather than an error: it is what a
model looks like when it uses one feature to correct another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seqmaestro.report.figures import style

#: Below this, a marginal effect has no reliable sign, so the two panels pointing opposite ways there
#: says nothing. Marking those rows would put a reader's attention on noise.
DISAGREEMENT_FLOOR = 0.05


def draw(internals: dict, path: Path, note: str, alone_label: str, title: str,
         decimals: int) -> Path:
    """Every weight, per standard deviation, and the marginal effect beside it where the record has
    one. `note` says what a bar to the right or to the left means, `alone_label` names the marginal
    effect's axis, and `decimals` is how many decimals a weight's label shows."""
    import matplotlib.pyplot as plt

    style.apply()
    rows = sorted(internals["features"], key=lambda row: abs(row["weight"]))
    order = [row["name"] for row in rows]
    moved = {row["name"]: row["weight"] for row in rows}
    marginal = {row["name"]: row["marginal_effect"] for row in rows
                if row["marginal_effect"] is not None}
    positions = np.arange(len(order))

    panels = 2 if marginal else 1
    figure, axes = plt.subplots(
        1, panels, figsize=(6.6 + 3.2 * (panels - 1), 0.42 * len(order) + 2.2), sharey=True,
        squeeze=False, gridspec_kw={"width_ratios": [2.1, 1] if panels == 2 else [1]})
    left = axes[0][0]

    values = [moved[name] for name in order]
    left.barh(positions, values, height=0.62, linewidth=0, color=[style.polarity(v) for v in values])
    left.axvline(0.0, color=style.AXIS, linewidth=0.9)
    left.set_yticks(positions)
    left.set_yticklabels([style.shorten(name, 44) for name in order], fontsize=8, color=style.INK)
    left.set_xlabel(f"{internals['scale']} per standard deviation of the feature", fontsize=8)
    left.set_title("What the model does with each feature", fontsize=9, loc="left", pad=6)
    left.grid(axis="x", zorder=0)
    left.set_axisbelow(True)
    span = max(abs(v) for v in values) or 1.0
    for position, value in zip(positions, values, strict=True):
        left.text(value + np.sign(value) * span * 0.03, position, f"{value:+.{decimals}f}",
                  va="center", ha="left" if value >= 0 else "right", fontsize=7.5,
                  color=style.INK_SECONDARY)
    left.set_xlim(-span * 1.32, span * 1.32)

    disagreeing = {name for name in order if name in marginal
                   and abs(marginal[name]) >= DISAGREEMENT_FLOOR and marginal[name] * moved[name] < 0}
    for position, name in zip(positions, order, strict=True):
        if name in disagreeing:
            for axis in axes[0]:
                axis.axhspan(position - 0.44, position + 0.44, color=style.GRID, zorder=0)

    if marginal:
        right = axes[0][1]
        effects = [marginal.get(name, float("nan")) for name in order]
        right.barh(positions, effects, height=0.62, linewidth=0,
                   color=[style.polarity(v) for v in effects])
        right.axvline(0.0, color=style.AXIS, linewidth=0.9)
        right.set_xlim(-1.05, 1.05)
        right.set_xlabel(alone_label, fontsize=8)
        right.set_title("What the feature does on its own", fontsize=9, loc="left", pad=6)
        right.grid(axis="x", zorder=0)
        right.set_axisbelow(True)
        for position, value in zip(positions, effects, strict=True):
            if value == value:
                right.text(value + np.sign(value) * 0.04, position, f"{value:+.2f}", va="center",
                           ha="left" if value >= 0 else "right", fontsize=7.5,
                           color=style.INK_SECONDARY)

    shaded = ""
    if disagreeing:
        shaded = (f"\nShaded rows are the {len(disagreeing)} feature(s) the model uses against the "
                  "direction they point in on their own.")
    figure.suptitle(style.caption(title, f"{note}{shaded}", width=96),
                    fontsize=10, color=style.INK, y=0.995, va="top")
    figure.subplots_adjust(top=1 - (1.25 if disagreeing else 0.95) / figure.get_figheight(),
                           bottom=0.9 / figure.get_figheight(),
                           left=0.30 if panels == 2 else 0.42, right=0.985, wspace=0.08)
    return style.save(figure, path)
