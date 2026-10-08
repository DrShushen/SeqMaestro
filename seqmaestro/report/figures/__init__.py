"""A report's figures, drawn from `figure_inputs.json`. **Needs matplotlib: `SeqMaestro_LocalUtils`.**

Core is the environment that produces results, and a plotting stack that no result depends on stays
out of it. `tests/test_report.py` checks that no module outside this package imports matplotlib, and
that the modules here import nothing from `seqmaestro` outside this package. Every figure is
therefore drawn from what `seqmaestro.report.figure_inputs` wrote, and nothing here reads a run.

`draw(entry, path)` draws one entry of that file. `style` holds the palette and defaults, and the
other modules hold the drawers.
"""

from __future__ import annotations

from pathlib import Path

from seqmaestro.report.figures import ebm_shapes, equation, l1_weights, performance, reliability, tree

#: What each `draw` value in `figure_inputs.json` is drawn by, from its `data` and the target path.
DRAWERS = {
    "grid": lambda data, path: performance.draw_grid(
        data["rows"], path, data["baselines"], data["title"], data["subtitle"],
        data["metric_label"], data["chance"]),
    "cost": lambda data, path: performance.draw_cost(
        data["rows"], path, data["reduced_label"], data["title"], data["subtitle"],
        data["metric_label"], data["chance"]),
    "ebm_shapes": lambda data, path: ebm_shapes.draw(
        data["internals"], path, data["note"], data["columns"], data["title"]),
    "l1_weights": lambda data, path: l1_weights.draw(
        data["internals"], path, data["note"], data["alone_label"], data["title"],
        data["decimals"]),
    "equation_shapley": lambda data, path: equation.draw_shapley(
        data["internals"], path, data["note"], data["columns"], data["title"]),
    "equation_front": lambda data, path: equation.draw_front(
        data["internals"], path, data["title"], data["note"]),
    "tree": lambda data, path: tree.draw(
        data["internals"], path, data["poles"], data["note"], data["max_depth"], data["title"]),
    "evidence_grid": lambda data, path: reliability.draw_evidence_grid(
        data["panels"], path, data["columns"], data["title"]),
    "threshold_sweep": lambda data, path: reliability.draw_threshold_sweep(
        data["rows"], path, data["n_reported"], data["n_word"], data["title"]),
    "concept_map": lambda data, path: reliability.draw_concept_map(
        data["concepts"], data["columns"], path, data["legend"], data["association"],
        data["n_word"], data["title"]),
}


def draw(entry: dict, path: Path) -> Path:
    """One figure, as its entry in `figure_inputs.json` describes it."""
    if entry["draw"] not in DRAWERS:
        raise KeyError(f"No drawer for {entry['draw']!r}; the drawers are {sorted(DRAWERS)}.")
    return DRAWERS[entry["draw"]](entry["data"], Path(path))
