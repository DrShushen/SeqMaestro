"""What each figure of a report draws, decided here and written to `figure_inputs.json`.

The figures are drawn under `SeqMaestro_LocalUtils` (`seqmaestro.report.figures`), and the drawing
code reads that file and nothing else. Every choice about what a figure shows is therefore made
here, in the environment that reads the run: which arms a figure shows, the numbers, the labels and
class names, the reference line, the caption and the width, and everything that differs between the
kinds of target (the metric and its chance level, what the poles of a direction are called). The
drawing side decides only how the figure looks. Every number it shows comes from that file, because
it is never handed a run.

Each entry is `{"draw": <drawer>, "caption", "width", "data": {...}}`, keyed by the name a report
source uses in `{{figure:NAME}}`.
"""

from __future__ import annotations

import math
from typing import Any

from seqmaestro.report import markdown, naming
from seqmaestro.report.concepts import Concept
from seqmaestro.report.facts import HEADLINE_FAMILIES
from seqmaestro.report.internals import NO_FEATURES
from seqmaestro.target import BINARY, CONTINUOUS

#: Levels of a tree drawn before a branch is folded into a summary wedge.
TREE_DEPTH = 3

#: Panels per row in the shape-function figure. Four fit across a portrait page.
SHAPE_COLUMNS = 4

#: The figure each drawn model class gets: its name in a report, its drawer, its caption, its width,
#: and what its title calls the model.
DRAWN_AS = {
    "ebm": ("stage3a_ebm", "ebm_shapes", "Shape functions of the additive model cut to {n} features",
            "100%", "The additive model"),
    "l1_logistic": ("stage3a_l1", "l1_weights",
                    "Weights of the sparse linear model cut to {n} features", "95%",
                    "The sparse linear model"),
    "l1_linear": ("stage3a_l1", "l1_weights",
                  "Weights of the sparse linear model cut to {n} features", "95%",
                  "The sparse linear model"),
    "igsr": ("stage3a_igsr", "l1_weights", "Weights of the IGSR model cut to {n} terms", "95%",
             "The IGSR model"),
    "deft_or_cart": ("stage3a_tree", "tree", "The decision tree cut to {n} features", "100%",
                     "The decision tree"),
    "pysr": ("stage3a_pysr", "equation_shapley", "Shapley values of the equation cut to {n} features",
             "100%", "The equation"),
}

#: The second figure of a symbolic model: the equations its search found, from which it chose one.
FRONT = ("stage3a_pysr_front", "equation_front", "The equations the search found, by complexity",
         "80%")

#: What the concept map calls a held-out association (Tier 2), and how far from the value that
#: means none it is shaded at full strength. An AUROC of 0.9 is a rank-biserial correlation of 0.8,
#: so both kinds reach full strength at the same strength of association.
ASSOCIATION_SHADING = {"auc": ("separation,\nheld out", 0.4), "rho": ("rank correlation,\nheld out", 0.8)}

#: What a feature's marginal effect is called on an axis: a rank-biserial correlation with the
#: classes, or Spearman's rho with a continuous target.
ALONE_LABEL = {BINARY: "separation on its own, -1 to +1",
               CONTINUOUS: "rank correlation on its own, -1 to +1"}


def polarity(facts: dict[str, Any]) -> dict[str, Any]:
    """How the figures say which way a feature points: the two poles, positive first, and the
    sentences and legend entries built on them.

    A binary target's poles are its classes, and a feature argues for one of them. Figures carry only
    the class names: a class's meaning is a whole clause, and the report states it once, beside the
    names, in its title block. A continuous target's poles are a higher and a lower value, and a
    feature raises or lowers it.
    """
    directions = facts["directions"]
    rising, falling = (directions["printed"][directions[side]] for side in ("rising", "falling"))
    if facts["target_kind"] == BINARY:
        return {"poles": [rising, falling],
                "shapes": f"Above the line the feature argues {rising}. Below it, the feature "
                          f"argues {falling}.",
                "shapley": f"Each dot is one training sequence. Above the line the feature argued "
                           f"{rising} for that sequence, and below it {falling}.",
                "weights": f"Bars to the right argue {rising}. Bars to the left argue {falling}.",
                "tree": f"Shading is the share of {rising} among the rows reaching that box, blue "
                        "high and red low.",
                "legend": [f"reached it; higher values argue {rising}",
                           f"reached it; higher values argue {falling}"]}
    return {"poles": [rising, falling],
            "shapes": "Above the line the feature raises the target. Below it, the feature lowers it.",
            "shapley": "Each dot is one training sequence. Above the line the feature raised that "
                       "sequence's prediction, and below it the feature lowered it.",
            "weights": "Bars to the right raise the target. Bars to the left lower it.",
            "tree": "Shading is the mean target of the rows reaching that box, blue above the mean "
                    "of every fitting row and red below it.",
            "legend": [f"reached it; higher values {rising}", f"reached it; higher values {falling}"]}


def _style(arm: str) -> dict[str, Any]:
    configuration = naming.parse(arm)
    return {"arm": arm, "label": configuration.label,
            "interpretable": configuration.is_interpretable,
            "writes_features": configuration.writes_features}


def weight_decimals(weights: list[float]) -> int:
    """How many decimals a weight chart labels its bars with: two where the largest weight reaches
    0.1, and one more for each factor of ten below that, so a target measured in small units does
    not label every weight 0.00."""
    span = max((abs(weight) for weight in weights), default=0.0)
    return max(2, 1 - math.floor(math.log10(span))) if span > 0 else 2


def build(facts: dict[str, Any], concepts: list[Concept], internals: dict[str, dict]) -> dict:
    """Every figure's inputs. `internals` maps a headline slot (`best_ebm`, ...) to what
    `seqmaestro.report.internals`, or DEFT's side for a tree, exported for its arm."""
    kind = facts["target_kind"]
    score, chance = facts["metrics"]["headline"], facts["metrics"]["chance"]
    scale = {"metric_label": f"{naming.METRICS[score]} on the held-out sequences", "chance": chance}
    poles = polarity(facts)
    budget = facts["stage3a"]["feature_budget"]
    n = markdown.word(budget)
    base = facts["baseline"]
    lines = [] if base is None else [{
        "value": base[score],
        "label": f"{base['label']}, {base['n_features']} numbers",
        "legend": "base composition of the sequence", "linestyle": "-"}]
    figures: dict[str, dict] = {}

    stage2 = facts["stage2"]["arms"]
    figures["stage2_grid"] = {
        "draw": "grid", "caption": "Every model configuration on held-out sequences", "width": "85%",
        "data": {"rows": [{**_style(arm), "value": entry[score],
                           "control": entry[f"shuffled_{score}"]}
                          for arm, entry in stage2.items() if entry[score] is not None],
                 "baselines": lines, **scale,
                 "title": "Every model configuration, on sequences none of them were fitted on",
                 "subtitle": "Bar length is how far the configuration scores above chance."}}

    stage3a = facts["stage3a"]["arms"]
    if stage3a:
        figures["stage3a_grid"] = {
            "draw": "grid", "caption": f"Every configuration cut to at most {n} features",
            "width": "85%",
            "data": {"rows": [{**_style(arm), "value": entry[f"reduced_{score}"],
                               "low": entry[f"reduced_{score}_low"],
                               "high": entry[f"reduced_{score}_high"]}
                              for arm, entry in stage3a.items()],
                     "baselines": lines, **scale,
                     "title": f"The same configurations cut to at most {n} features each",
                     "subtitle": ("Whiskers are a 95% interval from resampling the held-out "
                                  "sequences."
                                  if any(entry[f"reduced_{score}_low"] is not None
                                         for entry in stage3a.values()) else "")}}
        figures["stage3a_cost"] = {
            "draw": "cost", "caption": f"What cutting to {n} features cost each configuration",
            "width": "85%",
            "data": {"rows": [{**_style(arm), "full": entry[f"full_{score}"],
                               "reduced": entry[f"reduced_{score}"]}
                              for arm, entry in stage3a.items()],
                     "reduced_label": f"cut to at most {n} features", **scale,
                     "title": f"What cutting to {n} features cost each configuration",
                     "subtitle": "An open circle is every feature the configuration selected; a filled "
                                 f"one is what the cut to at most {n} left it."}}

    headline = facts["headline_set"]
    families = HEADLINE_FAMILIES[kind]
    for family in families:
        slot = f"best_{family}"
        # A model that used no feature has nothing to draw (`internals.no_features`).
        if slot not in internals or internals[slot]["kind"] == NO_FEATURES:
            continue
        name, drawer, caption, width, model = DRAWN_AS[family]
        label = naming.label(headline[slot])
        # How many columns the arm's reduction kept (`n_final`), in place of the budget. It falls
        # below the budget when the arm's Stage 2 model used fewer features than the budget, as an
        # equation often does. It counts a kept column the reduced model gives no weight, so the
        # words say what the model was cut to, and the drawing shows what it uses.
        own = markdown.word(facts["stage3a"]["arms"][headline[slot]]["n_final"])
        data = {"internals": internals[slot], "title": f"{model}: {label}, cut to {own} features"}
        if drawer == "ebm_shapes":
            data.update(note=poles["shapes"], columns=SHAPE_COLUMNS)
        elif drawer == "equation_shapley":
            data.update(note=poles["shapley"], columns=SHAPE_COLUMNS)
            front_name, front_drawer, front_caption, front_width = FRONT
            figures[front_name] = {
                "draw": front_drawer, "caption": front_caption, "width": front_width,
                "data": {"internals": internals[slot],
                         "title": f"The equations the search found for {label}",
                         "note": "Each point is the best equation the search found at that "
                                 "complexity. The chosen equation has the lowest loss on the "
                                 "rows set aside."}}
        elif drawer == "l1_weights":
            data.update(note=poles["weights"], alone_label=ALONE_LABEL[kind],
                        decimals=weight_decimals([row["weight"]
                                                  for row in internals[slot]["features"]]))
        else:
            data.update(poles=poles["poles"], note=poles["tree"], max_depth=TREE_DEPTH)
        figures[name] = {"draw": drawer, "caption": caption.format(n=own), "width": width,
                         "data": data}

    stage3b = facts["stage3b"]["arms"]
    if stage3b:
        panels = {arm: {"label": entry["label"],
                        "subtitle": f"chose from {entry['pool_size']:,} candidates"
                                    + (f" · {entry['evaluation_rows']} held-out sequences"
                                       if entry["evaluation_rows"] else ""),
                        "features": [{key: row[key] for key in
                                      ("feature", "cluster_pi_hat", "q_bh", "p_holm")}
                                     for row in entry["reported"]]}
                  for arm, entry in stage3b.items()}
        shown = [arm for arm in dict.fromkeys(headline.get(slot) for slot in
                                              (*(f"best_{family}" for family in families),
                                               "best_enumerated"))
                 if arm in panels]
        figures["stage3b_evidence"] = {
            "draw": "evidence_grid", "caption": "The three tiers, for the models this report shows",
            "width": "100%",
            "data": {"panels": [panels[arm] for arm in shown], "columns": 2,
                     "title": "Three pieces of evidence, for the models the body of this report shows"}}
        figures["stage3b_evidence_all"] = {
            "draw": "evidence_grid", "caption": "Three pieces of evidence for every reported feature",
            "width": "100%",
            "data": {"panels": list(panels.values()), "columns": 3,
                     "title": "Three pieces of evidence for every feature the interpretable models "
                              "reported"}}
        figures["stage3b_thresholds"] = {
            "draw": "threshold_sweep", "caption": "How many features survive at each cut-off",
            "width": "88%",
            "data": {"rows": threshold_rows(stage3b), "n_reported": budget, "n_word": n,
                     "title": "How many features survive, at every cut-off rather than at one"}}
        multi = [concept for concept in concepts if len(concept.arms) > 1]
        arms = list(stage3b)
        column, full = ASSOCIATION_SHADING[facts["metrics"]["association"]]
        association = {"label": column, "centre": facts["metrics"]["association_centre"],
                       "full": full}
        figures["stage3b_concepts"] = {
            "draw": "concept_map", "caption": "Ideas that more than one configuration arrived at",
            "width": "100%",
            "data": {"columns": [naming.label(arm) for arm in arms],
                     "concepts": [{"label": concept.label,
                                   "reached": [arms.index(arm) for arm in concept.arms if arm in arms],
                                   "direction": concept.direction,
                                   "best_stability": concept.best_stability,
                                   "strongest_association": concept.strongest_association,
                                   "best_association_q": concept.best_association_q}
                                  for concept in multi],
                     "legend": poles["legend"], "association": association, "n_word": n,
                     "title": "Ideas that more than one model configuration arrived at"}}
    return figures


def threshold_rows(stage3b: dict[str, Any]) -> list[dict]:
    """For every profiled arm, what survives at each cut-off at Tier 1's cluster level, where
    near-duplicate candidates count as one: how many of its reported features, how many clusters of
    its whole pool, and the chance bound at that cut-off."""
    rows = []
    for entry in stage3b.values():
        curve = entry["curve_cluster"]
        taus = [float(point["tau"]) for point in curve]
        rows.append({
            "label": entry["label"], "taus": taus,
            "reported": [sum(1 for feature in entry["reported"]
                             if feature["cluster_pi_hat"] is not None
                             and feature["cluster_pi_hat"] >= tau) for tau in taus],
            "pool": [float(point["n_selected"]) for point in curve],
            "bound": [float(point["bound_expected_false"]) for point in curve]})
    return sorted(rows, key=lambda row: (-sum(row["reported"]), row["label"]))
