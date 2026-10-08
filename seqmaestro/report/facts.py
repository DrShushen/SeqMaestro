"""Every number a report quotes, keyed and structured, with the files it was read from.

`facts.json` is one of the two sources a report is written from, and `tables.py` renders the same
numbers as markdown. A writer includes a table and says what it means, and quotes a number in prose
only from `facts.json`, so the report and its evidence cannot drift apart.

**Performance is on the test split unless a key says `val_`.** Every test number is the one the
reveal scored and wrote into a cell's `metrics.json`. Nothing here scores anything.

**A score's key is named by the metric of the target's kind**: `auroc` and `auprc` for a binary
target, `r2` and `spearman_rho` for a continuous one, so `reduced_auroc` is `reduced_r2` there.
`metrics` names them, with the chance level and the panel's association and loss, and every reader
of the facts builds its keys from it.

**A fixed rule chooses which configurations lead the report.** The headline set is the best
configuration of each interpretable model class, the best one built on features enumerated in
advance, and the strongest black-box reference. Applying the rule here means every dataset's report
leads with the same kinds of configuration, and a reader can see what the rule was.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from seqmaestro import arms as registry
from seqmaestro.arms import split_arm
from seqmaestro.dataset import NOT_APPLICABLE, PASSED
from seqmaestro.harness.cell import METRICS_FILE
from seqmaestro.harness.direction import NON_MONOTONE
from seqmaestro.harness.heldout import TEST_SPLIT
from seqmaestro.reliability import tables as reliability_tables
from seqmaestro.reliability import tier2, tier3
from seqmaestro.reliability.files import CHECKS_FILE, RECORD_FILE, TIER3_FILE
from seqmaestro.report import markdown, naming
from seqmaestro.report.sources import Cell, ReducedArm, Run
from seqmaestro.target import BINARY, CONTINUOUS, target_for

#: A reported feature chosen in at least this share of half-samples is counted as chosen again.
STABLE_PI = 0.7

#: The level for the corrected tests. A feature with a q-value at or below it counts as holding, and
#: a removal with a Holm-corrected p-value below it counts as costing the model.
ALPHA = 0.05

#: The interpretable model classes that each get one configuration in the headline set, in the
#: order a report shows them, by the kind of target. They follow from the registry
#: (`seqmaestro.arms`): every model that is read feature by feature and fits the kind, so the sparse
#: linear model is the logistic one for a binary target and the least-squares one for a continuous
#: one.
HEADLINE_FAMILIES = {
    kind: tuple(name for name in naming.MODELS
                if naming.model_interpretable(name) and kind in registry.spec(name).target_kinds)
    for kind in (BINARY, CONTINUOUS)}


def metrics(target_kind: str) -> dict[str, Any]:
    """The names a `target_kind` run's numbers go under: the headline and its companion, the
    headline's chance level, the panel's association (Tier 2) with the value that means none, and
    the loss its refits are compared in (Tier 3)."""
    target = target_for(target_kind)
    association, centre = tier2.ASSOCIATION[target_kind]
    return {"headline": target.headline, "companion": target.companion, "chance": target.chance,
            "association": association, "association_centre": centre,
            "loss": tier3.LOSS[target_kind][0]}


def _number(value) -> float | None:
    """A float, or None for a value that is missing, blank or NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


def _mean(values) -> float | None:
    found = [v for v in values if v is not None]
    return float(np.mean(found)) if found else None


def target(dataset) -> dict[str, Any]:
    """What the target is, in the dataset's words: its classes for a binary target, and its name
    and meaning for a continuous one."""
    if dataset.target_kind == BINARY:
        return {"classes": {"positive": dataset.classes[1].name, "negative": dataset.classes[0].name,
                            "positive_meaning": dataset.classes[1].meaning,
                            "negative_meaning": dataset.classes[0].meaning}}
    declared = dataset.record["target"]
    return {"target": {"name": declared["name"], "meaning": declared["meaning"]}}


def directions(dataset) -> dict[str, Any]:
    """The words a feature's direction is recorded in and how a report prints each. `rising` is the
    word for a feature whose higher values go with the positive class or a higher target, and
    `falling` its opposite. A binary target's words print as the class they argue for."""
    rising, falling = dataset.target.directions
    if dataset.target_kind == BINARY:
        printed = {rising: dataset.classes[1].name, falling: dataset.classes[0].name}
    else:
        printed = {word: naming.CONTINUOUS_DIRECTIONS[word] for word in (rising, falling)}
    return {"rising": rising, "falling": falling,
            "printed": {**printed, NON_MONOTONE: naming.BOTH_WAYS}}


# ------------------------------------------------------------------------------ the headline rule


def headline_set(scores: dict[str, float | None], target_kind: str) -> dict[str, str]:
    """The configurations the body of a report leads with, keyed by why each is there.

    The set holds the best configuration of each interpretable model class (`best_<class>`), the
    best interpretable configuration drawing on features enumerated in advance (`best_enumerated`),
    which is the contrast the design exists to make, and the strongest black-box reference
    (`black_box_reference`). The reference shows how much signal the data holds and says nothing
    about features. `scores` is each arm's held-out score at the stage the set is chosen for.
    """
    scored = [(arm, value) for arm, value in scores.items() if value is not None]
    chosen: dict[str, str] = {}

    def best(candidates):
        # Ties go to the arm named first alphabetically, so the rule has one answer.
        return max(sorted(candidates), key=lambda pair: pair[1])[0]

    for family in HEADLINE_FAMILIES[target_kind]:
        candidates = [(arm, value) for arm, value in scored
                      if naming.parse(arm).model_key == family and naming.parse(arm).is_interpretable]
        if candidates:
            chosen[f"best_{family}"] = best(candidates)
    enumerated = [(arm, value) for arm, value in scored
                  if naming.parse(arm).is_interpretable and not naming.parse(arm).writes_features]
    if enumerated:
        chosen["best_enumerated"] = best(enumerated)
    reference = [(arm, value) for arm, value in scored if not naming.parse(arm).is_interpretable]
    if reference:
        chosen["black_box_reference"] = best(reference)
    return chosen


# ------------------------------------------------------------------------------ stage 1


def stage1(run: Run) -> dict[str, Any]:
    """What Stage 1 recorded about the dataset, as its `dataset.json` says it."""
    dataset = run.dataset
    record = dataset.record
    return {
        "splits_directory": str(dataset.directory),
        "readme": str(dataset.directory.parent / "README.md"),
        "classes": {str(label): {"name": info.name, "meaning": info.meaning}
                    for label, info in dataset.classes.items()},
        **{key: record.get(key) for key in ("columns", "sequences", "split", "splits", "filters",
                                            "design_power", "notes")},
        # Each covariate with what it measures, in the declaration's words, and its spread on the
        # training rows. Empty for a dataset that supplies none.
        "covariates": record.get("covariates") or {},
        # Whether the language models were kept from a fact the judge was told, and the fact. A
        # report says so, since a result from a blinded run and one from a run that named the
        # answer cannot otherwise be told apart.
        "blinding": run.blinding,
        "checks": {name: result["status"] for name, result in record["checks"].items()},
        # Each finding is kept in full, because a reader needs what was found and what was said
        # about it, and the status alone says only "acknowledged". For example, if some held-out
        # rows were already seen by the model, the report must give their share and the reason the
        # finding was accepted.
        "findings": {name: result for name, result in record["checks"].items()
                     if result["status"] not in (PASSED, NOT_APPLICABLE)},
        "files": [str(dataset.directory / "dataset.json")],
    }


# ------------------------------------------------------------------------------ stage 2


def _analysis_row(frame: pd.DataFrame | None, **match) -> dict[str, Any]:
    if frame is None:
        return {}
    rows = frame
    for column, value in match.items():
        if column not in rows.columns:
            return {}
        rows = rows[rows[column].astype(str) == value]
    return rows.iloc[0].to_dict() if len(rows) else {}


def _paired_gap(real: dict[int, Cell], permuted: dict[int, Cell], headline: str) -> float | None:
    """Mean of real minus shuffled test headline, paired within seed, because the real and
    shuffled cells at one seed share one random draw."""
    gaps = [real[seed].test[headline] - permuted[seed].test[headline]
            for seed in sorted(set(real) & set(permuted))]
    return _mean(gaps)


def stage2(run: Run, names: dict[str, Any]) -> dict[str, Any]:
    """Every arm of the grid before any reduction: its scores over seeds and the Stage 2 analyses.
    `names` is `metrics(kind)`."""
    headline, companion = names["headline"], names["companion"]
    analyses = run.analyses
    recurrence = {entry.get("arm"): entry
                  for entry in (analyses.get("concept_recurrence") or {}).get("arms", [])}
    judge = (analyses.get("judge") or {}).get("arms", {})
    arms = {}
    for arm, by_labels in run.suite.items():
        model, pool = split_arm(arm)
        configuration = naming.parse(arm)
        real, permuted = by_labels.get("real", {}), by_labels.get("permuted", {})
        values = [cell.test[headline] for cell in real.values()]
        shuffled = [cell.test[headline] for cell in permuted.values()]
        stability = _analysis_row(analyses.get("stability"), model=model, pool=pool,
                                  labels="real")
        co_discovery = _analysis_row(analyses.get("co_discovery_by_arm"), arm=arm)
        arms[arm] = {
            "label": configuration.label,
            "model": configuration.model_long,
            "source": configuration.source_long,
            "interpretable": configuration.is_interpretable,
            "writes_features": configuration.writes_features,
            headline: _mean(values),
            f"{headline}_min": min(values) if values else None,
            f"{headline}_max": max(values) if values else None,
            f"val_{headline}": _mean(cell.val[headline] for cell in real.values()),
            companion: _mean(cell.test[companion] for cell in real.values()),
            f"shuffled_{headline}": _mean(shuffled),
            f"shuffled_{headline}_min": min(shuffled) if shuffled else None,
            f"shuffled_{headline}_max": max(shuffled) if shuffled else None,
            "paired_gap": _paired_gap(real, permuted, headline),
            "n_features_used": _mean(_number(cell.run.get("n_features_used"))
                                     for cell in real.values()),
            "n_seeds": len(real),
            "reselection_kappa": _number(stability.get("top_n_kappa")),
            "judge_mean": _number((judge.get(f"{arm}/real") or {}).get("mean")),
            "cross_source_agreement": _number(co_discovery.get("cross_pool_share")),
            "concepts": _number((recurrence.get(arm) or {}).get("n_concepts")),
            "concepts_recurring": _number((recurrence.get(arm) or {}).get("n_concepts_recurring")),
            "files": [str(cell.path / METRICS_FILE)
                      for cells in by_labels.values() for cell in cells.values()],
        }
    cross_arm = analyses.get("cross_arm_recurrence")
    return {
        "arms": arms,
        "n_arms": len(arms),
        "seeds": sorted({seed for by_labels in run.suite.values()
                         for cells in by_labels.values() for seed in cells}),
        "cross_arm_concepts": [] if cross_arm is None else cross_arm.to_dict("records"),
        "analyses_absent": sorted(name for name, found in analyses.items() if found is None),
        # Whether the run's config ran the judge, which decides whether the report has its column.
        "judged": "judge" in analyses,
    }


# ------------------------------------------------------------------------------ the baseline


def baseline(run: Run, budget: int | None, names: dict[str, Any]) -> dict[str, Any] | None:
    """The composition baseline: one reference line, averaged over the seeds it was fitted at."""
    if not run.baseline:
        return None
    headline, companion = names["headline"], names["companion"]
    cells = [run.baseline[seed] for seed in sorted(run.baseline)]
    values = [cell.test[headline] for cell in cells]
    n_features = {int(cell.run["n_features_available"]) for cell in cells}
    if len(n_features) != 1:
        raise ValueError(f"The baseline cells were fitted on different numbers of columns: {n_features}.")
    n = n_features.pop()
    return {
        "label": naming.BASELINE["short"],
        "measures": naming.BASELINE["measures"],
        "model": naming.MODELS[cells[0].run["model_settings"]["model"]]["long"],
        "n_features": n,
        "within_budget": None if budget is None else n <= budget,
        headline: _mean(values),
        f"{headline}_min": min(values),
        f"{headline}_max": max(values),
        companion: _mean(cell.test[companion] for cell in cells),
        f"val_{headline}": _mean(cell.val[headline] for cell in cells),
        "n_seeds": len(cells),
        "model_settings": cells[0].run["model_settings"],
        "files": [str(cell.path / METRICS_FILE) for cell in cells],
    }


# ------------------------------------------------------------------------------ stage 3A


def ranked_features(reduced: ReducedArm) -> list[dict[str, Any]]:
    """The columns the reduction kept, strongest first by the importance the reduction recorded,
    with a column the reduced model gives no weight included.

    Direction and marginal effect come from the Stage 2 cell, the only place they are recorded for
    every arm. The direction is therefore the one the arm's full model gave the feature, before the
    reduction.
    """
    record = reduced.record
    importance = dict(zip(record["final_columns"], record["final_importance"], strict=True))
    order = sorted(record["final_columns"], key=lambda name: -abs(importance[name]))
    rows = []
    for rank, name in enumerate(order, start=1):
        source = reduced.source_records[name]
        rows.append({
            "rank": rank, "name": name, "kind": source.kind,
            "description": " ".join((source.description or "").split()),
            "reduction_importance": importance[name],
            "direction": source.direction,
            "marginal_effect": _number(source.marginal_effect),
        })
    return rows


def stage3a(run: Run, names: dict[str, Any]) -> dict[str, Any]:
    headline, companion = names["headline"], names["companion"]
    arms = {}
    for arm, reduced in run.reduced.items():
        configuration = naming.parse(arm)
        record, roles = reduced.record, reduced.roles
        test = roles["reduced"].test
        # A run with `grid.n_bootstrap: 0` scores without intervals.
        low, high = test.get("ci", {}).get(headline, (None, None))
        arms[arm] = {
            "label": configuration.label,
            "interpretable": configuration.is_interpretable,
            "writes_features": configuration.writes_features,
            "seed": record["seed"],
            "n_start": record["n_start"],
            "n_final": record["n_final"],
            "n_target": record["n_target"],
            f"full_{headline}": roles["full"].test[headline],
            f"reduced_{headline}": test[headline],
            f"reduced_{headline}_low": low,
            f"reduced_{headline}_high": high,
            f"reduced_{companion}": test[companion],
            # Scored at the threshold chosen on validation, which only a binary target has.
            **({"reduced_balanced_accuracy": test["balanced_accuracy"]}
               if "balanced_accuracy" in test else {}),
            f"reduced_val_{headline}": roles["reduced"].val[headline],
            f"top_n_{headline}": roles["top_n"].test[headline],
            "features": ranked_features(reduced),
            "files": [str(reduced.directory / "reduction.json"),
                      *(str(cell.path / METRICS_FILE) for cell in roles.values()),
                      record["source_cell"] + "/features.json"],
        }
    budgets = {entry["n_target"] for entry in arms.values()}
    if len(budgets) > 1:
        raise ValueError(f"The reduced arms were cut to different sizes: {sorted(budgets)}.")
    return {"arms": arms, "feature_budget": budgets.pop() if budgets else None}


# ------------------------------------------------------------------------------ stage 3B


def _reported(features: pd.DataFrame, names: dict[str, Any]) -> list[dict[str, Any]]:
    rows = features[features["in_reported_model"].astype(bool)]
    direction = f"direction_{TEST_SPLIT}"
    association, loss = names["association"], names["loss"]
    return [{
        "feature": row["feature"],
        "kind": row["kind"],
        "description": " ".join(str(row["description"] if isinstance(row["description"], str)
                                    else "").split()),
        "pi_hat": _number(row["pi_hat"]),
        "cluster_pi_hat": _number(row["cluster_pi_hat"]),
        "cluster_size": None if _number(row["cluster_size"]) is None else int(row["cluster_size"]),
        association: _number(row[association]),
        "q_bh": _number(row["q_bh"]),
        "q_by": _number(row["q_by"]),
        "p_westfall_young": _number(row["p_westfall_young"]),
        "p_holm": _number(row["loco_p_holm"]),
        f"delta_{loss}": _number(row[f"loco_mean_delta_{loss}"]),
        f"delta_{names['headline']}": _number(row[f"loco_delta_{names['headline']}"]),
        "direction": row[direction] if isinstance(row[direction], str) and row[direction] else None,
    } for _, row in rows.iterrows()]


def _counts(reported: list[dict[str, Any]]) -> dict[str, int]:
    def at_least(value, level):
        return value is not None and value >= level

    def at_most(value, level):
        return value is not None and value <= level

    return {
        "stable_exact": sum(at_least(r["pi_hat"], STABLE_PI) for r in reported),
        "stable": sum(at_least(r["cluster_pi_hat"], STABLE_PI) for r in reported),
        "associated": sum(at_most(r["q_bh"], ALPHA) for r in reported),
        "necessary": sum(r["p_holm"] is not None and r["p_holm"] < ALPHA for r in reported),
        "stable_and_associated": sum(at_least(r["cluster_pi_hat"], STABLE_PI)
                                     and at_most(r["q_bh"], ALPHA) for r in reported),
    }


def stage3b(run: Run, names: dict[str, Any]) -> dict[str, Any]:
    headline = names["headline"]
    arms = {}
    for arm, profiled in run.profiled.items():
        stability, association, necessity = profiled.tier1, profiled.tier2, profiled.tier3
        reported = _reported(profiled.features, names)
        arms[arm] = {
            "label": naming.parse(arm).label,
            "pool_source": profiled.record["candidates"]["source"],
            "pool_caveat": profiled.record["candidates"].get("caveat"),
            "pool_size_offered": profiled.record["candidates"]["p"],
            "pool_size": stability["feature_level"]["p"],
            "half_samples": stability["feature_level"]["n_half_samples"],
            "selected_set_size": stability["feature_level"]["q_hat"],
            "clusters": stability["cluster_level"]["p"],
            "curve_feature": stability["feature_level"]["curve"],
            "curve_cluster": stability["cluster_level"]["curve"],
            "smallest_tau_bound_one": stability["feature_level"].get("tau_bound_1"),
            "tier2_refused": profiled.record["tier2"]["refused"],
            "evaluation_rows": None if association is None else association["n_rows"],
            "permutations": None if association is None else association["n_permutations"],
            "smallest_reachable_p": None if association is None else association["resolution"],
            "pool_called": {} if association is None else association["called"],
            "plug_in_fdr": [] if association is None else association["plug_in_fdr"],
            "reported": reported,
            "reported_summary": _counts(reported),
            f"reduced_{headline}": necessity[f"{headline}_model"],
            "tier3_refits": necessity["n_tests"],
            "is_check_arm": profiled.record["is_check_arm"],
            "checks": profiled.checks,
            "reproduced_from_pool": profiled.prepare["reproduction"].get("from_pool", {}).get(
                "identical"),
            "summary": profiled.summary,
            "files": [str(profiled.directory / name) for name in
                      (RECORD_FILE, reliability_tables.FEATURES_FILE, TIER3_FILE, CHECKS_FILE)],
        }
    return {"arms": arms, "not_profiled": run.not_profiled}


# ------------------------------------------------------------------------------ parameters


def parameters(run: Run, facts: dict[str, Any], concept_similarity: float) -> dict[str, str]:
    """Settings a dataset-independent explanation has to quote, read from what the run recorded.

    These differ between runs. A component that stated one as a constant would be silently wrong on
    a run configured otherwise, so it asks for them through `{{param:NAME}}` instead.
    """
    word = markdown.word
    panel = next(iter(run.profiled.values()), None)
    settings = panel.record["settings"] if panel else {}
    tier2 = next((p.tier2 for p in run.profiled.values() if p.tier2 is not None), None)
    budget = facts["stage3a"]["feature_budget"]
    test = run.dataset.split_counts[TEST_SPLIT]
    base = facts["baseline"] or {}
    return {
        "half_samples": word(2 * settings["pairs"]) if settings else markdown.MISSING,
        "half_sample_pairs": word(settings["pairs"]) if settings else markdown.MISSING,
        "permutations": f"{tier2['n_permutations']:,}" if tier2 else markdown.MISSING,
        "smallest_p": f"{tier2['resolution']:.5f}".rstrip("0") if tier2 else markdown.MISSING,
        "n_features": word(budget),
        "n_features_minus_one": word(budget - 1) if budget else markdown.MISSING,
        "n_repeats": word(len(facts["stage2"]["seeds"])),
        "n_configurations": word(len(facts["stage2"]["arms"])),
        "n_reduced": word(len(facts["stage3a"]["arms"])),
        "n_profiled": word(len(facts["stage3b"]["arms"])),
        "evaluation_rows": f"{test['rows']:,}",
        "cluster_similarity": f"{settings['cluster_rho']:g}" if settings else markdown.MISSING,
        "concept_similarity": f"{concept_similarity:g}",
        "baseline_n_features": word(base.get("n_features")),
        **_target_parameters(run),
    }


def _target_parameters(run: Run) -> dict[str, str]:
    """What a component says about the target: the classes and the held-out positive rate for a
    binary target, and the target's name for a continuous one."""
    dataset = run.dataset
    if dataset.target_kind == BINARY:
        return {"positive_rate": markdown.percentage(
                    dataset.split_counts[TEST_SPLIT]["positive_rate"], 1),
                "positive_class": dataset.classes[1].name,
                "negative_class": dataset.classes[0].name}
    return {"target_name": dataset.record["target"]["name"]}
