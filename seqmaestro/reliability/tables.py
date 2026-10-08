"""The three tables a panel comes down to, and the per-arm summary a report is written from.

    profile_features.csv   one row per candidate feature, with all three tiers on it
    profile_clusters.csv   one row per cluster of near-duplicates
    profile_curves.csv     every threshold curve of Tiers 1 and 2, stacked with a `kind` column

They are assembled after the reveal, from the files the tiers wrote. A report reads its numbers from
these tables and not from the arrays behind them, so a number in the prose cannot drift from the one
in a table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seqmaestro.harness.heldout import TEST_SPLIT
from seqmaestro.reliability import prefilter, tier1, tier2, tier3
from seqmaestro.reliability.files import (
    PREPARE_ARRAYS,
    PREPARE_FILE,
    REAL,
    TIER3_FILE,
    load_arrays,
    load_tier2,
    read_json,
    tier1_name,
    tier2_name,
)
from seqmaestro.target import BINARY, CONTINUOUS, recorded_kind, target_for

FEATURES_FILE = "profile_features.csv"
CLUSTERS_FILE = "profile_clusters.csv"
CURVES_FILE = "profile_curves.csv"

#: A feature left out of the reduced model with pi-hat at or above this is an alternative worth
#: naming.
ALTERNATIVE_PI = 0.7

#: A reported feature with pi-hat below this is one the procedure would often not have chosen.
UNSTABLE_PI = 0.5

#: The column Tier 2's statistic takes in the features table, by the kind of target.
STATISTIC_COLUMN = {BINARY: "abs_auc_minus_half", CONTINUOUS: "abs_rho"}


def tier2_columns(target_kind: str) -> dict[str, str]:
    """Tier 2's per-feature arrays, as the features table names its columns: the association of the
    kind (`auc` or `rho`) and its statistic, then the p-values, and the feature's mean and non-zero
    rate on the test split."""
    association, _ = tier2.ASSOCIATION[target_kind]
    return {association: association, "statistic": STATISTIC_COLUMN[target_kind],
            "p_value": "p_permutation", "q_bh": "q_bh", "q_by": "q_by", "p_holm": "p_holm",
            "p_westfall_young": "p_westfall_young", "mean": f"mean_{TEST_SPLIT}",
            "nonzero_rate": f"nonzero_rate_{TEST_SPLIT}"}


def loco_columns(target_kind: str) -> dict[str, str]:
    """Tier 3's per-refit results, as the features table names its columns: the change in the kind's
    loss and headline, then the tests."""
    loss, _ = tier3.LOSS[target_kind]
    headline = target_for(target_kind).headline
    return {f"mean_delta_{loss}": f"loco_mean_delta_{loss}", "ci_low": "loco_ci_low",
            "ci_high": "loco_ci_high", f"delta_{headline}": f"loco_delta_{headline}",
            "p_t_one_sided": "loco_p_one_sided", "p_sign_flip": "loco_p_sign_flip",
            "p_holm": "loco_p_holm", "q_bh": "loco_q_bh", "verdict": "loco_verdict"}


def _called(called: dict, level: float, method: str) -> int:
    """`called` keys survive a JSON round trip as strings; read either."""
    return int(called[level if level in called else str(level)][method])


def features_table(arm_dir: Path, record: dict) -> pd.DataFrame:
    """One row per candidate feature, with all three tiers on it."""
    kind = recorded_kind(record)
    association, _ = tier2.ASSOCIATION[kind]
    prepared = load_arrays(arm_dir / PREPARE_ARRAYS)
    pi = load_arrays(arm_dir / f"{tier1_name(REAL)}.npz")
    names = [str(name) for name in prepared["names"]]
    kept = prepared["kept"]
    kept_index = np.flatnonzero(kept)
    reported = set(record["reported"])

    frame = pd.DataFrame({
        "feature": names,
        "kind": [str(kind) for kind in prepared["kinds"]],
        "description": [str(text) for text in prepared["descriptions"]],
        "filtered_out": ~kept,
        "modal_share_D": prepared["modal_share"],
        "nonzero_rate_D": prepared["nonzero_rate"],
        "cluster_id": prepared["cluster"],
        "position_group": [str(group) for group in prepared["position_group"]],
        "pi_hat": pi["pi_hat"],
        "cluster_pi_hat": pi["cluster_pi_hat"],
        "in_reported_model": [name in reported for name in names],
    })
    # A candidate the prefilter removed was never offered to the selection, so Tier 1 has no value
    # for it. It gets NaN, because a zero would say "never chosen" and blame the feature for a
    # decision the filter made.
    frame.loc[frame["filtered_out"], ["pi_hat", "cluster_pi_hat"]] = np.nan
    group_pi = dict(zip([str(g) for g in pi["group_labels"]], pi["group_pi"].tolist(), strict=True))
    frame["position_group_pi_hat"] = [group_pi.get(group, np.nan) for group in frame["position_group"]]

    direction = np.full(len(names), "", dtype=object)
    if record["tier2"]["refused"]:
        for column in tier2_columns(kind).values():
            frame[column] = np.nan
    else:
        result = load_tier2(arm_dir, tier2_name(TEST_SPLIT, REAL))
        if result["names"] != [names[i] for i in kept_index]:
            raise AssertionError(f"{arm_dir}: Tier 2's columns are not the prefiltered pool.")
        for key, column in tier2_columns(kind).items():
            values = np.full(len(names), np.nan)
            values[kept_index] = result[key]
            frame[column] = values
        direction[kept_index] = [str(d) for d in result["direction"]]
    frame[f"direction_{TEST_SPLIT}"] = direction

    representative = _representatives(names, kept_index, prepared, pi["pi_hat"])
    frame["cluster_representative"] = [representative.get(int(c), "") for c in frame["cluster_id"]]
    sizes = {int(label): int((prepared["cluster"] == label).sum())
             for label in np.unique(prepared["cluster"])}
    frame["cluster_size"] = [sizes[int(label)] for label in prepared["cluster"]]

    loco = {row["label"]: row for row in read_json(arm_dir / TIER3_FILE)["rows"]
            if row["kind"] == "feature"}
    for key, column in loco_columns(kind).items():
        missing = "" if key == "verdict" else np.nan
        frame[column] = [loco.get(name, {}).get(key, missing) for name in names]

    return frame.sort_values(["in_reported_model", "pi_hat", association],
                             ascending=[False, False, False]).reset_index(drop=True)


def _representatives(names, kept_index, prepared, pi_hat) -> dict[int, str]:
    kinds = [str(prepared["kinds"][i]) for i in kept_index]
    kept_names = [names[i] for i in kept_index]
    chosen = prefilter.representatives(prepared["cluster"][kept_index], pi_hat[kept_index], kinds,
                                       kept_names)
    return {label: kept_names[index] for label, index in chosen.items()}


def clusters_table(features: pd.DataFrame, target_kind: str) -> pd.DataFrame:
    """One row per cluster: how stable it is, what is in it, and what it is worth on its own."""
    association, centre = tier2.ASSOCIATION[target_kind]
    kept = features[~features["filtered_out"]]
    rows = []
    for cluster_id, block in kept.groupby("cluster_id"):
        block = block.sort_values("pi_hat", ascending=False)
        rows.append({
            "cluster_id": int(cluster_id),
            "size": len(block),
            "representative": block["cluster_representative"].iloc[0],
            "cluster_pi_hat": float(block["cluster_pi_hat"].iloc[0]),
            "max_member_pi_hat": float(block["pi_hat"].max()),
            "n_reported": int(block["in_reported_model"].sum()),
            "reported": "; ".join(block.loc[block["in_reported_model"], "feature"]),
            "kinds": "/".join(sorted(set(block["kind"]))),
            f"best_{association}": float(
                block.loc[block[association].sub(centre).abs().idxmax(), association])
            if block[association].notna().any() else np.nan,
            "min_q_bh": float(block["q_bh"].min()) if block["q_bh"].notna().any() else np.nan,
            "members": "; ".join(block["feature"].head(12)) + ("; ..." if len(block) > 12 else ""),
        })
    return pd.DataFrame(rows).sort_values(
        ["n_reported", "cluster_pi_hat"], ascending=[False, False]).reset_index(drop=True)


def curves_table(arm_dir: Path, record: dict) -> pd.DataFrame:
    """Every threshold curve of the panel, stacked: tau, q and alpha rows in one file."""
    summary = read_json(arm_dir / f"{tier1_name(REAL)}.json")
    rows = []
    for level, key in (("feature", "feature_level"), ("cluster", "cluster_level"),
                       ("position_group", "position_group_level")):
        if summary[key] is None:
            continue
        for entry in summary[key]["curve"]:
            rows.append({"kind": "tier1_tau", "level": level, "x": entry["tau"],
                         "n_selected": entry["n_selected"],
                         "bound_expected_false": entry["bound_expected_false"],
                         "p": summary[key]["p"], "q_hat": summary[key]["q_hat"]})
    if not record["tier2"]["refused"]:
        result = read_json(arm_dir / f"{tier2_name(TEST_SPLIT, REAL)}.json")
        for level, counts in result["called"].items():
            for method, count in counts.items():
                rows.append({"kind": "tier2_called", "level": method, "x": float(level),
                             "n_selected": count})
        for entry in result["plug_in_fdr"]:
            rows.append({"kind": "tier2_plug_in_fdr", "level": "raw_p", "x": entry["alpha"],
                         "n_selected": entry["n_called"],
                         "expected_null_called": entry["expected_null_called"],
                         "plug_in_fdr": entry["plug_in_fdr"]})
    return pd.DataFrame(rows)


def summary(arm_dir: Path, record: dict, features: pd.DataFrame) -> dict:
    """The per-arm numbers a report quotes, computed once and read back from here."""
    prepared = read_json(arm_dir / PREPARE_FILE)
    levels = read_json(arm_dir / f"{tier1_name(REAL)}.json")
    refits = read_json(arm_dir / TIER3_FILE)
    reported = features[features["in_reported_model"]]
    kept = features[~features["filtered_out"]]
    alternatives = kept[(kept["pi_hat"] >= ALTERNATIVE_PI) & ~kept["in_reported_model"]]
    q_hat = levels["feature_level"]["q_hat"]
    pool = record["candidates"]
    headline = target_for(recorded_kind(record)).headline

    tier2_numbers = {"tier2_refused": record["tier2"]["refused"] is not None}
    if not record["tier2"]["refused"]:
        result = read_json(arm_dir / f"{tier2_name(TEST_SPLIT, REAL)}.json")
        tier2_numbers.update({
            "n_called_bh_0.05": _called(result["called"], 0.05, "bh"),
            "n_called_wy_0.05": _called(result["called"], 0.05, "westfall_young"),
            "n_reported_called_bh_0.05": int((reported["q_bh"] <= 0.05).sum()),
            "n_test_rows": result["n_rows"], "n_permutations": result["n_permutations"]})
    return {
        "model": record["model"], "pool": record["pool"],
        "seed": record["seed"], "pool_source": pool["source"], "p_pool": pool["p"],
        "p_after_prefilter": pool["p_after_prefilter"], "n_clusters": pool["n_clusters"],
        "n_reported": len(record["reported"]), "n_pairs": record["settings"]["pairs"],
        # None where the check was not run, for a model that is not reproducible.
        "reproduced_from_pool": prepared["reproduction"].get("from_pool", {}).get("identical"),
        "q_hat": q_hat,
        "bound_at_0.8": tier1.expected_false_positives(q_hat, pool["p_after_prefilter"], 0.8),
        "tau_bound_1": levels["feature_level"]["tau_bound_1"],
        "cluster_q_hat": levels["cluster_level"]["q_hat"],
        "cluster_bound_at_0.8": tier1.expected_false_positives(
            levels["cluster_level"]["q_hat"], levels["cluster_level"]["p"], 0.8),
        "median_reported_pi": float(reported["pi_hat"].median()),
        "n_reported_pi_ge_0.7": int((reported["pi_hat"] >= ALTERNATIVE_PI).sum()),
        "n_reported_pi_lt_0.5": int((reported["pi_hat"] < UNSTABLE_PI).sum()),
        "n_alternatives": len(alternatives),
        **tier2_numbers,
        "n_necessary": sum(1 for row in refits["rows"]
                           if row["kind"] == "feature" and row["verdict"] == "necessary"),
        "n_loco_tests": refits["n_tests"],
        f"{headline}_reduced_model": refits[f"{headline}_model"],
    }


def write(arm_dir: str | Path, record: dict) -> dict:
    """Write the three tables beside the arm's other files, and return its summary."""
    arm_dir = Path(arm_dir)
    features = features_table(arm_dir, record)
    features.to_csv(arm_dir / FEATURES_FILE, index=False)
    clusters_table(features, recorded_kind(record)).to_csv(arm_dir / CLUSTERS_FILE, index=False)
    curves_table(arm_dir, record).to_csv(arm_dir / CURVES_FILE, index=False)
    return summary(arm_dir, record, features)
