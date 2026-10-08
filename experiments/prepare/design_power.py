#!/usr/bin/env python
"""What a design can detect, worked out before a run is spent on it.

    python experiments/prepare/design_power.py --n-positive 334 --n-negative 334
    python experiments/prepare/design_power.py --n-rows 11000
    python experiments/prepare/design_power.py --splits data/<dataset>/<splits> \
        --target-column <target> --target-kind binary|continuous [--weight-column <weights>] \
        [--out <path>.csv]

A planning tool for Stage 1. `finalise_dataset.py` records these numbers in every finalised
dataset's `dataset.json`, and reports quote them from there. This script answers the earlier
question: what could a split of a given size, or each of several candidate splits, show?

A full grid costs tens of millions of tokens and several hours. A design whose test set cannot
separate the effect from chance spends all of that and reports near chance with a wide interval,
which cannot be told apart from the effect being absent. The arithmetic here takes seconds.

It reports two numbers, and a design can fail on either:

- **Test-set precision.** For a binary target, the interval on a held-out AUROC, from the test
  split's class counts. For a continuous target, the interval on a held-out correlation between
  predictions and targets, from its row count. Metrics are computed on unweighted rows, so this
  ignores any fitting weights. This is what decides whether a result can be *stated*.
- **Training information.** The effective sample size of the training split, which for a weighted
  design is Kish's figure (per class and combined harmonically, for a binary target) and not the
  row count. This is what decides whether there is anything to state.

The arithmetic is in `seqmaestro.harness.power`, and a split is sized by its kind's object in
`seqmaestro.target`. This script formats the table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seqmaestro.harness.power import (
    auroc_half_width,
    correlation_half_width,
    smallest_detectable_auroc,
    smallest_detectable_correlation,
)
from seqmaestro.target import AUROCS_AT, BINARY, CORRELATIONS_AT, TARGET_KINDS, target_for


def describe(name: str, n_positive: float, n_negative: float) -> dict:
    """One binary design's power, as the row of a table."""
    return {
        "design": name,
        "n_positive": round(float(n_positive), 1),
        "n_negative": round(float(n_negative), 1),
        "effective_n": round(float(2 / (1 / n_positive + 1 / n_negative)), 1)
        if n_positive and n_negative else 0.0,
        "smallest_detectable_auroc": round(smallest_detectable_auroc(n_positive, n_negative), 3),
        **{f"half_width_at_{a:.2f}": round(auroc_half_width(a, n_positive, n_negative), 4)
           for a in AUROCS_AT},
    }


def describe_rows(name: str, n_rows: float) -> dict:
    """One continuous design's power, as the row of a table."""
    smallest = smallest_detectable_correlation(n_rows)
    return {
        "design": name,
        "rows": int(n_rows),
        "smallest_detectable_r": round(smallest, 3),
        "smallest_detectable_r2": round(smallest ** 2, 4),
        **{f"half_width_at_{r:.2f}": round(correlation_half_width(r, n_rows), 4)
           for r in CORRELATIONS_AT},
    }


def from_splits(directory: Path, target_column: str, target_kind: str,
                weight_column: str | None) -> pd.DataFrame:
    """Both numbers for a split directory: what its test set can show, what its train set holds."""
    test = pd.read_csv(directory / "test.csv")
    train = pd.read_csv(directory / "train.csv")
    weights = (train[weight_column].to_numpy(dtype=float)
               if weight_column and weight_column in train.columns else None)
    target = target_for(target_kind)
    power = {"test": target.held_out_power(test[target_column].to_numpy()),
             "train": target.training_information(train[target_column].to_numpy(), weights)}
    held = power["train"]
    note = ("unweighted" if not held["weighted"] else
            f"row counts ignoring the weights would say {held['effective_n_counting_rows']:,.0f}")
    stated = f"{directory.name}/test (what can be stated)"
    there = f"{directory.name}/train (what is there to state)"
    unweighted = "metrics are unweighted, so any fitting weights do not enter here"

    if target_kind == BINARY:
        rows = [{**describe(stated, power["test"]["n_positive"], power["test"]["n_negative"]),
                 "note": unweighted},
                {"design": there, "n_positive": max(held["n_positive"], 1),
                 "n_negative": max(held["n_negative"], 1), "effective_n": held["effective_n"],
                 "smallest_detectable_auroc": None, "note": note}]
    else:
        rows = [{**describe_rows(stated, power["test"]["rows"]), "note": unweighted},
                {"design": there, "rows": held["rows"], "effective_n": held["effective_n"],
                 "smallest_detectable_r": None, "note": note}]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=Path, nargs="*", default=[],
                        help="Split directories to size up. Repeatable, to compare designs.")
    parser.add_argument("--n-positive", type=float, default=None)
    parser.add_argument("--n-negative", type=float, default=None)
    parser.add_argument("--n-rows", type=float, default=None,
                        help="the held-out rows of a design with a continuous target")
    # No defaults: which column holds the target, and what kind of target it is, are facts about
    # the dataset, and a default here would be one dataset's answer applied silently to the next.
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--target-kind", choices=TARGET_KINDS, default=None)
    parser.add_argument("--weight-column", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the table as CSV.")
    args = parser.parse_args()

    if not args.splits and args.n_positive is None and args.n_rows is None:
        parser.error("give --splits, --n-positive with --n-negative, or --n-rows")
    if args.splits and not (args.target_column and args.target_kind):
        parser.error("--splits needs --target-column and --target-kind")

    frames = [from_splits(d, args.target_column, args.target_kind, args.weight_column)
              for d in args.splits]
    if args.n_positive is not None:
        negative = args.n_negative if args.n_negative is not None else args.n_positive
        frames.append(pd.DataFrame([describe("given", args.n_positive, negative)]))
    if args.n_rows is not None:
        frames.append(pd.DataFrame([describe_rows("given", args.n_rows)]))

    table = pd.concat(frames, ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    if "smallest_detectable_auroc" in table:
        print("\nsmallest_detectable_auroc: the lowest AUROC whose 95% interval clears 0.5 on this "
              "test set.")
    if "smallest_detectable_r" in table:
        print("\nsmallest_detectable_r: the lowest correlation between predictions and targets whose "
              "95% interval clears 0 on this test set.")
    print("An effect below it cannot be separated from chance however many arms are run.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
