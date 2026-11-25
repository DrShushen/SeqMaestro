#!/usr/bin/env python
"""Finalise a prepared dataset: count it, check it, and write its `dataset.json`.

    python experiments/prepare/finalise_dataset.py --splits data/<dataset>/<splits> --declared <file>
    python experiments/prepare/finalise_dataset.py --splits data/<dataset>/<splits> --check

This is the last step of Stage 1 for every dataset, whichever script built it. The declaration
states what only the builder knows: the columns, the kind of target and what it means (each class,
or what the number measures), what each covariate measures, the window, the grouping, and every
filter with the rows it removed. This script counts and checks everything else from the files. The
logic is in `seqmaestro.dataset`, whose `validate_declared` lists the declaration's fields and
`run_checks` the checks, and `docs/developer/dataset.md` explains the record.

The Stage 1 checkpoint shows the printed summary to the user before Stage 2 may start. The summary
can report two kinds of problem:

- A **failing** check means the dataset is broken. The file is still written, so the failure can be
  read, but Stage 2 refuses the dataset until the check passes or the declaration accepts it with a
  reason.
- A **flagged** finding means the dataset has a property that only somebody who knows how it was
  produced can judge. It does not stop a run, but it is printed here with the question it raises,
  so that it is settled at Stage 1 instead of being rediscovered in a report.

In either case the script exits non-zero, so neither can scroll past unanswered.

`--check` reads an existing `dataset.json` back and prints the same summary without rewriting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from seqmaestro.dataset import (
    DATASET_FILE,
    FAILED,
    SPLITS,
    DatasetError,
    failed_checks,
    finalise,
    flagged_checks,
    target_kind,
)
from seqmaestro.provenance import git_is_dirty, git_sha

REPO = Path(__file__).resolve().parents[2]

#: The question each advisory check puts to the user. The check can measure the property, but only
#: somebody who knows how the data was produced can say whether it is a defect, so the summary ends
#: each finding with a question, which is answered at the Stage 1 checkpoint.
QUESTIONS = {
    "no_sequence_in_two_splits":
        "Is the same sequence appearing more than once expected here (one measurement repeated "
        "across replicates, one context hit by several events), or is it a duplicate that should "
        "not be there? If it is expected, say so and say what it means for a held-out score, since "
        "a model has seen those rows already. If it is not, the split or the build needs fixing.",
    "sequence_lengths":
        "Do the lengths go with the target (differ between the classes, or correlate with the "
        "value) because of the biology the task asks about, or because of how the rows were "
        "collected? A model can take a sequence's length from its k-mer counts and from padded "
        "positions, so a length that goes with the target can predict it on its own. Say which it "
        "is, and whether the design should match lengths instead. Where a positional window is "
        "declared, confirm that padding shorter sequences, or filtering them out, is the decision "
        "you want.",
    "target_distribution":
        "Are the target's tails the signal the task is about, or the noise of how it was measured "
        "(a low count, a detection limit)? R-squared weighs a row by its squared distance from the "
        "mean, so a handful of rows in a tail can decide a model's score. Say which, and whether "
        "the target should be transformed or filtered, as a recorded step before anything is "
        "fitted. Where one value covers many rows, say what it means (no change, below detection) "
        "and whether those rows belong in the task.",
}


def summary(record: dict) -> str:
    """The dataset as the Stage 1 checkpoint presents it: the target, the splits, the filters, the
    checks and the design power."""
    binary = target_kind(record) == "binary"
    lines = []
    if binary:
        for label in ("1", "0"):
            entry = record["classes"][label]
            lines.append(f"class {label}  {entry['name']}: {entry['meaning']}")
    else:
        target = record["target"]
        lines.append(f"target     {target['name']}, continuous: {target['meaning']}")
    sequences = record["sequences"]
    lines.append(f"sequences  {sequences['window']}; anchored on {sequences['anchor']}; "
                 f"alphabet {''.join(sequences['alphabet'])}; length "
                 f"{sequences['observed_length']['min']} to {sequences['observed_length']['max']}")
    window = sequences.get("positional_window")
    if window:
        lines.append(f"window     positional features read {window['length']} bases from the "
                     f"{window['anchor']}; shorter sequences "
                     + ("padded" if window["pad"] else "not allowed (no padding)"))
    for name, covariate in (record.get("covariates") or {}).items():
        spread = covariate.get("training")
        range_text = (f"; training {spread['min']:g} to {spread['max']:g}, "
                      f"{spread['distinct']:,} distinct" if spread else "")
        lines.append(f"covariate  {name}: {covariate['description']}{range_text}")
    unused = record["columns"].get("unused")
    if unused:
        lines.append(f"unused     columns in the split files that nothing reads: {', '.join(unused)}")
    split = record["split"]
    lines.append(f"split      {split['method']}; grouped by {', '.join(split['grouped_by']) or 'nothing'}; "
                 f"stratified on {', '.join(split['stratified_on']) or 'nothing'}")
    lines.append("")
    lines += _split_table(record, binary)
    if record["filters"]:
        lines.append("")
        lines += _filter_lines(record["filters"], binary)
    lines.append("")
    lines.append("checks:")
    for name, result in record["checks"].items():
        answer = f"  [{result['acknowledged']}]" if "acknowledged" in result else ""
        lines.append(f"  {result['status']:<15} {name}: {result['detail']}{answer}")
    power = record["design_power"]
    lines.append("")
    if binary:
        lines.append(f"design power  test: smallest detectable AUROC "
                     f"{power['test']['smallest_detectable_auroc']} "
                     f"({power['test']['n_positive']:,} positive, {power['test']['n_negative']:,} "
                     f"negative)")
    else:
        lines.append(f"design power  test: smallest detectable correlation "
                     f"{power['test']['smallest_detectable_r']} (r-squared "
                     f"{power['test']['smallest_detectable_r2']}) over {power['test']['rows']:,} rows")
    train = power["train"]
    weighted = (f", {train['effective_n_counting_rows']:,} if the weights were ignored"
                if train["weighted"] else "")
    lines.append(f"              train: effective n {train['effective_n']:,}{weighted}")
    return "\n".join(lines)


def _split_table(record: dict, binary: bool) -> list[str]:
    """Each split's rows, share and groups, with its classes or its target's distribution."""
    if binary:
        lines = [f"{'split':<6} {'rows':>9} {'share':>6} {'positive':>9} {'negative':>9} "
                 f"{'rate':>6} {'groups':>7}"]
    else:
        lines = [f"{'split':<6} {'rows':>9} {'share':>6} {'mean':>10} {'sd':>10} {'median':>10} "
                 f"{'groups':>7}"]
    for name in SPLITS:
        counts = record["splits"][name]
        groups = "" if counts["groups"] is None else str(counts["groups"])
        if binary:
            middle = (f"{counts['positive']:>9,} {counts['negative']:>9,} "
                      f"{counts['positive_rate']:>6.3f}")
        else:
            target = counts["target"] or {}
            middle = " ".join(f"{target.get(key, float('nan')):>10.4g}"
                              for key in ("mean", "sd", "median"))
        lines.append(f"{name:<6} {counts['rows']:>9,} {counts['share']:>6.3f} {middle} {groups:>7}")
    return lines


def _filter_lines(filters: list[dict], binary: bool) -> list[str]:
    """Each filter's rows before and after it, per class for a binary target."""
    if not binary:
        return ["filters, rows before and after each step:"] + [
            f"  {step['step']}: {step['rows_in']:,} -> {step['rows_out']:,}  "
            f"(lost {step['rows_in'] - step['rows_out']:,})" for step in filters]
    lines = ["filters, rows per class (1 / 0) before and after each step:"]
    for step in filters:
        before, after = step["rows_in"], step["rows_out"]
        lost = (before["1"] - after["1"], before["0"] - after["0"])
        lines.append(f"  {step['step']}: {before['1']:,} / {before['0']:,} -> "
                     f"{after['1']:,} / {after['0']:,}  (lost {lost[0]:,} / {lost[1]:,})")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=Path, required=True,
                        help="the directory holding train.csv, val.csv and test.csv")
    parser.add_argument("--declared", type=Path, default=None,
                        help="the builder's declaration, YAML or JSON; see docs/developer/dataset.md")
    parser.add_argument("--check", action="store_true",
                        help="print the summary of the existing dataset.json without rewriting it")
    args = parser.parse_args(argv)

    if args.check == (args.declared is not None):
        parser.error("give exactly one of --declared and --check")

    try:
        if args.check:
            path = args.splits / DATASET_FILE
            if not path.exists():
                print(f"{path} does not exist; finalise the dataset with --declared first.",
                      file=sys.stderr)
                return 2
            record = json.loads(path.read_text())
        else:
            declared = yaml.safe_load(args.declared.read_text())
            record = finalise(args.splits, declared, provenance={
                "finalised_by": "experiments/prepare/finalise_dataset.py",
                "declaration": str(args.declared),
                "git_sha": git_sha(REPO),
                "git_dirty": git_is_dirty(REPO),
            })
    except (DatasetError, OSError, yaml.YAMLError) as error:
        print(error, file=sys.stderr)
        return 2

    print(summary(record))
    if not args.check:
        print(f"\nwritten to {args.splits / DATASET_FILE}")

    flagged = flagged_checks(record)
    if flagged:
        print(f"\n{len(flagged)} finding(s) need the judgement of somebody who knows this data. "
              "Nothing here says the dataset is wrong. These are properties that mean different "
              "things on different designs, so they are put to you instead of being decided here. Answer "
              "each in the declaration under `acknowledged`, and the answer travels with the dataset "
              "and is printed in the report.", file=sys.stderr)
        for name, result in flagged.items():
            print(f"\n  {name}\n    {result['detail']}\n    -> {QUESTIONS[name]}", file=sys.stderr)

    failed = failed_checks(record)
    if failed:
        print(f"\n{len(failed)} check(s) {FAILED}. Stage 2 will refuse this dataset until each passes "
              "or the declaration accepts it under `acknowledged`, with a reason.", file=sys.stderr)
    return 1 if (failed or flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
