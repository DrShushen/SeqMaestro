"""Test discovered features against the label directly, matched and permuted within a confounder.

    experiments/analysis/direct_effect_test.py --dataset data/<dataset>/<file>.csv \\
        --label-column <target> \\
        --features experiments/runs/<run>/results/<model>/<pool>/real/seed_*/features.json \\
        --group-column <confounder> --out <path>.csv

The features are compiled from the code each cell recorded in its `features.json`, so the test runs
on the features the arm used, and no other part of a run is read. The driver does not start it.

`--group-column` is the confounder the split already groups on. When it is given, the classes are
matched 1:1 within every level of it, and the null distribution shuffles the labels within each
level. `--no-match` keeps every row and stratifies the null only. `--within` repeats the whole
analysis within each level of a second column, which checks whether a feature is only another view
of that column: a feature that holds within every level of it is not. `--columns` tests columns of
the dataset instead of, or beside, proposed features, which needs no run at all.

**Multiplicity is not corrected.** These are a handful of named features, which the caller has
already judged worth testing, so correcting for a set that was not chosen by scanning would only
lose power. If the set *was* chosen by scanning, correct across it yourself.

The command prints the table and the design, and with `--out` it writes the table as CSV and the
design beside it as JSON. `docs/developer/analyses/direct_measurement.md` explains the test and its
output.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from seqmaestro.features.base import Inputs
from seqmaestro.harness.effect import (
    DEFAULT_DRAWS,
    DEFAULT_MATCH_REPEATS,
    direct_effect,
    matched_effect,
)
from seqmaestro.harness.features import DEFAULT_TOP_N, load_features, top_features
from seqmaestro.pipeline.repository import REPO
from seqmaestro.proposer import GeneratedFeaturiser
from seqmaestro.provenance import record_invocation


def gather_records(paths: list[Path], kinds: set[str], top_n: int | None
                   ) -> tuple[list, str | None, tuple[str, ...]]:
    """The features to test, pooled across files and deduplicated by name.

    Several files are the normal case, because each seed of an arm reports its own set, and a
    feature worth a direct test is usually one that turned up in more than one. Importance is
    compared by its size, whatever its sign. The most important copy of a repeated name is kept,
    and its code is the code compiled, which matters where two files give one name different code.

    It also returns the sequence column and the covariates the features were written against,
    taken from the files rather than assumed, because generated code indexes those columns by name,
    so a wrong guess is a crash at best and a silently different feature at worst.
    """
    best: dict[str, object] = {}
    columns, covariates = set(), set()
    for path in paths:
        records, header = load_features(path)
        described = header.get("featuriser") or {}
        columns.add(described.get("sequence_column"))
        covariates.add(tuple(described.get("covariate_columns") or ()))
        if kinds:
            records = [r for r in records if r.kind in kinds]
        for record in top_features(records, top_n or max(len(records), 1)):
            seen = best.get(record.name)
            if seen is None or abs(record.importance or 0.0) > abs(seen.importance or 0.0):
                best[record.name] = record
    columns.discard(None)
    if len(columns) > 1:
        raise SystemExit(f"The feature files disagree about the sequence column: {sorted(columns)}. "
                         f"They came from different runs; test them separately.")
    if len(covariates) > 1:
        raise SystemExit(f"The feature files disagree about the covariates: {sorted(covariates)}. "
                         f"They came from different arms or runs; test them separately.")
    return (list(best.values()), (columns.pop() if columns else None),
            (covariates.pop() if covariates else ()))


def evaluate(records: list, frame: pd.DataFrame, sequence_column: str,
             covariates: tuple[str, ...]) -> tuple[list[str], np.ndarray]:
    """Compile the proposed features and evaluate them on every row of the dataset.

    Fitting on the rows it reports on would be wrong for a model, but it is correct here, because
    nothing is being trained: `fit` only decides which features compile and vary at all. Constant
    features are dropped, because a feature that takes one value has no class difference to measure.
    """
    missing = [column for column in (sequence_column, *covariates) if column not in frame.columns]
    if missing:
        raise SystemExit(f"The features read column(s) {missing}, which the dataset does not have.")
    featuriser = GeneratedFeaturiser.from_records(records, sequence_column=sequence_column)
    values = featuriser.fit_transform(Inputs(frame[sequence_column], frame[list(covariates)]))
    dropped = featuriser.fit_report.get("dropped", {})
    if any(dropped.values()):
        print(f"  {featuriser.fit_report['n_kept']} of {featuriser.fit_report['n_proposed']} "
              f"compiled and varied; dropped {dict(dropped)}")
    return [record.name for record in featuriser.records], values.to_numpy(dtype=float)


def analyse(values: np.ndarray, names: list[str], label: np.ndarray, groups, *,
            match: bool, repeats: int, draws: int, seed: int) -> pd.DataFrame:
    """One table: match if asked, then contrast the classes against a within-group null."""
    if len(np.unique(label)) != 2:
        return pd.DataFrame()
    if groups is not None and match:
        return matched_effect(values, label, groups, names,
                              repeats=repeats, draws=draws, seed=seed)
    return direct_effect(values, label, names, strata=groups, draws=draws, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--features", type=Path, nargs="*", default=[],
                        help="One or more features.json; pooled and deduplicated by name.")
    parser.add_argument("--columns", nargs="*", default=[],
                        help="Numeric columns of the dataset to test as features.")
    parser.add_argument("--sequence-column", default=None,
                        help="The column the features' code indexes. Needed only where the feature "
                             "files do not record it.")
    parser.add_argument("--kinds", nargs="*", default=[],
                        help="Feature kinds to keep; empty for all.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Features per file, by importance. 0 for all.")
    parser.add_argument("--group-column", default=None,
                        help="The confounder: matched on, and the null's strata.")
    parser.add_argument("--no-match", action="store_true",
                        help="Stratify the null only; keep every row in the estimate.")
    parser.add_argument("--within", default=None,
                        help="Repeat the analysis inside each level of this column.")
    parser.add_argument("--min-rows", type=int, default=50,
                        help="Skip a --within level holding fewer rows than this.")
    parser.add_argument("--match-repeats", type=int, default=DEFAULT_MATCH_REPEATS,
                        help="Matched samples drawn, to price the draw. Estimates only.")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="CSV path; JSON written beside it.")
    args = parser.parse_args(argv)

    if not args.features and not args.columns:
        parser.error("give --features, --columns, or both")

    frame = pd.read_csv(args.dataset)
    levels = frame[args.label_column].nunique()
    if levels != 2:
        parser.error(f"--label-column {args.label_column} holds {levels} distinct values, and the "
                     f"direct effect is a contrast between two classes. For a continuous target, "
                     f"a feature's association on its own is its marginal effect (Spearman's rho) "
                     f"and Tier 2 of the reliability panel.")
    label = (frame[args.label_column].to_numpy() == frame[args.label_column].max()).astype(int)

    names: list[str] = []
    blocks: list[np.ndarray] = []
    sequence_column = args.sequence_column
    if args.features:
        records, recorded, covariates = gather_records(args.features, set(args.kinds),
                                                       args.top_n or None)
        sequence_column = sequence_column or recorded
        if sequence_column is None:
            parser.error("the feature files do not record the column their code indexes; give "
                         "--sequence-column")
        print(f"{len(records)} feature(s) from {len(args.features)} file(s), "
              f"evaluated on `{sequence_column}`"
              + (f" and {', '.join(covariates)}" if covariates else "") + ":")
        compiled, values = evaluate(records, frame, sequence_column, covariates)
        names += compiled
        blocks.append(values)
    if args.columns:
        names += list(args.columns)
        blocks.append(frame[list(args.columns)].to_numpy(dtype=float))
    values = np.column_stack(blocks)

    groups = frame[args.group_column].to_numpy() if args.group_column else None
    design = "unstratified" if groups is None else (
        f"stratified on {args.group_column}" + ("" if args.no_match else ", matched 1:1 within it"))
    print(f"{len(frame):,} rows, {int(label.sum()):,} positive, {design}, {args.draws:,} draws")

    levels = [("(all rows)", np.ones(len(frame), dtype=bool))]
    if args.within:
        for level in sorted(frame[args.within].dropna().unique()):
            mask = (frame[args.within] == level).to_numpy()
            if mask.sum() >= args.min_rows:
                levels.append((f"{args.within}={level}", mask))

    tables = []
    for name, mask in levels:
        table = analyse(values[mask], names, label[mask],
                        None if groups is None else groups[mask],
                        match=not args.no_match, repeats=args.match_repeats,
                        draws=args.draws, seed=args.seed)
        if table.empty:
            print(f"\n{name}: no contrast after matching; skipped.")
            continue
        table.insert(0, "subset", name)
        tables.append(table)
        print(f"\n{name}   n={table.attrs['design']['n']:,}  "
              f"positive={table.attrs['design']['n_positive']:,}  "
              f"strata={table.attrs['design']['strata']}")
        shown = table.drop(columns=["subset", "binary"]).round(4)
        print(shown.to_string(index=False))

    if not tables:
        print("Nothing measurable.", file=sys.stderr)
        return 1
    combined = pd.concat(tables, ignore_index=True)

    print("\ndifference is the comparable column; ratio is how an enrichment reads and is left "
          "empty\nwhere it would carry a sign rather than a magnitude. p is two-sided and "
          "uncorrected.")
    if groups is not None and not args.no_match:
        print("ratio_draw_min/max is the same ratio with the matched sample drawn again, which the "
              "\ninterval does not cover. Where that range is wide the matching seed is doing the "
              "talking.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out, index=False)
        record_invocation(
            args.out.parent, repo_root=REPO, arguments=vars(args),
            filename=args.out.with_suffix(".json").name,
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            designs={name: table.attrs["design"] for name, table in
                     zip([t.subset.iloc[0] for t in tables], tables, strict=True)},
        )
        print(f"\nWritten to {args.out} and {args.out.with_suffix('.json')}")
    return 0
