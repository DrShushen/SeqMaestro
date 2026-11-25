"""How much of each feature is the thing a split groups on, checked before a run.

    experiments/analysis/feature_group_dependence.py \
        --dataset data/<dataset>/<file>.csv --sequence-column <sequence> \
        --group-column <group> --pools positional kmer --k-range 1 6 --kmer-normalise counts \
        --features experiments/runs/<run>/results/<model>/<pool>/real/seed_<n>/features.json

A grouped split promises that the same input never appears on both sides of it. **Whether that
promise holds depends on the pool of features.** For example, a split grouped by sequence window
can hold for positional features and fail for k-mer ones: two different windows of one gene share
nearly every k-mer, so a k-mer model recognises the gene across the boundary the grouping drew, and
can score well for that reason alone.

This check finds such a leak before any arm is fitted, and it needs no model fit.

**The measure is the share of a feature's variance explained by group identity**: a one-way ANOVA
eta-squared, `SS_between / SS_total`, with the split's grouping variable as the factor.

- **Near 0**: the feature varies within a group, so it describes the row. It cannot carry a
  group-level confound across a split.
- **Near 1**: the feature is constant within a group and differs between groups. It is a group
  indicator under another name, and grouping on that variable will not stop a model using it.

**A high value does not show that a feature is useless.** If the differences between groups are
the biology, a group-dependent feature measures biology. A high value says that the feature's
information is at the level of the group, so a split grouped on that variable cannot be trusted to
have isolated it, and a matched design is needed.

**A pool is featurised as its arm would featurise it.** `--k-range` and `--kmer-normalise` say how
k-mers are counted, and a pool with k-mers needs both, as the run's config states them: counts
grow with a sequence's length, so on sequences of different lengths they can depend on a group
through its lengths alone. Where the sequences differ in length, a positional pool uses the
positional window the dataset declares, which `--positional-window` gives, because the table this
check reads may not have a `dataset.json` yet. Without the flag, a positional pool is refused on
such sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from seqmaestro.features import (
    COVARIATES,
    FIXED_POOLS,
    MOTIF_DISCOVERED,
    MOTIF_KNOWN,
    PositionalWindow,
    make_featuriser,
    parse_pool,
)
from seqmaestro.features.base import Inputs
from seqmaestro.features.kmer import NORMALISATIONS
from seqmaestro.harness.features import load_features
from seqmaestro.proposer.featuriser import GeneratedFeaturiser


def eta_squared(X: pd.DataFrame, groups: np.ndarray) -> pd.Series:
    """The share of each column's variance explained by `groups`, vectorised over columns.

    Columns with no variance return 0 instead of raising a division warning, because a constant
    feature holds no information about anything, group identity included.
    """
    codes, _ = pd.factorize(groups)
    n, n_groups = len(codes), int(codes.max()) + 1
    indicator = sparse.csr_matrix(
        (np.ones(n), (codes, np.arange(n))), shape=(n_groups, n)
    )
    sizes = np.asarray(indicator.sum(axis=1)).ravel()

    values = X.to_numpy(np.float64)
    mean = values.sum(0) / n
    ss_total = (values * values).sum(0) - n * mean**2
    group_sums = indicator @ values
    ss_between = (group_sums**2 / sizes[:, None]).sum(0) - n * mean**2

    share = np.divide(ss_between, ss_total, out=np.zeros_like(ss_total), where=ss_total > 1e-12)
    return pd.Series(np.clip(share, 0.0, 1.0), index=X.columns)


#: The pools this check can score: those whose columns follow from the sequences alone, by a rule
#: its command line states. Which motif collection a run scans with, and how it discovers motifs,
#: are settings of the run, which this check does not have at Stage 1, and a discovery uses the
#: labels as well, so the pools holding either motif source are left out.
POOLS = tuple(pool for pool in FIXED_POOLS
              if not {COVARIATES, MOTIF_KNOWN, MOTIF_DISCOVERED} & set(parse_pool(pool)))

#: How `--positional-window` says whether a shorter sequence is padded.
PAD_CHOICES = {"pad": True, "no-pad": False}


def positional_window(values: list[str] | None) -> PositionalWindow | None:
    """The window `--positional-window` states, as the declaration would state it, or None."""
    if values is None:
        return None
    length, anchor, pad = values
    if pad not in PAD_CHOICES:
        raise SystemExit(f"--positional-window: PAD is {' or '.join(PAD_CHOICES)}, not {pad!r}.")
    try:
        return PositionalWindow(length=int(length), anchor=anchor, pad=PAD_CHOICES[pad])
    except ValueError as error:
        raise SystemExit(f"--positional-window: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Built CSV holding the sequences and the grouping column.")
    parser.add_argument("--group-column", required=True,
                        help="The variable the split groups on, which is what this asks each feature about.")
    parser.add_argument("--sequence-column", required=True,
                        help="The column holding the sequences. Required: no dataset's name for it "
                             "is assumed.")
    parser.add_argument("--pools", nargs="*", default=[], choices=POOLS, metavar="POOL",
                        help=f"The pools to score: {', '.join(POOLS)}.")
    parser.add_argument("--k-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="The lengths of k-mer counted, as the run's config states them. A "
                             "pool with k-mers needs it.")
    parser.add_argument("--kmer-normalise", choices=list(NORMALISATIONS),
                        help="Count k-mers in total, or per k-mer position in the sequence, as "
                             "the run's config states. A pool with k-mers needs it.")
    parser.add_argument("--positional-window", nargs=3, metavar=("LENGTH", "ANCHOR", "PAD"),
                        help="The positional window the dataset declares: its length, `start` or "
                             "`end`, and `pad` or `no-pad`. A positional pool needs it where the "
                             "sequences differ in length.")
    parser.add_argument("--features", type=Path, nargs="*", default=[],
                        help="features.json files whose generated features should also be scored, "
                             "each listed in the output under its path as given here.")
    parser.add_argument("--out", type=Path, help="Write the per-feature table here as CSV.")
    args = parser.parse_args(argv)
    if not args.pools and not args.features:
        parser.error("give --pools, --features, or both")
    counted = [pool for pool in args.pools if "kmer" in parse_pool(pool)]
    if counted and (args.k_range is None or args.kmer_normalise is None):
        parser.error(f"{', '.join(counted)} holds k-mers, so --k-range and --kmer-normalise must "
                     f"say how they are counted")
    window = positional_window(args.positional_window)

    frame = pd.read_csv(args.dataset)
    for column in (args.sequence_column, args.group_column):
        if column not in frame.columns:
            raise SystemExit(f"{args.dataset} has no column {column!r}.")
    groups = frame[args.group_column].to_numpy()
    print(f"{len(frame):,} rows, {len(np.unique(groups)):,} groups of {args.group_column!r}")

    lengths = frame[args.sequence_column].str.len()
    positional = [pool for pool in args.pools if "positional" in parse_pool(pool)]
    if positional and window is None and lengths.nunique() > 1:
        raise SystemExit(
            f"The sequences in {args.dataset} differ in length, from {lengths.min():,} to "
            f"{lengths.max():,}, so {', '.join(positional)} needs the positional window the "
            f"dataset declares: pass --positional-window.")

    scored: dict[str, pd.Series] = {}
    for pool in args.pools:
        featuriser = make_featuriser(pool, k_range=args.k_range and tuple(args.k_range),
                                     window=window, kmer_normalise=args.kmer_normalise)
        X = featuriser.fit_transform(frame[args.sequence_column])
        scored[pool] = eta_squared(X, groups)
        del X
    for path in args.features:
        records, header = load_features(path)
        # The covariates the features' code reads, as the file records them.
        covariates = list((header.get("featuriser") or {}).get("covariate_columns") or ())
        absent = [column for column in covariates if column not in frame.columns]
        if absent:
            raise SystemExit(f"{path} reads covariate(s) {absent}, which {args.dataset} does not have.")
        featuriser = GeneratedFeaturiser.from_records(records, sequence_column=args.sequence_column)
        # Named by the file's path as it was given, because the files of two arms at one seed are
        # in directories of the same name.
        scored[str(path)] = eta_squared(
            featuriser.fit_transform(Inputs(frame[args.sequence_column], frame[covariates])), groups
        )

    print(f"\n{'pool':<24}{'n':>7}{'median':>9}{'p90':>8}{'max':>8}{'>0.25':>8}{'>0.5':>8}")
    summary = {}
    for name, share in scored.items():
        summary[name] = {
            "n": len(share), "median": round(float(share.median()), 4),
            "p90": round(float(share.quantile(0.9)), 4), "max": round(float(share.max()), 4),
            "share_above_0.25": round(float((share > 0.25).mean()), 4),
            "share_above_0.5": round(float((share > 0.5).mean()), 4),
        }
        print(f"{name:<24}{len(share):>7}{share.median():>9.3f}{share.quantile(.9):>8.3f}"
              f"{share.max():>8.3f}{(share > 0.25).mean():>8.1%}{(share > 0.5).mean():>8.1%}")

    for name, share in scored.items():
        if len(share) <= 40:
            print(f"\n--- {name}, every feature ---")
            for feature, value in share.sort_values(ascending=False).items():
                print(f"  {value:.3f}  {feature}")

    if args.out:
        table = pd.concat(scored, names=["pool", "feature"]).rename("eta_squared")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out)
        args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWritten to {args.out}")
    return 0
