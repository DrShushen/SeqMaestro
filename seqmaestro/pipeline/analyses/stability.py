"""Stability of every arm across seeds, against its own permuted cells.

    python experiments/analysis/analyse_stability.py --run-dir experiments/runs/<date>_<name>

`results/stability.csv` has one row per model, pool and label condition over the run's seeds, with
two views of each arm: the full set its model reports, and its top N by importance. **Report
`top_n_kappa`**, the agreement of the top N corrected for the arm's own chance level,
`(J_real - J_permuted) / (1 - J_permuted)`. `full_phi`, `top_n_phi`, `full_jaccard` and
`top_n_jaccard` are written for diagnosis and are not reported.
`docs/developer/analyses/stability_and_co_discovery.md` explains each number and how to interpret
it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from seqmaestro.harness import cell as cells
from seqmaestro.harness.stability import (
    chance_corrected_agreement,
    load_run,
    non_finite_columns,
    stability_views,
)
from seqmaestro.pipeline.config import ConfigError, load_recorded
from seqmaestro.pipeline.layout import RESULTS_DIR
from seqmaestro.provenance import record_call

#: The name of this call (`seqmaestro.pipeline.layout.call_name`), and the table it writes under
#: the run's `results/`.
CALL = "analyse_stability"
TABLE = "stability.csv"


def arms(results: Path) -> list[tuple[str, str, str]]:
    """Every (model, pool, labels) present, found from the run's directories instead of enumerated.

    A hardcoded list would silently skip every arm it does not name, such as the DEFT arms and the
    arms on written sources, which are the ones the comparison is about. A missing row looks the
    same as an arm that was never run, so the omission would go unnoticed.
    """
    found = set()
    for run_file in results.rglob(cells.RUN_FILE):
        seed_dir = run_file.parent
        found.add((seed_dir.parents[2].name, seed_dir.parents[1].name, seed_dir.parent.name))
    return sorted(found)


def universe_size(cell: Path, runs: list) -> int | None:
    """The Nogueira `d`, or `None` when the arm has no fixed vocabulary to quote one from.

    It is read from the cells instead of from a table here, so it cannot go stale when a
    featuriser's parameters change.

    **The test is whether the recorded vocabulary covers the features actually seen**, which is the
    estimator's precondition. A weaker test, such as the recorded count agreeing across seeds,
    is not enough. A *generative* arm records how many features it proposed for each seed, and
    different seeds propose different features, so the count can be identical across seeds while
    their union is larger. Given such a `d`, the estimator refuses outright, which is correct, and
    this test exists to avoid that failure.

    `None` means "no fixed vocabulary". The estimator then falls back to the union of what was
    discovered, which is a stated convention and never comparable with a supplied `d`.
    """
    recorded = {
        cells.read_run(seed_dir).get("n_features_available")
        for seed_dir in sorted(cell.iterdir())
        if (seed_dir / cells.RUN_FILE).exists()
    }
    if len(recorded) != 1 or (size := recorded.pop()) is None:
        return None

    # An upper bound on the distinct features the estimator will see. Behavioural matching can only
    # merge names, never split them, so a `d` clearing this clears the real count too.
    observed = len({name for run in runs for name in run.names})
    return size if size >= observed else None


def add_chance_corrected(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `top_n_kappa` to every row of an arm that has both label conditions.

    Kappa is a property of an *arm*, not of one of its label conditions, because it needs the
    real-label agreement and the permuted-label agreement of the same (model, pool). Both rows
    hold the same value, so that a filter on either condition still finds it. An arm missing one
    condition gets NaN instead of a number computed against an absent null.

    **Kappa is computed on the top-N view only.** The full view cannot support a chance-corrected
    figure, because an arm that keeps every feature it is given agrees with itself almost perfectly
    on shuffled labels too. The division by `1 - chance` then turns a small difference into a huge
    one, and where the permuted Jaccard reaches exactly 1.0 the figure is undefined. Both are
    artifacts of comparing sets of very different sizes, which the top-N view exists to remove. A
    column that mixes a well-behaved value for an arm that selects with a large negative one for an
    arm that does not is worse than no column.
    """
    if frame.empty:
        return frame

    chance = (frame[frame.labels == cells.PERMUTED]
              .set_index(["model", "pool"])["top_n_jaccard"])
    observed = (frame[frame.labels == cells.REAL]
                .set_index(["model", "pool"])["top_n_jaccard"])
    kappa = {
        arm: round(chance_corrected_agreement(value, chance[arm]), 4)
        for arm, value in observed.items() if arm in chance.index
    }
    frame["top_n_kappa"] = [
        kappa.get((row.model, row.pool), float("nan"))
        for row in frame.itertuples()
    ]
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run whose results/ to analyse, with the cut its config states "
                             "(`analyses.top_n`)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = args.run_dir / RESULTS_DIR
    try:
        args.top_n = load_recorded(args.run_dir).analyses.top_n
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    record_call(args.run_dir, CALL, vars(args))

    rows = []
    for model, pool, labels in arms(results):
        cell = results / model / pool / labels
        seed_dirs = [d for d in sorted(cell.iterdir()) if (d / cells.FEATURES_FILE).exists()]
        if len(seed_dirs) < 2:
            print(f"  {model}/{pool}/{labels}: {len(seed_dirs)} seed(s). Stability is a property "
                  f"of a *set* of runs and is undefined below two. Skipped.", flush=True)
            continue

        runs = [load_run(seed_dir, label=seed_dir.name) for seed_dir in seed_dirs]
        universe = universe_size(cell, runs)

        # Behavioural matching imputes a feature's non-finite values to its column median, so
        # say how much was imputed. A generated feature can be legitimately undefined (a
        # CpG-cluster span is NaN for a sequence with no cluster), and a Jaccard computed over an
        # imputed column is a weaker statement than one computed over a measured one.
        undefined = sum(non_finite_columns(run.values) for run in runs)
        if undefined:
            print(f"  {model}/{pool}/{labels}: {undefined} feature-run(s) carry values that are "
                  f"undefined on some reference rows; imputed to the column median for matching.",
                  flush=True)

        views = stability_views(runs, universe_size=universe, top_n=args.top_n)
        # The two things that would quietly invalidate the numbers
        # (`seqmaestro.harness.stability._diagnostics`).
        found = views["top_n"]["diagnostics"]
        if found["runs_with_chained_clusters"]:
            print(f"  {model}/{pool}/{labels}: in {found['runs_with_chained_clusters']} seed(s) one "
                  f"cluster of the top {args.top_n} holds two features of that seed, so the matcher "
                  f"chained two properties together or the features are near-duplicates.", flush=True)
        if found["constant_features"]:
            print(f"  {model}/{pool}/{labels}: features that never vary on the validation rows, by "
                  f"seed: {found['constant_features']}. They match nothing.", flush=True)

        row = {"model": model, "pool": pool, "labels": labels,
               "universe": universe, "undefined_columns": undefined,
               "n_used": max(views["full"]["n_features_per_run"].values())}
        for view, result in views.items():
            row[f"{view}_jaccard"] = round(result["jaccard"]["mean_pairwise"], 4)
            row[f"{view}_phi"] = round(result["nogueira"]["phi"], 4)
        rows.append(row)
        print(f"  {model:12} {pool:16} {labels:9} "
              f"full J {row['full_jaccard']:.3f} phi {row['full_phi']:.3f}   "
              f"top{args.top_n} J {row['top_n_jaccard']:.3f} phi {row['top_n_phi']:.3f}", flush=True)

    if not rows:
        # A run with one seed has no set of fits that could agree, so there is nothing to write. A
        # table left by an earlier attempt would be taken for this one's, so it is removed, and the
        # report's inputs then say that stability was not measured, which is true.
        (results / TABLE).unlink(missing_ok=True)
        print("\nNo arm has two seeds or more, so stability is undefined for every arm and no table "
              "is written.", flush=True)
        return 0

    frame = pd.DataFrame(rows)
    frame = add_chance_corrected(frame)
    frame.to_csv(results / TABLE, index=False)

    # The reported figure: one number per arm, comparable across all of them.
    reported = (frame[frame.labels == cells.REAL]
                .set_index(["model", "pool"])[["top_n_kappa", "top_n_jaccard"]]
                .sort_values("top_n_kappa", ascending=False))
    print("\nChance-corrected agreement (kappa), the comparable stability figure:")
    print(reported.to_string())

    # Diagnostic only: phi-hat needs a universe size that generative arms do not have, and raw
    # Jaccard ignores a chance level that differs sharply from one arm to the next.
    wide = frame.pivot(index=["model", "pool"], columns="labels",
                       values=["full_phi", "top_n_phi", "full_jaccard", "top_n_jaccard"])
    print("\nDiagnostic only, superseded by kappa above:")
    print(wide.to_string())
    print(f"\nWritten to {results / TABLE}")
    return 0

