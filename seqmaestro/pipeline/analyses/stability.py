"""Stability of every arm across seeds, against its own permuted-label null.

One row per (model, pool, labels) over the run's seeds, so a row describes an *arm* under
one label condition (a cell is one seed of one arm). Each seed was fitted on its own stratified
subsample of the training rows, and reseeding is the perturbation that stability is measured
against. There are two views per arm: the full set each model reports, and its top N by importance.
The top-N view is the comparable one, because these models disagree by orders of magnitude about how
many features they "use".

The permuted rows are essential. A stability number on real labels means nothing on its own, because
it has to clear the number the same arm returns on noise.

**Report `top_n_kappa`.** It is the chance-corrected agreement
`(J_real - J_permuted) / (1 - J_permuted)`, and the only figure here that means the same thing for
every arm: 0 when an arm reproduces its features no better than on shuffled labels, 1 when the seeds
returned identical sets. Each arm is corrected against its own measured chance level, because that
level differs sharply between arms. A fixed vocabulary is drawn from many columns and rarely agrees
with itself by accident. A language-model proposer draws from a small, concentrated prior and agrees
with itself readily on shuffled labels, because it proposes similar features whatever the labels say.
Both columns are in `stability.csv`, so check the gap instead of assuming it.

**It measures whether an arm *selects* the same set again, and that is not the only sense of stable.**
An arm that writes its features renames one idea between seeds, so it scores near zero here while
proposing the same repertoire every time. `analyse_recurrence.py` is the companion that asks the
other question, and neither replaces the other.

`full_phi`, `top_n_phi`, `full_jaccard` and `top_n_jaccard` are computed and written for
diagnosis, but they are **not the reported numbers**. Phi-hat needs a universe size that a
generative arm does not have, and its union-of-discovered fallback both breaks comparability between
arms and drives the value negative. Raw Jaccard ignores the chance level entirely, which flatters
whichever arm has the highest chance level. That is often a generative arm, because two selections
agree cheaply when both are drawn from a small pool.

    python experiments/analysis/analyse_stability.py --run-dir experiments/runs/<date>_<name>
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
    """Every (model, pool, labels) present, discovered rather than enumerated.

    A hardcoded list would silently skip every arm it does not name, such as the DEFT and
    language-model arms, which are the ones the comparison is about. A missing row looks the same as
    an arm that was never run, so nothing would reveal the omission.
    """
    found = set()
    for run_file in results.rglob(cells.RUN_FILE):
        seed_dir = run_file.parent
        found.add((seed_dir.parents[2].name, seed_dir.parents[1].name, seed_dir.parent.name))
    return sorted(found)


def universe_size(cell: Path, runs: list) -> int | None:
    """The Nogueira `d`, or `None` when the arm has no fixed vocabulary to quote one from.

    Read from the cells and not from a table here, so it cannot go stale when a featuriser's
    parameters change.

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
    carry the same value so that a filter on either condition still finds it. An arm missing one
    condition gets NaN instead of a number computed against an absent null.

    **Only the top-N view.** The full view cannot support a chance-corrected figure, because an arm
    that keeps every feature it is given agrees with itself almost perfectly on shuffled labels too.
    Dividing by `1 - chance` then explodes, and where the permuted Jaccard reaches exactly 1.0 the
    figure is undefined. Both are artifacts of comparing sets of very different sizes, which the
    top-N view exists to remove. A column that mixes a well-behaved value for an arm that selects
    with a large negative one for an arm that does not is worse than no column.
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
        # A run of one seed has no set of runs to agree, so there is nothing to write, and a table
        # left by an earlier attempt would be taken for this one's. The report inputs then say
        # stability was not measured, which is what happened.
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

