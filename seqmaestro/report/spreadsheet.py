"""A finished run's reported features as a spreadsheet, for people who want to sort and filter.

    $LOCALUTILS experiments/analysis/export_features_xlsx.py --run-dir experiments/runs/<date>_<name>

**It is written under `SeqMaestro_LocalUtils`.** Writing the file needs `openpyxl`, which nothing
in the pipeline needs, so it stays out of the environment that produces results
(`ENVIRONMENTS.md`). The sheets themselves are built with pandas alone.

The spreadsheet adds nothing a run does not already contain. `features.json` in each cell is the
record, and this presents it in a format a collaborator can open. A written report quotes a handful
of features, and past a few dozen rows a spreadsheet is easier to work with.

It has four sheets: what the columns mean, one row for each of the `--top-n` most important
features every cell reported, one row per arm with its performance, and the concepts several arms
agreed on. Only real-label cells appear among the features, because the feature list of a
shuffled-label cell is a control.

**Performance is on the test split only once the run has been revealed**, and on validation before
that, with the split named in the column headers. A spreadsheet made before the reveal is a
working view, and no test number can leave a run through it early. The number is the headline
metric of the run's kind of target (AUROC or R-squared), named in the headers.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from seqmaestro.arms import arm_name
from seqmaestro.harness import cell as cells
from seqmaestro.harness.features import DEFAULT_TOP_N, load_features, top_features
from seqmaestro.harness.heldout import TEST_SPLIT, reveal_record
from seqmaestro.judge.files import SCORES as JUDGE_SCORES
from seqmaestro.pipeline.analyses import co_discovery
from seqmaestro.pipeline.layout import JUDGE_DIR, RESULTS_DIR
from seqmaestro.report import naming
from seqmaestro.target import BINARY, CONTINUOUS, recorded_kind, target_for

#: Where the spreadsheet goes when no destination is named, under the run.
FILE = "features.xlsx"

#: How wide each column is opened, by header. Anything unlisted gets a default.
WIDTHS = {"description": 110, "feature": 42, "concept members": 90, "model": 22,
          "feature source": 14, "seed": 7, "rank": 6, "importance": 10, "judge mean": 11}

#: The read-me sheet's rows. `{top_n}` is the number of features the export took from each cell.
READ_ME = [
    ("What this is", "The {top_n} most important features each arm reported at each seed, one row "
                     "per feature. An arm whose model used fewer lists them all."),
    ("Only real labels", "The features are those of the arms fitted on the real labels. Where the "
                         "run also fitted each arm on shuffled labels as a control, those feature "
                         "lists are not findings and are left out."),
    ("rank / importance", "Each model scores how much each feature contributed, and the features "
                          "are ranked by it. How that score is computed differs by model: a "
                          "coefficient, a gain, an impurity decrease. So importance is comparable "
                          "WITHIN one model, feature source and seed, and never between them. Rank "
                          "is the position in that ordering."),
    ("seed", "The run was repeated with different random seeds. Rows are matched by name, so a "
             "feature listed under one seed only was not among that arm's most important at the "
             "other seeds under that name. A feature a language model wrote can return under "
             "another name."),
    ("judge mean", "Mean of the independent ratings the feature got in this arm, 1 to 5, of how "
                   "biologically meaningful it would be if the association held. A feature rated "
                   "at several seeds shows one mean over all of them. Blank where the feature "
                   "was not put to the rater."),
    ("description", "How the feature is computed. For a feature a language model wrote, this is "
                    "the description that model gave it."),
]

#: What `direction` and `marginal effect` mean, by the run's kind of target.
READ_ME_BY_KIND = {
    BINARY: [
        ("direction", "Which class the model uses the feature to argue for: higher_in_positive "
                      "means the model's score for the positive class rises as the feature does, "
                      "higher_in_negative that it falls, and non_monotone that the model uses the "
                      "feature one way at low values and the other way at high ones. It is "
                      "derived from how each model works, so the word is comparable between models "
                      "even though the quantities behind it are not."),
        ("marginal effect", "How far the feature separates the two classes ON ITS OWN, ignoring "
                            "every other feature, from -1 to 1. Positive means higher values are "
                            "more common in the positive class. At 0.4, of one example from each "
                            "class, the positive one holds the higher value 70% of the time. Where "
                            "this and `direction` disagree, the model is using the feature to "
                            "correct for something else rather than for its own association with "
                            "the outcome."),
    ],
    CONTINUOUS: [
        ("direction", "Which way the model moves its prediction as the feature rises: "
                      "increases_target means the predicted value rises with the feature, "
                      "decreases_target that it falls, and non_monotone that the model uses the "
                      "feature one way at low values and the other way at high ones. It is "
                      "derived from how each model works, so the word is comparable between models "
                      "even though the quantities behind it are not."),
        ("marginal effect", "How far the feature rises or falls with the target ON ITS OWN, "
                            "ignoring every other feature: Spearman's rank correlation, from -1 to "
                            "1. Positive means higher values of the feature go with higher values "
                            "of the target. Where this and `direction` disagree, the model is using "
                            "the feature to correct for something else rather than for its own "
                            "association with the outcome."),
    ],
}


def read_me(target_kind: str, top_n: int) -> list[tuple[str, str]]:
    """The read-me sheet for a run of `target_kind`: the rows every run shares, with the two rows
    whose wording depends on the kind placed before the description. `top_n` is how many features
    the export takes from each cell."""
    rows = [*READ_ME[:-1], *READ_ME_BY_KIND[target_kind], READ_ME[-1]]
    return [(column, meaning.format(top_n=top_n) if column == "What this is" else meaning)
            for column, meaning in rows]


def cells_of(results: Path, labels: str = "*") -> list[Path]:
    """The finished cells under a run's results, of one label condition or of every one."""
    return sorted(path.parent for path in results.glob(f"*/*/{labels}/seed_*/{cells.METRICS_FILE}"))


def run_kind(results: Path) -> str:
    """The run's kind of target, as its cells record it."""
    kinds = {recorded_kind(cells.read_run(cell)) for cell in cells_of(results)}
    if len(kinds) != 1:
        raise SystemExit(f"{results}: the cells record {sorted(kinds) or 'no'} kinds of target.")
    return kinds.pop()


def arm_of(cell: Path) -> tuple[str, str]:
    """A cell's arm as its directories spell it: the model and the pool."""
    return cell.parts[-4], cell.parts[-3]


def named(cell: Path) -> tuple[str, str]:
    """(model, feature source) for a cell, named as a reader sees them.

    Named by `seqmaestro.report.naming`, as the report is, so the spreadsheet and the report a
    collaborator is handed use one vocabulary. An arm `naming` does not know raises, so an arm never
    reaches a sheet under its directory name.
    """
    configuration = naming.parse(arm_name(*arm_of(cell)))
    return configuration.model_short, configuration.source_short


def judge_means(run_dir: Path) -> dict[tuple[str, str], float]:
    """Mean rating per (arm, feature), for the real-label items the judge was shown."""
    path = run_dir / JUDGE_DIR / JUDGE_SCORES
    if not path.exists():
        return {}
    gathered: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for item in json.loads(path.read_text()):
        if item.get("labels") == cells.REAL:
            gathered[(item["arm"], item["feature"])].extend(item["scores"])
    return {key: sum(v) / len(v) for key, v in gathered.items() if v}


def features(results: Path, ratings: dict, top_n: int) -> pd.DataFrame:
    """One row per feature a real-label cell reported, the `top_n` most important of each cell."""
    rows = []
    for cell in cells_of(results, cells.REAL):
        model, source = named(cell)
        # The judge's records name an arm as its directories spell it, since that is the name the
        # judge was handed.
        arm = arm_name(*arm_of(cell))
        records, _ = load_features(cell / cells.FEATURES_FILE)
        for rank, record in enumerate(top_features(records, top_n), start=1):
            rows.append({
                "model": model, "feature source": source,
                "seed": int(cell.name.removeprefix("seed_")), "rank": rank,
                "feature": record.name,
                "importance": round(abs(record.importance or 0), 6),
                "judge mean": ratings.get((arm, record.name)),
                "direction": record.direction,
                "marginal effect": (None if record.marginal_effect is None
                                    else round(record.marginal_effect, 4)),
                "provenance": record.provenance,
                "description": " ".join((record.description or "").split()),
            })
    return pd.DataFrame(rows)


def performance(results: Path, split: str, target_kind: str) -> pd.DataFrame:
    """Mean and spread of the kind's headline per arm on `split`, with the split in the header of
    each and the metric in the mean's. A cell with no block for `split` counts towards neither, as
    happens after the reveal to a cell the run's config does not name, which the reveal does not
    score."""
    headline = target_for(target_kind).headline
    name = naming.METRICS[headline]
    rows = []
    for cell in cells_of(results):
        model, source = named(cell)
        rows.append({"model": model, "feature source": source, "labels": cell.parent.name,
                     headline: cells.read_metrics(cell).get(split, {}).get(headline),
                     "features used": cells.read_run(cell).get("n_features_used")})
    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(index=["model", "feature source"], columns="labels",
                             values=headline, aggfunc=["mean", "std"])
    wide.columns = [f"{split} {name} {b}" if a == "mean" else f"{split} sd {b}"
                    for a, b in wide.columns]
    used = (frame[frame.labels == cells.REAL].groupby(["model", "feature source"])["features used"]
            .mean())
    return (wide.join(used.rename(f"features used ({cells.REAL})")).round(4)
            .sort_values(f"{split} {name} {cells.REAL}", ascending=False).reset_index())


def concepts(results: Path, min_witnesses: int) -> pd.DataFrame:
    """The real-label concepts that at least `min_witnesses` independent arms reached, from the
    co-discovery table where the run has one."""
    path = results / co_discovery.TABLE
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[(frame.labels == cells.REAL) & (frame.n_witnesses >= min_witnesses)]
    return (frame.rename(columns={"n_arms": "arms", "n_witnesses": "independent witnesses",
                                  "n_pools": "distinct feature sources",
                                  "names": "concept members", "seed": "seed"})
            [["seed", "arms", "independent witnesses", "distinct feature sources", "concept members"]]
            .sort_values("independent witnesses", ascending=False).reset_index(drop=True))


def write(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """One sheet per frame, each with its header row frozen, a filter, and columns opened to a
    readable width."""
    from openpyxl.styles import Alignment, Font  # in SeqMaestro_LocalUtils alone
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for index, column in enumerate(frame.columns, start=1):
                letter = get_column_letter(index)
                sheet.column_dimensions[letter].width = WIDTHS.get(str(column), 18)
                sheet[f"{letter}1"].font = Font(bold=True)
            sheet.freeze_panes = "A2"
            if len(frame):
                sheet.auto_filter.ref = sheet.dimensions
            for row in sheet.iter_rows(min_row=2):
                for value in row:
                    value.alignment = Alignment(vertical="top", wrap_text=True)


def sheets_of(run_dir: Path, top_n: int, min_witnesses: int) -> dict[str, pd.DataFrame]:
    """The four sheets of a run, by name."""
    results = run_dir / RESULTS_DIR
    # Test numbers are read once the reveal has finished. Before it there are none, and during
    # it only some cells have theirs.
    revealed = reveal_record(run_dir)
    split = TEST_SPLIT if revealed and revealed.get("completed_utc") else cells.VALIDATION
    kind = run_kind(results)
    return {
        "Read me": pd.DataFrame(read_me(kind, top_n), columns=["column", "what it means"]),
        "Features": features(results, judge_means(run_dir), top_n),
        "Arms": performance(results, split, kind),
        "Agreed concepts": concepts(results, min_witnesses),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Destination .xlsx. Without it, <run-dir>/{FILE}.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Features per cell, as many as the analyses take.")
    parser.add_argument("--min-witnesses", type=int, default=4,
                        help="Concepts reached by at least this many independent arms.")
    args = parser.parse_args(argv)

    if not (args.run_dir / RESULTS_DIR).exists():
        parser.error(f"no results under {args.run_dir}")
    out = args.out or args.run_dir / FILE
    sheets = sheets_of(args.run_dir, args.top_n, args.min_witnesses)
    write(out, sheets)
    for name, frame in sheets.items():
        print(f"  {name:<16} {len(frame):>5} rows")
    print(f"\nWritten to {out}")
    return 0
