#!/usr/bin/env python
"""Build the example dataset: made-up binding sites, from a delivery to a grouped split.

    python experiments/prepare/build_example_dataset.py
    python experiments/prepare/finalise_dataset.py --splits data/example/splits \\
        --declared experiments/prepare/declarations/example__splits.yaml

The example shows every file a prepared dataset has, on data anyone can make, and gives a run
something to work on before there is real data. **Nothing in it is biology.** The sequences are
random bases with one rule planted in them: most sites of the bound class carry one motif at their
centre.

It takes the steps a builder of a real delivery takes, one function each:

1. `write_delivery` stands in for what the people who measured the data would send, and writes it
   to `raw/sites.csv`: one candidate site per row, with the region it lies in, its sequence at the
   length it was delivered at, and whether the factor was found bound there. A real dataset starts
   with files like this one already in `raw/`.
2. `build` cuts one window around the centre of each site, so that a position is the same distance
   from the site in every row. A site delivered shorter than the window cannot supply one and is
   dropped. The step counts what it dropped from each class, and refuses to go on when that is more
   than `MAX_DROPPED_SHARE` of a class, since a filter that removes much of one class changes the
   task.
3. `split` assigns each region whole to training, validation or test. Sites of one region are
   neighbours, so a model that has seen one has in effect seen the others, and a split by row would
   report a score it could not reach on a new region. The split files hold the sequence and the
   label alone, and `groups.csv` records the region of each sequence, which is what the grouping is
   checked against when the dataset is finalised.
4. `write_readme` writes the README a person reads before trusting a number from the dataset.

What these steps decided is stated in the declaration, `declarations/example__splits.yaml`.
`finalise_dataset.py` counts and checks everything else and writes `dataset.json` beside the splits
(`docs/dataset.md`). The description the language models are told is
`experiments/datasets/example.json`, and the run config is `experiments/configs/example.yaml`.

The number of sites of each class and of each length is fixed, and only the bases and the order
are drawn. The rows the declaration states for each step therefore hold on any machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from seqmaestro.dataset import GROUP_FILE
from seqmaestro.grouping import group_map

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "data" / "example"

DELIVERY, BUILT, MANIFEST, SPLITS = "raw/sites.csv", "sites_60nt.csv", "build_manifest.json", "splits"
SITE_COLUMN, GROUP_COLUMN, SEQUENCE_COLUMN, TARGET_COLUMN = "site_id", "region", "sequence", "bound"
CLASS_NAMES = {1: "bound", 0: "not bound"}

#: The delivery: this many candidate sites, `SITES_PER_REGION` to a region, alternately not bound
#: and bound.
N_SITES, SITES_PER_REGION = 1200, 4
#: The lengths the sites of each class are delivered at, in equal numbers. The bound sites are
#: delivered a little longer, so the window costs the two classes different shares of their rows.
DELIVERED_LENGTHS = {1: tuple(range(56, 136, 8)), 0: tuple(range(48, 128, 8))}
#: The planted rule: this share of the bound sites carry the motif, centred on the site.
MOTIF, MOTIF_SHARE = "GATTACA", 0.7
DELIVERY_SEED = 20261008

#: The window cut around each site's centre, and the largest share of a class the cut may drop.
WINDOW = 60
MAX_DROPPED_SHARE = 0.25

#: Ten folds, assigned seven, one and two to training, validation and test. Assigning whole folds
#: keeps the grouping and the stratification exact together.
FOLDS, TRAIN_FOLDS, VAL_FOLDS = 10, 7, 1
SPLIT_SEED = 20261008


def write_delivery(path: Path) -> None:
    """The stand-in for a delivery: `N_SITES` candidate sites of a made-up DNA-binding factor."""
    rng = np.random.default_rng(DELIVERY_SEED)
    per_class = N_SITES // 2
    lengths = {label: rng.permutation(np.resize(DELIVERED_LENGTHS[label], per_class))
               for label in CLASS_NAMES}
    rows = []
    for site in range(N_SITES):
        bound = site % 2
        length = int(lengths[bound][site // 2])
        bases = rng.choice(list("ACGT"), length)
        if bound and rng.random() < MOTIF_SHARE:
            start = length // 2 - len(MOTIF) // 2
            bases[start:start + len(MOTIF)] = list(MOTIF)
        rows.append({SITE_COLUMN: f"site_{site:04d}",
                     GROUP_COLUMN: f"region_{site // SITES_PER_REGION:03d}",
                     SEQUENCE_COLUMN: "".join(bases), TARGET_COLUMN: bound})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def per_class(frame: pd.DataFrame) -> dict[int, int]:
    """The rows of each class, positive first."""
    return {label: int((frame[TARGET_COLUMN] == label).sum()) for label in CLASS_NAMES}


def build(delivery: Path) -> tuple[pd.DataFrame, dict]:
    """One window per site, cut around its centre, with what the cut cost each class."""
    sites = pd.read_csv(delivery, dtype={SEQUENCE_COLUMN: str})
    missing = {SITE_COLUMN, GROUP_COLUMN, SEQUENCE_COLUMN, TARGET_COLUMN} - set(sites.columns)
    if missing:
        raise ValueError(f"{delivery} has no column {', '.join(sorted(missing))}.")
    if not sites[TARGET_COLUMN].isin(list(CLASS_NAMES)).all():
        raise ValueError(f"{delivery}: `{TARGET_COLUMN}` must be 1 or 0 on every row.")
    if sites[SITE_COLUMN].duplicated().any():
        raise ValueError(f"{delivery} lists a site twice, so a row is no longer one site.")
    sites[SEQUENCE_COLUMN] = sites[SEQUENCE_COLUMN].str.upper()
    unexpected = sorted(set("".join(sites[SEQUENCE_COLUMN])) - set("ACGT"))
    if unexpected:
        raise ValueError(f"{delivery} holds the symbols {unexpected}, and the build expects ACGT.")

    sites["delivered_length"] = sites[SEQUENCE_COLUMN].str.len()
    kept = sites[sites["delivered_length"] >= WINDOW].copy()
    rows_in, rows_out = per_class(sites), per_class(kept)
    for label, name in CLASS_NAMES.items():
        dropped = 1 - rows_out[label] / rows_in[label]
        if dropped > MAX_DROPPED_SHARE:
            raise ValueError(
                f"A {WINDOW} nt window drops {dropped:.0%} of the {name} sites, above the "
                f"{MAX_DROPPED_SHARE:.0%} this build allows. Dropping that many changes what the "
                f"task is, so choose a narrower window or another anchor.")

    # The same number of bases either side of the centre, so the centre is at one position in
    # every row whatever length the site was delivered at.
    first = (kept["delivered_length"] - WINDOW) // 2
    kept[SEQUENCE_COLUMN] = [sequence[start:start + WINDOW]
                             for sequence, start in zip(kept[SEQUENCE_COLUMN], first, strict=True)]
    built = kept[[SEQUENCE_COLUMN, TARGET_COLUMN, GROUP_COLUMN, SITE_COLUMN,
                  "delivered_length"]].reset_index(drop=True)
    step = {"step": f"sites delivered shorter than the {WINDOW} nt window dropped",
            "rows_in": rows_in, "rows_out": rows_out}
    return built, step


def split(built: pd.DataFrame, directory: Path) -> dict[str, pd.DataFrame]:
    """Each region whole in one of training, validation and test, with the classes in proportion."""
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold = pd.Series(-1, index=built.index, dtype=int)
    for number, (_, held) in enumerate(splitter.split(built, built[TARGET_COLUMN],
                                                      built[GROUP_COLUMN])):
        fold.iloc[held] = number
    parts = {"train": built[fold < TRAIN_FOLDS],
             "val": built[(fold >= TRAIN_FOLDS) & (fold < TRAIN_FOLDS + VAL_FOLDS)],
             "test": built[fold >= TRAIN_FOLDS + VAL_FOLDS]}

    placed: dict[str, str] = {}
    for name, part in parts.items():
        for region in part[GROUP_COLUMN].unique():
            if region in placed:
                raise ValueError(f"{region} is in both {placed[region]} and {name}.")
            placed[region] = name

    directory.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        # The sequence and the label alone, so a column that is no feature cannot become one.
        part[[SEQUENCE_COLUMN, TARGET_COLUMN]].to_csv(directory / f"{name}.csv", index=False)
    group_map(built, SEQUENCE_COLUMN, [GROUP_COLUMN]).to_csv(directory / GROUP_FILE, index=False)
    return parts


def write_readme(path: Path, step: dict, parts: dict[str, pd.DataFrame]) -> None:
    """The dataset's README, with the sections `docs/dataset.md` lists for one."""
    rows_in, rows_out = step["rows_in"], step["rows_out"]
    centre, half = WINDOW // 2, len(MOTIF) // 2
    lost = {label: 1 - rows_out[label] / rows_in[label] for label in CLASS_NAMES}
    delivered = [length for lengths in DELIVERED_LENGTHS.values() for length in lengths]
    sizes = " / ".join(f"{len(part):,}" for part in parts.values())
    path.write_text(f"""# The example dataset: made-up binding sites

**Nothing here is biology.** `experiments/prepare/build_example_dataset.py` made every file in
this folder. The sequences are random bases with one rule planted in them, so that the files of a
prepared dataset can be seen whole, and a run has something to work on before there is real data.
The README of a real dataset has the same sections, written by whoever prepared it.

| | |
|---|---|
| Rows | {sum(rows_out.values()):,}: {rows_out[1]:,} bound, {rows_out[0]:,} not bound |
| Sequence | {WINDOW} nt, upper case `ACGT`, with the site's centre at position {centre} |
| Split | {sizes} rows in training, validation and test, grouped on the region |
| Declaration | `experiments/prepare/declarations/example__splits.yaml` |
| Description | `experiments/datasets/example.json` |
| Config | `experiments/configs/example.yaml` |

## The source

`{DELIVERY}` stands in for a delivery: {N_SITES:,} candidate sites of a made-up DNA-binding
factor, each with the region it lies in, its sequence centred on the site, at the length it was
delivered at ({min(delivered)} to {max(delivered)} nt), and whether the factor was found bound
there. The builder writes that file. A real dataset starts with the delivered files already in
`raw/`, kept exactly as they arrived.

## The target

`{TARGET_COLUMN}` is 1 for a site where the factor was found bound, and 0 for a candidate site
where it was not. Both classes come from the one delivered table. What separates them is planted:
{MOTIF_SHARE:.0%} of the bound sites carry the motif `{MOTIF}` centred on the site, and beyond that
both classes are random bases.

## The window and its anchor

One window of {WINDOW} nt, cut around the centre of each site. Position {centre} is the site's
centre in every row, and the motif, where a site carries it, covers positions {centre - half} to
{centre + half}. The sites are delivered at different lengths, and the bound ones a little longer.
A model given the whole sequences could predict from the length alone, which the window removes.

## The preparation steps, and what each cost

| step | bound | not bound |
|---|---:|---:|
| delivered | {rows_in[1]:,} | {rows_in[0]:,} |
| {step["step"]} | {rows_out[1]:,} | {rows_out[0]:,} |

The cut drops {lost[1]:.0%} of the bound sites and {lost[0]:.0%} of the others, because the
shorter deliveries are more often sites that are not bound. A narrower window would drop fewer
sites and show less of the sequence around each.

## The grouping

Each region is assigned whole to training, validation or test. Sites of one region are neighbours,
so a model that has seen one has in effect seen the others, and a split by row would report a score
it could not reach on a new region. The split is stratified on the label
(`StratifiedGroupKFold` over {FOLDS} folds: {TRAIN_FOLDS} training, {VAL_FOLDS} validation,
{FOLDS - TRAIN_FOLDS - VAL_FOLDS} test, seed {SPLIT_SEED}), and `{SPLITS}/groups.csv` records the
region of each sequence.

## What the language models are told

`experiments/datasets/example.json`: that each sequence is a window around a candidate binding
site, labelled by whether the factor was found bound there, what the positions mean, and that the
data is synthetic. The motif is not named to them. No fact is kept for the judge alone, so the
run's config states `blinded: false`.

## Open questions for the people who supplied the data

None, since nobody supplied it. In a real dataset this section holds what only they can answer,
each question with what would change depending on the answer.
""")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATASET,
                        help="the dataset's folder, which the run config names as data/example")
    args = parser.parse_args(argv)

    delivery = args.out / DELIVERY
    write_delivery(delivery)
    built, step = build(delivery)
    built.to_csv(args.out / BUILT, index=False)
    parts = split(built, args.out / SPLITS)
    write_readme(args.out / "README.md", step, parts)
    manifest = {
        "delivery": DELIVERY,
        "delivery_sha256": hashlib.sha256(delivery.read_bytes()).hexdigest(),
        "window": WINDOW,
        "anchor": "the centre of the site",
        "filters": [step],
        "rows": len(built),
        "regions": int(built[GROUP_COLUMN].nunique()),
        "split_seed": SPLIT_SEED,
    }
    (args.out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{step['step']}:")
    for label, name in CLASS_NAMES.items():
        print(f"  {name:<10} {step['rows_in'][label]:>5,} -> {step['rows_out'][label]:>5,}")
    for name, part in parts.items():
        print(f"  {name:<6} {len(part):>5,} rows   {part[TARGET_COLUMN].mean():.3f} bound   "
              f"{part[GROUP_COLUMN].nunique():>4,} regions")
    out = args.out.resolve()
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"Written to {shown}")
    print(f"Next: experiments/prepare/finalise_dataset.py --splits {shown / SPLITS} --declared "
          f"experiments/prepare/declarations/example__splits.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
