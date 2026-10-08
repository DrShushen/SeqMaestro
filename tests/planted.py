"""A small prepared dataset with a planted signal, finalised the way Stage 1 finalises a real one.

Used by the tests that need a run's worth of data, such as the reveal and the composition baseline.
The positive class carries a GC-rich motif at a fixed position, so a positional arm has something
to find and composition also shifts with the class.

`planted_dataset(..., covariates=True)` adds two covariates. `signal` runs higher in the positive
class and `noise` does not, and the motif is planted in only some of the positives, so the
covariate carries signal the sequence does not.

`planted_dataset(..., variable=True)` draws each sequence's length, longer in the positive class on
purpose, keeps the motif at the same offset from the start, and declares a positional window over
the first `LENGTH` bases that pads the shorter sequences.

`planted_dataset(..., continuous=True)` makes the target a number instead of a label: the rows that
would be positive get `SHIFT` added to unit Gaussian noise, so the motif (and, with covariates,
`signal`) predicts it. It composes with the two variants above.

The column names match no real dataset. The library has no default for either column, and a
realistic name here would let such a default come back without any test failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seqmaestro.dataset import GROUP_FILE, Dataset, finalise, load_dataset
from seqmaestro.harness.runner import run_model

SEQUENCE, TARGET = "window", "outcome"
LENGTH = 20
MOTIF, AT = "GGCGCC", 5
SIZES = {"train": 160, "val": 60, "test": 60}
#: The covariates of the variant that has them, with what each measures.
COVARIATES = {"signal": "a measurement that runs higher in the motif class",
              "noise": "a measurement unrelated to the class"}
#: With covariates, the share of positives that carry the motif.
MOTIF_SHARE = 0.6
#: With variable lengths, the range each class's lengths are drawn from, inclusive.
LENGTHS = {1: (16, 40), 0: (12, 30)}
#: With a continuous target, how far the rows carrying the signal sit above the rest, in units of
#: the noise's standard deviation.
SHIFT = 1.5


def planted_frame(n: int, rng: np.random.Generator, covariates: bool = False,
                  variable: bool = False, continuous: bool = False) -> pd.DataFrame:
    """`n` random sequences, about half of them positive. Without covariates every positive carries
    the motif. With them, `MOTIF_SHARE` of the positives do, and `signal` is shifted up in all.
    With `variable`, each sequence's length is drawn from its class's range in `LENGTHS`."""
    rows, labels = [], []
    for _ in range(n):
        if variable:
            positive = bool(rng.random() < 0.5)
            low, high = LENGTHS[int(positive)]
            sequence = list(rng.choice(list("ACGT"), int(rng.integers(low, high + 1))))
        else:
            sequence = list(rng.choice(list("ACGT"), LENGTH))
            positive = bool(rng.random() < 0.5)
        if positive and (not covariates or rng.random() < MOTIF_SHARE):
            sequence[AT:AT + len(MOTIF)] = list(MOTIF)
        rows.append("".join(sequence))
        labels.append(int(positive))
    frame = pd.DataFrame({SEQUENCE: rows, TARGET: labels})
    if covariates:
        frame["signal"] = np.round(rng.normal(1.5 * frame[TARGET].to_numpy(), 1.0), 4)
        frame["noise"] = np.round(rng.normal(0.0, 1.0, n), 4)
    if continuous:
        frame[TARGET] = np.round(SHIFT * frame[TARGET].to_numpy() + rng.normal(0.0, 1.0, n), 4)
    return frame


def planted_dataset(root: Path, seed: int = 3, covariates: bool = False,
                    variable: bool = False, window: bool = True, continuous: bool = False) -> Dataset:
    """Write the three splits under `root`, finalise them, and load the result. `window=False`
    leaves a variable-length dataset without its positional window."""
    rng = np.random.default_rng(seed)
    directory = root / "planted" / "splits"
    directory.mkdir(parents=True)
    frames = {split: planted_frame(n, rng, covariates, variable, continuous)
              for split, n in SIZES.items()}
    groups = []
    for split, frame in frames.items():
        frame.to_csv(directory / f"{split}.csv", index=False)
        # One group per row, because the sequences are drawn independently and nothing relates two
        # of them. Writing the group map records that explicitly, so the grouping is not unknown.
        groups += [{"sequence": sequence, "group": f"{split}_{row}"}
                   for row, sequence in enumerate(frame[SEQUENCE])]
    pd.DataFrame(groups).drop_duplicates("sequence").to_csv(directory / GROUP_FILE, index=False)
    (directory.parent / "README.md").write_text("# A planted-motif dataset, for tests\n")

    totals = ({label: sum(int((frame[TARGET] == label).sum()) for frame in frames.values())
               for label in (1, 0)} if not continuous else sum(len(frame) for frame in frames.values()))
    declared_covariates = list(COVARIATES) if covariates else []
    target = ({"target": {"kind": "continuous", "name": "planted response",
                          "meaning": "a number that runs higher in the rows carrying the planted "
                                     "signal, by 1.5 standard deviations of its noise"}}
              if continuous else
              {"target": {"kind": "binary"},
               "classes": {1: {"name": "motif",
                               "meaning": ("the planted class: a raised signal, and the motif in most "
                                           "rows" if covariates else "carries the planted motif")},
                           0: {"name": "background", "meaning": "does not"}}})
    finalise(directory, {
        "columns": {"sequence": SEQUENCE, "target": TARGET, "weight": None,
                    "covariates": declared_covariates},
        "covariates": {name: {"description": COVARIATES[name]} for name in declared_covariates},
        **target,
        "sequences": ({"window": "random bases, of a length drawn per class", "anchor": "the first base",
                       "orientation": "as generated", "length": "variable",
                       **({"positional_window": {"length": LENGTH, "anchor": "start", "pad": True}}
                          if window else {})}
                      if variable else
                      {"window": f"{LENGTH} random bases", "anchor": "the first base",
                       "orientation": "as generated", "length": LENGTH}),
        "split": {"method": "random", "grouped_by": ["unit"], "stratified_on": [], "seed": seed},
        "filters": [{"step": "generated", "why": "synthetic", "rows_in": totals,
                     "rows_out": totals}],
        "builders": ["tests/planted.py"],
        **({"acknowledged": {"sequence_lengths": "synthetic: the positive class is drawn longer on "
                                                 "purpose, so the length finding is by design"}}
           if variable else {}),
    })
    return load_dataset(directory)


#: The `analyses` section of a test's run config, as the shared config states it.
ANALYSES = {
    "top_n": 20,
    "judge": {"repeats": 3, "show_code": False, "labels": ["real"], "min_answered": 0.9, "seed": 0},
    "recurrence": {"kinds": ["llm_generated"], "items_per_group": 6, "draws": 2000,
                   "across_arms": True, "seed": 0},
}


#: The settings of a run's grid that fitting a cell takes, at the values a test fits with where it
#: states none. The library has no default for any of them, since a run states each in its config.
GRID = {"seed": 0, "train_fraction": 0.8, "permute_labels": False, "permutation_scope": "seed",
        "n_bootstrap": 1000, "permutation_repeats": 5}


def run_cell(model, featuriser, train, val, **given):
    """`run_model`, with each of the grid's settings the test does not state taken from `GRID`."""
    return run_model(model, featuriser, train, val, **{**GRID, **given})
