"""What a vendored arm is handed across the environment boundary, and how a cell of one is fitted.

A vendored method runs under its own interpreter, so each of its cells is prepared here as files,
handed to its adapter, and read back as a bundle (`experiments/adapters/README.md`). DEFT's arms
(`deft.py`) and IGSR's (`igsr.py`) do this the same way, and share this module so they cannot
drift apart:

- `write_inputs`: the fitting rows at the seed with the cell's labels, the validation rows and
  every training row with the real target, and `columns.json`, which records what each column is.
  Each frame holds the sequence column and the covariates the cell is handed. A method that chooses
  on the validation rows is also handed the labels it chooses against there, shuffled on a control
  arm.
- `input_frame`: any later rows as the cell's model was fitted on them, from `columns.json`.
- `predict_numbered`: those rows handed to an adapter's predict mode, numbered and unlabelled, with
  the order checked on return.
- `VendoredArm`: one arm of a method, whose `fit` draws a cell's rows, writes its inputs, runs the
  adapter and scores the bundle that comes back.
- `config_overrides`: the settings a method is started with, from the run's config.

Nothing here imports a vendored method.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from seqmaestro.dataset import Columns
from seqmaestro.features import PositionalWindow
from seqmaestro.features.base import Featuriser, Inputs, as_inputs
from seqmaestro.harness.cell import PERMUTED
from seqmaestro.harness.fitted import load_featuriser, save_featuriser
from seqmaestro.harness.runner import (
    ingest_bundle,
    labels_to_choose_on,
    permutation_key,
    subsample_training_rows,
)

if TYPE_CHECKING:
    from seqmaestro.pipeline.config import ExperimentConfig
    from seqmaestro.pipeline.rows import Rows

#: Inside a cell: what the adapter was handed, and what it handed back.
INPUTS, BUNDLE = "inputs", "bundle"
#: The featuriser behind any columns we computed, kept beside the inputs so the same columns can be
#: computed for rows the adapter was never handed.
SUPPLIED_FEATURISER = "supplied_featuriser.pkl"
#: Written beside the inputs, and read by every later stage and by the adapter on the far side: the
#: sequence column with its positional window, and each other column the method is given, with its
#: recipe and its description. `{"sequence": {"column", "window"}, "columns": {name: {"kind",
#: "params", "description"}}}`.
COLUMNS_MANIFEST = "columns.json"
#: The identifier column `predict_numbered` numbers the rows with, which the adapter echoes back.
ROW_ID = "row_id"


def write_inputs(fit_rows: pd.DataFrame, y_fit: np.ndarray, train: pd.DataFrame,
                 val: pd.DataFrame, into: Path, *, columns: Columns,
                 choose_on: np.ndarray | None,
                 descriptions: Mapping[str, str] | None = None,
                 window: PositionalWindow | None = None,
                 supplied: Featuriser | None = None) -> dict:
    """Materialise what an adapter reads. Returns the paths.

    The fitting rows are written out in full. Reproducing the seeded subsample on the far side
    would implement the same subsampling twice, and the two copies could come to disagree without
    anything failing.

    `train` is written too, as `effect.csv`: every training row with its **real** label, which is
    where `marginal_effect` is measured. It differs from `train.csv` in both respects, since that
    file is this seed's subsample and carries the shuffled labels on a control arm.

    `val.csv` carries the validation rows' **real** labels, because the cell is scored against
    them. `choose_on` is for a method that chooses anything on those rows (the iteration a search
    keeps, the terms it prunes): the labels it chooses against instead, one per validation row in
    their order (`labels_to_choose_on`), written as `choose_on.csv` under the target's column name.
    On a control arm they are shuffled, so nothing the method chooses reads the real target. It is
    required, so each arm says whether its method chooses on the validation rows: None for one that
    does not, and no file is written.

    `columns` are the columns an arm on this pool reads (`Dataset.columns_for`). The raw sequence
    is always written, and the covariates as they are where the pool holds them. `supplied`, when
    given, is an unfitted featuriser of columns computed on this side: it is fitted on the fitting
    rows, its columns are written beside the rest, and it is saved so the reveal can compute the
    same columns for the test sequences. `columns.json` records each column besides the sequence,
    with its recipe and its description (for a covariate, what `descriptions` says it measures),
    and the sequence column with the dataset's positional `window`. The method is told from it what
    each column means, and a column it selects comes back as a known feature, not as an
    unrecognised name.
    """
    into.mkdir(parents=True, exist_ok=True)
    paths = {name: into / f"{name}.csv" for name in ("train", "val", "effect")}
    sequence_column, target_column, weight_column = columns.sequence, columns.target, columns.weight
    described = dict(descriptions or {})
    undescribed = [name for name in columns.covariates if name not in described]
    if undescribed:
        raise ValueError(f"The covariate(s) {undescribed} have no description, and the method is "
                         f"told what each column it is given means. Pass the dataset's.")

    rows = {"train": (fit_rows, y_fit), "val": (val, val[target_column].to_numpy()),
            "effect": (train, train[target_column].to_numpy())}
    frames = {name: columns.inputs(frame).frame(sequence_column).assign(**{target_column: labels})
              for name, (frame, labels) in rows.items()}
    handed = {name: {"kind": "covariate", "params": {"column": name},
                     "description": described[name]} for name in columns.covariates}
    if supplied is not None:
        # Fitted on the rows the model trains on, as the in-process arms do, because a vocabulary
        # learned from rows the model never saw is a leak, whichever environment fits it.
        blocks = {
            "train": supplied.fit_transform(columns.inputs(fit_rows)),
            "val": supplied.transform(columns.inputs(val)),
            "effect": supplied.transform(columns.inputs(train)),
        }
        for name, block in blocks.items():
            frames[name] = pd.concat([frames[name].reset_index(drop=True),
                                      block.reset_index(drop=True)], axis=1)
        handed |= {record.name: {"kind": record.kind, "params": record.params,
                                 "description": record.description}
                   for record in supplied.records}
        save_featuriser(into / SUPPLIED_FEATURISER, supplied)
    (into / COLUMNS_MANIFEST).write_text(json.dumps({
        "sequence": {"column": sequence_column,
                     "window": None if window is None else window.as_dict()},
        "columns": handed,
    }, indent=2) + "\n")

    # Weights go with the training rows only. The model fits on those, and held-out rows are scored
    # unweighted because a model is scored against the prevalence the data actually has. They are
    # taken from `fit_rows` so the seed's subsample selects rows and weights together.
    if weight_column is not None:
        if weight_column not in fit_rows.columns:
            raise ValueError(
                f"The weight column {weight_column!r} is not in the split; columns are "
                f"{sorted(fit_rows.columns)}."
            )
        frames["train"] = frames["train"].assign(
            **{weight_column: fit_rows[weight_column].to_numpy(dtype=float)}
        )
    if choose_on is not None:
        paths["choose_on"] = into / "choose_on.csv"
        frames["choose_on"] = pd.DataFrame({target_column: np.asarray(choose_on)})

    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)
    return paths


def handed_covariates(inputs_dir: Path) -> list[str]:
    """The covariates a cell was handed, in the order it was handed them."""
    columns = json.loads((Path(inputs_dir) / COLUMNS_MANIFEST).read_text())["columns"]
    return [name for name, entry in columns.items() if entry["kind"] == "covariate"]


def input_frame(cell: Path, inputs: Inputs, sequence_column: str) -> pd.DataFrame:
    """These rows (`Inputs`, or the sequences alone) as a cell's model was fitted on them: the
    sequence under the dataset's column name, the covariates the cell was handed, and any columns
    we computed, by the featuriser saved with it."""
    inputs = as_inputs(inputs)
    handed = cell / INPUTS
    frame = inputs.select(handed_covariates(handed)).frame(sequence_column)
    if (handed / SUPPLIED_FEATURISER).exists():
        supplied = load_featuriser(handed / SUPPLIED_FEATURISER).transform(inputs)
        frame = pd.concat([frame, supplied.reset_index(drop=True)], axis=1)
    if ROW_ID in frame.columns:
        raise ValueError(f"{cell}: a column is already called {ROW_ID!r}, which numbers the rows.")
    return frame


def predict_numbered(command: list[str], frame: pd.DataFrame, where: Path) -> np.ndarray:
    """Hand `frame` to an adapter's predict mode, numbered and unlabelled, and return its scores.

    The adapter hands the numbers back beside the scores, and they are checked before anything is
    returned, since a reordered result scores plausibly and means nothing. `command` is the
    adapter's invocation up to its row arguments.
    """
    ids = np.arange(len(frame))
    numbered = frame.copy()
    numbered.insert(0, ROW_ID, ids)
    with tempfile.TemporaryDirectory(prefix="predict_") as scratch:
        rows, scores = Path(scratch) / "rows.csv", Path(scratch) / "scores.npz"
        numbered.to_csv(rows, index=False)
        completed = subprocess.run(
            [*command, "--rows", str(rows), "--id-column", ROW_ID, "--out", str(scores)],
            capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{where}: the adapter's predict exited {completed.returncode}:\n"
                               f"{completed.stderr[-2000:]}")
        with np.load(scores) as returned:
            returned_ids, returned_scores = returned["row_id"], returned["score"]
    if not np.array_equal(returned_ids, ids):
        raise RuntimeError(f"{where}: the adapter returned rows in a different order or number "
                           f"from the {len(ids)} it was given.")
    return np.asarray(returned_scores, dtype=float)


# ================================================================================
# One arm's cells, and the settings its method is started with
# ================================================================================


class VendoredArm:
    """One arm of a vendored method: `model` on `pool`, in a run.

    A method's module subclasses this with what differs between methods: what the cell is handed
    (`write`), how its adapter is started (`run`), and what its cells record in `predict_with`.
    """

    #: What the arm's cells record in `predict_with`, which is how the reveal finds the side that
    #: predicts them (`seqmaestro.pipeline.sides`).
    predict_with: str

    def __init__(self, config: ExperimentConfig, rows: Rows, model: str, pool: str) -> None:
        self.config, self.rows = config, rows
        self.model, self.pool = model, pool
        self.dataset = rows.dataset

    def write(self, fit_rows: pd.DataFrame, y_fit: np.ndarray, y_choose: np.ndarray,
              into: Path) -> dict:
        """Write a cell's inputs and return their paths. `y_choose` is what the cell chooses against
        on the validation rows (`labels_to_choose_on`), which a method is handed only if it chooses
        on them (`write_inputs`)."""
        raise NotImplementedError

    def run(self, paths: dict, bundle: Path, seed: int) -> None:
        """Run the adapter's fit on the inputs at `paths`, into `bundle`. Raises on failure."""
        raise NotImplementedError

    def fit(self, cell: Path, labels: str, seed: int) -> dict:
        """One cell: the fitting rows drawn, the inputs written, the adapter run, and the bundle
        scored on validation into the cell. Returns the cell's manifest."""
        grid, columns = self.config.grid, self.dataset.columns
        permuted = labels == PERMUTED
        key = permutation_key(grid.permutation_scope, (self.model, self.pool))
        fit_rows, y_fit = subsample_training_rows(
            self.rows.train, seed=seed, train_fraction=grid.train_fraction, columns=columns,
            permute_labels=permuted, permutation_key=key)
        y_choose = labels_to_choose_on(self.rows.val, seed=seed, columns=columns,
                                       permute_labels=permuted, permutation_key=key)
        paths = self.write(fit_rows, y_fit, y_choose, cell / INPUTS)
        self.run(paths, cell / BUNDLE, seed)
        return ingest_bundle(
            cell / BUNDLE, self.rows.val,
            target_column=columns.target, target_kind=self.dataset.target_kind, out_dir=cell,
            seed=seed, n_bootstrap=grid.n_bootstrap, labels=labels,
            arm=(self.model, self.pool), permutation_scope=grid.permutation_scope,
            train_fraction=grid.train_fraction, n_fit_rows=len(fit_rows),
            weight_column=columns.weight, predict_with=self.predict_with)


def config_overrides(config: ExperimentConfig, model: str, stated: Mapping[str, object],
                     language_model: tuple[str, ...]) -> list[str]:
    """What a vendored model's own config is overridden with, as `key=value`: what the grid states
    for it (`stated`), then the `config_overrides` of its `models` settings.

    A key in both is refused, because two places that set one value can come to disagree. So is a
    key that says which language model the method calls (`language_model`): the run's environment
    names one provider for every part of the run (`seqmaestro.llm.PROVIDERS`), and each arm's stage
    passes it to the method as it starts.
    """
    from seqmaestro.pipeline.config import ConfigError

    given = list(config.models.get(model, {}).get("config_overrides", ()))
    conflicting = [item for item in given if item.split("=")[0].strip() in stated]
    if conflicting:
        raise ConfigError(
            f"{config.source}: `models.{model}.config_overrides` sets {', '.join(conflicting)}, "
            f"which the grid already states. Remove it.")
    naming = [item for item in given if item.split("=")[0].strip() in language_model]
    if naming:
        raise ConfigError(
            f"{config.source}: `models.{model}.config_overrides` sets {', '.join(naming)}. The "
            f"language model is the one the environment names (SEQMAESTRO_LLM_PROVIDER, in .env), "
            f"for every arm of the run. Remove it.")
    return [*(f"{key}={value}" for key, value in stated.items()), *given]
