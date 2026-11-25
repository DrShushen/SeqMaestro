"""The fitted arm, saved so that a finished cell can be loaded again and used on new sequences.

A cell records what a model *found* (its features, their importances, its scores on the validation
rows) and also keeps the fitted model, so that a new question can be answered without refitting the
cell, even a question about sequences the run never saw.

**The model alone is not enough.** It was fitted on the columns a featuriser produced, and
predicting on a new sequence means producing those columns again, in the same order and from the
same learned vocabulary. The featuriser and the model are therefore saved and loaded together, as a
`FittedArm`.

    from seqmaestro.harness.fitted import load_fitted

    arm = load_fitted("experiments/runs/<run>/results/ebm/positional/real/seed_42")
    arm.predict(["ACGT...", "TTGC..."])            # sequences alone
    arm.predict(dataset.inputs(frame))             # rows with their covariates

**The reveal depends on it.** A cell is scored on validation when it is fitted, and on the test
split once, at the reveal. The reveal predicts with the arm saved here and never refits, so a cell
without its fitted arm can never be scored on the test split.

**DEFT's and IGSR's Stage 2 cells keep their model in their bundle.** Each method saves the model
it fitted into the bundle it hands back, in the method's format: DEFT its tree, and IGSR the model
of its search. Using either needs the method, which this package must not import, so the reveal
has the method's adapter predict with it. The only fitted part of such a cell that this side saves
is the featuriser for any columns this side computed and handed across, which `save_featuriser`
writes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqmaestro import files
from seqmaestro.features.base import Featuriser
from seqmaestro.models.base import Model, filled_rows

#: The file's name inside a cell.
FILENAME = "fitted.pkl"

#: What the `run.json` of a cell that keeps this file records under `predict_with`, which tells the
#: reveal to load the file and predict in its process instead of through an adapter.
PREDICT_WITH = "fitted"

#: Bumped when the saved layout changes in a way a reader must notice. A pickle has no header, so
#: the version is saved beside the object in the pickled payload.
FITTED_VERSION = 1


@dataclass(frozen=True)
class FittedArm:
    """A fitted featuriser and the model fitted on top of it, kept together because neither is
    usable without the other."""

    featuriser: Featuriser
    model: Model
    #: Which column of the dataset held the sequences the arm was fitted on. The reveal checks it
    #: against the dataset it scores. `predict` takes rows rather than a frame, so it needs no name.
    sequence_column: str | None = None

    def transform(self, inputs) -> pd.DataFrame:
        """The feature columns for these rows, as the model expects them. Takes what
        `Featuriser.transform` takes: `Inputs`, or the sequences alone."""
        return self.featuriser.transform(inputs)

    def predict(self, inputs) -> np.ndarray:
        """The model's score per row: the probability of the positive class for a binary target,
        the predicted value for a continuous one."""
        return np.asarray(self.model.predict(self.transform(inputs)), dtype=float)

    def predict_counting(self, inputs) -> tuple[np.ndarray, int | None]:
        """`predict`, plus how many rows took the model's constant prediction because the fitted
        model is undefined there (`None` for a model defined everywhere), both from one transform."""
        X = self.transform(inputs)
        return np.asarray(self.model.predict(X), dtype=float), filled_rows(self.model, X)

    def describe(self) -> dict[str, Any]:
        return {"featuriser": self.featuriser.describe(), "model_settings": self.model.describe(),
                "sequence_column": self.sequence_column}


def save_fitted(out_dir: str | Path, featuriser: Featuriser, model: Model,
                sequence_column: str | None = None) -> Path:
    """Write the fitted pair into a cell and return the path.

    A pickling failure is raised, not swallowed. Every featuriser and model in this package pickles
    (the generated featurisers because they define how to restore themselves), so a failure here
    means a new component that cannot be saved. It is far better to find that out on the run that
    produced it than later, when the cell is needed.
    """
    if not featuriser.is_fitted:
        raise ValueError("Refusing to save an unfitted featuriser: it has no columns to produce.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILENAME
    payload = {"version": FITTED_VERSION,
               "arm": FittedArm(featuriser=featuriser, model=model,
                                sequence_column=sequence_column)}
    return files.write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def save_featuriser(path: str | Path, featuriser: Featuriser) -> Path:
    """Write a fitted featuriser on its own, for columns handed to an arm in another environment.

    That arm's model stays in its method's environment. Predicting with it later needs the same
    columns computed for the new rows, and only this featuriser, fitted on the rows the arm trained
    on, can produce them.
    """
    if not featuriser.is_fitted:
        raise ValueError("Refusing to save an unfitted featuriser: it has no columns to produce.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": FITTED_VERSION, "featuriser": featuriser}
    return files.write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def load_featuriser(path: str | Path) -> Featuriser:
    """A featuriser written by `save_featuriser`, trusted on the same terms as in `load_fitted`."""
    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("version") != FITTED_VERSION or "featuriser" not in payload:
        raise ValueError(f"{path} is not a featuriser written at version {FITTED_VERSION}.")
    return payload["featuriser"]


def load_fitted(path: str | Path) -> FittedArm:
    """Load a saved arm, given the cell directory or the file itself.

    **Unpickling runs code and requires the libraries the arm was fitted with.** A cell is written
    by one of our runs, so it is trusted like any other file in the run directory. The versions of
    `xgboost` and `interpret-core` still matter, and the environment snapshot beside the run records
    them.
    """
    path = Path(path)
    if path.is_dir():
        path = path / FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No fitted arm at {path}. A DEFT or an IGSR cell has none: its run.json names its "
            f"model under `fitted_model`."
        )
    payload = pickle.loads(path.read_bytes())
    version = payload.get("version")
    if version != FITTED_VERSION:
        raise ValueError(
            f"{path} was written at version {version}; this reader is version {FITTED_VERSION}."
        )
    return payload["arm"]
