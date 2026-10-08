"""An IGSR arm's Stage 2 cells: the pools it can be given, and the fit and the prediction made on
IGSR's side through `experiments/adapters/igsr_bundle.py`.

IGSR fits a continuous target by least squares over terms a language model writes during its
search. It chooses on the validation rows, so a cell is handed the labels to choose against there
(`handover.write_inputs`), which on a control arm are shuffled as the fitting rows' are.

This side never imports `igsr`. It prepares the inputs, starts the adapter under IGSR's
interpreter, and takes the bundle that comes back.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from seqmaestro.arms import IGSR, IGSR_ENVIRONMENT
from seqmaestro.features.base import Inputs
from seqmaestro.llm import provider_from_environment
from seqmaestro.pipeline.repository import ADAPTERS
from seqmaestro.pipeline.vendored.handover import (
    BUNDLE,
    COLUMNS_MANIFEST,
    VendoredArm,
    config_overrides,
    input_frame,
    predict_numbered,
    write_inputs,
)
from seqmaestro.provenance import environment_python
from seqmaestro.target import CONTINUOUS

if TYPE_CHECKING:
    from seqmaestro.pipeline.config import ExperimentConfig

ADAPTER = ADAPTERS / "igsr_bundle.py"
#: IGSR's interpreter on this machine, in the environment `seqmaestro.arms` names (see
#: ENVIRONMENTS.md for the override). Everything that shells out to IGSR uses it.
IGSR_PYTHON = environment_python(IGSR_ENVIRONMENT)
#: What an IGSR cell's `run.json` says about how the reveal predicts it: `predict_cell`, below.
PREDICT_WITH = "igsr_bundle"


def run_adapter(paths: dict, bundle_dir: Path, *, task: str, seed: int, target_column: str,
                overrides: list[str], permutation_repeats: int) -> None:
    """Cross the boundary: the adapter runs under IGSR's interpreter, not this one."""
    command = [
        str(IGSR_PYTHON), str(ADAPTER), "fit",
        "--train", str(paths["train"]), "--val", str(paths["val"]),
        # What the search chooses against on the validation rows. `--val` keeps the real labels,
        # which the cell is scored against.
        "--choose-on", str(paths["choose_on"]),
        # Where `marginal_effect` is measured: every training row, with the real target.
        "--effect", str(paths["effect"]),
        "--columns", str(Path(paths["train"]).parent / COLUMNS_MANIFEST),
        "--out-bundle", str(bundle_dir),
        "--target-column", target_column, "--target-kind", CONTINUOUS,
        "--task", task, "--seed", str(seed),
        # Passed explicitly, so every arm's cross-arm importance is measured the same number of
        # times.
        "--permutation-repeats", str(permutation_repeats), "--permutation-seed", str(seed),
    ]
    for override in overrides:
        command += ["--config-override", override]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"adapter exited {completed.returncode}")


def predict_cell(cell: Path, inputs: Inputs, sequence_column: str) -> np.ndarray:
    """The values a Stage 2 cell's saved model predicts for these rows, on IGSR's side. The reveal
    calls this for an IGSR cell, and the rows cross without their target."""
    return predict_numbered(
        [str(IGSR_PYTHON), str(ADAPTER), "predict", "--bundle", str(cell / BUNDLE)],
        input_frame(cell, inputs, sequence_column), cell)


def stated(config: ExperimentConfig) -> dict[str, object]:
    """What a run's grid states for IGSR, as overrides of IGSR's config: its policy for the terms
    its language model writes, the grid's `generated_features`. IGSR applies the time limit to each
    term on each split, where the proposer scales it with the rows. Its budget is its `models`
    settings, and its language model is the one the environment names (`overrides`)."""
    generated = config.grid.generated_features
    return {
        "experiment.function_terms.repair_rounds": generated.repair_rounds,
        "experiment.function_terms.max_undefined_share": generated.max_undefined_share,
        "experiment.function_terms.max_example_length": generated.max_example_length,
        "experiment.function_terms.timeout_seconds": generated.max_seconds,
    }


#: For each provider a run can call (`seqmaestro.llm.PROVIDERS`), IGSR's config for it.
LANGUAGE_MODEL = {"azure": "seqmaestro_azure", "openai": "seqmaestro_openai",
                  "claude": "seqmaestro_claude"}
LANGUAGE_MODEL_KEYS = ("llm",)


def run_overrides(config: ExperimentConfig) -> list[str]:
    """What the run's config makes of IGSR's config: what the grid states, then the run's
    `models.igsr.config_overrides`. It reads no environment, so the driver can refuse a
    conflicting config before anything starts."""
    return config_overrides(config, IGSR, stated(config), LANGUAGE_MODEL_KEYS)


def overrides(config: ExperimentConfig) -> list[str]:
    """Every override IGSR's config takes in a run: the language model of the provider the
    environment names, and then `run_overrides`."""
    return [f"llm={LANGUAGE_MODEL[provider_from_environment().name]}", *run_overrides(config)]


class Arm(VendoredArm):
    """An IGSR arm of a run: its search on the sequence, with or without the covariates.

    Nothing is computed from the sequence on this side, since the terms read it in their code, at
    whatever length it comes. IGSR's search chooses on the validation rows (the iteration it keeps,
    the terms it prunes), and its prompts show the language model the error and each term's
    influence there. So a cell is handed the labels to choose against, which on a control arm are
    shuffled as the fitting rows' are, and is scored against the real ones.
    """

    predict_with = PREDICT_WITH

    def write(self, fit_rows: pd.DataFrame, y_fit: np.ndarray, y_choose: np.ndarray,
              into: Path) -> dict:
        return write_inputs(fit_rows, y_fit, self.rows.train, self.rows.val, into,
                            columns=self.dataset.columns_for(self.pool), choose_on=y_choose,
                            descriptions=self.dataset.covariates_for(self.pool),
                            window=self.dataset.positional_window)

    def run(self, paths: dict, bundle: Path, seed: int) -> None:
        run_adapter(paths, bundle, task=self.config.dataset.task, seed=seed,
                    target_column=self.dataset.columns.target, overrides=overrides(self.config),
                    permutation_repeats=self.config.grid.permutation_repeats)
