"""An IGSR arm's Stage 2 cells: the settings a run starts IGSR with, and the fit and the
prediction made on IGSR's side through `experiments/adapters/igsr_bundle.py`.

IGSR fits a continuous target by least squares over terms a language model writes during its
search. It chooses on the validation rows, so a cell is handed the labels to choose against there
(`handover.write_inputs`), which on a permuted cell are shuffled as the fitting rows' are.

This side never imports `igsr`. It prepares the inputs, starts the adapter under IGSR's
interpreter, and takes the bundle that comes back.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from seqmaestro.arms import IGSR, IGSR_ENVIRONMENT
from seqmaestro.features.base import Inputs
from seqmaestro.llm import provider_from_environment
from seqmaestro.pipeline.repository import ADAPTERS, REPO
from seqmaestro.pipeline.vendored.handover import (
    BUNDLE,
    COLUMNS_MANIFEST,
    VendoredArm,
    call_overrides,
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
                overrides: list[str], permutation_repeats: int, guidance: str | None) -> None:
    """Cross the boundary: the adapter runs under IGSR's interpreter, not this one. `guidance`,
    where there is any, is what the person running the analysis asks of the form of the terms (the
    grid's `generated_features.guidance`), which the adapter adds to IGSR's data dictionary."""
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
    if guidance is not None:
        command += ["--guidance", guidance]
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
    term on each split, whereas the proposer scales it with the rows. Its budget is its `models`
    settings, and its language model is the one the environment names (`overrides`)."""
    generated = config.grid.generated_features
    return {
        "experiment.function_terms.repair_rounds": generated.repair_rounds,
        "experiment.function_terms.max_undefined_share": generated.max_undefined_share,
        "experiment.function_terms.max_example_length": generated.max_example_length,
        "experiment.function_terms.timeout_seconds": generated.max_seconds,
    }


#: IGSR's own configuration, whose values hold where a run states no override.
IGSR_CONF = REPO / "methods" / "igsr" / "src" / "igsr" / "conf"


def call_settings(config: ExperimentConfig) -> dict[str, int]:
    """What decides how many calls IGSR makes in a cell, from its configuration with the run's
    overrides: its search's budget of nodes (`experiment.total_budget`) and the successors its agent
    writes at each node it expands (`experiment.n_successors`)."""
    stated = dict(item.split("=", 1)
                  for item in config.models.get(IGSR, {}).get("config_overrides", []))
    experiment = yaml.safe_load((IGSR_CONF / "experiment" / "igsr.yaml").read_text())
    return {"budget": int(stated.get("experiment.total_budget", experiment["total_budget"])),
            "successors": int(stated.get("experiment.n_successors", experiment["n_successors"]))}


#: For each provider a run can call (`seqmaestro.llm.PROVIDERS`), IGSR's config for it.
LANGUAGE_MODEL = {"azure": "seqmaestro_azure", "azure_foundry": "seqmaestro_azure_foundry",
                  "openai": "seqmaestro_openai", "claude": "seqmaestro_claude"}
LANGUAGE_MODEL_KEYS = ("llm",)
#: The settings of IGSR's language model (`llm`) that the run's `language_model` states, by its
#: field.
CALL_KEYS = {"reasoning_effort": "llm.reasoning_effort", "retries": "llm.num_retries",
             "timeout_seconds": "llm.request_timeout"}


def run_overrides(config: ExperimentConfig) -> list[str]:
    """The overrides the run's config makes to IGSR's config: what the grid states, then the run's
    `models.igsr.config_overrides`. The function does not read the environment, so the driver can
    refuse a conflicting config before anything starts."""
    return config_overrides(config, IGSR, stated(config), (*LANGUAGE_MODEL_KEYS, *CALL_KEYS.values()))


def overrides(config: ExperimentConfig) -> list[str]:
    """Every override IGSR's config takes in a run: the language model of the provider the
    environment names and how the run calls it (`call_overrides`), and then `run_overrides`."""
    provider = provider_from_environment()
    return [f"llm={LANGUAGE_MODEL[provider.name]}",
            *call_overrides(config.language_model, provider, CALL_KEYS), *run_overrides(config)]


class Arm(VendoredArm):
    """An IGSR arm of a run: its search on the sequence, with or without the covariates.

    This side computes nothing from the sequence, since the terms' code uses the sequence directly,
    at whatever length it comes. IGSR's search chooses on the validation rows (the iteration it
    keeps, the terms it prunes), and its prompts show the language model the error and each term's
    influence there. A cell is therefore handed the labels to choose against, which on a permuted
    cell are shuffled as the fitting rows' are, and it is scored against the real ones.
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
                    permutation_repeats=self.config.grid.permutation_repeats,
                    guidance=self.config.grid.generated_features.guidance)
