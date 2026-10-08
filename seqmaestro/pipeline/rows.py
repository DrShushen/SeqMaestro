"""The rows a run's stages work on before the reveal: the training and the validation split.

Every stage before the reveal reads these two files and no other. The reveal opens the third
(`seqmaestro.harness.heldout`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from seqmaestro.dataset import Dataset, load_dataset

if TYPE_CHECKING:
    from seqmaestro.pipeline.config import ExperimentConfig


@dataclass(frozen=True)
class Rows:
    """A run's dataset, with its training and validation rows read once."""

    dataset: Dataset
    train: pd.DataFrame
    val: pd.DataFrame

    @classmethod
    def of(cls, config: ExperimentConfig) -> Rows:
        dataset = load_dataset(config.dataset.directory)
        return cls(dataset, pd.read_csv(dataset.train_path), pd.read_csv(dataset.val_path))

    def discovery(self) -> pd.DataFrame:
        """D, the rows the reliability panel works on: the training rows, then the validation
        rows, numbered anew. Its first `len(self.train)` rows are the training rows."""
        return pd.concat([self.train, self.val], ignore_index=True)
