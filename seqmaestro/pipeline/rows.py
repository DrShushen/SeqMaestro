"""The rows a run's stages work on before the reveal: the training and the validation split.

Every stage before the reveal reads these two files only, whereas the reveal opens the third, the
held-out split (`seqmaestro.harness.heldout`).
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
        """The rows the reliability panel works on (D): the training rows followed by the
        validation rows, with a fresh index. The first `len(self.train)` rows are therefore the
        training rows."""
        return pd.concat([self.train, self.val], ignore_index=True)
