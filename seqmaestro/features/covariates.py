"""The covariates as they are: one feature per number measured beside each sequence.

A dataset may declare covariates at Stage 1, such as a chromatin signal over the window or an
experimental measurement (`docs/developer/dataset.md`). This featuriser hands them to a model
unchanged, so an arm whose pool holds `covariates` can use them beside the features it takes from
the sequence. It never uses the sequence, and a sequence featuriser never uses a covariate.

Each feature is named `covariate_<column>`. The prefix says what kind of feature it is in any
table, as `pos_` and `kmer_` do, and keeps the name apart from any feature a language model writes
about the same column.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from seqmaestro.features.base import (
    ORIGIN_SEQMAESTRO,
    PROVENANCE_SELECTED,
    FeatureRecord,
    Featuriser,
    Inputs,
)

#: The `kind` of a covariate's feature, and the prefix of its name.
KIND_COVARIATE = "covariate"


def covariate_feature_name(column: str) -> str:
    """The name of the feature that holds a covariate column as it is."""
    return f"{KIND_COVARIATE}_{column}"


class CovariateFeaturiser(Featuriser):
    """One column per covariate of the rows, holding its values as they are.

    `descriptions` maps each covariate to what it measures, as the dataset declares it
    (`Dataset.covariates`), and that text becomes the description of its feature. Fitting records
    which covariates the rows held, and transforming selects those by name, so rows that lack one
    are refused.
    """

    def __init__(self, descriptions: Mapping[str, str]) -> None:
        super().__init__()
        self.descriptions = dict(descriptions)
        self.columns: tuple[str, ...] | None = None

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        # `y` is ignored: the columns are the covariates as they are. See `Featuriser._fit`.
        if not inputs.covariate_columns:
            raise ValueError(
                "The rows carry no covariates, so there is nothing to hold. A pool with "
                "`covariates` needs a dataset that declares them.")
        undescribed = [name for name in inputs.covariate_columns if name not in self.descriptions]
        if undescribed:
            raise ValueError(f"No description was given for covariate(s) {undescribed}. The "
                             f"dataset declares one for each, and a feature needs it.")
        self.columns = inputs.covariate_columns
        return [
            FeatureRecord(
                name=covariate_feature_name(column),
                description=self.descriptions[column],
                kind=KIND_COVARIATE,
                origin=ORIGIN_SEQMAESTRO,
                params={"column": column},
                provenance=PROVENANCE_SELECTED,
            )
            for column in self.columns
        ]

    def _transform(self, inputs: Inputs) -> np.ndarray:
        return inputs.select(self.columns).covariates.to_numpy(dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "covariates",
            "columns": None if self.columns is None else list(self.columns),
            "n_features": len(self._records) if self.is_fitted else None,
        }
