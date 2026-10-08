from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from deft.data.dataset_plugins import BaseDatasetPlugin
from deft.targets import BINARY, check_target, checked_kind
from deft.utils.window import checked_window

logger = logging.getLogger(__name__)


# Parameters that used to make this plugin split the CSV itself. Named individually so that an
# old config fails with an explanation rather than with `unexpected keyword argument`.
_SPLIT_PARAMETERS = {
    "test_size": "the fraction held out for testing",
    "val_size": "the fraction held out for validation",
    "train_size": "the fraction kept for training",
    "random_state": "the seed for the internal split",
    "stratify": "the column stratified on",
    "shuffle": "whether rows were shuffled before splitting",
}


class GenericCsvDatasetPlugin(BaseDatasetPlugin):
    """A CSV of training rows: a target column, optionally a sequence column, and covariates.

    **This plugin does not split.** It trains on every row of the file it is given, so the
    caller passes the training split and keeps the others. That is not only a separation of
    concerns: the bundle returned here has no place for a test set, so a plugin that split
    internally would *discard* the held-out rows. Point the old version at a 3,500-row training
    file and it trained on 2,800 while reporting nothing amiss. Passing a split parameter is
    therefore an error rather than a no-op -- see `_SPLIT_PARAMETERS`.

    It also matters for comparability: several methods must see identical rows, and
    `train_test_split` over a differently-ordered frame selects different rows at the same seed.

    **Columns.** Everything that is not the target is handed to DEFT: the sequence column if
    there is one, and every other column as a covariate. Excluding a column that should not be
    learned from -- an identifier, a leaked score, a redundant encoding -- belongs to whatever
    prepared the file, not here. `covariate_columns` and `drop_columns` exist for the cases
    where re-preparing the file is not worth it.

    **What each column means** can be stated in `column_descriptions`, `{column: text}`. The
    prompts show each described column's text beside its type and range, since a covariate's name
    rarely says what it measures. A description of a column DEFT is not given is refused.

    **Sequences of different lengths** need a `positional_window` (`{length, anchor, pad}`, see
    `deft.utils.window`), in which the positional features read their positions. Without one,
    every sequence must be one length, as positions index into the string by offset.

    **The target's kind** (`target_kind`, `deft.targets`) says how the target column is read: the
    labels 0 and 1 for a binary target, finite numbers for a continuous one. Either is refused when
    the column holds anything else.
    """

    def __init__(
        self,
        csv_path: str,
        target_column: str = "label",
        sequence_column: str | None = "raw_sequence",
        covariate_columns: Sequence[str] | None = None,
        drop_columns: Sequence[str] | None = None,
        weight_column: str | None = None,
        tree_filename: str = "tree_generic.dill",
        column_descriptions: Mapping[str, str] | None = None,
        positional_window: Mapping[str, Any] | None = None,
        target_kind: str = BINARY,
        **unsupported: Any,
    ):
        self._reject_unsupported(unsupported)
        self.target_kind = checked_kind(target_kind)

        csv_path_obj = Path(csv_path)
        if not csv_path_obj.is_absolute():
            csv_path_obj = (Path.cwd() / csv_path_obj).resolve()
        self.csv_path = csv_path_obj
        self.target_column = str(target_column)
        self.sequence_column = (
            None if sequence_column in (None, "", "null") else str(sequence_column)
        )
        self.covariate_columns = _as_list(covariate_columns)
        self.weight_column = (
            None if weight_column in (None, "", "null") else str(weight_column)
        )
        self.drop_columns = _as_list(drop_columns)
        self.tree_filename = str(tree_filename)
        self.column_descriptions = {
            str(column): str(text) for column, text in dict(column_descriptions or {}).items()
        }
        self.positional_window = checked_window(positional_window)
        if self.positional_window is not None and self.sequence_column is None:
            raise ValueError(
                "A positional window is read over the sequence column, and `sequence_column` is "
                "null. Name the sequence column, or remove the window."
            )

    @staticmethod
    def _reject_unsupported(unsupported: dict[str, Any]) -> None:
        """Refuse a config that expects this plugin to split, instead of quietly ignoring it."""
        split_args = [name for name in unsupported if name in _SPLIT_PARAMETERS]
        if split_args:
            described = ", ".join(f"`{n}` ({_SPLIT_PARAMETERS[n]})" for n in sorted(split_args))
            raise TypeError(
                f"This plugin no longer splits the CSV, so {described} would have no effect. "
                f"It trains on every row of `csv_path`, so pass the training split as "
                f"`csv_path` and hold the rest back yourself. Remove these keys from the "
                f"dataset config."
            )
        other = sorted(set(unsupported) - set(_SPLIT_PARAMETERS))
        if other:
            raise TypeError(
                f"Unknown dataset plugin parameter(s): {', '.join(other)}."
            )

    def _read(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError(f"CSV has no rows: {self.csv_path}")
        return df

    def _select_columns(self, df: pd.DataFrame) -> list[str]:
        """Which columns DEFT is given, in file order."""
        if self.target_column not in df.columns:
            raise KeyError(
                f"Target column `{self.target_column}` not found in {self.csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        if self.sequence_column is not None and self.sequence_column not in df.columns:
            raise KeyError(
                f"Sequence column `{self.sequence_column}` not found in {self.csv_path}. "
                f"Available columns: {list(df.columns)}. Set `sequence_column: null` for a "
                f"dataset with no sequence."
            )

        if self.covariate_columns is not None:
            missing = [c for c in self.covariate_columns if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Covariate column(s) not found in {self.csv_path}: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )
            keep = set(self.covariate_columns)
        else:
            keep = set(df.columns)

        keep.discard(self.target_column)
        # A fitting weight is not a feature. Excluded here rather than left to the caller's
        # `drop_columns`, because forgetting it would silently make the weights a covariate.
        if self.weight_column is not None:
            if self.weight_column not in df.columns:
                raise KeyError(
                    f"Weight column `{self.weight_column}` not found in {self.csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )
            keep.discard(self.weight_column)
        if self.drop_columns is not None:
            missing = [c for c in self.drop_columns if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Cannot drop column(s) absent from {self.csv_path}: {missing}."
                )
            keep -= set(self.drop_columns)
            if self.sequence_column in set(self.drop_columns):
                raise ValueError(
                    f"`{self.sequence_column}` is both the sequence column and in "
                    f"`drop_columns`. Set `sequence_column: null` if the sequence is not to "
                    f"be used."
                )
        if self.sequence_column is not None:
            keep.add(self.sequence_column)

        selected = [c for c in df.columns if c in keep]
        if not selected:
            raise ValueError(
                f"No feature columns left in {self.csv_path} after selection; DEFT has "
                f"nothing to build features from."
            )
        return selected

    def _check_sequences(self, sequences: pd.Series) -> tuple[pd.Series, list[int]]:
        """The sequences, and the shortest and longest length.

        Without a positional window, equal length is a precondition, since positional features
        index into the string. With one, lengths may differ, and a sequence shorter than a window
        declared without padding is refused.

        Missing values are caught here rather than after the cast to `str`, which would turn
        them into the literal three-character string "nan" and report a length mismatch.
        """
        if sequences.isna().any():
            n = int(sequences.isna().sum())
            raise ValueError(
                f"Sequence column `{self.sequence_column}` has {n} missing value(s) in "
                f"{self.csv_path}."
            )
        sequences = sequences.astype(str)
        lengths = sequences.str.len()
        distinct = sorted(lengths.unique().tolist())
        window = self.positional_window
        if window is not None:
            short = int((lengths < window["length"]).sum())
            if short and not window["pad"]:
                raise ValueError(
                    f"Sequence column `{self.sequence_column}` holds {short} sequence(s) shorter "
                    f"than the {window['length']}-position window, which is declared without "
                    f"padding (shortest {distinct[0]}). Filter them out before DEFT, or declare "
                    f"the window with padding."
                )
            return sequences, [int(distinct[0]), int(distinct[-1])]
        if len(distinct) != 1:
            counts = lengths.value_counts().head(5).to_dict()
            raise ValueError(
                f"Sequence column `{self.sequence_column}` holds sequences of different "
                f"lengths {distinct[:5]}{'...' if len(distinct) > 5 else ''} "
                f"(most common: {counts}). DEFT's positional features index into the string by "
                f"offset, so every sequence must be the same length and aligned to the same "
                f"landmark, or the dataset must declare a `positional_window`."
            )
        return sequences, [int(distinct[0]), int(distinct[0])]

    def load_for_deft(self) -> dict[str, Any]:
        df = self._read()
        selected = self._select_columns(df)

        X_train = df[selected].reset_index(drop=True)
        try:
            check_target(df[self.target_column], self.target_kind)
        except ValueError as error:
            raise ValueError(f"Target column `{self.target_column}` in {self.csv_path}: {error}") \
                from error
        y_train = df[[self.target_column]].astype(
            int if self.target_kind == BINARY else float).reset_index(drop=True)

        undescribed = sorted(set(self.column_descriptions) - set(selected))
        if undescribed:
            raise KeyError(
                f"`column_descriptions` describes column(s) DEFT is not given: {undescribed}. "
                f"Given: {selected}."
            )

        if self.sequence_column is not None:
            sequences, (shortest, longest) = self._check_sequences(X_train[self.sequence_column])
            X_train[self.sequence_column] = sequences
            if self.positional_window is None:
                logger.info(
                    "Sequence column `%s`: %d rows of length %d.",
                    self.sequence_column,
                    len(X_train),
                    shortest,
                )
            else:
                logger.info(
                    "Sequence column `%s`: %d rows of lengths %d to %d, read by position in "
                    "a %d-position window from the %s%s.",
                    self.sequence_column,
                    len(X_train),
                    shortest,
                    longest,
                    self.positional_window["length"],
                    self.positional_window["anchor"],
                    ", padded" if self.positional_window["pad"] else "",
                )
        else:
            logger.info("No sequence column; DEFT will work from covariates alone.")

        n_covariates = len(selected) - (1 if self.sequence_column is not None else 0)
        target = y_train[self.target_column]
        logger.info(
            "Loaded %d training rows and %d covariate column(s) from %s; %s target `%s` (%s).",
            len(X_train),
            n_covariates,
            self.csv_path,
            self.target_kind,
            self.target_column,
            f"positive rate {target.mean():.4f}" if self.target_kind == BINARY
            else f"mean {target.mean():.4f}, standard deviation {target.std():.4f}",
        )

        bundle = {
            "X_train": X_train,
            "y_train": y_train,
            "tree_filename": self.tree_filename,
            "prompt_context": {},
            "sequence_column": self.sequence_column,
            "column_descriptions": dict(self.column_descriptions),
            "positional_window": self.positional_window,
            "target_kind": self.target_kind,
        }
        if self.weight_column is not None:
            # Index-aligned to `y_train`, which is what the tree's per-row weighting matches on.
            bundle["sample_weight"] = (
                df[self.weight_column].astype(float).reset_index(drop=True)
            )
            logger.info(
                "Per-row fitting weights from `%s`: %d rows, total weight %.4f.",
                self.weight_column, len(bundle["sample_weight"]),
                float(bundle["sample_weight"].sum()),
            )
        return bundle


def _as_list(value: Iterable[str] | None) -> list[str] | None:
    """Accept a Hydra list, a comma-separated string, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]
