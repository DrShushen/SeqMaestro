import ast
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from deft.provenance import ORIGIN_NAIVE, ORIGIN_UNKNOWN
from deft.utils.window import PAD, window_symbols


def _normalise_source(source: str) -> str | None:
    """The feature function's body, with names, formatting and comments normalised away.

    Two candidates computing the same thing are the same candidate, whatever the model chose to
    call them. Returns None if the source will not parse or defines no function, in which case
    the caller falls back to comparing the raw text.
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None

    body = functions[-1].body
    # A `#` comment is discarded by the parser, but a docstring is a real statement in the body.
    # Neither changes what the feature computes, so drop it too.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return None

    return ast.dump(ast.Module(body=body, type_ignores=[]))


@dataclass
class FeatureInfo:
    fn: Callable = None
    name: str = None
    description: str = None
    string: str = None
    threshold: float = None
    score: float = None
    rationale: str = None
    # Whether this candidate came from the language model or from the local naive-feature
    # generator below. Read by `src/deft/provenance.py` to answer "how much of this tree is
    # actually DEFT?" -- see `src/deft/conf/integrity/strict.yaml`.
    origin: str = ORIGIN_UNKNOWN
    # Set on a language-model candidate whose first code failed on a node's rows and which the
    # repair round corrected (`LLMFeatureFinder._repair`). `renamed_from` is the name it was
    # proposed under, where the correction computes something new and so carries a new name.
    repaired: bool = False
    renamed_from: str | None = None


    def __str__(self):
        """Return the string representation."""
        base_str = f"""
    Name: {self.name}
    Description: {self.description}
    Rationale: {self.rationale}
    Code: {self.string}
    """
        return base_str

    def _identity(self) -> tuple:
        """What makes two candidates the same feature: what it computes.

        Upstream compared `score` alone, so `remove_duplicates` deduplicated by *score
        collision*: two genuinely different features that happened to tie collapsed into one.
        Ties are routine — many single-position naive features score identically on the same
        node — so this quietly shrank the candidate pool.

        Identity is the normalised body of the generated function, which means formatting,
        comments and the function's *name* are all ignored. That matters because the model
        readily produces the same feature several times under different names; upstream's own
        notes describe ten variants of `pos_51_is_G` from a single generation round.

        Falls back to the raw source, then to name and description, for candidates that have not
        reached code generation yet.
        """
        source = self.string
        cached = getattr(self, "_identity_cache", None)
        if cached is not None and cached[0] == source:
            return cached[1]

        if source:
            normalised = _normalise_source(source)
            identity = ("code", normalised) if normalised else ("source", " ".join(source.split()))
        else:
            identity = ("semantic", self.name, self.description)

        self._identity_cache = (source, identity)
        return identity

    def __eq__(self, value) -> bool:
        """Handle eq."""
        if not isinstance(value, FeatureInfo):
            return NotImplemented
        return self._identity() == value._identity()

    def __hash__(self) -> int:
        """Handle hash."""
        return hash(self._identity())


def list_to_str(list_features):
    """Handle list to str."""
    return "\n".join(f"Feature {i+1}:{obj!s}" for i, obj in enumerate(list_features))


def _compile_naive_feature(func_name, func_str, *, name, description, rationale):
    """Build a `FeatureInfo` from a locally-generated source string.

    This source is ours, not the language model's, so it does not go through the code guard.
    Column names reach it only as `repr`'d string literals, never as identifiers. It may call
    `window_symbols`, the one rule for reading a sequence in a positional window.
    """
    feature = FeatureInfo(
        name=name,
        description=description,
        rationale=rationale,
        string=func_str,
        origin=ORIGIN_NAIVE,
    )
    namespace = {"window_symbols": window_symbols}
    exec(func_str, namespace)
    feature.fn = namespace[func_name]
    return feature


def _safe_identifier(text, used):
    """A unique, valid Python function name derived from arbitrary text.

    Column names in real tabular data are not identifiers -- `SIG:NET-seq_COND:+Spt5` is one of
    ours. The generated function needs a legal name, and it needs to stay unique after two
    different columns sanitise to the same thing.
    """
    cleaned = re.sub(r"\W+", "_", str(text)).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"col_{cleaned}" if cleaned else "col"
    candidate = cleaned
    suffix = 2
    while candidate in used:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _sequence_position_features(X, sequence_column, window=None):
    """One indicator per (position, nucleotide) over a string sequence column.

    With a positional window (`{length, anchor, pad}`, `deft.utils.window`), the positions are
    the window's, read by `window_symbols`, and a window that pads adds one indicator per position
    for the padding, `pos_<i>_is_pad`: set where the sequence stops before that position.
    """
    sequences = X[sequence_column]
    # The alphabet as it actually occurs, not an assumed ACGT: real files carry `N`, and a
    # non-DNA sequence alphabet is a legitimate use of this method. Read from whole sequences, so
    # a symbol outside the window is still known.
    nucleotides = sorted(set("".join(sequences.astype(str).tolist())))
    if window is None:
        seq_length = len(sequences.iloc[0])
        symbols = nucleotides
    else:
        seq_length = window["length"]
        symbols = [*nucleotides, PAD] if window["pad"] else nucleotides

    features = []
    used_names = set()
    for pos in range(seq_length):
        for nuc in symbols:
            func_name = _safe_identifier(f"pos_{pos}_is_{nuc}", used_names)
            if window is None:
                read = f"x[{pos}]"
                description = f"Check if position {pos} of `{sequence_column}` is {nuc}"
            else:
                read = (f"window_symbols(x, {window['length']}, {window['anchor']!r}, "
                        f"{window['pad']})[{pos}]")
                description = _window_description(sequence_column, window, pos, nuc)
            func_str = (
                f"def {func_name}(X):\n"
                f"    return X[{sequence_column!r}].apply(lambda x: {read} == {nuc!r})"
            )
            features.append(
                _compile_naive_feature(
                    func_name,
                    func_str,
                    name=f"pos_{pos}_is_{nuc}",
                    description=description,
                    rationale=(
                        f"Basic sequence composition feature checking for {nuc} at "
                        f"position {pos}"
                    ),
                )
            )
    return features


def _window_description(sequence_column, window, pos, symbol):
    """What a positional indicator over a window checks, in words."""
    frame = (f"the window of the first {window['length']} positions of `{sequence_column}`"
             if window["anchor"] == "start"
             else f"the window of the last {window['length']} positions of `{sequence_column}`")
    if symbol != PAD:
        return f"Check if position {pos} of {frame} is {symbol}"
    shorter = pos + 1 if window["anchor"] == "start" else window["length"] - pos
    return (f"Check if the sequence is shorter than {shorter}, so position {pos} of {frame} is "
            f"padding")


def _split_position_columns(columns, position_prefix):
    """Match `seq_-50 … seq_50`-style columns, returning (column, position) pairs.

    Positions are read from the suffix rather than inferred from how many columns matched, so
    a partial or one-based range works. A column whose suffix is not an integer is not a
    position column and is left to be treated as a covariate.
    """
    matched = []
    for column in columns:
        if not column.startswith(position_prefix):
            continue
        try:
            matched.append((column, int(column[len(position_prefix) :])))
        except ValueError:
            continue
    return matched


def _position_column_features(position_columns, nucleotides=("A", "C", "G", "T")):
    """One indicator per (position column, nucleotide) for a pre-split sequence."""
    features = []
    used_names = set()
    for column, pos in position_columns:
        pos_str = f"neg{abs(pos)}" if pos < 0 else str(pos)
        for nuc in nucleotides:
            func_name = _safe_identifier(f"pos_{pos_str}_is_{nuc}", used_names)
            func_str = (
                f"def {func_name}(X):\n" f"    return X[{column!r}] == {nuc!r}"
            )
            features.append(
                _compile_naive_feature(
                    func_name,
                    func_str,
                    name=f"pos_{pos}_is_{nuc}",
                    description=f"Check if nucleotide at position {pos} (`{column}`) is {nuc}",
                    rationale=(
                        f"Basic sequence composition feature checking for {nuc} at "
                        f"position {pos}"
                    ),
                )
            )
    return features


def _covariate_features(X, columns, max_categorical_levels):
    """Naive features over non-sequence columns.

    Without these, `llm_split_share` would be measured against a sequence-only baseline while
    the model is handed covariates for free -- which flatters the model rather than measuring
    it.

    Numeric and boolean columns get the identity feature, leaving `optimize_threshold` to find
    the cut point; that is exactly what CART does. Low-cardinality non-numeric columns get one
    indicator per level, capped so a high-cardinality identifier cannot swamp the pool.
    """
    used_names = set()
    features = []
    for column in columns:
        series = X[column]

        if series.isna().any():
            logging.warning(
                "Naive features: skipping covariate `%s`, which has %d missing value(s). "
                "DEFT's threshold search has no missing-value handling and would silently "
                "sort them to one side of every split.",
                column,
                int(series.isna().sum()),
            )
            continue
        if series.nunique(dropna=False) <= 1:
            logging.info(
                "Naive features: skipping covariate `%s`, which is constant.", column
            )
            continue

        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            func_name = _safe_identifier(f"value_of_{column}", used_names)
            func_str = f"def {func_name}(X):\n    return X[{column!r}]"
            features.append(
                _compile_naive_feature(
                    func_name,
                    func_str,
                    name=f"value_of_{column}",
                    description=f"The value of covariate `{column}`",
                    rationale=(
                        f"Basic covariate feature: split directly on `{column}`, as a "
                        f"conventional decision tree would"
                    ),
                )
            )
            continue

        levels = series.astype(str).unique().tolist()
        if len(levels) > max_categorical_levels:
            logging.info(
                "Naive features: skipping covariate `%s`, which has %d distinct values "
                "(cap is %d). Raise `naive_max_categorical_levels` to include it.",
                column,
                len(levels),
                max_categorical_levels,
            )
            continue

        for level in sorted(levels):
            func_name = _safe_identifier(f"{column}_is_{level}", used_names)
            func_str = (
                f"def {func_name}(X):\n"
                f"    return X[{column!r}].astype(str) == {level!r}"
            )
            features.append(
                _compile_naive_feature(
                    func_name,
                    func_str,
                    name=f"{column}_is_{level}",
                    description=f"Check if covariate `{column}` equals '{level}'",
                    rationale=(
                        f"Basic covariate feature: indicator for one level of the "
                        f"categorical column `{column}`"
                    ),
                )
            )
    return features


def get_naive_features(
    X,
    sequence_column="raw_sequence",
    position_prefix="seq_",
    max_categorical_levels=20,
    positional_window=None,
):
    """The local baseline population every node is seeded with.

    Two things live here, and both must, because this population is the yardstick the
    language model is measured against (`src/deft/provenance.py`). Anything DEFT is given but the
    naive pool cannot reach is a column the model gets for free.

    - **Sequence**, either as `sequence_column` holding equal-length strings, or strings of
      different lengths read in `positional_window`, or, when that is None, as pre-split
      per-position columns matching `position_prefix`.
    - **Covariates**: every remaining column.

    A `sequence_column` that is named but absent is an error rather than a silent fallback to
    covariates only -- a typo there would otherwise remove the sequence from the baseline
    without saying so.
    """
    columns = list(X.columns)
    features = []
    consumed = set()

    if sequence_column is not None:
        if sequence_column not in X.columns:
            raise KeyError(
                f"Sequence column `{sequence_column}` is not in the data. Available "
                f"columns: {columns}. Pass `sequence_column=None` for a dataset with no "
                f"sequence."
            )
        features.extend(_sequence_position_features(X, sequence_column, positional_window))
        consumed.add(sequence_column)
    else:
        position_columns = _split_position_columns(columns, position_prefix)
        if position_columns:
            features.extend(_position_column_features(position_columns))
            consumed.update(column for column, _ in position_columns)
        else:
            logging.info(
                "Naive features: no sequence column and nothing matching `%s*`; building "
                "from covariates alone.",
                position_prefix,
            )

    n_sequence = len(features)
    features.extend(
        _covariate_features(
            X,
            [column for column in columns if column not in consumed],
            max_categorical_levels,
        )
    )
    logging.info(
        "Naive features: %d from the sequence, %d from %d covariate column(s).",
        n_sequence,
        len(features) - n_sequence,
        len(columns) - len(consumed),
    )
    return features
