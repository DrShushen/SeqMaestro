import gzip
import io
from copy import deepcopy

import dill
import numpy as np
import pandas as pd


def evaluate_feature_function(feature_fn, X):
    """Evaluate `feature_fn` on a defensive copy to prevent in-place mutations."""
    return feature_fn(X.copy(deep=True))


def _prediction_to_scalar(prediction):
    """Normalize node predictions to a scalar float."""
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.squeeze()
    if isinstance(prediction, pd.Series):
        if prediction.empty:
            return np.nan
        return float(prediction.iloc[0])

    arr = np.asarray(prediction)
    if arr.ndim == 0:
        return float(arr.item())
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def feature_output_to_series(feature_output, index):
    """Convert a feature function output to a Series aligned with `index`."""
    if isinstance(feature_output, pd.DataFrame):
        feature_output = (
            feature_output.iloc[:, 0]
            if feature_output.shape[1] == 1
            else feature_output.squeeze()
        )

    if isinstance(feature_output, pd.Series):
        if not feature_output.index.equals(index):
            try:
                feature_output = feature_output.reindex(index)
            except Exception:
                feature_output = pd.Series(feature_output.to_numpy(), index=index)
        return feature_output

    arr = np.asarray(feature_output)
    if arr.ndim == 0:
        return pd.Series(np.repeat(arr.item(), len(index)), index=index)

    arr = arr.reshape(-1)
    if arr.size != len(index):
        raise ValueError(
            "Feature function output size does not match number of input samples."
        )
    return pd.Series(arr, index=index)


def _compute_split_masks(
    node, X_subset, right_is_complement=True, as_numpy=False
):
    """
    Compute left/right split masks for a node on `X_subset`.
    """
    feature_values = evaluate_feature_function(node.feature.fn, X_subset)
    feature_values = feature_output_to_series(feature_values, X_subset.index)
    left_mask = feature_values <= node.feature.threshold

    right_mask = (
        ~left_mask if right_is_complement else feature_values > node.feature.threshold
    )

    if as_numpy:
        return (
            left_mask.to_numpy(dtype=bool),
            right_mask.to_numpy(dtype=bool),
        )
    return left_mask, right_mask


def _finalize_predictions(predictions, return_proba):
    """Return probabilities or thresholded predictions with consistent behavior."""
    if return_proba:
        return predictions
    return predictions > 0.5


def predict_all_depths_vectorized(root, X, max_depth, return_proba=False):
    """
    Predict for depths 0..max_depth in one incremental, vectorized traversal.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    n_samples = len(X)
    predictions = np.zeros((max_depth + 1, n_samples), dtype=float)
    if n_samples == 0:
        return predictions if return_proba else predictions > 0.5

    # Frontier stores disjoint sample groups currently at each node for this depth.
    frontier = [(root, np.arange(n_samples, dtype=int))]

    for depth in range(max_depth + 1):
        # Current-depth predictions come directly from each frontier node.
        for node, indices in frontier:
            if len(indices) == 0:
                continue
            predictions[depth, indices] = _prediction_to_scalar(node.prediction)

        if depth == max_depth:
            break

        next_frontier = []
        for node, indices in frontier:
            if len(indices) == 0:
                continue

            # Leaf predictions stay unchanged at deeper depths.
            if node._is_leaf():
                next_frontier.append((node, indices))
                continue

            X_subset = X.iloc[indices]
            left_mask, right_mask = _compute_split_masks(
                node=node,
                X_subset=X_subset,
                right_is_complement=True,
                as_numpy=True,
            )

            if left_mask.size != len(indices):
                raise ValueError(
                    "Feature mask size does not match number of node samples."
                )

            left_indices = indices[left_mask]
            right_indices = indices[right_mask]

            if len(left_indices) > 0:
                next_frontier.append((node.left, left_indices))
            if len(right_indices) > 0:
                next_frontier.append((node.right, right_indices))

        frontier = next_frontier
        if not frontier:
            break

    return _finalize_predictions(predictions, return_proba)


def predict_at_depth_vectorized(root, X, depth, return_proba=False):
    """
    Vectorized single-depth prediction helper.
    """
    all_predictions = predict_all_depths_vectorized(
        root=root, X=X, max_depth=depth, return_proba=True
    )
    predictions = all_predictions[depth]
    return _finalize_predictions(predictions, return_proba)


def compress_node(node):
    # Deep copy to avoid modifying original
    """Handle compress node."""
    node_copy = deepcopy(node)

    # Recursively compress artifacts in all nodes
    def compress_artifacts_recursive(n):
        """Handle compress artifacts recursive."""
        if n is None:
            return

        if n.artifacts:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9) as f:
                dill.dump(n.artifacts, f)
            n.artifacts = buffer.getvalue()

        compress_artifacts_recursive(n.left)
        compress_artifacts_recursive(n.right)

    compress_artifacts_recursive(node_copy)
    return node_copy


def decompress_node(compressed_node):
    """Read back a node tree written by `compress_node`.

    Nothing in this package calls it, so a dead-code sweep will keep finding it. It is not
    dead: `compress_node` runs on every node expansion and gzips each node's artifacts into
    `intermediate_tree.pkl`, and this is the only thing that can read them. Deleting it would
    leave that snapshot unreadable whenever `store_node_artifacts` is true.
    """
    if compressed_node is None:
        return None

    def decompress_recursive(node):
        """Handle decompress recursive."""
        if node is None:
            return

        if node.artifacts:
            buffer = io.BytesIO(node.artifacts)
            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                node.artifacts = dill.load(f)

        decompress_recursive(node.left)
        decompress_recursive(node.right)

    node_copy = deepcopy(compressed_node)
    decompress_recursive(node_copy)
    return node_copy
