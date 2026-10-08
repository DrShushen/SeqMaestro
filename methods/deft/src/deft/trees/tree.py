import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import dill
import numpy as np
import pandas as pd

from deft.budget import BudgetExceeded
from deft.targets import BINARY, CONTINUOUS, check_target, checked_kind
from deft.trees.split_criterion import SplitCriterion, normalize_class_label
from deft.utils.feature import FeatureInfo
from deft.utils.tree import (
    compress_node,
    evaluate_feature_function,
    feature_output_to_series,
    predict_at_depth_vectorized,
)

logger = logging.getLogger(__name__)

ClassWeight = Mapping[Any, float] | str | None


@dataclass
class Node:
    feature: Optional[
        "FeatureInfo"
    ]  # Contains: fn, name, description, string, threshold, score
    history: list | None
    left: Optional["Node"]  # This is the left node
    right: Optional["Node"]  # This is the right node
    prediction: float | None
    artifacts: dict | None = None  # Used for logging and post-hoc debugging purposes

    # Implement a function to check if the node is a leaf node
    def _is_leaf(self):
        """Check whether leaf."""
        return self.left is None and self.right is None


class AdaptiveDecisionTree:
    #: A tree saved before trees recorded their kind of target is binary, the only kind there was.
    target_kind = BINARY

    def __init__(
        self,
        feature_finder,
        splitting_criterion: SplitCriterion,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        store_node_artifacts=False,
        save_locally=True,
        folder_path=None,
        class_weight: ClassWeight = None,
        sample_weight=None,
        target_kind: str = BINARY,
    ):
        """Initialize the instance. `target_kind` is `deft.targets`' binary or continuous: what
        the leaves predict, and which splitting criteria the tree accepts."""
        self.target_kind = checked_kind(target_kind)
        self.feature_finder = feature_finder
        self.splitting_criterion = splitting_criterion
        self.max_depth = max_depth
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.store_node_artifacts = store_node_artifacts
        self.n_leaves = 0
        self.save_locally = save_locally
        self.folder_path = folder_path
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.class_weight_: dict[Any, float] | None = None

        # Set by `run_experiment` after instantiation, to a `deft.provenance.RunProvenance`.
        # None means nothing is recorded and no integrity check runs.
        self.provenance = None

        # Set by `run_experiment` to a `deft.budget.RunBudget`. None means no limits.
        self.budget = None
        # True once a budget limit stopped growth. The tree is then complete but shallower than
        # `max_depth`, and says so about itself.
        self.budget_stopped = False
        self.budget_stopped_reason = None

    def _make_leaf(self, node, node_y):
        """Turn a node into a leaf carrying its rows' prediction (`_get_prediction`)."""
        node.feature = FeatureInfo()
        node.prediction = self._get_prediction(node_y)

    def _budget_start_node(self, depth):
        """Open per-node cost accounting, so cost per node can be read off a run."""
        if self.budget is not None:
            self.budget.start_node(index=len(self.budget.nodes), depth=depth)

    def _budget_finish_node(self):
        if self.budget is not None:
            self.budget.finish_node()

    def _provenance_start_node(self, depth, n_samples):
        """Open a provenance record for the node about to be expanded."""
        if self.provenance is not None:
            self.provenance.start_node(depth=depth, n_samples=int(n_samples))

    def _provenance_finish_node(self, split_feature=None, budget_stopped=False):
        """Close the current provenance record and apply the per-node integrity policy.

        May raise `deft.provenance.DegradedRunError`, which is deliberately not caught here: it
        should stop the run at the first node the language model failed to contribute to,
        rather than after a whole tree has been built. Pass `split_feature=None` when the node
        ended as a leaf despite being expanded, and `budget_stopped=True` when the run budget
        cut it short -- those nodes are recorded but exempt from the checks.
        """
        if self.provenance is not None:
            self.provenance.finish_node(
                split_feature=split_feature, budget_stopped=budget_stopped
            )

    def fit(self, X, y, **kwargs):
        """Fit values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AdaptiveDecisionTree.fit expects X to be a pandas DataFrame. "
                f"Received {type(X).__name__}."
            )
        if self.target_kind not in self.splitting_criterion.target_kinds:
            raise ValueError(
                f"The splitting criterion {type(self.splitting_criterion).__name__} is written for "
                f"a {' or '.join(self.splitting_criterion.target_kinds) or 'no'} target, and this "
                f"tree's target is {self.target_kind}. For a binary target use "
                f"`splitting_criterion=gini` or `information_gain`, and for a continuous one "
                f"`squared_error`."
            )
        check_target(y, self.target_kind)
        if self.save_locally and not self.folder_path:
            logger.warning(
                "save_locally=True but folder_path is not set; disabling local snapshots."
            )
        # Per-row weights, when given, *replace* the class-weight mapping rather than compounding
        # with it: they already encode whatever balancing the design intends, and the criterion
        # accepts either. Held as an index-aligned Series so every node subset aligns by index.
        if self.sample_weight is not None:
            # Length first: attaching the labels' index to a mismatched vector raises from pandas
            # with a message about index lengths, which says nothing about what the caller got wrong.
            values = np.asarray(self.sample_weight, dtype=float).ravel()
            if len(values) != len(y):
                raise ValueError(
                    f"`sample_weight` has {len(values)} entries for {len(y)} training rows."
                )
            if not np.all(np.isfinite(values)) or (values < 0).any():
                raise ValueError("`sample_weight` must be finite and non-negative.")
            index = getattr(y, "index", None)
            self.class_weight_ = pd.Series(values, index=index) if index is not None \
                else pd.Series(values)
        else:
            self.class_weight_ = self._resolve_class_weight(y)
        self.splitting_criterion.prepare(y, self.class_weight_)
        self.root = self._grow_tree(X, y, depth=0, history=[], **kwargs)

    def _resolve_class_weight(self, y) -> dict[Any, float] | None:
        """Resolve class_weight into a concrete class-to-weight mapping."""
        if self.class_weight is None:
            return None
        if self.target_kind == CONTINUOUS:
            raise ValueError(
                "`class_weight` weights the classes of a binary target, and a continuous target "
                "has none. Leave it null, or weight the rows with `sample_weight`."
            )

        y_values = np.asarray(y)
        classes, counts = np.unique(y_values, return_counts=True)

        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(
                    "`class_weight` must be None, 'balanced', or a mapping from "
                    "class labels to weights."
                )
            n_samples = len(y_values)
            n_classes = len(classes)
            return {
                normalize_class_label(class_label): n_samples
                / (n_classes * count)
                for class_label, count in zip(classes, counts, strict=True)
            }

        if not isinstance(self.class_weight, Mapping):
            raise TypeError(
                "`class_weight` must be None, 'balanced', or a mapping from "
                "class labels to weights."
            )

        resolved_class_weight = {
            normalize_class_label(class_label): float(weight)
            for class_label, weight in self.class_weight.items()
        }
        if any(weight < 0 for weight in resolved_class_weight.values()):
            raise ValueError("`class_weight` values must be non-negative.")

        has_positive_weight = any(
            resolved_class_weight.get(normalize_class_label(class_label), 1.0) > 0
            for class_label in classes
        )
        if not has_positive_weight:
            raise ValueError("At least one observed class must have positive weight.")

        return resolved_class_weight

    def _grow_tree(self, X, y, depth=0, history=None, **kwargs):
        """Handle grow tree."""
        if history is None:
            history = []

        # Create root node
        root = Node(
            feature=None,
            history=history,
            left=None,
            right=None,
            prediction=self._get_prediction(y),
        )
        # Root starts as a leaf; each successful split increases this count by 1.
        self.n_leaves = 1
        self.budget_stopped = False
        self.budget_stopped_reason = None
        should_save_locally = self.save_locally and bool(self.folder_path)
        if should_save_locally:
            os.makedirs(self.folder_path, exist_ok=True)

        # Queue of nodes to process: (node, X subset, y subset, depth, history)
        queue = [(root, X, y, depth, history)]

        while queue:
            # save it locally
            if should_save_locally:
                # Save the root node
                compressed_root = compress_node(root)
                with open(f"{self.folder_path}/intermediate_tree.pkl", "wb") as f:
                    dill.dump(compressed_root, f)

            current_node, node_X, node_y, node_depth, node_history = queue.pop(0)

            # Check stopping criteria. Once the run budget has stopped growth, every node still
            # queued becomes a leaf, so `fit` returns a complete but shallower tree rather than
            # discarding work already paid for.
            if (
                self.budget_stopped
                or node_depth >= self.max_depth
                or self._should_stop(node_y)
            ):
                self._make_leaf(current_node, node_y)
                continue

            # Find optimal split
            self._provenance_start_node(node_depth, len(node_y))
            self._budget_start_node(node_depth)
            try:
                optimal_feature = self.feature_finder.get_optimal_feature(
                    node_X,
                    node_y,
                    history=node_history,
                    min_samples_leaf=self.min_samples_leaf,
                    splitting_criterion=self.splitting_criterion,
                    class_weight=self.class_weight_,
                    **kwargs,
                )
            except BudgetExceeded as exceeded:
                logger.warning("%s", exceeded)
                self.budget_stopped = True
                self.budget_stopped_reason = exceeded.reason
                self._budget_finish_node()
                self._provenance_finish_node(split_feature=None, budget_stopped=True)
                self._make_leaf(current_node, node_y)
                continue

            self._budget_finish_node()

            if self.store_node_artifacts:
                current_node.artifacts = self.feature_finder.logging_artifacts
            else:
                current_node.artifacts = None

            # If no valid split found, make current node a leaf
            if not optimal_feature:
                self._provenance_finish_node(split_feature=None)
                self._make_leaf(current_node, node_y)
                continue

            current_node.feature = optimal_feature
            # Guarded because this is the last place a feature function runs, and it used to be
            # the only step with nothing around it: a candidate that raises here took the whole
            # run down, after every token for the node had been spent. `filter_features` and
            # `optimize_threshold` both drop a misbehaving feature; this one should cost a node,
            # not the run.
            try:
                feature_values = evaluate_feature_function(optimal_feature.fn, node_X)
                feature_values = feature_output_to_series(feature_values, node_X.index)
            except Exception:
                logger.exception(
                    "The chosen feature `%s` failed when applied to this node's data, so the "
                    "node cannot be split on it and becomes a leaf.",
                    optimal_feature.name,
                )
                self._provenance_finish_node(split_feature=None)
                self._make_leaf(current_node, node_y)
                continue

            # left = samples at or below the threshold, right = everything above it
            left_mask = feature_values <= optimal_feature.threshold
            right_mask = ~left_mask

            # Validate split
            if not self._is_split_valid(node_y[left_mask], node_y[right_mask]):
                self._provenance_finish_node(split_feature=None)
                self._make_leaf(current_node, node_y)
                continue

            self._provenance_finish_node(split_feature=optimal_feature)

            # Create child nodes with updated history
            updated_history_left = [*node_history, (optimal_feature, "smaller")]
            updated_history_right = [*node_history, (optimal_feature, "greater")]

            current_node.left = Node(
                feature=None,
                history=updated_history_left,
                left=None,
                right=None,
                prediction=self._get_prediction(node_y[left_mask]),
            )

            current_node.right = Node(
                feature=None,
                history=updated_history_right,
                left=None,
                right=None,
                prediction=self._get_prediction(node_y[right_mask]),
            )
            # One leaf replaced by two leaves => net +1 leaf.
            self.n_leaves += 1

            # Add child nodes to queue
            queue.append(
                (
                    current_node.left,
                    node_X[left_mask],
                    node_y[left_mask],
                    node_depth + 1,
                    updated_history_left,
                )
            )

            queue.append(
                (
                    current_node.right,
                    node_X[right_mask],
                    node_y[right_mask],
                    node_depth + 1,
                    updated_history_right,
                )
            )

        return root

    def _is_split_valid(self, left_y, right_y):
        """Check whether split valid."""
        return not (
            len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf
        )

    def _get_prediction(self, y):
        """Return a node's prediction: the weighted mean of its targets.

        For a binary target that is a score in ``[0, 1]``, the positive proportion. Without
        weighting it is the raw positive rate ``mean(y)``. When a class weighting is active
        (``self.class_weight_`` resolved in :meth:`fit`), it is instead the class-weighted
        positive proportion ``sum_i w(y_i) * y_i / sum_i w(y_i)`` -- so a leaf predicts the
        positive class iff ``w_pos * n_pos > w_neg * n_neg``, matching sklearn's weighted-vote rule
        under the 0.5 threshold in :meth:`predict`. This lets ``class_weight`` shift the actual
        predictions toward the minority class, not only the choice of splits. For a continuous
        target it is the mean of the values, weighted by the rows' ``sample_weight`` when there
        are any.
        """
        if self.class_weight_ is None:
            return np.mean(y)

        # `ravel`, because the labels do not always arrive as a flat array. The generic-CSV plugin
        # hands `fit` a single-column *DataFrame*, so `asarray` gives shape (n, 1) and iterating it
        # yields one-element arrays, which are unhashable and cannot index the weight map. The
        # unweighted path never noticed: `np.mean` does not care about the extra axis.
        y_arr = np.asarray(y).ravel()
        if isinstance(self.class_weight_, pd.Series):
            index = getattr(y, "index", None)
            if index is None:
                raise TypeError("Per-row `sample_weight` needs labels that carry an index.")
            weights = self.class_weight_.reindex(index).to_numpy(dtype=float)
        else:
            weights = np.array(
                [self.class_weight_.get(normalize_class_label(label), 1.0) for label in y_arr],
                dtype=float,
            )
        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            return np.mean(y)
        return float(np.sum(weights * y_arr) / total_weight)

    def _should_stop(self, y):
        """Check whether stop."""
        n_samples = len(y)

        if n_samples < self.min_samples_split:
            return True

        # A node with fewer than two leaves' worth of samples cannot be split at all: every
        # candidate cut is rejected by `_is_split_valid`, so the node becomes a leaf whatever
        # the feature finder returns. Stopping here rather than there is the same tree, reached
        # without generating and scoring a pool of features that cannot win. This matters
        # because the generation is paid for in API calls -- a depth-3 node of 48 samples under
        # `min_samples_leaf = 32` spent 11 semantic prompts and 110 code prompts to produce
        # nothing, and then tripped the integrity check for having no usable feature, whose
        # message points at credentials rather than at arithmetic. sklearn applies the same
        # criterion for the same reason.
        if n_samples < 2 * self.min_samples_leaf:
            return True

        if self.max_leaf_nodes and self.n_leaves >= self.max_leaf_nodes:
            return True

        # A pure node cannot be improved by any split.
        return len(np.unique(y)) == 1

    def predict(self, X, return_proba=False):
        """Predict values: for a binary target the class, or with `return_proba` the positive
        proportion of the leaf; for a continuous target the leaf's mean, with nothing to threshold
        whatever `return_proba` says."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AdaptiveDecisionTree.predict expects X to be a pandas DataFrame. "
                f"Received {type(X).__name__}."
            )
        return predict_at_depth_vectorized(
            root=self.root,
            X=X,
            depth=self.max_depth,
            return_proba=return_proba or self.target_kind == CONTINUOUS,
        )

    def format_tree(self, node=None, indent="", prefix="") -> str:
        """The tree as text: every split's feature, threshold, description, code and rationale.

        Split out of `pretty_print` so the fitted tree can be *logged*. It used to be printed
        to stdout only, so a run whose terminal output was not captured left no readable record
        of its own result -- the `.dill` was the only copy, and reading that needs this
        environment. The pilot runs sweeps detached, so that record has to be in the log file.

        The layout is unchanged; `pretty_print` still prints exactly what it always did.
        """
        lines = []
        if node is None:
            node = self.root
            lines.append("Decision Tree Structure:")
            if node is None:
                lines.append("(the tree has not been fitted)")
                return "\n".join(lines)

        if node._is_leaf():
            lines.append(f"{indent}└── Prediction: {node.prediction:.4f}")
            return "\n".join(lines)

        feature = node.feature
        lines.append(f"{indent}{prefix}Feature: {feature.name}")
        lines.append(f"{indent}    Threshold: {feature.threshold:.4f}")
        if feature.description:
            lines.append(f"{indent}    Description: {feature.description}")
        if feature.string:
            lines.append(f"{indent}    String: {feature.string}")
        if feature.rationale:
            lines.append(f"{indent}    Rationale: {feature.rationale}")

        if node.left:
            lines.append(self.format_tree(node.left, indent + "    ", "≤ :"))
        if node.right:
            lines.append(self.format_tree(node.right, indent + "    ", "> :"))

        return "\n".join(lines)

    def pretty_print(self, node=None, indent="", prefix=""):
        """Print the tree. See `format_tree`, which is what a run records."""
        print(self.format_tree(node=node, indent=indent, prefix=prefix))
