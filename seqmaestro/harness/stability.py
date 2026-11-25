"""Matching the features of several fits, and how far an arm's selections agree across seeds.

**Features are matched on what they compute rather than on what they are called**, because an arm
that writes its features names one property differently at each seed. Every feature is evaluated on
a common reference matrix, and two features from different fits are the same when their values
agree. A `FeatureRun` holds one such fit: a cell's feature records with their values.

Agreement is measured by the mean pairwise Jaccard index, from which the reported figure is built
(`chance_corrected_agreement`), and by the Nogueira estimator, a diagnostic only, because it assumes
a universe of features of known size. `docs/developer/analyses/index.md` describes the matching,
and `docs/developer/analyses/stability_and_co_discovery.md` what is computed from it and how to
interpret it.

**The code here does not know which arm it compares.** Its input is what a cell wrote (feature
records and their values on the reference split), so a new arm takes part by writing those two
files.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.cell import FEATURE_VALUES_FILE, FEATURES_FILE
from seqmaestro.harness.features import (
    DEFAULT_TOP_N,
    definition_digest,
    load_feature_values,
    load_features,
    top_features,
)


@dataclass
class FeatureRun:
    """One fit's discovered features, and what each computes on the reference split."""

    label: str
    records: list[FeatureRecord]
    values: np.ndarray  # (n_reference_rows, n_features), column i belongs to records[i]
    header: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.records):
            raise ValueError(
                f"{self.label}: {self.values.shape} values for {len(self.records)} records."
            )

    @property
    def names(self) -> list[str]:
        return [record.name for record in self.records]

    def top(self, n: int) -> FeatureRun:
        """This fit restricted to its `n` most important features, with its values restricted to
        match.

        This is the view that is comparable between arms. Without it, a model that fits a shape
        function to every input is trivially "perfectly stable", because it selects everything
        every time, whereas a shallow tree that picks a handful out of thousands scores near zero
        unless two fits agree almost exactly. Neither number measures agreement.
        """
        kept = top_features(self.records, n)
        positions = [self.records.index(record) for record in kept]
        return FeatureRun(self.label, kept, self.values[:, positions], self.header)


def load_run(directory: str | Path, label: str | None = None) -> FeatureRun:
    """Read a cell written by any arm: its `features.json` and `feature_values.npz`."""
    directory = Path(directory)
    records, header = load_features(directory / FEATURES_FILE)
    names, values = load_feature_values(directory / FEATURE_VALUES_FILE)

    if names != [record.name for record in records]:
        raise ValueError(
            f"{directory}: the value matrix columns do not match the records, in name or in order. "
            f"They are paired by position, so a mismatch would compare the wrong features."
        )
    return FeatureRun(label or directory.name, records, values, header)


# ------------------------------------------------------------------------------------ matchers


class Matcher(Protocol):
    """Produces a similarity in [0, 1] for every pair of features across two fits.

    `match_pair` and `stability` need three things from a matcher: `threshold` decides what counts
    as a match, `similarity` scores every pair, and `describe` names the choice in the output, so
    that a number is judged against the notion of sameness that produced it. Any object providing
    these can be passed in.
    """

    threshold: float

    def similarity(self, run_a: FeatureRun, run_b: FeatureRun) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


class BehaviouralMatcher:
    """Absolute Spearman rank correlation between two features' values on the reference split.

    **The correlation is Spearman's rather than Pearson's**, because a feature and a monotone
    transform of it (a GC count and a GC fraction over the same window) are the same property and
    should match perfectly, which only a rank correlation ensures.

    **The correlation is absolute**, because a feature and its negation hold the same information,
    so the sign a model happened to learn does not change what was discovered.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold}.")
        self.threshold = threshold

    def similarity(self, run_a: FeatureRun, run_b: FeatureRun) -> np.ndarray:
        if run_a.values.shape[0] != run_b.values.shape[0]:
            raise ValueError(
                f"{run_a.label} was evaluated on {run_a.values.shape[0]} reference rows and "
                f"{run_b.label} on {run_b.values.shape[0]}. Matching is meaningless unless every "
                f"run is evaluated on the same rows in the same order."
            )
        return np.abs(_rank_normalise(run_a.values).T @ _rank_normalise(run_b.values))

    def describe(self) -> dict[str, Any]:
        return {"matcher": "behavioural", "statistic": "abs_spearman", "threshold": self.threshold}


# ------------------------------------------------------------------------------------ matching


def match_pair(
    run_a: FeatureRun, run_b: FeatureRun, matcher: Matcher, exact_first: bool = True
) -> list[tuple[int, int]]:
    """A one-to-one correspondence between two fits' features, maximising total similarity.

    It uses Hungarian assignment rather than greedy nearest-neighbour matching, because greedy
    matching lets one strong match take a feature that was the only plausible partner for another,
    and the result then depends on the order in which the features appear.

    `exact_first` pairs features with identical definitions before the matcher runs, because two
    features that compute the same thing are the same feature whether or not a statistic can show
    it. This matters for a feature that is **constant on the reference split**: it has no rank
    variation, so behavioural matching scores it 0 against everything, including itself in another
    fit. Without exact matching it becomes a new singleton cluster in every fit, and a fixed
    vocabulary can then appear to produce *more* "distinct" features across its seeds than the
    vocabulary contains. Matching exact definitions first loses nothing, because an identical
    definition would score the maximum similarity anyway.
    """
    matched: list[tuple[int, int]] = []
    taken_a: set[int] = set()
    taken_b: set[int] = set()

    if exact_first:
        by_digest: dict[str, list[int]] = {}
        for index, record in enumerate(run_b.records):
            digest = _identifying_digest(record)
            if digest is not None:
                by_digest.setdefault(digest, []).append(index)

        for index, record in enumerate(run_a.records):
            digest = _identifying_digest(record)
            candidates = by_digest.get(digest) if digest is not None else None
            if candidates:
                matched.append((index, candidates.pop()))
                taken_a.add(index)
                taken_b.add(matched[-1][1])

    rest_a = [index for index in range(len(run_a.records)) if index not in taken_a]
    rest_b = [index for index in range(len(run_b.records)) if index not in taken_b]
    if not rest_a or not rest_b:
        return sorted(matched)

    similarity = matcher.similarity(run_a, run_b)[np.ix_(rest_a, rest_b)]
    rows, columns = linear_sum_assignment(similarity, maximize=True)
    matched.extend(
        (rest_a[i], rest_b[j])
        for i, j in zip(rows, columns, strict=True)
        if similarity[i, j] >= matcher.threshold
    )
    return sorted(matched)


def jaccard(run_a: FeatureRun, run_b: FeatureRun, matcher: Matcher) -> float:
    """The number of matched features over the size of the union: 1.0 when both fits found the same
    features, and 0.0 when they share none.

    Two fits that both found nothing score 0.0, because they have no feature to agree on. Counting
    them as agreeing would raise an arm's chance level whenever its fits on shuffled labels select
    nothing (a control behaving as it should), and the chance-corrected figure would then fall far
    below zero for that arm.
    """
    matched = len(match_pair(run_a, run_b, matcher))
    union = len(run_a.records) + len(run_b.records) - matched
    return matched / union if union else 0.0


def consensus_labels(runs: list[FeatureRun], matcher: Matcher) -> list[list[int]]:
    """Assign every feature in every fit to a shared cluster, so fits can be compared as sets.

    The clusters are the connected components of the graph that joins features matched pairwise.
    Unlike a consensus grown fit by fit, this does not depend on the order of the fits. It is
    transitive, though, so a chain of borderline matches can merge two different properties, and
    the `diagnostics` block of `stability` counts the symptom: a cluster that holds two features
    from one fit.
    """
    offsets, total = [], 0
    for run in runs:
        offsets.append(total)
        total += len(run.records)

    parent = list(range(total))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (index_a, run_a), (index_b, run_b) in itertools.combinations(enumerate(runs), 2):
        for i, j in match_pair(run_a, run_b, matcher):
            root_a, root_b = find(offsets[index_a] + i), find(offsets[index_b] + j)
            if root_a != root_b:
                parent[root_a] = root_b

    cluster_of: dict[int, int] = {}
    labels = []
    for index, run in enumerate(runs):
        run_labels = []
        for i in range(len(run.records)):
            root = find(offsets[index] + i)
            run_labels.append(cluster_of.setdefault(root, len(cluster_of)))
        labels.append(run_labels)
    return labels


# ----------------------------------------------------------------------------------- estimators


def nogueira_phi(selection: np.ndarray, universe_size: int | None = None) -> float:
    """The stability estimator of Nogueira, Sechidis and Brown on a binary matrix of fits by
    features.

    `selection[m, f]` is 1 when fit `m` selected feature `f`. The estimator is corrected for chance:
    it is 1 when every fit selects the same set, and about 0 when fits select independently at the
    same rate.

    `universe_size` defaults to the number of columns given, which is the union of everything
    discovered. Any larger universe is a **convention**, and Φ̂ is not comparable between two
    conventions.
    """
    selection = np.asarray(selection, dtype=float)
    if selection.ndim != 2:
        raise ValueError(f"Expected a runs-by-features matrix; got shape {selection.shape}.")

    n_runs, n_observed = selection.shape
    if n_runs < 2:
        raise ValueError("Stability needs at least two runs.")

    universe = n_observed if universe_size is None else universe_size
    if universe < n_observed:
        raise ValueError(f"universe_size {universe} is smaller than the {n_observed} features observed.")
    if universe == 0:
        # The fits selected nothing, so there is no selection whose stability could be measured.
        return float("nan")

    # Features in the universe but never selected contribute zero variance, so only the observed
    # columns need summing. The unselected ones still count in the denominator, through `universe`.
    frequency = selection.mean(axis=0)
    unbiased_variance = (n_runs / (n_runs - 1)) * frequency * (1 - frequency)

    mean_selected = selection.sum(axis=1).mean()
    expected = (mean_selected / universe) * (1 - mean_selected / universe)
    if expected == 0:
        # Every fit selected everything, or nothing, which is perfectly but vacuously stable.
        return 1.0
    return float(1 - (unbiased_variance.sum() / universe) / expected)


def chance_corrected_agreement(observed: float, chance: float) -> float:
    """How far an arm's agreement across seeds sits above its chance level, scaled so that 0 is
    chance and 1 is identical sets.

        kappa = (observed - chance) / (1 - chance)

    **This is the stability figure to report, because it is comparable across all arms.**
    `observed` is the mean pairwise Jaccard over the arm's `real` cells, whereas `chance` is the same
    quantity over its `permuted` cells: the agreement this procedure produces when there is nothing
    to find. 0 means an arm reproduces its features no better than it does on shuffled labels, and 1
    means the seeds returned identical sets.

    The chance level is measured per arm instead of taken from a formula, because it differs widely
    between arms. A fixed vocabulary has a chance Jaccard near zero, whereas the proposer draws from
    a concentrated prior, so it proposes similar features whatever the labels say and agrees with
    itself substantially on shuffled labels. Its raw agreement can therefore be high while its
    agreement above chance is low, and any figure that ignores the chance level flatters generative
    arms.

    `nogueira_phi` is not used here, because Φ̂ needs a universe size `d`, which a generative arm
    does not have. Falling back to the union of what was discovered makes Φ̂ incomparable between
    arms and drives it negative: disjoint selections by three fits score -0.5 under that
    convention, whereas the same sets score 0.0 in the limit of large `d`. Kappa needs no universe,
    so it means the same thing for every arm.

    Use it on the **top-N view**, so that arms which select everything and arms which name a
    handful contribute the same number of features.

    Kappa is not bounded below: an arm that agrees *less* on real labels than on shuffled ones
    scores negative. It is NaN when `chance` is 1.0, where the scale has no room left.
    """
    if not np.isfinite(observed) or not np.isfinite(chance):
        return float("nan")
    if chance >= 1.0:
        return float("nan")
    return float((observed - chance) / (1.0 - chance))


def selection_matrix(labels: list[list[int]]) -> np.ndarray:
    """The binary matrix of fits by consensus features, from `consensus_labels`."""
    n_clusters = max((label for run in labels for label in run), default=-1) + 1
    matrix = np.zeros((len(labels), n_clusters), dtype=int)
    for run_index, run_labels in enumerate(labels):
        for label in run_labels:
            matrix[run_index, label] = 1
    return matrix


def stability(
    runs: list[FeatureRun],
    matcher: Matcher | None = None,
    universe_size: int | None = None,
    top_n: int | None = None,
) -> dict:
    """Both measures, with the diagnostics needed to decide whether to believe them.

    `universe_size` is the Nogueira convention. Leave it as `None` to take the universe as what was
    discovered, or pass the size of the pool's vocabulary for arms that draw from a fixed one.
    Report which was used.

    `top_n` first restricts each fit to its most important features, which is the view to compare
    *across* arms. The unrestricted view says something real about arms that do select, but nothing
    about arms that keep every feature. `stability_views` returns both.
    """
    if len(runs) < 2:
        raise ValueError("Stability needs at least two runs.")

    matcher = matcher or BehaviouralMatcher()
    if top_n is not None:
        runs = [run.top(top_n) for run in runs]
    rows = {run.values.shape[0] for run in runs}
    if len(rows) > 1:
        raise ValueError(f"Runs were evaluated on different numbers of reference rows: {sorted(rows)}.")

    pairwise = [
        {"runs": [a.label, b.label], "jaccard": jaccard(a, b, matcher)}
        for a, b in itertools.combinations(runs, 2)
    ]
    labels = consensus_labels(runs, matcher)
    matrix = selection_matrix(labels)

    return {
        "n_runs": len(runs),
        "top_n": top_n,
        "n_features_per_run": {run.label: len(run.records) for run in runs},
        "matcher": matcher.describe(),
        "jaccard": {
            "mean_pairwise": float(np.mean([entry["jaccard"] for entry in pairwise])),
            "pairwise": pairwise,
        },
        "nogueira": {
            "phi": nogueira_phi(matrix, universe_size),
            "universe_size": universe_size if universe_size is not None else int(matrix.shape[1]),
            "convention": "supplied" if universe_size is not None else "union_of_discovered",
        },
        "consensus": {
            "n_clusters": int(matrix.shape[1]),
            "n_found_by_every_run": int((matrix.sum(axis=0) == len(runs)).sum()),
            "n_found_by_one_run": int((matrix.sum(axis=0) == 1).sum()),
        },
        "diagnostics": _diagnostics(runs, labels),
    }


def stability_views(
    runs: list[FeatureRun],
    matcher: Matcher | None = None,
    universe_size: int | None = None,
    top_n: int | None = DEFAULT_TOP_N,
) -> dict:
    """The two views a result should report, computed together.

    `top_n` is the view to quote when comparing arms, because every arm contributes at most that
    many features, however many it nominally used. `full` is the view to quote about a single arm:
    "it kept a small fraction of the vocabulary and a third of those recurred" is a finding, whereas
    the same sentence about a model that keeps everything is not.
    """
    views = {"full": stability(runs, matcher, universe_size)}
    if top_n is not None:
        views["top_n"] = stability(runs, matcher, universe_size, top_n=top_n)
    return views


def _diagnostics(runs: list[FeatureRun], labels: list[list[int]]) -> dict:
    """Two conditions that would quietly invalidate the numbers above."""
    # A cluster holding two features from one fit means transitive chaining merged them. Either the
    # matcher's threshold is too loose or the features really are near-duplicates.
    chained = sum(len(run_labels) != len(set(run_labels)) for run_labels in labels)

    # A feature that never varies on the reference split cannot be matched to anything by its
    # values, and is usually a sign that the split is too small or the feature is degenerate.
    constant = {
        run.label: int((run.values.std(axis=0) == 0).sum())
        for run in runs
        if (run.values.std(axis=0) == 0).any()
    }
    return {"runs_with_chained_clusters": chained, "constant_features": constant}


def _identifying_digest(record: FeatureRecord) -> str | None:
    """A definition digest, returned only when the record has a definition.

    `definition_digest` always returns a hash, falling back to the kind and parameters when there
    is no code. For a *k*-mer that is a real definition, whereas for a record with neither generated
    code nor parameters (which a hand-written file may contain) the fallback hashes nothing
    distinguishing, and every such record in a fit would collide. Returning `None` sends those
    records to behavioural matching, which is the only way to tell them apart.
    """
    return definition_digest(record) if (record.source or record.params) else None


def non_finite_columns(values: np.ndarray) -> int:
    """How many columns hold at least one non-finite value, so that the imputation by
    `impute_non_finite` can be reported."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return 0
    return int((~np.isfinite(values)).any(axis=0).sum())


def impute_non_finite(values: np.ndarray) -> np.ndarray:
    """The values with each non-finite entry replaced by its column's median over the finite ones.

    A NaN cannot be ranked, so the rank statistics that compare or test feature columns impute
    first: the matching here, and the reliability panel's clusters and Tier 2. A column with no
    finite entry has nothing to impute from and becomes zeros, which the zero-norm guard in
    `_rank_normalise` then scores 0 against everything, as it does a constant column and for the
    same reason.
    """
    filled = np.array(values, dtype=float)
    bad = ~np.isfinite(filled)
    for column in np.where(bad.any(axis=0))[0]:
        finite = filled[~bad[:, column], column]
        filled[bad[:, column], column] = float(np.median(finite)) if finite.size else 0.0
    return filled


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    """Columns rank-transformed and scaled so that a dot product is a Spearman correlation.

    A constant column has no rank variation, so its correlation is undefined. It becomes zeros,
    which score a similarity of 0 against everything, so it never matches by accident.

    **Non-finite entries are imputed to their column's median before ranking.** A generated
    feature can be undefined on some rows without being wrong: `CpG_cluster_span` is NaN for a
    sequence containing no CpG cluster. `rankdata` would propagate a NaN across the whole column,
    and that NaN would pass the zero-norm guard below, because `nan != 0` is True. It would then
    reach `linear_sum_assignment`, which refuses a matrix with invalid entries, and the whole
    stability analysis would fail for every arm, not just the one holding the feature.

    The median is the convention `GeneratedFeaturiser` uses when a feature raises an error at
    transform time, so the two agree. Count the affected columns with `non_finite_columns` and
    report the count beside any figure computed here, because being undefined on a large share of
    rows is a real property of a feature, which imputation must not hide.
    """
    ranks = rankdata(impute_non_finite(values), axis=0)
    centred = ranks - ranks.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    return np.divide(centred, norms, out=np.zeros_like(centred), where=norms != 0)
