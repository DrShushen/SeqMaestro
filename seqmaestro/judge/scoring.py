"""Running the panel, and summarising its scores.

The summary answers two questions, the second only when a control was judged:

**What did each arm score?** A mean over its features, each feature itself a mean over repeats.

**What did the noise score?** When features from permuted-label runs are on the panel, they are
blinded identically and `control_gap` reports how far each arm sits above their mean. Judging them
is off by default, for the reasons in the docstring of `seqmaestro.pipeline.analyses.judge`. Without
them the summary omits `control_gap`, so an arm's mean describes its features and is compared with
the other arms.

Unanswered items are counted, never imputed. A refusal folded into the mean would pull every arm
towards the middle and flatter whichever arm is hardest to parse.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from seqmaestro.judge.engine import Judgement, ScoringEngine
from seqmaestro.judge.panel import PERMUTED, PanelEntry
from seqmaestro.judge.rubric import TaskContext, render

DEFAULT_REPEATS = 3


@dataclass
class ItemScore:
    """One feature's judgements. `scores` excludes refusals; `n_failed` counts them."""

    item_id: str
    arm: str
    labels: str
    feature: str
    scores: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    n_failed: int = 0

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.scores) if self.scores else None

    @property
    def spread(self) -> float | None:
        """Standard deviation across repeats: how reproducible this judgement was."""
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0 if self.scores else None


def score_panel(
    panel: list[PanelEntry],
    engine: ScoringEngine,
    context: TaskContext,
    repeats: int = DEFAULT_REPEATS,
    include_code: bool = False,
    include_direction: bool = False,
    progress=None,
) -> list[ItemScore]:
    """Judge every entry `repeats` times. Independent calls, so the spread is a real measurement.

    `context` is required: a feature described in coordinates the judge cannot interpret is not
    being judged on its biology.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1; got {repeats}.")

    results = []
    for index, entry in enumerate(panel, start=1):
        rendered = render(entry, context, include_code=include_code,
                          include_direction=include_direction)
        item = ItemScore(
            item_id=entry.item_id, arm=entry.arm, labels=entry.labels, feature=entry.record.name
        )
        for _ in range(repeats):
            judgement: Judgement = engine.judge(rendered)
            if judgement.score is None:
                item.n_failed += 1
            else:
                item.scores.append(judgement.score)
                if judgement.reason:
                    item.reasons.append(judgement.reason)
        results.append(item)
        if progress:
            progress(index, len(panel), item)
    return results


def summarise(items: list[ItemScore], context: TaskContext | None = None) -> dict:
    """Each arm's mean under each label condition, plus the noise floor and the gap to it when a
    control was judged."""
    control = [item.mean for item in items if item.labels == PERMUTED and item.mean is not None]
    control_mean = statistics.fmean(control) if control else None

    arms: dict[str, dict] = {}
    for arm in sorted({item.arm for item in items}):
        for labels in sorted({item.labels for item in items if item.arm == arm}):
            means = [
                item.mean
                for item in items
                if item.arm == arm and item.labels == labels and item.mean is not None
            ]
            spreads = [
                item.spread
                for item in items
                if item.arm == arm and item.labels == labels and item.spread is not None
            ]
            failed = sum(
                item.n_failed for item in items if item.arm == arm and item.labels == labels
            )
            mean = statistics.fmean(means) if means else None
            entry = {
                "n_features": len(means),
                "mean": round(mean, 3) if mean is not None else None,
                "median": round(statistics.median(means), 3) if means else None,
                "mean_spread_across_repeats": round(statistics.fmean(spreads), 3) if spreads else None,
                "n_unanswered": failed,
            }
            if control_mean is not None:
                # How far above the noise floor the arm sits. The key is absent when no control was
                # judged, because a column of nulls in every table would look like a measurement
                # that came back empty.
                entry["control_gap"] = round(mean - control_mean, 3) if mean is not None else None
            arms[f"{arm}/{labels}"] = entry

    control_block: dict = {
        "n_features": len(control),
        "mean": round(control_mean, 3) if control_mean is not None else None,
    }
    if control:
        control_block["why"] = (
            "Features from permuted-label runs, blinded identically. If these score as highly "
            "as the real ones, the judge is scoring how a feature is written rather than what "
            "produced it, and no arm mean above is interpretable."
        )

    return {
        "question": "significance_if_true",
        "task_context": context.as_dict() if context else None,
        "n_items": len(items),
        "n_unanswered": sum(item.n_failed for item in items),
        "control": control_block,
        "arms": arms,
    }
