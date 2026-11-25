"""Assembling the panel of features the judge rates.

The panel holds the top *N* features by importance of each source (one cell of an arm), each under
an identifier that reveals nothing, with its arm, seed and label condition kept on the entry beside
the feature and out of every prompt (`seqmaestro.judge.rubric`). The entries are shuffled with a
seed before the identifiers are assigned, so that an identifier reveals nothing either.
`docs/developer/analyses/judge.md` gives the reasons for each step.

Permuted-label features, when they are judged, need no separate mechanism. They enter the panel as
ordinary entries, which the judge cannot tell from the rest, and are separated again only when the
scores are summarised.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness import cell as cells
from seqmaestro.harness.features import DEFAULT_TOP_N, top_features

#: The label condition of a noise control. Real-label features and their controls must be
#: indistinguishable in the panel, so the condition is recorded on each entry and used only when
#: the scores are summarised.
PERMUTED = cells.PERMUTED


@dataclass(frozen=True)
class PanelSource:
    """One cell's contribution: where the features came from, and what they are."""

    arm: str
    labels: str
    records: list[FeatureRecord]
    #: The kind of target the arm was fitted against (`seqmaestro.target`), which decides the words
    #: a feature's direction is shown in.
    target_kind: str
    seed: int | None = None


@dataclass(frozen=True)
class PanelEntry:
    """One feature as the judge will see it, plus the provenance the judge will not see."""

    item_id: str
    record: FeatureRecord
    arm: str
    labels: str
    target_kind: str
    seed: int | None = None


def build_panel(
    sources: list[PanelSource],
    top_n: int | None = DEFAULT_TOP_N,
    seed: int = 0,
) -> list[PanelEntry]:
    """Select, hide the arms and shuffle. `top_n=None` judges everything, which is rarely affordable."""
    entries = []
    for source in sources:
        chosen = source.records if top_n is None else top_features(source.records, top_n)
        for record in chosen:
            entries.append((source, record))

    random.Random(seed).shuffle(entries)

    # Identifiers are assigned *after* shuffling, so the id itself leaks nothing about the arm.
    # Sequential ids assigned before shuffling would group each arm into a contiguous block.
    return [
        PanelEntry(
            item_id=f"F{index:04d}",
            record=record,
            arm=source.arm,
            labels=source.labels,
            seed=source.seed,
            target_kind=source.target_kind,
        )
        for index, (source, record) in enumerate(entries)
    ]


def panel_key(panel: list[PanelEntry]) -> dict:
    """What each identifier stands for: the arm, the label condition and the seed the feature
    came from, its name and its importance. It is written beside the scores.

    The judge is shown each item under its id and none of this. The scores file names each item's
    arm and label condition as well, since the summary is taken by arm, so the key adds the seed
    and the importance, and holds every item whether or not it was answered.
    """
    return {
        entry.item_id: {
            "arm": entry.arm,
            "labels": entry.labels,
            "seed": entry.seed,
            "feature": entry.record.name,
            "importance": entry.record.importance,
        }
        for entry in panel
    }
