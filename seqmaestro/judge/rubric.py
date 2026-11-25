"""The question the judge is asked, and how a feature is shown to it.

`docs/developer/analyses/judge.md` is the account of the whole protocol and of the reasons for it.
**A change to the wording here changes what every past score means**, so change that page in the
same commit.

The code here does not know which arm a feature came from. The arm stays hidden by construction,
because the renderer is given a `PanelEntry` and uses only its fields that reveal nothing, and
`tests/test_judge.py` checks that a prompt never names an arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from seqmaestro.harness.direction import (
    DECREASES_TARGET,
    HIGHER_IN_NEGATIVE,
    HIGHER_IN_POSITIVE,
    INCREASES_TARGET,
    from_sign,
)
from seqmaestro.llm import TaskContext
from seqmaestro.target import target_for

QUESTION = (
    "How biologically significant or interesting would this feature be, if it were true?"
)

#: The scale is anchored, because an unanchored scale of 1 to 5 drifts between calls and cannot be
#: compared across arms.
SCALE = {
    1: "Trivial or uninformative. Tells a biologist nothing they would act on.",
    2: "Marginal. A weak or very generic observation.",
    3: "Moderately interesting. Plausible and worth noting, but not surprising.",
    4: "Clearly interesting. Would prompt a biologist to look further.",
    5: "Highly significant. Would change how someone thinks about the mechanism.",
}

MIN_SCORE, MAX_SCORE = min(SCALE), max(SCALE)

SYSTEM_PROMPT = (
    "You are an expert molecular biologist assessing candidate sequence features proposed by "
    "an automated discovery system. You judge how biologically meaningful each feature would be "
    "if it held, not whether it in fact holds -- you cannot verify that from a description, and "
    "guessing would be worse than abstaining. Be discriminating: most candidates are ordinary, "
    "and a scale on which everything scores 4 measures nothing."
)


# `TaskContext` is defined in `seqmaestro.llm`, because the proposer needs it too, for the same
# reason as the judge. It is re-exported here so that a reader of the rubric finds it where they
# expect it.
__all__ = ["DIRECTION_LINE", "MAX_SCORE", "MIN_SCORE", "QUESTION", "SCALE", "SYSTEM_PROMPT",
           "Rendered", "TaskContext", "direction_shown", "render"]


@dataclass(frozen=True)
class Rendered:
    """A prompt pair, plus the item it belongs to, so a reply can be matched back."""

    item_id: str
    system: str
    user: str


#: Which class the feature runs with, or which way it moves a continuous target, as the judge is
#: shown it. The classes and the target are named by the dataset's description, which the judge is
#: given in the same prompt, so the wording here does not depend on the dataset.
DIRECTION_LINE = {
    HIGHER_IN_POSITIVE: "Observed in the data: this feature takes higher values in the positive class.",
    HIGHER_IN_NEGATIVE: "Observed in the data: this feature takes higher values in the negative class.",
    INCREASES_TARGET: "Observed in the data: higher values of this feature go with a higher target.",
    DECREASES_TARGET: "Observed in the data: higher values of this feature go with a lower target.",
}


def direction_shown(record, target_kind: str) -> str | None:
    """The sentence the judge would be shown for this feature, or None if it would be shown none.

    It is separate from `render`, so that a caller can count how many features of a panel have a
    direction without rendering the prompts. Asking for directions and getting none would otherwise
    go unnoticed: the setting would be recorded as on, and every prompt would go out without a
    direction.
    """
    return DIRECTION_LINE.get(from_sign(record.marginal_effect, target_for(target_kind).directions))


def render(entry, context: TaskContext, include_code: bool = False,
           include_direction: bool = False) -> Rendered:
    """Build the prompt for one feature of the panel.

    `include_code` shows the feature's implementation, which may help the judge with an ambiguous
    description or bias it towards whatever looks sophisticated, so it is off by default.
    `include_direction` adds a line saying which class the feature runs with, or which way a continuous
    target moves with it: the sign of its marginal effect only, and no line where it has none. A run
    states it once (`grid.judge_direction`), since it changes what a score means.
    `docs/developer/analyses/judge.md` gives the reasons for both.

    The judge is told the task description with what it withholds from the models that write features
    (`TaskContext.render_for_judge`), because rating what a feature would mean needs the phenotype
    named, whereas it is never told which arm produced the feature.
    """
    lines = [
        f"Feature name: {entry.record.name}",
        f"Description: {entry.record.description or '(none given)'}",
    ]
    if entry.record.rationale:
        lines.append(f"Stated rationale: {entry.record.rationale}")
    if include_direction and (shown := direction_shown(entry.record, entry.target_kind)):
        lines.append(shown)
    if include_code and entry.record.source:
        lines.append(f"Implementation:\n{entry.record.source}")

    scale = "\n".join(f"  {value} = {meaning}" for value, meaning in sorted(SCALE.items()))
    user = (
        f"{context.render_for_judge()}\n\n"
        f"{QUESTION}\n\n"
        f"{chr(10).join(lines)}\n\n"
        f"Scale:\n{scale}\n\n"
        f"Answer with JSON only, no other text:\n"
        f'{{"score": <integer {MIN_SCORE}-{MAX_SCORE}>, "reason": "<one sentence>"}}'
    )
    return Rendered(item_id=entry.item_id, system=SYSTEM_PROMPT, user=user)
