"""The question the judge is asked, and how a feature is shown to it.

`docs/analyses/judge.md` is the maintained account of the whole protocol. **A change to the
wording here changes what every past score means**, so change that page in the same commit.

**One question, not a multi-part rubric.** The obvious alternative scores separate parts
(mechanistic plausibility, specificity, standing relative to known biology, testability), but those
parts mostly co-vary. Averaging them gives a number nobody can interpret, and reporting them
separately quadruples the output without changing any decision.

**"If it were true" separates significance from correctness.** The judge is not asked to decide
truth from a description it cannot verify, a task it would do badly while sounding confident. The
phrase also sharpens the negative control. A noise-derived feature *can* be interesting if true,
and the question allows that, so if permuted features score as highly as real ones, the score
tracks how a feature is written and not what produced it.

Nothing here knows which arm a feature came from. Blinding is enforced by construction, because
the renderer is given a `PanelEntry` and reads only its blinded fields, and `tests/test_judge.py`
checks that no arm name reaches a prompt.
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

#: Anchored, because an unanchored 1-5 drifts between calls and cannot be compared across arms.
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


# `TaskContext` lives in `seqmaestro.llm` because the proposer needs it too, for the same reason.
# It is re-exported here so a reader of the rubric finds it where they expect it.
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
#: given in the same prompt, so no dataset-specific wording lives here.
DIRECTION_LINE = {
    HIGHER_IN_POSITIVE: "Observed in the data: this feature takes higher values in the positive class.",
    HIGHER_IN_NEGATIVE: "Observed in the data: this feature takes higher values in the negative class.",
    INCREASES_TARGET: "Observed in the data: higher values of this feature go with a higher target.",
    DECREASES_TARGET: "Observed in the data: higher values of this feature go with a lower target.",
}


def direction_shown(record, target_kind: str) -> str | None:
    """The sentence a judge would be shown for this feature, or None if it would be shown none.

    Separate from `render` so a caller can ask how much of a panel carries a direction without
    rendering it. Asking for directions and getting none is otherwise silent: the setting is
    recorded as on, every prompt goes out without one, and nothing says so.
    """
    return DIRECTION_LINE.get(from_sign(record.marginal_effect, target_for(target_kind).directions))


def render(entry, context: TaskContext, include_code: bool = False,
           include_direction: bool = False) -> Rendered:
    """Build the prompt for one blinded feature.

    `include_code` is off by default. Showing the implementation may help the judge understand an
    ambiguous description, or may bias it towards whatever looks sophisticated. Which of the two
    happens is a question to test separately.

    `include_direction` is off by default because it changes what is being asked. "How
    significant would this feature be if it were true" is barely answerable for a count of upstream
    start codons, which is plausible either way round. Told which way it runs, the judge scores a
    specific claim and can mark down one that contradicts what is known. That is a better
    instrument but a different one, so scores from a run with it on are not comparable with scores
    from a run without it. A run therefore states it once, as `grid.judge_direction` in its config,
    and the judge's summary records it beside the results.

    **The sign, never the size.** The judge is shown which class the feature runs with, or which
    way a continuous target moves with it, and nothing about how strongly, because an effect size
    would let it tell the real features from the permuted-label controls, which an arm's mean is
    compared with when they are judged. A sign is a coin flip for a control feature, so it
    discriminates nothing. The direction shown is the marginal one, and not the model's, for two
    reasons. It is a statement about the data, which is what the question asks about. And it is
    always one of two words, while the model's admits a third (`non_monotone`) that a noise feature
    might earn more often than a real one.

    A feature with no direction to show (its marginal effect is missing, or exactly zero, which is
    rare) gets no line, and no direction is invented for it.

    **The judge gets the unblinded context**, from `TaskContext.render_for_judge` and not
    `TaskContext.render`. Two senses of blinding meet here, and neither implies the other. The
    *feature* is blinded: the judge is not told which arm produced it or how it scored. The
    *dataset* may withhold a fact from the systems that proposed features and disclose it here,
    because rating what a feature would mean needs the phenotype named, while a proposer must not be
    handed it.
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
