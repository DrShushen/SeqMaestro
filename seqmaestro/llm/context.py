"""What a language model must be told about the problem to make sense of what it is shown.

Shared by the proposer, the judge and the recurrence analysis. The first two fail the same way
without it. "GC fraction of positions 41-49" cannot be judged without knowing the organism, the
assay and what position 41 is measured from, and a judge given only the description scores the
writing. A proposer given only a sequence column and a label column proposes generic string
statistics, because that is all it has been told the problem is about.

Every consumer requires a context, and **each dataset has one description, loaded from a file**.
The facts are the same for every consumer, and a dataset described separately in each place will
eventually be described differently in each place. `load_task` reads the descriptions from a
directory of files. This library ships none (see `experiments/datasets/`).

**One field may be told to the judge and withheld from the models**: `judge_only`. This is
blinding, and carries no consumer-specific framing. The proposer and a generative method are what
an experiment tests, so a fact they could *retrieve* instead of discover (the name of the protein
whose loss defines the classes, say) makes the result uninterpretable if they are given it. The
judge is not under test. It is the instrument that scores what they found, and it cannot say how
significant a feature would be without knowing what the phenotype is. The field still lives in the
same file, so the asymmetry is written down with everything else and not arranged by whoever
assembles a prompt.

The blind rendering is the default. A caller that reaches for `render()` gets what the models get,
so picking the wrong one of the two under-discloses and never leaks.

**A description is only part of what the models are told.** They also see the names of the columns
they are handed (the target column's name ends every example row of a proposer's prompt), and each
covariate's description from the dataset's declaration. A blinded run that withholds which
cells were treated, and whose target column is `is_treated`, tells every model what its
`judge_only` withholds. `blinding_leaks` finds where a withheld word shows in those names and
descriptions, and a run set up as blinded is refused before it starts when one does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: The fields a description file may carry. Unknown keys are refused rather than ignored, because a
#: typo in a dataset description is otherwise a silently missing fact.
FIELDS = ("task", "sequence_layout", "notes", "target_name", "judge_only")

#: The fields a model consumer may see. Derived from `FIELDS` instead of listed again, so a field
#: added there is disclosed to the models unless it is excluded here. `judge_only` is the one
#: exclusion, and the reason for the split.
BLIND_FIELDS = tuple(field for field in FIELDS if field != "judge_only")

#: The shortest word that counts as withheld. Shorter words ("is", "a") are too common to give
#: anything away, and a column name holds them by chance.
MIN_WITHHELD_LENGTH = 3


def _words(text: str | None) -> set[str]:
    """The words of `text` in lower case, split at anything but a letter or a digit, so that "type2"
    stays one word. Words shorter than `MIN_WITHHELD_LENGTH` are left out."""
    return {word.lower() for word in re.split(r"[^A-Za-z0-9]+", text or "")
            if len(word) >= MIN_WITHHELD_LENGTH}


@dataclass(frozen=True)
class TaskContext:
    """The problem, in the terms a domain expert would state it.

    **Factual about the task and silent about known confounds.** Telling the judge that GC content
    is confounded in this dataset would have it mark GC features down, which is a judgement about
    whether a feature is real, the question "if it were true" brackets off. Telling the proposer
    would steer it away from a hypothesis the experiment is meant to let it reach. Such warnings
    belong in the report. `notes` can carry one if you decide otherwise, but that is a different
    experiment and should be recorded as one.

    **No framing here is consumer-specific.** The judge knows it is judging and the proposer knows
    it is proposing, and that framing belongs in their own prompts. Putting it in the description
    of the data is how two consumers of one dataset drift apart.

    `judge_only` does not break that rule, because it is a fact about the data like every other
    field. It is disclosed to one consumer and withheld from the others, and which consumers are
    blind is a property of the experiment and not of any prompt.
    """

    task: str
    sequence_layout: str | None = None
    notes: str | None = None
    #: A short noun phrase for the prediction target, for consumers whose prompt templates
    #: interpolate one (DEFT's do). It is **not** part of `render()`: it exists so a config can be
    #: generated from this description, and rendering it would change prompts that have already
    #: been measured.
    target_name: str | None = None
    #: The blinded fact: told to the judge by `render_for_judge`, and reaching no model consumer
    #: through any path. Set it only for what a proposer could look up instead of finding, and say
    #: in the run report that the run was blinded and to what. A blinding nobody records cannot be
    #: told apart afterwards from a description that was simply thin.
    judge_only: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> TaskContext:
        payload = json.loads(Path(path).read_text())
        unknown = set(payload) - set(FIELDS)
        if unknown:
            raise ValueError(
                f"{path}: unknown field(s) {sorted(unknown)}. Allowed: {', '.join(FIELDS)}. A "
                f"misspelled key would otherwise be a fact silently missing from every prompt."
            )
        if not payload.get("task"):
            raise ValueError(f"{path}: `task` is required; every consumer reads it.")
        return cls(**payload)

    def render(self) -> str:
        """What a model consumer in this package is told (the proposer, the recurrence analysis). It
        never carries `judge_only` (see the module doc)."""
        parts = [f"Task: {self.task}"]
        if self.sequence_layout:
            parts.append(f"Sequence layout: {self.sequence_layout}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return "\n".join(parts)

    def render_for_judge(self) -> str:
        """`render`'s text, plus whatever this dataset withholds from the models.

        Built on top of `render`, so a field added there appears in both. The disclosure is
        appended as plain context, with no label saying it was withheld. The judge is asked what a
        feature would mean if it held, and telling it that the systems were working blind invites
        it to score the achievement instead of the biology.
        """
        if not self.judge_only:
            return self.render()
        return f"{self.render()}\nAdditional context: {self.judge_only}"

    def as_dict(self) -> dict:
        """Every field, for recording beside the output. A result is comparable only to another
        asked the same way.

        Carries `judge_only`, because a record that omits what was withheld cannot be told apart
        from a run that withheld nothing. This dict is a record and never goes into a prompt.
        """
        return {field: getattr(self, field) for field in FIELDS}

    def as_dict_blind(self) -> dict:
        """Every field except `judge_only`, for keying anything that only the models answer.

        A proposal cache is keyed on everything that would change the reply. `judge_only` cannot
        change it, since no model consumer is ever shown it, so including it would discard cached
        proposals whenever the judge's context was edited and pay again for identical answers.
        """
        return {field: getattr(self, field) for field in BLIND_FIELDS}

    def withheld_words(self) -> frozenset[str]:
        """The words of `judge_only` that no field the models are told holds, in lower case: what
        the description withholds, word by word. Empty where it withholds nothing."""
        told = set().union(*(_words(getattr(self, field)) for field in BLIND_FIELDS))
        return frozenset(_words(self.judge_only) - told)


def blinding_leaks(context: TaskContext, columns: Mapping[str, str | None]) -> list[str]:
    """Each place a word `context` withholds shows in what the models are told about the columns,
    in words. Empty where there is none, or where the description withholds nothing.

    `columns` maps each column name the models are shown to its description, or to None where it
    has none. **A name holds a word** where the word appears anywhere in it once everything but its
    letters and digits is removed, so `is_type2` and `isType2` both hold "type2". **A description
    holds a word** only as a whole word, since prose holds short words inside longer ones by chance
    ("cap" in "capture").

    Every withheld word counts, the common ones too, so a name can be refused for holding a word
    such as "line" by chance. The remedy is to rename the column, or to reword `judge_only` so the
    word is not in it.
    """
    withheld = sorted(context.withheld_words())
    leaks = []
    for name, description in columns.items():
        letters = re.sub(r"[^a-z0-9]", "", name.lower())
        leaks += [f"the column name {name!r} holds the withheld word {word!r}"
                  for word in withheld if word in letters]
        leaks += [f"the description of {name!r} holds the withheld word {word!r}"
                  for word in withheld if word in _words(description)]
    return leaks


def load_task(directory: str | Path, name: str) -> TaskContext:
    """Load `<directory>/<name>.json`. A name with no file is refused with a list of those there.

    A run's task describes the same dataset to the proposer, the judge and a vendored method,
    because all three read this file.
    """
    directory = Path(directory)
    path = directory / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in directory.glob("*.json"))
        raise ValueError(
            f"No dataset description {name!r} in {directory}. "
            f"Available: {', '.join(available) or '(none)'}."
        )
    return TaskContext.from_json(path)
