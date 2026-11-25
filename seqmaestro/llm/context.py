"""What a language model must be told about the problem to make sense of what it is shown.

A judge given only a feature's description scores the writing, and a proposer given only a
sequence column and a label column proposes generic string statistics, so every consumer requires a
context. **Each dataset has one description, loaded from a file** (`load_task` reads the
descriptions from a directory, and the library ships none), so that every consumer is told the same
facts. `experiments/datasets/README.md` describes the fields, blinding (`judge_only`), and what a
description leaves out.

The blind rendering is the default: a caller that reaches for `render()` gets what the models that
write features get, so a caller that picks the wrong one of the two discloses too little and never
reveals the withheld fact.

A description is only part of what the models are told, so `blinding_leaks` finds where a withheld
word appears in the column names and covariate descriptions they are shown, and `guidance_leaks`
where one appears in the guidance. A run set up as blinded is refused before it starts when either
finds one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: The fields a description file may hold. Unknown keys are refused rather than ignored, because a
#: typo in a dataset description would otherwise be a silently missing fact.
FIELDS = ("task", "sequence_layout", "notes", "target_name", "judge_only")

#: The fields a model consumer may see. They are derived from `FIELDS` instead of listed again, so
#: a field added there is disclosed to the models unless it is excluded here. `judge_only` is the
#: only exclusion, and the reason for the split.
BLIND_FIELDS = tuple(field for field in FIELDS if field != "judge_only")

#: The length of the shortest word that counts as withheld. Shorter words ("is", "a") are too
#: common to give anything away, and a column name holds them by chance.
MIN_WITHHELD_LENGTH = 3


def _words(text: str | None) -> set[str]:
    """The words of `text` in lower case, split at anything but a letter or a digit, so that "type2"
    stays one word. Words shorter than `MIN_WITHHELD_LENGTH` are left out."""
    return {word.lower() for word in re.split(r"[^A-Za-z0-9]+", text or "")
            if len(word) >= MIN_WITHHELD_LENGTH}


@dataclass(frozen=True)
class TaskContext:
    """The problem, in the terms a domain expert would state it: what is predicted, the layout of the
    sequences, any notes, and the fact told to the judge alone (`judge_only`).

    `experiments/datasets/README.md` says what each field holds, why a description states facts and no
    known confounds, and why it holds no framing specific to one consumer.
    """

    task: str
    sequence_layout: str | None = None
    notes: str | None = None
    #: A short noun phrase for the prediction target, for consumers whose prompt templates
    #: interpolate one (DEFT's do). It is **not** part of `render()`: it exists so that a config can
    #: be generated from this description, and rendering it would change prompts whose results have
    #: already been measured.
    target_name: str | None = None
    #: The blinded fact: told to the judge by `render_for_judge`, and never reaching a model
    #: consumer by any path. Set it only for what a proposer could look up instead of finding, and
    #: say in the run's report that the run was blinded, and to what. A blinding that is not
    #: recorded cannot be told apart afterwards from a description that was simply thin.
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
        """What a model consumer in this package (the proposer, the recurrence analysis) is told. It
        never includes `judge_only` (see the module docstring)."""
        parts = [f"Task: {self.task}"]
        if self.sequence_layout:
            parts.append(f"Sequence layout: {self.sequence_layout}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return "\n".join(parts)

    def render_for_judge(self) -> str:
        """`render`'s text, plus whatever this dataset withholds from the models.

        It is built on top of `render`, so a field added there appears in both. The disclosure is
        appended as plain context, without a label saying it was withheld, because the judge is
        asked what a feature would mean if it held, and telling it that the systems were working
        blind invites it to score the achievement instead of the biology.
        """
        if not self.judge_only:
            return self.render()
        return f"{self.render()}\nAdditional context: {self.judge_only}"

    def as_dict(self) -> dict:
        """Every field, for recording beside the output, because a result is comparable only to
        another asked the same way.

        It includes `judge_only`, because a record that omits what was withheld cannot be told
        apart from a run that withheld nothing. This dict is a record and never goes into a prompt.
        """
        return {field: getattr(self, field) for field in FIELDS}

    def as_dict_blind(self) -> dict:
        """Every field except `judge_only`, for keying anything that only the models answer.

        A proposal cache is keyed on everything that would change the reply. `judge_only` cannot
        change it, since a model consumer is never shown it, so including it would discard the
        cached proposals whenever the judge's context was edited, and pay again for identical
        answers.
        """
        return {field: getattr(self, field) for field in BLIND_FIELDS}

    def withheld_words(self) -> frozenset[str]:
        """The words of `judge_only`, in lower case, that are in none of the fields the models are
        told: what the description withholds, word by word. The set is empty where it withholds
        nothing."""
        told = set().union(*(_words(getattr(self, field)) for field in BLIND_FIELDS))
        return frozenset(_words(self.judge_only) - told)


def blinding_leaks(context: TaskContext, columns: Mapping[str, str | None]) -> list[str]:
    """Each place where a word that `context` withholds appears in what the models are told about
    the columns, as a sentence. The list is empty where there is none, or where the description
    withholds nothing.

    `columns` maps each column name the models are shown to its description, or to None where it
    has none. **A name holds a word** where the word appears anywhere in it once everything but its
    letters and digits is removed, so `is_type2` and `isType2` both hold "type2". **A description
    holds a word** only as a whole word, since prose holds short words inside longer ones by chance
    ("cap" in "capture").

    Every withheld word counts, including the common ones, so a name can be refused for holding a
    word such as "line" by chance. The remedy is to rename the column, or to reword `judge_only` so
    the word is not in it.
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


def guidance_leaks(context: TaskContext, guidance: str | None) -> list[str]:
    """Each word that `context` withholds and `guidance` (the guidance on the form of the features)
    contains, as a sentence. The list is empty where there is no guidance, or where the description
    withholds nothing.

    Every model that writes features is shown the guidance word for word, so on a blinded run it
    must not name anything withheld. Because the guidance is prose, it holds a word only as a whole
    word, as a covariate's description does (`blinding_leaks`).
    """
    held = _words(guidance)
    return [f"the feature guidance holds the withheld word {word!r}"
            for word in sorted(context.withheld_words()) if word in held]


def load_task(directory: str | Path, name: str) -> TaskContext:
    """Load `<directory>/<name>.json`. A name without a file is refused, with a list of the files
    there.

    The proposer, the judge and a vendored method are told the same task description, because all
    three read this file.
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
