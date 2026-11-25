"""The backends a score comes from: an interface, the real backend and a test double.

The scoring backend is kept apart from the protocol (selection, hiding the arms, ordering,
repeats and aggregation), so that the protocol can be tested exhaustively without a network call
or a credential, and a different model can be swapped in without changing it.

**A failed call gives no score.** A call that fails, times out, or returns something unparseable
yields `None`, which is passed through to the summary and counted. Treating a failure as a
middling score would pull every arm towards the mean, and would flatter most the arms whose
features are hardest to parse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from seqmaestro.judge.rubric import MAX_SCORE, MIN_SCORE, Rendered
from seqmaestro.llm import CallSettings, LiteLLMClient

#: The tokens a judgement's reply may take. The reply is one integer and one sentence, so the limit
#: is short, whereas a proposal's reply of several features in code needs far more.
REPLY_TOKENS = 512


@dataclass(frozen=True)
class Judgement:
    """One reply. `score is None` means the judge did not answer, not that it answered neutrally."""

    item_id: str
    score: int | None
    reason: str | None = None
    error: str | None = None


class ScoringEngine(Protocol):
    def judge(self, rendered: Rendered) -> Judgement: ...


def parse_reply(item_id: str, text: str) -> Judgement:
    """Pull a score out of a reply, tolerating how it is wrapped but inventing nothing.

    Models wrap JSON in prose or code fences often enough that insisting on clean JSON would
    discard good answers, but inventing a score from a reply that contains none would be worse.
    The parser therefore takes the JSON object wherever it is in the reply, or a reply that is the
    score alone, and otherwise gives up. It never picks a number out of a sentence, because a reply
    that restates the scale before answering would then be scored by the scale's first number.
    """
    if not text or not text.strip():
        return Judgement(item_id, None, error="empty reply")

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1).strip()

    brace = re.search(r"\{.*\}", candidate, re.S)
    if brace:
        try:
            payload = json.loads(brace.group(0))
            score = int(payload["score"])
            if MIN_SCORE <= score <= MAX_SCORE:
                reason = payload.get("reason")
                return Judgement(item_id, score, reason=str(reason) if reason else None)
            return Judgement(item_id, None, error=f"score {score} out of range")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    if candidate.isdigit() and MIN_SCORE <= int(candidate) <= MAX_SCORE:
        return Judgement(item_id, int(candidate))

    return Judgement(item_id, None, error=f"unparseable: {text[:120]!r}")


class LiteLLMEngine:
    """The real backend: `seqmaestro.llm.LiteLLMClient`, one call per judgement.

    Repeats are separate calls instead of `n>1` on one call, so they are independent samples of the
    judge. Their spread is then a property of the instrument, which is what is being characterised.
    Every call is made under the run's `settings`, as the proposer's are, and its reply may take up
    to `REPLY_TOKENS`.
    """

    def __init__(self, settings: CallSettings):
        self.client = LiteLLMClient(settings)

    def judge(self, rendered: Rendered) -> Judgement:
        completion = self.client.complete(rendered.system, rendered.user,
                                          max_completion_tokens=REPLY_TOKENS)
        if not completion.ok:
            return Judgement(rendered.item_id, None, error=completion.error)
        return parse_reply(rendered.item_id, completion.text)

    def describe(self) -> dict[str, object]:
        """Which model the judgements were asked of, and how (`LiteLLMClient.describe`)."""
        return self.client.describe()

    def usage(self) -> dict:
        return self.client.usage()


class ScriptedEngine:
    """A deterministic stand-in, so the protocol can be tested without a network or a bill.

    It takes a callable from the rendered prompt to a reply string, or a fixed reply, so a test can
    simulate a well-behaved judge, a malformed one, or an outright failure (by returning an
    exception instance). `described` is the record it gives of the model it stands for, which a
    test can vary to check what is stored under that record.
    """

    def __init__(self, reply, described: dict[str, object] | None = None):
        self.reply = reply
        self.described = dict(described) if described is not None else {"model": "scripted"}
        self.seen: list[Rendered] = []

    def judge(self, rendered: Rendered) -> Judgement:
        self.seen.append(rendered)
        text = self.reply(rendered) if callable(self.reply) else self.reply
        if isinstance(text, BaseException):
            return Judgement(rendered.item_id, None, error=str(text))
        return parse_reply(rendered.item_id, text)

    def describe(self) -> dict[str, object]:
        return dict(self.described)

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.seen)}
