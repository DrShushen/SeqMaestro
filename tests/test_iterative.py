"""The iterative source: the language model proposes features, is shown what a fit of the arm's
model made of them, and proposes again.

Everything here runs offline against `ScriptedClient`. The tests check the loop itself: what each
round is shown (round one is blind, since there is no fit to show yet), what happens to a repeated
feature, and the two equivalences that make the source comparable with the one-round source, to
which a single round should reduce it.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from seqmaestro.features import make_featuriser
from seqmaestro.features.base import PROVENANCE_GENERATED, PROVENANCE_SELECTED
from seqmaestro.llm import ScriptedClient, TaskContext
from seqmaestro.models import make_model
from seqmaestro.proposer import (
    GeneratedFeaturiser,
    IterativeGeneratedFeaturiser,
    ProposalSettings,
    Proposer,
)

CONTEXT = TaskContext(task="Predicting a binary outcome from DNA sequence.",
                      sequence_layout="12 nucleotides, 0-based.")
SEQUENCE, TARGET = "window", "outcome"
COLUMNS = {"sequence_column": SEQUENCE, "target_column": TARGET, "target_kind": "binary"}


def code(name, symbol):
    return (f"def {name}(X):\n"
            f"    return X['{SEQUENCE}'].str.count('{symbol}')")


def reply(*features):
    return json.dumps([{"name": n, "description": f"count of {s}", "rationale": "because",
                        "code": code(n, s)} for n, s in features])


def sequences(n=60):
    rng = np.random.default_rng(7)
    rich = ["".join(row) for row in rng.choice(list("GGGCCCAT"), size=(n // 2, 12))]
    poor = ["".join(row) for row in rng.choice(list("AAATTTGC"), size=(n - n // 2, 12))]
    return pd.Series(rich + poor)


def labels(n=60):
    return np.array([1] * (n // 2) + [0] * (n // 2))


def by_round(*replies):
    """A client that answers each call with the next scripted reply, then repeats the last."""
    calls = {"n": 0}

    def answer(system, user):
        index = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return replies[index]

    return ScriptedClient(answer)


def proposer_for(client, **kwargs):
    settings = kwargs.pop("settings", ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4))
    return Proposer(client, CONTEXT, settings=settings, **{**COLUMNS, **kwargs})


def iterative(client, rounds=2, pool=None, model="l1_logistic"):
    return IterativeGeneratedFeaturiser(
        proposer_for(client), lambda: make_model(model, target_kind="binary", seed=0),
        validation=(sequences(20), labels(20)), pool=pool, rounds=rounds,
    )


# --------------------------------------------------------------------------------------------
# What each round is shown
# --------------------------------------------------------------------------------------------


def test_the_first_round_is_blind_and_later_rounds_are_not():
    """Round one is shown no fit, so its request is the one-round source's, which the proposal
    cache key relies on."""
    client = by_round(reply(("gc", "G")), reply(("at", "A")))
    iterative(client, rounds=2).fit(sequences(), labels())

    first, second = (user for _, user in client.seen)
    assert "What it reported" not in first
    assert "A model was fitted" in second


def test_the_feedback_is_the_model_own_report():
    """`Model.feedback` decides what a round is told, and the featuriser passes it on unchanged."""
    client = by_round(reply(("gc", "G")), reply(("at", "A")))
    iterative(client, rounds=2, model="xgboost").fit(sequences(), labels())

    _, second = client.seen[1]
    assert "xgb_gain" in second, "the model named its own importance scale"
    assert "cover" in second, "and added what only it knows"


def test_a_symbolic_model_shows_the_proposer_its_equation_in_the_features_names():
    client = by_round(reply(("gc", "G")), reply(("at", "A")))
    iterative(client, rounds=2, model="pysr").fit(sequences(), labels())

    _, second = client.seen[1]
    assert "mean_abs_shapley" in second and '"equation"' in second and '"front"' in second
    assert not re.search(r"\bx\d+\b", second), "the search's placeholder names stay in its process"


def test_a_rounds_fit_takes_the_weights_the_arms_model_is_fitted_with():
    """On a dataset with per-row weights, the fit a round reports on is weighted as the arm's model
    is. An unweighted fit would show the language model what a different model relies on."""
    fitted_with = []

    class Recording:
        def __init__(self):
            self.model = make_model("l1_logistic", target_kind="binary", seed=0)

        def fit(self, X, y, sample_weight=None):
            fitted_with.append(sample_weight)
            self.model.fit(X, y, sample_weight=sample_weight)
            return self

        def feedback(self, X, y):
            return self.model.feedback(X, y)

    weights = np.linspace(0.5, 1.5, 60)
    client = by_round(reply(("gc", "G")), reply(("at", "A")))
    featuriser = IterativeGeneratedFeaturiser(proposer_for(client), Recording,
                                              validation=(sequences(20), labels(20)), rounds=2)
    featuriser.fit(sequences(), labels(), sample_weight=weights)

    assert len(fitted_with) == 1 and fitted_with[0] is weights


def test_a_later_round_sees_what_the_model_could_not_use():
    client = by_round(reply(("gc", "G")), reply(("at", "A")))
    iterative(client, rounds=2).fit(sequences(), labels())
    assert "unused" in client.seen[1][1]


# --------------------------------------------------------------------------------------------
# Repeated features, which a language model asked twice will produce
# --------------------------------------------------------------------------------------------


def test_a_feature_proposed_twice_is_kept_once():
    """`Featuriser.fit` refuses duplicate names, so without this the arm would not run at all."""
    client = by_round(reply(("gc", "G")), reply(("gc", "G"), ("at", "A")))
    featuriser = iterative(client, rounds=2).fit(sequences(), labels())

    names = [record.name for record in featuriser.records]
    assert names.count("gc") == 1
    assert set(names) == {"gc", "at"}


def test_the_round_report_says_what_each_round_actually_added():
    client = by_round(reply(("gc", "G")), reply(("gc", "G"), ("at", "A")))
    featuriser = iterative(client, rounds=2).fit(sequences(), labels())

    report = featuriser.round_report
    assert [r["round"] for r in report] == [1, 2]
    assert [r["had_feedback"] for r in report] == [False, True]
    assert report[1]["already_present"] == 1, "the repeat is counted, not silently dropped"
    assert report[1]["kept"] == 1


# --------------------------------------------------------------------------------------------
# The two equivalences
# --------------------------------------------------------------------------------------------


def test_one_round_over_a_generated_pool_is_the_one_round_source():
    """With one round and a pool that the blind proposal duplicates, the iterative source reduces to
    the one-round source of an arm such as `ebm/llm_generated`."""
    one_shot = GeneratedFeaturiser(proposer_for(by_round(reply(("gc", "G")))))
    one_shot.fit(sequences(), labels())

    pool = GeneratedFeaturiser(proposer_for(by_round(reply(("gc", "G")))))
    looped = iterative(by_round(reply(("gc", "G"))), rounds=1, pool=pool)
    looped.fit(sequences(), labels())

    assert [r.name for r in looped.records] == [r.name for r in one_shot.records]
    assert looped.transform(sequences()).equals(one_shot.transform(sequences()))


def test_one_round_over_a_fixed_pool_is_that_pool_plus_one_blind_proposal():
    pool = make_featuriser("positional")
    looped = iterative(by_round(reply(("gc", "G"))), rounds=1, pool=pool)
    looped.fit(sequences(), labels())

    names = [record.name for record in looped.records]
    assert names[-1] == "gc"
    assert all(name.startswith("pos_") for name in names[:-1])


# --------------------------------------------------------------------------------------------
# Provenance, which tells pooled features from generated ones of the same kind
# --------------------------------------------------------------------------------------------


def test_the_pool_is_selected_and_the_rounds_are_generated():
    """One cell holds features of one kind with two provenances, which `kind` alone cannot express."""
    pool = GeneratedFeaturiser(proposer_for(by_round(reply(("pooled", "G")))))
    looped = iterative(by_round(reply(("added", "A"))), rounds=1, pool=pool)
    looped.fit(sequences(), labels())

    provenance = {r.name: r.provenance for r in looped.records}
    assert provenance == {"pooled": PROVENANCE_SELECTED, "added": PROVENANCE_GENERATED}
    assert {r.kind for r in looped.records} == {"llm_generated"}, "same kind throughout"


def test_a_fixed_pool_is_selected_too():
    looped = iterative(by_round(reply(("gc", "G"))), rounds=1, pool=make_featuriser("positional"))
    looped.fit(sequences(), labels())
    provenance = {r.provenance for r in looped.records if r.kind == "positional"}
    assert provenance == {PROVENANCE_SELECTED}


# --------------------------------------------------------------------------------------------
# Degenerate rounds
# --------------------------------------------------------------------------------------------


def test_a_round_that_produces_nothing_usable_does_not_kill_the_arm():
    """A language model can return a constant feature or code that does not compile. The round is
    then empty and the arm continues."""
    constant = json.dumps([{"name": "always_one", "description": "d", "rationale": "r",
                            "code": f"def always_one(X):\n    return X['{SEQUENCE}'].map(lambda s: 1)"}])
    featuriser = iterative(by_round(reply(("gc", "G")), constant), rounds=2)
    featuriser.fit(sequences(), labels())

    assert [r.name for r in featuriser.records] == ["gc"]
    assert featuriser.round_report[1]["kept"] == 0


def test_an_arm_that_produces_nothing_at_all_is_refused():
    empty = json.dumps([])
    with pytest.raises(RuntimeError, match="No features survived"):
        iterative(by_round(empty), rounds=2).fit(sequences(), labels())


def test_zero_rounds_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        iterative(by_round(reply(("gc", "G"))), rounds=0)


def test_every_round_keeps_the_precision_the_arm_was_given():
    client = by_round(reply(("gc", "G")), reply(("at", "A")))
    featuriser = IterativeGeneratedFeaturiser(
        proposer_for(client), lambda: make_model("l1_logistic", target_kind="binary", seed=0),
        validation=(sequences(20), labels(20)), rounds=2, significant_digits=7)
    featuriser.fit(sequences(), labels())
    assert {part.significant_digits for part in featuriser._parts} == {7}
    assert featuriser.describe()["significant_digits"] == 7


# --------------------------------------------------------------------------------------------
# What a cell records about its written features
# --------------------------------------------------------------------------------------------


def constant(name):
    return {"name": name, "description": "d", "rationale": "r",
            "code": f"def {name}(X):\n    return X['{SEQUENCE}'].map(lambda s: 1)"}


def counting(name, symbol):
    return {"name": name, "description": f"count of {symbol}", "rationale": "because",
            "code": code(name, symbol)}


def test_the_fit_report_counts_the_written_features_of_every_round_and_nothing_else():
    """The agent checks these counts when most of an arm's features failed. They cover the blind
    proposal the pool holds, each informed round and a round that kept nothing, whereas the k-mer
    columns beside them are left out. The pool's proposal is round one, so it is not requested
    twice."""
    client = by_round(json.dumps([counting("gc", "G"), constant("flat_one")]),
                      json.dumps([counting("gc", "G"), counting("at", "A"), constant("flat_two")]),
                      json.dumps([constant("flat_three")]))
    proposer = proposer_for(client, settings=ProposalSettings(n_rounds=1, n_per_round=3,
                                                              n_examples=4, repair_rounds=0))
    pool = make_featuriser("kmer+llm_generated", k_range=(1, 1), kmer_normalise="counts",
                           proposer=proposer)
    featuriser = IterativeGeneratedFeaturiser(
        proposer, lambda: make_model("l1_logistic", target_kind="binary", seed=0),
        validation=(sequences(20), labels(20)), pool=pool, rounds=3)
    featuriser.fit(sequences(), labels())

    assert len(client.seen) == 3, "one request a round, the pool's being round one"
    assert [record.name for record in featuriser.records if record.kind == "llm_generated"] == [
        "gc", "at"]
    counts = [(entry["proposed"], entry["already_present"], entry["kept"],
               entry["dropped"].get("constant", 0)) for entry in featuriser.round_report]
    assert counts == [(2, 0, 1, 1), (3, 1, 1, 1), (1, 0, 0, 1)]
    report = featuriser.fit_report
    assert (report["n_proposed"], report["n_already_present"], report["n_kept"]) == (6, 1, 2)
    assert report["dropped"]["constant"] == 3
    assert report["n_proposed"] == (report["n_already_present"] + report["n_kept"]
                                    + sum(report["dropped"].values()))
    assert proposer.total.returned == 6 and proposer.stats.returned == 1, (
        "the total covers every round, where the last proposal's stats cover one")


def test_a_time_limit_that_cannot_be_set_stops_the_arm(monkeypatch):
    """A round that kept nothing lets the arm go on. A time limit that cannot be set, by contrast,
    says nothing about the round's features, so it fails the arm instead of leaving an empty
    round."""
    from seqmaestro.proposer.checks import TimeLimitUnavailable

    def refuse(self, inputs, y=None, sample_weight=None):
        raise TimeLimitUnavailable("no timer can be set here")

    monkeypatch.setattr(GeneratedFeaturiser, "fit", refuse)
    with pytest.raises(TimeLimitUnavailable):
        iterative(by_round(reply(("gc", "G"))), rounds=2).fit(sequences(), labels())
