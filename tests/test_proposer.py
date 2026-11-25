"""The proposer, the proposal cache, and the featuriser that runs the code a language model wrote.

Everything here runs offline against `ScriptedClient`, which works because the call to the language
model sits behind an interface. Proposal parsing, rejection accounting, caching, the fit/transform
discipline and the failure-filling policy are all tested without a credential or a bill. Only the
wording of the prompt needs a real call to evaluate.

The code guard is tested in `test_code_guard.py`, a port of DEFT's guard tests.
"""

from __future__ import annotations

import json
import multiprocessing
import pickle
import re
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqmaestro.dataset import Columns
from seqmaestro.features.base import ORIGIN_SEQMAESTRO, Inputs
from seqmaestro.harness.features import load_features
from seqmaestro.llm import ScriptedClient, TaskContext
from seqmaestro.proposer import (
    KIND_LLM_GENERATED,
    GeneratedFeaturiser,
    ProposalCache,
    ProposalSettings,
    Proposer,
    digest_rows,
    parse_proposals,
)
from seqmaestro.proposer.checks import (
    CONSTANT,
    RAISED,
    SLOW,
    Check,
    TimeLimitUnavailable,
    check,
    scaled_limit,
    time_limit,
    to_significant_digits,
)
from seqmaestro.proposer.guard import compile_feature_function
from seqmaestro.proposer.prompt import GUIDANCE_INTRODUCTION, render_repair_request
from seqmaestro.target import BINARY
from tests.planted import run_cell

CONTEXT = TaskContext(
    task="Predicting a binary outcome from DNA sequence.",
    sequence_layout="12 nucleotides, 0-based, position 6 is the site.",
)

# Neither is a real dataset's column name. The proposer requires both and defaults neither, and a
# plausible name here would let a default slip in unnoticed.
SEQUENCE, TARGET = "window", "outcome"
COLUMNS = {"sequence_column": SEQUENCE, "target_column": TARGET, "target_kind": "binary"}
SPLIT_COLUMNS = Columns(SEQUENCE, TARGET, weight=None, covariates=(), target_kind=BINARY)
#: What the covariate the tests use measures, as a dataset would declare it.
DESCRIBED = {"signal": "a signal measured over the window"}

# The name neither contains nor appears inside the name of the worked example in `CODE_CONTRACT`
# (`gc_fraction_first_<n>nt`). Otherwise a test asserting that a name is absent from a prompt
# would pass or fail on the example instead of on the proposal.
GC_CODE = (
    "def whole_window_gc(X):\n"
    "    seqs = X['window'].astype(str)\n"
    "    return (seqs.str.count('G') + seqs.str.count('C')) / seqs.str.len()"
)
A_COUNT_CODE = "def adenine_count(X):\n    return X['window'].str.count('A')"


def sequences(n=40):
    """Sequences of varied composition, correlated with the label.

    With only one distinct sequence per class, almost any proposed feature would be constant. The
    featuriser drops constants, so the fixture would then decide what the test sees.
    """
    rng = np.random.default_rng(7)
    gc_rich = ["".join(row) for row in rng.choice(list("GGGCCCAT"), size=(n // 2, 12))]
    at_rich = ["".join(row) for row in rng.choice(list("AAATTTGC"), size=(n - n // 2, 12))]
    return pd.Series(gc_rich + at_rich)


def labels(n=40):
    return np.array([1] * (n // 2) + [0] * (n // 2))


def reply(*features) -> str:
    return json.dumps([
        {"name": name, "description": f"description of {name}", "rationale": "because", "code": code}
        for name, code in features
    ])


def proposer(reply_text, **kwargs):
    settings = kwargs.pop("settings", ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4))
    return Proposer(ScriptedClient(reply_text), CONTEXT, settings=settings, **{**COLUMNS, **kwargs})


# --------------------------------------------------------------------------------------------
# What the language model returns becomes a record or is counted as a rejection
# --------------------------------------------------------------------------------------------


def test_a_well_formed_proposal_becomes_a_feature_record():
    records = proposer(reply(("whole_window_gc", GC_CODE))).propose(sequences(), labels())

    assert len(records) == 1
    record = records[0]
    assert record.name == "whole_window_gc"
    assert record.kind == KIND_LLM_GENERATED
    assert record.origin == ORIGIN_SEQMAESTRO
    assert record.source == GC_CODE
    assert record.rationale == "because"


def test_kind_says_a_model_wrote_it_and_origin_says_whose_method():
    """`kind` records that a language model wrote the feature, whereas `origin` records whose method
    produced it. They answer different questions, and every adapter must keep them apart."""
    record = proposer(reply(("whole_window_gc", GC_CODE))).propose(sequences(), labels())[0]

    assert record.kind == KIND_LLM_GENERATED, "same kind whoever's model wrote it"
    assert record.origin == ORIGIN_SEQMAESTRO, "ours, as against DEFT's"


def test_an_illegal_identifier_is_rejected_and_counted():
    """Language models often write names such as `upstream_-15nt_GC` that are not legal
    identifiers. DEFT discards them silently, so here each rejection is counted."""
    p = proposer(reply(("upstream_-15nt_GC", GC_CODE), ("whole_window_gc", GC_CODE)))
    records = p.propose(sequences(), labels())

    assert [r.name for r in records] == ["whole_window_gc"]
    assert p.stats.rejected_name == 1
    assert p.stats.returned == 2 and p.stats.accepted == 1


def test_unsafe_code_is_rejected_and_counted_separately_from_bad_code():
    """Code that does not compile and code that tries to escape the guard are different events,
    so they are counted separately."""
    unsafe = 'def f(X):\n    return __import__("os").getpid()'
    broken = "def f(X):\n    return X[[[["

    p = proposer(reply(("escape", unsafe), ("broken", broken), ("whole_window_gc", GC_CODE)))
    records = p.propose(sequences(), labels())

    assert [r.name for r in records] == ["whole_window_gc"]
    assert p.stats.rejected_unsafe == 1
    assert p.stats.rejected_syntax == 1


def test_a_failed_call_is_recorded_and_does_not_abort_the_remaining_rounds():
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        return RuntimeError("upstream timeout") if calls["n"] == 1 else reply(("whole_window_gc", GC_CODE))

    p = proposer(flaky, settings=ProposalSettings(n_rounds=2, n_per_round=1, n_examples=4))
    records = p.propose(sequences(), labels())

    assert [r.name for r in records] == ["whole_window_gc"]
    assert p.stats.rounds == 2
    assert len(p.stats.errors) == 1 and "timeout" in p.stats.errors[0]


def test_a_reply_that_is_not_json_is_recorded_rather_than_guessed_at():
    p = proposer("I would suggest looking at GC content.")
    assert p.propose(sequences(), labels()) == []
    assert p.stats.errors and "no JSON array" in p.stats.errors[0]


def test_the_same_feature_proposed_twice_is_kept_once():
    p = proposer(
        reply(("whole_window_gc", GC_CODE)),
        settings=ProposalSettings(n_rounds=3, n_per_round=1, n_examples=4),
    )
    records = p.propose(sequences(), labels())

    assert len(records) == 1
    assert p.stats.duplicates == 2


def test_later_rounds_are_told_what_earlier_rounds_proposed():
    """Without this, every round would re-propose GC content and the extra calls would add
    nothing."""
    client = ScriptedClient(reply(("whole_window_gc", GC_CODE)))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=2, n_per_round=1, n_examples=4),
                 **COLUMNS)
    p.propose(sequences(), labels())

    first, second = (user for _, user in client.seen)
    assert "whole_window_gc" not in first
    assert "whole_window_gc" in second


# --------------------------------------------------------------------------------------------
# Label-awareness: without it the permuted cells would have no features derived from noise
# --------------------------------------------------------------------------------------------


def test_the_model_is_shown_the_labels():
    client = ScriptedClient(reply(("whole_window_gc", GC_CODE)))
    Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4),
             **COLUMNS).propose(sequences(), labels())

    _, user = client.seen[0]
    assert f"{TARGET}=1" in user and f"{TARGET}=0" in user


def test_the_model_is_shown_the_guidance_word_for_word_and_nothing_where_there_is_none():
    asked = "Write each feature as a ratio of two counts, measured within 20 nt of the centre."
    guided, plain = (ScriptedClient(reply(("whole_window_gc", GC_CODE))) for _ in range(2))
    settings = ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4)
    Proposer(guided, CONTEXT, settings=replace(settings, guidance=asked), **COLUMNS).propose(
        sequences(), labels())
    Proposer(plain, CONTEXT, settings=settings, **COLUMNS).propose(sequences(), labels())

    _, user = guided.seen[0]
    assert f"{GUIDANCE_INTRODUCTION}\n{asked}" in user
    _, user = plain.seen[0]
    assert GUIDANCE_INTRODUCTION not in user


def test_the_examples_shown_are_class_balanced():
    """An unbalanced draw of so few examples can show one class three times out of four, and a
    language model shown almost no positives proposes features about the negatives."""
    client = ScriptedClient(reply(("whole_window_gc", GC_CODE)))
    skewed = pd.Series(["GGCCGGCCGGCC"] * 38 + ["AATTAATTAATT"] * 2)
    y = np.array([1] * 38 + [0] * 2)

    Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4),
             **COLUMNS).propose(skewed, y)

    _, user = client.seen[0]
    assert user.count(f"{TARGET}=0") == 2, "both minority rows shown, not a proportional two-in-forty"


def test_permuted_labels_change_the_cache_key():
    """If shuffling the labels did not change the key, a permuted cell would be handed the real
    cell's features, and the judge would have no null to score against."""
    p = proposer(reply(("whole_window_gc", GC_CODE)))
    real = labels()
    permuted = np.random.default_rng(0).permutation(real)

    assert p.key_for(sequences(), real) != p.key_for(sequences(), permuted)


# --------------------------------------------------------------------------------------------
# The cache, so that a proposal is requested once
# --------------------------------------------------------------------------------------------


def test_the_second_asker_gets_the_first_asker_s_features_without_a_call(tmp_path):
    """A proposal is a language model's answer, paid for once: a second request under the same key
    is answered from the cache, as when a cell is fitted again or a process beside this one asks
    for the same proposal. A proposer that is given its arm includes it in the key, so two arms
    never share a proposal."""
    cache = ProposalCache(tmp_path)
    first = proposer(reply(("whole_window_gc", GC_CODE)), cache=cache)
    records_g2 = first.propose(sequences(), labels())

    second = proposer(reply(("something_else", A_COUNT_CODE)), cache=cache)
    records_g3 = second.propose(sequences(), labels())

    assert [r.name for r in records_g3] == [r.name for r in records_g2] == ["whole_window_gc"]
    assert second.client.seen == [], "the second asker made no call at all"
    assert second.stats.cache_hit is True


def _propose_beside_another(cache: Path, name: str, code: str, calls: Path, out: Path) -> None:
    """Make one process's proposal, with a language model that takes a second to answer and notes
    each call."""
    def answer(system, user):
        with calls.open("a") as handle:
            handle.write(f"{name}\n")
        time.sleep(1.0)
        return reply((name, code))

    asker = Proposer(ScriptedClient(answer), CONTEXT,
                     settings=ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4),
                     cache=ProposalCache(cache), **COLUMNS)
    out.write_text(json.dumps([record.name for record in asker.propose(sequences(), labels())]))


def test_processes_asking_for_one_proposal_at_once_share_one_answer(tmp_path):
    """Two processes ask for one entry at the same moment, as a process started by hand beside the
    driver's may. The second waits for the first's answer and takes it, so both fit the same
    features and the language model is called once."""
    calls = tmp_path / "calls.txt"
    askers = [multiprocessing.get_context("fork").Process(
                  target=_propose_beside_another,
                  args=(tmp_path / "cache", name, code, calls, tmp_path / f"{name}.json"))
              for name, code in (("whole_window_gc", GC_CODE), ("something_else", A_COUNT_CODE))]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(60)
    assert [asker.exitcode for asker in askers] == [0, 0]
    assert len(calls.read_text().splitlines()) == 1, "the model was asked once"
    answers = [json.loads((tmp_path / f"{name}.json").read_text())
               for name in ("whole_window_gc", "something_else")]
    assert answers[0] == answers[1]
    assert [path.name for path in (tmp_path / "cache").glob("*.json")] != []
    assert not list((tmp_path / "cache").glob("*.partial")), "an entry is written whole"


def test_a_cache_entry_is_a_readable_features_file(tmp_path):
    """A cache entry needs no special tooling: `load_features` reads and validates it like any
    other feature file."""
    p = proposer(reply(("whole_window_gc", GC_CODE)), cache=ProposalCache(tmp_path))
    p.propose(sequences(), labels())

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    records, header = load_features(written[0])
    assert [r.name for r in records] == ["whole_window_gc"]
    assert header["proposal_stats"]["accepted"] == 1
    assert header["context"]["task"] == CONTEXT.task


@pytest.mark.parametrize(
    "label, change",
    [
        ("different rows", lambda p: p.key_for(sequences(20), labels(20))),
        ("different seed", lambda p: Proposer(p.client, CONTEXT, p.settings, seed=1, **COLUMNS)
                                     .key_for(sequences(), labels())),
        ("different settings",
         lambda p: Proposer(p.client, CONTEXT, ProposalSettings(n_rounds=9), **COLUMNS)
                   .key_for(sequences(), labels())),
        ("different task",
         lambda p: Proposer(p.client, TaskContext(task="something else"), **COLUMNS)
                   .key_for(sequences(), labels())),
        ("covariates beside the rows",
         lambda p: Proposer(p.client, CONTEXT, p.settings, covariates=DESCRIBED, **COLUMNS)
                   .key_for(with_covariates(sequences()), labels())),
        ("guidance on the form of the features",
         lambda p: Proposer(p.client, CONTEXT, replace(p.settings, guidance="Use ratios."), **COLUMNS)
                   .key_for(sequences(), labels())),
        ("another model",
         lambda p: Proposer(ScriptedClient(p.client.reply, described={"model": "another"}), CONTEXT,
                            p.settings, **COLUMNS).key_for(sequences(), labels())),
        ("another reasoning effort",
         lambda p: Proposer(ScriptedClient(p.client.reply, described={
             **p.client.describe(), "reasoning_effort": "high"}), CONTEXT, p.settings, **COLUMNS)
                   .key_for(sequences(), labels())),
    ],
)
def test_anything_that_would_change_the_answer_changes_the_key(label, change):
    p = proposer(reply(("whole_window_gc", GC_CODE)))
    assert change(p) != p.key_for(sequences(), labels())


def test_how_failed_calls_are_retried_and_waited_for_leaves_the_key_as_it_is():
    """Retries and the wait do not change a reply, so a run whose config changes them is answered
    from the proposals it made. The cell's record still states them."""
    def made(retries, timeout_seconds):
        client = ScriptedClient(reply(("whole_window_gc", GC_CODE)), described={
            "model": "scripted", "reasoning_effort": "minimal", "retries": retries,
            "timeout_seconds": timeout_seconds})
        return Proposer(client, CONTEXT, ProposalSettings(), **COLUMNS)

    assert made(5, 120).key_for(sequences(), labels()) == made(0, 600).key_for(sequences(), labels())
    assert made(0, 600).describe()["language_model"]["timeout_seconds"] == 600


def test_editing_the_shared_prompt_changes_the_key(monkeypatch):
    """The cache key must cover everything that would change a proposal, and the prompt wording is
    the only such input that is neither the data nor a setting. A cache lives inside one run
    directory, so only a resume can reach a stale entry, and a resume after a prompt edit is where
    an answer to the old prompt would go unnoticed."""
    from seqmaestro.proposer import prompt as prompt_module

    p = proposer(reply(("whole_window_gc", GC_CODE)))
    before = p.key_for(sequences(), labels())

    monkeypatch.setattr(prompt_module, "SYSTEM_PROMPT",
                        prompt_module.SYSTEM_PROMPT + " Prefer features about periodicity.")
    assert p.key_for(sequences(), labels()) != before


@pytest.mark.parametrize("edit", ["request", "guidance", "template", "example window", "rounding"])
def test_editing_any_part_of_a_request_changes_the_key(monkeypatch, edit):
    """The wording enters the key wherever it is written: in a constant, such as what the features
    are asked to do, how guidance is introduced or how wide the worked example's window is, or in
    the text a request's template assembles, such as the format of the answer. So does how values
    are rounded before they are checked, which decides which functions go back for repair."""
    from seqmaestro.proposer import prompt as prompt_module
    from seqmaestro.proposer import proposer as proposer_module

    p = proposer(reply(("whole_window_gc", GC_CODE)))
    before = p.key_for(sequences(), labels())
    if edit == "request":
        monkeypatch.setitem(prompt_module.REQUEST, BINARY, "tell the classes apart")
    elif edit == "guidance":
        monkeypatch.setattr(prompt_module, "GUIDANCE_INTRODUCTION", "Guidance:")
    elif edit == "template":
        monkeypatch.setattr(prompt_module, "_answer_format", lambda repair=False: ["Answer in JSON."])
    elif edit == "example window":
        monkeypatch.setattr(prompt_module, "EXAMPLE_WINDOW", prompt_module.EXAMPLE_WINDOW + 1)
    else:
        monkeypatch.setattr(proposer_module, "SIGNIFICANT_DIGITS", 6)
    assert p.key_for(sequences(), labels()) != before


def test_the_wording_digest_is_the_same_on_every_call():
    """The key is compared across processes and resumed runs, so the digest of unchanged wording
    must not vary."""
    from seqmaestro.proposer.prompt import wording_digest

    assert wording_digest() == wording_digest()


def test_the_shared_prompt_names_no_molecule():
    """Which nucleic acid a run is about belongs in the dataset's `sequence_layout`, because it
    differs between datasets and the shared prompt does not. Naming one here would contradict the
    description of every dataset of the other kind, and a language model told it is looking at DNA
    proposes fewer of the features that only make sense for RNA."""
    from seqmaestro.proposer.prompt import SYSTEM_PROMPT

    named = [molecule for molecule in ("DNA", "RNA") if molecule in SYSTEM_PROMPT]
    assert not named, f"the shared system prompt names {named}; that is the dataset's to say"


def with_covariates(rows: pd.Series, **columns) -> Inputs:
    """The rows with covariates beside them: `signal` by default, rising along the rows."""
    columns = columns or {"signal": np.arange(len(rows), dtype=float)}
    return Inputs(rows, pd.DataFrame(columns, index=rows.index))


def test_a_covariate_described_otherwise_asks_another_question():
    rows = with_covariates(sequences())
    keys = {Proposer(ScriptedClient("[]"), CONTEXT, covariates={"signal": text}, **COLUMNS)
            .key_for(rows, labels()) for text in ("a signal over the window", "a count of peaks")}
    assert len(keys) == 2


def test_rows_with_a_covariate_the_proposer_cannot_describe_are_refused():
    """A language model shown a bare column name would guess at what it measures."""
    with pytest.raises(ValueError, match="no description"):
        proposer(reply(("whole_window_gc", GC_CODE))).propose(with_covariates(sequences()), labels())


def test_the_model_is_told_the_covariates_and_shown_their_values():
    client = ScriptedClient(reply(("whole_window_gc", GC_CODE)))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4),
                 covariates=DESCRIBED, **COLUMNS)
    p.propose(with_covariates(sequences()), labels())
    user = client.seen[0][1]

    assert "Each row has these columns:" in user
    assert "'signal' (a number, 0 to 39 on the training rows): a signal measured over the window" in user
    assert re.search(r"\n  row 1: [ACGT]{12}   -> outcome=[01]\n", user)
    assert "\n  row,signal\n" in user
    assert re.search(r"\n  4,\d+\n", user)
    # Each name appears once, in the table's header.
    assert "signal=" not in user
    assert "and a column of numbers for each covariate listed above" in user
    assert p.describe()["covariates"] == DESCRIBED


def test_rows_of_different_lengths_are_asked_for_code_that_works_at_any_length():
    rng = np.random.default_rng(3)
    rows = pd.Series(["".join(rng.choice(list("ACGT"), size)) for size in (12, 30, 55, 80) * 10])
    client = ScriptedClient(reply(("whole_window_gc", GC_CODE)))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4),
                 **COLUMNS)
    p.propose(rows, labels())
    user = client.seen[0][1]

    assert "'window' (text, 12 to 80 characters, varying by row): the sequence" in user
    assert "the sequences differ in length, from 12 to 80 characters" in user
    assert "return NaN for that row instead of raising an error" in user
    assert "def gc_fraction_last_40nt(X):" in user and "gc_fraction_first_" not in user


def test_the_worked_example_reads_a_window_that_fits_the_rows_it_is_shown_with():
    """The worked example does not name a dataset. Its window is taken from the sequences' length,
    so the code a language model is shown is right for the rows beside it, and where the length
    varies or is not known, the example shown is the one that works at any length."""
    from seqmaestro.proposer.prompt import EXAMPLE_WINDOW, Example, example_window, render_request

    assert example_window(101) == EXAMPLE_WINDOW == 20
    assert (example_window(12), example_window(1)) == (6, 1)

    def shown(length) -> str:
        columns = None if length is None else {"sequence": {"name": SEQUENCE, "length": length},
                                               "covariates": []}
        return render_request(CONTEXT, n_features=1, examples=[Example("ACGTACGTACGT", 1)],
                              already_proposed=[], columns=columns, **COLUMNS)

    assert "def gc_fraction_first_20nt(X):" in shown(101) and ".str.slice(0, 20)" in shown(101)
    assert "def gc_fraction_first_6nt(X):" in shown(12) and ") / 6.0" in shown(12)
    for varying_or_unknown in ({"min": 12, "max": 80}, None):
        text = shown(varying_or_unknown)
        assert "def gc_fraction_last_40nt(X):" in text and "gc_fraction_first_" not in text


def test_a_long_example_is_shown_by_its_ends_with_its_length():
    from seqmaestro.proposer.prompt import shown_sequence

    sequence = "A" * 10 + "C" * 980 + "G" * 10
    assert shown_sequence(sequence, 1000) == sequence
    assert shown_sequence(sequence, 20) == "A" * 10 + "..." + "G" * 10 + \
        "  (1,000 characters, the middle not shown)"
    assert shown_sequence(sequence, None) == sequence


def test_editing_the_any_length_example_changes_the_key(monkeypatch):
    from seqmaestro.proposer import prompt as prompt_module

    p = proposer(reply(("whole_window_gc", GC_CODE)))
    before = p.key_for(sequences(), labels())
    monkeypatch.setattr(prompt_module, "CODE_CONTRACT_ANY_LENGTH",
                        prompt_module.CODE_CONTRACT_ANY_LENGTH + "\n")
    assert p.key_for(sequences(), labels()) != before


def test_covariates_enter_the_rows_digest_by_value_and_by_name():
    rows, y = sequences(), labels()
    measured = digest_rows(with_covariates(rows), y)
    assert measured != digest_rows(rows, y)
    assert digest_rows(with_covariates(rows, signal=np.arange(40.0) + 1), y) != measured
    assert digest_rows(with_covariates(rows, other=np.arange(40.0)), y) != measured


# --------------------------------------------------------------------------------------------
# The plain case: rows with no covariates, all one length
# --------------------------------------------------------------------------------------------

PLAIN_PROMPTS = Path(__file__).parent / "fixtures" / "plain_case_prompts.txt"
PLAIN_ROWS = pd.Series(["GGGCCCATGCGC", "GCGCGGCCATGG", "CCGGGCGCATCC", "GGCCGCGGCATG",
                        "AATTTAAGCTAT", "TATAAATTGCAA", "ATTAATATGCTT", "TTAATTAAGCAT"])
PLAIN_LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0])


def plain_case_prompts() -> dict[str, str]:
    """The fixture's sections by name. Each runs from its marker to the next, less the newline the
    fixture adds after it."""
    sections: dict[str, str] = {}
    name = None
    for line in PLAIN_PROMPTS.read_text().splitlines(keepends=True):
        marker = re.fullmatch(r"=== (\w+) ===\n", line)
        if marker:
            name = marker.group(1)
            sections[name] = ""
        elif name is not None:
            sections[name] += line
    return {name: text.removesuffix("\n") for name, text in sections.items()}


def test_the_plain_case_is_asked_as_the_fixture_records():
    """Rows with no covariates and one length are put to the language model byte for byte as the
    fixture records. What the prompt says about covariates and varying lengths renders only where
    it applies, so it cannot change the request that a run on such a dataset sends."""
    replies = iter([reply(("gc_share", "def gc_share(X):\n    return X['window'].str.count('G')")),
                    reply(("at_share", "def at_share(X):\n    return X['window'].str.count('A')")),
                    reply(("t_share", "def t_share(X):\n    return X['window'].str.count('T')"))])
    client = ScriptedClient(lambda system, user: next(replies))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=2, n_per_round=3, n_examples=4),
                 seed=11, **COLUMNS)
    p.propose(PLAIN_ROWS, PLAIN_LABELS)
    p.settings = ProposalSettings(n_rounds=1, n_per_round=3, n_examples=4)
    p.propose(PLAIN_ROWS, PLAIN_LABELS, feedback={"auroc": 0.71, "unused": ["at_share"]})

    expected = plain_case_prompts()
    (system, first), (_, second), (_, informed) = client.seen
    assert system == expected["system"]
    assert first == expected["blind_round_1"]
    assert second == expected["blind_round_2"]
    assert informed == expected["informed_round"]


def test_the_sequences_alone_digest_as_rows_with_no_covariates_do():
    assert digest_rows(PLAIN_ROWS, PLAIN_LABELS) == \
        digest_rows(Inputs.from_sequences(PLAIN_ROWS), PLAIN_LABELS)


def test_caching_is_off_by_default():
    """In a research repository, a cache nobody asked for is likely to serve a stale result."""
    p = proposer(reply(("whole_window_gc", GC_CODE)))
    assert p.cache.enabled is False
    assert p.cache.get("anything") is None


# --------------------------------------------------------------------------------------------
# GeneratedFeaturiser: executing what was proposed
# --------------------------------------------------------------------------------------------


def test_fit_transform_produces_a_named_matrix():
    featuriser = GeneratedFeaturiser(proposer(reply(("whole_window_gc", GC_CODE), ("adenine_count", A_COUNT_CODE))))
    rows = sequences()
    matrix = featuriser.fit_transform(rows, labels())

    assert list(matrix.columns) == ["whole_window_gc", "adenine_count"]
    assert matrix.shape == (40, 2)
    # Compared with a direct computation instead of a literal, so the fixture can change without
    # the test quietly becoming an assertion about the fixture.
    assert matrix["adenine_count"].tolist() == [sequence.count("A") for sequence in rows]
    assert matrix["whole_window_gc"].tolist() == pytest.approx(
        [(sequence.count("G") + sequence.count("C")) / 12 for sequence in rows]
    )


def test_the_model_is_called_at_fit_and_never_at_transform():
    """The split discipline depends on this, because validation and test rows must not reach the
    language model. A call per transform would also make the arm's cost unbounded."""
    client = ScriptedClient(reply(("whole_window_gc", GC_CODE)))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4),
                 **COLUMNS)
    featuriser = GeneratedFeaturiser(p)

    featuriser.fit(sequences(), labels())
    calls_after_fit = len(client.seen)
    featuriser.transform(sequences(10))
    featuriser.transform(sequences(10))

    assert calls_after_fit == 1
    assert len(client.seen) == 1


def test_a_feature_that_raises_on_the_training_rows_is_dropped():
    exploding = "def boom(X):\n    return X['window'].apply(lambda s: 1 / 0)"
    featuriser = GeneratedFeaturiser(proposer(reply(("boom", exploding), ("whole_window_gc", GC_CODE))))
    featuriser.fit(sequences(), labels())

    assert featuriser.feature_names == ["whole_window_gc"]
    assert featuriser.fit_report["dropped"]["raised"] == 1


def test_a_constant_feature_is_dropped():
    """A constant feature cannot help a model, and the stability estimator would match it to
    anything."""
    constant = "def always_seven(X):\n    return np.full(len(X), 7.0)"
    featuriser = GeneratedFeaturiser(proposer(reply(("always_seven", constant), ("whole_window_gc", GC_CODE))))
    featuriser.fit(sequences(), labels())

    assert featuriser.feature_names == ["whole_window_gc"]
    assert featuriser.fit_report["dropped"]["constant"] == 1


# A value as two computations can return it: every other row has a relative error of a few units
# in the last place, as a sum taken in a different order would leave.
NOISY_GC_CODE = (
    "def noisy_gc_tenths(X):\n"
    "    seqs = X['window'].astype(str)\n"
    "    gc = (seqs.str.count('G') + seqs.str.count('C')).to_numpy()\n"
    "    return (gc / 10) * (1 + np.where(np.arange(len(gc)) % 2 == 0, 4e-16, 0.0))"
)


def test_values_that_differ_only_in_their_last_bits_are_one_value():
    """Values are rounded to significant digits. Without that, every exact comparison downstream
    (a rank, a count of distinct values, a cut point) would treat 0.2 and 0.20000000000000004 as
    two values."""
    gc = sequences().str.count("G") + sequences().str.count("C")
    rounded = GeneratedFeaturiser(proposer(reply(("noisy_gc_tenths", NOISY_GC_CODE))))
    raw = GeneratedFeaturiser(proposer(reply(("noisy_gc_tenths", NOISY_GC_CODE))),
                              significant_digits=None)
    kept = rounded.fit_transform(sequences(), labels())["noisy_gc_tenths"].to_numpy()
    noisy = raw.fit_transform(sequences(), labels())["noisy_gc_tenths"].to_numpy()

    assert len(np.unique(noisy)) > len(np.unique(gc)), "the fixture must hold the noise"
    assert len(np.unique(kept)) == len(np.unique(gc))
    assert np.array_equal(kept, (gc / 10).to_numpy())


def test_a_feature_constant_but_for_its_noise_is_dropped_as_constant():
    noisy_seven = ("def noisy_seven(X):\n"
                   "    return 7.0 * (1 + np.where(np.arange(len(X)) % 2 == 0, 4e-16, 0.0))")
    features = reply(("noisy_seven", noisy_seven), ("whole_window_gc", GC_CODE))
    featuriser = GeneratedFeaturiser(proposer(features))
    featuriser.fit(sequences(), labels())
    assert featuriser.feature_names == ["whole_window_gc"]
    assert featuriser.fit_report["dropped"]["constant"] == 1

    raw = GeneratedFeaturiser(proposer(features), significant_digits=None)
    raw.fit(sequences(), labels())
    assert "noisy_seven" in raw.feature_names, "without rounding the noise passes for variation"


def test_the_precision_is_recorded_and_kept_through_a_save():
    featuriser = GeneratedFeaturiser(proposer(reply(("noisy_gc_tenths", NOISY_GC_CODE))),
                                     significant_digits=9)
    featuriser.fit(sequences(), labels())
    assert featuriser.describe()["significant_digits"] == 9

    restored = pickle.loads(pickle.dumps(featuriser))
    assert restored.significant_digits == 9
    assert restored.transform(sequences()).equals(featuriser.transform(sequences()))

    with pytest.raises(ValueError, match="positive integer"):
        GeneratedFeaturiser(proposer(reply(("whole_window_gc", GC_CODE))), significant_digits=0)


def test_rounding_to_significant_digits():
    values = np.array([0.123456789012345, -98765.4321098765, 3e-300, 0.0, -0.0, np.nan, np.inf,
                       0.30000000000000004, 1e300])
    rounded = to_significant_digits(values, 12)
    assert rounded[0] == 0.123456789012 and rounded[1] == -98765.4321099
    assert rounded[2] == 3e-300 and rounded[7] == 0.3 and rounded[8] == 1e300
    assert rounded[3] == 0 and np.isnan(rounded[5]) and rounded[6] == np.inf
    assert np.array_equal(to_significant_digits(rounded, 12), rounded, equal_nan=True)


def test_a_scalar_return_is_refused_rather_than_broadcast():
    """Broadcasting a scalar would hide a feature with no information behind a real-looking column."""
    scalar = "def just_one(X):\n    return 1.0"
    featuriser = GeneratedFeaturiser(proposer(reply(("just_one", scalar), ("whole_window_gc", GC_CODE))))
    featuriser.fit(sequences(), labels())

    assert featuriser.feature_names == ["whole_window_gc"]
    assert featuriser.fit_report["dropped"]["wrong_shape"] == 1


def test_nothing_surviving_is_an_error_rather_than_an_empty_arm():
    exploding = "def boom(X):\n    return X['window'].apply(lambda s: 1 / 0)"
    featuriser = GeneratedFeaturiser(proposer(reply(("boom", exploding))))

    with pytest.raises(RuntimeError, match="No proposed feature survived"):
        featuriser.fit(sequences(), labels())


def test_a_transform_time_failure_is_filled_with_the_fit_time_median_and_counted():
    """A feature that fails at transform time cannot be dropped, because the model is already
    fitted on its column."""
    # Works on 12-nt sequences, raises on anything shorter.
    fragile = (
        "def strict_g_count(X):\n"
        "    def check(s):\n"
        "        if len(s) < 12:\n"
        "            raise ValueError('too short')\n"
        "        return float(s.count('G'))\n"
        "    return X['window'].apply(check)"
    )
    rows = sequences()
    featuriser = GeneratedFeaturiser(proposer(reply(("strict_g_count", fragile))))
    featuriser.fit(rows, labels())

    values = featuriser.transform(pd.Series(["ACGT", "ACGT"]))
    expected = float(np.median([sequence.count("G") for sequence in rows]))

    assert featuriser.transform_failures == {"strict_g_count": 1}, "one failed call, not one per row"
    assert values["strict_g_count"].tolist() == [expected, expected], "the fit-time median"


def test_non_finite_values_are_filled_rather_than_handed_to_a_model():
    """A NaN that reaches XGBoost silently gives a different model, and one that reaches an EBM
    raises an exception."""
    sometimes_nan = (
        "def ratio(X):\n"
        "    seqs = X['window'].astype(str)\n"
        "    return seqs.str.count('G') / seqs.str.count('T')"
    )
    featuriser = GeneratedFeaturiser(proposer(reply(("ratio", sometimes_nan))))
    featuriser.fit(sequences(), labels())

    values = featuriser.transform(pd.Series(["GGGGGGGGGGGG", "TTTTTTTTTTTT"]))
    assert np.isfinite(values["ratio"]).all()


# --------------------------------------------------------------------------------------------
# The repair round and the undefined-value tolerance
# --------------------------------------------------------------------------------------------

DIVIDES_BY_T = ("def g_per_t(X):\n    s = X['window']\n"
                "    return s.str.count('G') / s.str.count('T').replace(0, np.nan)")
FIXED = "def g_per_t(X):\n    s = X['window']\n    return s.str.count('G') / (s.str.count('T') + 1)"
RAISES = "def g_per_t(X):\n    return X['window'].str.count('G') / 0 + X['missing_column']"


def repairing_client(first: str, corrected: str) -> ScriptedClient:
    """A client that answers a proposal request with `first` and a repair request with `corrected`."""
    return ScriptedClient(lambda system, user: reply(
        ("g_per_t", corrected if "failed when they were run" in user else first)))


def test_a_candidate_that_fails_on_the_rows_goes_back_once_and_is_kept_when_corrected():
    client = repairing_client(RAISES, FIXED)
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4),
                 **COLUMNS)
    featuriser = GeneratedFeaturiser(p)
    featuriser.fit(sequences(), labels())

    assert featuriser.feature_names == ["g_per_t"]
    record = featuriser.records[0]
    assert record.source == FIXED and record.extra["repaired"] is True
    assert (p.stats.repair_calls, p.stats.repaired, p.stats.unrepaired) == (1, 1, 0)
    repair = client.seen[1][1]
    assert "Function: g_per_t\nWhat went wrong: it raised KeyError" in repair and RAISES in repair
    assert "Requirements for the code:" in repair, "the repair is asked under the same rules"


def test_a_candidate_still_failing_stays_as_proposed_and_the_featuriser_drops_it():
    client = repairing_client(RAISES, RAISES)
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4,
                                                            repair_rounds=2), **COLUMNS)
    with pytest.raises(RuntimeError, match="No proposed feature survived"):
        GeneratedFeaturiser(p).fit(sequences(), labels())
    assert (p.stats.repair_calls, p.stats.repaired, p.stats.unrepaired) == (2, 0, 1)
    assert len(client.seen) == 3, "one proposal call and two repair calls"


CONSTANT_CODE = "def g_per_t(X):\n    return X['window'].str.len() * 1.0"
G_COUNT = "def g_count(X):\n    return X['window'].str.count('G').astype(float)"
UNSAFE = "import os\n\ndef g_per_t(X):\n    return X['window'].str.len() * os.getpid()"


def answering(*codes: str) -> ScriptedClient:
    """A client that answers each call in turn with the next version of `g_per_t`."""
    replies = iter(codes)
    return ScriptedClient(lambda system, user: reply(("g_per_t", next(replies))))


def test_a_second_repair_is_sent_the_code_last_tried_with_its_own_failure():
    client = answering(RAISES, CONSTANT_CODE, FIXED)
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4,
                                                            repair_rounds=2), **COLUMNS)
    [record] = p.propose(sequences(), labels())

    assert record.source == FIXED and record.extra["repaired"] is True
    assert (p.stats.repair_calls, p.stats.repaired, p.stats.unrepaired) == (2, 1, 0)
    second = client.seen[2][1]
    assert CONSTANT_CODE in second and "the same value on every row" in second
    assert RAISES not in second, "the first code is not sent with the second code's failure"


def test_a_correction_the_guard_refuses_leaves_the_code_before_it_with_both_failures():
    client = answering(RAISES, UNSAFE, FIXED)
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4,
                                                            repair_rounds=2), **COLUMNS)
    [record] = p.propose(sequences(), labels())

    assert record.source == FIXED
    second = client.seen[2][1]
    assert RAISES in second and UNSAFE not in second
    assert "raised KeyError" in second and "its correction was refused" in second


def correction(replaces: str, name: str, code: str, description: str) -> str:
    """A reply to a repair request: one corrected function, naming the function it replaces."""
    return json.dumps([{"replaces": replaces, "name": name, "description": description,
                        "rationale": "because", "code": code}])


def test_the_request_says_how_to_correct_each_kind_of_failure_it_sends():
    """A code fault is fixed without changing what the feature measures, whereas a constant feature
    has to measure something a little different, under a name for it. Only the failures sent are
    explained, and every entry of the answer names the function it replaces."""
    code_fault = render_repair_request(
        CONTEXT, [("g_per_t", RAISES, Check(RAISED, "raised KeyError: 'missing_column'"))],
        sequence_column=SEQUENCE)
    constant = render_repair_request(
        CONTEXT, [("g_per_t", CONSTANT_CODE, Check(CONSTANT, "returned the same value on every row"))],
        sequence_column=SEQUENCE)
    assert "the code is at fault" in code_fault and "cannot tell one row" not in code_fault
    assert "cannot tell one row" in constant and "the code is at fault" not in constant
    for request in (code_fault, constant):
        assert "needs a new name and a description" in request
        assert '"replaces": "<the name of the function this entry corrects>"' in request


def test_a_constant_feature_comes_back_under_a_name_for_what_it_now_measures():
    replies = iter([reply(("g_per_t", CONSTANT_CODE), ("whole_window_gc", GC_CODE)),
                    correction("g_per_t", "g_count", G_COUNT, "the number of G in the sequence")])
    client = ScriptedClient(lambda system, user: next(replies))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4),
                 **COLUMNS)
    records = p.propose(sequences(), labels())

    assert [record.name for record in records] == ["g_count", "whole_window_gc"], "in its place"
    renamed = records[0]
    assert (renamed.source, renamed.description) == (G_COUNT, "the number of G in the sequence")
    assert renamed.extra == {"repaired": True, "renamed_from": "g_per_t"}
    assert (p.stats.repaired, p.stats.unrepaired) == (1, 0)


@pytest.mark.parametrize(("name", "description", "told"), [
    ("9_g", "the number of G", "which is not a legal Python identifier"),
    ("whole_window_gc", "the number of G", "which another feature already has"),
    ("g_count", "", "with no description of what it now measures"),
])
def test_a_correction_whose_new_name_cannot_be_kept_goes_back_with_the_reason(name, description,
                                                                               told):
    replies = iter([reply(("g_per_t", CONSTANT_CODE), ("whole_window_gc", GC_CODE)),
                    correction("g_per_t", name, G_COUNT, description),
                    correction("g_per_t", "g_count", G_COUNT, "the number of G")])
    client = ScriptedClient(lambda system, user: next(replies))
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4,
                                                            repair_rounds=2), **COLUMNS)
    records = p.propose(sequences(), labels())

    second = client.seen[2][1]
    assert told in second and CONSTANT_CODE in second, "the code before it, with both reasons"
    assert [record.name for record in records] == ["g_count", "whole_window_gc"]


def test_no_repair_is_asked_for_when_repair_rounds_is_zero():
    client = repairing_client(RAISES, FIXED)
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4,
                                                            repair_rounds=0), **COLUMNS)
    p.propose(sequences(), labels())
    assert len(client.seen) == 1 and p.stats.unrepaired == 1


def test_a_feature_undefined_on_a_few_rows_is_kept_and_says_how_many():
    """Below the tolerance, a feature is kept and filled with its median where it is undefined, and
    its record states on what share of the training rows. Above the tolerance, it is dropped as
    undefined."""
    code = ("def gc_mostly(X):\n    counts = X['window'].str.count('G').astype(float)\n"
            "    return counts.where(np.arange(len(X)) >= 4)")
    lenient = ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4, max_undefined_share=0.2)
    kept = GeneratedFeaturiser(proposer(reply(("gc_mostly", code)), settings=lenient))
    kept.fit(sequences(), labels())
    assert kept.records[0].extra["undefined_share"] == 0.1
    assert kept.max_undefined_share == 0.2, "taken from the proposer's settings"

    strict = ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4, max_undefined_share=0.05)
    dropped = GeneratedFeaturiser(proposer(reply(("gc_mostly", code), ("whole_window_gc", GC_CODE)),
                                           settings=strict))
    dropped.fit(sequences(), labels())
    assert dropped.feature_names == ["whole_window_gc"]
    assert dropped.fit_report["dropped"]["undefined"] == 1


def test_undefined_values_filled_at_transform_are_counted():
    featuriser = GeneratedFeaturiser(proposer(reply(("g_per_t", DIVIDES_BY_T))))
    featuriser.fit(sequences(), labels())
    values = featuriser.transform(pd.Series(["GGGGGGGGGGGG", "GGGGGGGGGGGT"]))
    assert np.isfinite(values["g_per_t"]).all()
    assert featuriser.transform_undefined == {"g_per_t": 1}
    assert featuriser.describe()["transform_undefined"] == {"g_per_t": 1}


def test_generated_code_reads_the_covariates_by_name():
    code = "def signal_plus_gc(X):\n    return X['signal'] + X['window'].str.count('G')"
    rows = with_covariates(sequences())
    featuriser = GeneratedFeaturiser(proposer(reply(("signal_plus_gc", code)), covariates=DESCRIBED))
    values = featuriser.fit_transform(rows, labels())["signal_plus_gc"]

    expected = rows.covariates["signal"] + rows.sequences.str.count("G")
    assert values.tolist() == pytest.approx(expected.tolist())
    assert featuriser.covariate_columns == ("signal",)
    assert featuriser.describe()["covariate_columns"] == ["signal"]


def test_a_fitted_featuriser_picks_its_own_covariates_and_refuses_rows_without_them():
    """A saved arm is handed every covariate the dataset declares. It uses the ones it was fitted
    with, and rows that lack one are refused instead of being evaluated against a different
    frame."""
    code = "def signal_twice(X):\n    return X['signal'] * 2 + X['window'].str.len() * 0"
    featuriser = GeneratedFeaturiser(proposer(reply(("signal_twice", code)), covariates=DESCRIBED))
    featuriser.fit(with_covariates(sequences()), labels())

    wider = with_covariates(sequences(10), signal=np.arange(10.0), noise=np.ones(10))
    assert featuriser.transform(wider)["signal_twice"].tolist() == [2.0 * i for i in range(10)]
    with pytest.raises(ValueError, match="no covariate signal"):
        featuriser.transform(sequences(10))


def test_from_records_needs_no_proposer_and_no_network():
    """Re-scoring a finished run, or running a test, must not call a language model."""
    records = proposer(reply(("whole_window_gc", GC_CODE))).propose(sequences(), labels())
    featuriser = GeneratedFeaturiser.from_records(records, sequence_column=SEQUENCE)

    matrix = featuriser.fit_transform(sequences(), labels())
    assert list(matrix.columns) == ["whole_window_gc"]
    assert "proposer" not in featuriser.describe()


def test_records_alone_do_not_say_which_column_their_code_indexes():
    """A record holds the code but not the frame it was written against, so the sequence column
    must be given."""
    records = proposer(reply(("whole_window_gc", GC_CODE))).propose(sequences(), labels())
    with pytest.raises(ValueError, match="sequence_column"):
        GeneratedFeaturiser.from_records(records)


def test_a_featuriser_takes_the_column_from_its_proposer_rather_than_being_told_twice():
    """The column is stated once, on the proposer, because two statements of it could
    disagree."""
    p = proposer(reply(("whole_window_gc", GC_CODE)), sequence_column="somewhere_else")
    assert GeneratedFeaturiser(p).sequence_column == "somewhere_else"


def test_describe_records_the_conditions_that_produced_the_features():
    featuriser = GeneratedFeaturiser(proposer(reply(("whole_window_gc", GC_CODE))))
    featuriser.fit(sequences(), labels())
    described = featuriser.describe()

    assert described["kind"] == "llm_generated"
    assert described["guard"]["enabled"] is True
    assert described["proposer"]["settings"]["n_rounds"] == 1
    assert described["proposer"]["context"]["task"] == CONTEXT.task
    assert described["fit_report"]["n_kept"] == 1


# --------------------------------------------------------------------------------------------
# End to end: a language-model arm through the harness, like any other arm
# --------------------------------------------------------------------------------------------


def planted_split(n, seed):
    """Sequences where GC content drives the label, so that a model on a proposed GC feature can
    learn it."""
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for _ in range(n):
        positive = rng.random() < 0.5
        pool = list("GGGCCCAT") if positive else list("AAATTTGC")
        rows.append("".join(rng.choice(pool, 12)))
        y.append(int(positive))
    return pd.DataFrame({SEQUENCE: rows, TARGET: y})


def test_an_arm_on_a_written_source_writes_the_same_cell_as_every_other_arm(tmp_path):
    """A proposer arm is a fixed-vocabulary arm with a different featuriser passed to `run_model`.

    Whatever reads a cell's directory (the stability estimator, the judge, the results table) must
    not be able to tell that a language model was involved.
    """
    from seqmaestro.models import make_model

    train, val = planted_split(300, 0), planted_split(120, 1)
    featuriser = GeneratedFeaturiser(
        proposer(reply(("whole_window_gc", GC_CODE), ("adenine_count", A_COUNT_CODE)))
    )
    out = tmp_path / "cell"

    run = run_cell(
        make_model("l1_logistic", target_kind="binary", seed=42), featuriser, train, val,
        columns=SPLIT_COLUMNS,
        seed=42, out_dir=out, n_bootstrap=0,
        arm=("l1_logistic", "llm_generated"),
    )

    assert {p.name for p in out.iterdir()} == {
        "run.json", "metrics.json", "predictions.npz", "features.json", "feature_values.npz",
        "fitted.pkl",
    }
    assert run["metrics"]["val"]["auroc"] > 0.8, "GC drives the label, so a GC feature should work"

    records, header = load_features(out / "features.json")
    assert {r.kind for r in records} == {KIND_LLM_GENERATED}
    assert all(r.source for r in records), "the code travels with the record, for the judge"
    assert all(r.importance is not None for r in records)
    assert header["featuriser"]["proposer"]["settings"]["n_rounds"] == 1


def test_a_permuted_cell_proposes_from_the_permuted_labels(tmp_path):
    """A permuted cell supplies the noise pool the judge scores against. If the proposer saw the
    real labels while the model fitted permuted ones, the control would be handed genuine
    features."""
    train, val = planted_split(300, 0), planted_split(120, 1)
    seen: list[np.ndarray] = []

    class Recording:
        model = "recording"
        # A proposer declares the column its generated code will index, and the featuriser reads it
        # from here, so the column is stated only once.
        sequence_column = SEQUENCE

        def propose(self, inputs, y=None):
            seen.append(np.asarray(y))
            return proposer(reply(("whole_window_gc", GC_CODE))).propose(inputs, y)

        def describe(self):
            return {"kind": "recording"}

        def usage(self):
            return {}

    from seqmaestro.models import make_model

    for permute in (False, True):
        run_cell(
            make_model("l1_logistic", target_kind="binary", seed=42), GeneratedFeaturiser(Recording()),
            train, val, columns=SPLIT_COLUMNS,
            seed=42, permute_labels=permute, n_bootstrap=0,
        )

    real, permuted = seen
    assert real.tolist() != permuted.tolist(), "the proposer saw the labels the model trains on"
    assert sorted(real) == sorted(permuted), "a permutation, not different rows"


# --------------------------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, text",
    [
        ("bare array", reply(("whole_window_gc", GC_CODE))),
        ("fenced", f"```json\n{reply(("whole_window_gc", GC_CODE))}\n```"),
        ("with prose around it", f"Here you go:\n{reply(("whole_window_gc", GC_CODE))}\nHope that helps."),
    ],
)
def test_the_json_is_found_through_the_usual_wrappings(label, text):
    proposals, error = parse_proposals(text)
    assert [p.name for p in proposals] == ["whole_window_gc"]
    assert error is None


def test_an_entry_missing_its_code_is_dropped_rather_than_repaired():
    text = json.dumps([{"name": "no_code", "description": "d"}, {"name": "ok", "code": GC_CODE}])
    proposals, error = parse_proposals(text)

    assert [p.name for p in proposals] == ["ok"]
    assert "1 entries lacked" in error


# --------------------------------------------------------------------------------------------
# The time limit on a generated feature
# --------------------------------------------------------------------------------------------

#: A feature that loops on every row and catches any error each row raises, as generated code
#: handling its own edge cases does. A limit that code could catch would stop one row and let the
#: others run. The loop is bounded, so that such a failure costs the test seconds instead of the
#: rest of the session.
SLOW_CODE = (
    "def g_per_t(X):\n"
    "    values = []\n"
    "    for s in X['window']:\n"
    "        n = 0\n"
    "        try:\n"
    "            for _ in range(10 ** 6):\n"
    "                n += 1\n"
    "        except Exception:\n"
    "            n = -1\n"
    "        values.append(float(n))\n"
    "    return values"
)
#: `SLOW_CODE` with a bare `except:`, which catches any exception at all.
SLOW_BARE_EXCEPT = SLOW_CODE.replace("except Exception:", "except:")
#: A feature that is fast on the rows it is fitted on and slow on any sequence longer than 20.
SLOW_ON_LONG = (
    "def g_per_t(X):\n"
    "    values = []\n"
    "    for s in X['window']:\n"
    "        n = 0\n"
    "        if len(s) > 20:\n"
    "            for _ in range(10 ** 8):\n"
    "                n += 1\n"
    "        values.append(float(s.count('G') + n))\n"
    "    return values"
)
LIMIT = 0.2


def windows(n=40) -> pd.DataFrame:
    return pd.DataFrame({SEQUENCE: sequences(n)})


@pytest.mark.parametrize("code", [SLOW_CODE, SLOW_BARE_EXCEPT], ids=["except Exception", "bare except"])
def test_a_feature_past_its_limit_fails_as_slow_even_where_it_catches_every_error(code):
    started = time.perf_counter()
    checked = check(compile_feature_function(code), windows(), max_undefined_share=0.2,
                    significant_digits=None, max_seconds=LIMIT)

    assert checked.failure == SLOW
    assert checked.detail == f"did not finish within {LIMIT:g} seconds on these 40 rows"
    assert time.perf_counter() - started < 2
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "the timer is cancelled"
    assert signal.getsignal(signal.SIGALRM) is signal.SIG_DFL, "the handler before it is restored"


def test_a_bare_except_still_catches_the_features_own_errors():
    code = ("def g_per_t(X):\n"
            "    values = []\n"
            "    for s in X['window']:\n"
            "        try:\n"
            "            values.append(s.count('G') / s.count('T'))\n"
            "        except:\n"
            "            values.append(0.0)\n"
            "    return values")
    assert compile_feature_function(code)(pd.DataFrame({SEQUENCE: ["GGG", "GT"]})) == [0.0, 1.0]


def test_a_feature_too_slow_goes_back_to_the_model_and_its_fast_correction_is_kept():
    client = repairing_client(SLOW_CODE, FIXED)
    p = Proposer(client, CONTEXT, settings=ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4,
                                                            max_seconds=LIMIT), **COLUMNS)
    featuriser = GeneratedFeaturiser(p)
    featuriser.fit(sequences(), labels())

    assert featuriser.records[0].source == FIXED and featuriser.records[0].extra["repaired"] is True
    repair = client.seen[1][1]
    assert f"it did not finish within {LIMIT:g} seconds on these 40 rows" in repair
    assert "grows in proportion to the length of a sequence" in repair
    assert featuriser.describe()["max_seconds"] == LIMIT, "taken from the proposer's settings"


def test_a_feature_that_runs_out_of_time_at_fit_is_dropped_and_counted():
    settings = ProposalSettings(n_rounds=1, n_per_round=2, n_examples=4, repair_rounds=0,
                                max_seconds=LIMIT)
    featuriser = GeneratedFeaturiser(proposer(reply(("g_per_t", SLOW_CODE), ("g_count", G_COUNT)),
                                              settings=settings))
    featuriser.fit(sequences(), labels())

    assert featuriser.feature_names == ["g_count"]
    assert featuriser.fit_report["dropped"][SLOW] == 1


def test_a_feature_out_of_time_at_transform_is_filled_with_its_median_and_counted():
    """The feature finished on the rows it was fitted on, so it is in the model and cannot be
    dropped."""
    settings = ProposalSettings(n_rounds=1, n_per_round=1, n_examples=4, max_seconds=LIMIT)
    rows = sequences()
    featuriser = GeneratedFeaturiser(proposer(reply(("g_per_t", SLOW_ON_LONG)), settings=settings))
    featuriser.fit(rows, labels())

    values = featuriser.transform(pd.Series(["G" * 30, "GGGGAAAACCCC"]))
    expected = float(np.median([sequence.count("G") for sequence in rows]))
    assert featuriser.transform_failures == {"g_per_t": 1}
    assert values["g_per_t"].tolist() == [expected, expected]
    assert featuriser.transform(pd.Series(["GGGGAAAACCCC"]))["g_per_t"].tolist() == [4.0]


def test_the_limit_grows_with_the_rows_and_never_falls_below_the_limit_checked_under():
    assert scaled_limit(60, checked_rows=1_000, rows=5_000) == 300
    assert scaled_limit(60, checked_rows=1_000, rows=10) == 60
    assert scaled_limit(None, checked_rows=1_000, rows=5_000) is None
    assert scaled_limit(60, checked_rows=None, rows=5_000) is None, "a feature fitted before limits"


def test_a_limit_of_zero_is_refused_since_the_timer_would_read_it_as_none():
    with pytest.raises(TimeLimitUnavailable, match="above 0"), time_limit(0):
        pass
    with pytest.raises(ValueError, match="above 0"):
        ProposalSettings(max_seconds=0)


def test_off_the_main_thread_the_limit_is_refused_and_not_blamed_on_the_feature():
    function = compile_feature_function(G_COUNT)
    caught = []

    def work():
        try:
            check(function, windows(), max_undefined_share=0.2, significant_digits=None,
                  max_seconds=5)
        except TimeLimitUnavailable as error:
            caught.append(error)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert len(caught) == 1 and "main thread" in str(caught[0])
