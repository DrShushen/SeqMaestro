"""The function term language: terms written as Python functions of the rows.

The language model is replaced by scripted replies at the point where the agent would call it, so
the agent's loop, the parser, the screening, the repair requests and the search all run as they do
live. Both input shapes are covered: a numeric table alone, and text sequences of different
lengths beside a numeric column.
"""

import json
import signal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from igsr.agent.agent import LiteLLMAgent
from igsr.code_guard import UnsafeCodeError, compile_feature_function
from igsr.compose import compose_config
from igsr.dataset.data_bundle import DataBundle
from igsr.method.fitted_model import FittedModel
from igsr.method.function_prompts import describe_columns, parse_function_terms, shown
from igsr.method.function_terms import (
    CONSTANT,
    NAMING,
    RAISED,
    REFUSED,
    UNDEFINED,
    WRONG_SHAPE,
    FunctionTermRegistry,
    FunctionTermRows,
    FunctionTermSettings,
    Proposal,
    call_with_timeout,
    normalised_source,
    screen,
)
from igsr.method.igsr import igsr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides) -> FunctionTermSettings:
    values = dict(max_undefined_share=0.2, repair_rounds=1, n_examples=6, max_example_length=12, timeout_seconds=5)
    values.update(overrides)
    return FunctionTermSettings(**values)


def _cfg(*overrides):
    """The copy's own configuration in the function form, one plain iteration, pruned by influence."""
    return compose_config([
        "llm=seqmaestro_azure", "llm.deployment=dummy", "llm.endpoint=https://dummy.invalid",
        "llm.api_key=dummy-key", "llm.api_version=2025-04-01-preview",
        "experiment.term_language=function", "experiment.tree=false", "experiment.n_iters=1",
        "experiment.algorithmic_pruning=true", "experiment.early_stop_enabled=false",
        "experiment.retry_attempts=1", "experiment.function_terms.n_examples=6",
        "experiment.function_terms.max_example_length=12", *overrides,
    ])


def _table_bundle(seed=0) -> DataBundle:
    """A numeric table alone: y = 2 x1 + 3 sin(x2)."""
    rng = np.random.default_rng(seed)

    def make(n):
        x1, x2 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": 2.0 * x1 + 3.0 * np.sin(x2)})

    return DataBundle(name="toy_table", data_settings=None, dataset_train=make(120), dataset_validation=make(40),
                      dataset_test=make(40), target_columns=["y"], data_dictionary="A toy numeric table.")


def _sequence_bundle(seed=0) -> DataBundle:
    """Nucleotide sequences of 8 to 30 characters beside a numeric covariate:
    y = 1.5 * (occurrences of GC) + 0.5 * c."""
    rng = np.random.default_rng(seed)

    def make(n):
        sequences = ["".join(rng.choice(list("ACGT"), size=length)) for length in rng.integers(8, 31, n)]
        c = rng.normal(size=n)
        gc = np.array([sequence.count("GC") for sequence in sequences])
        return pd.DataFrame({"seq": sequences, "c": c, "y": 1.5 * gc + 0.5 * c})

    return DataBundle(name="toy_sequences", data_settings=None, dataset_train=make(150),
                      dataset_validation=make(60), dataset_test=make(60), target_columns=["y"],
                      data_dictionary="Toy nucleotide sequences with one covariate.",
                      column_descriptions={"seq": "a nucleotide sequence", "c": "a measured covariate"})


def _term(name, code, description="what it measures", replaces=None):
    entry = {"name": name, "description": description, "rationale": "why it might predict", "code": code}
    return {**entry, "replaces": replaces} if replaces else entry


def _reply(*terms) -> str:
    return "TERMS:\n```json\n" + json.dumps(list(terms)) + "\n```"


def _script(monkeypatch, replies):
    """Replace the language model with `replies`, in turn. Returns the messages it was sent."""
    sent = []
    replies = iter(replies)

    def complete(self, messages, completion_kwargs):
        sent.append(messages[-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))],
                               usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 3, "completion_tokens": 2}))

    monkeypatch.setattr(LiteLLMAgent, "_create_chat_completion", complete)
    return sent


GC_COUNT = 'def gc_count(X):\n    return X["seq"].str.count("GC")'
COVARIATE = 'def covariate(X):\n    return X["c"]'


def _training_rows(frame, settings=None):
    registry = FunctionTermRegistry(settings or _settings())
    return registry, FunctionTermRows("train", frame, registry)


# ---------------------------------------------------------------------------
# The two input shapes, end to end
# ---------------------------------------------------------------------------


def test_a_table_alone_is_fitted_through_function_terms(monkeypatch):
    _script(monkeypatch, [_reply(_term("x1_itself", 'def x1_itself(X):\n    return X["x1"]'),
                                 _term("sin_x2", 'def sin_x2(X):\n    return np.sin(X["x2"])'))])
    bundle = _table_bundle()
    result = igsr(_cfg(), seed=0, data_bundle=bundle)

    assert set(result["best_terms"]) == {"x1_itself", "sin_x2"} and result["best_metric"] < 1e-12
    rows = bundle.dataset_test[["x1", "x2"]]
    np.testing.assert_allclose(result["best_model"].predict(rows).ravel(), bundle.dataset_test["y"], atol=1e-9)


def test_sequences_of_different_lengths_beside_a_table_are_read_whole(monkeypatch):
    sent = _script(monkeypatch, [_reply(_term("gc_count", GC_COUNT), _term("covariate", COVARIATE))])
    bundle = _sequence_bundle()
    result = igsr(_cfg(), seed=0, data_bundle=bundle)

    assert set(result["best_terms"]) == {"gc_count", "covariate"} and result["best_metric"] < 1e-12
    lengths = bundle.dataset_train["seq"].str.len()
    prompt = sent[0]
    assert f"from {lengths.min()} to {lengths.max()} characters: lengths differ between rows" in prompt
    assert "`c` (numeric): a measured covariate" in prompt and "Toy nucleotide sequences" in prompt
    assert "..." in prompt and "Example 6: y = " in prompt  # long sequences by their ends; six examples
    assert 'X["seq"]' in prompt  # the worked example reads the text column


def test_the_saved_function_model_predicts_new_rows_behind_the_guard(monkeypatch):
    _script(monkeypatch, [_reply(_term("gc_count", GC_COUNT), _term("covariate", COVARIATE))])
    bundle = _sequence_bundle()
    saved = json.loads(json.dumps(igsr(_cfg(), seed=0, data_bundle=bundle)["best_model"].to_dict()))

    rows = bundle.dataset_test[["seq", "c"]]
    np.testing.assert_allclose(FittedModel.from_dict(saved).predict(rows).ravel(), bundle.dataset_test["y"],
                               atol=1e-9)
    saved["functions"][0]["code"] = "import os\ndef gc_count(X):\n    return X['c']"
    with pytest.raises(UnsafeCodeError):
        FittedModel.from_dict(saved).predict(rows)


# ---------------------------------------------------------------------------
# Repairs through the agent's loop
# ---------------------------------------------------------------------------


def test_a_failing_term_goes_back_with_its_failure_and_its_correction_is_kept(monkeypatch):
    broken = 'def covariate(X):\n    return X["no_such_column"]'
    sent = _script(monkeypatch, [
        _reply(_term("gc_count", GC_COUNT), _term("covariate", broken)),
        _reply(_term("covariate", COVARIATE, replaces="covariate")),
    ])
    result = igsr(_cfg(), seed=0, data_bundle=_sequence_bundle())

    assert "Term `covariate` raised KeyError" in sent[1] and "no_such_column" in sent[1]
    records = {record["name"]: record for record in result["term_records"]}
    assert records["covariate"]["repaired"] and not records["gc_count"]["repaired"]
    assert result["proposal_stats"] == {"proposed": 2, "corrections": 1, "accepted": 2, "repaired": 1,
                                        "repair_requests": 1, "left_out": {}}
    assert result["best_metric"] < 1e-12


def test_a_correction_that_measures_something_else_is_renamed(monkeypatch):
    constant = 'def long_rows(X):\n    return (X["seq"].str.len() > 1000).astype(float)'
    _script(monkeypatch, [
        _reply(_term("gc_count", GC_COUNT), _term("long_rows", constant)),
        _reply(_term("longer_than_20", 'def longer_than_20(X):\n    return (X["seq"].str.len() > 20).astype(float)',
                     description="whether the sequence is longer than 20", replaces="long_rows")),
    ])
    result = igsr(_cfg(), seed=0, data_bundle=_sequence_bundle())
    records = {record["name"]: record for record in result["term_records"]}
    assert records["longer_than_20"]["renamed_from"] == "long_rows" and records["longer_than_20"]["repaired"]


def test_without_repair_rounds_a_failing_term_is_left_out(monkeypatch):
    sent = _script(monkeypatch, [_reply(_term("gc_count", GC_COUNT),
                                        _term("broken", 'def broken(X):\n    return X["nope"]'))])
    result = igsr(_cfg("experiment.function_terms.repair_rounds=0"), seed=0, data_bundle=_sequence_bundle())

    assert len(sent) == 1 and result["best_terms"] == ["gc_count"]
    assert result["proposal_stats"]["left_out"] == {RAISED: 1}


def test_an_unreadable_reply_goes_back_with_the_format(monkeypatch):
    sent = _script(monkeypatch, ["TERMS:\n```json\n[{\"name\": \"x\"}]\n```", _reply(_term("gc_count", GC_COUNT))])
    result = igsr(_cfg(), seed=0, data_bundle=_sequence_bundle())
    assert "Entry 1 must be an object" in sent[1] and result["best_terms"] == ["gc_count"]


def test_the_pruner_is_shown_what_each_term_measures(monkeypatch):
    sent = _script(monkeypatch, [
        _reply(_term("gc_count", GC_COUNT, description="occurrences of GC"), _term("covariate", COVARIATE)),
        'DECISION:\n```\n{"keep": ["gc_count", "covariate"], "drop": []}\n```',
    ])
    igsr(_cfg("experiment.algorithmic_pruning=false"), seed=0, data_bundle=_sequence_bundle())
    assert "description" in sent[1] and "occurrences of GC" in sent[1]


# ---------------------------------------------------------------------------
# Screening, one proposal at a time
# ---------------------------------------------------------------------------


def test_screening_names_each_way_a_term_can_fail():
    frame = pd.DataFrame({"seq": ["ACGT" * k for k in range(1, 11)], "c": np.arange(10.0)})
    registry, training = _training_rows(frame)
    proposals = [
        Proposal("raises", "", "", 'def raises(X):\n    return X["missing"]'),
        Proposal("scalar", "", "", "def scalar(X):\n    return 3.0"),
        Proposal("mostly_nan", "", "", 'def mostly_nan(X):\n    return X["c"].where(X["c"] > 6)'),
        Proposal("constant", "", "", 'def constant(X):\n    return X["c"] * 0'),
        Proposal("imports_os", "", "", 'import os\ndef imports_os(X):\n    return X["c"]'),
        Proposal("bad-name", "", "", 'def bad(X):\n    return X["c"]'),
        Proposal("elsewhere", "", "", 'def other_name(X):\n    return X["c"]'),
        Proposal("some_nan", "", "", 'def some_nan(X):\n    return X["c"].where(X["c"] > 0)'),
    ]
    found = screen(proposals, registry, training, first_seen="0")

    assert {name: checked.failure for name, (_, checked) in found.failing.items()} == {
        "raises": RAISED, "scalar": WRONG_SHAPE, "mostly_nan": UNDEFINED, "constant": CONSTANT,
        "imports_os": REFUSED, "bad-name": NAMING, "elsewhere": NAMING}
    assert found.accepted == ["some_nan"]
    term = registry["some_nan"]
    assert term.undefined_share == pytest.approx(0.1) and term.fill == 5.0
    assert registry.column("some_nan", training)[0] == 5.0


def test_the_same_computation_is_one_term_and_a_taken_name_is_renamed():
    frame = pd.DataFrame({"c": np.arange(10.0)})
    registry, training = _training_rows(frame)
    first = screen([Proposal("double_c", "", "", 'def double_c(X):\n    """Twice c."""\n    return 2 * X["c"]')],
                   registry, training, first_seen="0")
    again = screen([Proposal("twice", "", "", 'def twice(X):\n    return 2*X["c"]  # the same'),
                    Proposal("double_c", "", "", 'def double_c(X):\n    return 3 * X["c"]')],
                   registry, training, first_seen="1")

    assert first.accepted == ["double_c"] and again.accepted == ["double_c", "double_c_2"]
    assert registry["double_c_2"].renamed_from == "double_c" and "def double_c_2(X)" in registry["double_c_2"].code
    assert normalised_source('def a(X):\n    """doc"""\n    return X["c"]', "a") == \
        normalised_source('def b(X):  # comment\n    return X[\'c\']', "b")


def test_a_term_that_fails_on_other_rows_takes_its_fill_there():
    registry, training = _training_rows(pd.DataFrame({"seq": ["ACGTACGT", "AAAACCCC", "GGGGTTTT"]}))
    screen([Proposal("eighth_is_t", "", "", 'def eighth_is_t(X):\n    return X["seq"].map(lambda s: float(s[7] == "T"))')],
           registry, training, first_seen="0")
    validation = FunctionTermRows("validation", pd.DataFrame({"seq": ["ACG", "ACGTACGT"]}), registry)

    np.testing.assert_array_equal(registry.column("eighth_is_t", validation), [1.0, 1.0])
    assert "IndexError" in registry.failures["eighth_is_t"]["validation"]


def test_a_term_that_runs_too_long_is_stopped():
    registry, training = _training_rows(pd.DataFrame({"c": np.arange(5.0)}), _settings(timeout_seconds=1))
    found = screen([Proposal("forever", "", "", "def forever(X):\n    while True:\n        pass")],
                   registry, training, first_seen="0")
    assert "took longer than 1 s" in found.failing["forever"][1].detail


# ---------------------------------------------------------------------------
# The prompt's parts, and the settings
# ---------------------------------------------------------------------------


def test_the_columns_and_examples_are_described_for_code_over_them():
    frame = pd.DataFrame({"seq": ["ACGTACGTACGTACGT", "ACG"], "depth": [1.0, 2.0]})
    assert describe_columns(frame, {"depth": "read depth"}) == (
        "- `seq` (text, from 3 to 16 characters: lengths differ between rows)\n- `depth` (numeric): read depth")
    assert shown("ACGTACGTACGTACGT", 8) == "ACGT...ACGT (16 characters)" and shown("ACG", 8) == "ACG"


def test_a_reply_must_carry_every_field():
    proposals, error = parse_function_terms(_reply(_term("a", "def a(X):\n    return X['c']")), "TERMS:")
    assert error is None and proposals[0].name == "a"
    assert parse_function_terms("no terms here", "TERMS:")[1] == "TERMS: was not found in the reply."
    assert "not valid JSON" in parse_function_terms("TERMS:\n```json\n[{]\n```", "TERMS:")[1]


def test_every_setting_of_the_function_form_is_stated():
    with pytest.raises(ValueError, match="must state max_undefined_share, repair_rounds"):
        FunctionTermSettings.from_config({"n_examples": 3})
    with pytest.raises(ValueError, match="max_undefined_share"):
        _settings(max_undefined_share=1.0)


# ---------------------------------------------------------------------------
# Opt-in: a live model (run with `-m slow`)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_live_model_writes_function_terms_over_sequences(live_llm_config):
    """One iteration of the function form against the model of the provider SeqMaestro's root .env
    names: the reply parses, its terms run on sequences of different lengths, and a model is fitted
    on them. Needs a valid key."""
    cfg = compose_config([live_llm_config, "experiment.term_language=function", "experiment.tree=false",
                          "experiment.n_iters=1", "experiment.algorithmic_pruning=true",
                          "experiment.early_stop_enabled=false"])
    result = igsr(cfg, seed=0, data_bundle=_sequence_bundle())

    assert result["term_records"] and result["best_terms"]
    assert result["compute_profiler"]["number_llm_calls"] >= 1
    assert np.isfinite(result["best_metric"])


@pytest.mark.parametrize("handler", ["except Exception:", "except:"])
def test_a_term_catching_every_error_still_stops_at_its_timeout(handler):
    """A term that catches its errors row by row, with `except Exception:` or a bare `except:`,
    cannot catch its timeout and run on with no limit left. Bounded, so that such a failure costs
    this test seconds."""
    term = compile_feature_function(
        "def slow(X):\n"
        "    total = 0\n"
        "    for _ in range(100):\n"
        "        try:\n"
        "            for _ in range(10 ** 6):\n"
        "                total += 1\n"
        f"        {handler}\n"
        "            total = -1\n"
        "    return X['v'] * total\n")
    with pytest.raises(TimeoutError, match="took longer than 1 s"):
        call_with_timeout(term, pd.DataFrame({"v": [1.0]}), 1)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "the alarm is cancelled"
    assert signal.getsignal(signal.SIGALRM) is signal.SIG_DFL, "the handler before it is restored"

