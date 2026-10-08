"""The registry of what an arm can be: which model can be fitted on which pool, and how.

The config loader refuses a combination from these answers before anything is spent, and Stage 2
chooses how each arm's cells are fitted from them, so a wrong answer here either refuses a run that
would work or starts one that cannot.
"""

from __future__ import annotations

import pytest

from seqmaestro import arms
from seqmaestro.target import BINARY, CONTINUOUS


def test_l1_names_the_sparse_linear_model_for_the_kind_of_target():
    assert arms.resolve("l1", BINARY) == "l1_logistic"
    assert arms.resolve("l1", CONTINUOUS) == "l1_linear"
    assert arms.resolve("ebm", CONTINUOUS) == "ebm"


def test_what_a_core_model_declares_is_read_from_its_class():
    assert arms.black_box("xgboost")
    assert not any(arms.black_box(name) for name in ("l1_logistic", "ebm", "pysr", "deft_or_cart", "igsr"))
    assert not arms.spec("pysr").reproducible, "a PySR search is not reproducible"
    assert arms.spec("pysr").environment == "SeqMaestro_Method_pysr"
    assert arms.spec("l1_logistic").target_kinds == (BINARY,)


@pytest.mark.parametrize(("model", "pool"), [
    ("ebm", "kmer"),
    ("l1_logistic", "covariates"),
    ("xgboost", "positional+kmer+llm_generated_iterative+covariates"),
    ("deft_or_cart", "positional"),
    ("deft_or_cart", "positional+kmer+covariates"),
    ("deft_or_cart", "positional+llm_generated_joint+covariates"),
])
def test_a_combination_the_code_runs_has_no_problems(model, pool):
    assert arms.problems(model, pool, BINARY, weighted=False) == []


@pytest.mark.parametrize(("model", "pool", "kind", "weighted", "problem"), [
    ("igsr", "llm_generated_joint", CONTINUOUS, True, "without per-row weights"),
    ("igsr", "llm_generated_joint", BINARY, False, "fits a continuous target"),
    ("igsr", "positional+llm_generated_joint", CONTINUOUS, False, "cannot be handed positional yet"),
    ("igsr", "covariates", CONTINUOUS, False, "needs llm_generated_joint"),
    ("deft_or_cart", "llm_generated_joint", BINARY, False, "needs positional"),
    ("deft_or_cart", "positional+llm_generated", BINARY, False, "cannot be handed llm_generated yet"),
    ("deft_or_cart", "positional+kmer+llm_generated_joint", BINARY, False,
     "cannot be handed kmer+llm_generated_joint together yet"),
    ("ebm", "positional+llm_generated_joint", BINARY, False, "only deft_or_cart's and igsr's fits write"),
])
def test_a_combination_the_code_cannot_run_says_why(model, pool, kind, weighted, problem):
    assert any(problem in found for found in arms.problems(model, pool, kind, weighted))


def test_a_pool_holds_at_most_one_written_source():
    assert arms.pool_problems("kmer+llm_generated") == []
    assert "at most one written source" in arms.pool_problems("llm_generated+llm_generated_joint")[0]


@pytest.mark.parametrize(("model", "pool", "kind", "retries", "calls"), [
    ("ebm", "positional+kmer", "fixed", True, False),
    ("ebm", "llm_generated+covariates", "proposer", True, True),
    ("pysr", "kmer+llm_generated_iterative", "proposer", True, True),
    ("deft_or_cart", "positional", "deft", True, False),
    ("deft_or_cart", "positional+llm_generated_joint", "deft", False, True),
    ("igsr", "llm_generated_joint", "igsr", False, True),
])
def test_each_arm_is_of_one_kind_and_retries_unless_its_model_writes_inside_the_fit(
        model, pool, kind, retries, calls):
    """A second start of an arm whose language model works inside the method's fit would pay again
    for every call the first made."""
    assert arms.kind(model, pool) == kind
    assert kind in (arms.FIXED, arms.PROPOSER, arms.DEFT, arms.IGSR_KIND)
    assert arms.retries(pool) is retries
    assert arms.calls_language_model(pool) is calls


def test_an_unknown_model_is_refused_with_the_names_it_could_be():
    with pytest.raises(ValueError, match=r"Unknown model 'cart'.*deft_or_cart"):
        arms.spec("cart")


def test_an_arm_is_spelled_one_way_in_text_and_one_way_in_a_file_name():
    assert arms.arm_name("ebm", "positional+kmer") == "ebm/positional+kmer"
    assert arms.split_arm("deft_or_cart/positional+llm_generated_joint") == (
        "deft_or_cart", "positional+llm_generated_joint")
    assert arms.arm_slug("l1_logistic", "llm_generated") == "l1_logistic__llm_generated"
    for malformed in ("ebm", "/positional", "ebm/", "ebm/positional/real"):
        with pytest.raises(ValueError, match="is not an arm"):
            arms.split_arm(malformed)


def test_only_a_vendored_methods_model_is_used_in_its_environment_alone():
    """PySR's search runs in another environment, and the equation it finds is evaluated in Core."""
    assert {name for name, spec in arms.MODEL_SPECS.items() if spec.vendored} == {
        arms.DEFT_OR_CART, arms.IGSR}
    assert arms.spec("pysr").environment is not None and not arms.spec("pysr").vendored


def test_every_model_is_presented_to_a_reader_and_every_interpretable_one_is_drawn():
    """A model added to the registry has no name a reader sees and no figure until someone decides
    both, so this fails for it instead of a report being built without them."""
    from seqmaestro.report import facts, figure_inputs, naming
    from seqmaestro.target import BINARY, CONTINUOUS

    assert set(naming.MODELS) == set(arms.MODEL_SPECS)
    assert facts.HEADLINE_FAMILIES == {BINARY: ("ebm", "l1_logistic", "deft_or_cart", "pysr"),
                                       CONTINUOUS: ("ebm", "l1_linear", "deft_or_cart", "igsr", "pysr")}
    drawn = {family for families in facts.HEADLINE_FAMILIES.values() for family in families}
    assert drawn == set(figure_inputs.DRAWN_AS) == {
        name for name, spec in arms.MODEL_SPECS.items() if not spec.black_box}


def test_every_kind_of_importance_and_direction_a_model_records_is_a_listed_one():
    from seqmaestro.harness.direction import DIRECTION_KINDS
    from seqmaestro.harness.features import IMPORTANCE_KINDS
    from seqmaestro.models import MODELS
    from seqmaestro.models.sparse import LeastSquares

    classes = {model for models in MODELS.values() for model in models.values()} | {LeastSquares}
    assert {model.importance_kind for model in classes} <= set(IMPORTANCE_KINDS)
    assert {model.direction_kind for model in classes} <= set(DIRECTION_KINDS)
