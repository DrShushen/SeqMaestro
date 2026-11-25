"""`seqmaestro.models.equation`: how Core evaluates the equation a symbolic search hands back, writes
it out and attributes it to its features."""

from __future__ import annotations

import numpy as np
import pytest

from seqmaestro.models import equation
from seqmaestro.models.equation import check, columns, evaluate, filled, remap, shapley, terms

NAMES = ["gc_content", "motif_count", "loop_length"]


def rows(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.2, 0.8, n), rng.integers(1, 6, n).astype(float),
                            rng.uniform(1.0, 9.0, n)])


def col(i: int) -> list:
    return ["col", i]


def const(c: float) -> list:
    return ["const", c]


# ------------------------------------------------------------------------------ evaluation

def test_every_operator_computes_what_numpy_computes():
    X = rows()
    x0, x1, x2 = X.T
    tree = ["add", const(0.5), ["mul", const(2.0), col(0)],
            ["mul", const(-1.0), ["mul", col(1), ["pow", col(2), -1]]],
            ["mul", ["exp", col(0)], ["log", col(1)]], ["pow", col(2), 2]]

    expected = 0.5 + 2.0 * x0 + -1.0 * (x1 * x2 ** -1.0) + np.exp(x0) * np.log(x1) + x2 ** 2.0
    np.testing.assert_allclose(evaluate(tree, X), expected, rtol=1e-15)


def test_a_constant_equation_gives_one_value_per_row():
    assert evaluate(const(1.25), rows(7)).tolist() == [1.25] * 7


def test_undefined_rows_are_not_finite_and_take_the_fill():
    X = np.array([[1.0, 0.0], [-1.0, 2.0], [800.0, 1.0], [2.0, 4.0]])
    tree = ["add", ["mul", col(0), ["pow", col(1), -1]], ["log", col(0)], ["exp", col(0)]]

    values = evaluate(tree, X)
    assert np.isfinite(values).tolist() == [False, False, False, True]
    kept, undefined = filled(values, 7.5)
    assert undefined.tolist() == [True, True, True, False]
    assert kept[:3].tolist() == [7.5] * 3
    assert kept[3] == pytest.approx(2.0 / 4.0 + np.log(2.0) + np.exp(2.0))


@pytest.mark.parametrize("tree", [
    ["sin", col(0)],
    ["add", col(0)],
    ["pow", col(0), col(1)],
    ["col", 3],
    ["col", True],
    ["const", float("inf")],
    ["exp", col(0), col(1)],
    "x0",
])
def test_a_tree_outside_the_vocabulary_is_refused(tree):
    with pytest.raises(ValueError):
        check(tree, n_columns=3)


def test_the_columns_a_tree_names_and_the_same_tree_over_other_positions():
    tree = ["add", ["pow", col(2), 2.0], ["mul", const(3.0), col(0), col(2)]]
    assert columns(tree) == [0, 2]
    moved = remap(tree, {0: 1, 2: 0})
    assert columns(moved) == [0, 1]
    X = rows()
    np.testing.assert_array_equal(evaluate(moved, X[:, [2, 0]]), evaluate(tree, X))


# ------------------------------------------------------------------------------ writing it out

def test_an_equation_is_written_in_the_features_names_term_by_term():
    tree = ["add", ["mul", const(1.23456), col(0)],
            ["mul", const(-1.0), ["mul", const(0.5), col(1), ["pow", col(2), -1]]],
            ["mul", const(-1.0), const(0.6221292)]]

    assert terms(tree, NAMES) == ["1.23 * gc_content", "- 0.5 * motif_count / loop_length",
                                  "- 0.622"]
    assert equation.text(tree, NAMES) == "1.23 * gc_content - 0.5 * motif_count / loop_length - 0.622"


def test_placeholder_names_never_appear_in_a_written_equation():
    tree = ["add", ["exp", ["mul", const(-2.0), col(1)]], ["log", col(0)], ["pow", col(2), 3]]
    written = equation.text(tree, NAMES)

    assert written == "exp(-2 * motif_count) + log(gc_content) + loop_length^3"
    assert "x0" not in written and "x1" not in written


def test_brackets_follow_precedence():
    tree = ["mul", ["add", col(0), const(-1.0)], ["pow", ["add", col(1), col(2)], -2]]
    assert equation.text(tree, NAMES) == "(gc_content - 1) / (motif_count + loop_length)^2"
    divided = ["mul", col(0), ["pow", ["mul", col(1), col(2)], -1]]
    assert equation.text(divided, NAMES) == "gc_content / (motif_count * loop_length)"


def test_latex_writes_names_as_upright_text_and_fractions_typeset():
    tree = ["add", ["mul", const(2.5e-5), col(0), ["pow", col(2), -1]], ["exp", col(1)]]
    assert terms(tree, NAMES, latex=True) == [
        "\\frac{2.5 \\times 10^{-5} \\cdot \\text{gc\\_content}}{\\text{loop\\_length}}",
        "+ \\exp\\left(\\text{motif\\_count}\\right)"]


# ------------------------------------------------------------------------------ Shapley values

def test_shapley_values_sum_to_the_output_less_the_background_mean():
    X, background = rows(40, seed=1), rows(30, seed=2)
    tree = ["add", ["mul", col(0), ["exp", col(2)]], ["mul", const(0.3), ["log", col(1)]],
            ["mul", col(1), col(2)]]
    values, used, share = shapley(tree, X, background, fill=0.0)

    assert used == [0, 1, 2] and values.shape == (40, 3) and share == 0.0
    expected = evaluate(tree, X) - evaluate(tree, background).mean()
    np.testing.assert_allclose(values.sum(axis=1), expected, rtol=1e-10, atol=1e-12)


def test_an_additive_equation_gives_each_term_its_centred_curve():
    X, background = rows(40, seed=3), rows(30, seed=4)
    tree = ["add", ["mul", const(2.0), col(0)], ["log", col(2)]]
    values, used, _ = shapley(tree, X, background, fill=0.0)

    assert used == [0, 2]
    np.testing.assert_allclose(values[:, 0], 2.0 * (X[:, 0] - background[:, 0].mean()), atol=1e-12)
    np.testing.assert_allclose(values[:, 1], np.log(X[:, 2]) - np.log(background[:, 2]).mean(),
                               atol=1e-12)


def test_a_product_splits_its_interaction_equally():
    X, background = rows(40, seed=5), rows(30, seed=6)
    values, _, _ = shapley(["mul", col(0), col(1)], X, background, fill=0.0)

    b0, b1 = background[:, 0], background[:, 1]
    x0, x1 = X[:, 0], X[:, 1]
    expected = 0.5 * (x0 * b1.mean() - (b0 * b1).mean() + x0 * x1 - x1 * b0.mean())
    np.testing.assert_allclose(values[:, 0], expected, atol=1e-12)


def test_a_named_feature_that_never_moves_the_output_gets_zero():
    X, background = rows(20, seed=7), rows(10, seed=8)
    values, used, _ = shapley(["add", col(0), ["mul", const(0.0), col(1)]], X, background, 0.0)

    assert used == [0, 1]
    assert np.all(values[:, 1] == 0)


def test_undefined_mixed_rows_take_the_fill_and_are_counted():
    X = np.array([[1.0, 2.0], [2.0, 0.0]])
    background = np.array([[1.0, 0.0], [1.0, 1.0]])
    tree = ["mul", col(0), ["pow", col(1), -1]]
    values, _, share = shapley(tree, X, background, fill=-3.0)

    # Of the eight evaluations per row pair, the ones with a zero in the second column are undefined.
    assert 0 < share < 1
    kept = filled(evaluate(tree, X), -3.0)[0] - filled(evaluate(tree, background), -3.0)[0].mean()
    np.testing.assert_allclose(values.sum(axis=1), kept, atol=1e-12)


def test_a_constant_equation_has_no_features_to_attribute():
    values, used, share = shapley(const(2.0), rows(5), rows(5), fill=0.0)
    assert values.shape == (5, 0) and used == [] and share == 0.0


def test_rows_are_drawn_without_replacement_or_all_taken():
    rng = np.random.default_rng(0)
    drawn = equation.sample_rows(1000, 100, rng)
    assert len(set(drawn)) == 100 and list(drawn) == sorted(drawn)
    assert equation.sample_rows(40, 100, rng).tolist() == list(range(40))
