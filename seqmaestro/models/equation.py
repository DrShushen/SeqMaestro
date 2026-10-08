"""An equation over a feature matrix's columns, in SeqMaestro's own form. **numpy only.**

A symbolic search (`seqmaestro.models.symbolic`) finds closed-form equations in another
environment and hands each one back as a tree built from a small vocabulary. Everything after the
search happens here, in Core: evaluating an equation on new rows, writing it in the features'
names, and attributing its output to the features it uses. Keeping the equation as our own data,
and not as the search library's object, means a fitted model loads and predicts without that
library, and a later version of it cannot refuse a saved cell.

**The tree** is nested lists, so it travels as JSON and pickles as plain data:

    ["add", a, b, ...]     a sum of two or more trees
    ["mul", a, b, ...]     a product of two or more trees
    ["pow", a, e]          a tree raised to a constant exponent e; a division is a product with a
                           power of -1
    ["exp", a], ["log", a]
    ["col", i]             the i-th column of the matrix the equation was found on
    ["const", c]           a constant, kept at full precision

**Undefined rows.** A division by zero, the log of a number that is not positive, and an overflow
give a value that is not finite. `evaluate` returns it as it is, and the model decides what such a
row gets (`fill`). A tree is evaluated in float64 with numpy, operator by operator, in the order the
tree holds them.

**Names.** A tree refers to columns by position and never by a name, so no placeholder name the
search used internally can reach a reader. `terms` writes an equation in the names it is given.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

#: The operators a tree may hold. `add` and `mul` take two or more arguments.
OPERATORS = ("add", "mul", "pow", "exp", "log", "col", "const")

#: Rows of the fitting data whose outputs other rows are compared against: the background of the
#: interventional Shapley values. A hundred rows keeps the exact computation affordable, since each
#: explained row is evaluated against every background row for every subset of features, and the
#: background's mean changes little with more rows.
SHAPLEY_BACKGROUND = 100

#: At most this many rows are explained. Importances and dependence curves are means over rows,
#: which settle long before this.
SHAPLEY_ROWS = 2000


def check(tree, n_columns: int) -> None:
    """Refuse a tree outside the vocabulary, or naming a column the matrix does not have."""
    if not isinstance(tree, list) or not tree or tree[0] not in OPERATORS:
        raise ValueError(f"Not an equation node: {tree!r}.")
    op, args = tree[0], tree[1:]
    if op in ("add", "mul"):
        if len(args) < 2:
            raise ValueError(f"`{op}` needs two or more arguments, not {len(args)}.")
        for arg in args:
            check(arg, n_columns)
    elif op == "pow":
        if len(args) != 2 or not _is_number(args[1]):
            raise ValueError(f"`pow` takes a tree and a constant exponent: {tree!r}.")
        check(args[0], n_columns)
    elif op in ("exp", "log"):
        if len(args) != 1:
            raise ValueError(f"`{op}` takes one argument: {tree!r}.")
        check(args[0], n_columns)
    elif op == "col":
        if (len(args) != 1 or not isinstance(args[0], int) or isinstance(args[0], bool)
                or not 0 <= args[0] < n_columns):
            raise ValueError(f"`col` names a column from 0 to {n_columns - 1}: {tree!r}.")
    elif len(args) != 1 or not _is_number(args[0]):
        raise ValueError(f"`const` holds one finite number: {tree!r}.")


def _is_number(value) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool)
            and bool(np.isfinite(value)))


def columns(tree) -> list[int]:
    """The columns a tree names, in ascending order."""
    found: set[int] = set()

    def visit(node) -> None:
        if node[0] == "col":
            found.add(node[1])
        elif node[0] != "const":
            for arg in node[1:2] if node[0] == "pow" else node[1:]:
                visit(arg)

    visit(tree)
    return sorted(found)


def remap(tree, mapping: dict[int, int]):
    """The same tree over other column positions: column `i` becomes `mapping[i]`."""
    op = tree[0]
    if op == "col":
        return ["col", mapping[tree[1]]]
    if op == "const":
        return list(tree)
    if op == "pow":
        return ["pow", remap(tree[1], mapping), tree[2]]
    return [op, *(remap(arg, mapping) for arg in tree[1:])]


def evaluate(tree, X: np.ndarray) -> np.ndarray:
    """The tree's value on each row of `X`, in float64. Not finite where the equation is undefined.

    Columns are taken by position from `X`, which must have every column the tree names.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Rows must be a two-dimensional array, not {X.ndim}-dimensional.")
    with np.errstate(all="ignore"):
        value = np.asarray(_value(tree, X), dtype=float)
    return np.broadcast_to(value, (len(X),)).copy()


def _value(node, X: np.ndarray):
    op = node[0]
    if op == "col":
        return X[:, node[1]]
    if op == "const":
        return float(node[1])
    if op == "add":
        total = _value(node[1], X)
        for arg in node[2:]:
            total = total + _value(arg, X)
        return total
    if op == "mul":
        total = _value(node[1], X)
        for arg in node[2:]:
            total = total * _value(arg, X)
        return total
    if op == "pow":
        return np.power(_value(node[1], X), float(node[2]))
    if op == "exp":
        return np.exp(_value(node[1], X))
    if op == "log":
        return np.log(_value(node[1], X))
    raise ValueError(f"Not an equation node: {node!r}.")


def filled(values: np.ndarray, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """`values` with every value that is not finite replaced by `fill`, and the mask of those rows."""
    values = np.asarray(values, dtype=float)
    undefined = ~np.isfinite(values)
    return np.where(undefined, fill, values), undefined


# ------------------------------------------------------------------------------ Shapley values

def shapley(tree, X: np.ndarray, background: np.ndarray, fill: float
            ) -> tuple[np.ndarray, list[int], float]:
    """Exact interventional Shapley values of the tree's output on each row of `X`.

    Returns the values, one column per column the tree names (in `columns(tree)`'s order), those
    columns, and the share of the evaluations that were undefined and took `fill`.

    A feature's value on a row is its average marginal contribution over every subset of the
    other features the tree names. A subset's worth on a row is the tree's mean output when the
    subset's features take the row's values and the rest take a background row's, averaged over the
    background: the interventional form, which asks what the equation does and not how the features
    are correlated. The subsets are enumerated, 2 to the power K of them for K features, so the
    values are exact for this background, and on each row they sum to the output (filled) less the
    background's mean output.

    An undefined output takes `fill` wherever it arises, on a mixed row as on a real one, so the
    values describe the model that predicts, which fills the same way.
    """
    X, background = np.asarray(X, dtype=float), np.asarray(background, dtype=float)
    used = columns(tree)
    k, n, m = len(used), len(X), len(background)
    if k == 0:
        return np.zeros((n, 0)), [], 0.0
    local = remap(tree, {column: j for j, column in enumerate(used)})
    rows, others = X[:, used], background[:, used]

    # worth[s] is subset s's mean output on each row; bit j of s says feature j takes the row's value.
    worth = np.empty((2 ** k, n))
    undefined = 0
    for subset in range(2 ** k):
        from_row = np.array([(subset >> j) & 1 for j in range(k)], dtype=bool)
        mixed = np.where(from_row, rows[:, None, :], others[None, :, :]).reshape(n * m, k)
        values, missing = filled(evaluate(local, mixed), fill)
        undefined += int(missing.sum())
        worth[subset] = values.reshape(n, m).mean(axis=1)

    weight = [factorial(size) * factorial(k - size - 1) / factorial(k) for size in range(k)]
    values = np.zeros((n, k))
    for j in range(k):
        others_of_j = [i for i in range(k) if i != j]
        for size in range(k):
            for members in combinations(others_of_j, size):
                subset = sum(1 << i for i in members)
                values[:, j] += weight[size] * (worth[subset | (1 << j)] - worth[subset])
    return values, used, undefined / (2 ** k * n * m)


def sample_rows(n_rows: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """`size` row indices drawn without replacement, in ascending order, or every row if fewer."""
    if n_rows <= size:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


# ------------------------------------------------------------------------------ writing it out

def terms(tree, names: list[str], *, digits: int = 3, latex: bool = False) -> list[str]:
    """The equation as its top-level terms, in the features' names, constants to `digits`
    significant figures. Every term after the first starts with its sign, so the terms joined by
    spaces are the whole equation, and a long equation can be broken between them.

    Plain text uses `*`, `/` and `^`; `latex=True` writes the names as upright text for a typeset
    equation. Numbers multiplied together in a product are shown as one coefficient, which is only
    a way of writing the same value.
    """
    writer = _Writer(names, digits, latex)
    parts = _summands(tree)
    written = []
    for index, part in enumerate(parts):
        negative, body = writer.signed(part)
        if index == 0:
            written.append(f"-{body}" if negative else body)
        else:
            written.append(f"- {body}" if negative else f"+ {body}")
    return written


def text(tree, names: list[str], *, digits: int = 3) -> str:
    """The equation on one line, in the features' names."""
    return " ".join(terms(tree, names, digits=digits))


def _summands(node) -> list:
    if node[0] != "add":
        return [node]
    return [part for arg in node[1:] for part in _summands(arg)]


def _factors(node) -> list:
    if node[0] != "mul":
        return [node]
    return [part for arg in node[1:] for part in _factors(arg)]


class _Writer:
    """Writes trees as text or LaTeX. Precedence: a sum inside a product or under a power is
    bracketed, and a function's argument is bracketed by the function."""

    def __init__(self, names: list[str], digits: int, latex: bool) -> None:
        self.names, self.digits, self.latex = names, digits, latex

    def number(self, value: float) -> str:
        shown = format(abs(value), f".{self.digits}g")
        if self.latex and "e" in shown:
            mantissa, exponent = shown.split("e")
            shown = f"{mantissa} \\times 10^{{{int(exponent)}}}"
        return shown

    def name(self, index: int) -> str:
        name = self.names[index]
        if not self.latex:
            return name
        for char in "\\_%&#${}":
            name = name.replace(char, "\\" + char)
        return f"\\text{{{name}}}"

    def signed(self, node) -> tuple[bool, str]:
        """Whether the node is negative as written, and the node written without that sign."""
        if node[0] == "const":
            return node[1] < 0, self.number(node[1])
        if node[0] == "mul":
            return self.product(_factors(node))
        return False, self.write(node)

    def product(self, factors: list) -> tuple[bool, str]:
        coefficient = 1.0
        above, below = [], []
        for factor in factors:
            if factor[0] == "const":
                coefficient *= factor[1]
            elif factor[0] == "pow" and factor[2] == -1:
                below.extend(_factors(factor[1]))
            elif factor[0] == "pow" and factor[2] < 0:
                below.append(["pow", factor[1], -factor[2]])
            else:
                above.append(factor)
        numerator = [self.write(factor, inside="mul") for factor in above]
        if abs(coefficient) != 1 or not numerator:
            numerator.insert(0, self.number(coefficient))
        top = (" \\cdot " if self.latex else " * ").join(numerator)
        if not below:
            return coefficient < 0, top
        denominator = [self.write(factor, inside="mul") for factor in below]
        if self.latex:
            bottom = " \\cdot ".join(denominator)
            return coefficient < 0, f"\\frac{{{top}}}{{{bottom}}}"
        bottom = " * ".join(denominator)
        return coefficient < 0, f"{top} / {bottom if len(denominator) == 1 else f'({bottom})'}"

    def write(self, node, inside: str | None = None) -> str:
        op = node[0]
        if op == "col":
            return self.name(node[1])
        if op == "const":
            written = self.number(node[1])
            if node[1] >= 0:
                return written
            return f"-{written}" if inside is None else self.bracket(f"-{written}")
        if op in ("exp", "log"):
            argument = self.write(node[1])
            if self.latex:
                return f"\\{op}\\left({argument}\\right)"
            return f"{op}({argument})"
        if op == "pow":
            exponent = float(node[2])
            shown = str(int(exponent)) if exponent.is_integer() else self.number(exponent)
            base = self.write(node[1], inside="pow")
            return f"{base}^{{{shown}}}" if self.latex else f"{base}^{shown}"
        if op == "mul":
            negative, body = self.product(_factors(node))
            written = f"-{body}" if negative else body
            return self.bracket(written) if inside == "pow" or (negative and inside) else written
        # A sum, bracketed wherever it sits inside something that binds tighter.
        written = " ".join(terms(node, self.names, digits=self.digits, latex=self.latex))
        return self.bracket(written) if inside in ("mul", "pow") else written

    def bracket(self, written: str) -> str:
        return f"\\left({written}\\right)" if self.latex else f"({written})"
