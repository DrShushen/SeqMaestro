"""A stand-in for PySR's search process, for Core's tests: no Julia, no PySR, a front in seconds.

It runs under Core's interpreter and serves through the search process's own loop
(`seqmaestro/models/symbolic_search.py`, loaded by its path), so the request, the reply and the
reply channel are the real ones. Only the search is replaced. By default the front is fitted on the
searched rows in the loss the request names, squared error or the logistic loss on labels of -1
and +1: the constant, then linear equations over the best 1, 2, 4 and 8 columns, the best being
those most correlated with the target.

A test that needs other equations copies this script into a directory of its own and writes
`front.json` beside it: `{"trees": [...], "losses": [...], "perturb": 0.0}`. The trees are
returned as the front, with the losses given as PySR's (or squared error when none are given), and
PySR's values are the trees evaluated here and multiplied by `1 + perturb`. A copy with a file named
`keep_requests` beside it keeps each request there as `last_request.npz`, with the thread count the
process was started with, so a test can see what the model sent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from seqmaestro.models import equation

HERE = Path(__file__).resolve().parent
SEARCH = Path(equation.__file__).with_name("symbolic_search.py")


def fitted_line(design, y, w, logistic):
    """Weighted least squares, or a weighted logistic regression of labels of -1 and +1 by
    Newton's method with a small ridge, which keeps a separable design finite."""
    if not logistic:
        root = np.sqrt(w)[:, None]
        return np.linalg.lstsq(design * root, y * root[:, 0], rcond=None)[0]
    coefficients = np.zeros(design.shape[1])
    positive = (y > 0).astype(float)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ coefficients))
        gradient = design.T @ (w * (p - positive)) + 1e-3 * coefficients
        hessian = (design * (w * p * (1 - p))[:, None]).T @ design + 1e-3 * np.eye(design.shape[1])
        coefficients = coefficients - np.linalg.solve(hessian, gradient)
    return coefficients


def least_squares_front(X, y, weights, logistic):
    w = np.ones(len(y)) if weights is None else weights
    trees = [["const", float(fitted_line(np.ones((len(y), 1)), y, w, logistic)[0])]]
    spread = X.std(axis=0)
    usable = np.flatnonzero(spread > 0)
    if usable.size:
        centred = X[:, usable] - X[:, usable].mean(axis=0)
        strength = np.abs(centred.T @ (y - y.mean())) / spread[usable]
        best = [int(usable[i]) for i in np.argsort(-strength)[:8]]
        for size in (1, 2, 4, 8):
            if size > len(best):
                break
            columns = best[:size]
            design = np.column_stack([X[:, columns], np.ones(len(y))])
            coefficients = fitted_line(design, y, w, logistic)
            terms = [["mul", ["const", float(c)], ["col", column]]
                     for c, column in zip(coefficients[:-1], columns, strict=True)]
            trees.append(["add", *terms, ["const", float(coefficients[-1])]])
    return trees, None, 0.0


def search(X, y, weights, options):
    logistic = options["elementwise_loss"] == "LogitMarginLoss()"
    given = HERE / "front.json"
    if given.exists():
        spec = json.loads(given.read_text())
        trees, losses, perturb = spec["trees"], spec.get("losses"), spec.get("perturb", 0.0)
    else:
        trees, losses, perturb = least_squares_front(X, y, weights, logistic)
    values = np.vstack([equation.evaluate(tree, X) for tree in trees])
    if losses is None:
        w = np.ones(len(y)) if weights is None else weights
        losses = [float(np.average(np.logaddexp(0, -y * value) if logistic else (y - value) ** 2,
                                   weights=w)) for value in values]
    if (HERE / "keep_requests").exists():
        np.savez(HERE / "last_request.npz", X=X, y=y,
                 weights=np.array([]) if weights is None else weights, options=json.dumps(options),
                 threads=os.environ.get("PYTHON_JULIACALL_THREADS", ""),
                 tmpdir=os.environ.get("TMPDIR", ""))
    front = [{"tree": tree, "complexity": 1 + 4 * index, "loss": loss}
             for index, (tree, loss) in enumerate(zip(trees, losses, strict=True))]
    reply = {"front": front, "left_out": [],
             "versions": {"pysr": "fake", "symbolic_regression_jl": "fake", "julia": "fake"}}
    return reply, values * (1.0 + perturb)


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("symbolic_search", SEARCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(sys.argv[1:], search)
