"""The search of a PySR model, run as a process of `SeqMaestro_Method_pysr`. It imports **numpy,
sympy, PySR and the standard library only**.

`seqmaestro.models.symbolic` starts this file by its path under the PySR environment's interpreter
the first time a model is fitted in a Core process, and keeps the process open for every later fit
of that Core process, so Julia compiles once per Core process and not once per fit. The PySR
environment has no `seqmaestro`, which is why this file does not import it (`tests/test_symbolic.py`
checks this), and why the equation format below is restated instead of imported.

**The protocol.** Requests and replies pass through two dedicated pipes, whose descriptors are this
script's first two arguments, because Julia takes over the standard streams when it starts: it
writes to standard output as it pleases, and it switches standard input to non-blocking mode, which
ends a Python loop reading from it. Standard input is `/dev/null`, and standard output and error go
to the log the caller opened. The third argument is the working directory the caller made for this
process, which is removed when the requests end. A request is one line naming a directory, which
holds the rows to search (`X.npy`), their targets (`y.npy`), their weights (`weights.npy`, absent
for an unweighted fit) and PySR's settings (`options.json`). The reply goes into the same
directory: `reply.json` with every equation on the front as a tree, its complexity and PySR's loss
on the rows searched, and `values.npy` with PySR's prediction of each equation on those rows, so
that Core can check its evaluation against PySR's. Then one line is written on the reply pipe. A
search that fails replies with its error, and the process goes on serving. It ends when the request
pipe closes.

**The tree** is `seqmaestro.models.equation`'s: nested lists of `add`, `mul`, `pow` with a constant
exponent, `exp`, `log`, `col` and `const`. It is built from each equation's sympy form, which is the
form PySR's `predict` evaluates. PySR parses that form unevaluated, so `x0 - 0.6` arrives as a sum
holding a product of -1 and 0.6, and a division as a product with a power of -1. `tree_of` keeps
every node as it comes, so the tree computes what PySR's `predict` computes. An equation holding
any other node is left out of the reply and counted.

**Julia stays inside the environment.** juliapkg decides where Julia, its packages and its project
live, and unless told otherwise it may use the home directory or whichever conda environment the
caller has active (`ENVIRONMENTS.md`). Before PySR is imported, the four variables that decide it
are set from this interpreter's prefix, and the process refuses to start if any of them is already
set to something else.

**The variables are named `x0` to `xk` here**, because PySR refuses names that sympy defines, and a
tree refers to a column by its position. An equation never leaves this process as text.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

#: Options set here and never by a request. `input_stream` stops PySR from watching standard input
#: for a `q` to stop, since nothing is typed into this process (its standard input is `/dev/null`).
#: `temp_equation_file` stops it from writing its equations under `outputs/` wherever it runs. The
#: last three keep the log quiet and make the thread count, which the caller sets when it starts the
#: process, the only thing that decides how the search runs.
PROCESS_OPTIONS = {"input_stream": "devnull", "temp_equation_file": True, "verbosity": 0,
                   "progress": False, "parallelism": "multithreading"}


#: The prefix of the working directory the caller makes for this process
#: (`seqmaestro.models.symbolic.WORKDIR_PREFIX`, restated because this file cannot import it). Only
#: a directory so named is removed.
WORKDIR_PREFIX = "seqmaestro-pysr-"


def julia_variables(prefix: Path) -> dict[str, str]:
    """The variables that keep Julia, its packages and its project inside the environment at
    `prefix`, where the environment's recipe put them."""
    return {"JULIA_DEPOT_PATH": str(prefix / "share" / "julia"),
            "PYTHON_JULIAPKG_PROJECT": str(prefix / "julia_env"),
            "PYTHON_JULIAPKG_EXE": str(prefix / "julia_env" / "pyjuliapkg" / "install" / "bin"
                                       / "julia"),
            "PYTHON_JULIAPKG_OFFLINE": "yes"}


def confine_julia(prefix: Path, environ) -> None:
    """Set the variables in `environ`, refusing one already set to anything else."""
    for name, value in julia_variables(prefix).items():
        given = environ.get(name)
        if given not in (None, "", value):
            raise RuntimeError(f"{name} is set to {given!r}. The search process keeps Julia inside "
                               f"its own environment and sets it to {value!r}; unset it.")
        environ[name] = value


class OutsideVocabulary(ValueError):
    """An equation holds a node the tree cannot express. The message names the kind of node and
    never the expression, whose variables are this process's placeholders."""


def tree_of(expression, n_columns: int) -> list:
    """The tree for one sympy expression, node for node as PySR parsed it."""
    import sympy

    if expression.is_Symbol:
        name = expression.name
        if not (name.startswith("x") and name[1:].isdigit() and int(name[1:]) < n_columns):
            raise OutsideVocabulary("a symbol that is not one of the searched columns")
        return ["col", int(name[1:])]
    if expression.is_Number:
        value = float(expression)
        if not math.isfinite(value):
            raise OutsideVocabulary("a constant that is not finite")
        return ["const", value]
    if isinstance(expression, sympy.Add):
        return ["add", *(tree_of(arg, n_columns) for arg in expression.args)]
    if isinstance(expression, sympy.Mul):
        return ["mul", *(tree_of(arg, n_columns) for arg in expression.args)]
    if isinstance(expression, sympy.Pow):
        base, exponent = expression.args
        if not exponent.is_Number or not math.isfinite(float(exponent)):
            raise OutsideVocabulary("a power whose exponent is not a finite constant")
        return ["pow", tree_of(base, n_columns), float(exponent)]
    if isinstance(expression, sympy.exp):
        return ["exp", tree_of(expression.args[0], n_columns)]
    if isinstance(expression, sympy.log) and len(expression.args) == 1:
        return ["log", tree_of(expression.args[0], n_columns)]
    raise OutsideVocabulary(f"a node of kind {type(expression).__name__}")


def versions() -> dict[str, str]:
    """The versions of PySR, of the Julia package that runs the search (SymbolicRegression.jl),
    and of Julia."""
    import pysr
    from pysr.julia_import import jl, jl_version

    return {"pysr": pysr.__version__,
            "symbolic_regression_jl": str(jl.seval("string(pkgversion(SymbolicRegression))")),
            "julia": ".".join(str(part) for part in jl_version)}


def search_with_pysr(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None, options: dict
                     ) -> tuple[dict, np.ndarray]:
    """One PySR search. Returns the reply (the front as trees, what was left out, the versions) and
    PySR's prediction of each equation on the searched rows, one row per equation."""
    from pysr import PySRRegressor

    clash = sorted(set(options) & set(PROCESS_OPTIONS))
    if clash:
        raise ValueError(f"{clash} are set by the search process, not by a request.")
    model = PySRRegressor(**options, **PROCESS_OPTIONS)
    model.fit(X, y, weights=weights, variable_names=[f"x{i}" for i in range(X.shape[1])])

    front, left_out, values = [], [], []
    for index, (_, row) in enumerate(model.equations_.iterrows()):
        try:
            tree = tree_of(row["sympy_format"], X.shape[1])
            predicted = np.asarray(model.predict(X, index=index), dtype=float).ravel()
        except Exception as error:  # an equation that cannot be exported is counted, not fatal
            # Only this module's message (`OutsideVocabulary`) is kept, because any other may quote
            # the expression.
            reason = str(error) if isinstance(error, OutsideVocabulary) else type(error).__name__
            left_out.append({"complexity": int(row["complexity"]), "reason": reason})
            continue
        front.append({"tree": tree, "complexity": int(row["complexity"]),
                      "loss": float(row["loss"])})
        values.append(predicted)
    return ({"front": front, "left_out": left_out, "versions": versions()},
            np.vstack(values) if values else np.zeros((0, len(X))))


def serve(requests, channel, search) -> None:
    """Answer each request line with a reply in its directory and a line on `channel`, until the
    requests end. `search` is `search_with_pysr`, or a stand-in in the tests."""
    for line in requests:
        if not line.strip():
            continue
        directory = Path(json.loads(line)["directory"])
        started = time.time()
        try:
            X, y = np.load(directory / "X.npy"), np.load(directory / "y.npy")
            weights = (np.load(directory / "weights.npy")
                       if (directory / "weights.npy").exists() else None)
            options = json.loads((directory / "options.json").read_text())
            reply, values = search(X, y, weights, options)
            np.save(directory / "values.npy", values)
        except Exception:
            reply = {"error": traceback.format_exc()}
        reply["seconds"] = round(time.time() - started, 2)
        (directory / "reply.json").write_text(json.dumps(reply))
        channel.write("done\n")
        channel.flush()


def run(argv: list[str], search) -> None:
    """Serve on the two pipes that `argv` names (the request pipe's descriptor, then the reply
    pipe's), then remove the working directory named third, if it is one the caller made.

    It is removed only when the requests end normally, so the log inside it survives a crash for
    the caller to quote.
    """
    with os.fdopen(int(argv[0]), "r") as requests, os.fdopen(int(argv[1]), "w") as replies:
        serve(requests, replies, search)
    if len(argv) > 2:
        workdir = Path(argv[2])
        if workdir.name.startswith(WORKDIR_PREFIX) and workdir.is_dir():
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    confine_julia(Path(sys.prefix), os.environ)
    run(sys.argv[1:] if argv is None else argv, search_with_pysr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
