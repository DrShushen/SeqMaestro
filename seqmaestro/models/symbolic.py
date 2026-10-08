"""Symbolic regression with PySR: one equation over the features, found by an evolutionary search.

PySR (Cranmer, 2023, arXiv:2305.01582) evolves closed-form equations over the columns it is given,
and returns a front: for each complexity (the number of nodes in the equation) up to `maxsize`, the
equation with the lowest loss it found. The models here choose one equation from that front and use
it as SeqMaestro uses any model: it predicts, says which features it used and how much, and which
way each of them points.

**Where it runs.** Only the search runs in PySR's environment, in a process of
`SeqMaestro_Method_pysr` started by the first fit in a Core process and kept for every later fit of
that process (`symbolic_search.py`). Core never starts Julia. The search hands back each equation on
the front as a tree (`seqmaestro.models.equation`), and Core evaluates it. Every front equation is
checked against PySR's own predictions on the searched rows, so a fault in the export stops the fit
instead of changing the model. A fitted model holds only these trees and numbers, so it pickles,
loads and predicts without PySR.

**The choice.** A fit sets aside `set_aside_fraction` of its fitting rows, drawn with its seed and
stratified as the harness stratifies (`seqmaestro.target`), and searches the rest. From the front it
keeps the equation with the lowest loss on the rows set aside, in the loss the search minimised and
weighted as the fit is, and a tie goes to the simpler equation. PySR's own choice weighs each
equation's training loss against its complexity, which suits clean data: on a noisy target nearly
the whole front has a training loss close to the lowest, and the choice then says little about new
rows. The lowest loss on rows the search did not see can still favour a longer equation whose gain
is noise; the extra features such an equation carries are what the reduction's ranking and the
reliability panel exist to find. The rows set aside come from the fitting rows and not from the
validation split, so a fit depends on its fitting rows alone, as XGBoost's regressor stops early on
its own rows. The chosen equation keeps the constants the search fitted, and nothing is refitted on
the rows set aside.

**Undefined rows.** An equation can be undefined on a row: a division by zero, the log of a number
that is not positive, an overflow. An equation undefined on any row set aside is never chosen. On
other rows the model predicts its constant for such a row: the weighted mean of the fitting targets,
or the weighted positive rate for a binary target. `filled(X)` says which rows those are, so the
harness can record how many there were.

**Importance and direction** come from exact interventional Shapley values of the equation's output
(`equation.shapley`), in the target's units or in log-odds for a binary target. A feature's
importance is its mean absolute value over up to `equation.SHAPLEY_ROWS` fitting rows, and a feature
the equation names whose values are all zero is unused. A direction comes from the dependence of a
feature's values on the feature, as XGBoost's does from its TreeSHAP contributions. The fit keeps
the values it ranked by, so a report draws the same ones.

**Not reproducible.** The search runs in PySR's multithreaded mode, on the threads its process was
given or on every core. That mode does not fix the order in which the populations evolve and
exchange equations, so two fits of the same rows with the same seed can return different fronts,
on one thread as on many (`reproducible = False`). Every stage keeps what it fitted, so no number
depends on fitting again, and the reliability panel does not rerun the reduction for this model
(`docs/reliability.md`).

**The binary target** is searched as labels of -1 and +1 under PySR's logistic margin loss, so the
equation is the log-odds of the positive class and `predict` returns its sigmoid.

**The settings are fixed here** and recorded by `describe()`, as XGBoost's are:

- `+ - * /`, `exp` and `log`, where `exp` and `log` may not contain `exp` or `log`: nested
  exponentials and logarithms are hard to follow, and PySR's tuning notes suggest the constraint.
- 200 iterations of 31 populations of 27 equations, PySR's population defaults. A dataset makes
  several hundred searches across its stages, which is what the budget is sized for.
- `maxsize` 25 nodes, below PySR's 30, so that an equation stays short enough to be followed.
- Precision 32, PySR's default. The search fits constants and compares equations to about seven
  significant figures, far finer than these targets' noise, and a constant comes back as the
  shortest decimal that gives back its 32-bit value. Core evaluates that decimal in float64, as
  PySR's own `predict` does.
- PySR's automatic batching: above about a thousand rows it evolves on batches of rows and scores
  its best equations on all of them.
- The seed is the cell's, although a search does not repeat itself under it.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split

from seqmaestro.execution import allowed_threads
from seqmaestro.harness import direction
from seqmaestro.models import equation
from seqmaestro.models.base import fit_summary
from seqmaestro.models.weights import resolve_class_weight, resolve_fit_weights
from seqmaestro.provenance import environment_python, missing_python
from seqmaestro.target import BINARY, CONTINUOUS, target_for

#: The environment the search runs in.
ENVIRONMENT = "SeqMaestro_Method_pysr"

#: The script the search process runs, by its path.
SEARCH_SCRIPT = Path(__file__).with_name("symbolic_search.py")

#: The prefix of a search process's working directory. The search process removes the directory it
#: is handed only when the name starts with this (`symbolic_search.WORKDIR_PREFIX`, held equal by a
#: test), so it can never remove anything else.
WORKDIR_PREFIX = "seqmaestro-pysr-"

#: Core's evaluation of a front equation may differ from PySR's own prediction by this much,
#: relative to the equation's largest value on the searched rows. Both evaluate the same
#: expression in float64 with numpy, and differ only by the order of a sum's terms.
PARITY = 1e-9


# ------------------------------------------------------------------------------ the search process

class SearchProcess:
    """A search process of the PySR environment, serving one Core process.

    Started with `threads` Julia threads, or one per core when `threads` is None, in a clean
    environment: only the home directory, a path to the interpreter's environment, and the thread
    count reach it. Requests and replies travel on two pipes of their own, since Julia takes over
    the standard streams. Its standard input is `/dev/null`, and its standard output and error go
    to a log in its working directory, a temporary directory. The end of that log is quoted in the
    error raised when the process ends without replying. The directory is its `TMPDIR` too, since
    PySR writes each search's equations to a temporary directory of its own and never removes it.
    The process removes its working directory when its requests end, which also covers a forked
    Tier 1 worker's process: the worker exits without running exit handlers, but its end closes the
    request pipe.
    """

    def __init__(self, python: Path, script: Path, threads: int | None) -> None:
        self.owner = os.getpid()
        self.workdir = Path(tempfile.mkdtemp(prefix=WORKDIR_PREFIX))
        self.log_path = self.workdir / "search.log"
        environment = {"HOME": os.environ.get("HOME", str(self.workdir)),
                       "PATH": f"{python.parent}:/usr/bin:/bin", "LANG": "C.UTF-8",
                       "TMPDIR": str(self.workdir),
                       "PYTHON_JULIACALL_THREADS": "auto" if threads is None else str(threads)}
        if threads is not None:
            # PySR otherwise gives Julia's garbage collector half the machine's cores, which
            # oversubscribes the machine when many single-threaded searches run side by side.
            environment["JULIA_NUM_GC_THREADS"] = str(threads)
        theirs_to_read, ours_to_write = os.pipe()
        ours_to_read, theirs_to_write = os.pipe()
        with open(self.log_path, "w") as log:
            self.process = subprocess.Popen(
                [str(python), str(script), str(theirs_to_read), str(theirs_to_write),
                 str(self.workdir)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(theirs_to_read, theirs_to_write), cwd=self.workdir, env=environment)
        os.close(theirs_to_read)
        os.close(theirs_to_write)
        self.requests = os.fdopen(ours_to_write, "w")
        self.replies = os.fdopen(ours_to_read, "r")

    def search(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None,
               options: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        """One search: the reply, and PySR's prediction of each front equation on `X`."""
        request = Path(tempfile.mkdtemp(dir=self.workdir))
        try:
            np.save(request / "X.npy", np.ascontiguousarray(X, dtype=float))
            np.save(request / "y.npy", np.asarray(y, dtype=float))
            if weights is not None:
                np.save(request / "weights.npy", np.asarray(weights, dtype=float))
            (request / "options.json").write_text(json.dumps(options))
            try:
                self.requests.write(json.dumps({"directory": str(request)}) + "\n")
                self.requests.flush()
            except BrokenPipeError:
                pass  # reported below, with the log
            if not self.replies.readline():
                raise RuntimeError(f"The PySR search process ended without replying. The end of "
                                   f"its log ({self.log_path}):\n{self.log_tail()}")
            reply = json.loads((request / "reply.json").read_text())
            if "error" in reply:
                raise RuntimeError(f"The PySR search failed:\n{reply['error']}")
            return reply, np.load(request / "values.npy")
        finally:
            shutil.rmtree(request, ignore_errors=True)

    def alive(self) -> bool:
        return self.process.poll() is None

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text().splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def close(self) -> None:
        """End the process by closing its request pipe, and remove its directory. Only the Core
        process that started it does this: a forked child holds copies of its pipes and nothing
        more."""
        if os.getpid() != self.owner:
            return
        try:
            self.requests.close()
            self.process.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        self.replies.close()
        shutil.rmtree(self.workdir, ignore_errors=True)


#: The search processes this Core process started, by the process id that started them, the
#: interpreter, the script and the thread count. A forked child finds none under its own id, so it
#: starts its own and never writes to its parent's.
_PROCESSES: dict[tuple[int, str, str, int | None], SearchProcess] = {}


def search_process(threads: int | None) -> SearchProcess:
    """This Core process's search process for `threads`, started if it has none that is alive."""
    python = environment_python(ENVIRONMENT)
    if not python.exists():
        raise FileNotFoundError(missing_python(ENVIRONMENT))
    key = (os.getpid(), str(python), str(SEARCH_SCRIPT), threads)
    process = _PROCESSES.get(key)
    if process is None or not process.alive():
        process = _PROCESSES[key] = SearchProcess(python, SEARCH_SCRIPT, threads)
    return process


@atexit.register
def close_search_processes() -> None:
    for key, process in list(_PROCESSES.items()):
        if key[0] == os.getpid():
            process.close()
            del _PROCESSES[key]


def check_parity(trees: list, X: np.ndarray, theirs: np.ndarray, complexities: list[int]) -> None:
    """Refuse a front whose trees Core evaluates differently from PySR's own predictions."""
    for tree, predicted, complexity in zip(trees, theirs, complexities, strict=True):
        ours = equation.evaluate(tree, X)
        if not np.array_equal(np.isfinite(ours), np.isfinite(predicted)):
            raise RuntimeError(f"The front equation of complexity {complexity} is undefined on "
                               f"different rows in Core than in PySR, so its export is wrong.")
        finite = np.isfinite(ours)
        if finite.any():
            scale = max(float(np.max(np.abs(predicted[finite]))), np.finfo(float).tiny)
            worst = float(np.max(np.abs(ours[finite] - predicted[finite]))) / scale
            if worst > PARITY:
                raise RuntimeError(f"Core's value of the front equation of complexity "
                                   f"{complexity} differs from PySR's by {worst:.2e} of its "
                                   f"scale, more than {PARITY}, so its export is wrong.")


# ------------------------------------------------------------------------------ the models

class _Equation:
    """What a symbolic model of either kind does. The subclasses say how the target is searched,
    what the loss is, and what the constant prediction and the link are."""

    name = "pysr"
    importance_kind = "mean_abs_shapley"
    direction_kind = "shapley_dependence"
    #: One equation, written in the features' names.
    interpretable = True
    #: The search's populations evolve in no fixed order, on one thread as on many.
    reproducible = False
    #: The environment the search runs in, which a stage records beside a run that fits this model.
    environment = ENVIRONMENT
    #: A search runs a Julia thread per core it is allowed.
    one_core: ClassVar[dict[str, Any]] = {"threads": 1}
    target_kind: str
    loss: str

    def __init__(self, seed: int = 0, threads: int | None = None, niterations: int = 200,
                 populations: int = 31, population_size: int = 27, maxsize: int = 25,
                 precision: int = 32, set_aside_fraction: float = 0.15) -> None:
        if not 0 < set_aside_fraction < 1:
            raise ValueError(f"set_aside_fraction is the share of the fitting rows the equation is "
                             f"chosen on, strictly between 0 and 1, not {set_aside_fraction!r}.")
        if threads is not None and threads < 1:
            raise ValueError(f"threads is a number of Julia threads, or None for the threads the "
                             f"process was given (`seqmaestro.execution`), not {threads!r}.")
        self.seed = seed
        self.threads = threads
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.maxsize = maxsize
        self.precision = precision
        self.set_aside_fraction = set_aside_fraction
        self._row_weights = False
        self._fitted: dict[str, Any] | None = None

    # -------------------------------------------------------------- what the subclasses decide

    def _search_target(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _row_losses(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The loss the search minimises, on each row."""
        raise NotImplementedError

    def _constant(self, y: np.ndarray, weights: np.ndarray | None) -> float:
        """The constant prediction, in the equation's units."""
        raise NotImplementedError

    def _link(self, output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _weights(self, y: np.ndarray, sample_weight) -> np.ndarray | None:
        raise NotImplementedError

    # -------------------------------------------------------------- fitting

    def options(self) -> dict[str, Any]:
        """The search's settings, as PySR takes them."""
        return {"binary_operators": ["+", "-", "*", "/"], "unary_operators": ["exp", "log"],
                "nested_constraints": {"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}},
                "niterations": self.niterations, "populations": self.populations,
                "population_size": self.population_size, "maxsize": self.maxsize,
                "precision": self.precision, "elementwise_loss": self.loss, "batching": "auto",
                "random_state": self.seed}

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> _Equation:
        self._row_weights = sample_weight is not None
        names = [str(name) for name in X.columns]
        values = X.to_numpy(dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        weights = self._weights(y, sample_weight)
        searched, aside = train_test_split(
            np.arange(len(y)), test_size=self.set_aside_fraction,
            stratify=target_for(self.target_kind).strata(y), random_state=self.seed)
        searched, aside = np.sort(searched), np.sort(aside)

        def part(rows):
            return None if weights is None else weights[rows]

        threads = self.threads if self.threads is not None else allowed_threads()
        reply, predicted = search_process(threads).search(
            values[searched], self._search_target(y[searched]), part(searched), self.options())
        front = reply["front"]
        trees = [entry["tree"] for entry in front]
        complexities = [entry["complexity"] for entry in front]
        for tree in trees:
            equation.check(tree, len(names))
        check_parity(trees, values[searched], predicted, complexities)

        losses = []
        for tree in trees:
            output = equation.evaluate(tree, values[aside])
            losses.append(float(np.average(self._row_losses(output, y[aside]), weights=part(aside)))
                          if np.isfinite(output).all() else float("inf"))
        eligible = [i for i in range(len(trees)) if np.isfinite(losses[i])]
        if not eligible:
            raise RuntimeError("No equation on the front is defined on every row set aside.")
        chosen = min(eligible, key=lambda i: (losses[i], complexities[i]))

        used = equation.columns(trees[chosen])
        local = equation.remap(trees[chosen], {column: j for j, column in enumerate(used)})
        constant = self._constant(y, weights)
        rng = np.random.default_rng(self.seed)
        background = equation.sample_rows(len(y), equation.SHAPLEY_BACKGROUND, rng)
        explained = equation.sample_rows(len(y), equation.SHAPLEY_ROWS, rng)
        shapley, _, filled_share = equation.shapley(local, values[explained][:, used],
                                                    values[background][:, used], constant)
        self._fitted = {
            "names": names, "trees": trees, "complexities": complexities,
            "searched_losses": [entry["loss"] for entry in front], "set_aside_losses": losses,
            "chosen": chosen, "used": [names[i] for i in used], "local": local,
            "constant": constant, "background": values[background][:, used],
            "explained": values[explained][:, used], "shapley": shapley,
            "shapley_filled_share": filled_share, "n_searched": len(searched),
            "n_set_aside": len(aside), "left_out": reply["left_out"],
            "versions": reply["versions"], "search_seconds": reply["seconds"],
        }
        return self

    # -------------------------------------------------------------- using the fit

    def _state(self) -> dict[str, Any]:
        if self._fitted is None:
            raise RuntimeError("The model has not been fitted.")
        return self._fitted

    def output(self, X: pd.DataFrame) -> np.ndarray:
        """The chosen equation on each row, in its own units, with undefined rows filled."""
        state = self._state()
        rows = X[state["used"]].to_numpy(dtype=float) if state["used"] else np.zeros((len(X), 0))
        return equation.filled(equation.evaluate(state["local"], rows), state["constant"])[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._link(self.output(X))

    def filled(self, X: pd.DataFrame) -> np.ndarray:
        """Which rows of `X` get the constant prediction, because the equation is undefined there."""
        state = self._state()
        rows = X[state["used"]].to_numpy(dtype=float) if state["used"] else np.zeros((len(X), 0))
        return ~np.isfinite(equation.evaluate(state["local"], rows))

    def importances(self) -> dict[str, float]:
        """Mean absolute Shapley value, over the fitting rows the fit explained, for each feature
        whose values are not all zero."""
        state = self._state()
        means = np.abs(state["shapley"]).mean(axis=0) if len(state["shapley"]) else []
        return {name: float(mean) for name, mean in zip(state["used"], means, strict=True)
                if mean > 0}

    def directions(self, X: pd.DataFrame) -> dict[str, str]:
        """Which way each used feature moves the equation, from its Shapley values on up to
        `equation.SHAPLEY_ROWS` of these rows, binned by the feature's values."""
        state = self._state()
        used = self.importances()
        if not used:
            return {}
        rows = X[state["used"]].to_numpy(dtype=float)
        sample = equation.sample_rows(len(rows), equation.SHAPLEY_ROWS,
                                      np.random.default_rng(self.seed))
        shapley, _, _ = equation.shapley(state["local"], rows[sample], state["background"],
                                         state["constant"])
        words = target_for(self.target_kind).directions
        found = {}
        for j, name in enumerate(state["used"]):
            if name not in used:
                continue
            curve = direction.dependence_curve(rows[sample, j], shapley[:, j])
            if (word := direction.from_curve(curve, words)) is not None:
                found[name] = word
        return found

    def shapley_sample(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """The features the equation names, their values on the fitting rows the fit explained, and
        their Shapley values there: the numbers `importances` averages, one column per feature."""
        state = self._state()
        return list(state["used"]), state["explained"], state["shapley"]

    def equation_terms(self, digits: int = 3, latex: bool = False) -> list[str]:
        """The chosen equation's top-level terms, in the features' names (`equation.terms`)."""
        state = self._state()
        return equation.terms(state["local"], state["used"], digits=digits, latex=latex)

    def equation_text(self, digits: int = 3) -> str:
        """The chosen equation on one line, in the features' names."""
        state = self._state()
        return equation.text(state["local"], state["used"], digits=digits)

    def front(self, digits: int = 3) -> list[dict[str, Any]]:
        """Every equation on the front, simplest first: its complexity, its loss on the rows
        searched and on the rows set aside, and the equation in the features' names."""
        state = self._state()
        return [{"complexity": state["complexities"][i],
                 "searched_loss": state["searched_losses"][i],
                 "set_aside_loss": (state["set_aside_losses"][i]
                                    if np.isfinite(state["set_aside_losses"][i]) else None),
                 "chosen": i == state["chosen"],
                 "equation": equation.text(state["trees"][i], state["names"], digits=digits)}
                for i in range(len(state["trees"]))]

    def describe(self) -> dict[str, Any]:
        """The settings, and under `fitted` what a fit found, which a rebuilt model does not have
        and the reduction's comparison leaves out."""
        described = {
            "model": self.name, **{key: value for key, value in self.options().items()
                                   if key != "random_state"},
            "set_aside_fraction": self.set_aside_fraction,
            "choice": "lowest set-aside loss",
            "shapley_background": equation.SHAPLEY_BACKGROUND,
            "shapley_rows": equation.SHAPLEY_ROWS,
            "threads": self.threads, "seed": self.seed, "target_kind": self.target_kind,
            "per_row_weights": self._row_weights,
        }
        if self._fitted is not None:
            state = self._fitted
            described["fitted"] = {
                "equation": self.equation_text(), "features": state["used"],
                "complexity": state["complexities"][state["chosen"]],
                "front": self.front(), "n_searched": state["n_searched"],
                "n_set_aside": state["n_set_aside"], "constant": state["constant"],
                "shapley_filled_share": state["shapley_filled_share"],
                "left_out": state["left_out"], "versions": state["versions"],
                "search_seconds": state["search_seconds"],
            }
        else:
            described["fitted"] = None
        return described

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        """The shared summary, plus the chosen equation and the front, in the features' names.

        The equation is the model, so a proposer sees how its features were combined: which were
        multiplied, which divided, which sit inside a logarithm. The front shows what a simpler or a
        longer equation would have given on the rows set aside.
        """
        summary = fit_summary(self, X, y)
        summary["equation"] = self.equation_text()
        summary["front"] = [{key: entry[key] for key in ("complexity", "set_aside_loss",
                                                         "equation")}
                            for entry in self.front()]
        return summary


class SymbolicClassification(_Equation):
    """A symbolic model of a binary target: an equation for the log-odds of the positive class,
    searched on labels of -1 and +1 under PySR's logistic margin loss, whose sigmoid is the
    predicted probability. Balanced class weights by default, as for every binary model."""

    target_kind = BINARY
    loss = "LogitMarginLoss()"

    def __init__(self, seed: int = 0, class_weight: str | None = "balanced", **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self.class_weight = resolve_class_weight(class_weight)

    def _weights(self, y, sample_weight):
        return resolve_fit_weights(y, self.class_weight, sample_weight)

    def _search_target(self, y):
        return 2.0 * y - 1.0

    def _row_losses(self, output, y):
        # The logistic margin loss on labels of -1 and +1 is the log-loss of the sigmoid.
        return np.logaddexp(0.0, -(2.0 * y - 1.0) * output)

    def _constant(self, y, weights):
        rate = float(np.clip(np.average(y, weights=weights), 1e-12, 1 - 1e-12))
        return float(np.log(rate / (1.0 - rate)))

    def _link(self, output):
        return expit(output)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "class_weight": self.class_weight}


class SymbolicRegression(_Equation):
    """A symbolic model of a continuous target: an equation for the target, searched under squared
    error. It takes no class weighting; per-row weights from the design are used as for every
    model."""

    target_kind = CONTINUOUS
    loss = "L2DistLoss()"

    def _weights(self, y, sample_weight):
        return resolve_fit_weights(y, None, sample_weight)

    def _search_target(self, y):
        return y

    def _row_losses(self, output, y):
        return (y - output) ** 2

    def _constant(self, y, weights):
        return float(np.average(y, weights=weights))

    def _link(self, output):
        return output
