"""Guardrails on executing language-model-written Python: the function term language's code.

In the function form (`igsr.method.function_terms`), a language model writes each term as a Python
function and the search runs it on the rows, so `exec()` of untrusted code is part of the method.
The prompts also show rows of the dataset, which makes a crafted dataset a prompt-injection path to
code execution on the host.

**This is a port of the guard in DEFT's vendored copy** (`src/deft/code_guard.py`, by way of
SeqMaestro's proposer), and its tests are ported alongside it as `tests/test_code_guard.py`, where
they are the acceptance bar. The copy stands alone, so it imports neither; tests are
specifications, and copying them keeps the three guards rejecting the same attacks.

## Scope

A **guardrail**, not a security boundary. It raises the cost of an accident or of casual
misbehaviour from zero to substantial. An allowlist does not stop a determined adversary with a
CPython `exec` and enough patience. Only process or container isolation does that, and it is not
implemented here.

Three layers:

1. **Static check.** The source is parsed and inspected before anything runs: no imports outside
   the allowlist, no dunder attribute access, no reference to `open` / `eval` / `exec` / `compile` /
   `getattr` and friends, and nothing at module level except imports and function definitions.
2. **Restricted namespace.** Execution gets a minimal `__builtins__` and a guarded `__import__`, so
   the blocked names are not reachable even if the static check were somehow evaded.
3. **What runs is what was checked.** The validated AST is compiled directly, without re-parsing
   the source string, so the two cannot drift apart.

**A bare `except:` is compiled as `except Exception:`.** A term's timeout
(`igsr.method.function_terms.call_with_timeout`) stops it with an exception that `except Exception`
does not catch, and a bare `except:` would catch that exception and let the term run on. For code
handling its own errors, the two clauses do the same.

**The contract is one function taking a single argument named `X`**, a frame of the rows.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class UnsafeCodeError(ValueError):
    """Generated code was rejected before it ran.

    A subclass of ValueError so a caller can treat it the same as code that fails to compile: log
    it and drop the feature. Rejections then show up as a falling usable-feature yield, which the
    proposer records, so an over-tight allowlist is visible instead of silently starving the
    candidate pool.
    """


# Modules a sequence feature may legitimately use.
#
# `re` earns its place because regex motifs (`[AG]GGTCA`) are a natural way to express a binding
# site. `collections` and `itertools` are here because `Counter` is the obvious tool for k-mer
# composition and `product` for k-mer enumeration, and a model writing sequence features reaches
# for both.
#
# Left out: `scipy`, whose uses here (entropy, periodicity) can be written in numpy, and Biopython,
# which the model has no way to know is installed. Add either when the yield metric shows features
# reaching for them.
DEFAULT_ALLOWED_MODULES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "re",
    "math",
    "collections",
    "itertools",
)

# Also bound into the namespace under these names, because a model usually writes `np.` and `pd.`
# without importing them. Pre-binding raises the yield.
MODULE_ALIASES: dict[str, str] = {"np": "numpy", "pd": "pandas"}

# Names that must not appear at all. Blocking `getattr` and `setattr` is essential: their attribute
# name is a runtime string, so they are the route that would slip past the static dunder check.
BLOCKED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)

# The builtins a feature is allowed. This list matters more than the module list, because without
# `len`, `sum`, `sorted` and the rest almost every real feature breaks.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    # So a feature can catch its own edge cases instead of raising and being discarded.
    "Exception", "IndexError", "KeyError", "TypeError", "ValueError", "ZeroDivisionError",
)


@dataclass(frozen=True)
class CodeGuardPolicy:
    """What generated code is allowed to do.

    The defaults are the strict policy. `enabled=False` is a supported escape hatch, but it
    restores unrestricted `exec` of model-written code, so a run that uses it should say so in its
    manifest.
    """

    enabled: bool = True
    allowed_modules: tuple[str, ...] = DEFAULT_ALLOWED_MODULES
    extra_allowed_modules: tuple[str, ...] = ()

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.allowed_modules, *self.extra_allowed_modules)))

    @classmethod
    def from_config(cls, cfg: Any) -> CodeGuardPolicy:
        """Build from a mapping (a parsed config block, or argparse output). `None` means strict."""
        if cfg is None:
            return cls()
        values: dict[str, Any] = {}
        if "enabled" in cfg:
            values["enabled"] = bool(cfg["enabled"])
        for key in ("allowed_modules", "extra_allowed_modules"):
            if key in cfg and cfg[key] is not None:
                values[key] = tuple(str(m) for m in cfg[key])
        return cls(**values)

    def describe(self) -> dict[str, Any]:
        """For the run manifest: what the code that produced these features was permitted to do."""
        return {"enabled": self.enabled, "allowed_modules": list(self.modules)}


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _check_import(node: ast.AST, allowed: Sequence[str]) -> None:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:  # ast.ImportFrom
        if node.level:
            raise UnsafeCodeError("Relative imports are not allowed in generated features.")
        names = [node.module or ""]
    for name in names:
        root = name.split(".")[0]
        if root not in allowed:
            raise UnsafeCodeError(
                f"Import of {name!r} is not allowed in generated features. "
                f"Allowed modules: {', '.join(allowed)}."
            )


def validate_source(module: ast.Module, policy: CodeGuardPolicy) -> list:
    """Reject anything a feature function has no business doing. Returns its function defs."""
    allowed = policy.modules

    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node, allowed)
        elif isinstance(node, ast.FunctionDef):
            continue
        else:
            raise UnsafeCodeError(
                f"Only imports and function definitions are allowed at the top level of a "
                f"generated feature; found {type(node).__name__}. Example usage and top-level "
                f"statements are not permitted."
            )

    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        raise UnsafeCodeError("Generated code defines no function.")

    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node, allowed)
        elif isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                raise UnsafeCodeError(
                    f"Attribute access to {node.attr!r} is not allowed: dunder attributes are "
                    f"the usual route out of a restricted namespace."
                )
        elif isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or _is_dunder(node.id)):
            raise UnsafeCodeError(f"Use of {node.id!r} is not allowed in generated features.")

    return functions


def _guarded_import(allowed: Sequence[str]) -> Callable:
    """An `__import__` that only resolves allowlisted modules.

    The static check already rejects a disallowed import, so this is defence in depth: it keeps the
    runtime namespace restricted even if the check were bypassed.
    """
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level or root not in allowed:
            raise UnsafeCodeError(f"Import of {name!r} is not allowed in generated features.")
        return real_import(name, globals, locals, fromlist, level)

    return guarded


def _build_namespace(policy: CodeGuardPolicy) -> dict[str, Any]:
    safe_builtins: dict[str, Any] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    safe_builtins["__import__"] = _guarded_import(policy.modules)

    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    for module_name in policy.modules:
        try:
            namespace[module_name] = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - an allowlisted module that is not installed
            logger.warning("Allowlisted module %r is not importable; skipping.", module_name)
    for alias, module_name in MODULE_ALIASES.items():
        if module_name in namespace:
            namespace[alias] = namespace[module_name]
    return namespace


class _NameBareExcept(ast.NodeTransformer):
    """Every bare `except:` becomes `except Exception:` (see the module docstring)."""

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        self.generic_visit(node)
        if node.type is None:
            node.type = ast.copy_location(ast.Name(id="Exception", ctx=ast.Load()), node)
        return node


def _select_feature_function(namespace: dict[str, Any], functions: list) -> Callable:
    """Pick the feature function out of what was defined.

    The contract in the prompt is a single function taking one argument named `X`, so prefer that.
    Taking the first callable in the namespace instead, as upstream DEFT does, picks a helper
    whenever the model defines one before the feature.
    """
    taking_x = [fn.name for fn in functions if len(fn.args.args) == 1 and fn.args.args[0].arg == "X"]
    chosen = (taking_x or [fn.name for fn in functions])[-1]
    return namespace[chosen]


def compile_feature_function(code_str: str, policy: CodeGuardPolicy | None = None) -> Callable:
    """Validate and compile generated source into a callable.

    Raises `UnsafeCodeError` if the code is rejected, or `ValueError` if it does not compile. The
    two are kept apart because `def upstream_-15nt_GC(X):` is a model writing a bad identifier, not
    an attack, and counting it as one would misreport a run's rejections.
    """
    policy = policy if policy is not None else CodeGuardPolicy()

    try:
        module = ast.parse(code_str)
    except SyntaxError as exc:
        raise ValueError(f"Failed to compile function: {exc}") from exc
    module = ast.fix_missing_locations(_NameBareExcept().visit(module))

    if not policy.enabled:
        # Unrestricted execution, opt-in. It still compiles the parsed AST, and still picks the
        # function by the documented contract instead of by namespace order.
        namespace: dict[str, Any] = {}
        exec(compile(module, "<llm-feature>", "exec"), namespace)
        functions = [n for n in module.body if isinstance(n, ast.FunctionDef)]
        if not functions:
            raise ValueError("Generated code defines no function.")
        return _select_feature_function(namespace, functions)

    functions = validate_source(module, policy)
    namespace = _build_namespace(policy)
    try:
        # Compiling the validated AST, without re-parsing the string, guarantees that what runs is
        # what was checked.
        exec(compile(module, "<llm-feature>", "exec"), namespace)
    except UnsafeCodeError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to compile function: {exc}") from exc

    return _select_feature_function(namespace, functions)
