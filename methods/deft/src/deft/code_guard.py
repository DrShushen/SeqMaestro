"""Guardrails on executing language-model-generated Python.

DEFT's whole method is to have a model write feature functions and then run them. So `exec()`
of untrusted code is not an incidental implementation detail -- it is the mechanism. Upstream
guarded it with a substring scan:

    if "exec" in code_str or "eval" in code_str:
        raise ValueError("Unsafe code detected")

`__import__("os").system(...)` walks straight through that, as do `open`, `compile` and
`getattr`. It also rejects perfectly good code that happens to contain `eval` in a comment or a
variable name.

**Why this matters here rather than in the abstract.** The prompts embed rows of the user's own
dataset (`serialize_in_context`). A crafted CSV is therefore a prompt-injection path to code
execution on the host, and SeqMaestro's entire premise is taking someone else's dataset.

## What this is, and what it is not

This is a **guardrail**, not a security boundary. It raises the cost of an accident or of casual
misbehaviour from zero to substantial. A determined adversary with a CPython `exec` and enough
patience is not stopped by an allowlist; only process or container isolation does that, and that
is a different piece of work which has deliberately not been done here.

Three layers:

1. **Static check.** The source is parsed and inspected before anything runs: no imports outside
   the allowlist, no dunder attribute access, no reference to `open` / `eval` / `exec` /
   `compile` / `getattr` and friends, and nothing at module level except imports and function
   definitions.
2. **Restricted namespace.** Execution gets a minimal `__builtins__` and a guarded `__import__`,
   so the blocked names are not reachable even if the static check were somehow evaded.
3. **What runs is what was checked.** The validated AST is compiled directly, rather than the
   source string being re-parsed, so the two cannot drift apart.

**A bare `except:` is compiled as `except Exception:`.** The timeout on a generated feature
(`deft.features.helpers.timeout`) stops it with an exception that `except Exception` does not
catch, and a bare `except:` would catch that exception and let the feature run on. For code handling
its own errors, the two clauses do the same.

Defensive copying of the data is handled elsewhere and is already correct: `filter_features` and
`optimize_threshold` pass `copy(X)`, and `copy.copy` on a DataFrame is a *deep* copy, while
`evaluate_feature_function` says `X.copy(deep=True)` outright. A feature cannot corrupt the
training data by mutating its argument.

Configured through the `code_guard` config group -- see `src/deft/conf/code_guard/strict.yaml`.
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

    A subclass of ValueError so the existing caller in `src/deft/utils/parsing.py` logs it and drops
    the feature, exactly as it does for code that fails to compile. Rejections therefore show up
    as a falling `min_usable_feature_yield`, which the integrity policy reports -- so an
    over-tight allowlist announces itself rather than quietly starving the candidate pool.
    """


# Modules a sequence feature may legitimately use.
#
# `re` earns its place because regex motifs (`[AG]GGTCA`) are a natural way to express a binding
# site. `collections` and `itertools` are here because `Counter` is the obvious tool for k-mer
# composition and `product` for k-mer enumeration, and a model writing sequence features reaches
# for both.
#
# Deliberately absent: `scipy`, which is not installed at all now that scikit-learn is gone from
# the dependency set, and whose uses here (entropy, periodicity) are writable in numpy; and
# Biopython, which is a new dependency the model has no way to know about. Add either when the
# yield metric shows features actually reaching for them.
DEFAULT_ALLOWED_MODULES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "re",
    "math",
    "collections",
    "itertools",
)

# Bound into the namespace under these names as well, because a model writes `np.` and `pd.`
# without importing them far more often than not. Pre-binding raises the yield rather than
# lowering it.
MODULE_ALIASES: dict[str, str] = {"np": "numpy", "pd": "pandas"}

# Names that must not appear at all. Blocking `getattr`/`setattr` matters more than it looks:
# the attribute name there is a runtime string, so it is the one route that would slip past the
# static dunder check.
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

# The builtins a feature is allowed. This half matters more than the module list: without `len`,
# `sum`, `sorted` and the rest, almost every real feature breaks.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    # So a feature may guard its own edge cases rather than raising and being discarded.
    "Exception", "IndexError", "KeyError", "TypeError", "ValueError", "ZeroDivisionError",
)


@dataclass(frozen=True)
class CodeGuardPolicy:
    """What generated code is allowed to do. See `src/deft/conf/code_guard/strict.yaml`."""

    enabled: bool = True
    allowed_modules: tuple[str, ...] = DEFAULT_ALLOWED_MODULES
    extra_allowed_modules: tuple[str, ...] = ()

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.allowed_modules, *self.extra_allowed_modules)))

    @classmethod
    def from_config(cls, cfg: Any) -> CodeGuardPolicy:
        """Build from a Hydra config node. A missing block means the strict defaults."""
        if cfg is None:
            return cls()
        values: dict[str, Any] = {}
        if "enabled" in cfg:
            values["enabled"] = bool(cfg["enabled"])
        for key in ("allowed_modules", "extra_allowed_modules"):
            if key in cfg and cfg[key] is not None:
                values[key] = tuple(str(m) for m in cfg[key])
        return cls(**values)


# Run-level setting, established once by `run_experiment`. It is module-level state because the
# parsing functions are bound dynamically through `importlib` in `LLMFeatureFinder`, which makes
# threading a policy argument through three call layers more trouble than it is worth for a
# single-run process. Tests pass a policy explicitly instead of touching this.
_ACTIVE_POLICY = CodeGuardPolicy()


def set_active_policy(policy: CodeGuardPolicy) -> None:
    """Set the policy used when none is passed explicitly."""
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
    if not policy.enabled:
        logger.warning(
            "Code guard is DISABLED. Model-generated Python will be exec'd with no restriction "
            "on imports, builtins or attribute access. See src/deft/conf/code_guard/strict.yaml."
        )
    else:
        logger.info("Code guard enabled; allowed modules: %s", ", ".join(policy.modules))


def active_policy() -> CodeGuardPolicy:
    return _ACTIVE_POLICY


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
        elif isinstance(node, ast.Name) and (
            node.id in BLOCKED_NAMES or _is_dunder(node.id)
        ):
            raise UnsafeCodeError(f"Use of {node.id!r} is not allowed in generated features.")

    return functions


def _guarded_import(allowed: Sequence[str]) -> Callable:
    """An `__import__` that only resolves allowlisted modules.

    The static check already rejects a disallowed import, so this is defence in depth: it keeps
    the runtime namespace honest even if the check were bypassed.
    """
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level or root not in allowed:
            raise UnsafeCodeError(f"Import of {name!r} is not allowed in generated features.")
        return real_import(name, globals, locals, fromlist, level)

    return guarded


def _build_namespace(policy: CodeGuardPolicy) -> dict[str, Any]:
    safe_builtins: dict[str, Any] = {
        name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
    }
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

    The contract in the prompt is a single function taking one argument named `X`, so prefer
    that. Upstream took the *first* callable in the namespace, which picks a helper rather than
    the feature whenever the model defines one.
    """
    taking_x = [
        fn.name
        for fn in functions
        if len(fn.args.args) == 1 and fn.args.args[0].arg == "X"
    ]
    chosen = (taking_x or [fn.name for fn in functions])[-1]
    return namespace[chosen]


def compile_feature_function(
    code_str: str, policy: CodeGuardPolicy | None = None
) -> Callable:
    """Validate and compile generated source into a callable.

    Raises `UnsafeCodeError` if the code is rejected, or `ValueError` if it does not compile.
    """
    policy = policy if policy is not None else active_policy()

    try:
        module = ast.parse(code_str)
    except SyntaxError as exc:
        raise ValueError(f"Failed to compile function: {exc}") from exc
    module = ast.fix_missing_locations(_NameBareExcept().visit(module))

    if not policy.enabled:
        # Upstream behaviour, opt-in. Still uses the AST we parsed, and still picks the function
        # by the documented contract rather than by namespace order.
        namespace: dict[str, Any] = {}
        exec(compile(module, "<llm-feature>", "exec"), namespace)
        functions = [n for n in module.body if isinstance(n, ast.FunctionDef)]
        if not functions:
            raise ValueError("Generated code defines no function.")
        return _select_feature_function(namespace, functions)

    functions = validate_source(module, policy)
    namespace = _build_namespace(policy)
    try:
        # Compiling the validated AST, rather than re-parsing the string, guarantees that what
        # runs is exactly what was checked.
        exec(compile(module, "<llm-feature>", "exec"), namespace)
    except UnsafeCodeError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to compile function: {exc}") from exc

    return _select_feature_function(namespace, functions)
