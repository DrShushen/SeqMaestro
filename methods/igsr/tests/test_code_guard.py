"""Guardrails on executing model-written Python (`igsr/code_guard.py`).

A port of the code guard tests of DEFT's vendored copy, by way of SeqMaestro's proposer. The copy
imports neither, and a guard written from scratch can reopen a hole the others closed. Tests are
specifications, so copying them creates no import between the copies. Keep the files diffable:
when one gains a case, add it to the others.

Both directions matter. Escapes must be rejected, and real terms must still compile and run,
because an allowlist that rejects valid code would shrink the candidates without any error. All
tests run offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from igsr.code_guard import (
    CodeGuardPolicy,
    UnsafeCodeError,
    compile_feature_function,
)

STRICT = CodeGuardPolicy()
OFF = CodeGuardPolicy(enabled=False)


def frame(n=8):
    return pd.DataFrame({"raw_sequence": ["ACGTACGTGGCC" * 9 for _ in range(n)]})


def compile_strict(code):
    return compile_feature_function(code, policy=STRICT)


# --------------------------------------------------------------------------------------------
# Escapes are rejected
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, code",
    [
        ("import os", "import os\ndef f(X):\n    return os.getpid()"),
        ("from os import", "from os import system\ndef f(X):\n    return 1"),
        ("import subprocess inside the function",
         "def f(X):\n    import subprocess\n    return 1"),
        ("relative import", "from . import something\ndef f(X):\n    return 1"),
        ("__import__", 'def f(X):\n    return __import__("os").getpid()'),
        ("open", 'def f(X):\n    return len(open("/etc/passwd").read())'),
        ("eval", 'def f(X):\n    return eval("1+1")'),
        ("exec", 'def f(X):\n    exec("x = 1")\n    return 1'),
        ("compile", 'def f(X):\n    return compile("1", "<s>", "eval")'),
        ("getattr, whose attribute name is a runtime string",
         'def f(X):\n    return getattr(X, "__class__")'),
        ("setattr", 'def f(X):\n    setattr(X, "a", 1)\n    return 1'),
        ("dunder attribute walk", "def f(X):\n    return len(().__class__.__bases__)"),
        ("dunder subclasses", "def f(X):\n    return len(X.__class__.__subclasses__())"),
        ("globals", "def f(X):\n    return len(globals())"),
        ("vars", "def f(X):\n    return len(vars())"),
        ("top-level example usage", "def f(X):\n    return 1\nprint(f(None))"),
        ("top-level assignment", "SECRET = 1\ndef f(X):\n    return SECRET"),
        ("no function at all", "X = 1"),
    ],
)
def test_rejects(label, code):
    with pytest.raises(UnsafeCodeError):
        compile_strict(code)


def test_the_specific_escape_that_walked_through_the_old_guard():
    """`__import__("os").system(...)` contains neither "exec" nor "eval" as a substring, so a
    substring scan for those words (the guard upstream DEFT used) lets it through."""
    code = 'def f(X):\n    return __import__("os").system("echo pwned")'
    assert "exec" not in code and "eval" not in code, "the old scan would have allowed this"

    with pytest.raises(UnsafeCodeError, match="__import__"):
        compile_strict(code)


def test_a_rejection_says_what_was_wrong():
    with pytest.raises(UnsafeCodeError) as excinfo:
        compile_strict("import socket\ndef f(X):\n    return 1")
    message = str(excinfo.value)
    assert "socket" in message
    assert "numpy" in message, "the message should list what *is* allowed"


def test_the_guard_holds_even_if_the_static_check_were_bypassed():
    """Defence in depth: the execution namespace has no `open` or `eval`, and its `__import__` is
    the guarded one."""
    from igsr.code_guard import _build_namespace

    namespace = _build_namespace(STRICT)
    assert "open" not in namespace["__builtins__"]
    assert "eval" not in namespace["__builtins__"]
    with pytest.raises(UnsafeCodeError):
        namespace["__builtins__"]["__import__"]("os")


# --------------------------------------------------------------------------------------------
# Real features still work, so the guard cannot silently shrink the candidate pool
# --------------------------------------------------------------------------------------------


def test_the_real_feature_the_model_actually_produced():
    """A feature a model wrote during a real proposer run. If this fails, the guard is too tight
    and is starving the proposer's candidate pool."""
    code = (
        "def compute_upstream_gc_fraction_0_49(X):\n"
        '    seqs = X["raw_sequence"].astype(str).str.upper().str.slice(0, 50)\n'
        '    return ((seqs.str.count("G") + seqs.str.count("C")) / 50.0)'
    )
    fn = compile_strict(code)
    values = fn(frame())

    assert len(values) == 8
    assert 0.0 <= values.iloc[0] <= 1.0


@pytest.mark.parametrize(
    "label, code",
    [
        ("numpy pre-bound, never imported",
         "def f(X):\n    return np.array([len(s) for s in X['raw_sequence']])"),
        ("pandas pre-bound",
         "def f(X):\n    return pd.Series([1] * len(X), index=X.index)"),
        ("explicit allowlisted import",
         "import numpy as np\ndef f(X):\n    return np.ones(len(X))"),
        ("regex motif, the reason `re` is allowed",
         "import re\n"
         "def f(X):\n"
         "    return X['raw_sequence'].apply(lambda s: len(re.findall(r'[AG]GGTCA', s)))"),
        ("Counter for k-mer composition, the reason `collections` is allowed",
         "from collections import Counter\n"
         "def f(X):\n"
         "    return X['raw_sequence'].apply(lambda s: Counter(s)['G'] / len(s))"),
        ("itertools for k-mer enumeration",
         "from itertools import product\n"
         "def f(X):\n"
         "    kmers = [''.join(p) for p in product('ACGT', repeat=2)]\n"
         "    return X['raw_sequence'].apply(lambda s: sum(s.count(k) for k in kmers))"),
        ("math",
         "import math\n"
         "def f(X):\n"
         "    return X['raw_sequence'].apply(lambda s: math.log(len(s)))"),
        ("the builtins that matter",
         "def f(X):\n"
         "    return X['raw_sequence'].apply(\n"
         "        lambda s: max(sum(1 for a, b in zip(s, s[1:]) if a == b), 0))"),
        ("guarding its own edge case",
         "def f(X):\n"
         "    def safe(s):\n"
         "        try:\n"
         "            return 1.0 / len(s)\n"
         "        except ZeroDivisionError:\n"
         "            return 0.0\n"
         "    return X['raw_sequence'].apply(safe)"),
    ],
)
def test_accepts_and_runs(label, code):
    values = compile_strict(code)(frame())
    assert len(values) == 8


def test_code_mentioning_eval_in_a_comment_is_no_longer_rejected():
    """A substring scan also has false positives: it would reject this code because a comment and
    a variable name contain "eval". The guard parses the code, so it accepts it."""
    code = (
        "def f(X):\n"
        "    # we evaluate GC content here; do not eval() anything\n"
        "    evaluation = X['raw_sequence'].str.count('G')\n"
        "    return evaluation"
    )
    assert len(compile_strict(code)(frame())) == 8


# --------------------------------------------------------------------------------------------
# Which function gets returned
# --------------------------------------------------------------------------------------------


def test_the_feature_function_is_chosen_over_a_helper():
    """Taking the first callable in the namespace instead would pick the helper."""
    code = (
        "def helper(s):\n"
        "    return s.count('G')\n"
        "def feature(X):\n"
        "    return X['raw_sequence'].apply(helper)"
    )
    fn = compile_strict(code)

    assert fn.__name__ == "feature"
    assert len(fn(frame())) == 8


def test_the_contract_is_a_single_argument_named_X():
    """Selection is by that contract, not by definition order."""
    code = (
        "def feature(X):\n"
        "    return X['raw_sequence'].str.count('G')\n"
        "def afterthought(s):\n"
        "    return s\n"
    )
    assert compile_strict(code).__name__ == "feature"


# --------------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------------


def test_disabling_the_guard_allows_unrestricted_code():
    """`enabled=False` turns the guard off entirely. It is an intended escape hatch."""
    fn = compile_feature_function("import os\ndef f(X):\n    return os.name", policy=OFF)

    assert isinstance(fn(frame()), str)


def test_extra_allowed_modules_widen_rather_than_replace():
    code = "import statistics\ndef f(X):\n    return statistics.mean([1, 2])"
    with pytest.raises(UnsafeCodeError):
        compile_strict(code)

    widened = CodeGuardPolicy(extra_allowed_modules=("statistics",))
    assert compile_feature_function(code, policy=widened)(frame()) == 1.5
    # numpy must still be there, i.e. the base list was extended and not overwritten.
    assert "numpy" in widened.modules


def test_policy_from_config():
    assert CodeGuardPolicy.from_config(None).enabled is True
    assert CodeGuardPolicy.from_config({"enabled": False}).enabled is False

    widened = CodeGuardPolicy.from_config({"extra_allowed_modules": ["scipy"]})
    assert "scipy" in widened.modules and "numpy" in widened.modules


def test_syntax_errors_are_not_reported_as_unsafe():
    """A model writing `def upstream_-15nt_GC(X):` has made a parse error, so it gets a plain
    `ValueError` and not `UnsafeCodeError`."""
    with pytest.raises(ValueError) as excinfo:
        compile_strict("def upstream_-15nt_GC(X):\n    return 1")
    assert not isinstance(excinfo.value, UnsafeCodeError)


# --------------------------------------------------------------------------------------------
# The default policy (DEFT's copy has no such test: it reads its policy from module-level state)
# --------------------------------------------------------------------------------------------


def test_the_default_policy_is_the_strict_one():
    """This guard has no module-level active policy, so an omitted `policy` must mean strict."""
    with pytest.raises(UnsafeCodeError):
        compile_feature_function("import os\ndef f(X):\n    return os.name")


def test_a_bare_except_still_handles_the_terms_own_errors():
    """A bare `except:` is compiled as `except Exception:`, so the timeout cannot be caught by it,
    and the term's own errors are handled as they were."""
    fn = compile_feature_function(
        "def ratio(X):\n"
        "    out = []\n"
        "    for s in X['s']:\n"
        "        try:\n"
        "            out.append(s.count('G') / s.count('T'))\n"
        "        except:\n"
        "            out.append(0.0)\n"
        "    return out\n"
    )
    assert fn(pd.DataFrame({"s": ["GGG", "GT"]})) == [0.0, 1.0]
