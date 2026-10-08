"""SeqMaestro's own shared code.

This package holds the code that belongs to neither a vendored method nor a single experiment:
the standard record a prepared dataset is described in and the checks made on one, sequence
featurisation and the feature-record schema, the models, the evaluation harness, the language-model
proposer and the judge, the reduction, the reliability panel, the report, and the stages of a run
that call them (`seqmaestro.pipeline`).

**This package must not import any vendored method.** The dependency runs one way: SeqMaestro may
read a method's artifacts, and a method never imports SeqMaestro. Each method has its own conda
environment to keep it that way, and `tests/test_boundaries.py` enforces the rule. The two sides
exchange files on disk: the rows a method is handed, and the predictions, the feature records and
the saved model it hands back.

Keep the base dependency set small (numpy, pandas and scikit-learn, with scipy and threadpoolctl,
which arrive with scikit-learn). Anything heavier belongs behind an optional extra in the repo-root
`pyproject.toml`, so this package can be installed into any environment that needs it.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
