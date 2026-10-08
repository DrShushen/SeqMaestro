"""Where things are in the repository: the one module that knows.

A run's code needs a few places outside the package: the task descriptions, the adapters that run
under a vendored method's interpreter, the scripts a run starts, and the directory runs go to.
They are stated here once, so that code which needs one imports it and no module works a path out
from its own location.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: One description per dataset, which every consumer that tells a language model about the data
#: takes from here (`experiments/datasets/README.md`).
DESCRIPTIONS = REPO / "experiments" / "datasets"

#: The scripts that run under a vendored method's interpreter (`experiments/adapters/README.md`).
ADAPTERS = REPO / "experiments" / "adapters"

#: The scripts a run starts: one for the arms, one for each other stage, and the analyses.
RUNNERS = REPO / "experiments" / "runners"
ANALYSIS = REPO / "experiments" / "analysis"

#: Where a run goes when no directory is named for it.
RUNS = REPO / "experiments" / "runs"
