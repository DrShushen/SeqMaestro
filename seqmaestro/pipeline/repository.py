"""Where the repository keeps the files a run's code needs outside the package.

A run's code needs a few places outside the package: the task descriptions, the adapters (scripts
that run under a vendored method's interpreter), the scripts a run starts, the directory runs are
written to, and the fetched motif collections. They are stated here once, so that code which needs
one imports it instead of working out a path from its own location.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The task descriptions, one per dataset. Every part of a run that tells a language model about
#: the data takes the description from here (`experiments/datasets/README.md`).
DESCRIPTIONS = REPO / "experiments" / "datasets"

#: The scripts that run under a vendored method's interpreter (`experiments/adapters/README.md`).
ADAPTERS = REPO / "experiments" / "adapters"

#: The scripts a run starts: one for the arms, one for each other stage, and the analyses.
RUNNERS = REPO / "experiments" / "runners"
ANALYSIS = REPO / "experiments" / "analysis"

#: Where a run is written when no directory is named for it.
RUNS = REPO / "experiments" / "runs"

#: The motif collections a run can scan with, one folder each, which Stage 1 fetches with
#: `experiments/prepare/fetch_motif_collection.py`. Because they are published files that each
#: machine downloads, they are kept under `data/`, which git ignores.
MOTIF_COLLECTIONS = REPO / "data" / "motif_collections"
