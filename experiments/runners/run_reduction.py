#!/usr/bin/env python
"""Stage 3A: reduce each arm of a run to a model of N features, on validation only.

    experiments/runners/run_reduction.py --run-dir experiments/runs/<date>_<name>

`--help` describes the stage, and `seqmaestro/pipeline/reduction.py` holds it.
"""

from seqmaestro.pipeline.reduction import main

if __name__ == "__main__":
    raise SystemExit(main())
