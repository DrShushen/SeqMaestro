#!/usr/bin/env python
"""The composition baseline of a run: base fractions into a sparse linear model, at every seed.

    experiments/runners/run_baseline.py --run-dir experiments/runs/<date>_<name>

`--help` describes the baseline, and `seqmaestro/pipeline/baseline.py` holds it.
"""

from seqmaestro.pipeline.baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
