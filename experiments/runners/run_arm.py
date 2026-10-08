#!/usr/bin/env python
"""Stage 2: fit cells of the arms a run's config names.

    experiments/runners/run_arm.py --run-dir experiments/runs/<date>_<name> --arms ebm/positional

`--help` describes the stage, and `seqmaestro/pipeline/stage2.py` holds it.
"""

from seqmaestro.pipeline.stage2 import main

if __name__ == "__main__":
    raise SystemExit(main())
