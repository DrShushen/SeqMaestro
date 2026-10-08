#!/usr/bin/env python
"""Stage 3B: the reliability panel of each arm's reduced model, before the reveal and after it.

    experiments/runners/run_reliability.py --run-dir experiments/runs/<date>_<name>
    experiments/runners/run_reliability.py --run-dir ... --after-reveal

`--help` describes the stage, and `seqmaestro/pipeline/reliability.py` holds it.
"""

from seqmaestro.pipeline.reliability import main

if __name__ == "__main__":
    raise SystemExit(main())
