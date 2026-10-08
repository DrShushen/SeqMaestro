#!/usr/bin/env python
"""Run a dataset through the whole pipeline, start to finish, unattended: every arm and analysis,
the reduction and the reliability panel, the reveal, and the report's inputs.

    experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml

`--help` describes the run, and `seqmaestro/pipeline/driver.py` holds it.
"""

from seqmaestro.pipeline.driver import main

if __name__ == "__main__":
    raise SystemExit(main())
