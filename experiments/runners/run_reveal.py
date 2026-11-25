#!/usr/bin/env python
"""Open a run's held-out split once and score every saved model on it.

    experiments/runners/run_reveal.py --run-dir experiments/runs/<date>_<name>
    experiments/runners/run_reveal.py --run-dir ... --check    # is it ready? reveals nothing

`--help` describes the reveal, and `seqmaestro/pipeline/reveal.py` holds it.
"""

from seqmaestro.pipeline.reveal import main

if __name__ == "__main__":
    raise SystemExit(main())
