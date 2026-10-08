#!/usr/bin/env python
"""Check a run config and print it as a run would record it, with `extends` folded in and any
`--set` applied. It starts no run and costs nothing.

    experiments/runners/experiment_config.py --config experiments/configs/<name>.yaml
"""

from seqmaestro.pipeline.config import main

if __name__ == "__main__":
    raise SystemExit(main())
