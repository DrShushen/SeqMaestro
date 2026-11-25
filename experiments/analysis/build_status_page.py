#!/usr/bin/env python
"""Build a run's status page (`status.html` at the run's root) again from the run's files.

    $CORE experiments/analysis/build_status_page.py --run-dir experiments/runs/<date>_<name>

The driver writes the page while it runs, so this is for a run it has finished, for instance once
the report is written. `seqmaestro/status/` holds the page.
"""

from seqmaestro.status import main

if __name__ == "__main__":
    raise SystemExit(main())
