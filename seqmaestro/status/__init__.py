"""A run's status page: `status.html` at the run's root, which shows where the run stands and what
each stage has found so far, with links to the run's files.

    experiments/analysis/build_status_page.py --run-dir <run>

The page has one column per stage (1, 2, 3A and 3B), a band above them for the whole run, and a band
below for the report. Each column shows that stage's key decisions or results, with its details
collapsed, and the stage's checkpoint. `seqmaestro.status.gather` says what each part shows, and
`docs/developer/status.md` explains the rules the page keeps and how to change it. The page is built
from the run's own files and the scheduler's snapshot of its jobs, so it holds nothing a run does not
already record, and building it again gives the same page.

**The driver keeps it current** (`Updater`): it writes the page when the run starts, after a process
starts or ends (at most every `MIN_SECONDS`), at least every `QUIET_SECONDS` while a long process
runs, and when the run stops. While the driver runs, an open page reloads itself. A failure to
build the page is written once in the driver's log and never stops the run, since the page only
reports. After the run, the script above builds it again, for instance once the report is written.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from seqmaestro import files
from seqmaestro.pipeline.config import ConfigError, load_recorded
from seqmaestro.pipeline.layout import LOGS_DIR
from seqmaestro.pipeline.schedule import JOBS_FILE, snapshot
from seqmaestro.status import gather, page

#: The page, at the run's root.
PAGE_FILE = "status.html"

#: The fewest seconds between two pages written because a process started or ended.
MIN_SECONDS = 5.0

#: The most seconds between two pages while the driver runs, so that the times it shows stay
#: current while a long process runs.
QUIET_SECONDS = 60.0


def write_page(run_dir: Path | str) -> Path:
    """Build the run's page from its files and write it at the run's root."""
    run_dir = Path(run_dir)
    config = load_recorded(run_dir)
    return page.write(run_dir / PAGE_FILE, page.render(gather.gather(run_dir, config)))


class Updater:
    """Keeps a run's page current while the driver runs: the scheduler's snapshot after each change,
    and the page as often as the module docstring says. `note` is told, once, when the page could
    not be built."""

    def __init__(self, run_dir: Path, note: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.run_dir, self.note, self.clock = Path(run_dir), note, clock
        self.written = float("-inf")
        self.failed = False

    def __call__(self, jobs, changed: bool) -> None:
        now = self.clock()
        if changed:
            self._snapshot(jobs)
        if (changed and now - self.written >= MIN_SECONDS) or now - self.written >= QUIET_SECONDS:
            if not changed:
                self._snapshot(jobs)
            self.page()

    def finish(self, jobs) -> None:
        """The invocation has ended: the last snapshot, marked as such, and the last page."""
        self._snapshot(jobs, finished=True)
        self.page()

    def page(self) -> None:
        try:
            write_page(self.run_dir)
        except Exception as error:
            # The page only reports, so no failure to build it may stop the run.
            if not self.failed:
                self.failed = True
                self.note(f"the status page could not be built: {error!r}\n"
                          + traceback.format_exc(limit=3))
        self.written = self.clock()

    def _snapshot(self, jobs, finished: bool = False) -> None:
        (self.run_dir / LOGS_DIR).mkdir(parents=True, exist_ok=True)
        files.write_json(self.run_dir / LOGS_DIR / JOBS_FILE, snapshot(jobs, finished))


def main(argv: list[str] | None = None) -> int:
    """Build a run's status page again from its files, and print where it is."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="the run's directory")
    args = parser.parse_args(argv)
    try:
        print(write_page(args.run_dir))
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    return 0
