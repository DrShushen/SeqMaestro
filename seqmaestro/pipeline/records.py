"""The records a call writes beside its results, apart from its invocation record: its failures
and the environments it ran code in.

A call is one process of a run (`layout.call_name`). Every call writes these records in the same
way, so the code that writes them is defined once, here:

- `failures/<call>.json`: each cell or arm of the call that raised an exception, with its
  traceback. A call that ends without a failure removes the file an earlier attempt left, so the
  directory lists only what is still broken, and a run without failures has no file there.
- `environment/<name>/`: what was installed in each environment the call ran code in, written
  once per run.

The invocation record, `invocations/<call>.json`, is written by
`seqmaestro.provenance.record_call`, which the scripts that run outside Core use too.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Iterable
from pathlib import Path

from seqmaestro import arms as registry
from seqmaestro import files
from seqmaestro.pipeline.layout import ENVIRONMENT_DIR, FAILURES_DIR
from seqmaestro.pipeline.repository import REPO
from seqmaestro.provenance import snapshot_directory, write_snapshot


def failures_file(run_dir: str | Path, call: str) -> Path:
    return Path(run_dir) / FAILURES_DIR / f"{call}.json"


class Failures:
    """What failed in one call. A failed item must not stop the others, so each stage catches an
    item's exception, records it here and goes on to the next."""

    def __init__(self, run_dir: str | Path, call: str) -> None:
        self.path = failures_file(run_dir, call)
        self.found: list[dict[str, str]] = []

    def record(self, item: str) -> None:
        """Keep the exception being handled under the name of the item it stopped, and print its
        traceback. Call it from an `except` block."""
        error = traceback.format_exc()
        self.found.append({"item": item, "error": error})
        print(f"FAILED {item}\n{error}", file=sys.stderr, flush=True)

    def close(self) -> int:
        """Write what failed, or remove the file an earlier attempt of this call left, and return
        the call's exit status."""
        if not self.found:
            self.path.unlink(missing_ok=True)
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        files.write_json(self.path, self.found)
        print(f"\n{len(self.found)} failed; see {self.path}", flush=True)
        return 1


def snapshot_environments(run_dir: str | Path, models: Iterable[str] = (),
                          environments: Iterable[str] = ()) -> None:
    """Record what is installed in Core and in every other environment the call ran code in: the
    environment that fitting each of `models` uses (a vendored method's, or the one PySR's search
    runs in), and any named in `environments` (such as the MEME Suite's, for a motif discovery).
    An environment the run has already recorded is left as it is."""
    others = ({registry.spec(model).environment for model in models} | set(environments)) - {None}
    for environment in (registry.CORE_ENVIRONMENT, *sorted(others)):
        write_snapshot(environment, Path(run_dir) / ENVIRONMENT_DIR / snapshot_directory(environment),
                       repo_root=REPO, skip_if_present=True)
