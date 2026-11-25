"""What an analysis that asks a language model was computed from, so that it asks once.

Stability and co-discovery are arithmetic on the cells, so computing them again gives the same
tables, whereas the recurrence analysis and the judge ask a language model, whose answers cost
tokens and differ from one call to the next. The driver starts every analysis again whenever a run
that is not yet revealed is continued, which the process has an agent do after it has stopped
before the reveal and checked the run.

Each of the two therefore records a digest of what it was computed from, and a later call that
would be computed from the same inputs keeps what is on file, so the tables an agent checked are
the tables the report is written from. A changed feature file of a cell, a changed setting, a
changed description, or another model or reasoning effort gives another digest, so the analysis
asks again. The model and the effort change the digest because each analysis counts them among
its settings (`reply_conditions`). `--force` asks again whatever the digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from seqmaestro.harness import cell as cells

#: The key of an analysis's record under which its digest is kept.
KEY = "computed_from"


def digest(results: Path, settings: Mapping[str, Any], context: Any) -> str:
    """One digest of what a language model's answers depend on: the feature file of every cell
    under `results`, the analysis's `settings`, and the dataset's description (`context`). Every
    field of the description is hashed, whether or not the analysis's prompts show it."""
    hashed = hashlib.sha256()
    for path in sorted(results.glob(f"*/*/*/seed_*/{cells.FEATURES_FILE}")):
        hashed.update(str(path.relative_to(results)).encode())
        hashed.update(hashlib.sha256(path.read_bytes()).digest())
    stated = {"settings": dict(settings), "context": asdict(context)}
    hashed.update(json.dumps(stated, sort_keys=True, default=str).encode())
    return hashed.hexdigest()


def on_file(record: Path) -> str | None:
    """The digest the record on file states, or None where there is no record, or where the record
    states none because its analysis did not finish."""
    try:
        stated = json.loads(record.read_text())
    except (OSError, ValueError):
        return None
    return stated.get(KEY) if isinstance(stated, dict) else None
