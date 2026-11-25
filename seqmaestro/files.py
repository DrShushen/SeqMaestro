"""Writing a file in one step, so that a reader never finds half of it.

Several files of a run mark work as finished by their existence: a cell's `metrics.json`, an arm's
`reduction.json`, a panel's record. A process killed in the middle of writing one would leave half
a file under the name that says the work is done, and the next call would skip it. Every function
here therefore writes under a temporary name in the same directory and renames the result into
place, which the file system does in one step, so a reader sees either the earlier file or the new
one.

The temporary name holds the ID of the writing process, so two processes that write the same file
never write into each other's temporary file.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_bytes(data)
    os.replace(partial, path)
    return path


def write_text(path: str | Path, text: str) -> Path:
    return write_bytes(path, text.encode())


def write_json(path: str | Path, payload: Any, default: Callable[[Any], Any] | None = None) -> Path:
    """Write `payload` as indented JSON. `default` converts a value JSON cannot represent into one
    it can, as in `json.dumps`."""
    return write_text(path, json.dumps(payload, indent=2, default=default) + "\n")


def write_arrays(path: str | Path, **arrays: Any) -> Path:
    """Write the arrays as a compressed `.npz`, which `numpy.load` reads back."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return write_bytes(path, buffer.getvalue())
