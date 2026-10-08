"""Writing a file whole.

Several files of a run mark work as finished by existing: a cell's `metrics.json`, an arm's
`reduction.json`, a panel's record. A process killed in the middle of writing one would leave half
a file under the name that says the work is done, and the next call would skip it. Every function
here therefore writes under a temporary name in the same directory and renames the result into
place, which the file system does in one step. A reader sees the earlier file or the new one.

The temporary name holds the number of the writing process, so two processes that write the same
file cannot write into each other's.
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
    """`payload` as indented JSON. `default` turns a value JSON has no form for into one that it
    has, as in `json.dumps`."""
    return write_text(path, json.dumps(payload, indent=2, default=default) + "\n")


def write_arrays(path: str | Path, **arrays: Any) -> Path:
    """The arrays as a compressed `.npz`, read back with `numpy.load`."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return write_bytes(path, buffer.getvalue())
