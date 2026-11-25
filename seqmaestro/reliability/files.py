"""The names of the files an arm's panel writes, and how its summaries and arrays are read back.

`panel.py` documents the layout and writes it, whereas the tables, the fits and the reveal find its
files through the names defined here, so that each file name is spelled in one place.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from seqmaestro.reliability import tier2

PREPARE_FILE = "prepare.json"
PREPARE_ARRAYS = "prepare.npz"
PLANTED_FILE = "planted.json"
RECORD_FILE = "reliability.json"
LOCO_DIR = "loco"
TIER3_FILE = "tier3.json"
CHECKS_FILE = "checks.json"

#: The name that the in-sample Tier 2's files give to D, the rows it is computed on.
DISCOVERY = "discovery"

#: The label sets: the real labels, the global null (the real labels permuted), and the planted
#: signal.
REAL, NULL, PLANTED = "real", "null", "planted"


def _suffix(label_set: str) -> str:
    return "" if label_set == REAL else f"_{label_set}"


def selections_file(label_set: str) -> str:
    return f"selections{_suffix(label_set)}.npz"


def tier1_name(label_set: str) -> str:
    """Tier 1 under a label set, as `<name>.json` and `<name>.npz`."""
    return f"tier1{_suffix(label_set)}"


def tier2_name(split: str, label_set: str) -> str:
    """Tier 2 on a split under a label set, as `<name>.json` and `<name>.npz`."""
    return f"tier2_{split}{_suffix(label_set)}"


def loco_cell(label: str) -> str:
    """A refit's cell, relative to its arm's directory, named for what the refit drops, which must
    be a plain name."""
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", label) or label in {".", ".."}:
        raise ValueError(f"{label!r} cannot name a cell directory.")
    return f"{LOCO_DIR}/{label}"


def _plain(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    """Replace a JSON file whole, so an interruption never leaves half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    partial.replace(path)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def save_tier2(arm_dir: Path, name: str, result: dict, **about: Any) -> dict:
    """A Tier 2 result as a summary and its per-feature arrays. Returns the summary."""
    result = dict(result)
    arrays = {key: np.asarray(result.pop(key)) for key in tier2.arrays(result["target_kind"])}
    names = result.pop("names")
    np.savez_compressed(arm_dir / f"{name}.npz", names=np.array(names, dtype=str), **arrays)
    summary = {**about, **result}
    write_json(arm_dir / f"{name}.json", summary)
    return summary


def load_tier2(arm_dir: Path, name: str) -> dict:
    """A saved Tier 2 result, with its arrays and names restored."""
    arrays = load_arrays(arm_dir / f"{name}.npz")
    return {**read_json(arm_dir / f"{name}.json"), "names": [str(n) for n in arrays.pop("names")],
            **arrays}
