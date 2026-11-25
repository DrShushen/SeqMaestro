"""Recording what produced a result: the environment and the command.

A number in a report can be trusted only as far as its origin is known, and a record of that origin
kept by hand goes stale. This module writes both halves of the record automatically.

**The environment.** The record has four layers, and the snapshot written here is the third:

1. **Declaration**: `pyproject.toml`, per package. What a thing *needs*.
2. **Recipe**: `ENVIRONMENTS.md`. Which environments exist, how to build them, and the
   constraints that keep them apart.
3. **Snapshot**: `write_snapshot`. What was *actually installed* when something ran. This can
   differ from the declaration, because an environment keeps a package its declaration has dropped
   until the environment is rebuilt. An environment with a Julia runtime inside it also has its
   Julia project recorded, since pip does not see the Julia packages.
4. **Attribution**: the manifest of each snapshot (`environment.json`), kept in the run beside its
   results, so that a number can be traced to the environment that produced it.

**The command.** `record_invocation` writes the command line, the resolved arguments and the
commit the code was at. It also records **whether the working tree was dirty**. A commit SHA
taken against modified files does not identify the code that ran, so a result produced that way
cannot be reproduced by anyone, including its author. Every stage of a run records its call itself
(`record_call`), so the record does not depend on a person remembering to make it.

A snapshot can also be written from the command line:

    python -m seqmaestro.provenance SeqMaestro_Core experiments/runs/<date>_<name>/environment/core

Run it with the interpreter of the environment being recorded. Called from another environment's
interpreter, `write_snapshot` runs this module under the named environment's interpreter, so a
stage running in Core can record the environment a model's search ran in (`snapshot_directory`
names its directory beside Core's).

**Which interpreter runs another environment's code.** A stage that shells out to DEFT's
environment, or the report's figure script to `SeqMaestro_LocalUtils`, asks `environment_python`
for the interpreter instead of holding an absolute path. The interpreter is found under the running
interpreter's conda root, and an environment variable overrides it on a machine laid out
differently.

*(`deft.provenance` in the vendored method is unrelated: it measures how much of a fitted tree the
language model contributed.)*
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from seqmaestro.execution import allowed_threads


def _conda_root(prefix: Path) -> Path:
    """The conda installation an interpreter belongs to: two levels up from a named environment
    (`<root>/envs/<name>`), or the prefix itself for the base environment."""
    return prefix.parent.parent if prefix.parent.name == "envs" else prefix


CONDA_ROOT = _conda_root(Path(sys.prefix))

#: Where an environment with a Julia runtime keeps its Julia project, relative to the environment's
#: prefix. The environment's recipe puts it there (ENVIRONMENTS.md), so that Julia's packages are
#: recorded and removed with the environment instead of living under the home directory.
JULIA_PROJECT = "julia_env"


def python_override_variable(environment: str) -> str:
    """The environment variable that points `environment_python` elsewhere, e.g.
    `SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_DEFT`."""
    return "SEQMAESTRO_PYTHON_" + "".join(c if c.isalnum() else "_" for c in environment).upper()


def environment_python(environment: str) -> Path:
    """The interpreter of a conda environment on this machine.

    It is `<conda root>/envs/<environment>/bin/python`, under the running interpreter's conda root,
    unless `python_override_variable(environment)` names another file. The path is not checked
    here, so a module can name an interpreter a machine does not have without failing to import.
    The caller checks that it exists before running it, as the driver's preflight does, which also
    names the variable to set.
    """
    override = os.environ.get(python_override_variable(environment))
    if override:
        return Path(override)
    return CONDA_ROOT / "envs" / environment / "bin" / "python"


def missing_python(environment: str) -> str | None:
    """Why `environment`'s interpreter cannot run here, or None when it can."""
    path = environment_python(environment)
    if path.exists():
        return None
    return (f"missing {environment} interpreter: {path}. Build the environment (./install.sh), "
            f"or set {python_override_variable(environment)} to its python.")


def _run(command: list[str]) -> str | None:
    """Return a command's stdout, or None where the command cannot run or fails, so that a missing
    conda is not fatal."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def pip_freeze() -> str:
    return _run([sys.executable, "-m", "pip", "freeze"]) or ""


def conda_export(env_name: str) -> str | None:
    conda = CONDA_ROOT / "bin" / "conda"
    if not conda.exists():
        return None
    return _run([str(conda), "env", "export", "-n", env_name, "--no-builds"])


def git_sha(repo_root: Path) -> str | None:
    return (_run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]) or "").strip() or None


def git_is_dirty(repo_root: Path) -> bool | None:
    """Whether tracked files differ from the recorded commit. None when git cannot answer.

    The commit alone does not identify the code: runs are usually made while something is
    half-edited, and a SHA recorded against a modified tree points at code that differs from what
    ran.
    """
    status = _run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"])
    return None if status is None else bool(status.strip())


def record_invocation(
    out_dir: Path | str,
    repo_root: Path | None = None,
    arguments: dict | None = None,
    *,
    filename: str,
    warn: bool = True,
    **extra,
) -> dict:
    """Write a record of the command, the code it ran and the conditions it ran under into
    `out_dir`.

    Each command calls this itself. `arguments` should be the *resolved* argument namespace,
    because an argument left at its default does not appear on the command line, so the record
    would not otherwise show the value it took.

    `filename` keeps apart the records of commands that write into one directory, so that a second
    command does not overwrite the first's record. It has no default, so every command names its
    record.
    """
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty = git_is_dirty(repo_root)
    record = {
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "arguments": {key: str(value) for key, value in (arguments or {}).items()},
        "git_sha": git_sha(repo_root),
        "git_dirty": dirty,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "executable": sys.executable,
        "seqmaestro_version": _version(),
        # The threads its fits were given (`seqmaestro.execution`), or None for every core.
        "threads": allowed_threads(),
        **extra,
    }
    (out_dir / filename).write_text(json.dumps(record, indent=2) + "\n")

    if warn and dirty:
        print(
            "WARNING: the working tree has uncommitted changes to tracked files. The commit "
            f"recorded ({record['git_sha'][:12] if record['git_sha'] else 'unknown'}) does not "
            "describe the code that just ran, and this result cannot be reproduced from it.",
            file=sys.stderr,
        )
    return record


#: Where a run keeps the invocation record of each of its calls.
INVOCATIONS_DIR = "invocations"


def record_call(run_dir: Path | str, call: str, arguments: dict, **extra) -> dict:
    """Write the invocation record of one call of a run, `invocations/<call>.json`. A call is one
    process of the run, and `call` is its name (`seqmaestro.pipeline.layout.call_name`), which its
    log and its failures file share. `arguments` are the parsed arguments, so a flag left out of the
    command shows the value it took."""
    return record_invocation(Path(run_dir) / INVOCATIONS_DIR, arguments=arguments,
                             filename=f"{call}.json", call=call, **extra)


def _version() -> str | None:
    try:
        from seqmaestro import __version__
    except ImportError:  # pragma: no cover (only if the package is broken)
        return None
    return __version__


def snapshot_directory(environment: str) -> str:
    """The directory under a run's `environment/` that holds an environment's snapshot: `core` for
    `SeqMaestro_Core`, `pysr` for `SeqMaestro_Method_pysr`."""
    return environment.removeprefix("SeqMaestro_Method_").removeprefix("SeqMaestro_").lower()


def write_snapshot(
    env_name: str, out_dir: Path, repo_root: Path | None = None, skip_if_present: bool = False
) -> dict:
    """Write the snapshot files and return the manifest that was written alongside them.

    A snapshot is a directory of files:

    - `pip-freeze.txt`: every package pip sees in the environment, with its version.
    - `conda-environment.yml`: the environment as conda exports it, where conda can export it.
    - `python-version.txt`: the interpreter's version.
    - `julia/`: the `Project.toml` and `Manifest.toml` of the environment's Julia project, for an
      environment that has one (`snapshot_julia`).
    - `environment.json`: the manifest, which names the environment, its interpreter, the platform
      and the commit, with a digest of the freeze, so that two snapshots can be compared without
      reading their package lists.

    A snapshot records what is installed in the environment it runs in. So when the named
    environment's interpreter exists and is not the one running this function, that interpreter
    runs this module to take the snapshot.

    With `skip_if_present`, an existing manifest is returned and nothing is rewritten. It is for
    invocations that run **concurrently into one run directory**, which would otherwise race on
    fixed filenames, each spending seconds shelling out to `pip` and `conda` to produce the same
    snapshot of the same environment.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    manifest_path = out_dir / "environment.json"
    if skip_if_present and manifest_path.exists():
        return json.loads(manifest_path.read_text())

    python = environment_python(env_name)
    if python.exists() and python.resolve().parent.parent != Path(sys.prefix).resolve():
        subprocess.run([str(python), "-m", "seqmaestro.provenance", env_name, str(out_dir)],
                       cwd=repo_root, check=True, capture_output=True, timeout=600)
        return json.loads(manifest_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)

    freeze = pip_freeze()
    (out_dir / "pip-freeze.txt").write_text(freeze)
    (out_dir / "python-version.txt").write_text(platform.python_version() + "\n")

    conda_yaml = conda_export(env_name)
    if conda_yaml is not None:
        (out_dir / "conda-environment.yml").write_text(conda_yaml)

    manifest = {
        "environment": env_name,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        # Identifies the exact package set without needing to diff two freeze files by eye.
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "n_packages": len([line for line in freeze.splitlines() if line and not line.startswith("#")]),
        "seqmaestro_git_sha": git_sha(repo_root),
        "conda_export_available": conda_yaml is not None,
        "julia": snapshot_julia(Path(sys.prefix), out_dir),
    }
    (out_dir / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def snapshot_julia(prefix: Path, out_dir: Path) -> dict | None:
    """Copy the Julia project of the environment at `prefix` into `out_dir / "julia"`, and return
    what identifies it, or None for an environment without a Julia project.

    The Julia packages an environment uses are fixed by its project's `Manifest.toml`, which pip
    does not see, so the freeze alone would leave them unrecorded. `Project.toml` states what was
    asked for, `Manifest.toml` what was resolved, and the manifest names the Julia version that
    resolved it.
    """
    project = prefix / JULIA_PROJECT
    if not (project / "Project.toml").exists():
        return None
    copies = out_dir / "julia"
    copies.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in ("Project.toml", "Manifest.toml"):
        if (project / name).exists():
            text = (project / name).read_text()
            (copies / name).write_text(text)
            digests[name] = hashlib.sha256(text.encode()).hexdigest()
    version = None
    if "Manifest.toml" in digests:
        version = tomllib.loads((project / "Manifest.toml").read_text()).get("julia_version")
    return {"project": str(project), "julia_version": version,
            "manifest_sha256": digests.get("Manifest.toml")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_name", help="conda environment name, e.g. SeqMaestro_Core")
    parser.add_argument("out_dir", type=Path, help="directory to write the snapshot into")
    args = parser.parse_args(argv)

    manifest = write_snapshot(args.env_name, args.out_dir)
    julia = "" if manifest["julia"] is None else f", julia {manifest['julia']['julia_version']}"
    print(f"Recorded {manifest['environment']} (python {manifest['python']}, "
          f"{manifest['n_packages']} packages{julia}) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
