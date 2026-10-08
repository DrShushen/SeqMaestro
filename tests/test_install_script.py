"""`install.sh`, run against a stand-in for conda: which environments it builds and from what, what
its checks report, and that it prints no secret.

Nothing is installed. The script is copied into a folder that looks like the repository, beside a
`conda` that lists the environments a test names and records every call, so the script's real
`conda`, the real environments and the real `.env` are never touched. What the script builds for
real is tested by running it (`ENVIRONMENTS.md`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from seqmaestro import arms
from seqmaestro.llm.client import AZURE, ENV_DEPLOYMENT, ENV_ENDPOINT, ENV_KEY, ENV_PROVIDER, PROVIDERS
from seqmaestro.models import symbolic
from seqmaestro.pipeline import driver

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "install.sh"

#: The environments a run needs, as the code that starts them names them, by the script's short
#: name for each.
NEEDED = {
    "core": arms.CORE_ENVIRONMENT,
    "deft": arms.DEFT_ENVIRONMENT,
    "igsr": arms.IGSR_ENVIRONMENT,
    "pysr": symbolic.ENVIRONMENT,
    "localutils": driver.LOCALUTILS_ENVIRONMENT,
}

FAKE_CONDA = """#!/usr/bin/env bash
echo "conda $*" >> "$FAKE_LOG"
if [ "$1" = env ] && [ "$2" = list ]; then
    echo "# conda environments:"
    echo "base                  *  /fake/root"
    for name in $FAKE_ENVS; do
        echo "$name                 /fake/root/envs/$name"
    done
fi
"""


@pytest.fixture
def machine(tmp_path):
    """A folder that looks like the repository, with the script in it, and a home with nothing."""
    repo = tmp_path / "repo"
    (repo / "methods" / "deft").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "install.sh")
    (repo / "pyproject.toml").write_text("[project]\nname = 'stand-in'\n")
    (repo / ".env.example").write_text(f"{ENV_KEY}=\n")
    for folder in ("bin", "home"):
        (tmp_path / folder).mkdir()
    conda = tmp_path / "bin" / "conda"
    conda.write_text(FAKE_CONDA)
    conda.chmod(0o755)
    return tmp_path


def install(machine: Path, *arguments: str, built: str = "", conda: bool = True):
    """Run the script there. `built` names the environments the stand-in conda lists."""
    path = f"{machine / 'bin'}:/usr/bin:/bin" if conda else "/usr/bin:/bin"
    return subprocess.run(
        ["bash", str(machine / "repo" / "install.sh"), *arguments], capture_output=True, text=True,
        stdin=subprocess.DEVNULL, check=False,
        # A session of its own has no terminal, so the script cannot stop to ask a question.
        start_new_session=True,
        env={"PATH": path, "HOME": str(machine / "home"), "FAKE_ENVS": built,
             "FAKE_LOG": str(machine / "conda.log"), "TMPDIR": str(machine)})


def commands(result) -> str:
    """What a dry run printed, without the shell's quoting."""
    return result.stderr.replace("\\", "")


def test_the_script_parses_and_uses_nothing_the_bash_of_macos_lacks():
    """macOS ships bash 3.2, which has no associative array, no `mapfile` and no case-changing
    expansion, and which parses a here-document inside a command substitution badly."""
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
    assert os.access(SCRIPT, os.X_OK), "the README runs it as ./install.sh"
    code = "\n".join(line for line in SCRIPT.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    newer = r"declare -A|mapfile|readarray|\$\{\w+(?:,,|\^\^)\}|&>>|\|&|local -n|coproc|\[\[ -v "
    assert not re.findall(newer, code)
    assert not re.search(r"\$\([^()]*<<", code), "a here-document inside a command substitution"


def test_a_dry_run_builds_the_environments_the_code_starts(machine):
    """Each environment is created under the name the pipeline looks for, on the Python the
    package declares, and given the packages `ENVIRONMENTS.md` lists for it."""
    result = install(machine, "--dry-run")
    printed = commands(result)

    assert result.returncode == 0, result.stderr
    for environment in NEEDED.values():
        assert (f"create -y -n {environment} --override-channels -c conda-forge python=3.12 pip"
                in printed)
    assert "SeqMaestro_Core/bin/python -m pip install -e .[models,llm,driver,dev]" in printed
    assert ("SeqMaestro_Method_deft/bin/python -m pip install -e methods/deft[dev] scikit-learn"
            in printed)
    assert "SeqMaestro_Method_igsr/bin/python -m pip install -e methods/igsr[dev]" in printed
    assert "SeqMaestro_Method_pysr/bin/python -m pip install pysr==2.5.1" in printed
    assert "juliapkg.require_julia('=1.11.9'); juliapkg.resolve()" in printed
    assert "SeqMaestro_LocalUtils/bin/python -m pip install -e .[localutils]" in printed
    assert "alphagenome" not in printed, "no run needs it, so it is built only when asked for"
    assert "cp " in printed and ".env.example" in printed, "the credentials file is made"
    assert not (machine / "repo" / ".env").exists(), "and a dry run makes nothing"
    assert "create" not in (machine / "conda.log").read_text()


def test_the_python_built_is_one_the_package_accepts():
    declared = re.search(r'requires-python = ">=(\d+)\.(\d+)"', (REPO / "pyproject.toml").read_text())
    built = re.search(r"^PYTHON_VERSION=(\d+)\.(\d+)$", SCRIPT.read_text(), re.MULTILINE)
    assert tuple(map(int, built.groups())) >= tuple(map(int, declared.groups()))


def test_the_environments_to_build_can_be_chosen(machine):
    def created(*arguments: str) -> list[str]:
        return re.findall(r"create -y -n (\S+)", commands(install(machine, "--dry-run", *arguments)))

    assert created() == list(NEEDED.values())
    assert created("--only", "core,pysr") == [NEEDED["core"], NEEDED["pysr"]]
    assert created("--skip=pysr") == [name for key, name in NEEDED.items() if key != "pysr"]
    assert created("--with-alphagenome")[-1] == "SeqMaestro_Method_alphagenome"

    refused = install(machine, "--only", "everything")
    assert refused.returncode == 1 and "no environment is called 'everything'" in refused.stderr


def test_an_environment_that_exists_is_kept_and_brought_up_to_date(machine):
    result = install(machine, "--dry-run", "--only", "core", built=NEEDED["core"])
    printed = commands(result)
    assert f"{NEEDED['core']} exists, and is kept." in printed
    assert "create -y" not in printed
    assert f"/fake/root/envs/{NEEDED['core']}/bin/python -m pip install -e" in printed


def test_the_interpreters_are_printed_for_the_environments_that_exist(machine):
    """`eval "$(./install.sh --interpreters)"` is how the pages get `$CORE` and the others, so the
    assignments are all that goes to standard output."""
    result = install(machine, "--interpreters", built=f"{NEEDED['core']} {NEEDED['deft']}")
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        f"CORE=/fake/root/envs/{NEEDED['core']}/bin/python",
        f"DEFT=/fake/root/envs/{NEEDED['deft']}/bin/python"]


def test_the_check_says_what_is_missing_and_how_to_build_it(machine):
    result = install(machine, "--check", built=NEEDED["deft"])
    assert result.returncode == 1
    assert f"PROBLEM  {NEEDED['core']} is not built. Run: ./install.sh --only core" in result.stderr
    assert f"PROBLEM  {NEEDED['deft']} cannot import what it needs" in result.stderr, (
        "an environment that is listed and has no working python is a problem too")
    assert "create" not in (machine / "conda.log").read_text(), "a check builds nothing"


def test_the_check_names_the_empty_credentials_and_prints_no_value(machine):
    secret = "a-key-that-must-not-be-printed"
    env = machine / "repo" / ".env"
    env.write_text(f"{ENV_PROVIDER}=azure\n{ENV_KEY}={secret}\n{ENV_ENDPOINT}=\n")
    result = install(machine, "--check")
    assert (f"still empty in .env, for the provider azure: {ENV_DEPLOYMENT} {ENV_ENDPOINT}"
            in result.stderr)
    assert secret not in result.stdout + result.stderr

    env.unlink()
    assert "there is no .env" in install(machine, "--check").stderr


@pytest.mark.parametrize("named", ["", "azure, openai", "gemini", "a-key-pasted-into-the-wrong-line"])
def test_the_check_asks_for_a_provider_where_env_names_none_of_them(machine, named):
    """The variable's value is never printed either: a key pasted into the wrong line stays out
    of the output."""
    (machine / "repo" / ".env").write_text(f"{ENV_PROVIDER}='{named}'\n{ENV_KEY}=x\n")
    result = install(machine, "--check")
    assert (f"{ENV_PROVIDER} in .env does not name the language model's provider. Set it to one of: "
            f"{' '.join(PROVIDERS)}") in result.stderr
    assert not named or named not in result.stdout + result.stderr


def test_the_check_knows_each_providers_settings_as_the_client_does(machine):
    """The script lists each provider's required variables itself, since its check must work
    before any environment is built, so the two lists are held to each other here."""
    listed = re.search(r'^PROVIDERS="(.*)"$', SCRIPT.read_text(), re.MULTILINE).group(1).split()
    assert listed == list(PROVIDERS)
    env = machine / "repo" / ".env"
    for name, provider in PROVIDERS.items():
        env.write_text(f"{ENV_PROVIDER}={name}\n")
        assert (f"still empty in .env, for the provider {name}: {' '.join(provider.required)}"
                in install(machine, "--check").stderr)
        env.write_text(f"{ENV_PROVIDER}={name}\n" + "".join(f"{v}=x\n" for v in provider.required))
        assert (f"the language model's provider is {name}, and its settings are filled in"
                in install(machine, "--check").stderr)
    assert AZURE.required == (ENV_KEY, ENV_DEPLOYMENT, ENV_ENDPOINT)


def test_without_conda_the_script_says_so_and_can_install_miniforge(machine):
    """Nothing is installed without the user's word: a check refuses, a build stops with the two
    ways on, and `--install-miniforge` is the word."""
    assert "no conda was found" in install(machine, "--check", conda=False).stderr

    stopped = install(machine, conda=False)
    assert stopped.returncode == 1
    assert "./install.sh --install-miniforge" in stopped.stderr

    result = install(machine, "--install-miniforge", "--dry-run", conda=False)
    printed = commands(result)
    home = machine / "home"
    assert result.returncode == 0, result.stderr
    assert re.search(r"conda-forge/miniforge/releases/latest/download/Miniforge3-\w+-\w+\.sh", printed)
    assert f"Miniforge3.sh -b -p {home}/miniforge3" in printed
    assert f"{home}/miniforge3/bin/conda create -y -n {NEEDED['core']}" in printed
    assert not (home / "miniforge3").exists()
