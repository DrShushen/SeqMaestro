#!/usr/bin/env bash
# SeqMaestro: build the conda environments a run needs, and check them.
#
#     ./install.sh                build every environment, then check them
#     ./install.sh --check        build nothing: check what is there
#     ./install.sh --check-llm    make one call to the language model that .env names
#     ./install.sh --help         every option
#
# For Linux and macOS. It can be run again at any time: an environment that exists is kept, and
# its packages are brought up to what the repository declares. ENVIRONMENTS.md says what each
# environment holds and why they are kept apart, and README.md what to do after this.
#
# Written for the bash that macOS ships (3.2), so it uses no associative array and no `mapfile`.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# What the environments are built from. ENVIRONMENTS.md gives the reason for each pin.
PYTHON_VERSION=3.12
PYSR_VERSION=2.5.1
JULIA_VERSION=1.11.9
ALPHAGENOME_VERSION=0.9.0
# One channel, named in full, so that an environment is built from the same packages whichever
# conda a machine has, and no channel's terms have to be accepted first.
CHANNEL=conda-forge

# The environments, by the short name this script takes. A run of the shared config needs the
# first five. The sixth serves one dataset-preparation script and is built only when asked for.
DEFAULT_SET="core deft igsr pysr localutils"
EVERY_NAME="core deft igsr pysr localutils alphagenome"

environment_of() {
    case "$1" in
        core) echo SeqMaestro_Core ;;
        deft) echo SeqMaestro_Method_deft ;;
        igsr) echo SeqMaestro_Method_igsr ;;
        pysr) echo SeqMaestro_Method_pysr ;;
        localutils) echo SeqMaestro_LocalUtils ;;
        alphagenome) echo SeqMaestro_Method_alphagenome ;;
        *) return 1 ;;
    esac
}

# The shell variable the pages use for each environment's interpreter.
variable_of() {
    case "$1" in
        core) echo CORE ;;
        deft) echo DEFT ;;
        igsr) echo IGSR ;;
        pysr) echo PYSR ;;
        localutils) echo LOCALUTILS ;;
        alphagenome) echo AG ;;
    esac
}

# What each environment must be able to import to do its part.
imports_of() {
    case "$1" in
        core) echo "seqmaestro, sklearn, xgboost, interpret, litellm, psutil, yaml, pytest" ;;
        deft) echo "deft, sklearn, litellm, hydra, dill" ;;
        igsr) echo "igsr, sklearn, litellm, hydra" ;;
        pysr) echo "pysr" ;;
        localutils) echo "seqmaestro, matplotlib, openpyxl, PIL" ;;
        alphagenome) echo "alphagenome" ;;
    esac
}

# The variable naming the provider every language-model call goes to, the providers it can name,
# and the variables a call to each cannot be made without (.env.example). The list is
# seqmaestro/llm/client.py's, which a test holds this one to.
PROVIDER_VARIABLE=SEQMAESTRO_LLM_PROVIDER
PROVIDERS="azure openai claude"
credentials_of() {
    case "$1" in
        azure) echo "SEQMAESTRO_AZURE_OPENAI_KEY SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT SEQMAESTRO_AZURE_OPENAI_ENDPOINT" ;;
        openai) echo "SEQMAESTRO_OPENAI_KEY SEQMAESTRO_OPENAI_MODEL" ;;
        claude) echo "SEQMAESTRO_CLAUDE_KEY SEQMAESTRO_CLAUDE_MODEL" ;;
    esac
}

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

With no option it builds the five conda environments a run needs (SeqMaestro_Core,
SeqMaestro_Method_deft, SeqMaestro_Method_igsr, SeqMaestro_Method_pysr and
SeqMaestro_LocalUtils), copies .env.example to .env where there is no .env, and checks the
result. An environment that exists is kept and its packages are brought up to date.

  --only NAMES          build these only, separated by commas:
                        core, deft, igsr, pysr, localutils, alphagenome
  --skip NAMES          build every default environment but these
  --with-alphagenome    also build SeqMaestro_Method_alphagenome, which one dataset-preparation
                        script uses and no run needs
  --check               build nothing: check the environments, the credentials file and the
                        programs a PDF report needs
  --check-llm           make one call to the language model .env names, and say what came back
  --interpreters        print each environment's python as a shell assignment (CORE=..., DEFT=...),
                        for:  eval "$(./install.sh --interpreters)"
  --install-miniforge   where no conda is found, install Miniforge into ~/miniforge3 without asking
  --dry-run             print the commands and run none of them
  -y, --yes             answer yes to every question
  -h, --help            this text

A conda other than the one on the PATH is named with SEQMAESTRO_CONDA=/path/to/conda.
EOF
}

say() { printf '%s\n' "$*" >&2; }
step() { printf '\n==> %s\n' "$*" >&2; }
fail() { printf '\ninstall.sh: %s\n' "$*" >&2; exit 1; }

DRY_RUN=no
ASSUME_YES=no
INSTALL_MINIFORGE=no
ACTION=install
ONLY=""
SKIP=""
WITH_ALPHAGENOME=no

# Run a command, printing it first. Under --dry-run it is printed and not run.
run() {
    printf '+' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    if [ "$DRY_RUN" = no ]; then
        "$@"
    fi
}

ask() {
    if [ "$ASSUME_YES" = yes ]; then
        return 0
    fi
    if [ ! -r /dev/tty ]; then
        return 1
    fi
    local reply
    printf '%s [y/N] ' "$1" >&2
    read -r reply </dev/tty || return 1
    case "$reply" in
        y | Y | yes | Yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

# ------------------------------------------------------------------------------ the arguments

names_from() {
    printf '%s' "$1" | tr ',' ' '
}

while [ $# -gt 0 ]; do
    case "$1" in
        --only) [ $# -ge 2 ] || fail "--only needs the environments to build"; ONLY=$(names_from "$2"); shift ;;
        --only=*) ONLY=$(names_from "${1#--only=}") ;;
        --skip) [ $# -ge 2 ] || fail "--skip needs the environments to leave out"; SKIP=$(names_from "$2"); shift ;;
        --skip=*) SKIP=$(names_from "${1#--skip=}") ;;
        --with-alphagenome) WITH_ALPHAGENOME=yes ;;
        --check) ACTION=check ;;
        --check-llm) ACTION=check_llm ;;
        --interpreters) ACTION=interpreters ;;
        --install-miniforge) INSTALL_MINIFORGE=yes ;;
        --dry-run) DRY_RUN=yes ;;
        -y | --yes) ASSUME_YES=yes ;;
        -h | --help) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
    shift
done

for name in $ONLY $SKIP; do
    environment_of "$name" >/dev/null || fail "no environment is called '$name'. They are: $EVERY_NAME"
done

listed() {  # whether the word $1 is among the words of $2
    case " $2 " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

SELECTED=""
if [ -n "$ONLY" ]; then
    SELECTED=$ONLY
else
    for name in $DEFAULT_SET; do
        listed "$name" "$SKIP" || SELECTED="$SELECTED $name"
    done
    if [ "$WITH_ALPHAGENOME" = yes ]; then
        SELECTED="$SELECTED alphagenome"
    fi
fi

# ------------------------------------------------------------------------------ conda

CONDA=""

find_conda() {
    local candidate found
    if [ -n "${SEQMAESTRO_CONDA:-}" ]; then
        [ -x "$SEQMAESTRO_CONDA" ] || fail "SEQMAESTRO_CONDA names $SEQMAESTRO_CONDA, which is no program"
        CONDA=$SEQMAESTRO_CONDA
        return 0
    fi
    for candidate in conda mamba micromamba; do
        found=$(type -P "$candidate" 2>/dev/null || true)
        if [ -n "$found" ]; then
            CONDA=$found
            return 0
        fi
    done
    # A conda that was installed and never put on the PATH.
    for candidate in "${CONDA_EXE:-}" "${MAMBA_EXE:-}" "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
        /opt/homebrew/Caskroom/miniforge/base/bin/conda /opt/conda/bin/conda; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            CONDA=$candidate
            return 0
        fi
    done
    return 1
}

download() {  # url, file
    if type -P curl >/dev/null 2>&1; then
        run curl -fsSL -o "$2" "$1"
    elif type -P wget >/dev/null 2>&1; then
        run wget -q -O "$2" "$1"
    else
        fail "neither curl nor wget is installed, so nothing can be downloaded"
    fi
}

# Where Miniforge's installer is downloaded to. `leave_nothing_behind` removes it however the
# script ends.
DOWNLOADS=""

leave_nothing_behind() {
    if [ -n "$DOWNLOADS" ] && [ -d "$DOWNLOADS" ]; then
        rm -f "$DOWNLOADS/Miniforge3.sh"
        rmdir "$DOWNLOADS" 2>/dev/null || true
    fi
}
trap leave_nothing_behind EXIT

install_miniforge() {
    local target=$HOME/miniforge3 url
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    step "Installing Miniforge into $target"
    if [ "$DRY_RUN" = no ]; then
        DOWNLOADS=$(mktemp -d "${TMPDIR:-/tmp}/seqmaestro-install.XXXXXX")
    else
        DOWNLOADS="<a temporary directory>"
    fi
    # The installer refuses to run from a file whose name does not end in .sh.
    download "$url" "$DOWNLOADS/Miniforge3.sh"
    # Batch mode: it writes into the target alone and edits no shell start-up file.
    run bash "$DOWNLOADS/Miniforge3.sh" -b -p "$target"
    CONDA=$target/bin/conda
}

# A conda installed under a long path starts through `env python`, since a script's first line
# cannot hold a path that long. That finds conda's python only with conda's bin on the PATH, so
# it is put there for the commands of this script.
conda_on_path() {
    local root
    root=$(cd "$(dirname "$CONDA")/.." 2>/dev/null && pwd) || return 0
    if [ -x "$root/bin/python" ]; then
        PATH="$root/bin:$PATH"
        export PATH
    fi
}

need_conda() {
    if find_conda; then
        conda_on_path
        return 0
    fi
    if [ "$ACTION" != install ]; then
        fail "no conda was found. Run ./install.sh first, or name one with SEQMAESTRO_CONDA."
    fi
    say "No conda was found on this machine. SeqMaestro keeps each part in a conda environment."
    if [ "$INSTALL_MINIFORGE" = yes ] || ask "Install Miniforge (conda, with conda-forge) into $HOME/miniforge3 now?"; then
        install_miniforge
        conda_on_path
    else
        fail "install conda first, for example Miniforge (https://conda-forge.org/download/), or run:
    ./install.sh --install-miniforge"
    fi
}

# The prefix of a named environment, or nothing where this conda has none of that name. It never
# fails, and awk takes in the whole listing, so that neither a conda that cannot start nor a
# closed pipe ends the script here without a word.
prefix_of() {
    { "$CONDA" env list 2>/dev/null || true; } |
        awk -v name="$1" '$1 == name && !found { found = $NF } END { if (found) print found }'
}

# The same once the environment is built. Under --dry-run, where nothing was built, it stands for
# the place the environment would be.
prefix_for() {
    local prefix
    prefix=$(prefix_of "$1")
    if [ -z "$prefix" ]; then
        if [ "$DRY_RUN" = no ]; then
            fail "$1 was created, and '$CONDA env list' does not name it. Check that conda's
environments are kept under one directory (conda config --show envs_dirs)."
        fi
        prefix="<conda envs>/$1"
    fi
    printf '%s' "$prefix"
}

ensure_environment() {  # environment, further conda packages
    local environment=$1
    shift
    if [ -n "$(prefix_of "$environment")" ]; then
        say "$environment exists, and is kept."
        return 0
    fi
    run "$CONDA" create -y -n "$environment" --override-channels -c "$CHANNEL" \
        "python=$PYTHON_VERSION" pip "$@"
}

# pip in an environment, by its interpreter's path. Packages under the home directory and a
# PYTHONPATH of the calling shell are kept out, so that what is installed is what is declared.
pip_install() {  # prefix, pip's arguments
    local prefix=$1
    shift
    run env -u PYTHONPATH PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
        "$prefix/bin/python" -m pip install "$@"
}

# ------------------------------------------------------------------------------ the environments

build_core() {
    step "SeqMaestro_Core: the library, the models fitted in it, the driver and the tests"
    if [ "$(uname)" = Darwin ]; then
        # XGBoost's wheel for macOS loads the OpenMP runtime from outside itself.
        ensure_environment SeqMaestro_Core llvm-openmp
    else
        ensure_environment SeqMaestro_Core
    fi
    local prefix
    prefix=$(prefix_for SeqMaestro_Core)
    pip_install "$prefix" -e ".[models,llm,driver,dev]"
}

build_deft() {
    step "SeqMaestro_Method_deft: DEFT's tree, and scikit-learn for its adapter"
    ensure_environment SeqMaestro_Method_deft
    local prefix
    prefix=$(prefix_for SeqMaestro_Method_deft)
    pip_install "$prefix" -e "methods/deft[dev]" scikit-learn
}

build_igsr() {
    step "SeqMaestro_Method_igsr: IGSR"
    ensure_environment SeqMaestro_Method_igsr
    local prefix
    prefix=$(prefix_for SeqMaestro_Method_igsr)
    pip_install "$prefix" -e "methods/igsr[dev]"
}

# The variables of the calling shell a download may depend on, which `env -i` would drop.
kept_variables() {
    local name
    for name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy \
        SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE TMPDIR LANG; do
        if [ -n "${!name:-}" ]; then
            printf '%s=%s\n' "$name" "${!name}"
        fi
    done
}

# Run the PySR environment's python with Julia, its packages and its project inside the
# environment, and nothing of the calling shell that could point them elsewhere (ENVIRONMENTS.md).
in_pysr() {  # prefix, further VARIABLE=value settings ..., --, python's arguments
    local prefix=$1 line
    shift
    local settings=("HOME=$HOME" "PATH=$prefix/bin:/usr/bin:/bin"
        "JULIA_DEPOT_PATH=$prefix/share/julia" "PYTHON_JULIAPKG_PROJECT=$prefix/julia_env")
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            settings+=("$line")
        fi
    done <<EOF
$(kept_variables)
EOF
    while [ "$1" != -- ]; do
        settings+=("$1")
        shift
    done
    shift
    run env -i "${settings[@]}" "$prefix/bin/python" "$@"
}

pysr_imports() {  # prefix: whether PySR imports there with nothing left to download
    local prefix=$1
    DRY_RUN=no in_pysr "$prefix" "PYTHON_JULIAPKG_EXE=$prefix/julia_env/pyjuliapkg/install/bin/julia" \
        PYTHON_JULIAPKG_OFFLINE=yes -- -c 'import pysr' >/dev/null 2>&1
}

build_pysr() {
    step "SeqMaestro_Method_pysr: PySR, with Julia and its packages inside the environment"
    ensure_environment SeqMaestro_Method_pysr
    local prefix julia
    prefix=$(prefix_for SeqMaestro_Method_pysr)
    julia=$prefix/julia_env/pyjuliapkg/install/bin/julia
    pip_install "$prefix" "pysr==$PYSR_VERSION"
    if [ -x "$julia" ] && pysr_imports "$prefix"; then
        say "Julia $JULIA_VERSION and PySR's packages are in the environment already."
        return 0
    fi
    say "Julia is downloaded into the environment and PySR's packages are compiled there."
    say "This is the longest step, of several minutes."
    run mkdir -p "$prefix/share/julia" "$prefix/julia_env"
    in_pysr "$prefix" -- -c \
        "import juliapkg; juliapkg.require_julia('=$JULIA_VERSION'); juliapkg.resolve()"
    # One import compiles SymbolicRegression.jl, so that the first search of a run does not.
    in_pysr "$prefix" "PYTHON_JULIAPKG_EXE=$julia" -- -c 'import pysr'
}

build_localutils() {
    step "SeqMaestro_LocalUtils: the report's figures and spreadsheets"
    ensure_environment SeqMaestro_LocalUtils
    local prefix
    prefix=$(prefix_for SeqMaestro_LocalUtils)
    pip_install "$prefix" -e ".[localutils]"
}

build_alphagenome() {
    step "SeqMaestro_Method_alphagenome: AlphaGenome's client, for one dataset-preparation script"
    ensure_environment SeqMaestro_Method_alphagenome
    local prefix
    prefix=$(prefix_for SeqMaestro_Method_alphagenome)
    pip_install "$prefix" "alphagenome==$ALPHAGENOME_VERSION"
}

credentials_file() {
    if [ -f "$REPO/.env" ]; then
        say ".env is there, and is left as it is."
        return 0
    fi
    step "Credentials: .env made from .env.example, for you to fill in"
    run cp "$REPO/.env.example" "$REPO/.env"
    run chmod 600 "$REPO/.env"
}

# ------------------------------------------------------------------------------ the checks

PROBLEMS=0
problem() {
    PROBLEMS=$((PROBLEMS + 1))
    say "  PROBLEM  $*"
}
fine() { say "  ok       $*"; }
remark() { say "  note     $*"; }

check_environment() {  # short name
    local name=$1 environment prefix modules reason
    environment=$(environment_of "$name")
    prefix=$(prefix_of "$environment")
    modules=$(imports_of "$name")
    if [ -z "$prefix" ]; then
        problem "$environment is not built. Run: ./install.sh --only $name"
        return 0
    fi
    if [ "$name" = pysr ]; then
        if pysr_imports "$prefix"; then
            fine "$environment imports PySR, with Julia inside the environment"
        else
            problem "$environment cannot import PySR with what is inside it. Run: ./install.sh --only pysr"
        fi
        return 0
    fi
    # -I: what is imported is what the environment holds, and not this folder or the home directory.
    if reason=$("$prefix/bin/python" -I -c "import $modules" 2>&1); then
        fine "$environment imports $modules"
    else
        problem "$environment cannot import what it needs: $(printf '%s' "$reason" | tail -n 1)"
        if [ "$name" = core ] && [ "$(uname)" = Darwin ]; then
            remark "on macOS XGBoost needs the OpenMP runtime: brew install libomp, then run this check again"
        fi
    fi
}

# Python for `check_discovery`: why each named environment's interpreter would not be found.
DISCOVERY='
import sys
from seqmaestro.provenance import missing_python
missing = [reason for reason in map(missing_python, sys.argv[1:]) if reason]
sys.exit("\n".join(missing) if missing else 0)
'

# Whether a stage running under Core finds the other environments' interpreters, which it looks
# for beside its own (ENVIRONMENTS.md, "How the pipeline finds the other interpreters").
check_discovery() {
    local core name environment reason expected=""
    core=$(prefix_of SeqMaestro_Core)
    if [ -z "$core" ]; then
        return 0
    fi
    for name in $SELECTED; do
        if [ "$name" != core ] && [ -n "$(prefix_of "$(environment_of "$name")")" ]; then
            expected="$expected $(environment_of "$name")"
        fi
    done
    if [ -z "$expected" ]; then
        return 0
    fi
    # shellcheck disable=SC2086  # the names are passed as separate arguments
    if reason=$("$core/bin/python" -I -c "$DISCOVERY" $expected 2>&1); then
        fine "a stage under SeqMaestro_Core finds the other environments"
    else
        problem "a stage under SeqMaestro_Core would not find an environment it starts:"
        say "$reason"
    fi
    if (cd "$REPO" && "$core/bin/python" experiments/runners/run_experiment.py --help >/dev/null 2>&1); then
        fine "the driver starts (experiments/runners/run_experiment.py --help)"
    else
        problem "the driver does not start: $core/bin/python experiments/runners/run_experiment.py --help"
    fi
}

# What the language model's settings lack once .env is read: on the first line the provider it
# names, or nothing where it names none of PROVIDERS, and on the second that provider's required
# variables that are empty. Only names are printed, never a value.
language_model_settings() {
    (
        set +eu
        if [ -f "$REPO/.env" ]; then
            set -a
            # shellcheck disable=SC1091
            . "$REPO/.env" >/dev/null 2>&1
            set +a
        fi
        provider=""
        for name in $PROVIDERS; do
            if [ "$name" = "${!PROVIDER_VARIABLE:-}" ]; then
                provider=$name
            fi
        done
        printf '%s\n' "$provider"
        for name in $(credentials_of "$provider"); do
            if [ -z "${!name:-}" ]; then
                printf '%s ' "$name"
            fi
        done
    )
}

check_credentials() {
    local settings provider empty
    if [ ! -f "$REPO/.env" ]; then
        remark "there is no .env. Copy .env.example to .env and fill it in (README.md, step 2)"
        return
    fi
    settings=$(language_model_settings)
    provider=$(printf '%s\n' "$settings" | sed -n 1p)
    empty=$(printf '%s\n' "$settings" | sed -n 2p)
    if [ -z "$provider" ]; then
        remark "$PROVIDER_VARIABLE in .env does not name the language model's provider. Set it to one of: $PROVIDERS"
    elif [ -z "$empty" ]; then
        fine "the language model's provider is $provider, and its settings are filled in. Try them with: ./install.sh --check-llm"
    else
        remark "still empty in .env, for the provider $provider: $empty"
        remark "they are needed by the arms in which a language model writes features, and by the judge"
    fi
}

check_report_tools() {
    local missing="" tool
    for tool in pandoc xelatex; do
        type -P "$tool" >/dev/null 2>&1 || missing="$missing $tool"
    done
    if [ -z "$missing" ]; then
        fine "pandoc and xelatex are installed, for a report as a PDF"
    else
        remark "not installed:$missing. Only a report as a PDF needs them (ENVIRONMENTS.md)"
    fi
}

check_everything() {
    local name
    step "Checking"
    for name in $SELECTED; do
        check_environment "$name"
    done
    check_discovery
    check_credentials
    check_report_tools
    if [ "$PROBLEMS" -gt 0 ]; then
        say ""
        say "$PROBLEMS problem(s) above."
        return 1
    fi
    return 0
}

check_llm() {
    local core
    core=$(prefix_of SeqMaestro_Core)
    [ -n "$core" ] || fail "SeqMaestro_Core is not built. Run ./install.sh first."
    step "One call to the language model"
    (
        set +u
        if [ -f "$REPO/.env" ]; then
            set -a
            # shellcheck disable=SC1091
            . "$REPO/.env"
            set +a
        fi
        "$core/bin/python" -I - <<'EOF'
import sys

from seqmaestro.llm.client import LiteLLMClient, missing_settings

missing = missing_settings()
if missing:
    sys.exit(f"not set: {', '.join(missing)}. Fill them in .env (.env.example says what each is).")
client = LiteLLMClient(num_retries=0, request_timeout=60)
reply = client.complete("You reply with one word.", "Reply with the word: ready")
if not reply.ok:
    sys.exit(f"The call to {client.model} failed.\n{reply.error}")
print(f"{client.model} answered {reply.text.strip()!r}, for {client.usage()['total_tokens']} "
      f"tokens. The proposer, the judge, DEFT and IGSR all call this model.")
EOF
    )
}

print_interpreters() {
    local name prefix
    for name in $EVERY_NAME; do
        prefix=$(prefix_of "$(environment_of "$name")")
        if [ -n "$prefix" ]; then
            printf '%s=%q\n' "$(variable_of "$name")" "$prefix/bin/python"
        fi
    done
}

# ------------------------------------------------------------------------------ what is done

case "$(uname)" in
    Linux | Darwin) ;;
    *) fail "this script is for Linux and macOS. On Windows, use it inside WSL." ;;
esac
cd "$REPO"
[ -f pyproject.toml ] && [ -d methods/deft ] || fail "$REPO does not look like the SeqMaestro repository"

need_conda

case "$ACTION" in
    interpreters)
        print_interpreters
        exit 0
        ;;
    check_llm)
        check_llm
        exit 0
        ;;
    check)
        check_everything
        exit $?
        ;;
esac

say "SeqMaestro: building$SELECTED"
say "with $CONDA, from $REPO"
for name in $SELECTED; do
    "build_$name"
done
credentials_file

if [ "$DRY_RUN" = yes ]; then
    say ""
    say "Dry run: nothing was built."
    exit 0
fi

if ! check_everything; then
    fail "the environments are not all in order. The lines marked PROBLEM say what to do."
fi

if [ "$SECONDS" -lt 90 ]; then
    took="about a minute"
else
    took="$((SECONDS / 60)) minutes"
fi
cat >&2 <<EOF

The environments are built and checked, in $took.

Next (README.md has the detail):
  1. Fill in .env with your language model's provider and key, then try it:   ./install.sh --check-llm
  2. Start a run with Claude Code:   set -a && source .env && set +a && claude "/seqmaestro ..."
EOF
