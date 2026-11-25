#!/usr/bin/env bash
# Start an AI agent in this repository as the SeqMaestro orchestrator. The agent checks the
# installation and asks where your data is and what you want to learn from it.
#
#     ./agent.sh claude           Claude Code
#     ./agent.sh codex            Codex
#     ./agent.sh openhands        OpenHands (Agent Canvas, in the browser)
#     ./agent.sh claude-science   Claude Science (in the browser)
#
# Claude Code and Codex open the conversation by themselves. The two that run in the browser cannot
# be handed a first message, so the script prints the steps and the message to type. Anything after
# the harness's name is passed to it, for example a model. Each harness runs on a language model of
# its own, set up in the harness, which is separate from the model SeqMaestro calls (.env).
# docs/user/agents.md says how to set up each one, and how to start any other harness.
set -euo pipefail
cd "$(dirname "$0")"

# The first message for a harness that reads AGENTS.md by itself, which tells the agent how to
# answer it. Claude Science reads no file of the folder, so it is told to read AGENTS.md.
OPENING="Start SeqMaestro."

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "$1"; }

installed() {  # command: whether it is installed, saying how to install it when it is not
    command -v "$1" > /dev/null && return 0
    echo "$1 is not installed. docs/user/agents.md says how to install it." >&2
    exit 1
}

steps() {  # first message, steps ...: the browser harness's steps, then the message to type
    local message=$1
    shift
    printf '\nIn the browser:\n'
    printf '%s\n' "$@" "  Then type: $message" ""
}

harness=${1:-}
[ $# -gt 0 ] && shift
case "$harness" in
    claude)
        installed claude
        exec claude "$@" "/seqmaestro" ;;
    codex)
        installed codex
        # Codex runs each command in a sandbox that may write only in this folder. PySR's Julia
        # takes a lock and keeps its compiled packages inside its environment, so those two folders
        # are opened to it as well.
        writable=()
        pysr=$(./install.sh --interpreters 2> /dev/null | sed -n 's/^PYSR=//p')
        if [ -n "$pysr" ] && [ -d "${pysr%/bin/python}/julia_env" ]; then
            writable=(--add-dir "${pysr%/bin/python}/julia_env" --add-dir "${pysr%/bin/python}/share/julia")
        fi
        exec codex ${writable[@]+"${writable[@]}"} "$@" "\$seqmaestro" ;;
    openhands)
        installed agent-canvas
        steps "$OPENING" \
              "  1. Open http://localhost:8000. The first time, set the agent's model under Settings > LLM." \
              "  2. Open Workspace > Add Workspace, choose $(pwd), keep Local, and confirm."
        exec agent-canvas "$@" ;;
    claude-science)
        installed claude-science
        steps "Read AGENTS.md in $(pwd) and follow it." \
              "  1. Sign in, the first time, with your Claude account." \
              "  2. Make a project, or open the one you made for SeqMaestro, and start a new session."
        # Its sandbox cannot write into a folder you give it, which every stage of a run does.
        exec claude-science serve --dangerously-no-sandbox "$@" ;;
    -h|--help) usage 0 ;;
    *) usage 2 ;;
esac
