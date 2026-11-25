"""The instructions an agent harness finds in the repository, and the script that starts one.

Every harness is pointed at one skill and the stage files it names (`process/`): `AGENTS.md`, which
most harnesses read at the start of a session, names the skill's path, Codex and OpenHands find the
skill under `.agents/skills/`, and Claude Code under `.claude/skills/`, through a link to the same
folder. These tests check that each of those pointers leads to the one skill, and that `agent.sh`
says how it is used.
"""

from __future__ import annotations

import re
import subprocess

from seqmaestro.pipeline.repository import REPO

SKILL = REPO / ".agents" / "skills" / "seqmaestro"


def frontmatter(path) -> dict[str, str]:
    """The `key: value` lines between the two `---` lines that open a skill."""
    block = path.read_text().split("---")[1]
    return dict(line.split(": ", 1) for line in block.strip().splitlines())


def test_there_is_one_skill_and_claude_codes_folder_links_to_it():
    link = REPO / ".claude" / "skills" / "seqmaestro"
    assert link.is_symlink(), "a copy would drift from the skill the other harnesses read"
    assert link.resolve() == SKILL.resolve()


def test_the_skill_is_named_for_its_folder_as_the_agent_skills_format_asks():
    fields = frontmatter(SKILL / "SKILL.md")
    assert fields["name"] == SKILL.name
    assert 0 < len(fields["description"]) <= 1024


def test_the_skill_holds_nothing_one_harness_alone_would_fill_in():
    """`$ARGUMENTS` is Claude Code's placeholder for the user's words, which any other harness would
    show to the agent as it is written."""
    assert "$ARGUMENTS" not in (SKILL / "SKILL.md").read_text()


def test_agents_md_names_the_skill_and_every_path_it_names_exists():
    text = (REPO / "AGENTS.md").read_text()
    assert ".agents/skills/seqmaestro/SKILL.md" in text
    for path in re.findall(r"`([\w./-]+/[\w./-]*)`", text):
        assert (REPO / path).exists(), f"AGENTS.md names {path}, which does not exist"


def test_the_launcher_names_every_harness_it_starts_and_refuses_any_other():
    shown = subprocess.run([str(REPO / "agent.sh"), "--help"], capture_output=True, text=True)
    assert shown.returncode == 0
    for harness in ("claude", "codex", "openhands", "claude-science"):
        assert f"./agent.sh {harness}" in shown.stdout
    unknown = subprocess.run([str(REPO / "agent.sh"), "another"], capture_output=True, text=True)
    assert unknown.returncode == 2
