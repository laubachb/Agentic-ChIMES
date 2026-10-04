"""Guards the Claude Code agent setup (CLAUDE.md, .claude/skills, .claude/agents,
.claude/settings.json) against rot: a skill with broken frontmatter is
silently never loaded, and a CLAUDE.md that stops listing a stage misleads
every future session."""

import json
import re
from pathlib import Path

import pytest
import yaml

from agentic_chimes import cli

ROOT = Path(__file__).resolve().parents[2]
CLAUDE_MD = ROOT / "CLAUDE.md"
SKILLS_DIR = ROOT / ".claude" / "skills"
AGENTS_DIR = ROOT / ".claude" / "agents"
SETTINGS = ROOT / ".claude" / "settings.json"

VALID_TOOLS = {"Bash", "Read", "Grep", "Glob", "Edit", "Write", "WebFetch", "WebSearch"}


def _frontmatter(path: Path) -> dict:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, f"{path} has no YAML frontmatter"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict), f"{path} frontmatter is not a mapping"
    return data


def _skill_files():
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def _agent_files():
    return sorted(AGENTS_DIR.glob("*.md"))


def test_setup_files_exist():
    assert CLAUDE_MD.is_file()
    assert _skill_files(), "no skills found"
    assert _agent_files(), "no subagents found"


@pytest.mark.parametrize("path", _skill_files(), ids=lambda p: p.parent.name)
def test_skill_frontmatter(path):
    fm = _frontmatter(path)
    assert fm.get("name") == path.parent.name, "skill name must match its directory"
    desc = fm.get("description", "")
    assert len(desc) > 60, "description is what triggers the skill; make it specific"


@pytest.mark.parametrize("path", _agent_files(), ids=lambda p: p.stem)
def test_agent_frontmatter(path):
    fm = _frontmatter(path)
    assert fm.get("name") == path.stem, "agent name must match its filename"
    assert len(fm.get("description", "")) > 60
    tools = {t.strip() for t in str(fm.get("tools", "")).split(",") if t.strip()}
    assert tools, "agents should declare an explicit tool allowlist"
    assert tools <= VALID_TOOLS, f"unknown tools: {tools - VALID_TOOLS}"
    assert path.read_text().split("---", 2)[2].strip(), "agent has no system prompt body"


def test_claude_md_covers_every_stage():
    text = CLAUDE_MD.read_text()
    missing = []
    import importlib

    for module_name in cli.STAGE_MODULE_NAMES:
        stage = importlib.import_module(f"agentic_chimes.stages.{module_name}").NAME
        if f"`{stage}`" not in text:
            missing.append(stage)
    assert not missing, f"CLAUDE.md does not mention stage(s): {missing}"


def test_claude_md_references_only_existing_skills_and_agents():
    text = CLAUDE_MD.read_text()
    skills = {p.parent.name for p in _skill_files()}
    agents = {p.stem for p in _agent_files()}
    referenced = set(re.findall(r"`(chimes-[a-z-]+)`", text))
    referenced.discard("chimes-agent")
    unknown = referenced - skills - agents
    assert not unknown, f"CLAUDE.md references undefined skills/agents: {unknown}"
    for name in skills | agents:
        assert f"`{name}`" in text, f"{name} exists but CLAUDE.md never mentions it"


def test_skills_reference_only_existing_skills_and_agents():
    skills = {p.parent.name for p in _skill_files()}
    agents = {p.stem for p in _agent_files()}
    known = skills | agents
    for path in _skill_files() + _agent_files():
        for ref in set(re.findall(r"`(chimes-[a-z-]+)`", path.read_text())) - {"chimes-agent"}:
            assert ref in known, f"{path.name} references undefined `{ref}`"


def test_skills_mention_only_real_stages():
    import importlib

    real = {importlib.import_module(f"agentic_chimes.stages.{m}").NAME for m in cli.STAGE_MODULE_NAMES}
    for path in _skill_files() + _agent_files():
        for stage in re.findall(r"chimes-agent ([a-z][a-z-]+)", path.read_text()):
            assert stage in real, f"{path.name}: `chimes-agent {stage}` is not a stage"


def test_settings_json_is_wellformed():
    data = json.loads(SETTINGS.read_text())
    perms = data["permissions"]
    assert set(perms) <= {"allow", "ask", "deny"}
    for rules in perms.values():
        assert all(isinstance(r, str) and "(" in r for r in rules)
    assert "Edit(codes/**)" in perms["deny"], "vendored forks must stay protected"
