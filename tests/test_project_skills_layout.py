from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_skills_are_single_sourced_for_agent_discovery() -> None:
    """Project skills are single-sourced under .agents/skills with .claude/skills
    as the Claude Code compatibility alias.

    Guards this PR's repository layout against drift between the portable agent
    and Claude Code skill discovery paths.
    """
    agent_skills = _REPO_ROOT / ".agents" / "skills"
    claude_skills = _REPO_ROOT / ".claude" / "skills"

    assert agent_skills.is_dir()
    assert claude_skills.is_symlink()
    assert claude_skills.readlink() == Path("../.agents/skills")
    assert claude_skills.resolve() == agent_skills.resolve()
    assert (agent_skills / "benchflow-experiment-review" / "SKILL.md").is_file()
