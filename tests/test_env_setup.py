"""Tests for benchflow._env_setup — Dockerfile skills injection."""

from pathlib import Path
from unittest.mock import patch

from benchflow._env_setup import (
    _inject_skills_into_dockerfile,
    _get_agent_skill_paths,
)


def _make_task(tmp_path: Path, dockerfile_content: str = "FROM ubuntu:22.04\n") -> Path:
    """Create a minimal task directory with a Dockerfile."""
    task_path = tmp_path / "task"
    env_dir = task_path / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text(dockerfile_content)
    return task_path


def _make_skills_dir(tmp_path: Path) -> Path:
    """Create a skills directory with two dummy skills."""
    skills = tmp_path / "skills"
    for name in ("skill-a", "skill-b"):
        d = skills / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    return skills


class TestInjectSkillsIntoDockerfile:
    """_inject_skills_into_dockerfile(task_path, skills_dir)"""

    def test_copies_skills_and_appends_dockerfile(self, tmp_path):
        task_path = _make_task(tmp_path)
        skills_dir = _make_skills_dir(tmp_path)

        _inject_skills_into_dockerfile(task_path, skills_dir)

        # Skills copied into _deps/skills/
        deps = task_path / "environment" / "_deps" / "skills"
        assert deps.is_dir()
        assert (deps / "skill-a" / "SKILL.md").exists()
        assert (deps / "skill-b" / "SKILL.md").exists()

        # Dockerfile has COPY line
        content = (task_path / "environment" / "Dockerfile").read_text()
        assert "COPY _deps/skills /skills/" in content
        assert "# Skills directory (injected by benchflow --skills-dir)" in content

    def test_appends_symlink_lines_for_agents(self, tmp_path):
        task_path = _make_task(tmp_path)
        skills_dir = _make_skills_dir(tmp_path)

        _inject_skills_into_dockerfile(task_path, skills_dir)

        content = (task_path / "environment" / "Dockerfile").read_text()
        # Should have at least one RUN ln -sf line for agent skill paths
        assert "ln -sf /skills" in content
        assert "mkdir -p" in content

    def test_preserves_original_dockerfile_content(self, tmp_path):
        original = "FROM python:3.12\nRUN pip install flask\n"
        task_path = _make_task(tmp_path, original)
        skills_dir = _make_skills_dir(tmp_path)

        _inject_skills_into_dockerfile(task_path, skills_dir)

        content = (task_path / "environment" / "Dockerfile").read_text()
        assert content.startswith(original)

    def test_noop_when_no_dockerfile(self, tmp_path):
        task_path = tmp_path / "task"
        task_path.mkdir()
        # No environment/ dir at all
        skills_dir = _make_skills_dir(tmp_path)

        _inject_skills_into_dockerfile(task_path, skills_dir)
        # Should not crash, no files created
        assert not (task_path / "environment").exists()

    def test_noop_when_skills_dir_missing(self, tmp_path):
        task_path = _make_task(tmp_path)
        original = (task_path / "environment" / "Dockerfile").read_text()

        _inject_skills_into_dockerfile(task_path, tmp_path / "nonexistent")

        # Dockerfile unchanged
        assert (task_path / "environment" / "Dockerfile").read_text() == original

    def test_ignores_venv_and_pycache(self, tmp_path):
        task_path = _make_task(tmp_path)
        skills_dir = _make_skills_dir(tmp_path)
        # Add dirs that should be ignored
        (skills_dir / "__pycache__").mkdir()
        (skills_dir / "__pycache__" / "foo.pyc").write_bytes(b"")
        (skills_dir / ".venv").mkdir()
        (skills_dir / ".venv" / "bin").mkdir()

        _inject_skills_into_dockerfile(task_path, skills_dir)

        deps = task_path / "environment" / "_deps" / "skills"
        assert not (deps / "__pycache__").exists()
        assert not (deps / ".venv").exists()

    def test_overwrites_existing_deps_skills(self, tmp_path):
        task_path = _make_task(tmp_path)
        skills_dir = _make_skills_dir(tmp_path)

        # Pre-existing _deps/skills with stale content
        stale = task_path / "environment" / "_deps" / "skills" / "old-skill"
        stale.mkdir(parents=True)
        (stale / "SKILL.md").write_text("stale")

        _inject_skills_into_dockerfile(task_path, skills_dir)

        deps = task_path / "environment" / "_deps" / "skills"
        assert not (deps / "old-skill").exists()
        assert (deps / "skill-a").exists()

    def test_double_inject_duplicates_lines(self, tmp_path):
        """Calling inject twice appends duplicate COPY/RUN lines (known issue)."""
        task_path = _make_task(tmp_path)
        skills_dir = _make_skills_dir(tmp_path)

        _inject_skills_into_dockerfile(task_path, skills_dir)
        _inject_skills_into_dockerfile(task_path, skills_dir)

        content = (task_path / "environment" / "Dockerfile").read_text()
        assert content.count("COPY _deps/skills /skills/") == 2


class TestGetAgentSkillPaths:
    """_get_agent_skill_paths() -> list[str]"""

    def test_returns_home_based_paths_only(self):
        paths = _get_agent_skill_paths()
        for p in paths:
            assert p.startswith("/root/"), f"Expected /root/ prefix, got {p}"

    def test_no_duplicates(self):
        paths = _get_agent_skill_paths()
        assert len(paths) == len(set(paths))

    def test_no_workspace_paths(self):
        paths = _get_agent_skill_paths()
        for p in paths:
            assert "$WORKSPACE" not in p

    def test_with_mock_agents(self):
        """Verify behavior with controlled agent configs."""
        from benchflow.agents.registry import AgentConfig

        mock_agents = {
            "agent-a": AgentConfig(
                name="agent-a",
                install_cmd="true",
                launch_cmd="true",
                requires_env=[],
                skill_paths=["$HOME/.a/skills"],
            ),
            "agent-b": AgentConfig(
                name="agent-b",
                install_cmd="true",
                launch_cmd="true",
                requires_env=[],
                skill_paths=["$HOME/.a/skills", "$WORKSPACE/skills"],
            ),
        }
        with patch("benchflow._env_setup.AGENTS", mock_agents):
            paths = _get_agent_skill_paths()
        assert paths == ["/root/.a/skills"]
