"""Tests for benchflow._env_setup — Dockerfile skills injection and dep staging."""

from pathlib import Path
from unittest.mock import patch

from benchflow._env_setup import (
    _dep_local_name,
    _get_agent_skill_paths,
    _inject_skills_into_dockerfile,
    stage_dockerfile_deps,
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


class TestDepLocalName:
    def test_single_component(self):
        assert _dep_local_name("claw-gmail") == "claw-gmail"

    def test_nested_component(self):
        assert _dep_local_name("packages/environments/claw-gmail") == "claw-gmail"

    def test_generic_basename(self):
        assert _dep_local_name("tasks/email-foo/data") == "email-foo__data"

    def test_skills_basename(self):
        assert (
            _dep_local_name("tasks/email-foo/environment/skills")
            == "environment__skills"
        )


class TestStageDockerfileDeps:
    def test_copies_and_rewrites(self, tmp_path):
        """COPY with repo-root-relative path gets staged into _deps/."""
        # Create repo structure
        repo_root = tmp_path / "repo"
        pkg_dir = repo_root / "packages" / "claw-gmail"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "app.py").write_text("print('gmail')")

        # Create task
        task_dir = repo_root / "tasks" / "my-task"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text('version = "1.0"')
        (env_dir / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nCOPY packages/claw-gmail /app\nRUN echo hello\n"
        )

        stage_dockerfile_deps(task_dir, repo_root)

        # Check _deps was created and content preserved
        staged_file = env_dir / "_deps" / "claw-gmail" / "app.py"
        assert staged_file.exists()
        assert staged_file.read_text() == "print('gmail')"

        # Check Dockerfile was rewritten
        rewritten = (env_dir / "Dockerfile").read_text()
        assert "COPY _deps/claw-gmail /app" in rewritten
        assert "packages/claw-gmail" not in rewritten

    def test_skips_absolute_paths(self, tmp_path):
        """COPY with absolute source is left unchanged."""
        task_dir = tmp_path / "task"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        original = "FROM ubuntu:24.04\nCOPY /etc/hosts /hosts\n"
        (env_dir / "Dockerfile").write_text(original)

        stage_dockerfile_deps(task_dir, tmp_path)

        assert (env_dir / "Dockerfile").read_text() == original

    def test_skips_dot_source(self, tmp_path):
        """COPY . /app is left unchanged."""
        task_dir = tmp_path / "task"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        original = "FROM ubuntu:24.04\nCOPY . /app\n"
        (env_dir / "Dockerfile").write_text(original)

        stage_dockerfile_deps(task_dir, tmp_path)

        assert (env_dir / "Dockerfile").read_text() == original

    def test_no_dockerfile(self, tmp_path):
        """No-op when Dockerfile doesn't exist."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        # Should not raise
        stage_dockerfile_deps(task_dir, tmp_path)

    def test_multiple_copy_instructions(self, tmp_path):
        """Multiple COPY instructions in the same Dockerfile are all rewritten."""
        repo_root = tmp_path / "repo"
        dep1 = repo_root / "packages" / "dep1"
        dep1.mkdir(parents=True)
        (dep1 / "a.txt").write_text("aaa")
        dep2 = repo_root / "packages" / "dep2"
        dep2.mkdir(parents=True)
        (dep2 / "b.txt").write_text("bbb")

        task_dir = repo_root / "tasks" / "multi"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text('version = "1.0"')
        (env_dir / "Dockerfile").write_text(
            "FROM ubuntu:24.04\n"
            "COPY packages/dep1 /app/dep1\n"
            "RUN echo hi\n"
            "COPY packages/dep2 /app/dep2\n"
        )

        stage_dockerfile_deps(task_dir, repo_root)

        rewritten = (env_dir / "Dockerfile").read_text()
        lines = rewritten.split("\n")
        assert "_deps/" in lines[1]  # first COPY rewritten
        assert "_deps/" in lines[3]  # second COPY rewritten
        assert "packages/dep1" not in rewritten
        assert "packages/dep2" not in rewritten
        # Content preserved
        assert (env_dir / "_deps" / "dep1" / "a.txt").read_text() == "aaa"
        assert (env_dir / "_deps" / "dep2" / "b.txt").read_text() == "bbb"

    def test_source_not_found(self, tmp_path):
        """COPY with nonexistent source is left unchanged."""
        task_dir = tmp_path / "task"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        original = "FROM ubuntu:24.04\nCOPY nonexistent/path /app\n"
        (env_dir / "Dockerfile").write_text(original)

        stage_dockerfile_deps(task_dir, tmp_path)

        assert (env_dir / "Dockerfile").read_text() == original
