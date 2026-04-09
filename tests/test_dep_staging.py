"""Tests for Dockerfile dependency staging."""

from pathlib import Path

from benchflow._env_setup import stage_dockerfile_deps, _dep_local_name


class TestDepLocalName:
    def test_single_component(self):
        assert _dep_local_name("claw-gmail") == "claw-gmail"

    def test_nested_component(self):
        assert _dep_local_name("packages/environments/claw-gmail") == "claw-gmail"

    def test_generic_basename(self):
        assert _dep_local_name("tasks/email-foo/data") == "email-foo__data"

    def test_skills_basename(self):
        assert _dep_local_name("tasks/email-foo/environment/skills") == "environment__skills"


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
            "FROM ubuntu:24.04\n"
            "COPY packages/claw-gmail /app\n"
            "RUN echo hello\n"
        )

        stage_dockerfile_deps(task_dir, repo_root)

        # Check _deps was created
        assert (env_dir / "_deps" / "claw-gmail" / "app.py").exists()

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

    def test_source_not_found(self, tmp_path):
        """COPY with nonexistent source is left unchanged."""
        task_dir = tmp_path / "task"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        original = "FROM ubuntu:24.04\nCOPY nonexistent/path /app\n"
        (env_dir / "Dockerfile").write_text(original)

        stage_dockerfile_deps(task_dir, tmp_path)

        assert (env_dir / "Dockerfile").read_text() == original
