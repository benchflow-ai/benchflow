"""Tests for Dockerfile dependency staging."""

from benchflow._env_setup import _dep_local_name, stage_dockerfile_deps


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
