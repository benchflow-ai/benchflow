"""Tests for benchflow.tasks — task authoring (check and init)."""

import shutil
from pathlib import Path

import pytest

from benchflow.tasks import check_task, init_task


class TestCheckTask:
    """check_task(task_dir) -> list[str]"""

    def _make_valid_task(self, tmp_path: Path) -> Path:
        """Create a minimal valid task directory."""
        task = tmp_path / "my-task"
        task.mkdir()
        (task / "task.toml").write_text(
            '[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n'
        )
        (task / "instruction.md").write_text("# Do something\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        tests = task / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        return task

    def test_valid_task_no_issues(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        assert check_task(task) == []

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "not-a-dir"
        f.write_text("hi")
        issues = check_task(f)
        assert issues == [f"Not a directory: {f}"]

    def test_nonexistent_path(self, tmp_path):
        p = tmp_path / "nonexistent"
        issues = check_task(p)
        assert issues == [f"Not a directory: {p}"]

    def test_missing_task_toml(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").unlink()
        issues = check_task(task)
        assert "Missing required file: task.toml" in issues

    def test_missing_instruction_md(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "instruction.md").unlink()
        issues = check_task(task)
        assert "Missing required file: instruction.md" in issues

    def test_missing_environment_dir(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        shutil.rmtree(task / "environment")
        issues = check_task(task)
        assert "Missing required directory: environment/" in issues

    def test_missing_dockerfile(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "environment" / "Dockerfile").unlink()
        issues = check_task(task)
        assert "Missing environment/Dockerfile" in issues

    def test_missing_tests_dir(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        shutil.rmtree(task / "tests")
        issues = check_task(task)
        assert "Missing tests/ directory (verifier needs test.sh or evaluate.py)" in issues

    def test_empty_tests_dir(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        for f in (task / "tests").iterdir():
            f.unlink()
        issues = check_task(task)
        assert "tests/ directory is empty" in issues

    def test_empty_instruction_md(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "instruction.md").write_text("")
        issues = check_task(task)
        assert "instruction.md is empty" in issues

    def test_task_toml_missing_agent_section(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text('[verifier]\ntimeout_sec = 120\n')
        issues = check_task(task)
        assert "task.toml missing [agent] section" in issues

    def test_task_toml_missing_timeout_sec(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text('[agent]\n')
        issues = check_task(task)
        assert "task.toml [agent] missing timeout_sec" in issues

    def test_task_toml_parse_error(self, tmp_path):
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text("not valid {{{{ toml")
        issues = check_task(task)
        assert any(i.startswith("task.toml parse error:") for i in issues)

    def test_multiple_issues_reported(self, tmp_path):
        """An empty directory should report multiple specific issues."""
        task = tmp_path / "empty-task"
        task.mkdir()
        issues = check_task(task)
        assert "Missing required file: task.toml" in issues
        assert "Missing required file: instruction.md" in issues
        assert "Missing required directory: environment/" in issues


class TestInitTask:
    """init_task(name, ...) -> Path"""

    def test_creates_complete_structure(self, tmp_path):
        task = init_task("my-task", parent_dir=tmp_path)
        assert task == tmp_path / "my-task"
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "tests" / "test.sh").exists()
        assert (task / "tests" / "test_outputs.py").exists()
        assert (task / "solution" / "solve.sh").exists()

    def test_task_toml_is_valid(self, tmp_path):
        """Scaffolded task.toml should pass check_task validation."""
        task = init_task("valid-task", parent_dir=tmp_path)
        issues = check_task(task)
        assert issues == []

    def test_no_pytest_skips_test_outputs(self, tmp_path):
        task = init_task("no-pytest", parent_dir=tmp_path, no_pytest=True)
        assert not (task / "tests" / "test_outputs.py").exists()
        assert (task / "tests" / "test.sh").exists()

    def test_no_solution_skips_solution_dir(self, tmp_path):
        task = init_task("no-sol", parent_dir=tmp_path, no_solution=True)
        assert not (task / "solution").exists()

    def test_test_sh_is_executable(self, tmp_path):
        task = init_task("exec-test", parent_dir=tmp_path)
        import os
        assert os.access(task / "tests" / "test.sh", os.X_OK)

    def test_solve_sh_is_executable(self, tmp_path):
        task = init_task("exec-sol", parent_dir=tmp_path)
        import os
        assert os.access(task / "solution" / "solve.sh", os.X_OK)

    def test_raises_on_existing_dir(self, tmp_path):
        (tmp_path / "dupe").mkdir()
        with pytest.raises(FileExistsError):
            init_task("dupe", parent_dir=tmp_path)

    def test_creates_parent_dirs(self, tmp_path):
        task = init_task("deep", parent_dir=tmp_path / "nested" / "tasks")
        assert task.exists()
