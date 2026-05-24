"""Tests for benchflow.tasks — task authoring (check and init)."""

import shutil
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import _check_ctrf_path, check_task, init_task


class TestCheckTask:
    """check_task(task_dir) -> list[str]"""

    def _make_valid_task(self, tmp_path: Path) -> Path:
        """Create a minimal valid task directory."""
        task = tmp_path / "my-task"
        task.mkdir()
        (task / "task.toml").write_text(
            "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n"
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
        assert (
            "Missing tests/ directory (verifier needs test.sh or evaluate.py)" in issues
        )

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

    def test_task_toml_missing_agent_section_is_allowed(self, tmp_path):
        """[agent] is optional — runtime AgentConfig defaults to no timeout.

        Guards #379: bench tasks check and bench eval create must agree on
        the missing-[agent] contract. The runtime accepts it, so check_task
        does too.
        """
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text("[verifier]\ntimeout_sec = 120\n")
        issues = check_task(task)
        assert not any("agent" in i for i in issues)

    def test_task_toml_missing_timeout_sec_is_allowed(self, tmp_path):
        """[agent].timeout_sec is optional — defaults to no wall-clock cap.

        Guards #379: keeps check_task in sync with AgentConfig (timeout_sec
        defaults to None).
        """
        task = self._make_valid_task(tmp_path)
        (task / "task.toml").write_text("[agent]\n")
        issues = check_task(task)
        assert not any("timeout_sec" in i for i in issues)

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

    def test_scaffold_fails_check_until_placeholders_replaced(self, tmp_path):
        """Freshly scaffolded task must FAIL `bench tasks check` (#360).

        Otherwise authors can `bench tasks init foo && bench tasks check foo`
        and get a green ✓ on a task that auto-passes with reward 1.0 —
        silent false positives in eval sets.
        """
        task = init_task("scaffolded", parent_dir=tmp_path)
        issues = check_task(task)
        assert issues, "scaffold must not pass check_task before editing"
        # instruction.md, tests/test.sh and solution/solve.sh all carry the
        # placeholder marker — each must be flagged so the author sees what
        # to replace.
        joined = "\n".join(issues)
        assert "instruction.md" in joined
        assert "tests/test.sh" in joined
        assert "solution/solve.sh" in joined

    def test_scaffold_passes_check_after_placeholders_replaced(self, tmp_path):
        """Once the author replaces every [REPLACE: ...] marker, check passes."""
        task = init_task("editable", parent_dir=tmp_path)
        (task / "instruction.md").write_text("# editable\n\nDo the thing.\n")
        (task / "tests" / "test.sh").write_text(
            '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n'
        )
        (task / "solution" / "solve.sh").write_text(
            "#!/bin/bash\ntouch /app/output.txt\n"
        )
        assert check_task(task) == []

    def test_scaffold_test_sh_defaults_to_failure(self, tmp_path):
        """Scaffolded verifier writes 0.0, not 1.0 (#360).

        Fail-closed default means even if check_task were skipped, the task
        cannot give a passing reward without explicit author edits.
        """
        task = init_task("fail-closed", parent_dir=tmp_path)
        test_sh = (task / "tests" / "test.sh").read_text()
        assert 'echo "0.0"' in test_sh
        assert 'echo "1.0"' not in test_sh

    def test_scaffold_solution_does_not_satisfy_verifier(self, tmp_path):
        """The placeholder solution must NOT make the placeholder verifier
        pass — issue #360 calls this out explicitly: solution and test must
        not collide on a default-pass.

        We simulate by reading both files and checking that solve.sh exits
        nonzero (it does not produce side effects) while test.sh writes 0.0.
        """
        task = init_task("collision", parent_dir=tmp_path)
        solve = (task / "solution" / "solve.sh").read_text()
        test_sh = (task / "tests" / "test.sh").read_text()
        # Solve.sh must not write a passing reward itself (no echo of 1.0
        # into the reward file).
        assert "reward.txt" not in solve
        assert 'echo "1.0"' not in solve
        # Test.sh must default to 0.0 reward.
        assert 'echo "0.0"' in test_sh
        # And the solve script signals it's a placeholder by exiting nonzero.
        assert "exit 1" in solve

    def test_scaffold_pytest_template_fails(self, tmp_path):
        """The pytest verifier template fails until edited (#360)."""
        task = init_task("py-fail", parent_dir=tmp_path)
        text = (task / "tests" / "test_outputs.py").read_text()
        assert "pytest.fail" in text
        assert "assert True" not in text

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


class TestCtrfPathLint:
    """_check_ctrf_path detects non-standard CTRF output paths (ENG-153).

    Guards PR #356 — ensures `bench tasks check` catches inconsistent
    CTRF paths before they reach production evals.
    """

    def test_standard_path_no_issues(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n")
        assert _check_ctrf_path(sh) == []

    def test_standard_path_equals_form(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf=/logs/verifier/ctrf.json /tests/test_outputs.py\n")
        assert _check_ctrf_path(sh) == []

    def test_nonstandard_filename_flagged(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf /logs/verifier/ctrf-report.json /tests\n")
        issues = _check_ctrf_path(sh)
        assert len(issues) == 1
        assert "ctrf-report.json" in issues[0]
        assert "non-standard" in issues[0]

    def test_relative_path_flagged(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest --ctrf ctrf.json /tests/test_outputs.py\n")
        issues = _check_ctrf_path(sh)
        assert len(issues) == 1
        assert "ctrf.json" in issues[0]

    def test_variable_path_skipped(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text('pytest --ctrf "$CTRF_DIR/ctrf.json" /tests\n')
        assert _check_ctrf_path(sh) == []

    def test_commented_ctrf_ignored(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("# pytest --ctrf /bad/path.json\npytest /tests\n")
        assert _check_ctrf_path(sh) == []

    def test_no_ctrf_flag_no_issues(self, tmp_path):
        sh = tmp_path / "test.sh"
        sh.write_text("pytest /tests/test_outputs.py\n")
        assert _check_ctrf_path(sh) == []

    def test_check_task_integrates_ctrf_lint(self, tmp_path):
        """check_task() calls _check_ctrf_path as part of validation."""
        task = tmp_path / "bad-ctrf"
        task.mkdir()
        (task / "task.toml").write_text(
            "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n"
        )
        (task / "instruction.md").write_text("# Task\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        tests = task / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text(
            "pytest --ctrf /logs/verifier/ctrf-report.json /tests\n"
        )
        issues = check_task(task)
        assert any("non-standard CTRF path" in i for i in issues)
