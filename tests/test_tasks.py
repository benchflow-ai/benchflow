"""Tests for benchflow.tasks — task authoring (check and init)."""

import shutil
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import (
    _check_ctrf_path,
    check_task,
    init_task,
    migrate_task_to_task_md,
)
from benchflow.cli.main import app
from benchflow.task import Task, TaskConfig, TaskDocument
from benchflow.task.paths import TaskPaths


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
            "Missing verifier/ directory (or legacy tests/; verifier needs "
            "test.sh or evaluate.py)" in issues
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


class TestTaskPathAliases:
    def test_native_dirs_are_preferred_over_legacy_aliases(self, tmp_path):
        """Guards oracle/verifier as native names with legacy aliases intact."""
        task = tmp_path / "task"
        (task / "oracle").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "verifier").mkdir()
        (task / "tests").mkdir()

        paths = TaskPaths(task)

        assert paths.solution_dir == task / "oracle"
        assert paths.tests_dir == task / "verifier"
        assert paths.uses_native_oracle_dir is True
        assert paths.uses_native_verifier_dir is True

    def test_legacy_dirs_remain_supported(self, tmp_path):
        """Guards compatibility for existing tests/ and solution/ packages."""
        task = tmp_path / "task"
        (task / "solution").mkdir(parents=True)
        (task / "tests").mkdir()

        paths = TaskPaths(task)

        assert paths.solution_dir == task / "solution"
        assert paths.tests_dir == task / "tests"
        assert paths.uses_native_oracle_dir is False
        assert paths.uses_native_verifier_dir is False

    def test_check_task_rejects_oracle_solution_alias_collision(self, tmp_path):
        """Guards task-standard alias drift when both native and legacy dirs exist."""
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n[verifier]\n')
        (task / "instruction.md").write_text("Do something\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "oracle").mkdir()
        (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho native\n")
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho legacy\n")
        (task / "verifier").mkdir()
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\nexit 0\n")

        issues = check_task(task)

        assert any("oracle/" in issue and "solution/" in issue for issue in issues)

    def test_check_task_rejects_verifier_tests_alias_collision(self, tmp_path):
        """Guards task-standard alias drift when both native and legacy dirs exist."""
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n[verifier]\n')
        (task / "instruction.md").write_text("Do something\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "verifier").mkdir()
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\necho native\n")
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/bash\necho legacy\n")

        issues = check_task(task)

        assert any("verifier/" in issue and "tests/" in issue for issue in issues)

    def test_task_load_rejects_oracle_solution_alias_collision(self, tmp_path):
        """Guards runtime fail-closed behavior for conflicting alias trees."""
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n[verifier]\n')
        (task / "instruction.md").write_text("Do something\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "oracle").mkdir()
        (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho native\n")
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho legacy\n")
        (task / "verifier").mkdir()
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\nexit 0\n")

        with pytest.raises(ValueError, match="oracle/"):
            Task(task)

    def test_check_task_accepts_byte_identical_alias_trees(self, tmp_path):
        """Equivalent native and legacy alias trees should still pass check."""
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n[verifier]\n')
        (task / "instruction.md").write_text("Do something\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "oracle").mkdir()
        (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho ok\n")
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho ok\n")
        (task / "verifier").mkdir()
        (task / "verifier" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")

        assert check_task(task) == []


class TestInitTask:
    """init_task(name, ...) -> Path"""

    def test_default_creates_task_md_structure(self, tmp_path):
        """Guards commit 67378ddd's task.md default authoring standard."""
        task = init_task("my-task", parent_dir=tmp_path)

        assert task == tmp_path / "my-task"
        assert (task / "task.md").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "verifier" / "test.sh").exists()
        assert (task / "verifier" / "test_outputs.py").exists()
        assert (task / "oracle" / "solve.sh").exists()
        assert not (task / "tests").exists()
        assert not (task / "solution").exists()

    def test_creates_legacy_structure_when_requested(self, tmp_path):
        task = init_task("my-task", parent_dir=tmp_path, task_format="legacy")
        assert task == tmp_path / "my-task"
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "tests" / "test.sh").exists()
        assert (task / "tests" / "test_outputs.py").exists()
        assert (task / "solution" / "solve.sh").exists()

    def test_tasks_init_cli_defaults_to_task_md(self, tmp_path):
        """Guards the CLI default for the task.md standard."""
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            app,
            ["tasks", "init", "cli-default", "--dir", str(tmp_path)],
        )

        task = tmp_path / "cli-default"
        assert result.exit_code == 0, result.output
        assert (task / "task.md").exists()
        assert (task / "verifier" / "test.sh").exists()
        assert (task / "oracle" / "solve.sh").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()

    def test_creates_task_md_structure(self, tmp_path):
        """Guards commit 67378ddd's 2026-06-04 task.md scaffold entrypoint."""
        task = init_task(
            "my-task-md",
            parent_dir=tmp_path,
            task_format="task-md",
        )

        assert (task / "task.md").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "verifier" / "test.sh").exists()

    def test_check_task_accepts_authored_task_md(self, tmp_path):
        """Guards commit 67378ddd's 2026-06-04 task.md check path."""
        task = init_task(
            "authored-task-md",
            parent_dir=tmp_path,
            no_pytest=True,
            no_oracle=True,
            task_format="task-md",
        )
        (task / "task.md").write_text(
            """---
version: "1.0"
metadata:
  category: capability
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 1
---
## prompt

Create hello.txt.
"""
        )
        (task / "verifier" / "test.sh").write_text(
            '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n'
        )

        assert check_task(task) == []

    def test_scaffold_fails_check_until_placeholders_replaced(self, tmp_path):
        """Freshly scaffolded task must FAIL `bench tasks check` (#360).

        Otherwise authors can `bench tasks init foo && bench tasks check foo`
        and get a green ✓ on a task that auto-passes with reward 1.0 —
        silent false positives in eval sets.
        """
        task = init_task("scaffolded", parent_dir=tmp_path, task_format="legacy")
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
        task = init_task("editable", parent_dir=tmp_path, task_format="legacy")
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
        task = init_task("fail-closed", parent_dir=tmp_path, task_format="legacy")
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
        task = init_task("collision", parent_dir=tmp_path, task_format="legacy")
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
        task = init_task("py-fail", parent_dir=tmp_path, task_format="legacy")
        text = (task / "tests" / "test_outputs.py").read_text()
        assert "pytest.fail" in text
        assert "assert True" not in text

    def test_no_pytest_skips_test_outputs(self, tmp_path):
        task = init_task(
            "no-pytest",
            parent_dir=tmp_path,
            no_pytest=True,
            task_format="legacy",
        )
        assert not (task / "tests" / "test_outputs.py").exists()
        assert (task / "tests" / "test.sh").exists()

    def test_no_oracle_skips_native_oracle_dir(self, tmp_path):
        task = init_task(
            "no-oracle",
            parent_dir=tmp_path,
            no_oracle=True,
        )
        assert not (task / "oracle").exists()

    def test_no_solution_remains_legacy_compat_alias(self, tmp_path):
        task = init_task(
            "no-sol",
            parent_dir=tmp_path,
            no_solution=True,
            task_format="legacy",
        )
        assert not (task / "solution").exists()

    def test_test_sh_is_executable(self, tmp_path):
        task = init_task("exec-test", parent_dir=tmp_path, task_format="legacy")
        import os

        assert os.access(task / "tests" / "test.sh", os.X_OK)

    def test_solve_sh_is_executable(self, tmp_path):
        task = init_task("exec-sol", parent_dir=tmp_path, task_format="legacy")
        import os

        assert os.access(task / "solution" / "solve.sh", os.X_OK)

    def test_raises_on_existing_dir(self, tmp_path):
        (tmp_path / "dupe").mkdir()
        with pytest.raises(FileExistsError):
            init_task("dupe", parent_dir=tmp_path)

    def test_creates_parent_dirs(self, tmp_path):
        task = init_task("deep", parent_dir=tmp_path / "nested" / "tasks")
        assert task.exists()


class TestMigrateTask:
    """migrate_task_to_task_md(task_dir) -> TaskMigrationResult"""

    def _make_legacy_task(self, tmp_path: Path) -> Path:
        task = tmp_path / "legacy-task"
        task.mkdir()
        (task / "task.toml").write_text(
            """schema_version = "1.3"
artifacts = [{ source = "/logs/artifacts", destination = "artifacts" }]

[metadata]
category = "migration"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120

[verifier.hardening]
cleanup_conftests = false

[environment]
network_mode = "no-network"
cpus = 2
memory_mb = 4096

[solution.env]
MODE = "legacy-oracle"
"""
        )
        (task / "instruction.md").write_text("# Legacy prompt\n\nCreate hello.txt.\n")
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
        return task

    def test_migrate_preserves_config_and_prompt_without_deleting_legacy(
        self, tmp_path
    ):
        """Guards commit 67378ddd's 2026-06-04 task.md migration."""
        task = self._make_legacy_task(tmp_path)
        legacy_config = TaskConfig.model_validate_toml(
            (task / "task.toml").read_text()
        )

        result = migrate_task_to_task_md(task)
        document = TaskDocument.from_path(result.task_md)

        assert result.removed_legacy is False
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert document.config.model_dump() == legacy_config.model_dump()
        assert document.instruction == "# Legacy prompt\n\nCreate hello.txt."
        assert document.frontmatter["oracle"]["env"] == {"MODE": "legacy-oracle"}
        assert "solution" not in document.frontmatter
        assert check_task(task) == []

    def test_migrate_can_remove_legacy_pair_after_verified_roundtrip(
        self, tmp_path
    ):
        """Guards explicit cleanup when adopting task.md as the canonical file."""
        task = self._make_legacy_task(tmp_path)

        result = migrate_task_to_task_md(task, remove_legacy=True)

        assert result.removed_legacy is True
        assert (task / "task.md").exists()
        assert not (task / "task.toml").exists()
        assert not (task / "instruction.md").exists()
        assert check_task(task) == []

    def test_migrate_refuses_to_overwrite_existing_task_md(self, tmp_path):
        """Guards commit 67378ddd's migration against task.md replacement."""
        task = self._make_legacy_task(tmp_path)
        (task / "task.md").write_text("---\n---\n\nExisting document.\n")

        with pytest.raises(FileExistsError):
            migrate_task_to_task_md(task)

    def test_tasks_migrate_cli_writes_task_md(self, tmp_path):
        """Guards the CLI adoption path for legacy task authors."""
        from typer.testing import CliRunner

        task = self._make_legacy_task(tmp_path)

        result = CliRunner().invoke(app, ["tasks", "migrate", str(task)])

        assert result.exit_code == 0, result.output
        assert (task / "task.md").exists()
        assert "kept task.toml and instruction.md" in result.output


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
