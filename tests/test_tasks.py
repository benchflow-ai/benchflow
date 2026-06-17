"""Tests for benchflow.tasks — task authoring (check and init)."""

import os
import shutil
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import (
    _check_ctrf_path,
    check_task,
    init_task,
    scaffold_task,
)


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
        """A missing legacy tests/ verifier dir is still reported.

        _make_valid_task uses the legacy split layout whose verifier package
        is the tests/ alias. That alias still ships, so removing it must make
        check_task report the missing verifier package and name the legacy
        tests/ fallback.
        """
        task = self._make_valid_task(tmp_path)
        shutil.rmtree(task / "tests")
        issues = check_task(task)
        assert (
            "Missing verifier/ directory (or legacy tests/; verifier needs "
            "test.sh or a verifier.md selected strategy)" in issues
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

        Guards #379: bench tasks check and bench eval run must agree on
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
    """init_task(name, ...) -> Path — default task.md format."""

    def test_no_solution_skips_solution_dir(self, tmp_path):
        task = init_task("no-sol", parent_dir=tmp_path, no_solution=True)
        assert not (task / "solution").exists()

    def test_raises_on_existing_dir(self, tmp_path):
        (tmp_path / "dupe").mkdir()
        with pytest.raises(FileExistsError):
            init_task("dupe", parent_dir=tmp_path)

    @pytest.mark.parametrize(
        "bad",
        ["../escape", "foo/bar", "my task", "..", ".hidden", " lead", "back\\slash"],
    )
    def test_rejects_unsafe_task_names(self, tmp_path, bad):
        # The name is a single dir segment under parent_dir; separators / '..' /
        # leading dots / whitespace are rejected (path traversal + tooling safety).
        with pytest.raises(ValueError, match="single path segment"):
            init_task(bad, parent_dir=tmp_path)
        # Nothing escaped the parent dir.
        assert not (tmp_path.parent / "escape").exists()

    def test_creates_parent_dirs(self, tmp_path):
        task = init_task("deep", parent_dir=tmp_path / "nested" / "tasks")
        assert task.exists()


class TestScaffoldTaskReportedFiles:
    """scaffold_task(...).files must equal exactly what landed on disk.

    The CLI ``Created:`` summary prints ``result.files`` verbatim, so this is
    the contract that keeps the summary from under-reporting (it used to omit
    verifier/test_outputs.py and verifier/rubrics/verifier.toml, both of which
    ``bench tasks check`` validates).
    """

    @staticmethod
    def _disk_files(task_dir: Path) -> set[str]:
        return {
            p.relative_to(task_dir).as_posix()
            for p in task_dir.rglob("*")
            if p.is_file()
        }

    def test_reported_set_equals_written_set_task_md(self, tmp_path):
        result = scaffold_task("rep", parent_dir=tmp_path)
        assert set(result.files) == self._disk_files(result.task_dir)
        assert result.files == sorted(result.files)
        assert set(result.files) == {
            "task.md",
            "environment/Dockerfile",
            "verifier/test.sh",
            "verifier/test_outputs.py",
            "verifier/verifier.md",
            "verifier/rubrics/verifier.md",
            "verifier/rubrics/verifier.toml",
            "oracle/solve.sh",
        }

    def test_reported_set_equals_written_set_legacy(self, tmp_path):
        result = scaffold_task("rep", parent_dir=tmp_path, task_format="legacy")
        assert set(result.files) == self._disk_files(result.task_dir)
        assert set(result.files) == {
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "tests/test.sh",
            "tests/test_outputs.py",
            "solution/solve.sh",
        }

    def test_reported_set_tracks_no_pytest_and_no_oracle_flags(self, tmp_path):
        result = scaffold_task(
            "rep", parent_dir=tmp_path, no_pytest=True, no_oracle=True
        )
        assert set(result.files) == self._disk_files(result.task_dir)
        assert "verifier/test_outputs.py" not in result.files
        assert "oracle/solve.sh" not in result.files
        # the rubric package (validated by `bench tasks check`) is still written
        assert "verifier/rubrics/verifier.toml" in result.files

    def test_cli_init_summary_lists_every_written_file(self, tmp_path):
        import click
        from typer.testing import CliRunner

        from benchflow.cli.main import app

        out = CliRunner().invoke(app, ["tasks", "init", "rep", "--dir", str(tmp_path)])
        assert out.exit_code == 0, out.output
        text = click.unstyle(out.output)
        reported = {line.strip() for line in text.splitlines() if line.startswith("  ")}
        assert reported == self._disk_files(tmp_path / "rep")
        # the two files the old hardcoded summary dropped (#regression)
        assert "verifier/test_outputs.py" in reported
        assert "verifier/rubrics/verifier.toml" in reported

    def test_cli_init_rejects_legacy_scaffold(self, tmp_path):
        from typer.testing import CliRunner

        from benchflow.cli.main import app

        out = CliRunner().invoke(
            app, ["tasks", "init", "old", "--dir", str(tmp_path), "--format", "legacy"]
        )
        assert out.exit_code == 1
        assert "no longer scaffolds the legacy split layout" in out.output
        assert not (tmp_path / "old").exists()


class TestLegacyScaffoldCompatibility:
    """init_task(..., task_format="legacy") — compatibility scaffold coverage.

    The CLI no longer creates new split-layout tasks in v0.6.2, but migration,
    export, and adapter tests still depend on the compatibility scaffolder.
    Keep these fail-closed defaults covered while the lower-level API remains.
    """

    def test_creates_complete_structure(self, tmp_path):
        task = init_task("my-task", parent_dir=tmp_path, task_format="legacy")
        assert task == tmp_path / "my-task"
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert (task / "environment" / "Dockerfile").exists()
        assert (task / "tests" / "test.sh").exists()
        assert (task / "tests" / "test_outputs.py").exists()
        assert (task / "solution" / "solve.sh").exists()

    def test_scaffold_fails_check_until_placeholders_replaced(self, tmp_path):
        """Freshly scaffolded legacy task must FAIL `bench tasks check` (#360).

        Otherwise compatibility-generated fixtures can pass `bench tasks check`
        before the placeholders are replaced, creating silent false positives in
        eval sets.
        """
        task = init_task("scaffolded", parent_dir=tmp_path, task_format="legacy")
        issues = check_task(task)
        assert issues, "scaffold must not pass check_task before editing"
        # instruction.md, tests/test.sh, tests/test_outputs.py and
        # solution/solve.sh all carry the placeholder marker — each must be
        # flagged so the author sees what to replace.
        joined = "\n".join(issues)
        assert "instruction.md" in joined
        assert "tests/test.sh" in joined
        assert "tests/test_outputs.py" in joined
        assert "solution/solve.sh" in joined

    def test_scaffold_passes_check_after_placeholders_replaced(self, tmp_path):
        """Once the author replaces every [REPLACE: ...] marker, check passes."""
        task = init_task("editable", parent_dir=tmp_path, task_format="legacy")
        (task / "instruction.md").write_text("# editable\n\nDo the thing.\n")
        (task / "tests" / "test.sh").write_text(
            '#!/bin/bash\necho "1.0" > /logs/verifier/reward.txt\n'
        )
        (task / "tests" / "test_outputs.py").write_text(
            "def test_outputs():\n    assert True\n"
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
            "no-pytest", parent_dir=tmp_path, no_pytest=True, task_format="legacy"
        )
        assert not (task / "tests" / "test_outputs.py").exists()
        assert (task / "tests" / "test.sh").exists()

    def test_test_sh_is_executable(self, tmp_path):
        task = init_task("exec-test", parent_dir=tmp_path, task_format="legacy")
        assert os.access(task / "tests" / "test.sh", os.X_OK)

    def test_solve_sh_is_executable(self, tmp_path):
        task = init_task("exec-sol", parent_dir=tmp_path, task_format="legacy")
        assert os.access(task / "solution" / "solve.sh", os.X_OK)


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


class TestTaskMdScaffoldAgentNeutral:
    """The default task.md scaffold must not pin an agent (#dogfood-1).

    A scenes block with a role-pinned agent silently overrides --agent on
    `bench eval run`, so the scaffold's own documented oracle smoke test
    (`--agent oracle`) ran claude-agent-acp instead and died on ACP auth.
    The scaffold stays bare prose; roles/scenes are opt-in via the authoring
    guide.
    """

    def test_scaffold_pins_no_agent_and_declares_no_scenes(self, tmp_path):
        from benchflow.task.document import TaskDocument

        task = init_task("agent-neutral", parent_dir=tmp_path)
        text = (task / "task.md").read_text()
        assert "claude-agent-acp" not in text
        frontmatter = text.split("---")[1]
        assert "scenes:" not in frontmatter
        assert "agents:" not in frontmatter
        doc = TaskDocument.from_path(task / "task.md")
        assert not doc.roles
        assert not doc.scenes


class TestTaskMdScaffoldCanonicalVersionKey:
    """The task.md scaffold must use the standard's canonical version key.

    init used to scaffold 'version:' while migrate emitted 'schema_version:',
    so freshly authored and migrated documents disagreed on spelling. The
    parser keeps accepting 'version' as an alias; generated documents pin
    the canonical 'schema_version'.
    """

    def test_scaffold_emits_schema_version_not_alias(self, tmp_path):
        import yaml

        from benchflow.task.document import TaskDocument

        task = init_task("canonical-key", parent_dir=tmp_path)
        frontmatter = yaml.safe_load((task / "task.md").read_text().split("---\n")[1])
        assert frontmatter["schema_version"] == "1.3"
        assert "version" not in frontmatter
        doc = TaskDocument.from_path(task / "task.md")
        assert doc.config.schema_version == "1.3"
