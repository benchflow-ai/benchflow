"""CLI tests for bench tasks export and sandbox-aware bench tasks check."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app

COMPAT_EXPORT_EXAMPLE = Path(
    "docs/examples/task-standard/benchflow-wanted-features/compat-export-loss-reports"
)


def _write_minimal_legacy_task(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n'
    )
    (task_dir / "instruction.md").write_text("Do the thing.\n")
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


class TestTasksExportCli:
    def test_export_writes_split_layout_and_prints_loss_summary(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "exported"
        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "export",
                str(COMPAT_EXPORT_EXAMPLE),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (output_dir / "task.toml").is_file()
        assert (output_dir / "instruction.md").is_file()
        assert (output_dir / "tests" / "test.sh").is_file()
        assert (output_dir / "solution" / "solve.md").is_file()
        assert "Export harbor (degraded)" in result.output
        assert "Losses (" in result.output
        assert "agents:" in result.output
        assert "verifier.verifier_md:" in result.output

    def test_export_defaults_output_dir_to_cwd_task_name_export(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        task_dir = COMPAT_EXPORT_EXAMPLE.resolve()
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            app,
            ["tasks", "export", str(task_dir)],
        )

        default_output = tmp_path / f"{task_dir.name}-export"
        assert result.exit_code == 0, result.output
        assert default_output.is_dir()
        assert (default_output / "task.toml").is_file()
        assert "Export harbor (degraded)" in result.output

    def test_export_accepts_pier_target(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "pier-export"
        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "export",
                str(COMPAT_EXPORT_EXAMPLE),
                "--target",
                "pier",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Export pier (degraded)" in result.output
        assert (output_dir / "task.toml").is_file()

    def test_export_fails_for_legacy_only_package(self, tmp_path: Path) -> None:
        task = _write_minimal_legacy_task(tmp_path / "legacy-only")
        result = CliRunner().invoke(
            app,
            ["tasks", "export", str(task), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code == 1, result.output
        assert "task.md" in result.output


class TestTasksCheckSandboxCli:
    def test_check_without_sandbox_skips_runtime_validation(self, tmp_path: Path) -> None:
        task = _write_minimal_legacy_task(tmp_path / "ok-task")
        result = CliRunner().invoke(app, ["tasks", "check", str(task)])

        assert result.exit_code == 0, result.output
        assert "valid" in result.output
        assert "runtime supported" not in result.output

    def test_check_with_sandbox_reports_unsupported_features(
        self, tmp_path: Path
    ) -> None:
        task = _write_minimal_legacy_task(tmp_path / "steps-task")
        (task / "task.toml").write_text(
            """\
version = "1.0"

[[steps]]
name = "scaffold"
"""
        )
        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--sandbox", "docker"],
        )

        assert result.exit_code == 1, result.output
        assert "valid" in result.output
        assert "unsupported feature(s) on docker" in result.output
        assert "steps:" in result.output

    def test_check_with_sandbox_passes_minimal_task(self, tmp_path: Path) -> None:
        task = _write_minimal_legacy_task(tmp_path / "ok-task")
        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--sandbox", "daytona"],
        )

        assert result.exit_code == 0, result.output
        assert "valid" in result.output
        assert "runtime supported on daytona" in result.output

    def test_check_with_modal_sandbox_passes_minimal_task(self, tmp_path: Path) -> None:
        task = _write_minimal_legacy_task(tmp_path / "ok-task")
        result = CliRunner().invoke(
            app,
            ["tasks", "check", str(task), "--sandbox", "modal"],
        )

        assert result.exit_code == 0, result.output
        assert "valid" in result.output
        assert "runtime supported on modal" in result.output


class TestTasksRoundTripCli:
    def test_round_trip_exports_and_validates_compat_example(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "round-trip"
        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "round-trip",
                str(COMPAT_EXPORT_EXAMPLE),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Round-trip parity OK" in result.output
        assert (output_dir / "compatibility" / "export-report.json").is_file()
