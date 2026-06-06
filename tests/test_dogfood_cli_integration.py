"""CLI integration tests for real task-standard dogfood packages."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app

DOGFOOD_ROOT = Path("docs/examples/task-standard/benchflow-wanted-features")
RUNTIME_GATE_TASK = DOGFOOD_ROOT / "runtime-capability-gate"
SUPPORTED_TASKS = sorted(
    path
    for path in DOGFOOD_ROOT.iterdir()
    if path.is_dir()
    and (path / "task.md").is_file()
    and path.name != RUNTIME_GATE_TASK.name
)


def _dogfood_task_dirs() -> list[Path]:
    return sorted(
        path
        for path in DOGFOOD_ROOT.iterdir()
        if path.is_dir() and (path / "task.md").is_file()
    )


class TestDogfoodTasksCheckCli:
    def test_all_dogfood_tasks_pass_structural_check(self) -> None:
        for task_dir in _dogfood_task_dirs():
            result = CliRunner().invoke(app, ["tasks", "check", str(task_dir)])
            assert result.exit_code == 0, f"{task_dir.name}: {result.output}"

    def test_supported_tasks_pass_docker_sandbox_check(self) -> None:
        for task_dir in SUPPORTED_TASKS:
            result = CliRunner().invoke(
                app,
                ["tasks", "check", str(task_dir), "--sandbox", "docker"],
            )
            assert result.exit_code == 0, f"{task_dir.name}: {result.output}"
            assert "runtime supported on docker" in result.output

    def test_supported_tasks_pass_modal_sandbox_check(self) -> None:
        for task_dir in SUPPORTED_TASKS:
            result = CliRunner().invoke(
                app,
                ["tasks", "check", str(task_dir), "--sandbox", "modal"],
            )
            assert result.exit_code == 0, f"{task_dir.name}: {result.output}"
            assert "runtime supported on modal" in result.output

    def test_runtime_capability_gate_fails_sandbox_check(self) -> None:
        for sandbox in ("docker", "modal"):
            result = CliRunner().invoke(
                app,
                ["tasks", "check", str(RUNTIME_GATE_TASK), "--sandbox", sandbox],
            )
            assert result.exit_code == 1, result.output
            assert "environment.healthcheck" in result.output

    def test_compat_export_round_trip_cli(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "compat-round-trip"
        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "round-trip",
                str(DOGFOOD_ROOT / "compat-export-loss-reports"),
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Round-trip parity OK" in result.output
        assert (output_dir / "compatibility" / "export-report.json").is_file()
