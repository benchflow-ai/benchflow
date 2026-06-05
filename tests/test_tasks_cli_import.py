"""CLI tests for bench tasks import and export→import round-trip validation."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app

COMPAT_EXPORT_EXAMPLE = Path(
    "docs/examples/task-standard/benchflow-wanted-features/compat-export-loss-reports"
)


def _export_compat_example(output_dir: Path) -> None:
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


class TestTasksImportCli:
    def test_import_split_package_from_exported_compat_example(
        self, tmp_path: Path
    ) -> None:
        exported_dir = tmp_path / "exported"
        _export_compat_example(exported_dir)

        result = CliRunner().invoke(
            app,
            ["tasks", "import", str(exported_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "Import" in result.output
        assert "split package" in result.output
        assert "oracle file(s)" in result.output
        assert "verifier file(s)" in result.output

    def test_import_native_round_trip_validation_passes(self, tmp_path: Path) -> None:
        exported_dir = tmp_path / "exported"
        _export_compat_example(exported_dir)

        result = CliRunner().invoke(
            app,
            [
                "tasks",
                "import",
                str(exported_dir),
                "--native",
                str(COMPAT_EXPORT_EXAMPLE),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Import" in result.output
        assert "split package" in result.output
        assert "Round-trip parity OK" in result.output
