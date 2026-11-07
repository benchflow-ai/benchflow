"""Guard the AppleDouble warning against the 61b4d227 baseline."""

from typer.testing import CliRunner

from benchflow.cli.main import app
from tests.test_check_run_plan import _task


def test_check_warns_on_nested_appledouble_without_failing(tmp_path):
    task = _task(tmp_path / "task")
    sidecar = task / "environment" / "._data.csv"
    sidecar.write_bytes(b"metadata")
    result = CliRunner().invoke(app, ["check", str(task)], color=False)
    assert result.exit_code == 0, result.output
    assert "AppleDouble" in result.output
    assert "environment/._data.csv" in result.output
    assert sidecar.read_bytes() == b"metadata"


def test_check_does_not_warn_about_external_sidecars(tmp_path):
    task = _task(tmp_path / "task")
    (tmp_path / "._external").write_bytes(b"metadata")
    result = CliRunner().invoke(app, ["check", str(task)], color=False)
    assert result.exit_code == 0, result.output
    assert "AppleDouble" not in result.output
