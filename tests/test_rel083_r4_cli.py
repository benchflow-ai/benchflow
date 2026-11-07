"""The CLI findings of the REL r4 review.

L4: ``bench agents oracle`` and ``bench agents nop`` show the two controls
``bench agents`` lists. L6: ``bench run --help-all`` does not end by telling
the user to run ``bench run --help-all``. M4: the unknown ``--target`` of
``bench export`` and the unknown ``--provider`` of ``bench hub list`` end
with a next step, as text and in ``--json``. M7: a BenchFlow without bfx
says so in ``bench doctor`` and at the start of a Docker run.
"""

from __future__ import annotations

import io
import json
import re

import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.mark.parametrize(
    ("name", "says"),
    [("oracle", "reference solution"), ("nop", "Does nothing")],
)
def test_bench_agents_shows_each_control_it_lists(name, says):
    listed = runner.invoke(app, ["agents"])
    assert listed.exit_code == 0
    assert name in _ANSI.sub("", listed.output)
    shown = runner.invoke(app, ["agents", name])
    assert shown.exit_code == 0, shown.output
    out = _ANSI.sub("", shown.output)
    assert out.splitlines()[0].strip() == name
    assert says in out
    assert "Unknown agent" not in out


def test_help_all_does_not_point_at_itself():
    short = _ANSI.sub("", runner.invoke(app, ["run", "--help"]).output)
    full = _ANSI.sub("", runner.invoke(app, ["run", "--help-all"]).output)
    assert "more run settings" in short
    assert "more run settings" not in full
    # Every setting is on the long page.
    assert "--max-sandbox-seconds" in full


def test_an_unknown_export_target_ends_with_a_next_step(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    result = runner.invoke(
        app, ["export", str(task), str(tmp_path / "o"), "--target", "foo"]
    )
    assert result.exit_code == 2
    lines = [line.strip() for line in result.stderr.strip().splitlines()]
    assert lines[0].startswith("--target 'foo' is an unknown value")
    assert lines[-1] == "next: pass one of those to --target"


def test_an_unknown_hub_provider_has_a_next_step_in_json(capsys):
    from benchflow.cli._errors import run_cli

    with pytest.raises(SystemExit) as stopped:
        run_cli(app, ["hub", "list", "--provider", "nosuch", "--json"])
    assert stopped.value.code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["error"].startswith("--provider 'nosuch' is an unknown value")
    assert obj["next"] == "pass one of those to --provider"


@pytest.fixture
def no_bfx(tmp_path, monkeypatch):
    """A BenchFlow built from source without `make bfx`: no binaries."""
    from benchflow._bfx import locate

    monkeypatch.setattr(locate, "BIN_DIR", tmp_path / "empty-bin")
    return tmp_path


def test_doctor_says_when_bfx_is_not_installed(no_bfx):
    from benchflow import doctor

    report = doctor.run_doctor(sandbox="docker", offline=True)
    (check,) = [c for c in report.checks if c.id == "bfx"]
    assert check.status == "warn"
    assert check.summary.startswith("bfx is not installed")
    assert "BENCHFLOW_BFX_DELIVERY=off" in check.summary
    assert "make bfx" in check.fix


class _DockerReady:
    def checks(self):
        return []


def _preflight_output(monkeypatch, sandbox: str) -> str:
    import benchflow.cli.doctor as cli_doctor
    from benchflow import doctor

    out = io.StringIO()
    monkeypatch.setattr(cli_doctor, "err_console", Console(file=out, width=400))
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(doctor.DoctorProbes, "from_host", classmethod(lambda c: None))
    cli_doctor.eval_preflight(sandbox=sandbox, docker=_DockerReady())
    return out.getvalue()


def test_a_docker_run_says_at_its_start_when_bfx_is_not_installed(no_bfx, monkeypatch):
    monkeypatch.delenv("BENCHFLOW_BFX_DELIVERY", raising=False)
    text = _preflight_output(monkeypatch, "docker")
    assert "Warning: bfx is not installed" in text
    assert "make bfx" in text


def test_no_bfx_notice_when_it_is_off_by_choice(no_bfx, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")
    assert "bfx" not in _preflight_output(monkeypatch, "docker")
