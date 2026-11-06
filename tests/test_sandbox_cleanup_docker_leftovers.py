"""`bench sandbox cleanup` sees the Docker leftovers of a killed run.

Guards DF-p3-eval-runner-9 (PY8 DX-G, base py8/dx-a 1ac5fffa): after a
``kill -9`` the dead run's containers kept running, and ``bench sandbox
cleanup --dry-run`` reported only ``Docker bf-snap images: 0 found``. It now
runs the jobs' own leftover sweep (tests/test_docker_sweep.py has its rules)
and says what it removes, or would remove.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from benchflow.cli import sandbox as sandbox_cli
from benchflow.cli.main import app
from benchflow.sandbox import _docker_sweep as sweep
from benchflow.sandbox._docker_sweep import SweepResult


@pytest.fixture
def host(monkeypatch):
    calls: list[bool] = []

    def fake_sweep(*, dry_run=False):
        calls.append(dry_run)
        return SweepResult(
            containers=["c-stopped", "c-up-1", "c-up-2"],
            networks=["n-1", "n-2"],
            running=["c-up-1", "c-up-2"],
            projects=["cancel-async-tasks__7e483de0", "log-summary__bb16ce4c"],
            # A real run: what Docker confirmed (the integ accounting).
            removed_containers=[] if dry_run else ["c-stopped", "c-up-1", "c-up-2"],
            removed_networks=[] if dry_run else ["n-1", "n-2"],
        )

    monkeypatch.setattr(sandbox_cli, "_daytona_sdk_available", lambda: False)
    monkeypatch.setattr(sandbox_cli, "_cleanup_agentcore_runtimes", lambda **_: False)
    monkeypatch.setattr(sandbox_cli, "cleanup_docker_snapshots", lambda **_: None)
    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(sweep, "sweep_leftovers", fake_sweep)
    return calls


@pytest.mark.parametrize(
    ("flags", "dry_run", "verb"),
    [(["--dry-run"], True, "would remove"), ([], False, "removed")],
)
def test_cleanup_reports_a_killed_runs_containers(host, flags, dry_run, verb):
    result = CliRunner().invoke(app, ["sandbox", "cleanup", *flags], terminal_width=200)
    assert result.exit_code == 0, result.output
    assert host == [dry_run]
    out = " ".join(result.output.split())
    assert (
        "Docker leftovers of exited BenchFlow processes: 3 container(s) "
        f"(2 running), 2 network(s) {verb}: cancel-async-tasks__7e483de0, "
        "log-summary__bb16ce4c" in out
    )
    assert "Nothing to clean up" not in out
    assert "Traceback" not in result.output


def test_no_docker_cli_runs_no_sweep(monkeypatch):
    called = []
    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(sweep, "sweep_leftovers", lambda **_: called.append(1))
    assert sandbox_cli.cleanup_docker_leftovers(dry_run=True) is False
    assert called == []
