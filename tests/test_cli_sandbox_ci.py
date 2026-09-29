"""bench sandbox list/cleanup for CI.

After a cancelled run, `cleanup --max-age 0` could not reclaim
the owner's STARTED sandbox (the 30-minute activity guard; no override).
With bad credentials cleanup printed a false 'not installed' line and
exited 0. `list` showed every sandbox on the account with no owner
column or filter and truncated ids. Every skip was reported as
'younger than Nm', including sandboxes skipped because they are not yours.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.cli import main as cli_main
from benchflow.cli import sandbox as sandbox_cli
from benchflow.cli.main import app
from benchflow.sandbox.daytona import reap_stale_sandboxes

ME = "ci-owner"


def _sb(
    sb_id: str, age: float, *, owner: str | None = ME, active_ago: float | None = None
):
    now = datetime.now(UTC)
    labels = (
        {"benchflow.managed": f"1:{owner}", "benchflow.owner": owner} if owner else {}
    )
    return SimpleNamespace(
        id=sb_id,
        state="SandboxState.STARTED",
        created_at=(now - timedelta(minutes=age)).isoformat().replace("+00:00", "Z"),
        last_activity_at=None
        if active_ago is None
        else (now - timedelta(minutes=active_ago)).isoformat().replace("+00:00", "Z"),
        labels=labels,
        target="us",
    )


class FakeClient:
    def __init__(self, sandboxes):
        self.sandboxes = sandboxes
        self.deleted: list[str] = []

    def list(self):
        return iter(self.sandboxes)

    def delete(self, sb):
        self.deleted.append(sb.id)

    @property
    def snapshot(self):
        return SimpleNamespace(
            list=lambda **_: SimpleNamespace(items=[], total_pages=1)
        )


@pytest.fixture(autouse=True)
def _no_host_docker_sweep(monkeypatch):
    """`bench sandbox cleanup` also removes the host's `bf-snap-*` images. These tests
    mock only Daytona, so without this stub they ran a real `docker rmi` over every kept
    checkpoint on the machine running the suite. The Docker sweep has its own tests."""
    monkeypatch.setattr(sandbox_cli, "cleanup_docker_snapshots", lambda **_: None)


@pytest.fixture
def fleet(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", ME)
    client = FakeClient(
        [
            _sb("mine-running-leak", 12, active_ago=1),
            _sb("mine-old-idle", 3000, active_ago=2000),
            _sb("mine-young", 5),
            _sb("theirs", 5000, owner="someone-else"),
        ]
    )
    monkeypatch.setattr(sandbox_cli, "_daytona_sdk_available", lambda: True)
    monkeypatch.setattr(cli_main, "_daytona_client_or_exit", lambda: client)
    return client


def test_reaper_ignore_age_takes_every_owned_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", ME)
    client = FakeClient(
        [_sb("a", 1, active_ago=0), _sb("b", 1), _sb("x", 1, owner="other")]
    )
    reasons: dict[str, str] = {}
    reap_stale_sandboxes(
        client, ignore_age=True, on_skip=lambda sb, why: reasons.update({sb.id: why})
    )
    assert sorted(client.deleted) == ["a", "b"]
    assert reasons == {"x": "foreign"}


def test_cleanup_all_reclaims_my_leaked_sandbox(fleet) -> None:
    result = CliRunner().invoke(
        app, ["sandbox", "cleanup", "--all"], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    assert sorted(fleet.deleted) == ["mine-old-idle", "mine-running-leak", "mine-young"]
    assert "theirs" not in fleet.deleted


def test_cleanup_all_needs_an_owner(fleet, monkeypatch) -> None:
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER")
    result = CliRunner().invoke(
        app, ["sandbox", "cleanup", "--all"], terminal_width=200
    )
    assert result.exit_code == 2 and "BENCHFLOW_DAYTONA_OWNER" in result.output
    assert fleet.deleted == []


def test_cleanup_gives_the_real_skip_reasons(fleet) -> None:
    result = CliRunner().invoke(
        app, ["sandbox", "cleanup", "--max-age", "0"], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    assert fleet.deleted == ["mine-old-idle", "mine-young"]
    out = " ".join(result.output.split())
    assert "1 another owner's (ignored)" in out
    assert "1 active in the last" in out


def test_cleanup_fails_loudly_when_listing_fails(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_cli, "_daytona_sdk_available", lambda: True)

    def boom(**_):
        raise RuntimeError("Failed to list sandboxes: Invalid credentials")

    monkeypatch.setattr(cli_main, "_cleanup_daytona_sandboxes", boom)
    result = CliRunner().invoke(
        app, ["sandbox", "cleanup", "--dry-run"], terminal_width=200
    )
    assert result.exit_code == 1
    assert "Invalid credentials" in result.output
    assert "is installed" not in result.output  # no false 'not installed' line


def test_list_is_filtered_to_my_owner_with_full_ids_and_json(fleet) -> None:
    mine = CliRunner().invoke(app, ["sandbox", "list", "--json"], terminal_width=200)
    assert mine.exit_code == 0, mine.output
    rows = json.loads(mine.stdout)
    assert sorted(r["id"] for r in rows) == [
        "mine-old-idle",
        "mine-running-leak",
        "mine-young",
    ]
    assert {r["owner"] for r in rows} == {ME} and rows[0]["state"] == "STARTED"
    everyone = CliRunner().invoke(
        app, ["sandbox", "list", "--all", "--json"], terminal_width=200
    )
    assert len(json.loads(everyone.stdout)) == 4
    table = CliRunner().invoke(app, ["sandbox", "list"], terminal_width=200)
    assert "mine-running-leak" in table.output and "Owner" in table.output
