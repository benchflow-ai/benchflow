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
    sb_id: str,
    age: float,
    *,
    owner: str | None = ME,
    active_ago: float | None = None,
    state: str = "SandboxState.STARTED",
):
    now = datetime.now(UTC)
    labels = (
        {"benchflow.managed": f"1:{owner}", "benchflow.owner": owner} if owner else {}
    )
    return SimpleNamespace(
        id=sb_id,
        state=state,
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

    def delete(self, sb, timeout=60, wait=False):
        # Guards PR #13: an acknowledged deletion is not yet a released lease.
        assert wait is True and 0 < timeout <= 60
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
    # Likewise the sweep of killed runs' containers and networks.
    monkeypatch.setattr(sandbox_cli, "cleanup_docker_leftovers", lambda **_: False)


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


def test_reaper_ignore_age_takes_every_owned_sandbox_not_started(monkeypatch) -> None:
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", ME)
    client = FakeClient(
        [
            _sb("a", 1, active_ago=0, state="SandboxState.STOPPED"),
            _sb("b", 1, state="SandboxState.ERROR"),
            _sb("running", 1),
            _sb("x", 1, owner="other"),
        ]
    )
    reasons: dict[str, str] = {}
    reap_stale_sandboxes(
        client, ignore_age=True, on_skip=lambda sb, why: reasons.update({sb.id: why})
    )
    assert sorted(client.deleted) == ["a", "b"]
    assert reasons == {"running": "started", "x": "foreign"}


def test_cleanup_all_keeps_a_started_sandbox_and_id_reclaims_it(fleet) -> None:
    """--all never deletes a STARTED sandbox (DAYTONA-START round 4): Daytona
    stops an abandoned one after its inactivity interval, and --id deletes
    it at once."""
    result = CliRunner().invoke(
        app, ["sandbox", "cleanup", "--all"], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    assert fleet.deleted == []
    assert "3 STARTED" in " ".join(result.output.split())

    result = CliRunner().invoke(
        app,
        ["sandbox", "cleanup", "--id", "mine-running-leak", "--id", "theirs"],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    # --id adds the named one; the age rule still takes the 50 h idle one,
    # and another owner's is never touched.
    assert sorted(fleet.deleted) == ["mine-old-idle", "mine-running-leak"]


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
