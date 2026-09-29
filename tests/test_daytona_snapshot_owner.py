"""Daytona branch snapshots carry their owner, and the reaper removes stale ones.

Regression test: ``DaytonaSandbox.snapshot()`` named snapshots
``bf-snap-<task dir>-<hex>`` without ``BENCHFLOW_DAYTONA_OWNER``, Daytona
snapshots carry no labels, and the reaper only looked at sandboxes, so an
owner-scoped sweep could not find a leaked snapshot. Names now start with
``bf-snap-<owner>-<owner hash>-``; the hash keeps owner ``team`` from matching
owner ``team-b``'s snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.daytona_reaper import (
    benchflow_snapshot_name,
    owner_snapshot_prefix,
    reap_stale_snapshots,
)
from benchflow.sandbox.protocol import ExecResult

NOW = datetime.now(UTC)


def _snap(name: str, age_min: float):
    return SimpleNamespace(
        name=name, id=f"id-{name}", created_at=NOW - timedelta(minutes=age_min)
    )


class _Client:
    def __init__(self, pages: list[list[SimpleNamespace]]):
        self.pages = pages
        self.deleted: list[str] = []
        self.listed = 0
        self.snapshot = self

    def list(self, page=None, limit=None):
        self.listed += 1
        items = self.pages[(page or 1) - 1] if self.pages else []
        return SimpleNamespace(items=items, total_pages=len(self.pages) or 1)

    def delete(self, snapshot):
        self.deleted.append(snapshot.name)


def test_snapshot_name_carries_the_owner(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "Team_A")
    name = benchflow_snapshot_name("hello-world-task")
    assert name.startswith("bf-snap-team-a-")
    assert name.split("-")[4] != "hello"  # the owner hash sits before the task
    assert "hello-world-task" in name
    assert name == name.lower()


def test_unscoped_snapshot_name_keeps_the_old_shape(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    name = benchflow_snapshot_name("hello_world_task")
    assert name.startswith("bf-snap-hello-world-task-")


async def test_daytona_direct_snapshot_uses_the_owner_name(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team")
    created: list[str] = []
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox.environment_name = "hello-world-task"
    sandbox._sandbox = SimpleNamespace(
        id="sb-1",
        _experimental_create_snapshot=AsyncMock(side_effect=created.append),
    )
    sandbox._strategy = _DaytonaDirect(sandbox)
    # No credential files in this sandbox: the scrub lists nothing.
    sandbox._strategy._credential_ops = lambda: SimpleNamespace(
        run=AsyncMock(return_value=ExecResult(0, "", ""))
    )
    image = await sandbox.snapshot()
    assert image.ref == created[0]
    assert image.ref.startswith(owner_snapshot_prefix())
    assert "hello-world-task" in image.ref


def test_reaper_deletes_only_this_owners_stale_snapshots(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team")
    mine_old = benchflow_snapshot_name("task-a")
    mine_new = benchflow_snapshot_name("task-b")
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team-b")
    other_owner = benchflow_snapshot_name("task-a")
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER")
    unscoped = benchflow_snapshot_name("task-a")
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team")
    client = _Client(
        [
            [_snap(mine_old, 2000), _snap(other_owner, 5000)],
            [
                _snap(unscoped, 5000),
                _snap("other-owner-image", 9000),
                _snap(mine_new, 5),
            ],
        ]
    )
    decisions = []
    counts = reap_stale_snapshots(
        client,
        max_age_minutes=1440,
        on_decision=lambda snap, age, will: decisions.append((snap.name, will)),
    )
    assert client.deleted == [mine_old]
    assert counts == {"found": 5, "deleted": 1, "skipped": 4, "failed": 0}
    assert decisions == [(mine_old, True), (mine_new, False)]


def test_reaper_dry_run_deletes_nothing(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team")
    client = _Client([[_snap(benchflow_snapshot_name("t"), 2000)]])
    counts = reap_stale_snapshots(client, dry_run=True)
    assert client.deleted == []
    assert counts["deleted"] == 1


def test_reaper_without_an_owner_never_lists_snapshots(monkeypatch):
    """Without an owner scope nothing proves a snapshot is ours."""
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    client = _Client([[_snap("bf-snap-task-a-0123456789ab", 9000)]])
    assert reap_stale_snapshots(client) == {
        "found": 0,
        "deleted": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert client.listed == 0


def test_reaper_rejects_a_negative_age(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team")
    with pytest.raises(ValueError):
        reap_stale_snapshots(_Client([]), max_age_minutes=-1)
