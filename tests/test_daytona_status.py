"""Unit tests for the dashboard Daytona panel's snapshot() domain logic.

Covers the SDK-backed path without touching the network: the daytona client
bootstrap (``benchflow.sandbox.daytona.build_sync_client``) is monkeypatched, so
these exercise row-building, state-enum stripping, ordering, and the
error-returning contract — the path the 49 existing dashboard tests don't reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import daytona_status  # noqa: E402


class _FakeSandbox:
    def __init__(self, id_: str, state: str, created_at: str, target: str = "us"):
        self.id = id_
        self.state = state
        self.created_at = created_at
        self.target = target


def _client_returning(sandboxes):
    class _Client:
        def list(self):
            return iter(sandboxes)

    return lambda api_key=None: _Client()


def test_snapshot_builds_rows_states_and_order(monkeypatch):
    import benchflow.sandbox.daytona as bd

    sandboxes = [
        _FakeSandbox("a", "SandboxState.STARTED", "2026-06-03T10:00:00+00:00"),
        _FakeSandbox("b", "SandboxState.DESTROYING", "2026-06-01T10:00:00+00:00"),
        _FakeSandbox("c", "SandboxState.STARTED", "2026-06-03T11:00:00+00:00"),
    ]
    monkeypatch.setattr(bd, "build_sync_client", _client_returning(sandboxes))

    r = daytona_status.snapshot("key")

    assert "error" not in r
    assert r["count"] == 3
    # enum prefix stripped to the bare state name
    assert r["by_state"] == {"DESTROYING": 1, "STARTED": 2}
    # sorted by created ascending -> the 06-01 sandbox is first
    assert [row["id"] for row in r["rows"]] == ["b", "a", "c"]
    assert r["rows"][0]["state"] == "DESTROYING"
    assert r["rows"][0]["target"] == "us"
    assert r["rows"][0]["age"] != "?"  # a parseable timestamp yields a real age
    assert r["as_of"]


def test_snapshot_missing_created_time_is_tolerated(monkeypatch):
    import benchflow.sandbox.daytona as bd

    monkeypatch.setattr(
        bd,
        "build_sync_client",
        _client_returning([_FakeSandbox("x", "SandboxState.ERROR", "")]),
    )

    r = daytona_status.snapshot("key")

    assert r["count"] == 1
    assert r["rows"][0]["created"] == "?"
    assert r["rows"][0]["age"] == "?"
    assert r["by_state"] == {"ERROR": 1}


def test_snapshot_returns_error_instead_of_raising(monkeypatch):
    import benchflow.sandbox.daytona as bd

    def _boom(api_key=None):
        raise RuntimeError("transport down")

    monkeypatch.setattr(bd, "build_sync_client", _boom)

    r = daytona_status.snapshot("key")

    assert r["count"] == 0
    assert r["rows"] == []
    assert "Daytona list() failed" in r["error"]
