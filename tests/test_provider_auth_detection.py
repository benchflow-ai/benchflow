"""Tests for provider-auth detection at the rollout boundary (PR #564 / #546).

The real failure mode: an invalid provider key surfaces at the ACP layer only
as a generic "ACP error -32603: Internal error"; the actual 401/403 is visible
only in the proxy-captured trajectory. The rollout snapshots that status (code
only — never body/headers) after the usage proxy imports its captures, so a
sanitized marker can be appended to ``result.error`` and the retry classifier
can fail fast.

PR #564 review finding 1: Daytona's SandboxUsageProxy only fills its trajectory
on ``stop()``, which runs during ``cleanup()`` — *after* the old code classified
the ACP error. Classification now happens after cleanup, reading a status
snapshot taken once captures are imported.
"""

from types import SimpleNamespace

from benchflow.rollout import _provider_auth_status_from_runtime


def _runtime_with_statuses(statuses):
    exchanges = [
        SimpleNamespace(response=SimpleNamespace(status_code=s)) for s in statuses
    ]
    trajectory = SimpleNamespace(exchanges=exchanges)
    return SimpleNamespace(server=SimpleNamespace(trajectory=trajectory))


def test_detects_401_in_trajectory():
    """Guards PR #564: a provider 401 in the proxy trajectory is surfaced."""
    assert _provider_auth_status_from_runtime(_runtime_with_statuses([200, 401])) == 401


def test_detects_403_in_trajectory():
    """Guards PR #564: a provider 403 in the proxy trajectory is surfaced."""
    assert _provider_auth_status_from_runtime(_runtime_with_statuses([403])) == 403


def test_returns_last_auth_status():
    """The most recent auth failure wins (trajectory scanned newest-first)."""
    assert (
        _provider_auth_status_from_runtime(_runtime_with_statuses([401, 200, 403]))
        == 403
    )


def test_no_auth_status_when_all_ok():
    """Guards PR #564: a healthy trajectory must not be flagged as auth failure."""
    assert (
        _provider_auth_status_from_runtime(_runtime_with_statuses([200, 200, 500]))
        is None
    )


def test_missing_runtime_is_safe():
    """No proxy runtime (e.g. oracle runs) must not raise — returns None."""
    assert _provider_auth_status_from_runtime(None) is None


def test_empty_trajectory_is_safe():
    assert _provider_auth_status_from_runtime(_runtime_with_statuses([])) is None


def test_snapshot_after_late_capture_import():
    """Guards PR #564 finding 1: a proxy that only populates its trajectory on
    stop() (Daytona's SandboxUsageProxy) must still yield the 401 once captures
    are imported — the snapshot is read after, not before, import."""

    class LateProxy:
        """server.trajectory is empty until stop() imports captures."""

        def __init__(self):
            self.server = SimpleNamespace(trajectory=SimpleNamespace(exchanges=[]))

        def stop(self):
            self.server.trajectory.exchanges = [
                SimpleNamespace(response=SimpleNamespace(status_code=401))
            ]

    proxy = LateProxy()
    # Before import: nothing visible (this is exactly the old-code bug).
    assert _provider_auth_status_from_runtime(proxy) is None
    proxy.stop()
    # After import (what cleanup() now does before snapshotting): 401 visible.
    assert _provider_auth_status_from_runtime(proxy) == 401
