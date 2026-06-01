"""Tests for provider-auth detection at the rollout boundary (PR #564 / #546).

The real failure mode: an invalid provider key surfaces at the ACP layer only
as a generic "ACP error -32603: Internal error"; the actual 401/403 is visible
only in the proxy-captured trajectory. ``Rollout._provider_auth_status`` reads
that status (status code only — never body/headers) so a sanitized auth marker
can be appended to ``result.error`` and the retry classifier can fail fast.
"""

from types import SimpleNamespace

from benchflow.rollout import Rollout


def _rollout_with_statuses(statuses):
    """Build a bare Rollout whose proxy trajectory has the given response statuses."""
    exchanges = [
        SimpleNamespace(response=SimpleNamespace(status_code=s)) for s in statuses
    ]
    trajectory = SimpleNamespace(exchanges=exchanges)
    server = SimpleNamespace(trajectory=trajectory)
    rollout = Rollout.__new__(Rollout)
    rollout._usage_runtime = SimpleNamespace(server=server)
    return rollout


def test_detects_401_in_trajectory():
    """Guards PR #564: a provider 401 in the proxy trajectory is surfaced."""
    assert _rollout_with_statuses([200, 401])._provider_auth_status() == 401


def test_detects_403_in_trajectory():
    """Guards PR #564: a provider 403 in the proxy trajectory is surfaced."""
    assert _rollout_with_statuses([403])._provider_auth_status() == 403


def test_returns_last_auth_status():
    """The most recent auth failure wins (trajectory scanned newest-first)."""
    assert _rollout_with_statuses([401, 200, 403])._provider_auth_status() == 403


def test_no_auth_status_when_all_ok():
    """Guards PR #564: a healthy trajectory must not be flagged as auth failure."""
    assert _rollout_with_statuses([200, 200, 500])._provider_auth_status() is None


def test_missing_usage_runtime_is_safe():
    """No proxy runtime (e.g. oracle runs) must not raise — returns None."""
    rollout = Rollout.__new__(Rollout)
    rollout._usage_runtime = None
    assert rollout._provider_auth_status() is None


def test_empty_trajectory_is_safe():
    assert _rollout_with_statuses([])._provider_auth_status() is None
