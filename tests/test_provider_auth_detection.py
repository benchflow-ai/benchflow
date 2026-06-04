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

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
from benchflow.rollout import _provider_auth_status_from_runtime


def _runtime_with_statuses(statuses):
    exchanges = [
        SimpleNamespace(response=SimpleNamespace(status_code=s)) for s in statuses
    ]
    trajectory = SimpleNamespace(exchanges=exchanges)
    return SimpleNamespace(server=SimpleNamespace(trajectory=trajectory))


def _runtime_with_trajectory(trajectory):
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


def test_litellm_auth_failure_import_drives_sanitized_provider_marker(tmp_path):
    """Guards PR #564: LiteLLM callback auth failures that only expose ``401
    Invalid bearer token`` inside the failure payload must still surface as
    provider_auth without leaking the payload into ``result.error``."""
    from benchflow.agents.errors import AgentProtocolError

    record = {
        "event": "failure",
        "request_model": "benchflow-claude",
        "provider_model": "anthropic/claude-opus-4-8",
        "request": {"method": "POST", "path": "/v1/messages", "body": {}},
        "response": {},
        "error": {
            "type": "AuthenticationError",
            "message": (
                "litellm.AuthenticationError: AnthropicException - "
                "API Error: 401 Invalid bearer token"
            ),
        },
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:00",
    }

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="openhands",
    )
    status = _provider_auth_status_from_runtime(_runtime_with_trajectory(trajectory))

    assert trajectory.exchanges[0].response.status_code == 401
    assert status == 401

    rollout = _auth_rollout(tmp_path)
    rollout._provider_auth_status_cached = status

    classified = rollout._classify_acp_error(
        AgentProtocolError("ACP error -32603: Internal error")
    )

    assert classified == (
        "ACP error -32603: Internal error | provider auth failed (HTTP 401)"
    )
    assert "Invalid bearer token" not in classified


def test_litellm_non_auth_failure_import_remains_generic_500():
    """Guards PR #564: non-auth LiteLLM failures stay 500/None."""
    record = {
        "event": "failure",
        "request_model": "benchflow-gpt",
        "request": {"method": "POST", "path": "/v1/chat/completions", "body": {}},
        "error": {
            "type": "APIConnectionError",
            "message": "upstream connection reset while reading response",
        },
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:00",
    }

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="codex-acp",
    )

    assert trajectory.exchanges[0].response.status_code == 500
    assert (
        _provider_auth_status_from_runtime(_runtime_with_trajectory(trajectory)) is None
    )


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


# ── Rollout-level run()/cleanup() ordering (PR #564 deeper review) ──


def _auth_rollout(tmp_path, *, usage_source="unavailable"):
    """Build a minimal real Rollout whose usage proxy carries a 401 but reports
    no provider token usage — the exact shape of a provider-auth failure under
    ``usage_tracking.mode == "required"`` (PR #564 / issue #546).

    The FakeServer's proxy trajectory holds a single 401 exchange, so
    cleanup()'s ``_provider_auth_status_from_runtime`` snapshot returns 401,
    while ``extract_usage`` yields ``usage_source`` (default ``"unavailable"``)
    so the required-usage enforcement path is reachable.
    """
    from benchflow.providers.runtime import ProviderRuntime
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    class FakeServer:
        trajectory = SimpleNamespace(
            exchanges=[SimpleNamespace(response=SimpleNamespace(status_code=401))]
        )

        async def stop(self):
            return None

    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=tmp_path / "task",
        usage_tracking=UsageTrackingConfig(mode="required"),
    )
    rollout._error = None
    rollout._trajectory = []
    rollout._acp_client = None
    rollout._agent_launch = ""
    rollout._env = SimpleNamespace(stop=AsyncMock())
    rollout._environment = None
    rollout._provider_runtime = None
    rollout._provider_auth_status_cached = None
    rollout._usage_runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url="http://host.docker.internal:32124",
        backend_model="gpt-5.5",
        server=FakeServer(),
    )
    rollout._planes = SimpleNamespace(
        stop_provider_runtime=lambda runtime: runtime.server.stop(),
        extract_usage=lambda runtime: {"usage_source": usage_source},
    )
    rollout._rollout_dir = tmp_path
    return rollout


def test_classify_acp_error_handles_base_error_without_message(tmp_path):
    """Guards PR #564 / issue #546: a base ``AgentProtocolError`` (which only
    annotates ``message: str`` and never assigns it) must not AttributeError in
    ``_classify_acp_error``; the sanitized ``provider auth failed (HTTP 401)``
    marker is still appended when a 401 snapshot is present.

    FAILS if FIX 2 is reverted to reading ``e.message`` directly.
    """
    from benchflow.agents.errors import AgentProtocolError

    rollout = _auth_rollout(tmp_path)
    rollout._provider_auth_status_cached = 401

    err = AgentProtocolError("ACP error -32603: Internal error")
    assert not hasattr(err, "message")

    classified = rollout._classify_acp_error(err)
    assert classified == (
        "ACP error -32603: Internal error | provider auth failed (HTTP 401)"
    )


@pytest.mark.asyncio
async def test_run_auth_failure_sets_provisional_error_before_cleanup(
    tmp_path, monkeypatch
):
    """KEY guarding test for FIX 1 (PR #564 / issue #546).

    Drives the real ``Rollout.run()`` ordering: install_agent raises an ACPError,
    cleanup() runs ``_enforce_required_usage_tracking`` (usage mode required, no
    provider usage captured), then the post-cleanup block refines self._error to
    the provider_auth marker.

    A spy wraps the real ``_classify_acp_error`` and records ``self._error`` at
    the seam — after cleanup's enforcement, before refinement. With FIX 1 the
    recorded value is the provisional ACP-error string (enforcement skipped);
    without it the recorded value becomes the spurious "Token usage tracking is
    required..." message, so this test FAILS if FIX 1 is reverted.
    """
    from benchflow.acp.client import ACPError

    rollout = _auth_rollout(tmp_path, usage_source="unavailable")
    # Lightweight RolloutResult path (no trial dir / _build_result needed).
    rollout._rollout_dir = None

    monkeypatch.setattr(rollout, "setup", AsyncMock())
    monkeypatch.setattr(rollout, "start", AsyncMock())
    monkeypatch.setattr(
        rollout,
        "install_agent",
        AsyncMock(side_effect=ACPError(-32603, "Internal error")),
    )

    recorded = {}
    real_classify = rollout._classify_acp_error

    def spy_classify(e):
        recorded["error_at_seam"] = rollout._error
        return real_classify(e)

    monkeypatch.setattr(rollout, "_classify_acp_error", spy_classify)

    result = await rollout.run()

    # Enforcement was skipped because run() set a provisional self._error — the
    # seam value is the ACP-error string, NOT the spurious usage message.
    assert recorded["error_at_seam"] == "ACP error -32603: Internal error"
    assert result.error == (
        "ACP error -32603: Internal error | provider auth failed (HTTP 401)"
    )
    assert rollout._provider_auth_status_cached == 401


def test_enforcement_still_fires_when_no_error_and_usage_missing(tmp_path):
    """Negative control for FIX 1 (PR #564 / issue #546): on a clean run with no
    ACP error and no captured provider usage, the legitimate required-usage
    enforcement message must STILL fire — the provisional-error guard only
    suppresses it when an error is already set.
    """
    rollout = _auth_rollout(tmp_path, usage_source="unavailable")
    rollout._usage_metrics = {"usage_source": "unavailable"}
    rollout._error = None

    rollout._enforce_required_usage_tracking()

    assert rollout._error == (
        "Token usage tracking is required, but no provider token usage was captured."
    )
