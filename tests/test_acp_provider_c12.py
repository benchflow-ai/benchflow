"""Missing-telemetry coverage omitted by b1838d3b."""

from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.acp.session import ACPSession
from tests.test_session_cost import _log, _response, _state, _unpriced_claude_rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order", [(True, False), (False, True), (True, False, True, False)]
)
@pytest.mark.parametrize("telemetry", ["absent", "zero", "adapter", "reported"])
@pytest.mark.parametrize("agent", ["codex-acp", "claude-agent-acp", "gemini", None])
@pytest.mark.parametrize("provider_cost", [0.25, 0.0, None])
async def test_missing_native_telemetry_never_makes_provider_subtotal_complete(
    tmp_path, monkeypatch, order, telemetry, agent, provider_cost
):
    home = tmp_path / "sandbox-home"
    _log(
        home / ".claude" / "projects",
        [_response("dummy"), _state(0.01, input_tokens=1000, output_tokens=50)],
    )
    rollout, trial = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
    rollout._native_usage_metrics = {}
    for covered in order:
        rollout._active_role = SimpleNamespace(agent=agent) if agent else object()
        rollout._native_usage_proxy_covered = covered
        rollout._native_usage_checkpoint = None
        rollout._session = ACPSession("dummy")
        if covered or telemetry in {"zero", "reported"}:
            n = 100 if covered or telemetry == "reported" else 0
            rollout._session.record_prompt_usage(
                SimpleNamespace(input_tokens=n, output_tokens=0, total_tokens=n)
            )
        elif telemetry == "adapter":
            rollout._session = SimpleNamespace()
        # disconnect is the timeout and role-transition collection path.
        rollout._collect_native_acp_usage()
        await rollout.disconnect()
    provider: dict[str, Any] = {
        "usage_source": "provider_response",
        "n_input_tokens": 100 * sum(order),
        "n_output_tokens": 0,
        "total_tokens": 100 * sum(order),
        "cost_usd": provider_cost,
        "price_source": "provider_response" if provider_cost is not None else None,
    }
    rollout._usage_metrics = dict(provider)
    for _ in range(2):
        await rollout._cleanup_before_teardown()
        metrics = rollout._usage_metrics
        assert metrics["cost_usd"] is None
        assert metrics["price_source"] is None
        assert metrics["total_tokens"] == 100 * (
            sum(order) + (order.count(False) if telemetry == "reported" else 0)
        )
        assert metrics["usage_details"]["provider_response"] == provider
        assert "cost_estimate" not in metrics["usage_details"]
    assert not (trial / "agent" / "claude-sessions").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("session", ["absent", "unreported", "zero"])
@pytest.mark.parametrize("provider_cost", [0.25, 0.0, None])
async def test_covered_or_absent_session_keeps_provider_total(
    tmp_path, monkeypatch, session, provider_cost
):
    rollout, _ = _unpriced_claude_rollout(
        tmp_path, monkeypatch, tmp_path / "sandbox-home", {}
    )
    rollout._native_usage_proxy_covered = True
    rollout._active_role = SimpleNamespace(agent="codex-acp")
    if session != "absent":
        rollout._session = ACPSession("dummy")
        if session == "zero":
            rollout._session.record_prompt_usage(
                SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)
            )
    await rollout.disconnect()
    provider = {
        "usage_source": "provider_response",
        "total_tokens": 100,
        "cost_usd": provider_cost,
    }
    rollout._usage_metrics = dict(provider)
    for _ in range(2):
        await rollout._cleanup_before_teardown()
        assert rollout._usage_metrics == provider
