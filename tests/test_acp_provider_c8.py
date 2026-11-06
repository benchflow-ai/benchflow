"""Cleanup must preserve the mixed totals from 74a369b2."""

from types import SimpleNamespace

import pytest

from benchflow.acp.session import ACPSession
from tests.test_session_cost import _log, _response, _state, _unpriced_claude_rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sequence", [(False, True), (True, False), (False, True, False, True)]
)
@pytest.mark.parametrize("provider_cost", [0.25, 0.0, None])
async def test_cleanup_cannot_price_mixed_roles_from_claude_log(
    tmp_path, monkeypatch, sequence, provider_cost
):
    home = tmp_path / "sandbox-home"
    _log(
        home / ".claude" / "projects",
        [
            _response("dummy-message", input_tokens=160, output_tokens=40),
            _state(0.01, input_tokens=160, output_tokens=40),
        ],
    )
    rollout, trial = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
    for covered in sequence:
        rollout._native_usage_proxy_covered = covered
        rollout._native_usage_checkpoint = None
        rollout._session = ACPSession("dummy-session")
        rollout._session.record_prompt_usage(
            SimpleNamespace(
                input_tokens=80 if covered else 160,
                output_tokens=20 if covered else 40,
                total_tokens=100 if covered else 200,
            )
        )
        rollout._collect_native_acp_usage()
        rollout._collect_native_acp_usage()
    rollout._session = None
    rollout._usage_metrics = {
        "usage_source": "provider_response",
        "n_input_tokens": 80 * sum(sequence),
        "n_output_tokens": 20 * sum(sequence),
        "total_tokens": 100 * sum(sequence),
        "cost_usd": provider_cost,
    }

    # Exercise the actual finalization -> pricing -> teardown order, which
    # the C6 test omitted. Repeated cleanup must leave the same totals.
    for _ in range(2):
        await rollout._cleanup_before_teardown()
        usage = rollout._usage_metrics
        assert usage["cost_usd"] is None
        assert usage["price_source"] is None
        assert usage["total_tokens"] == 100 * sum(sequence) + 200 * sequence.count(
            False
        )
        assert usage["usage_details"]["provider_response"]["cost_usd"] == provider_cost
        assert "cost_estimate" not in usage["usage_details"]
    assert not (trial / "agent" / "claude-sessions").exists()


@pytest.mark.asyncio
async def test_unpriced_provider_usage_is_not_replaced_by_agent_estimate(
    tmp_path, monkeypatch
):
    home = tmp_path / "sandbox-home"
    _log(
        home / ".claude" / "projects",
        [
            _response("dummy-message"),
            _state(0.01, input_tokens=1000, output_tokens=50),
        ],
    )
    rollout, trial = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
    rollout._usage_metrics = {
        "usage_source": "provider_response",
        "total_tokens": 100,
        "cost_usd": None,
    }
    await rollout.cleanup()
    assert rollout._usage_metrics["cost_usd"] is None
    assert not (trial / "agent" / "claude-sessions").exists()
