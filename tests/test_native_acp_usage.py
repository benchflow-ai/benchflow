from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_acp_client_records_prompt_response_usage():
    """Guards PR #613 follow-up: native ACP usage is captured from prompt results."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession

    session = ACPSession("session-1")
    client = ACPClient.__new__(ACPClient)
    client._session = session

    async def fake_send_request(method, params):
        assert method == "session/prompt"
        assert params["sessionId"] == "session-1"
        return {
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "totalTokens": 16,
                "cachedReadTokens": 2,
                "cachedWriteTokens": 1,
                "thoughtTokens": 1,
            },
        }

    client._send_request = fake_send_request

    result = await client.prompt("solve")

    assert result.stop_reason == "end_turn"
    assert session.latest_usage_totals() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 16,
        "cached_read_tokens": 2,
        "cached_write_tokens": 1,
        "thought_tokens": 1,
    }


@pytest.mark.asyncio
async def test_disconnect_preserves_native_usage_in_final_metrics():
    """Guards issue #933 against usage loss demonstrated by PR #1051."""
    from benchflow.acp.session import ACPSession
    from benchflow.rollout import Rollout

    session = ACPSession("timeout-session")
    session.record_prompt_usage(
        SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14)
    )
    client = SimpleNamespace(session=session, close=AsyncMock())
    rollout = Rollout.__new__(Rollout)
    rollout._acp_client = client
    rollout._session = session
    rollout._session_adapter = None
    rollout._is_session_factory = False
    rollout._trajectory = []
    rollout._session_traj_count = 0
    rollout._native_usage_checkpoint = None
    rollout._native_usage_metrics = {"total_tokens": 0}
    rollout._usage_metrics = {"usage_source": "unavailable"}
    rollout._agent_launch = ""
    rollout._env = None
    rollout._phase = "connected"

    await rollout.disconnect()
    rollout._finalize_usage_metrics()

    assert rollout._usage_metrics["usage_source"] == "agent_native_acp"
    assert rollout._usage_metrics["total_tokens"] == 14


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fx_reason,spec_reason",
    [
        ("refused", "refusal"),
        ("max_output_tokens", "max_tokens"),
        ("max_model_turns", "max_turn_requests"),
    ],
)
async def test_acp_client_normalizes_fx_stop_reasons(fx_reason, spec_reason):
    """Guards PR #1052: fx's non-spec stop reasons parse as spec values."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession

    client = ACPClient.__new__(ACPClient)
    client._session = ACPSession("session-1")

    async def fake_send_request(method, params):
        return {"stopReason": fx_reason}

    client._send_request = fake_send_request

    result = await client.prompt("solve")

    assert result.stop_reason == spec_reason


@pytest.mark.asyncio
async def test_acp_client_normalizes_fx_usage_keys():
    """Guards PR #1052: fx's usage keys (cacheReadTokens, reasoningTokens, no
    totalTokens) are captured instead of failing PromptResponse validation."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession

    client = ACPClient.__new__(ACPClient)
    client._session = ACPSession("session-1")

    async def fake_send_request(method, params):
        return {
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadTokens": 2,
                "cacheWriteTokens": 1,
                "reasoningTokens": 3,
            },
        }

    client._send_request = fake_send_request

    await client.prompt("solve")

    assert client._session.latest_usage_totals() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "cached_read_tokens": 2,
        "cached_write_tokens": 1,
        "thought_tokens": 3,
    }


def test_rollout_native_acp_usage_uses_cumulative_deltas():
    """Guards PR #613 follow-up: ACP cumulative usage is not double-counted."""
    from benchflow.acp.session import ACPSession
    from benchflow.rollout import Rollout

    session = ACPSession("session-1")
    rollout = Rollout.__new__(Rollout)
    rollout._session = session
    rollout._native_usage_checkpoint = None

    session.record_prompt_usage(
        SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            total_tokens=16,
            cached_read_tokens=2,
            cached_write_tokens=1,
            thought_tokens=1,
        )
    )
    rollout._collect_native_acp_usage()
    session.record_prompt_usage(
        SimpleNamespace(
            input_tokens=13,
            output_tokens=9,
            total_tokens=27,
            cached_read_tokens=3,
            cached_write_tokens=1,
            thought_tokens=3,
        )
    )
    rollout._collect_native_acp_usage()

    assert rollout._native_usage_metrics == {
        "n_input_tokens": 13,
        "n_output_tokens": 9,
        "n_cache_read_tokens": 3,
        "n_cache_creation_tokens": 1,
        "total_tokens": 27,
        "cost_usd": None,
        "usage_source": "agent_native_acp",
        "price_source": None,
        "usage_details": {"thought_tokens": 3},
    }


def test_rollout_provider_usage_wins_over_native_acp_usage():
    """Guards PR #613 follow-up: LiteLLM provider telemetry remains authoritative."""
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._usage_metrics = {
        "n_input_tokens": 100,
        "n_output_tokens": 20,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 120,
        "cost_usd": 0.01,
        "usage_source": "provider_response",
        "price_source": "litellm",
    }
    rollout._native_usage_metrics = {
        "n_input_tokens": 10,
        "n_output_tokens": 5,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 15,
        "cost_usd": None,
        "usage_source": "agent_native_acp",
        "price_source": None,
        "usage_details": {"thought_tokens": 0},
    }

    rollout._finalize_usage_metrics()

    assert rollout._usage_metrics["usage_source"] == "provider_response"
    assert rollout._usage_metrics["total_tokens"] == 120


def test_required_usage_accepts_native_acp_usage(tmp_path):
    """Guards PR #613 follow-up: required tracking accepts native ACP telemetry."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=tmp_path / "task",
        usage_tracking=UsageTrackingConfig(mode="required"),
    )
    rollout._error = None
    rollout._usage_metrics = {"usage_source": "unavailable"}
    rollout._native_usage_metrics = {
        "n_input_tokens": 10,
        "n_output_tokens": 5,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 15,
        "cost_usd": None,
        "usage_source": "agent_native_acp",
        "price_source": None,
        "usage_details": {"thought_tokens": 0},
    }

    rollout._finalize_usage_metrics()
    rollout._enforce_required_usage_tracking()

    assert rollout._error is None
    assert rollout._usage_metrics["usage_source"] == "agent_native_acp"


def test_native_acp_usage_result_json_and_metadata(tmp_path):
    """Guards PR #613 follow-up: result.json exposes native ACP usage metadata."""
    from benchflow.rollout import _build_rollout_result
    from benchflow.usage_tracking import UsageTrackingConfig

    _build_rollout_result(
        tmp_path,
        task_name="usage-task",
        rollout_name="usage-rollout",
        agent="codex-acp",
        agent_name="Codex",
        model="gpt-4o",
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 6, 4, 12, 0),
        timing={},
        n_input_tokens=10,
        n_output_tokens=5,
        n_cache_read_tokens=2,
        n_cache_creation_tokens=1,
        total_tokens=20,
        usage_source="agent_native_acp",
        usage_details={"thought_tokens": 2},
        usage_tracking=UsageTrackingConfig(mode="required").to_result_metadata(
            environment="docker",
            status="enabled",
            usage_source="agent_native_acp",
        ),
    )

    data = json.loads((tmp_path / "result.json").read_text())

    assert data["agent_result"]["usage_source"] == "agent_native_acp"
    assert data["agent_result"]["usage_details"] == {"thought_tokens": 2}
    assert data["usage_tracking"]["endpoint_kind"] == "agent_native"


def test_usage_summary_includes_native_acp_usage():
    """Guards PR #613 follow-up: native ACP usage counts as telemetry coverage."""
    from benchflow._utils.evaluation_results import usage_summary

    summary = usage_summary(
        {
            "provider": {
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "agent_result": {
                    "usage_source": "provider_response",
                    "n_input_tokens": 10,
                    "n_output_tokens": 5,
                    "n_cache_read_tokens": 0,
                    "n_cache_creation_tokens": 0,
                    "total_tokens": 15,
                    "cost_usd": 0.01,
                },
            },
            "native": {
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "agent_result": {
                    "usage_source": "agent_native_acp",
                    "n_input_tokens": 20,
                    "n_output_tokens": 7,
                    "n_cache_read_tokens": 1,
                    "n_cache_creation_tokens": 2,
                    "total_tokens": 30,
                    "cost_usd": None,
                },
            },
        }
    )

    assert summary["total_input_tokens"] == 30
    assert summary["total_output_tokens"] == 12
    assert summary["total_tokens"] == 45
    assert summary["total_cost_usd"] == 0.01
    assert summary["telemetry_coverage"] == 1.0
