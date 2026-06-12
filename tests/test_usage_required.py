"""Tests for required provider token usage enforcement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


def _trajectory(body: dict) -> Trajectory:
    trajectory = Trajectory(session_id="s1", agent_name="agent")
    trajectory.exchanges.append(
        LLMExchange(
            request=LLMRequest(body={"model": "gpt-5.5", "messages": []}),
            response=LLMResponse(body=body),
        )
    )
    return trajectory


@pytest.mark.asyncio
async def test_required_usage_tracking_fails_when_provider_usage_missing(tmp_path):
    """Guards PR #587: required usage must not silently pass without tokens."""
    from benchflow.providers.runtime import ProviderRuntime, extract_usage
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.usage_tracking import UsageTrackingConfig

    class FakeServer:
        trajectory = _trajectory({"error": {"type": "budget_exceeded"}})

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
    rollout._usage_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url="http://host.docker.internal:32124",
        backend_model="gpt-5.5",
        server=FakeServer(),
    )
    rollout._planes = SimpleNamespace(
        stop_provider_runtime=lambda runtime: runtime.server.stop(),
        extract_usage=extract_usage,
    )
    rollout._rollout_dir = tmp_path

    await rollout.cleanup()

    assert rollout._usage_metrics["usage_source"] == "unavailable"
    assert rollout._error == (
        "Token usage tracking is required, but no provider token usage was captured."
    )
