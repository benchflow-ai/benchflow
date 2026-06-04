"""Tests for provider token/cost telemetry serialization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.providers.litellm_logging import extract_usage_from_trajectory
from benchflow.providers.runtime import ProviderRuntime, extract_usage
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


def _build_result(rollout_dir: Path, **overrides):
    from benchflow.rollout import _build_rollout_result

    defaults = dict(
        task_name="usage-task",
        rollout_name="usage-rollout",
        agent="claude-agent-acp",
        agent_name="Claude",
        model="claude-haiku-4-5-20251001",
        n_tool_calls=3,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 5, 18, 12, 0),
        timing={},
    )
    defaults.update(overrides)
    return _build_rollout_result(rollout_dir, **defaults)


def _result_json(rollout_dir: Path) -> dict:
    return json.loads((rollout_dir / "result.json").read_text())


def _trajectory(*bodies: dict, model: str = "claude-haiku-4-5-20251001") -> Trajectory:
    trajectory = Trajectory(session_id="s", agent_name="agent")
    for body in bodies:
        trajectory.exchanges.append(
            LLMExchange(
                request=LLMRequest(body={"model": model}),
                response=LLMResponse(body=body),
            )
        )
    return trajectory


def test_result_json_contains_unavailable_usage_defaults(tmp_path):
    result = _build_result(tmp_path)
    data = _result_json(tmp_path)

    assert result.usage_source == "unavailable"
    assert data["agent_result"]["usage_source"] == "unavailable"
    assert data["agent_result"]["n_input_tokens"] is None
    assert data["agent_result"]["n_output_tokens"] is None
    assert data["final_metrics"]["total_prompt_tokens"] is None
    assert data["trajectory_summary"]["partial_trajectory"] is False


def test_result_json_contains_provider_usage_when_supplied(tmp_path):
    result = _build_result(
        tmp_path,
        n_input_tokens=100,
        n_output_tokens=20,
        n_cache_read_tokens=7,
        n_cache_creation_tokens=3,
        total_tokens=130,
        cost_usd=0.0012,
        usage_source="provider_response",
        price_source="pricing_table_2026-05",
    )
    data = _result_json(tmp_path)

    assert result.total_tokens == 130
    assert data["agent_result"]["n_input_tokens"] == 100
    assert data["agent_result"]["n_output_tokens"] == 20
    assert data["agent_result"]["n_cache_read_tokens"] == 7
    assert data["agent_result"]["n_cache_creation_tokens"] == 3
    assert data["agent_result"]["total_tokens"] == 130
    assert data["agent_result"]["cost_usd"] == 0.0012
    assert data["agent_result"]["usage_source"] == "provider_response"
    assert data["final_metrics"] == {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 20,
        "total_cached_tokens": 7,
        "total_cost_usd": 0.0012,
    }


def test_result_json_includes_harbor_style_trajectory_summary(tmp_path):
    _build_result(
        tmp_path,
        trajectory=[
            {"type": "user_message", "text": "solve"},
            {"type": "agent_thought", "text": "thinking"},
            {"type": "tool_call", "status": "completed"},
            {"type": "agent_message", "text": "done"},
            {"type": "tool_call", "status": "failed"},
            {"type": "tool_result", "text": "legacy"},
        ],
        partial_trajectory=True,
        trajectory_source="partial_acp",
    )

    summary = _result_json(tmp_path)["trajectory_summary"]

    assert summary["steps"] == 6
    assert summary["tool_call_steps"] == 2
    assert summary["event_type_counts"]["tool_call"] == 2
    assert summary["tool_call_status_counts"] == {"completed": 1, "failed": 1}
    assert summary["partial_trajectory"] is True
    assert summary["trajectory_source"] == "partial_acp"


def test_extract_usage_none_runtime():
    usage = extract_usage(None)

    assert usage["usage_source"] == "unavailable"
    assert usage["total_tokens"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"message": {"content": "hi"}}]},
        {"usage": {}},
        {"usage": {"input_tokens": None, "output_tokens": None}},
    ],
)
def test_extract_usage_requires_provider_usage_fields(body):
    usage = extract_usage_from_trajectory(_trajectory(body), fallback_model=None)

    assert usage["usage_source"] == "unavailable"


def test_extract_usage_with_anthropic_exchanges():
    usage = extract_usage_from_trajectory(
        _trajectory(
            {
                "model": "claude-haiku-4-5-20251001",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 1,
                },
            },
            {
                "model": "claude-haiku-4-5-20251001",
                "usage": {"input_tokens": 3, "output_tokens": 5},
            },
        ),
        fallback_model="claude-haiku-4-5-20251001",
    )

    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 18
    assert usage["n_output_tokens"] == 7
    assert usage["n_cache_read_tokens"] == 4
    assert usage["n_cache_creation_tokens"] == 1
    assert usage["total_tokens"] == 25


def test_extract_usage_with_openai_and_gemini_shapes():
    openai = extract_usage_from_trajectory(
        _trajectory(
            {
                "model": "gpt-4.1-mini",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 5},
                    "total_tokens": 28,
                },
            }
        ),
        fallback_model="gpt-4.1-mini",
    )
    gemini = extract_usage_from_trajectory(
        _trajectory(
            {
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 4,
                    "cachedContentTokenCount": 3,
                    "totalTokenCount": 16,
                }
            },
            model="gemini-3.5-flash",
        ),
        fallback_model="gemini-3.5-flash",
    )

    assert openai["n_input_tokens"] == 20
    assert openai["n_cache_read_tokens"] == 5
    assert openai["total_tokens"] == 28
    assert gemini["n_input_tokens"] == 12
    assert gemini["n_cache_read_tokens"] == 3
    assert gemini["total_tokens"] == 16


def test_extract_usage_uses_litellm_computed_cost():
    # Cost is whatever LiteLLM computed (summed into metadata by the importer);
    # BenchFlow no longer estimates from tokens.
    trajectory = _trajectory(
        {
            "model": "bedrock/us.anthropic.claude-opus-4-8",
            "usage": {
                "inputTokens": 1000,
                "outputTokens": 200,
                "cacheReadInputTokens": 500,
            },
        }
    )
    trajectory.metadata["cost_usd"] = 0.0123
    usage = extract_usage_from_trajectory(trajectory, fallback_model=None)

    assert usage["usage_source"] == "provider_response"
    assert usage["cost_usd"] == 0.0123
    assert usage["price_source"] == "litellm"


def test_extract_usage_cost_none_when_litellm_did_not_price():
    # Custom/unpriced model with no per-route override: tokens still recorded,
    # cost is honestly None (no hand-rolled estimate).
    trajectory = _trajectory(
        {
            "model": "openai/some-unpriced-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    )
    usage = extract_usage_from_trajectory(trajectory, fallback_model=None)

    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 10
    assert usage["cost_usd"] is None
    assert usage["price_source"] is None


def test_extract_usage_reads_litellm_runtime_trajectory():
    trajectory = _trajectory(
        {
            "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url="http://127.0.0.1:4000",
        backend_model="claude-haiku-4-5-20251001",
        server=SimpleNamespace(trajectory=trajectory),
    )

    usage = extract_usage(runtime)

    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 10
    assert usage["n_output_tokens"] == 2


@pytest.mark.asyncio
async def test_rollout_cleanup_extracts_usage_and_writes_llm_trajectory(tmp_path):
    from benchflow.rollout import Rollout, RolloutConfig

    class FakeServer:
        def __init__(self):
            self.trajectory = _trajectory(
                {
                    "model": "claude-haiku-4-5-20251001",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            )

        async def stop(self):
            return None

    server = FakeServer()
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=tmp_path / "task")
    rollout._trajectory = []
    rollout._acp_client = None
    rollout._agent_launch = ""
    rollout._env = SimpleNamespace(stop=AsyncMock())
    rollout._environment = None
    rollout._usage_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url="http://127.0.0.1:4000",
        backend_model="claude-haiku-4-5-20251001",
        server=server,
    )
    rollout._planes = SimpleNamespace(
        stop_provider_runtime=lambda runtime: runtime.server.stop(),
        extract_usage=extract_usage,
    )
    rollout._rollout_dir = tmp_path
    rollout._env_externally_owned = False

    await rollout.cleanup()

    assert rollout._usage_metrics["usage_source"] == "provider_response"
    assert (tmp_path / "trajectory" / "llm_trajectory.jsonl").exists()

