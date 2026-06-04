from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ProviderRuntime, ensure_litellm_runtime


class FakeSandboxLiteLLM:
    def __init__(self, base_url: str, route):
        self._base_url = base_url
        self.route = route
        self.trajectory = None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def is_running(self) -> bool:
        return True

    async def stop(self) -> None:
        return None


@pytest.mark.asyncio
async def test_registered_provider_uses_sandbox_local_litellm(monkeypatch):
    async def fake_start(**kwargs):
        return FakeSandboxLiteLLM("http://127.0.0.1:45678", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_start)
    updated, provider_runtime = await ensure_litellm_runtime(
        agent="opencode",
        agent_env={
            "MINIMAX_API_KEY": "key",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
        model="minimax/MiniMax-M3",
        runtime=None,
        environment="daytona",
        session_id="run",
        sandbox=SimpleNamespace(),
    )

    assert provider_runtime.kind == "litellm"
    assert provider_runtime.backend_model == "openai/MiniMax-M3"
    assert updated["OPENAI_BASE_URL"] == "http://127.0.0.1:45678/v1"


@pytest.mark.asyncio
async def test_dead_daytona_runtime_is_replaced(monkeypatch):
    async def is_running():
        return False

    old_server = SimpleNamespace(is_running=is_running)

    async def fake_stop():
        old_server.stopped = True

    old_server.stop = fake_stop

    async def fake_start(**kwargs):
        return FakeSandboxLiteLLM("http://127.0.0.1:45679", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_start)
    old_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url="http://127.0.0.1:45678",
        server=old_server,
        config_key="stale",
    )

    updated, new_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-openai"},
        model="openai/gpt-4.1-mini",
        runtime=old_runtime,
        environment="daytona",
        session_id="run",
        sandbox=SimpleNamespace(),
    )

    assert getattr(old_server, "stopped", False) is True
    assert new_runtime is not old_runtime
    assert updated["OPENAI_BASE_URL"] == "http://127.0.0.1:45679/v1"


def test_extract_usage_accepts_bedrock_converse_usage_shape():
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    trajectory = Trajectory(session_id="s", agent_name="openhands")
    trajectory.exchanges.append(
        LLMExchange(
            request=LLMRequest(body={"model": "bedrock/us.anthropic.claude-opus-4-8"}),
            response=LLMResponse(
                body={
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 2,
                        "cacheReadInputTokens": 4,
                    }
                }
            ),
        )
    )

    usage = extract_usage_from_trajectory(
        trajectory,
        fallback_model="bedrock/us.anthropic.claude-opus-4-8",
    )
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 14
    assert usage["n_output_tokens"] == 2
