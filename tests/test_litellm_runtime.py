from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import LITELLM_MODEL_ALIAS_ENV
from benchflow.providers.runtime import (
    ProviderRuntime,
    ensure_litellm_runtime,
    stop_provider_runtime,
)


class FakeLiteLLMServer:
    def __init__(self, base_url: str, route):
        self._base_url = base_url
        self.route = route
        self.stopped = False
        self.trajectory = None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def is_running(self) -> bool:
        return not self.stopped

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_host_litellm_rewrites_codex_env(monkeypatch):
    async def fake_start(**kwargs):
        return FakeLiteLLMServer("http://host.docker.internal:32123", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
        },
        model="aws-bedrock/us.anthropic.claude-opus-4-8",
        runtime=None,
        environment="docker",
        session_id="run-1",
        usage_tracking="required",
    )

    assert provider_runtime is not None
    assert provider_runtime.kind == "litellm"
    assert provider_runtime.backend_model == "bedrock/us.anthropic.claude-opus-4-8"
    assert updated["OPENAI_BASE_URL"] == "http://host.docker.internal:32123/v1"
    assert updated["OPENAI_API_KEY"] == provider_runtime.master_key
    assert updated[LITELLM_MODEL_ALIAS_ENV] == (
        "benchflow-aws-bedrock-us.anthropic.claude-opus-4-8"
    )
    assert (
        '"model":"benchflow-aws-bedrock-us.anthropic.claude-opus-4-8"'
        in updated["CODEX_CONFIG"]
    )


@pytest.mark.asyncio
async def test_claude_agent_uses_anthropic_compatible_litellm_endpoint(monkeypatch):
    async def fake_start(**kwargs):
        return FakeLiteLLMServer("http://127.0.0.1:4000", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    updated, _runtime = await ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_API_KEY": "sk-ant"},
        model="claude-sonnet-4-6",
        runtime=None,
        environment="local",
        session_id="run-1",
    )

    assert updated["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert updated["ANTHROPIC_AUTH_TOKEN"].startswith("sk-benchflow-")
    assert updated["ANTHROPIC_MODEL"] == "benchflow-claude-sonnet-4-6"
    assert "CLAUDE_CODE_USE_BEDROCK" not in updated


@pytest.mark.asyncio
async def test_daytona_uses_sandbox_local_litellm(monkeypatch):
    starts = []

    async def fake_sandbox_start(**kwargs):
        starts.append(kwargs)
        return FakeLiteLLMServer("http://127.0.0.1:45678", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_sandbox_start)
    sandbox = SimpleNamespace()

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="openhands",
        agent_env={
            "AWS_BEARER_TOKEN_BEDROCK": "token",
            "AWS_REGION": "us-west-2",
        },
        model="aws-bedrock/us.anthropic.claude-opus-4-8",
        runtime=None,
        environment="daytona",
        session_id="run-1",
        sandbox=sandbox,
    )

    assert starts[0]["sandbox"] is sandbox
    assert provider_runtime.base_url == "http://127.0.0.1:45678"
    assert updated["LLM_BASE_URL"] == "http://127.0.0.1:45678/v1"
    assert updated["LLM_MODEL"].startswith("openai/benchflow-aws-bedrock")


@pytest.mark.asyncio
async def test_runtime_reuse_and_stop(monkeypatch):
    created = []

    async def fake_start(**kwargs):
        server = FakeLiteLLMServer("http://127.0.0.1:4000", kwargs["route"])
        created.append(server)
        return server

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    env = {"OPENAI_API_KEY": "sk-openai"}
    _updated, first = await ensure_litellm_runtime(
        agent="opencode",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="local",
        session_id="run-1",
    )
    _updated, second = await ensure_litellm_runtime(
        agent="opencode",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=first,
        environment="local",
        session_id="run-1",
    )

    assert second is first
    assert len(created) == 1
    await stop_provider_runtime(second)
    assert created[0].stopped is True


@pytest.mark.asyncio
async def test_required_usage_fails_when_litellm_lacks_provider_key():
    with pytest.raises(RuntimeError, match="requires OPENAI_API_KEY"):
        await ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={},
            model="openai/gpt-4.1-mini",
            runtime=None,
            environment="docker",
            usage_tracking="required",
        )


@pytest.mark.asyncio
async def test_usage_tracking_off_leaves_provider_env_untouched(monkeypatch):
    """Guards the follow-up to PR #613: off must not proxy provider traffic."""

    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start when usage tracking is off")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)
    env = {"OPENAI_API_KEY": "sk-openai"}

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        usage_tracking="off",
    )

    assert updated == env
    assert provider_runtime is None


@pytest.mark.asyncio
async def test_usage_tracking_off_stops_existing_litellm_runtime():
    """Guards the follow-up to PR #613: off must clear stale LiteLLM routing."""

    server = FakeLiteLLMServer("http://127.0.0.1:4000", route=None)
    existing = ProviderRuntime(
        kind="litellm",
        agent_base_url=server.base_url,
        server=server,
        config_key="old",
    )

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-openai"},
        model="openai/gpt-4.1-mini",
        runtime=existing,
        environment="docker",
        usage_tracking="off",
    )

    assert updated == {"OPENAI_API_KEY": "sk-openai"}
    assert provider_runtime is None
    assert server.stopped is True


@pytest.mark.asyncio
async def test_auto_usage_falls_back_when_litellm_lacks_provider_key(monkeypatch):
    """Guards the follow-up to PR #613: auto should not fail just to track usage."""

    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start without provider credentials")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={},
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        usage_tracking="auto",
    )

    assert updated == {}
    assert provider_runtime is None


@pytest.mark.asyncio
async def test_auto_usage_falls_back_when_litellm_start_fails(monkeypatch):
    """Guards the follow-up to PR #613: auto should survive proxy startup errors."""

    async def fail_start(**_kwargs):
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)
    env = {"OPENAI_API_KEY": "sk-openai"}

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        usage_tracking="auto",
    )

    assert updated == env
    assert provider_runtime is None


@pytest.mark.asyncio
async def test_required_usage_propagates_litellm_start_failure(monkeypatch):
    """Guards the follow-up to PR #613: required must still fail closed."""

    async def fail_start(**_kwargs):
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    with pytest.raises(RuntimeError, match="Token usage tracking is required"):
        await ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={"OPENAI_API_KEY": "sk-openai"},
            model="openai/gpt-4.1-mini",
            runtime=None,
            environment="docker",
            usage_tracking="required",
        )


@pytest.mark.asyncio
async def test_required_usage_fails_for_native_agent_without_litellm_route():
    """Guards the follow-up to PR #613: required cannot silently go unavailable."""

    with pytest.raises(RuntimeError, match="cannot be routed through LiteLLM"):
        await ensure_litellm_runtime(
            agent="gemini",
            agent_env={"GEMINI_API_KEY": "gemini-key"},
            model="gemini-2.5-flash",
            runtime=None,
            environment="docker",
            usage_tracking="required",
        )


@pytest.mark.asyncio
async def test_oracle_does_not_start_litellm(monkeypatch):
    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start for oracle")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)
    env = {"OPENAI_API_KEY": "sk-openai"}

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="oracle",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
    )

    assert updated == env
    assert provider_runtime is None
