"""Tests for Daytona-specific provider usage runtime wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.trajectories.types import Trajectory


@pytest.mark.asyncio
async def test_usage_runtime_reconnect_ignores_own_proxy_url(monkeypatch):
    """Guards PR #587: reconnects must not point a new proxy at the old proxy."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    started = []
    stopped = []

    class FakeSandboxUsageProxy:
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            started.append(self.target)

        async def is_running(self):
            return True

        async def stop(self):
            stopped.append(self.target)

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    env = {
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_API_KEY": "sk-real-key",
    }
    first_env, first_runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env=env,
        model="claude-haiku-4-5-20251001",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )
    second_env, second_runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env=first_env,
        model="claude-haiku-4-5-20251001",
        runtime=first_runtime,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert first_runtime is not None
    assert second_runtime is first_runtime
    assert started == ["https://api.anthropic.com"]
    assert stopped == []
    assert first_runtime.server.target == "https://api.anthropic.com"
    assert second_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:49001"


@pytest.mark.asyncio
async def test_dead_usage_runtime_reconnect_uses_original_upstream(monkeypatch):
    """Guards PR #587: stale-proxy replacement must dial the provider, not itself."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ProviderRuntime, ensure_usage_proxy_runtime

    stopped = []
    started = []

    class DeadServer:
        target = "https://api.anthropic.com"

        async def is_running(self):
            return False

        async def stop(self):
            stopped.append("dead")

    class FakeSandboxUsageProxy:
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            started.append(self.target)

        async def stop(self):
            stopped.append("new")

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    stale_runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url="http://127.0.0.1:49000",
        backend_model="claude-haiku-4-5-20251001",
        server=DeadServer(),
    )

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="claude-agent-acp",
        agent_env={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:49000",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:49000",
            "ANTHROPIC_API_KEY": "sk-real-key",
        },
        model="claude-haiku-4-5-20251001",
        runtime=stale_runtime,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert stopped == ["dead"]
    assert started == ["https://api.anthropic.com"]
    assert runtime is not None
    assert runtime is not stale_runtime
    assert runtime.server.target == "https://api.anthropic.com"
    assert updated["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:49001"


@pytest.mark.asyncio
async def test_codex_provider_config_is_repointed_at_usage_proxy(monkeypatch):
    """Guards PR #587: Codex custom providers must not bypass telemetry proxy."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    class FakeSandboxUsageProxy:
        target = "https://example-resource.openai.azure.com/openai/v1"
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": (
            "https://example-resource.openai.azure.com/openai/v1"
        ),
        "OPENAI_BASE_URL": "https://example-resource.openai.azure.com/openai/v1",
        "BENCHFLOW_PROVIDER_MODEL": "gpt-5.5",
        "OPENAI_API_KEY": "az-test",
        "MODEL_PROVIDER": "benchflow-azure-foundry-openai",
        "CODEX_CONFIG": json.dumps(
            {
                "model_provider": "benchflow-azure-foundry-openai",
                "model": "gpt-5.5",
                "model_providers": {
                    "benchflow-azure-foundry-openai": {
                        "name": "azure-foundry-openai",
                        "base_url": (
                            "https://example-resource.openai.azure.com/openai/v1"
                        ),
                        "env_key": "OPENAI_API_KEY",
                        "wire_api": "responses",
                        "supports_websockets": False,
                    }
                },
            }
        ),
    }

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="codex-acp",
        agent_env=env,
        model="azure-foundry-openai/gpt-5.5",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert runtime is not None
    assert updated["OPENAI_BASE_URL"] == "http://127.0.0.1:49001"
    codex_config = json.loads(updated["CODEX_CONFIG"])
    provider = codex_config["model_providers"]["benchflow-azure-foundry-openai"]
    assert provider["base_url"] == "http://127.0.0.1:49001"


@pytest.mark.asyncio
async def test_codex_native_openai_gets_usage_proxy_provider_config(monkeypatch):
    """Guards PR #587: native Codex OpenAI runs must not bypass telemetry."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import ensure_usage_proxy_runtime

    class FakeSandboxUsageProxy:
        target = "https://api.openai.com/v1"
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    updated, runtime = await ensure_usage_proxy_runtime(
        agent="codex-acp",
        agent_env={
            "OPENAI_API_KEY": "sk-test",
            "BENCHFLOW_PROVIDER_MODEL": "gpt-5.4-mini",
        },
        model="gpt-5.4-mini",
        runtime=None,
        environment="daytona",
        session_id="rollout-1",
        sandbox=object(),
    )

    assert runtime is not None
    assert updated["OPENAI_BASE_URL"] == "http://127.0.0.1:49001"
    assert updated["MODEL_PROVIDER"] == "benchflow-openai"
    codex_config = json.loads(updated["CODEX_CONFIG"])
    assert codex_config["model"] == "gpt-5.4-mini"
    assert codex_config["model_provider"] == "benchflow-openai"
    provider = codex_config["model_providers"]["benchflow-openai"]
    assert provider == {
        "name": "openai",
        "base_url": "http://127.0.0.1:49001",
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
        "supports_websockets": False,
    }


@pytest.mark.asyncio
async def test_daytona_openhands_bedrock_usage_proxy_sets_aws_endpoint(monkeypatch):
    """Guards PR #587: remote Bedrock-direct OpenHands is metered in-sandbox."""
    from benchflow.providers import runtime as provider_runtime_mod
    from benchflow.providers.runtime import (
        ensure_bedrock_proxy_runtime,
        ensure_usage_proxy_runtime,
    )
    from benchflow.usage_tracking import UsageTrackingConfig

    class FakeSandboxUsageProxy:
        base_url = "http://127.0.0.1:49001"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trajectory = Trajectory(
                session_id=kwargs["session_id"], agent_name=kwargs["agent_name"]
            )

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(
        provider_runtime_mod, "SandboxUsageProxy", FakeSandboxUsageProxy
    )

    agent_env = {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
        "AWS_REGION": "us-west-2",
        "LLM_BASE_URL": "",
        "LLM_MODEL": "anthropic/us.anthropic.claude-opus-4-7",
    }
    bedrock_env, bedrock_runtime = await ensure_bedrock_proxy_runtime(
        agent="openhands",
        agent_env=agent_env,
        model="aws-bedrock/us.anthropic.claude-opus-4-7",
        runtime=None,
        environment="daytona",
    )

    assert bedrock_runtime is None
    assert "LLM_BASE_URL" not in bedrock_env
    assert bedrock_env["LLM_MODEL"] == "bedrock/us.anthropic.claude-opus-4-7"

    usage_env, usage_runtime = await ensure_usage_proxy_runtime(
        agent="openhands",
        agent_env=bedrock_env,
        model="aws-bedrock/us.anthropic.claude-opus-4-7",
        runtime=None,
        environment="daytona",
        usage_tracking=UsageTrackingConfig(mode="required"),
        sandbox=object(),
    )

    assert usage_runtime is not None
    assert (
        usage_runtime.server.target
        == "https://bedrock-runtime.us-west-2.amazonaws.com"
    )
    assert usage_env["LLM_BASE_URL"] == usage_runtime.base_url
    assert "BENCHFLOW_PROVIDER_BASE_URL" not in usage_env
    assert usage_env["AWS_REGION_NAME"] == "us-west-2"
    assert usage_env["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"] == usage_runtime.base_url
    assert usage_env["AWS_ENDPOINT_URL_BEDROCK"] == usage_runtime.base_url


def test_extract_usage_accepts_bedrock_converse_usage_shape():
    """Guards PR #587: Bedrock Converse usage fields count as provider usage."""
    from benchflow.providers.runtime import ProviderRuntime, extract_usage
    from benchflow.trajectories.types import LLMExchange, LLMRequest, LLMResponse

    trajectory = Trajectory(session_id="rollout-1", agent_name="openhands")
    trajectory.exchanges.append(
        LLMExchange(
            request=LLMRequest(
                path="/model/us.anthropic.claude-opus-4-7/converse",
                body={"modelId": "us.anthropic.claude-opus-4-7"},
            ),
            response=LLMResponse(
                status_code=200,
                body={
                    "usage": {
                        "cacheReadInputTokens": 100,
                        "cacheWriteInputTokens": 200,
                        "inputTokens": 34,
                        "outputTokens": 13,
                        "totalTokens": 347,
                    }
                },
            ),
            duration_ms=12,
        )
    )

    runtime = ProviderRuntime(
        kind="usage-proxy",
        agent_base_url="http://127.0.0.1:49000",
        backend_model="us.anthropic.claude-opus-4-7",
        server=SimpleNamespace(trajectory=trajectory),
    )
    usage = extract_usage(runtime)

    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 34
    assert usage["n_output_tokens"] == 13
    assert usage["n_cache_read_tokens"] == 100
    assert usage["n_cache_creation_tokens"] == 200
    assert usage["total_tokens"] == 347
