"""Tests for provider runtime startup helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from benchflow import _provider_runtime as provider_runtime_mod
from benchflow._provider_runtime import (
    ProviderRuntime,
    _bedrock_frontend_model,
    ensure_bedrock_proxy_runtime,
    needs_provider_runtime,
    stop_provider_runtime,
)


class TestProviderRuntimeSelection:
    def test_needs_provider_runtime_for_bedrock(self):
        assert needs_provider_runtime("aws-bedrock/openai.gpt-oss-20b-1:0") is True

    def test_needs_provider_runtime_for_non_bedrock(self):
        assert needs_provider_runtime("claude-sonnet-4-6") is False
        assert needs_provider_runtime(None) is False

    def test_claude_frontend_model_normalizes_supported_version(self):
        assert (
            _bedrock_frontend_model(
                agent="claude-agent-acp",
                backend_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            )
            == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )


class TestBedrockProxyRuntime:
    @pytest.mark.asyncio
    async def test_non_bedrock_model_is_noop(self):
        agent_env = {"OPENAI_API_KEY": "sk-test"}
        updated, runtime = await ensure_bedrock_proxy_runtime(
            agent="codex-acp",
            agent_env=agent_env,
            model="gpt-4.1-mini",
            runtime=None,
            environment="docker",
        )
        assert updated == agent_env
        assert runtime is None

    @pytest.mark.asyncio
    async def test_starts_proxy_and_rewrites_codex_env(self, monkeypatch):
        class FakeServer:
            def __init__(self, host, port, backend_model=None, frontend_model=None):
                self.host = host
                self.port = 32123
                self.backend_model = backend_model
                self.frontend_model = frontend_model
                self.started = False
                self.stopped = False

            async def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

        monkeypatch.setattr(provider_runtime_mod, "BedrockProxyServer", FakeServer)
        updated, runtime = await ensure_bedrock_proxy_runtime(
            agent="codex-acp",
            agent_env={
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_REGION": "us-east-1",
                "OPENAI_API_KEY": "bedrock-proxy",
            },
            model="aws-bedrock/openai.gpt-oss-20b-1:0",
            runtime=None,
            environment="docker",
        )

        assert runtime is not None
        assert runtime.host == "host.docker.internal"
        assert runtime.port == 32123
        assert runtime.backend_model == "openai.gpt-oss-20b-1:0"
        assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == "http://host.docker.internal:32123"
        assert updated["OPENAI_BASE_URL"] == "http://host.docker.internal:32123"
        assert runtime.server.started is True

    @pytest.mark.asyncio
    async def test_starts_proxy_and_rewrites_claude_env(self, monkeypatch):
        class FakeServer:
            def __init__(self, host, port, backend_model=None, frontend_model=None):
                self.port = 32123
                self.backend_model = backend_model
                self.frontend_model = frontend_model

            async def start(self):
                return None

            async def stop(self):
                return None

        monkeypatch.setattr(provider_runtime_mod, "BedrockProxyServer", FakeServer)
        updated, runtime = await ensure_bedrock_proxy_runtime(
            agent="claude-agent-acp",
            agent_env={
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_REGION": "us-east-1",
                "ANTHROPIC_AUTH_TOKEN": "bedrock-proxy",
            },
            model="aws-bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            runtime=None,
            environment="docker",
        )

        assert runtime is not None
        assert updated["BENCHFLOW_PROVIDER_BASE_URL"] == "http://host.docker.internal:32123"
        assert "ANTHROPIC_BASE_URL" not in updated
        assert "ANTHROPIC_AUTH_TOKEN" not in updated
        assert updated["CLAUDE_CODE_USE_BEDROCK"] == "1"
        assert updated["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] == "1"
        assert updated["ANTHROPIC_BEDROCK_BASE_URL"] == "http://host.docker.internal:32123"
        assert updated["ANTHROPIC_MODEL"] == "anthropic.claude-haiku-4-5-20251001-v1:0"
        assert "BENCHFLOW_CLAUDE_FRONTEND_MODEL" not in updated

    @pytest.mark.asyncio
    async def test_claude_bedrock_alias_preserves_backend_model(self, monkeypatch):
        class FakeServer:
            def __init__(self, host, port, backend_model=None, frontend_model=None):
                self.port = 32123
                self.backend_model = backend_model
                self.frontend_model = frontend_model

            async def start(self):
                return None

            async def stop(self):
                return None

        monkeypatch.setattr(provider_runtime_mod, "BedrockProxyServer", FakeServer)
        updated, runtime = await ensure_bedrock_proxy_runtime(
            agent="claude-agent-acp",
            agent_env={
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_REGION": "us-east-1",
            },
            model="aws-bedrock/us.anthropic.claude-sonnet-4-5",
            runtime=None,
            environment="docker",
        )

        assert runtime is not None
        assert runtime.backend_model == "us.anthropic.claude-sonnet-4-5"
        assert updated["ANTHROPIC_MODEL"] == "us.anthropic.claude-sonnet-4-5"
        assert "BENCHFLOW_CLAUDE_FRONTEND_MODEL" not in updated


    @pytest.mark.asyncio
    async def test_reuses_existing_runtime(self):
        runtime = ProviderRuntime(kind="aws-bedrock", host="host.docker.internal", port=8099)
        updated, returned = await ensure_bedrock_proxy_runtime(
            agent="codex-acp",
            agent_env={"OPENAI_API_KEY": "bedrock-proxy"},
            model="aws-bedrock/openai.gpt-oss-20b-1:0",
            runtime=runtime,
            environment="docker",
        )
        assert returned is runtime
        assert updated["OPENAI_BASE_URL"] == "http://host.docker.internal:8099"

    @pytest.mark.asyncio
    async def test_stop_provider_runtime_stops_server(self):
        server = AsyncMock()
        runtime = ProviderRuntime(
            kind="aws-bedrock",
            host="host.docker.internal",
            port=8099,
            server=server,
        )
        await stop_provider_runtime(runtime)
        server.stop.assert_awaited_once()
