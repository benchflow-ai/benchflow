"""Tests for provider runtime startup helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from benchflow.providers import runtime as provider_runtime_mod
from benchflow.providers.runtime import (
    ProviderRuntime,
    _bedrock_frontend_model,
    ensure_bedrock_proxy_runtime,
    ensure_usage_proxy_runtime,
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
        monkeypatch.setattr(
            provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
        )

        class FakeServer:
            def __init__(
                self,
                host,
                port,
                backend_model=None,
                frontend_model=None,
                runtime_env=None,
            ):
                self.host = host
                self.port = 32123
                self.backend_model = backend_model
                self.frontend_model = frontend_model
                self.runtime_env = runtime_env
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
        assert runtime.base_url == "http://host.docker.internal:32123"
        assert runtime.backend_model == "openai.gpt-oss-20b-1:0"
        assert (
            updated["BENCHFLOW_PROVIDER_BASE_URL"]
            == "http://host.docker.internal:32123"
        )
        assert updated["OPENAI_BASE_URL"] == "http://host.docker.internal:32123"
        assert runtime.server.runtime_env["AWS_BEARER_TOKEN_BEDROCK"] == "bedrock-token"
        assert runtime.server.started is True

    @pytest.mark.asyncio
    async def test_starts_proxy_and_rewrites_claude_env(self, monkeypatch):
        monkeypatch.setattr(
            provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
        )

        class FakeServer:
            def __init__(
                self,
                host,
                port,
                backend_model=None,
                frontend_model=None,
                runtime_env=None,
            ):
                self.port = 32123
                self.backend_model = backend_model
                self.frontend_model = frontend_model
                self.runtime_env = runtime_env

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
        assert (
            updated["BENCHFLOW_PROVIDER_BASE_URL"]
            == "http://host.docker.internal:32123"
        )
        assert "ANTHROPIC_BASE_URL" not in updated
        assert "ANTHROPIC_AUTH_TOKEN" not in updated
        assert updated["CLAUDE_CODE_USE_BEDROCK"] == "1"
        assert updated["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] == "1"
        assert (
            updated["ANTHROPIC_BEDROCK_BASE_URL"] == "http://host.docker.internal:32123"
        )
        assert updated["ANTHROPIC_MODEL"] == "anthropic.claude-haiku-4-5-20251001-v1:0"
        assert runtime.server.runtime_env["AWS_REGION"] == "us-east-1"
        assert "BENCHFLOW_CLAUDE_FRONTEND_MODEL" not in updated

    @pytest.mark.asyncio
    async def test_claude_bedrock_alias_preserves_backend_model(self, monkeypatch):
        class FakeServer:
            def __init__(
                self,
                host,
                port,
                backend_model=None,
                frontend_model=None,
                runtime_env=None,
            ):
                self.port = 32123
                self.backend_model = backend_model
                self.frontend_model = frontend_model
                self.runtime_env = runtime_env

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
        runtime = ProviderRuntime(
            kind="aws-bedrock",
            agent_base_url="http://host.docker.internal:8099",
        )
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
            agent_base_url="http://host.docker.internal:8099",
            server=server,
        )
        await stop_provider_runtime(runtime)
        server.stop.assert_awaited_once()


class TestBedrockProxyRemoteSandbox:
    """Guards the fix from PR #329: the Bedrock proxy is load-bearing; on a
    remote sandbox where the host proxy is unreachable the run must fail fast
    rather than inject an unreachable 127.0.0.1 base URL (the Daytona
    telemetry-proxy twin bug)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("environment", ["daytona", "modal"])
    async def test_bedrock_on_remote_sandbox_fails_fast(self, environment):
        with pytest.raises(RuntimeError, match="not supported on the"):
            await ensure_bedrock_proxy_runtime(
                agent="claude-agent-acp",
                agent_env={
                    "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                    "AWS_REGION": "us-east-1",
                },
                model="aws-bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
                runtime=None,
                environment=environment,
            )

    @pytest.mark.asyncio
    async def test_remote_sandbox_error_is_actionable(self):
        with pytest.raises(RuntimeError) as exc:
            await ensure_bedrock_proxy_runtime(
                agent="codex-acp",
                agent_env={"AWS_REGION": "us-east-1"},
                model="aws-bedrock/openai.gpt-oss-20b-1:0",
                runtime=None,
                environment="daytona",
            )
        message = str(exc.value)
        assert "daytona" in message
        assert "--sandbox docker" in message

    @pytest.mark.asyncio
    async def test_non_bedrock_model_on_remote_sandbox_is_noop(self):
        # A non-Bedrock model never needs the proxy, so a remote sandbox is fine.
        agent_env = {"ANTHROPIC_API_KEY": "sk-test"}
        updated, runtime = await ensure_bedrock_proxy_runtime(
            agent="claude-agent-acp",
            agent_env=agent_env,
            model="claude-haiku-4-5",
            runtime=None,
            environment="daytona",
        )
        assert updated == agent_env
        assert runtime is None

    @pytest.mark.asyncio
    async def test_stale_runtime_stopped_when_environment_unreachable(self):
        server = AsyncMock()
        runtime = ProviderRuntime(
            kind="aws-bedrock",
            agent_base_url="http://host.docker.internal:8099",
            server=server,
        )
        with pytest.raises(RuntimeError, match="not supported on the"):
            await ensure_bedrock_proxy_runtime(
                agent="claude-agent-acp",
                agent_env={"AWS_REGION": "us-east-1"},
                model="aws-bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
                runtime=runtime,
                environment="modal",
            )
        server.stop.assert_awaited_once()

    def test_bedrock_proxy_command_rejects_unreachable_environment(self):
        from benchflow.providers.runtime import _bedrock_proxy_command

        with pytest.raises(AssertionError):
            _bedrock_proxy_command(environment="daytona")


class TestUsageProxyRuntime:
    @pytest.mark.asyncio
    async def test_gemini_proxy_uses_cli_base_url_env(self, monkeypatch):
        """Guards the follow-up to PR #483: Gemini CLI 0.42.0 reads
        GOOGLE_GEMINI_BASE_URL, not GEMINI_API_BASE_URL (#375)."""

        monkeypatch.setattr(
            provider_runtime_mod, "_docker_host_address", lambda: "host.docker.internal"
        )

        class FakeProxy:
            def __init__(
                self,
                *,
                target,
                session_id,
                agent_name,
                host,
                port,
                prompt_cache_retention=None,
            ):
                self.target = target
                self.session_id = session_id
                self.agent_name = agent_name
                self.host = host
                self.port = 43210
                self.prompt_cache_retention = prompt_cache_retention

            async def start(self):
                return None

            async def stop(self):
                return None

        monkeypatch.setattr(provider_runtime_mod, "TrajectoryProxy", FakeProxy)

        updated, runtime = await ensure_usage_proxy_runtime(
            agent="gemini",
            agent_env={"GEMINI_API_KEY": "test-key"},
            model="gemini-2.5-flash",
            runtime=None,
            environment="docker",
        )

        assert runtime is not None
        assert runtime.server.target == "https://generativelanguage.googleapis.com"
        assert (
            updated["BENCHFLOW_PROVIDER_BASE_URL"]
            == "http://host.docker.internal:43210"
        )
        assert updated["GOOGLE_GEMINI_BASE_URL"] == "http://host.docker.internal:43210"
        assert "GEMINI_API_BASE_URL" not in updated
