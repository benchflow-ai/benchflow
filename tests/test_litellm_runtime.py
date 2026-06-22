from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.agents.codex_config import CODEX_DEFAULT_AUTH_REQUEST_ENV
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_bedrock_preflight import BedrockPatchPreflightError
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
            CODEX_DEFAULT_AUTH_REQUEST_ENV: json.dumps(
                {"methodId": "api-key", "_meta": {"api-key": {"apiKey": "old"}}}
            ),
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
    auth_request = json.loads(updated[CODEX_DEFAULT_AUTH_REQUEST_ENV])
    assert auth_request == {
        "methodId": "gateway",
        "_meta": {
            "gateway": {
                "baseUrl": "http://host.docker.internal:32123/v1",
                "providerName": "BenchFlow LiteLLM",
                "headers": {"Authorization": f"Bearer {provider_runtime.master_key}"},
            }
        },
    }


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
async def test_openhands_registered_provider_can_route_via_explicit_proxy(monkeypatch):
    """Guards PR #780: OpenHands keeps BenchFlow tracking over explicit proxy env."""
    starts = []

    async def fake_start(**kwargs):
        starts.append(kwargs)
        return FakeLiteLLMServer("http://172.17.0.1:45678", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="openhands",
        agent_env={
            "BENCHFLOW_PROVIDER_BASE_URL": "https://llm-proxy.example.test/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "sk-proxy",
        },
        model="deepseek/deepseek-v4-flash",
        runtime=None,
        environment="docker",
        session_id="run-1",
        usage_tracking="required",
    )

    assert provider_runtime is not None
    route = starts[0]["route"]
    assert route.litellm_params["api_base"] == "https://llm-proxy.example.test/v1"
    assert route.litellm_params["api_key"] == ("os.environ/BENCHFLOW_PROVIDER_API_KEY")
    assert updated["LLM_BASE_URL"] == "http://172.17.0.1:45678/v1"
    assert updated["LLM_API_KEY"] == provider_runtime.master_key
    assert updated["LLM_MODEL"] == "openai/benchflow-deepseek-deepseek-v4-flash"
    assert updated["BENCHFLOW_PROVIDER_MODEL"] == (
        "benchflow-deepseek-deepseek-v4-flash"
    )


@pytest.mark.asyncio
async def test_pi_acp_proxy_preserves_provider_model_metadata(monkeypatch):
    """Guards PR #803: Pi metadata follows the LiteLLM alias in proxy mode."""

    async def fake_start(**kwargs):
        return FakeLiteLLMServer("http://172.17.0.1:45678", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    provider_models = [
        {
            "id": "Qwen/Qwen3-4B",
            "name": "Qwen/Qwen3-4B",
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 16384,
            "maxTokens": 1024,
        }
    ]

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="pi-acp",
        agent_env={
            "BENCHFLOW_PROVIDER_BASE_URL": "http://172.17.0.1:8000/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "dummy",
            "BENCHFLOW_PROVIDER_MODELS": json.dumps(provider_models),
        },
        model="vllm/Qwen/Qwen3-4B",
        runtime=None,
        environment="docker",
        session_id="run-1",
        usage_tracking="required",
    )

    assert provider_runtime is not None
    assert updated["BENCHFLOW_PROVIDER_MODEL"] == "benchflow-vllm-Qwen-Qwen3-4B"
    models = json.loads(updated["BENCHFLOW_PROVIDER_MODELS"])
    alias = next(m for m in models if m["id"] == "benchflow-vllm-Qwen-Qwen3-4B")
    assert alias["name"] == "benchflow-vllm-Qwen-Qwen3-4B"
    assert alias["maxTokens"] == 1024
    assert alias["contextWindow"] == 16384


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
async def test_required_usage_fails_when_litellm_lacks_provider_key(monkeypatch):
    monkeypatch.setattr(runtime_mod, "uses_native_subscription_auth", lambda *_: False)

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
async def test_required_usage_skips_litellm_for_codex_subscription(monkeypatch):
    """Guards PR #613 follow-up: Codex subscription usage comes from ACP."""

    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start for Codex subscription auth")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)
    env = {"CODEX_AUTH_JSON": '{"tokens": {"access_token": "access-token"}}'}

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        usage_tracking="required",
    )

    assert updated == env
    assert provider_runtime is None


@pytest.mark.asyncio
async def test_required_usage_skips_litellm_for_claude_subscription(monkeypatch):
    """Guards PR #613 follow-up: Claude Code subscription usage comes from ACP."""

    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start for Claude subscription auth")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)
    env = {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env=env,
        model="claude-sonnet-4-6",
        runtime=None,
        environment="docker",
        usage_tracking="required",
    )

    assert updated == env
    assert provider_runtime is None


@pytest.mark.asyncio
async def test_usage_tracking_off_still_routes_through_proxy(monkeypatch):
    """off no longer disables the proxy: routable traffic is still captured and
    the raw provider key never reaches the agent."""

    async def fake_start(**kwargs):
        return FakeLiteLLMServer("http://127.0.0.1:4000", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    env = {"OPENAI_API_KEY": "sk-openai"}

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env=env,
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        usage_tracking="off",
    )

    assert provider_runtime is not None
    # Agent points at the proxy, holding the master key — not the raw provider key.
    assert updated["OPENAI_BASE_URL"] == "http://127.0.0.1:4000/v1"
    assert updated["OPENAI_API_KEY"] == provider_runtime.master_key
    assert "sk-openai" not in updated.values()


@pytest.mark.asyncio
async def test_usage_tracking_off_replaces_stale_litellm_runtime(monkeypatch):
    """off with a stale runtime stops the old proxy and starts a fresh one
    (it no longer tears the proxy down and goes direct)."""

    old_server = FakeLiteLLMServer("http://127.0.0.1:4000", route=None)
    existing = ProviderRuntime(
        kind="litellm",
        agent_base_url=old_server.base_url,
        server=old_server,
        config_key="old",
    )

    async def fake_start(**kwargs):
        return FakeLiteLLMServer("http://127.0.0.1:4001", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-openai"},
        model="openai/gpt-4.1-mini",
        runtime=existing,
        environment="docker",
        usage_tracking="off",
    )

    assert old_server.stopped is True
    assert provider_runtime is not None
    assert updated["OPENAI_BASE_URL"] == "http://127.0.0.1:4001/v1"


@pytest.mark.asyncio
async def test_openhands_azure_never_bypasses_proxy(monkeypatch):
    """The bug this PR fixes: a caller-supplied Azure LLM_BASE_URL must never let
    OpenHands reach Azure directly. The proxy is forced on even under off, the
    raw Azure key and endpoints are stripped, and LLM_BASE_URL points at the proxy."""

    async def fake_start(**kwargs):
        return FakeLiteLLMServer("http://127.0.0.1:45678", kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="openhands",
        agent_env={
            "AZURE_API_KEY": "azure-secret",
            "AZURE_API_ENDPOINT": "https://my-resource.openai.azure.com",
            "AZURE_API_VERSION": "preview",
            "LLM_BASE_URL": "https://my-resource.openai.azure.com/openai/v1",
            "OPENAI_BASE_URL": "https://my-resource.openai.azure.com/openai/v1",
        },
        model="azure-foundry-openai/gpt-5.5",
        runtime=None,
        environment="daytona",
        session_id="run-1",
        usage_tracking="off",
        sandbox=SimpleNamespace(),
    )

    assert provider_runtime is not None
    assert updated["LLM_BASE_URL"] == "http://127.0.0.1:45678/v1"
    assert updated["LLM_API_KEY"] == provider_runtime.master_key
    assert "AZURE_API_KEY" not in updated
    assert "OPENAI_BASE_URL" not in updated
    assert not any("azure.com" in str(v) for v in updated.values())


def test_proxy_isolation_guard_blocks_leaked_secret():
    """The fail-closed guard refuses to run if a raw provider key would survive."""

    with pytest.raises(RuntimeError, match="isolation breached"):
        runtime_mod._assert_proxy_isolated(
            "openhands", {"OPENAI_API_KEY": "sk-leak"}, master_key="sk-benchflow-x"
        )

    # Proxy master key in a provider slot + a non-secret var are fine.
    runtime_mod._assert_proxy_isolated(
        "codex-acp",
        {"OPENAI_API_KEY": "sk-benchflow-x", "AZURE_API_VERSION": "preview"},
        master_key="sk-benchflow-x",
    )


def test_proxy_docs_disable_env_neutralizes_stray_docs_url(monkeypatch):
    """A stray DOCS_URL (e.g. baked into a sandbox base image) used to crash the
    proxy at startup ('Routed paths must start with /'). The proxy launch env
    must disable Swagger docs so litellm registers no docs route."""
    from litellm.proxy.utils import _get_docs_url

    # Reproduce the crash trigger: an inherited non-"/" DOCS_URL.
    monkeypatch.setenv("DOCS_URL", "stray-without-leading-slash")
    assert _get_docs_url() == "stray-without-leading-slash"  # would crash add_route

    # Applying the proxy launch overrides makes litellm skip the docs route.
    for key, value in runtime_mod._PROXY_DOCS_DISABLE_ENV.items():
        monkeypatch.setenv(key, value)
    assert _get_docs_url() is None


@pytest.mark.asyncio
async def test_auto_usage_fails_closed_when_litellm_lacks_provider_key(monkeypatch):
    """The proxy is mandatory: missing credentials are fatal, never a silent
    fall back to direct provider access — even under auto."""

    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start without provider credentials")

    monkeypatch.setattr(runtime_mod, "uses_native_subscription_auth", lambda *_: False)
    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    with pytest.raises(RuntimeError, match="requires OPENAI_API_KEY"):
        await ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={},
            model="openai/gpt-4.1-mini",
            runtime=None,
            environment="docker",
            usage_tracking="auto",
        )


@pytest.mark.asyncio
async def test_auto_usage_fails_closed_when_route_resolution_fails(monkeypatch):
    """Route resolution failure is fatal: BenchFlow never bypasses the proxy."""

    def fail_route(*_args, **_kwargs):
        raise ValueError("missing AZURE_RESOURCE")

    async def fail_start(**_kwargs):
        raise AssertionError("LiteLLM should not start without a resolved route")

    monkeypatch.setattr(runtime_mod, "resolve_litellm_route", fail_route)
    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    with pytest.raises(RuntimeError, match="cannot resolve"):
        await ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={"AZURE_API_KEY": "azure-key"},
            model="azure-foundry-openai/gpt-4.1-mini",
            runtime=None,
            environment="docker",
            usage_tracking="auto",
        )


@pytest.mark.asyncio
async def test_auto_usage_fails_closed_when_litellm_start_fails(monkeypatch):
    """Proxy startup failure is fatal under auto too (no direct-provider fallback)."""

    async def fail_start(**_kwargs):
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    with pytest.raises(RuntimeError, match="failed to start"):
        await ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={"OPENAI_API_KEY": "sk-openai"},
            model="openai/gpt-4.1-mini",
            runtime=None,
            environment="docker",
            usage_tracking="auto",
        )


@pytest.mark.asyncio
async def test_auto_usage_does_not_fallback_on_bedrock_patch_preflight(monkeypatch):
    """Guards PR #668's fail-closed fix for issue #602: an inactive Bedrock patch
    must abort even when usage_tracking=auto would normally skip LiteLLM."""

    async def fail_start(**_kwargs):
        raise BedrockPatchPreflightError("patch inactive")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    with pytest.raises(BedrockPatchPreflightError, match="patch inactive"):
        await ensure_litellm_runtime(
            agent="openhands",
            agent_env={
                "AWS_BEARER_TOKEN_BEDROCK": "tok",
                "AWS_REGION": "us-west-2",
            },
            model="aws-bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0",
            runtime=None,
            environment="docker",
            usage_tracking="auto",
        )


@pytest.mark.asyncio
async def test_auto_usage_requires_sandbox_handle_for_sandbox_local_litellm():
    """Guards the follow-up to PR #620: auto must not hide wiring bugs."""

    with pytest.raises(RuntimeError, match="sandbox-local LiteLLM"):
        await ensure_litellm_runtime(
            agent="openhands",
            agent_env={"OPENAI_API_KEY": "sk-openai"},
            model="openai/gpt-4.1-mini",
            runtime=None,
            environment="daytona",
            usage_tracking="auto",
            sandbox=None,
        )


@pytest.mark.asyncio
async def test_auto_usage_checks_sandbox_handle_before_route_fallback(monkeypatch):
    """Guards the follow-up to PR #620: sandbox wiring wins over route fallback."""

    route_calls = []

    def fail_route(*_args, **_kwargs):
        route_calls.append(True)
        raise ValueError("missing AZURE_RESOURCE")

    monkeypatch.setattr(runtime_mod, "resolve_litellm_route", fail_route)

    with pytest.raises(RuntimeError, match="sandbox-local LiteLLM"):
        await ensure_litellm_runtime(
            agent="openhands",
            agent_env={},
            model="azure-foundry-openai/gpt-4.1-mini",
            runtime=None,
            environment="daytona",
            usage_tracking="auto",
            sandbox=None,
        )

    assert route_calls == []


@pytest.mark.asyncio
async def test_required_usage_propagates_litellm_start_failure(monkeypatch):
    """Guards the follow-up to PR #613: required must still fail closed."""

    async def fail_start(**_kwargs):
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fail_start)

    with pytest.raises(RuntimeError, match="failed to start"):
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
