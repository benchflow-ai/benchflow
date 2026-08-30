from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.providers.runtime import ensure_litellm_runtime


class _SubscriptionIsolationSandbox:
    def __init__(self, *, return_code: int = 0) -> None:
        self.return_code = return_code
        self.commands: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        assert kwargs["user"] == "root"
        return SimpleNamespace(return_code=self.return_code, stdout="", stderr="")


@pytest.mark.asyncio
@pytest.mark.parametrize(("return_code", "trusted"), [(0, True), (1, False)])
async def test_claude_api_proxy_isolates_preexisting_subscription_credential(
    monkeypatch, return_code: int, trusted: bool
) -> None:
    """Guards PR #1057 against a reused Claude OAuth file bypassing capture."""

    async def fake_start(**_kwargs):
        return SimpleNamespace(base_url="http://127.0.0.1:4000")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    sandbox = _SubscriptionIsolationSandbox(return_code=return_code)

    _, provider_runtime = await ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_API_KEY": "sk-ant-provider"},
        model="claude-sonnet-4-6",
        runtime=None,
        environment="docker",
        session_id="run-claude-api-with-stale-oauth",
        sandbox=sandbox,
        sandbox_user="agent",
    )

    assert provider_runtime is not None
    assert provider_runtime.capture_trusted is trusted
    assert len(sandbox.commands) == 1
    assert "/home/agent/.claude/.credentials.json" in sandbox.commands[0]
    assert "rm -f" in sandbox.commands[0]
    assert "! test -e" in sandbox.commands[0]
    assert "! test -L" in sandbox.commands[0]


@pytest.mark.asyncio
async def test_vertex_adc_provider_capture_remains_audit_only_on_host(monkeypatch):
    """Guards PR #1057 against trusting agent-accessible Vertex credentials."""

    async def fake_start(**_kwargs):
        return SimpleNamespace(base_url="http://127.0.0.1:4000")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"type":"authorized_user"}',
            "GOOGLE_APPLICATION_CREDENTIALS": (
                "/home/agent/.config/gcloud/application_default_credentials.json"
            ),
            "GOOGLE_CLOUD_PROJECT": "project",
            "GOOGLE_CLOUD_LOCATION": "global",
        },
        model="google-vertex/gemini-2.5-flash",
        runtime=None,
        environment="docker",
        session_id="run-vertex-adc",
    )

    assert provider_runtime is not None
    assert provider_runtime.capture_trusted is False
    assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in updated
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in updated


@pytest.mark.asyncio
async def test_vertex_adc_stays_audit_only_with_verified_sandbox_artifacts(monkeypatch):
    """Guards PR #1057 against equating file custody with ADC isolation."""

    async def fake_start(**_kwargs):
        return SimpleNamespace(
            base_url="http://127.0.0.1:4000",
            runtime_dir="/tmp/benchflow-litellm/test-runtime",
        )

    async def fake_artifact_custody(**_kwargs):
        return True

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_start)
    monkeypatch.setattr(
        runtime_mod,
        "_provider_capture_has_verified_custody",
        fake_artifact_custody,
    )

    _, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"type":"authorized_user"}',
            "GOOGLE_CLOUD_PROJECT": "project",
            "GOOGLE_CLOUD_LOCATION": "global",
        },
        model="google-vertex/gemini-2.5-flash",
        runtime=None,
        environment="daytona",
        session_id="run-vertex-adc-sandbox",
        sandbox=SimpleNamespace(),
        sandbox_user="agent",
    )

    assert provider_runtime is not None
    assert provider_runtime.capture_trusted is False


def test_proxy_strips_every_supported_alternate_credential_alias():
    """Guards PR #1057 against alternate credentials bypassing the proxy."""

    master_key = "sk-benchflow-master"
    route = resolve_litellm_route(
        "openai/gpt-5.6-luna",
        {"OPENAI_API_KEY": "sk-provider"},
    )
    raw_credentials = {
        "OPENAI_API_KEY": "sk-provider",
        "CODEX_API_KEY": "sk-codex",
        "CODEX_ACCESS_TOKEN": "codex-access",
        "CODEX_AUTH_JSON": '{"tokens":{"access_token":"codex-json"}}',
        "CLAUDE_CODE_OAUTH_TOKEN": "claude-code-oauth",
        "CLAUDE_OAUTH_TOKEN": "claude-oauth",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-auth",
        "AWS_ACCESS_KEY_ID": "aws-access",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AWS_SESSION_TOKEN": "aws-session",
        "CUSTOM_API_KEY": "custom-provider-key",
    }

    agent_env = {
        **raw_credentials,
        "DEFAULT_AUTH_REQUEST": json.dumps(
            {
                "methodId": "api-key",
                "_meta": {"api-key": {"apiKey": raw_credentials["CODEX_API_KEY"]}},
            }
        ),
    }
    updated = runtime_mod._wire_litellm_agent_env(
        agent="codex-acp",
        agent_env=agent_env,
        route=route,
        base_url="http://127.0.0.1:4000",
        master_key=master_key,
    )

    for name, value in raw_credentials.items():
        assert updated.get(name) != value, name
        assert value not in json.dumps(updated), name
    assert updated["OPENAI_API_KEY"] == master_key
    assert updated["BENCHFLOW_PROVIDER_API_KEY"] == master_key
    runtime_mod._assert_proxy_isolated(
        "codex-acp",
        updated,
        master_key=master_key,
    )


def test_proxy_rebinds_custom_codex_provider_to_master_key():
    """Guards PR #1057 against retaining custom Codex provider credentials."""

    master_key = "sk-benchflow-master"
    literal_key = "literal-provider-credential"
    route = resolve_litellm_route(
        "openai/gpt-5.6-luna",
        {"OPENAI_API_KEY": "sk-provider"},
    )
    updated = runtime_mod._wire_litellm_agent_env(
        agent="codex-acp",
        agent_env={
            "CODEX_API_KEY": "sk-codex-provider",
            "CODEX_CONFIG": json.dumps(
                {
                    "top_level_secret": literal_key,
                    "model_provider": "custom",
                    "model_providers": {
                        "custom": {
                            "env_key": "CODEX_API_KEY",
                            "wire_api": "responses",
                            "http_headers": {"Authorization": literal_key},
                        },
                        "unused": {
                            "env_key": "UNUSED_API_KEY",
                            "http_headers": {"Authorization": literal_key},
                        },
                    },
                }
            ),
        },
        route=route,
        base_url="http://127.0.0.1:4000",
        master_key=master_key,
    )

    config = json.loads(updated["CODEX_CONFIG"])
    assert config == {
        "model_providers": {
            "benchflow-litellm": {
                "name": "litellm",
                "base_url": "http://127.0.0.1:4000/v1",
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
                "supports_websockets": False,
            }
        },
        "model_provider": "benchflow-litellm",
        "model": route.model_alias,
    }
    assert updated["OPENAI_API_KEY"] == master_key
    assert "CODEX_API_KEY" not in updated
    assert literal_key not in json.dumps(updated)
    runtime_mod._assert_proxy_isolated(
        "codex-acp",
        updated,
        master_key=master_key,
    )


def test_proxy_isolation_guard_detects_unregistered_api_key_alias():
    """Guards PR #1057 against custom API-key aliases escaping the guard."""

    with pytest.raises(RuntimeError, match="CUSTOM_API_KEY"):
        runtime_mod._assert_proxy_isolated(
            "custom-agent",
            {"CUSTOM_API_KEY": "raw-provider-key"},
            master_key="sk-benchflow-master",
        )
