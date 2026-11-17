"""a downstream harness patch: Claude Code on an Azure chat-completions deployment.

The proxy serves Claude Code on a chat-completions model through LiteLLM's
messages bridge (``litellm_messages_bridge_patch``), and ``aws-bedrock`` was
admitted that way. ``azure-foundry-openai`` listed no Anthropic endpoint, so
``-a claude-agent-acp -m azure-foundry-openai/DeepSeek-V4-Pro`` was refused
before the job existed.
"""

from __future__ import annotations

import pytest

import benchflow.providers.litellm_runtime as runtime_mod
from benchflow.agents.env import resolve_agent_env, unroutable_agent_model

MODEL = "azure-foundry-openai/DeepSeek-V4-Pro"
AZURE = {"AZURE_API_KEY": "azure-key", "AZURE_RESOURCE": "res"}


class _Server:
    def __init__(self, route):
        self.route = route
        self.trajectory = None

    @property
    def base_url(self) -> str:
        return "http://host.docker.internal:32123"

    async def is_running(self) -> bool:
        return True

    async def stop(self) -> None:
        return None


@pytest.fixture(autouse=True)
def no_host_env(monkeypatch):
    monkeypatch.setattr(
        "benchflow.agents.env.auto_inherit_env", lambda env, **kwargs: None
    )


def test_claude_code_is_admitted_on_an_azure_chat_deployment():
    assert unroutable_agent_model("claude-agent-acp", MODEL) is None
    assert unroutable_agent_model("claude", MODEL) is None


@pytest.mark.asyncio
async def test_claude_code_reaches_azure_only_through_the_proxy(monkeypatch):
    async def start(**kwargs):
        return _Server(kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", start)
    resolved = resolve_agent_env("claude-agent-acp", MODEL, dict(AZURE))
    assert resolved["BENCHFLOW_PROVIDER_NAME"] == "azure-foundry-openai"
    updated, provider = await runtime_mod.ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env=resolved,
        model=MODEL,
        runtime=None,
        environment="docker",
        usage_tracking="auto",
    )
    assert provider is not None and provider.kind == "litellm"
    # The proxy calls the deployment's chat route; Claude Code calls the proxy.
    assert provider.backend_model == "azure/DeepSeek-V4-Pro"
    assert updated["ANTHROPIC_BASE_URL"] == "http://host.docker.internal:32123"
    assert updated["ANTHROPIC_AUTH_TOKEN"] == provider.master_key
    assert "azure.com" not in updated["ANTHROPIC_BASE_URL"]


@pytest.mark.parametrize(
    ("agent", "protocol"),
    [("pi-acp", "openai-completions"), ("codex-acp", "openai-responses")],
)
def test_other_agents_keep_their_azure_protocol(agent, protocol):
    resolved = resolve_agent_env(agent, MODEL, dict(AZURE))
    assert resolved["BENCHFLOW_PROVIDER_PROTOCOL"] == protocol
    assert resolved["BENCHFLOW_PROVIDER_BASE_URL"] == (
        "https://res.openai.azure.com/openai/v1"
    )
