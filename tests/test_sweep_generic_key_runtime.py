"""Guards FF-sw-cli-other-C9: the proxy serves the key and endpoint the resolver selected.

A generic key with an endpoint (BENCHFLOW_PROVIDER_API_KEY plus
BENCHFLOW_PROVIDER_BASE_URL) passes resolution for every family. The proxy
the runtime then starts must authenticate with that key at that endpoint, for
bare and provider-qualified spellings, whether or not the host also exports an
unrelated family key. Every credential here is a dummy; no proxy is started.
"""

from __future__ import annotations

import pytest

from benchflow.agents.env import resolve_agent_env
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime

SELECTED = "dummy-account-a"
UNRELATED = "dummy-host-account-b"
FAMILY_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "CODEX_AUTH_JSON",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "BENCHFLOW_PROVIDER_API_KEY",
    "BENCHFLOW_PROVIDER_BASE_URL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
)

# (agent, model spelling, the family key the host might export, endpoint)
CASES = [
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com",
    ),
    (
        "claude-agent-acp",
        "anthropic/claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com",
    ),
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        "https://anthropic-gateway.example.invalid",
    ),
    ("codex-acp", "gpt-5.5", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    (
        "codex-acp",
        "gpt-5.5",
        "OPENAI_API_KEY",
        "https://openai-gateway.example.invalid/v1",
    ),
    ("codex-acp", "openai/gpt-5.5", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    ("gemini", "gemini-2.5-pro", "GEMINI_API_KEY", "https://gemini.example.invalid"),
    (
        "gemini",
        "gemini/gemini-2.5-pro",
        "GEMINI_API_KEY",
        "https://gemini.example.invalid",
    ),
]


@pytest.fixture(autouse=True)
def bare_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in FAMILY_KEYS:
        monkeypatch.delenv(name, raising=False)


async def _proxy_inputs(monkeypatch, agent, model, explicit):
    seen: dict[str, object] = {}

    class _Server:
        base_url = "http://127.0.0.1:4998"
        trajectory = None

        async def is_running(self):
            return True

        async def stop(self):
            return None

    async def fake_start(**kwargs):
        seen.update(kwargs)
        return _Server()

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    resolved = resolve_agent_env(agent, model, dict(explicit))
    _, runtime = await ensure_litellm_runtime(
        agent=agent,
        agent_env=resolved,
        model=model,
        runtime=None,
        environment="docker",
        session_id="c9",
        usage_tracking="off",
    )
    assert runtime is not None
    return seen["route"], seen["agent_env"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host_family_key", [False, True], ids=["bare-host", "host-key-b"]
)
@pytest.mark.parametrize(
    ("agent", "model", "family_key", "endpoint"),
    CASES,
    ids=[f"{a}:{m}:{e.split('//')[1]}" for a, m, _, e in CASES],
)
async def test_proxy_uses_the_resolved_generic_key_and_endpoint(
    monkeypatch, agent, model, family_key, endpoint, host_family_key
):
    """Resolution accepts the generic key, and the proxy serves exactly it."""
    if host_family_key:
        monkeypatch.setenv(family_key, UNRELATED)
    route, proxy_env = await _proxy_inputs(
        monkeypatch,
        agent,
        model,
        {
            "BENCHFLOW_PROVIDER_API_KEY": SELECTED,
            "BENCHFLOW_PROVIDER_BASE_URL": endpoint,
        },
    )
    params = route.litellm_params
    key_ref = str(params["api_key"])
    assert key_ref.startswith("os.environ/")
    assert proxy_env[key_ref.removeprefix("os.environ/")] == SELECTED
    assert params.get("api_base") == endpoint
    assert not runtime_mod._missing_required_env(route, proxy_env)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "family_key"),
    [
        ("claude-agent-acp", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
        ("claude-agent-acp", "anthropic/claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
        ("codex-acp", "gpt-5.5", "OPENAI_API_KEY"),
        ("gemini", "gemini/gemini-2.5-pro", "GEMINI_API_KEY"),
    ],
)
async def test_family_key_route_keeps_its_key_and_default_endpoint(
    monkeypatch, agent, model, family_key
):
    """Control: a family key alone still authenticates the proxy at the default endpoint."""
    route, proxy_env = await _proxy_inputs(
        monkeypatch, agent, model, {family_key: SELECTED}
    )
    key_ref = str(route.litellm_params["api_key"])
    assert proxy_env[key_ref.removeprefix("os.environ/")] == SELECTED
    assert "api_base" not in route.litellm_params
