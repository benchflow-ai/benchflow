"""Guards FF-sw-cli-other-C10, C11 and C12: one selected login decides what goes in.

C10: a native route presents exactly the login resolution selected (an
explicit one first, then the family's order) and its same-value alias; another
account's login of the same family reaches neither the launch environment nor
a staged credential file. C11: an agent-native bridge never replaces a
higher-ranked family key, so the proxy serves the selected account. C12: a raw
command with no model keeps the credential slots its caller passed. Every
value is a dummy; no sandbox, proxy or agent starts.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.acp import runtime as acp_runtime
from benchflow.agents import credentials as cred_mod
from benchflow.agents.env import login_label, resolve_agent_env
from benchflow.agents.registry import AGENTS
from benchflow.native_harness.harnesses import native_harness_for
from benchflow.native_harness.runtime import check_model_route
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime

A = "dummy-account-a"
B = "dummy-account-b"
B2 = "dummy-account-b2"
LOGINS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "CODEX_AUTH_JSON",
    "BENCHFLOW_PROVIDER_API_KEY",
    "BENCHFLOW_PROVIDER_BASE_URL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def bare_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in LOGINS:
        monkeypatch.delenv(name, raising=False)


def _value(name: str, account: str) -> str:
    return json.dumps({"dummy": account}) if name.endswith("_JSON") else account


class _Launched(Exception):
    pass


async def _launch_env(monkeypatch, tmp_path, agent, model, env):
    seen: dict[str, dict[str, str]] = {}

    def transport(**kwargs):
        seen["env"] = dict(kwargs["env"])
        raise _Launched

    monkeypatch.setattr(acp_runtime, "ContainerTransport", transport)
    sandbox = SimpleNamespace(live_process=AsyncMock(return_value=object()))
    with pytest.raises(_Launched):
        await acp_runtime.connect_acp(
            env=sandbox,
            agent=agent,
            agent_launch="agent-under-test",
            agent_env=env,
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    return seen["env"]


async def _native(agent, model, explicit):
    resolved = resolve_agent_env(agent, model, dict(explicit))
    env, runtime = await ensure_litellm_runtime(
        agent=agent,
        agent_env=resolved,
        model=model,
        runtime=None,
        environment="docker",
        session_id="c10",
        usage_tracking="off",
    )
    assert runtime is None, "these inputs select the native route"
    return env


def _logins_in(launch: dict[str, str]) -> dict[str, str]:
    # Empty slots disable inherited credentials; only values are logins.
    return {name: launch[name] for name in LOGINS if launch.get(name)}


# (agent, model, explicit login, host logins of the same family (another
#  account), the logins the launch must hold exactly)
C10_CASES = [
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"CLAUDE_CODE_OAUTH_TOKEN": A},
        {"CLAUDE_OAUTH_TOKEN": B, "ANTHROPIC_AUTH_TOKEN": B2},
        {"CLAUDE_CODE_OAUTH_TOKEN": A},
    ),
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"CLAUDE_OAUTH_TOKEN": A},
        {"CLAUDE_CODE_OAUTH_TOKEN": B},
        {"CLAUDE_OAUTH_TOKEN": A, "CLAUDE_CODE_OAUTH_TOKEN": A},
    ),
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"ANTHROPIC_AUTH_TOKEN": A},
        {"CLAUDE_CODE_OAUTH_TOKEN": B, "CLAUDE_OAUTH_TOKEN": B2},
        {"ANTHROPIC_AUTH_TOKEN": A},
    ),
    (
        "codex-acp",
        "gpt-5.5",
        {"CODEX_ACCESS_TOKEN": A},
        {"CODEX_AUTH_JSON": B},
        {"CODEX_ACCESS_TOKEN": A},
    ),
    (
        "codex-acp",
        "openai/gpt-5.5",
        {"CODEX_AUTH_JSON": A},
        {"CODEX_ACCESS_TOKEN": B},
        {"CODEX_AUTH_JSON": A},
    ),
]
C10_IDS = [f"{c[0]}:{'+'.join(c[2])}-host:{'+'.join(c[3])}" for c in C10_CASES]


def _export(monkeypatch, host):
    for name, account in host.items():
        monkeypatch.setenv(name, _value(name, account))


def _expected(want):
    return {name: _value(name, account) for name, account in want.items()}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "explicit", "host", "want"), C10_CASES, ids=C10_IDS
)
async def test_native_launch_holds_only_the_selected_login(
    monkeypatch, tmp_path, agent, model, explicit, host, want
):
    """C10: another account's same-family login never reaches the ACP process."""
    _export(monkeypatch, host)
    explicit = {k: _value(k, v) for k, v in explicit.items()}
    env = await _native(agent, model, explicit)
    launch = await _launch_env(monkeypatch, tmp_path, agent, model, env)
    assert _logins_in(launch) == _expected(want)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "explicit", "host", "want"),
    [c for c in C10_CASES if c[0] == "claude-agent-acp"],
    ids=[
        i for i, c in zip(C10_IDS, C10_CASES, strict=True) if c[0] == "claude-agent-acp"
    ],
)
async def test_native_harness_holds_only_the_selected_login(
    monkeypatch, agent, model, explicit, host, want
):
    """C10: the native CLI harness admits the same one login."""
    _export(monkeypatch, host)
    explicit = {k: _value(k, v) for k, v in explicit.items()}
    env = await _native(agent, model, explicit)
    launch = check_model_route(native_harness_for(agent), agent, model, env)
    assert _logins_in(launch) == _expected(want)


@pytest.mark.asyncio
async def test_inherited_logins_resolve_to_one_and_the_label_names_it(
    monkeypatch, tmp_path
):
    """C10: with two inherited Claude logins exactly one is presented and labelled."""
    _export(monkeypatch, {"CLAUDE_CODE_OAUTH_TOKEN": B, "ANTHROPIC_AUTH_TOKEN": B2})
    model = "claude-sonnet-4-6"
    env = await _native("claude-agent-acp", model, {})
    launch = await _launch_env(monkeypatch, tmp_path, "claude-agent-acp", model, env)
    logins = _logins_in(launch)
    assert len(logins) == 1
    (name,) = logins
    assert login_label("claude-agent-acp", model, env) == f"{name} (environment)"


async def _staged(monkeypatch, agent, model, env):
    uploads: dict[str, str] = {}

    async def upload(sandbox, path, content, *, owner=None):
        uploads[path] = content

    monkeypatch.setattr(cred_mod, "upload_credential", upload)
    await cred_mod.write_credential_files(
        object(), agent, dict(env), AGENTS[agent], model, "/home/agent"
    )
    return uploads


@pytest.mark.asyncio
async def test_unselected_codex_auth_json_is_not_staged(monkeypatch):
    """C10: an inherited auth.json of account B is not written next to A's token."""
    _export(monkeypatch, {"CODEX_AUTH_JSON": B})
    env = await _native("codex-acp", "gpt-5.5", {"CODEX_ACCESS_TOKEN": A})
    uploads = await _staged(monkeypatch, "codex-acp", "gpt-5.5", env)
    assert not any(B in content for content in uploads.values()), sorted(uploads)


@pytest.mark.asyncio
async def test_selected_codex_auth_json_is_staged(monkeypatch):
    """Control: the selected auth.json is still written."""
    _export(monkeypatch, {"CODEX_ACCESS_TOKEN": B})
    env = await _native(
        "codex-acp", "openai/gpt-5.5", {"CODEX_AUTH_JSON": _value("CODEX_AUTH_JSON", A)}
    )
    uploads = await _staged(monkeypatch, "codex-acp", "openai/gpt-5.5", env)
    assert uploads == {"/home/agent/.codex/auth.json": _value("CODEX_AUTH_JSON", A)}


async def _proxy_key(monkeypatch, agent, model, explicit):
    seen: dict[str, object] = {}

    class _Server:
        base_url = "http://127.0.0.1:4997"
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
        session_id="c11",
        usage_tracking="off",
    )
    assert runtime is not None
    route, proxy_env = seen["route"], seen["agent_env"]
    ref = str(route.litellm_params["api_key"]).removeprefix("os.environ/")
    assert not runtime_mod._missing_required_env(route, proxy_env)
    return proxy_env[ref]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"])
@pytest.mark.parametrize(
    ("explicit", "host", "served"),
    [
        ({"ANTHROPIC_API_KEY": A}, {"ANTHROPIC_AUTH_TOKEN": B}, A),
        ({"ANTHROPIC_API_KEY": A}, {"CLAUDE_CODE_OAUTH_TOKEN": B}, A),
        ({"ANTHROPIC_AUTH_TOKEN": A}, {"ANTHROPIC_API_KEY": B}, A),
        ({}, {"ANTHROPIC_API_KEY": B, "ANTHROPIC_AUTH_TOKEN": B2}, B),
    ],
    ids=[
        "explicit-key-vs-host-token",
        "explicit-key-vs-host-oauth",
        "explicit-token-vs-host-key",
        "both-inherited",
    ],
)
async def test_proxy_serves_the_selected_family_login(
    monkeypatch, model, explicit, host, served
):
    """C11: a bridge never replaces a higher-ranked family key; the proxy serves the selection."""
    _export(monkeypatch, host)
    assert await _proxy_key(monkeypatch, "claude-agent-acp", model, explicit) == served


RAW = "/opt/benchflow/bin/claude-agent-acp"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "explicit",
    [{"ANTHROPIC_API_KEY": A}, {"CLAUDE_CODE_OAUTH_TOKEN": A}, {"OPENAI_API_KEY": A}],
    ids=lambda e: next(iter(e)),
)
async def test_raw_command_keeps_its_explicit_login(monkeypatch, tmp_path, explicit):
    """C12: a raw command with no model is launched with the slot its caller passed."""
    assert RAW not in AGENTS
    env = resolve_agent_env(RAW, None, dict(explicit))
    env, runtime = await ensure_litellm_runtime(
        agent=RAW,
        agent_env=env,
        model=None,
        runtime=None,
        environment="docker",
        session_id="c12",
        usage_tracking="off",
    )
    assert runtime is None
    launch = await _launch_env(monkeypatch, tmp_path, RAW, None, env)
    (name,) = explicit
    assert launch.get(name) == A


@pytest.mark.asyncio
async def test_raw_command_gets_no_inherited_login(monkeypatch, tmp_path):
    """C12 control: only the caller's slot is admitted, not the host's other logins."""
    _export(monkeypatch, {"CODEX_AUTH_JSON": B, "CLAUDE_CODE_OAUTH_TOKEN": B2})
    env = resolve_agent_env(RAW, None, {"ANTHROPIC_API_KEY": A})
    launch = await _launch_env(monkeypatch, tmp_path, RAW, None, env)
    assert _logins_in(launch) == {"ANTHROPIC_API_KEY": A}
