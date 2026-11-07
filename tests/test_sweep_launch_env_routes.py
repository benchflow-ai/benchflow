"""Guards FF-sw-cli-other-C8: an agent process holds only its selected route's credentials.

The host exports every credential variable BenchFlow knows of, each with a
distinct dummy value. For every route, the environment the agent process is
actually launched with (the ACP transport, the session-factory sandbox and the
native-harness launch) may hold a credential only when it is that route's own:
the proxy's master key on a proxy route, the selected family's login on a
native route, the agent's own slot on a no-model route. All values are dummies.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.acp import runtime as acp_runtime
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.providers import PROVIDERS
from benchflow.agents.registry import AGENTS
from benchflow.native_harness.harnesses import native_harness_for
from benchflow.native_harness.runtime import check_model_route
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime
from benchflow.rollout import session_factory_runtime as sf_runtime

MASTER_KEY_ENV = "BENCHFLOW_LITELLM_MASTER_KEY"
SELECTED = "dummy-selected-a"
CLAUDE_LOGINS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
)
CODEX_LOGINS = ("CODEX_ACCESS_TOKEN", "CODEX_AUTH_JSON")


def _known_host_credentials() -> list[str]:
    names = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AZURE_API_KEY",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "CODEX_ACCESS_TOKEN",
        "CODEX_API_KEY",
        "CODEX_AUTH_JSON",
        "DAYTONA_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "LLM_API_KEY",
        "META_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MUSE_AUTH_JSON",
        "OPENAI_API_KEY",
    }
    names |= {cfg.auth_env for cfg in PROVIDERS.values() if cfg.auth_env}
    names |= {
        cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"]
        for cfg in AGENTS.values()
        if "BENCHFLOW_PROVIDER_API_KEY" in cfg.env_mapping
    }
    return sorted(names)


HOST_CREDENTIALS = _known_host_credentials()


def _dummy(name: str) -> str:
    token = f"dummy-host-{name.lower()}"
    if name.endswith("_JSON"):
        return json.dumps({"dummy": token})
    return token


def _host_token(name: str) -> str:
    return f"dummy-host-{name.lower()}"


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host with no login files that exports the credentials it is given."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in (
        *HOST_CREDENTIALS,
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_NAME",
        "OPENAI_BASE_URL",
        "LLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    def export(*, without: tuple[str, ...] = ()) -> None:
        for name in HOST_CREDENTIALS:
            if name not in without:
                monkeypatch.setenv(name, _dummy(name))

    return export


class _Launched(Exception):
    pass


async def _acp_launch_env(monkeypatch, tmp_path, agent, model, agent_env):
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
            agent_env=agent_env,
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    return seen["env"]


async def _session_factory_env(monkeypatch, tmp_path, agent, model, agent_env):
    seen: dict[str, dict[str, str]] = {}

    class _Sandbox:
        def __init__(self, env, connect_env, cwd, sandbox_user=None):
            seen["env"] = dict(connect_env)

    class _Agent:
        async def connect(self, sandbox, name):
            return object()

    monkeypatch.setattr(sf_runtime, "SessionFactorySandbox", _Sandbox)
    monkeypatch.setattr(sf_runtime, "_load_session_factory", lambda spec: _Agent)
    await sf_runtime.connect_session_factory(
        env=object(),
        agent=agent,
        session_factory="dummy.module:factory",
        agent_env=agent_env,
        sandbox_user=None,
        model=model,
        rollout_dir=tmp_path,
        timeout=0,
    )
    return seen["env"]


async def _proxied(monkeypatch, agent, model, explicit):
    class _Server:
        base_url = "http://127.0.0.1:4999"
        trajectory = None

        async def is_running(self):
            return True

        async def stop(self):
            return None

    async def fake_start(**kwargs):
        return _Server()

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    resolved = resolve_agent_env(agent, model, dict(explicit))
    wired, runtime = await ensure_litellm_runtime(
        agent=agent,
        agent_env=resolved,
        model=model,
        runtime=None,
        environment="docker",
        session_id="c8",
        usage_tracking="off",
    )
    assert runtime is not None, "these inputs select the proxy route"
    return wired


def _master_slots(launch: dict[str, str]) -> set[str]:
    """The proxy route's own credential: the master key, wherever wiring put it."""
    master = launch.get("RUNTIME_LITELLM_MASTER_KEY") or launch[MASTER_KEY_ENV]
    assert master.startswith("sk-runtime-")
    return {name for name, value in launch.items() if value == master}


def _assert_only(launch: dict[str, str], allowed: set[str]) -> None:
    """No host credential outside ``allowed`` survives, including aliases."""
    for name in HOST_CREDENTIALS:
        if name in allowed:
            continue
        # Explicit empty slots suppress container inheritance; they carry no
        # credential. Continue checking every value for the host's token.
        assert not launch.get(name), f"{name} reached the agent"
        token = _host_token(name)
        carriers = sorted(key for key, value in launch.items() if token in value)
        assert not carriers, f"{name}'s value reached the agent as {carriers}"


GENERIC = {
    "BENCHFLOW_PROVIDER_API_KEY": SELECTED,
    "BENCHFLOW_PROVIDER_BASE_URL": "https://provider.example.invalid/v1",
}
# (agent, model, the explicit inputs that select a provider route)
PROXY_ROUTES = [
    ("codex-acp", "openai/gpt-5.5", GENERIC),
    ("codex-acp", "openai/gpt-5.5", {"OPENAI_API_KEY": SELECTED}),
    ("codex-acp", "gpt-5.5", {"OPENAI_API_KEY": SELECTED}),
    ("claude-agent-acp", "claude-sonnet-4-6", {"ANTHROPIC_API_KEY": SELECTED}),
    (
        "claude-agent-acp",
        "anthropic/claude-sonnet-4-6",
        {"ANTHROPIC_API_KEY": SELECTED},
    ),
    ("opencode", "openai/gpt-5.5", GENERIC),
    ("openhands", "openai/gpt-5.5", GENERIC),
    ("pi-acp", "openai/gpt-5.5", GENERIC),
    ("gemini", "gemini/gemini-2.5-pro", {"GEMINI_API_KEY": SELECTED}),
    ("deepagents", "deepseek/deepseek-v4-pro", {"DEEPSEEK_API_KEY": SELECTED}),
]
PROXY_IDS = [f"{agent}:{model}:{'+'.join(env)}" for agent, model, env in PROXY_ROUTES]


@pytest.mark.asyncio
@pytest.mark.parametrize(("agent", "model", "explicit"), PROXY_ROUTES, ids=PROXY_IDS)
async def test_proxy_route_acp_process_holds_only_the_proxy_key(
    host, monkeypatch, tmp_path, agent, model, explicit
):
    """Provider routes: no host login and no upstream key reaches the ACP process."""
    host()
    wired = await _proxied(monkeypatch, agent, model, explicit)
    launch = await _acp_launch_env(monkeypatch, tmp_path, agent, model, wired)
    _assert_only(launch, _master_slots(launch))
    assert SELECTED not in "\n".join(launch.values())
    assert launch["RUNTIME_LITELLM_MASTER_KEY"] == wired[MASTER_KEY_ENV]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "explicit"), PROXY_ROUTES[:4], ids=PROXY_IDS[:4]
)
async def test_proxy_route_session_factory_holds_only_the_proxy_key(
    host, monkeypatch, tmp_path, agent, model, explicit
):
    """The in-process session-factory sandbox gets the same launch environment."""
    host()
    wired = await _proxied(monkeypatch, agent, model, explicit)
    launch = await _session_factory_env(monkeypatch, tmp_path, agent, model, wired)
    _assert_only(launch, _master_slots(launch))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "explicit"),
    [
        ("claude-agent-acp", "claude-sonnet-4-6", {"ANTHROPIC_API_KEY": SELECTED}),
        ("codex-acp", "openai/gpt-5.5", GENERIC),
    ],
    ids=["claude", "codex"],
)
async def test_proxy_route_native_harness_holds_only_the_proxy_key(
    host, monkeypatch, agent, model, explicit
):
    """The native CLI harness launches with the same allowlist."""
    host()
    wired = await _proxied(monkeypatch, agent, model, explicit)
    launch = check_model_route(native_harness_for(agent), agent, model, wired)
    _assert_only(launch, _master_slots(launch))


# (agent, model, explicit login, host names left unset because they would
#  select a provider route instead, the selected family's login names)
NATIVE_ROUTES = [
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"CLAUDE_CODE_OAUTH_TOKEN": SELECTED},
        ("ANTHROPIC_API_KEY",),
        set(CLAUDE_LOGINS),
    ),
    (
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"ANTHROPIC_AUTH_TOKEN": SELECTED},
        ("ANTHROPIC_API_KEY",),
        set(CLAUDE_LOGINS),
    ),
    (
        "codex-acp",
        "gpt-5.5",
        {"CODEX_ACCESS_TOKEN": SELECTED},
        ("OPENAI_API_KEY", "CODEX_API_KEY"),
        set(CODEX_LOGINS),
    ),
    (
        "codex-acp",
        "openai/gpt-5.5",
        {"CODEX_AUTH_JSON": json.dumps({"dummy": SELECTED})},
        ("OPENAI_API_KEY", "CODEX_API_KEY"),
        set(CODEX_LOGINS),
    ),
    (
        "muse-code",
        "muse-spark",
        {"MUSE_AUTH_JSON": json.dumps({"dummy": SELECTED})},
        (),
        {"MUSE_AUTH_JSON"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model", "explicit", "unset", "family"),
    NATIVE_ROUTES,
    ids=[f"{r[0]}:{'+'.join(r[2])}" for r in NATIVE_ROUTES],
)
async def test_native_route_keeps_only_its_own_family_login(
    host, monkeypatch, tmp_path, agent, model, explicit, unset, family
):
    """A native login reaches the agent only when its own native route is selected."""
    host(without=unset)
    resolved = resolve_agent_env(agent, model, dict(explicit))
    env, runtime = await ensure_litellm_runtime(
        agent=agent,
        agent_env=resolved,
        model=model,
        runtime=None,
        environment="docker",
        session_id="c8",
        usage_tracking="off",
    )
    assert runtime is None, "these inputs select the native route"
    launch = await _acp_launch_env(monkeypatch, tmp_path, agent, model, env)
    _assert_only(launch, family)
    assert SELECTED in "\n".join(launch[name] for name in family if name in launch)


@pytest.mark.asyncio
async def test_no_model_route_keeps_only_the_agents_own_slot(
    host, monkeypatch, tmp_path
):
    """A caller-configured endpoint (a continuation replay) gets only its own key."""
    host()
    explicit = {
        "LLM_BASE_URL": "http://replay.example.invalid/v1",
        "LLM_API_KEY": SELECTED,
        "LLM_MODEL": "openai/replay",
    }
    resolved = resolve_agent_env("openhands", None, dict(explicit))
    env, runtime = await ensure_litellm_runtime(
        agent="openhands",
        agent_env=resolved,
        model=None,
        runtime=None,
        environment="docker",
        session_id="c8",
        usage_tracking="off",
    )
    assert runtime is None
    launch = await _acp_launch_env(monkeypatch, tmp_path, "openhands", None, env)
    _assert_only(launch, {"LLM_API_KEY"})
    assert launch["LLM_API_KEY"] == SELECTED


def test_every_known_host_credential_is_classified_as_a_credential():
    """The allowlist's credential set covers every name this file enumerates,
    and an inherited name is a runtime setting only by being named one."""
    from benchflow.agents.env import inheritable_env_names
    from benchflow.agents.launch_env import (
        _NON_CREDENTIAL_ENV,
        host_credential_env_names,
    )

    known = host_credential_env_names()
    assert set(HOST_CREDENTIALS) <= known
    url_params = {n for cfg in PROVIDERS.values() for n in cfg.url_params.values()}
    assert inheritable_env_names() - known <= _NON_CREDENTIAL_ENV | url_params


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_scripted_agents_get_no_credential(agent):
    from benchflow.agents.launch_env import agent_launch_env

    env = {name: _dummy(name) for name in HOST_CREDENTIALS}
    env["BENCHFLOW_SETTING"] = "kept"
    launch = agent_launch_env(agent, None, env)
    _assert_only(launch, set())
    assert launch == {"BENCHFLOW_SETTING": "kept"}
