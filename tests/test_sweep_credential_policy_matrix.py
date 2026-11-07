"""The credential policy table, checked against what reaches the sandbox.

Every cell is (agent, route, host credential set). The host exports every
known credential with a distinct dummy value (or nothing), holds an ADC login
and, in the full set, Claude and Codex login files. A cell runs the real
resolution, install-time staging, proxy wiring (with a fake proxy) and ACP
launch, then lists every dummy token found in a file written into the
sandbox or in the agent's launch variables. That list must equal the cell's
expected entries exactly, no more and no less, and the run's recorded table
(``launch_env.credential_policy``) must name exactly those credentials.

Named rows keep the FF-sw-cli-other C8-C19 inputs. All values are dummies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.acp import runtime as acp_runtime
from benchflow.agents import credentials as cred_mod
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    register_agent,
)
from benchflow.errors import MissingCredentialError
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime
from tests.test_sweep_launch_env_routes import HOST_CREDENTIALS, _dummy

try:
    from benchflow.agents.launch_env import credential_policy
except ImportError:  # trees before the policy table: their own staging rule
    credential_policy = None


def _stages_host_login(agent, model, env) -> bool:
    if credential_policy is not None:
        return credential_policy(agent, model, env).host_login
    try:
        from benchflow.agents.launch_env import stages_host_login
    except ImportError:
        return bool(env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"))
    return stages_host_login(agent, model, env)


HOME = "/home/agent"
ADC_TOKEN = "dummy-host-adc"
CLAUDE_FILE_TOKEN = "dummy-host-claude-login-file"
CODEX_FILE_TOKEN = "dummy-host-codex-login-file"
RAW = "/opt/runtime/bin/raw-acp"
TWO_PART = "matrix-two-part"
CLAUDE_WRAPPER = "matrix-claude-wrapper"
SETTINGS = (
    "BENCHFLOW_PROVIDER_BASE_URL",
    "BENCHFLOW_PROVIDER_NAME",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AZURE_RESOURCE",
    "AZURE_API_ENDPOINT",
    "XIAOMI_BASE_URL",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "PROJECT_ID",
)


def explicit(name: str) -> str:
    token = f"dummy-explicit-{name.lower()}"
    if name.endswith("_BASE_URL"):
        return "https://provider.example.invalid/v1"
    return json.dumps({"dummy": token}) if name.endswith("_JSON") else token


def token_of(name: str) -> str:
    return f"dummy-explicit-{name.lower()}"


@dataclass(frozen=True)
class Cell:
    agent: str
    model: str | None
    passed: tuple[str, ...] = ()
    host: str = "bare"  # "bare": no credential variables; "all": every one
    unset: tuple[str, ...] = ()
    settings: dict[str, str] = field(default_factory=dict)
    login_files: bool = False
    # The credentials the table admits, by variable name ("HOST_LOGIN" for
    # host login files), and the tokens that may reach the sandbox.
    names: frozenset[str] = frozenset()
    tokens: frozenset[str] = frozenset()


@pytest.fixture(autouse=True)
def custom_agents():
    register_agent(
        TWO_PART,
        "true",
        f"{TWO_PART} --acp",
        requires_env=["CLIENT_ID", "CLIENT_SECRET"],
    )
    register_agent(
        CLAUDE_WRAPPER,
        "true",
        f"{CLAUDE_WRAPPER} --acp",
        requires_env=["ANTHROPIC_API_KEY", "PROJECT_ID"],
        subscription_auth=AGENTS["claude-agent-acp"].subscription_auth,
    )
    yield
    for name in (TWO_PART, CLAUDE_WRAPPER):
        for registry in (AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH):
            registry.pop(name, None)


def _prepare_host(tmp_path, monkeypatch, cell: Cell) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in (*HOST_CREDENTIALS, *SETTINGS):
        monkeypatch.delenv(name, raising=False)
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text(json.dumps({"refresh_token": ADC_TOKEN}))
    if cell.host == "all":
        for name in HOST_CREDENTIALS:
            if name not in cell.unset:
                monkeypatch.setenv(name, _dummy(name))
    for name, value in cell.settings.items():
        monkeypatch.setenv(name, value)
    if cell.login_files or cell.host == "all":
        for folder, filename, token in (
            (".claude", ".credentials.json", CLAUDE_FILE_TOKEN),
            (".codex", "auth.json", CODEX_FILE_TOKEN),
        ):
            path = tmp_path / folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"dummy": token}))


class _Launched(Exception):
    pass


async def _run(tmp_path, monkeypatch, cell: Cell):
    """(sandbox tokens, recorded table) for one cell; None table if refused."""
    _prepare_host(tmp_path, monkeypatch, cell)
    files: dict[str, str] = {}

    async def upload(sandbox, path, content, *, owner=None):
        files[path] = content

    monkeypatch.setattr(cred_mod, "upload_credential", upload)
    try:
        resolved = resolve_agent_env(
            cell.agent, cell.model, {name: explicit(name) for name in cell.passed}
        )
    except (ValueError, MissingCredentialError) as exc:
        return None, files, {}, exc
    staged = dict(resolved)
    policy = (
        credential_policy(cell.agent, cell.model, staged)
        if credential_policy is not None
        else None
    )
    await cred_mod.write_credential_files(
        object(), cell.agent, staged, AGENTS.get(cell.agent), cell.model, HOME
    )
    if _stages_host_login(cell.agent, cell.model, staged):
        await cred_mod.upload_subscription_auth(object(), cell.agent, HOME)

    class _Server:
        base_url = "http://127.0.0.1:4995"
        trajectory = None

        async def is_running(self):
            return True

        async def stop(self):
            return None

    async def fake_start(**kwargs):
        return _Server()

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    wired, _ = await ensure_litellm_runtime(
        agent=cell.agent,
        agent_env=staged,
        model=cell.model,
        runtime=None,
        environment="docker",
        session_id="matrix",
        usage_tracking="off",
    )
    seen: dict[str, dict[str, str]] = {}

    def transport(**kwargs):
        seen["env"] = dict(kwargs["env"])
        raise _Launched

    monkeypatch.setattr(acp_runtime, "ContainerTransport", transport)
    with pytest.raises(_Launched):
        await acp_runtime.connect_acp(
            env=SimpleNamespace(live_process=AsyncMock(return_value=object())),
            agent=cell.agent,
            agent_launch="agent-under-test",
            agent_env=wired,
            sandbox_user="agent",
            model=cell.model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    return policy, files, seen["env"], None


def _tokens(files: dict[str, str], launch: dict[str, str]) -> set[str]:
    found: set[str] = set()
    for text in [*files.values(), *launch.values()]:
        for raw in text.replace('"', " ").replace(",", " ").split():
            if raw.startswith("dummy-"):
                found.add(raw.rstrip("}"))
    return found


def _table_names(policy) -> set[str]:
    """What the table admits besides the proxy key, by variable name."""
    names = set(policy._own())
    if policy.host_login:
        names.add("HOST_LOGIN")
    return names


async def _check(tmp_path, monkeypatch, cell: Cell) -> None:
    policy, files, launch, refused = await _run(tmp_path, monkeypatch, cell)
    assert refused is None, f"refused: {refused}"
    assert _tokens(files, launch) == set(cell.tokens), (sorted(files), launch)
    if policy is not None:
        assert _table_names(policy) == set(cell.names), policy


# ── Proxy routes: every agent x every provider family x host set ──

FAMILIES = {
    "anthropic": ("claude-sonnet-4-6", ("ANTHROPIC_API_KEY",), {}),
    "openai": ("openai/gpt-5.5", ("OPENAI_API_KEY",), {}),
    "gemini": ("gemini/gemini-2.5-pro", ("GEMINI_API_KEY",), {}),
    "vertex": ("google-vertex/gemini-2.5-pro", (), {"GOOGLE_CLOUD_PROJECT": "p-a"}),
    "anthropic-vertex": (
        "anthropic-vertex/claude-sonnet-4-6",
        (),
        {"GOOGLE_CLOUD_PROJECT": "p-a", "GOOGLE_CLOUD_LOCATION": "global"},
    ),
    "bedrock": (
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        ("AWS_BEARER_TOKEN_BEDROCK",),
        {"AWS_REGION": "us-west-2"},
    ),
    "azure": (
        "azure-foundry-openai/gpt-5.5",
        ("AZURE_API_KEY",),
        {"AZURE_RESOURCE": "resource-a"},
    ),
    "xiaomi": (
        "xiaomi/mimo-v2.6-flash",
        ("XIAOMI_API_KEY",),
        {"XIAOMI_BASE_URL": "https://xiaomi.example.invalid/v1"},
    ),
    "deepseek": ("deepseek/deepseek-v4-pro", ("DEEPSEEK_API_KEY",), {}),
}
PROXY_AGENTS = (
    "claude-agent-acp",
    "codex-acp",
    "gemini",
    "opencode",
    "openhands",
    "pi-acp",
    "mimo",
    "deepagents",
    TWO_PART,
    RAW,
)


def _proxy_cell(agent: str, family: str, host: str) -> Cell:
    model, keys, settings = FAMILIES[family]
    passed = keys
    names: frozenset[str] = frozenset()
    tokens: frozenset[str] = frozenset()
    if agent == TWO_PART:
        passed = (*keys, "CLIENT_ID", "CLIENT_SECRET")
        names = frozenset({"CLIENT_ID", "CLIENT_SECRET"})
        tokens = frozenset({token_of("CLIENT_ID"), token_of("CLIENT_SECRET")})
    if agent == "gemini" and family == "vertex":
        # The one native Vertex route; the ADC file is its login.
        names = frozenset({"GOOGLE_APPLICATION_CREDENTIALS_JSON"})
        tokens = frozenset({ADC_TOKEN})
    return Cell(
        agent, model, passed, host=host, settings=settings, names=names, tokens=tokens
    )


# Agent/provider pairs whose protocols do not match: refused at resolution.
PROTOCOL_REFUSED = frozenset(
    {
        ("claude-agent-acp", "openai"),
        ("claude-agent-acp", "vertex"),
        ("claude-agent-acp", "azure"),
        ("claude-agent-acp", "deepseek"),
        ("codex-acp", "vertex"),
        ("codex-acp", "anthropic-vertex"),
        ("codex-acp", "xiaomi"),
        ("codex-acp", "deepseek"),
        ("mimo", "anthropic-vertex"),
    }
)

PROXY_CELLS = [
    pytest.param(
        _proxy_cell(agent, family, host),
        (agent, family) in PROTOCOL_REFUSED,
        id=f"{agent.rsplit('/', 1)[-1]}-{family}-{host}",
    )
    for agent in PROXY_AGENTS
    for family in FAMILIES
    for host in ("bare", "all")
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("cell", "protocol_refused"), PROXY_CELLS)
async def test_proxy_matrix(tmp_path, monkeypatch, cell, protocol_refused):
    """Each agent on each provider family: the sandbox holds exactly its table."""
    policy, files, launch, refused = await _run(tmp_path, monkeypatch, cell)
    assert (refused is not None) == protocol_refused, refused
    if refused is not None:
        # The agent cannot speak this provider's protocol: refused during
        # resolution, before anything was staged.
        assert files == {}
        assert isinstance(refused, ValueError)
        assert "provider protocol" in str(refused)
        return
    assert _tokens(files, launch) == set(cell.tokens), (sorted(files), launch)
    if policy is not None:
        assert _table_names(policy) == set(cell.names), policy


# ── Native logins, no-model and raw routes, and the named findings ──

CLAUDE_KEYS = ("ANTHROPIC_API_KEY",)
CODEX_KEYS = ("OPENAI_API_KEY", "CODEX_API_KEY")
NAMED = {
    # C8: provider route with every host login exported.
    "C8-codex-provider-route": Cell(
        "codex-acp",
        "openai/gpt-5.5",
        ("BENCHFLOW_PROVIDER_API_KEY", "BENCHFLOW_PROVIDER_BASE_URL"),
        host="all",
    ),
    # C10: another account's same-family login on a native route.
    "C10-claude-oauth": Cell(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        ("CLAUDE_CODE_OAUTH_TOKEN",),
        host="all",
        unset=CLAUDE_KEYS,
        names=frozenset({"CLAUDE_CODE_OAUTH_TOKEN"}),
        tokens=frozenset({token_of("CLAUDE_CODE_OAUTH_TOKEN")}),
    ),
    "C10-claude-oauth-alias": Cell(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        ("CLAUDE_OAUTH_TOKEN",),
        host="all",
        unset=CLAUDE_KEYS,
        names=frozenset({"CLAUDE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}),
        tokens=frozenset({token_of("CLAUDE_OAUTH_TOKEN")}),
    ),
    "C10-claude-auth-token": Cell(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        ("ANTHROPIC_AUTH_TOKEN",),
        host="all",
        unset=CLAUDE_KEYS,
        names=frozenset({"ANTHROPIC_AUTH_TOKEN"}),
        tokens=frozenset({token_of("ANTHROPIC_AUTH_TOKEN")}),
    ),
    "C10-codex-access-token": Cell(
        "codex-acp",
        "gpt-5.5",
        ("CODEX_ACCESS_TOKEN",),
        host="all",
        unset=CODEX_KEYS,
        names=frozenset({"CODEX_ACCESS_TOKEN"}),
        tokens=frozenset({token_of("CODEX_ACCESS_TOKEN")}),
    ),
    "C10-codex-auth-json": Cell(
        "codex-acp",
        "openai/gpt-5.5",
        ("CODEX_AUTH_JSON",),
        host="all",
        unset=CODEX_KEYS,
        names=frozenset({"CODEX_AUTH_JSON"}),
        tokens=frozenset({token_of("CODEX_AUTH_JSON")}),
    ),
    "native-muse": Cell(
        "muse-code",
        "muse-spark",
        ("MUSE_AUTH_JSON",),
        host="all",
        names=frozenset({"MUSE_AUTH_JSON"}),
        tokens=frozenset({token_of("MUSE_AUTH_JSON")}),
    ),
    "native-claude-host-login-file": Cell(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        login_files=True,
        names=frozenset({"HOST_LOGIN"}),
        tokens=frozenset({CLAUDE_FILE_TOKEN}),
    ),
    "native-codex-host-login-file": Cell(
        "codex-acp",
        "gpt-5.5",
        login_files=True,
        names=frozenset({"HOST_LOGIN"}),
        tokens=frozenset({CODEX_FILE_TOKEN}),
    ),
    # C11: an explicit family key beats an inherited token on the proxy.
    "C11-explicit-key-proxy": Cell(
        "claude-agent-acp", "claude-sonnet-4-6", ("ANTHROPIC_API_KEY",), host="all"
    ),
    # C12: a raw command keeps the slot its caller passed.
    "C12-raw-command-key": Cell(
        RAW,
        None,
        ("ANTHROPIC_API_KEY",),
        host="all",
        names=frozenset({"ANTHROPIC_API_KEY"}),
        tokens=frozenset({token_of("ANTHROPIC_API_KEY")}),
    ),
    # C14: Vertex proxy runs carry no ADC; Gemini CLI's native route does.
    "C14-vertex-proxy": Cell(
        "openhands",
        "google-vertex/gemini-2.5-pro",
        host="all",
        settings={"GOOGLE_CLOUD_PROJECT": "p-a"},
    ),
    "D847-gemini-native-vertex": Cell(
        "gemini",
        "google-vertex/gemini-2.5-pro",
        host="all",
        settings={"GOOGLE_CLOUD_PROJECT": "p-a"},
        names=frozenset({"GOOGLE_APPLICATION_CREDENTIALS_JSON"}),
        tokens=frozenset({ADC_TOKEN}),
    ),
    # C15: jointly required parts reach their agent together.
    "C15-two-part-no-model": Cell(
        TWO_PART,
        None,
        ("CLIENT_ID", "CLIENT_SECRET"),
        host="all",
        names=frozenset({"CLIENT_ID", "CLIENT_SECRET"}),
        tokens=frozenset({token_of("CLIENT_ID"), token_of("CLIENT_SECRET")}),
    ),
    # C17: another agent's declared credentials never reach this agent.
    "C17-openhands-shared-env": Cell(
        "openhands",
        "openai/gpt-5.5",
        ("OPENAI_API_KEY", "CLIENT_ID", "CLIENT_SECRET"),
        host="all",
    ),
    "C17-claude-native-shared-env": Cell(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        ("CLAUDE_CODE_OAUTH_TOKEN", "CLIENT_ID", "CLIENT_SECRET"),
        unset=CLAUDE_KEYS,
        names=frozenset({"CLAUDE_CODE_OAUTH_TOKEN"}),
        tokens=frozenset({token_of("CLAUDE_CODE_OAUTH_TOKEN")}),
    ),
    # C18: an ordinary requirement does not suppress the host login.
    "C18-wrapper-host-login": Cell(
        CLAUDE_WRAPPER,
        None,
        ("PROJECT_ID",),
        login_files=True,
        names=frozenset({"PROJECT_ID", "HOST_LOGIN"}),
        tokens=frozenset({token_of("PROJECT_ID"), CLAUDE_FILE_TOKEN}),
    ),
    # C19: a raw command's explicit ADC variable is kept.
    "C19-raw-command-adc": Cell(
        RAW,
        None,
        ("GOOGLE_APPLICATION_CREDENTIALS_JSON",),
        host="all",
        names=frozenset({"GOOGLE_APPLICATION_CREDENTIALS_JSON"}),
        tokens=frozenset({token_of("GOOGLE_APPLICATION_CREDENTIALS_JSON")}),
    ),
    "raw-command-two-slots": Cell(
        RAW,
        None,
        ("CLIENT_ID", "CLIENT_SECRET"),
        host="all",
        names=frozenset({"CLIENT_ID", "CLIENT_SECRET"}),
        tokens=frozenset({token_of("CLIENT_ID"), token_of("CLIENT_SECRET")}),
    ),
    "no-model-openhands-replay": Cell(
        "openhands",
        None,
        ("LLM_API_KEY",),
        host="all",
        names=frozenset({"LLM_API_KEY"}),
        tokens=frozenset({token_of("LLM_API_KEY")}),
    ),
    "no-model-claude-key": Cell(
        "claude-agent-acp",
        None,
        ("ANTHROPIC_API_KEY",),
        host="all",
        # A generic key with no model is refused by design (C7).
        unset=("BENCHFLOW_PROVIDER_API_KEY",),
        names=frozenset({"ANTHROPIC_API_KEY"}),
        tokens=frozenset({token_of("ANTHROPIC_API_KEY")}),
    ),
    "scripted-oracle": Cell("oracle", None, host="all"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("cell", [pytest.param(c, id=k) for k, c in NAMED.items()])
async def test_named_rows(tmp_path, monkeypatch, cell):
    """C8-C19 and the native, no-model and raw routes: exactly the table."""
    await _check(tmp_path, monkeypatch, cell)


# ── An agent that routes its own model calls (model_route = "agent") ──

OWN_ROUTE = "matrix-own-route"


@pytest.fixture
def own_route_agent():
    register_agent(
        OWN_ROUTE,
        "true",
        f"{OWN_ROUTE} --acp",
        requires_env=["CLIENT_ID", "CLIENT_SECRET"],
        model_route="agent",
    )
    yield OWN_ROUTE
    for registry in (AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH):
        registry.pop(OWN_ROUTE, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label",
    [None, *(model for model, _keys, _settings in FAMILIES.values())],
    ids=["no-label", *FAMILIES],
)
async def test_own_model_route_admits_only_its_declared_credentials(
    tmp_path, monkeypatch, own_route_agent, label
):
    """An agent with model_route = "agent" is on the direct
    route whatever model label it carries. With every host credential
    exported, only its own declared parts reach the sandbox: the label never
    selects a provider, a login or a proxy key."""
    cell = Cell(
        own_route_agent,
        label,
        ("CLIENT_ID", "CLIENT_SECRET"),
        host="all",
        names=frozenset({"CLIENT_ID", "CLIENT_SECRET"}),
        tokens=frozenset({token_of("CLIENT_ID"), token_of("CLIENT_SECRET")}),
    )
    policy, files, launch, refused = await _run(tmp_path, monkeypatch, cell)
    assert refused is None, f"refused: {refused}"
    assert policy is not None and policy.route == "direct"
    assert _tokens(files, launch) == set(cell.tokens), (sorted(files), launch)
    assert _table_names(policy) == set(cell.names), policy
    for name in (
        "BENCHFLOW_LITELLM_MASTER_KEY",
        "BENCHFLOW_PROVIDER_MODEL",
        "BENCHFLOW_PROVIDER_NAME",
    ):
        assert name not in launch, name
