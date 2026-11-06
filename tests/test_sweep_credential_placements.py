"""Guards FF-sw-cli-other-C14 and C15: every credential placement follows the route.

For each route (Vertex, Bedrock, Azure and plain API-key proxy routes, native
logins, a host login file, no-model and multi-part custom agents) this lists
everything that reaches the sandbox: each file written into the agent's home
and each variable the agent process is launched with. Only the route's own
selected agent credential may appear there; on a proxy route that is nothing
but the proxy key, and the proxy's credentials (Vertex ADC) stay in the proxy's
own storage. The host exports every known credential with a distinct dummy
value; nothing real is read and nothing starts.
"""

from __future__ import annotations

import json
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
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.providers.runtime import ensure_litellm_runtime
from tests.test_litellm_hardening import _FakeSandbox
from tests.test_sweep_launch_env_routes import HOST_CREDENTIALS, _dummy

try:  # the rollout's rule for copying host login files, at this tree
    from benchflow.agents.launch_env import credential_policy

    def stages_host_login(agent, model, env):
        return credential_policy(agent, model, env).host_login

except ImportError:  # older trees
    try:
        from benchflow.agents.launch_env import stages_host_login
    except ImportError:

        def stages_host_login(agent, model, env):
            return bool(env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"))


A = "dummy-selected-a"
HOME = "/home/agent"
ADC = json.dumps({"type": "authorized_user", "refresh_token": "dummy-host-adc"})
HOST_LOGIN = json.dumps({"dummy": "dummy-host-login-file"})
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
    "CLIENT_ID",
    "CLIENT_SECRET",
    "XIAOMI_BASE_URL",
)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host with an ADC login and, on request, a Claude login file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in (*HOST_CREDENTIALS, *SETTINGS):
        monkeypatch.delenv(name, raising=False)
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text(ADC)

    def export(*, without=(), login_file=False, settings=None):
        for name in HOST_CREDENTIALS:
            if name not in without:
                monkeypatch.setenv(name, _dummy(name))
        for name, value in (settings or {}).items():
            monkeypatch.setenv(name, value)
        if login_file:
            login = tmp_path / ".claude/.credentials.json"
            login.parent.mkdir(parents=True)
            login.write_text(HOST_LOGIN)

    return export


class _Launched(Exception):
    pass


async def _sandbox_view(monkeypatch, tmp_path, agent, model, explicit):
    """(files written into the sandbox, the agent's launch variables)."""
    files: dict[str, str] = {}

    async def upload(sandbox, path, content, *, owner=None):
        files[path] = content

    monkeypatch.setattr(cred_mod, "upload_credential", upload)
    resolved = resolve_agent_env(agent, model, dict(explicit))
    # Install time, before any proxy: the order rollout.install_agent uses.
    staged = dict(resolved)
    await cred_mod.write_credential_files(
        object(), agent, staged, AGENTS.get(agent), model, HOME
    )
    if stages_host_login(agent, model, staged):
        await cred_mod.upload_subscription_auth(object(), agent, HOME)

    class _Server:
        base_url = "http://127.0.0.1:4996"
        trajectory = None

        async def is_running(self):
            return True

        async def stop(self):
            return None

    async def fake_start(**kwargs):
        return _Server()

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    wired, _ = await ensure_litellm_runtime(
        agent=agent,
        agent_env=staged,
        model=model,
        runtime=None,
        environment="docker",
        session_id="c14",
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
            agent=agent,
            agent_launch="agent-under-test",
            agent_env=wired,
            sandbox_user="agent",
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )
    return files, seen["env"]


def _dummies(files: dict[str, str], launch: dict[str, str]) -> dict[str, list[str]]:
    """Every dummy credential token visible in the sandbox, with where it is."""
    found: dict[str, list[str]] = {}
    places = [(f"file {p}", c) for p, c in files.items()]
    places += [(f"env {k}", v) for k, v in launch.items()]
    for where, text in places:
        for token in text.replace('"', " ").replace(",", " ").split():
            if token.startswith("dummy-"):
                found.setdefault(token.rstrip("}"), []).append(where)
    return found


PROXY_ROUTES = [
    pytest.param(
        "openhands",
        "google-vertex/gemini-2.5-pro",
        {},
        {"GOOGLE_CLOUD_PROJECT": "project-a"},
        id="vertex-openhands",
    ),
    pytest.param(
        "claude-agent-acp",
        "anthropic-vertex/claude-sonnet-4-6",
        {},
        {"GOOGLE_CLOUD_PROJECT": "project-a", "GOOGLE_CLOUD_LOCATION": "global"},
        id="vertex-claude",
    ),
    pytest.param(
        "claude-agent-acp",
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {"AWS_BEARER_TOKEN_BEDROCK": A},
        {"AWS_REGION": "us-west-2"},
        id="bedrock-claude",
    ),
    pytest.param(
        "codex-acp",
        "azure-foundry-openai/gpt-5.5",
        {"AZURE_API_KEY": A},
        {"AZURE_RESOURCE": "resource-a"},
        id="azure-codex",
    ),
    pytest.param(
        "codex-acp", "openai/gpt-5.5", {"OPENAI_API_KEY": A}, {}, id="openai-codex"
    ),
    pytest.param(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"ANTHROPIC_API_KEY": A},
        {},
        id="anthropic-claude",
    ),
    pytest.param(
        "mimo",
        "xiaomi/mimo-v2.6-flash",
        {"XIAOMI_API_KEY": A},
        {"XIAOMI_BASE_URL": "https://xiaomi.example.invalid/v1"},
        id="xiaomi-mimo",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("agent", "model", "explicit", "settings"), PROXY_ROUTES)
async def test_proxy_route_places_no_credential_in_the_sandbox(
    host, monkeypatch, tmp_path, agent, model, explicit, settings
):
    """C14: no upstream key, ADC, login or login file reaches a proxy-route agent."""
    host(settings=settings, login_file=True)
    files, launch = await _sandbox_view(monkeypatch, tmp_path, agent, model, explicit)
    assert _dummies(files, launch) == {}
    assert not any("application_default_credentials" in p for p in files), files
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in launch
    assert launch["BENCHFLOW_LITELLM_MASTER_KEY"].startswith("sk-benchflow-")


# (agent, model, explicit login, host names left unset because they would select
#  a provider route, the selected credential's file path if one is staged)
NATIVE_ROUTES = [
    pytest.param(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"CLAUDE_CODE_OAUTH_TOKEN": A},
        ("ANTHROPIC_API_KEY",),
        None,
        id="claude-oauth",
    ),
    pytest.param(
        "codex-acp",
        "gpt-5.5",
        {"CODEX_ACCESS_TOKEN": A},
        ("OPENAI_API_KEY", "CODEX_API_KEY"),
        None,
        id="codex-access-token",
    ),
    pytest.param(
        "codex-acp",
        "openai/gpt-5.5",
        {"CODEX_AUTH_JSON": json.dumps({"dummy": A})},
        ("OPENAI_API_KEY", "CODEX_API_KEY"),
        f"{HOME}/.codex/auth.json",
        id="codex-auth-json",
    ),
    pytest.param(
        "muse-code",
        "muse-spark",
        {"MUSE_AUTH_JSON": json.dumps({"dummy": A})},
        (),
        f"{HOME}/.config/muse/auth.json",
        id="muse",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("agent", "model", "explicit", "unset", "file"), NATIVE_ROUTES)
async def test_native_route_places_only_its_selected_login(
    host, monkeypatch, tmp_path, agent, model, explicit, unset, file
):
    """C14/C10: a native route's sandbox holds its selected login and nothing else."""
    host(without=unset, login_file=True)
    files, launch = await _sandbox_view(monkeypatch, tmp_path, agent, model, explicit)
    assert set(_dummies(files, launch)) == {A}
    if file is not None:
        assert A in files[file]
    assert not any("application_default_credentials" in p for p in files), files


@pytest.mark.asyncio
async def test_host_login_file_is_the_only_placement_of_a_file_login(
    host, monkeypatch, tmp_path
):
    """C14: a host-login route copies that file and no credential variable."""
    host(without=("ANTHROPIC_API_KEY", *_claude_logins()), login_file=True)
    files, launch = await _sandbox_view(
        monkeypatch, tmp_path, "claude-agent-acp", "claude-sonnet-4-6", {}
    )
    assert set(_dummies(files, launch)) == {"dummy-host-login-file"}
    assert [p for p, c in files.items() if "dummy-host-login-file" in c] == [
        f"{HOME}/.claude/.credentials.json"
    ]


def _claude_logins():
    return ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN")


@pytest.fixture
def two_part_agent():
    register_agent(
        "c15-two-part",
        "true",
        "c15-two-part --acp",
        protocol="acp",
        requires_env=["CLIENT_ID", "CLIENT_SECRET"],
    )
    yield "c15-two-part"
    for registry in (AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH):
        registry.pop("c15-two-part", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "openai/gpt-5.5"], ids=["no-model", "proxy"])
async def test_jointly_required_parts_reach_the_agent_together(
    host, monkeypatch, tmp_path, two_part_agent, model
):
    """C15: both parts of a declared two-part credential are one selection."""
    host()
    explicit = {"CLIENT_ID": "dummy-client-a", "CLIENT_SECRET": "dummy-secret-a"}
    if model is not None:
        explicit["OPENAI_API_KEY"] = A
    files, launch = await _sandbox_view(
        monkeypatch, tmp_path, two_part_agent, model, explicit
    )
    assert launch["CLIENT_ID"] == "dummy-client-a"
    assert launch["CLIENT_SECRET"] == "dummy-secret-a"
    assert set(_dummies(files, launch)) == {"dummy-client-a", "dummy-secret-a"}


@pytest.mark.asyncio
async def test_sandbox_proxy_keeps_vertex_adc_in_its_own_storage():
    """C14 control: a sandbox-local proxy gets ADC only in its mode-600 runtime dir."""
    route = resolve_litellm_route(
        "google-vertex/gemini-2.5-pro",
        {"GOOGLE_CLOUD_PROJECT": "project-a", "GOOGLE_CLOUD_LOCATION": "global"},
    )
    sandbox = _FakeSandbox()
    await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-benchflow-dummy",
        agent_env={"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC},
        session_id="c14",
        agent_name="openhands",
    )
    holders = [p for p, c in sandbox.uploaded.items() if "dummy-host-adc" in c]
    assert len(holders) == 1
    (adc_path,) = holders
    assert adc_path.startswith(runtime_mod.LITELLM_SANDBOX_ROOT + "/")
    assert sandbox.uploaded_modes[adc_path] == "600"
    assert not any(p.startswith("/home/") for p in sandbox.uploaded)


@pytest.mark.asyncio
async def test_gemini_vertex_native_route_places_only_the_adc_file(
    host, monkeypatch, tmp_path
):
    """Gemini CLI on Vertex is a native route whose only login is the ADC file."""
    host(settings={"GOOGLE_CLOUD_PROJECT": "project-a"}, login_file=True)
    gemini_login = tmp_path / ".gemini/oauth_creds.json"
    gemini_login.parent.mkdir(parents=True)
    gemini_login.write_text(json.dumps({"dummy": "dummy-host-gemini-login"}))
    model = "google-vertex/gemini-2.5-pro"
    files, launch = await _sandbox_view(monkeypatch, tmp_path, "gemini", model, {})
    adc_path = f"{HOME}/.config/gcloud/application_default_credentials.json"
    assert set(_dummies(files, launch)) == {"dummy-host-adc"}
    assert [p for p, c in files.items() if "dummy-host-adc" in c] == [adc_path]
    assert sorted(files) == [adc_path, f"{HOME}/.gemini/settings.json"]
    assert json.loads(files[f"{HOME}/.gemini/settings.json"]) == {
        "security": {"auth": {"selectedType": "vertex-ai"}}
    }
    assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in launch
    assert launch["GOOGLE_APPLICATION_CREDENTIALS"] == adc_path
    assert launch["GOOGLE_CLOUD_PROJECT"] == "project-a"
    assert "BENCHFLOW_LITELLM_MASTER_KEY" not in launch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "model"),
    [
        ("openhands", "google-vertex/gemini-2.5-pro"),
        ("opencode", "google-vertex/gemini-2.5-pro"),
        ("claude-agent-acp", "anthropic-vertex/claude-sonnet-4-6"),
    ],
)
async def test_other_vertex_runs_stay_on_the_proxy_without_adc(
    host, monkeypatch, tmp_path, agent, model
):
    """Only Gemini CLI on google-vertex/ is native; the rest proxy."""
    host(settings={"GOOGLE_CLOUD_PROJECT": "project-a"})
    files, launch = await _sandbox_view(monkeypatch, tmp_path, agent, model, {})
    assert _dummies(files, launch) == {}
    assert not any("application_default_credentials" in p for p in files), files
    assert not any(p.endswith(".gemini/settings.json") for p in files), files
    assert launch["BENCHFLOW_LITELLM_MASTER_KEY"].startswith("sk-benchflow-")
