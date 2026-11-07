"""Credential files live for one connection, staged after proxy wiring.

FF-sw-cli-other C23-C30. A rollout drives real resolution, the real
credential writers and real proxy wiring (a fake proxy server); the sandbox
is a dict of files. The removal script itself runs on a real temp directory.
All values are dummies.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow.rollout as rollout_mod
from benchflow.agents import credentials as cred_mod
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    CredentialFile,
    register_agent,
)
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime
from benchflow.rollout import Role, Rollout, RolloutConfig
from tests.test_sweep_credential_policy_matrix import SETTINGS
from tests.test_sweep_launch_env_routes import HOST_CREDENTIALS

HOME = "/home/agent"
PROXY_FILE = "lifecycle-proxy-file"
DIRECT_FILE = "lifecycle-direct-file"
TWO_FILES = "lifecycle-two-files"
APP_FILE = "lifecycle-app-file"
WEB_CONFIG = "lifecycle-web-config"
AUTH_PATH = f"{HOME}/.custom/auth.json"
EXTRA_PATH = f"{HOME}/.custom/extra.json"
CONFIG_PATH = f"{HOME}/.custom/config.json"
CODEX_AUTH = f"{HOME}/.codex/auth.json"
KEY_PATH = f"{HOME}/.custom/key"
CLAUDE_LOGIN = f"{HOME}/.claude/.credentials.json"
HOST_LOGIN = "dummy-host-claude-login"
UPSTREAM = "dummy-upstream-openai"
EXPLICIT = "dummy-explicit-openai"


@pytest.fixture(autouse=True)
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    for name in (*HOST_CREDENTIALS, *SETTINGS):
        monkeypatch.delenv(name, raising=False)
    login = tmp_path / ".claude/.credentials.json"
    login.parent.mkdir(parents=True)
    login.write_text(json.dumps({"dummy": HOST_LOGIN}))
    register_agent(
        PROXY_FILE,
        "true",
        f"{PROXY_FILE} --acp",
        requires_env=["CUSTOM_API_KEY"],
        env_mapping={"BENCHFLOW_PROVIDER_API_KEY": "CUSTOM_API_KEY"},
        credential_files=[CredentialFile("{home}/.custom/key", "CUSTOM_API_KEY")],
    )
    register_agent(
        DIRECT_FILE,
        "true",
        f"{DIRECT_FILE} --acp",
        credential_files=[CredentialFile("{home}/.custom/key", "OPENAI_API_KEY")],
    )
    register_agent(
        TWO_FILES,
        "true",
        f"{TWO_FILES} --acp",
        credential_files=[
            CredentialFile("{home}/.custom/auth.json", "CUSTOM_AUTH_JSON"),
            CredentialFile("{home}/.custom/extra.json", "CUSTOM_EXTRA_JSON"),
        ],
    )
    register_agent(
        APP_FILE,
        "true",
        f"{APP_FILE} --acp",
        credential_files=[CredentialFile("/app/.custom/auth.json", "CUSTOM_AUTH_JSON")],
    )
    register_agent(
        WEB_CONFIG,
        "true",
        f"{WEB_CONFIG} --acp",
        credential_files=[
            CredentialFile("{home}/.custom/config.json", "CUSTOM_AUTH_JSON")
        ],
        disallow_web_tools_setup_cmd=(
            "sed -i 's/fetch/nofetch/' \"$BENCHFLOW_AGENT_HOME/.custom/config.json\""
        ),
    )
    yield
    for name in (PROXY_FILE, DIRECT_FILE, TWO_FILES, APP_FILE, WEB_CONFIG):
        for registry in (AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH):
            registry.pop(name, None)


class Sandbox:
    """Files written into the sandbox, and the order of lifecycle events."""

    def __init__(
        self, monkeypatch, *, fail_removal: bool = False, fail_upload: str = ""
    ):
        self.files: dict[str, str] = {}
        self.events: list[tuple[str, object]] = []
        self.launches: list[dict[str, str]] = []

        async def upload(env, path, content, *, owner=None):
            if fail_upload and path == fail_upload:
                # E.g. a previous agent left a directory at this destination.
                raise RuntimeError("Agent credential installation failed")
            self.events.append(("upload", path))
            self.files[path] = content

        async def remove(env, sandbox_user, *rest):
            paths = rest[-1]  # (paths) now; (cred_home, paths) before C29
            self.events.append(("remove", tuple(paths)))
            if fail_removal:
                raise RuntimeError("Credential cleanup failed")
            for path in paths:
                self.files.pop(path, None)

        class _Server:
            base_url = "http://127.0.0.1:4995"
            trajectory = None

            async def is_running(self):
                return True

            async def stop(self):
                return None

        async def start(**kwargs):
            return _Server()

        monkeypatch.setattr(cred_mod, "upload_credential", upload)
        # raising=False: the fails-before tree has no lifecycle to patch.
        monkeypatch.setattr(
            rollout_mod, "remove_staged_credentials", remove, raising=False
        )
        monkeypatch.setattr(runtime_mod, "_start_host_litellm", start)

    def planes(self):
        async def quiesce(env, sandbox_user, **kwargs):
            self.events.append(("quiesce", sandbox_user))

        async def connect_acp(**kwargs):
            self.events.append(("launch", kwargs["agent"]))
            self.launches.append(dict(kwargs["agent_env"]))
            key = kwargs["agent_env"].get("OPENAI_API_KEY")
            if kwargs["agent"] == "codex-acp" and key:
                # What CODEX_ACP_BUILTIN_LAUNCH writes before exec'ing codex.
                self.files[CODEX_AUTH] = json.dumps({"OPENAI_API_KEY": key})

            return (AsyncMock(), AsyncMock(), AsyncMock(), kwargs["agent"])

        async def web_policy(env, agent, agent_cfg, home, *, disallow, **kwargs):
            if disallow and getattr(agent_cfg, "disallow_web_tools_setup_cmd", ""):
                # The setup edits files in place: they must exist by now.
                self.events.append(("web", tuple(sorted(self.files))))

        async def ensure(**kwargs):
            for name in ("sandbox", "sandbox_setup_timeout", "prior_runtimes"):
                kwargs.pop(name, None)
            for name in (
                "required_skill_names",
                "live_trajectory_path",
                "force_sandbox_local",
                "disallow_web_tools",
                "install_layer_bypass_reason",
            ):
                kwargs.pop(name, None)
            return await ensure_litellm_runtime(**kwargs)

        return SimpleNamespace(
            agent_launch=lambda agent, **kwargs: agent,
            resolve_agent_env=resolve_agent_env,
            ensure_litellm_runtime=ensure,
            install_agent=AsyncMock(
                side_effect=lambda env, agent, *a, **k: AGENTS[agent]
            ),
            write_credential_files=cred_mod.write_credential_files,
            upload_subscription_auth=cred_mod.upload_subscription_auth,
            apply_web_tool_policy=web_policy,
            quiesce_agent=quiesce,
            connect_acp=connect_acp,
        )


def _rollout(
    tmp_path,
    sandbox: Sandbox,
    agent: str,
    model: str | None,
    passed,
    *,
    no_web: bool = False,
):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent=agent,
            model=model,
            environment="docker",
            agent_env=dict(passed),
            skip_agent_install=True,
        )
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "lifecycle"
    rollout._agent_launch = agent
    rollout._agent_cwd = "/app"
    rollout._env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
        )
    )
    rollout._task = SimpleNamespace(config=SimpleNamespace())
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None
    # The fake ACP session reports no usage or trajectory at disconnect.
    rollout._collect_native_acp_usage = lambda: None
    rollout._capture_partial_acp_trajectory = lambda: None
    rollout._planes = sandbox.planes()
    rollout._disallow_web_tools = no_web
    if agent != "oracle":
        # What install_agent() resolved for the primary agent.
        rollout._agent_env = resolve_agent_env(agent, model, dict(passed))
        rollout._agent_cfg = AGENTS[agent]
    return rollout


@pytest.fixture(autouse=True)
def no_kept_services(monkeypatch):
    monkeypatch.setattr(
        "benchflow.task.keep_services.trial_keep", lambda *a: None, raising=False
    )


async def _connect(tmp_path, sandbox, entry, agent, model, passed):
    if entry == "primary":
        rollout = _rollout(tmp_path, sandbox, agent, model, passed)
        await rollout.connect()
    else:
        rollout = _rollout(tmp_path, sandbox, "oracle", None, {})
        await rollout.connect_as(
            Role(name="solver", agent=agent, model=model, env=dict(passed))
        )
    return rollout


ROUTES = {
    # C24: the proxy key a custom agent reads from its file.
    "proxy": (PROXY_FILE, "openai/gpt-5.5", {"OPENAI_API_KEY": UPSTREAM}),
    # C25: a file-only direct login in a known credential slot.
    "direct": (DIRECT_FILE, None, {"OPENAI_API_KEY": EXPLICIT}),
    # The host login file of a native route.
    "native": ("claude-agent-acp", "claude-sonnet-4-6", {}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("entry", ["primary", "connect_as"])
async def test_connection_stages_exactly_its_admitted_files(
    tmp_path, monkeypatch, entry, route
):
    """Primary and connect_as stage the same files on every route."""
    sandbox = Sandbox(monkeypatch)
    agent, model, passed = ROUTES[route]
    await _connect(tmp_path, sandbox, entry, agent, model, passed)
    launched = sandbox.launches[-1]
    if route == "proxy":
        master = launched["BENCHFLOW_LITELLM_MASTER_KEY"]
        assert sandbox.files == {KEY_PATH: master}
        assert UPSTREAM not in json.dumps(launched)
    elif route == "direct":
        assert sandbox.files == {KEY_PATH: EXPLICIT}
        assert launched["OPENAI_API_KEY"] == EXPLICIT
    else:
        assert sandbox.files == {CLAUDE_LOGIN: json.dumps({"dummy": HOST_LOGIN})}


@pytest.mark.asyncio
async def test_c23_role_change_removes_the_previous_login(tmp_path, monkeypatch):
    """Claude's host login is gone, its agent stopped, before OpenHands runs."""
    sandbox = Sandbox(monkeypatch)
    rollout = _rollout(tmp_path, sandbox, "claude-agent-acp", "claude-sonnet-4-6", {})
    await rollout.connect()
    assert CLAUDE_LOGIN in sandbox.files
    await rollout.disconnect()
    await rollout.connect_as(
        Role(
            name="reviewer",
            agent="openhands",
            model="openai/gpt-5.5",
            env={"OPENAI_API_KEY": UPSTREAM},
        )
    )
    assert CLAUDE_LOGIN not in sandbox.files
    assert HOST_LOGIN not in json.dumps(sandbox.files)
    removed = [i for i, e in enumerate(sandbox.events) if e[0] == "remove"]
    launches = [i for i, e in enumerate(sandbox.events) if e[0] == "launch"]
    assert removed and removed[0] < launches[1]


@pytest.mark.asyncio
async def test_c23_another_agent_quiesces_before_removal(tmp_path, monkeypatch):
    """Without a disconnect, the next agent first stops the previous one."""
    sandbox = Sandbox(monkeypatch)
    rollout = _rollout(tmp_path, sandbox, "claude-agent-acp", "claude-sonnet-4-6", {})
    await rollout.connect()
    await rollout.connect_as(
        Role(
            name="reviewer",
            agent="openhands",
            model="openai/gpt-5.5",
            env={"OPENAI_API_KEY": UPSTREAM},
        )
    )
    kinds = [k for k, _ in sandbox.events if k in ("launch", "quiesce", "remove")]
    assert kinds == ["launch", "quiesce", "remove", "launch"]
    assert CLAUDE_LOGIN not in sandbox.files


@pytest.mark.asyncio
async def test_c23_failed_removal_admits_no_next_role(tmp_path, monkeypatch):
    sandbox = Sandbox(monkeypatch, fail_removal=True)
    rollout = _rollout(tmp_path, sandbox, "claude-agent-acp", "claude-sonnet-4-6", {})
    await rollout.connect()
    await rollout.disconnect()  # logs the failure; the login stays pending
    with pytest.raises(RuntimeError, match="Credential cleanup failed"):
        await rollout.connect_as(
            Role(
                name="reviewer",
                agent="openhands",
                model="openai/gpt-5.5",
                env={"OPENAI_API_KEY": UPSTREAM},
            )
        )
    assert [kind for kind, _ in sandbox.events].count("launch") == 1


# ── The removal script, on a real directory ──


async def remove_staged_credentials(env, sandbox_user, home, paths):
    """The removal helper; the fails-before tree also took the home."""
    import inspect

    from benchflow.agents.credentials import remove_staged_credentials as remove

    if "cred_home" in inspect.signature(remove).parameters:
        return await remove(env, sandbox_user, home, paths)
    return await remove(env, sandbox_user, paths)


class LocalSandbox:
    async def exec(self, command, user=None, timeout_sec=None, **kwargs):
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(return_code=done.returncode)


@pytest.mark.asyncio
async def test_removal_script_removes_files_and_skips_missing_parents(tmp_path):
    home = tmp_path / "home"
    login = home / ".claude/.credentials.json"
    login.parent.mkdir(parents=True)
    login.write_text("dummy")
    await remove_staged_credentials(
        LocalSandbox(),
        None,
        str(home),
        [str(login), str(home / ".absent/dir/auth.json")],
    )
    assert not login.exists()
    assert login.parent.is_dir()


@pytest.mark.asyncio
async def test_removal_script_refuses_a_linked_parent(tmp_path):
    home = tmp_path / "home"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "auth.json").write_text("dummy")
    home.mkdir()
    (home / ".codex").symlink_to(elsewhere)
    with pytest.raises(RuntimeError, match="Credential cleanup failed"):
        await remove_staged_credentials(
            LocalSandbox(), None, str(home), [str(home / ".codex/auth.json")]
        )
    assert (elsewhere / "auth.json").exists()


@pytest.mark.asyncio
async def test_c29_removal_accepts_a_declared_destination_outside_the_home(tmp_path):
    """A CredentialFile may name any container path (e.g. /app/...)."""
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "app/.custom/auth.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("dummy")
    await remove_staged_credentials(LocalSandbox(), None, str(home), [str(outside)])
    assert not outside.exists()


@pytest.mark.asyncio
async def test_removal_refuses_a_relative_path(tmp_path):
    with pytest.raises(RuntimeError, match="Credential cleanup refused"):
        await remove_staged_credentials(
            LocalSandbox(), None, str(tmp_path), ["relative/auth.json"]
        )


# ── C26-C30 through the rollout ──

OPENHANDS_ROLE = Role(
    name="reviewer",
    agent="openhands",
    model="openai/gpt-5.5",
    env={"OPENAI_API_KEY": UPSTREAM},
)


@pytest.mark.asyncio
async def test_c26_ordinary_disconnect_still_quiesces_before_the_next_role(
    tmp_path, monkeypatch
):
    """OpenHands, disconnect, then native Claude: OpenHands' survivors are
    stopped before Claude's login is written."""
    sandbox = Sandbox(monkeypatch)
    rollout = _rollout(tmp_path, sandbox, "oracle", None, {})
    await rollout.connect_as(OPENHANDS_ROLE)
    await rollout.disconnect()
    await rollout.connect_as(
        Role(name="solver", agent="claude-agent-acp", model="claude-sonnet-4-6")
    )
    order = [e for e in sandbox.events if e[0] in ("quiesce", "upload")]
    assert ("quiesce", "agent") in order
    assert order.index(("quiesce", "agent")) < order.index(("upload", CLAUDE_LOGIN))


@pytest.mark.asyncio
async def test_c26_same_role_reconnect_quiesces_before_cleanup(tmp_path, monkeypatch):
    """Guards 7516a36a: identical roles still must stop credential writers."""
    sandbox = Sandbox(monkeypatch)
    rollout = _rollout(tmp_path, sandbox, "oracle", None, {})
    for _ in range(2):
        await rollout.connect_as(OPENHANDS_ROLE)
        await rollout.disconnect()
    assert len([e for e in sandbox.events if e[0] == "quiesce"]) == 2


@pytest.mark.asyncio
async def test_c27_partial_staging_is_still_removed(tmp_path, monkeypatch):
    sandbox = Sandbox(monkeypatch, fail_upload=EXTRA_PATH)
    rollout = _rollout(tmp_path, sandbox, "oracle", None, {})
    role = Role(
        name="solver",
        agent=TWO_FILES,
        model=None,
        env={"CUSTOM_AUTH_JSON": '{"a": 1}', "CUSTOM_EXTRA_JSON": '{"b": 2}'},
    )
    with pytest.raises(RuntimeError, match="installation failed"):
        await rollout.connect_as(role)
    assert AUTH_PATH in sandbox.files
    await rollout.disconnect()
    await rollout.connect_as(OPENHANDS_ROLE)
    assert AUTH_PATH not in sandbox.files


@pytest.mark.asyncio
async def test_c28_launcher_written_codex_key_is_removed(tmp_path, monkeypatch):
    async def admitted(env, **kwargs):
        return kwargs["agent_env"]

    monkeypatch.setattr("benchflow.rollout.enforce_codex_apps_policy", admitted)
    sandbox = Sandbox(monkeypatch)
    rollout = _rollout(
        tmp_path, sandbox, "codex-acp", None, {"OPENAI_API_KEY": EXPLICIT}
    )
    await rollout.connect()
    assert EXPLICIT in sandbox.files[CODEX_AUTH]
    await rollout.disconnect()
    await rollout.connect_as(OPENHANDS_ROLE)
    assert CODEX_AUTH not in sandbox.files
    assert EXPLICIT not in json.dumps(sandbox.files)


@pytest.mark.asyncio
async def test_c29_destination_outside_the_home_is_staged_and_removed(
    tmp_path, monkeypatch
):
    sandbox = Sandbox(monkeypatch)
    rollout = _rollout(
        tmp_path, sandbox, APP_FILE, None, {"CUSTOM_AUTH_JSON": '{"a": 1}'}
    )
    for _ in range(2):
        await rollout.connect()
        assert "/app/.custom/auth.json" in sandbox.files
        await rollout.disconnect()
        assert "/app/.custom/auth.json" not in sandbox.files


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["primary", "connect_as"])
async def test_c30_web_policy_setup_runs_after_its_credential_file(
    tmp_path, monkeypatch, entry
):
    sandbox = Sandbox(monkeypatch)
    passed = {"CUSTOM_AUTH_JSON": '{"fetch": true}'}
    if entry == "primary":
        rollout = _rollout(tmp_path, sandbox, WEB_CONFIG, None, passed, no_web=True)
        await rollout.connect()
    else:
        rollout = _rollout(tmp_path, sandbox, "oracle", None, {}, no_web=True)
        # S3 907cf44a audits a restricted role before launch. Exercise its
        # protected-shell/receipt interface with the existing image fixture.
        from tests.test_firewall_power import FixtureEnv

        rollout._env = FixtureEnv("cap")
        await rollout.connect_as(
            Role(name="solver", agent=WEB_CONFIG, model=None, env=passed)
        )
        record = json.loads((tmp_path / "firewall-power.json").read_text())
        assert record["state"] == "disarmed"
    web = [files for kind, files in sandbox.events if kind == "web"]
    assert web and CONFIG_PATH in web[-1]
