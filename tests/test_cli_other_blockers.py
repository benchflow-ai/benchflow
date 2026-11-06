"""Guards PY8 C23/C26-C33 at 7516a36a; only dummy login markers.

Real shell checks run in the lane's mount-free throwaway container. Lifecycle
checks drive both public connection entry points with the existing fake backend.
"""

import json
import os
import stat
import subprocess
from types import SimpleNamespace

import pytest

import benchflow.rollout as rollout_mod
from benchflow.agents import credentials
from benchflow.agents.registry import AGENTS
from benchflow.rollout import Role
from tests.test_sweep_credential_lifecycle import (
    EXPLICIT,
    HOME,
    OPENHANDS_ROLE,
    Sandbox,
    _rollout,
    host,  # noqa: F401 -- isolate host auth and register dummy agents
    no_kept_services,  # noqa: F401
)


@pytest.mark.parametrize("entry", ["primary", "role"])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_c31_barrier_precedes_every_credential_consumer(
    tmp_path, monkeypatch, entry, disconnect
):
    """7516a36a ran Codex probes and proxy wiring before the role barrier."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "codex-acp", None, {"OPENAI_API_KEY": EXPLICIT})
    await trial.connect_as(OPENHANDS_ROLE)
    if disconnect:
        await trial.disconnect()

    original = trial._planes.ensure_litellm_runtime

    async def ensure(**kwargs):
        sandbox.events.append(("credential-consumer", "runtime"))
        return await original(**kwargs)

    async def admit(env, **kwargs):
        sandbox.events.append(("credential-consumer", "admission"))
        return kwargs["agent_env"]

    trial._planes.ensure_litellm_runtime = ensure
    monkeypatch.setattr(rollout_mod, "enforce_codex_apps_policy", admit)
    if entry == "primary":
        await trial.connect()
    else:
        await trial.connect_as(
            Role(
                name="next",
                agent="codex-acp",
                model=None,
                env={"OPENAI_API_KEY": EXPLICIT},
            )
        )
    events = [kind for kind, _ in sandbox.events]
    assert events.index("quiesce") < events.index("credential-consumer")
    if "remove" in events:
        assert events.index("quiesce") < events.index("remove")


async def test_c26_same_role_reconnect_still_stops_credential_writers(
    tmp_path, monkeypatch
):
    """7516a36a skipped the barrier for identical role/environment tuples."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "claude-agent-acp", "claude-sonnet-4-6", {})
    await trial.connect()
    await trial.disconnect()
    await trial.connect()
    events = [kind for kind, _ in sandbox.events]
    assert events.index("quiesce") < events.index("remove")


async def test_c26_failed_quiescence_keeps_ledger_and_blocks_all_next_consumers(
    tmp_path, monkeypatch
):
    """7516a36a cleaned before proving the previous connection was stopped."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "claude-agent-acp", "claude-sonnet-4-6", {})
    await trial.connect()
    before = dict(sandbox.files)

    async def refuse(*args, **kwargs):
        raise RuntimeError("dummy quiescence refusal")

    trial._planes.quiesce_agent = refuse
    with pytest.raises(RuntimeError, match="dummy quiescence refusal"):
        await trial.disconnect()
    assert sandbox.files == before
    with pytest.raises(RuntimeError, match="dummy quiescence refusal"):
        await trial.connect_as(OPENHANDS_ROLE)
    assert len(sandbox.launches) == 1


@pytest.mark.parametrize(
    "agent,paths",
    [
        ("codex-acp", ["{home}/.codex/auth.json"]),
        ("openhands", ["{home}/.openhands/agent_settings.json"]),
        ("pi-acp", ["{home}/.pi/agent/models.json"]),
        ("opencode", ["{home}/.config/opencode/opencode.json"]),
        (
            "mimo",
            ["{home}/.config/mimocode/mimocode.json", "{workspace}/mimocode.json"],
        ),
    ],
)
async def test_c32_every_builtin_launcher_destination_is_recorded_before_launch(
    tmp_path, monkeypatch, agent, paths
):
    """7516a36a inventoried only Codex, missing the other bundled writers."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env.exec = _local_unused_exec
    # Activate every conditional writer: inactive destinations are task-owned
    # and covered by CONF-3's preservation matrix.
    await trial._stage_connection_credentials(
        agent,
        {
            "OPENAI_API_KEY": "dummy-login",
            "OPENAI_BASE_URL": "http://dummy.invalid",
            "BENCHFLOW_LITELLM_MODEL_ALIAS": "dummy-model",
            "BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://dummy.invalid",
        },
        AGENTS[agent],
        None,
        key=(agent, None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    expected = [p.format(home=HOME, workspace="/app") for p in paths]
    assert set(expected).issubset(trial._credential_ledger)
    await trial._remove_ledger()
    assert set(expected).issubset(sandbox.events[-1][1])


async def _local_unused_exec(*args, **kwargs):
    # Ledger test only: real owner-side scripts are exercised below.
    return SimpleNamespace(return_code=0, stdout="", stderr="")


class LocalSandbox:
    async def exec(self, command, **kwargs):
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


@pytest.mark.parametrize(
    "kind", ["symlink", "dangling", "hardlink", "regular", "absent"]
)
async def test_c33_launcher_preparation_never_writes_through_an_alias(
    tmp_path, monkeypatch, kind
):
    """7516a36a allowed the next Codex launch to write beyond its ledger."""
    home = tmp_path / "dummy-home"
    auth = home / ".codex/auth.json"
    auth.parent.mkdir(parents=True)
    saved = tmp_path / "dummy-other-file.json"
    if kind != "dangling":
        saved.write_text("{}")
    if kind in {"symlink", "dangling"}:
        auth.symlink_to(saved)
    elif kind == "hardlink":
        os.link(saved, auth)
    elif kind == "regular":
        auth.write_text("{}")
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env = LocalSandbox()
    # Test the real preparation at a harmless temp path, not the fixture's home.
    cfg = SimpleNamespace(credential_files=[], launcher_credential_files=[str(auth)])
    await trial._stage_connection_credentials(
        "codex-acp",
        {},
        cfg,
        None,
        key=("codex-acp", None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    # 092eb18ad scopes the writer in a subshell followed by && exec. Run the
    # complete writer/control flow and stub only the external agent executable.
    command = AGENTS["codex-acp"].launch_cmd.replace(
        "/opt/benchflow/bin/codex-acp", "true", 1
    )
    subprocess.run(
        ["sh", "-c", command],
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "BENCHFLOW_AGENT_HOME": str(home),
            "OPENAI_API_KEY": "dummy-codex-login",
        },
    )
    assert not saved.exists() if kind == "dangling" else saved.read_text() == "{}"
    assert not auth.is_symlink()
    assert json.loads(auth.read_text())["OPENAI_API_KEY"] == "dummy-codex-login"
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600
    await credentials.remove_staged_credentials(
        LocalSandbox(), None, trial._credential_ledger
    )
    assert not auth.exists()


async def test_c33_linked_parent_refused_before_any_launcher_can_write(
    tmp_path, monkeypatch
):
    """7516a36a checked parents only at cleanup, after credential exposure."""
    home = tmp_path / "dummy-home"
    other = tmp_path / "dummy-other"
    home.mkdir()
    other.mkdir()
    (home / ".codex").symlink_to(other)
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env = LocalSandbox()
    cfg = SimpleNamespace(
        credential_files=[], launcher_credential_files=[str(home / ".codex/auth.json")]
    )
    with pytest.raises(RuntimeError, match="credential"):
        await trial._stage_connection_credentials(
            "codex-acp",
            {},
            cfg,
            None,
            key=("codex-acp", None, ()),
            disallow_web_tools=False,
            disallow_hosted_search=False,
        )
    assert not list(other.iterdir())


@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_c31_apps_probes_have_no_credential_environment(tmp_path, harness):
    """7516a36a passed every resolved credential to authentication-free probes."""
    from unittest.mock import AsyncMock

    from benchflow.agents.codex_connector_policy import enforce_codex_apps_policy
    from benchflow.agents.registry import pinned_npm_package
    from tests.test_codex_connector_policy import (
        APPS_OFF,
        INSTALLED,
        admission,
        args,
        response,
    )
    from tests.test_sweep_launch_env_routes import HOST_CREDENTIALS

    replies = (
        admission()
        if harness == "acp"
        else [response("1000"), response("codex-cli " + pinned_npm_package("codex")[1])]
    )
    env = SimpleNamespace(exec=AsyncMock(side_effect=[*replies, INSTALLED, APPS_OFF]))
    passed = {name: "dummy-probe-login" for name in HOST_CREDENTIALS}
    passed.update(
        CUSTOM_AUTH_JSON="dummy-custom-login", CODEX_CONFIG='{"features":{"apps":true}}'
    )
    updated = await enforce_codex_apps_policy(
        env, **{**args(tmp_path), "agent_env": passed, "harness": harness}
    )
    for call in env.exec.await_args_list:
        assert set(call.kwargs.get("env") or {}) <= {
            "CODEX_HOME",
            "BENCHFLOW_AGENT_HOME",
            "CODEX_PATH",
            "NODE_OPTIONS",
            "NODE_PATH",
        }
    assert updated["CUSTOM_AUTH_JSON"] == "dummy-custom-login"


@pytest.mark.parametrize("path_kind", ["home", "workspace", "absolute"])
@pytest.mark.parametrize("obstacle", ["directory", "fifo", "parent-link"])
async def test_partial_launcher_preparation_keeps_each_attempt_on_one_ledger(
    tmp_path, monkeypatch, path_kind, obstacle
):
    """Guards C27/C32 at 7516a36a: a later preparation failure cannot erase earlier obligations."""
    folder = tmp_path / path_kind
    folder.mkdir()
    first = folder / "first.json"
    first.write_text('{"dummy":"original"}')
    blocked = folder / "blocked.json"
    if obstacle == "directory":
        blocked.mkdir()
    elif obstacle == "fifo":
        os.mkfifo(blocked)
    else:
        linked = folder / "linked"
        linked.symlink_to(folder, target_is_directory=True)
        blocked = linked / "blocked.json"
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env = LocalSandbox()
    cfg = SimpleNamespace(
        credential_files=[], launcher_credential_files=[str(first), str(blocked)]
    )
    with pytest.raises(RuntimeError, match="credential"):
        await trial._stage_connection_credentials(
            "codex-acp",
            {},
            cfg,
            None,
            key=("codex-acp", None, ()),
            disallow_web_tools=False,
            disallow_hosted_search=False,
        )
    assert trial._credential_ledger == [str(first), str(blocked)]
    assert first.read_text() == '{"dummy":"original"}'
    assert not list(folder.glob(".benchflow-credential-*"))


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
async def test_c32_openhands_login_removed_for_every_backend(
    tmp_path, monkeypatch, backend
):
    """Guards C32 at 7516a36a through backend-independent rollout planes."""
    from unittest.mock import AsyncMock

    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._config.environment = backend
    trial._planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kw: (kw["agent_env"], None)
    )
    original = trial._planes.connect_acp
    path = f"{HOME}/.openhands/agent_settings.json"

    async def connect(**kwargs):
        if kwargs["agent"] == "openhands":
            # Faithful launcher placement, independent of the backend exec API.
            sandbox.files[path] = "dummy-openhands-login"
        return await original(**kwargs)

    trial._planes.connect_acp = connect
    await trial.connect_as(
        Role(
            name="first",
            agent="openhands",
            model=None,
            env={
                "LLM_MODEL": "openai/gpt-test",
                "LLM_API_KEY": "dummy-openhands-login",
            },
        )
    )
    assert path in sandbox.files
    await trial.disconnect()
    await trial.connect_as(Role(name="second", agent="claude-agent-acp", model=None))
    assert path not in sandbox.files
    kinds = [kind for kind, _ in sandbox.events]
    assert kinds.index("quiesce") < kinds.index("remove")


@pytest.mark.parametrize(
    "agent", ["codex-acp", "opencode", "mimo", "pi-acp", "openhands"]
)
async def test_c33_launcher_home_override_cannot_escape_managed_ledger(
    tmp_path, monkeypatch, agent
):
    """Guards C32/C33 at 7516a36a: launch and staging must use the same home."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env.exec = _local_unused_exec
    env = {
        "BENCHFLOW_AGENT_HOME": "/tmp/dummy-override",
        "CODEX_HOME": "/tmp/dummy-codex-override",
    }
    await trial._stage_connection_credentials(
        agent,
        env,
        AGENTS[agent],
        None,
        key=(agent, None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    assert env["BENCHFLOW_AGENT_HOME"] == HOME
    if agent == "codex-acp":
        assert env["CODEX_HOME"] == f"{HOME}/.codex"


async def test_c32_codex_provider_config_does_not_claim_the_image_config(
    tmp_path, monkeypatch
):
    """Guards ac3bedaa1: global provider overrides must not erase image config."""
    from benchflow.agents.codex_config import apply_codex_launch_config

    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env.exec = _local_unused_exec
    env = {
        "CODEX_CONFIG": json.dumps(
            {
                "model_provider": "dummy",
                "model_providers": {
                    "dummy": {
                        "base_url": "http://127.0.0.1:8991",
                        "http_headers": {"Authorization": "dummy-provider-login"},
                    }
                },
            }
        )
    }
    await trial._stage_connection_credentials(
        "codex-acp",
        env,
        AGENTS["codex-acp"],
        None,
        key=("codex-acp", None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    launched, _ = apply_codex_launch_config(
        "codex-acp", env, model=None, reasoning_effort=None
    )
    path = f"{HOME}/.codex/config.toml"
    assert "dummy-provider-login" in launched["BENCHFLOW_CODEX_HOME_CONFIG"]
    sandbox.files[path] = 'model_reasoning_effort = "high"'
    await trial._remove_ledger()
    assert sandbox.files[path] == 'model_reasoning_effort = "high"'


async def test_codex_non_provider_config_is_not_claimed_by_credential_cleanup(
    tmp_path, monkeypatch
):
    """Guards C32's fix: preserve config the credential launcher does not write."""
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "oracle", None, {})
    trial._env.exec = _local_unused_exec
    path = f"{HOME}/.codex/config.toml"
    sandbox.files[path] = 'model_reasoning_effort = "high"'
    await trial._stage_connection_credentials(
        "codex-acp",
        {},
        AGENTS["codex-acp"],
        None,
        key=("codex-acp", None, ()),
        disallow_web_tools=False,
        disallow_hosted_search=False,
    )
    await trial._remove_ledger()
    assert sandbox.files[path] == 'model_reasoning_effort = "high"'
