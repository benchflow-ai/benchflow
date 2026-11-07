"""Guards CONF-R3-1, the inline-login reconnect regression at 5e77deca.

Exercise real file cleanup and the shipped Codex writer with dummy logins.
Only transfer, process quiescence and the final third-party client are faked;
backend parameters exercise the common connection plane, not cloud services.
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow.rollout as rollout_mod
from benchflow.agents import credentials
from benchflow.agents.env import CREDENTIAL_POLICY_MARKER, resolve_agent_env
from benchflow.agents.launch_env import agent_launch_env
from benchflow.agents.registry import AGENTS, CODEX_ACP_BUILTIN_LAUNCH
from benchflow.rollout import Role
from tests.test_cli_other_confirm3 import LocalSandbox
from tests.test_sweep_credential_lifecycle import (
    Sandbox,
    _rollout,
    host,  # noqa: F401
    no_kept_services,  # noqa: F401
)

INLINE = json.dumps({"tokens": {"access_token": "dummy-subscription-B"}})
OTHER_INLINE = json.dumps({"tokens": {"access_token": "dummy-subscription-A"}})
API_KEY = "dummy-api-A"
BACKENDS = ["docker", "remote-docker", "daytona", "modal"]


def file_trial(tmp_path, monkeypatch, backend, passed):
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "codex-acp", None, passed)
    trial._config.environment = backend
    home_dir = tmp_path / "login-home"
    home_dir.mkdir()
    events = []
    logins = []
    local = LocalSandbox()

    class ConnectionSandbox:
        async def exec(self, command, **kwargs):
            # No persistent dummy client exists. Never run process-pattern
            # cleanup against the pytest process that owns this fixture.
            if command.startswith("pkill "):
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            return await local.exec(command, **kwargs)

    trial._env = ConnectionSandbox()

    def pin(_agent, values):
        values.update(
            HOME=str(home_dir),
            BENCHFLOW_AGENT_HOME=str(home_dir),
            CODEX_HOME=str(home_dir / ".codex"),
        )
        return str(home_dir)

    async def upload(env, path, content, *, owner=None):
        assert path in trial._credential_ledger
        events.append("upload")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    async def remove(env, user, paths):
        events.append("clean")
        await credentials.remove_staged_credentials(env, user, paths)

    async def quiesce(*args, **kwargs):
        events.append("quiesce")

    async def admitted(env, **kwargs):
        return kwargs["agent_env"]

    async def connect_acp(**kwargs):
        events.append("launch")
        launch = agent_launch_env("codex-acp", None, kwargs["agent_env"])
        # Execute the real writer; the dummy client reads exactly the auth
        # file Codex would discover. No provider or real login is contacted.
        command = CODEX_ACP_BUILTIN_LAUNCH.rsplit("exec ", 1)[0] + "true"
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            env={"PATH": os.defpath, **launch},
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        auth = home_dir / ".codex/auth.json"
        logins.append(json.loads(auth.read_text()) if auth.exists() else None)
        return AsyncMock(), AsyncMock(), AsyncMock(), "codex-acp"

    trial._pin_credential_homes = pin
    trial._planes.quiesce_agent = quiesce
    trial._planes.connect_acp = connect_acp
    monkeypatch.setattr(credentials, "upload_credential", upload)
    monkeypatch.setattr(rollout_mod, "remove_staged_credentials", remove)
    monkeypatch.setattr(rollout_mod, "enforce_codex_apps_policy", admitted)
    return trial, home_dir / ".codex/auth.json", events, logins


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entry", ["primary", "role"])
@pytest.mark.parametrize("explicit_disconnect", [False, True])
@pytest.mark.parametrize("login", ["inline", "api"])
async def test_repeated_connections_stage_the_selected_login(
    tmp_path, monkeypatch, backend, entry, explicit_disconnect, login
):
    """5e77deca drops inline auth on primary reconnect; API/role are controls."""
    passed = (
        {"CODEX_AUTH_JSON": INLINE}
        if login == "inline"
        else {"OPENAI_API_KEY": API_KEY, "CODEX_AUTH_JSON": INLINE}
    )
    expected = json.loads(INLINE) if login == "inline" else {"OPENAI_API_KEY": API_KEY}
    trial, auth, events, logins = file_trial(tmp_path, monkeypatch, backend, passed)
    role = Role(name="solver", agent="codex-acp", model=None, env=passed)
    for _ in range(3):
        if entry == "primary":
            await trial.connect()
        else:
            await trial.connect_as(role)
        assert logins[-1] == expected
        assert trial._credential_ledger == [str(auth)]
        if login == "inline" and entry == "primary":
            assert trial._agent_env["OPENAI_API_KEY"] == ""
        if explicit_disconnect:
            await trial.disconnect()
            assert not auth.exists()
            assert trial._credential_ledger == []
    await trial.disconnect()
    assert not auth.exists()
    assert (
        events
        == (([] if login == "api" else ["upload"]) + ["launch", "quiesce", "clean"]) * 3
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("role_login", ["inline", "api", "none"])
async def test_primary_reconnect_replaces_every_previous_role_login(
    tmp_path, monkeypatch, backend, role_login
):
    """5e77deca must restore only primary B after quiescing/cleaning role A."""
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, backend, {"CODEX_AUTH_JSON": INLINE}
    )
    role_values, expected_role = {
        "inline": ({"CODEX_AUTH_JSON": OTHER_INLINE}, json.loads(OTHER_INLINE)),
        "api": ({"OPENAI_API_KEY": API_KEY}, {"OPENAI_API_KEY": API_KEY}),
        "none": ({"CODEX_AUTH_JSON": "", "OPENAI_API_KEY": ""}, None),
    }[role_login]
    await trial.connect()
    await trial.connect_as(
        Role(name="reviewer", agent="codex-acp", model=None, env=role_values)
    )
    assert logins == [json.loads(INLINE), expected_role]
    assert trial._credential_ledger == ([] if role_login == "none" else [str(auth)])
    boundary = len(events)
    await trial.connect()
    assert logins[-1] == json.loads(INLINE)
    assert json.loads(auth.read_text()) == json.loads(INLINE)
    assert events[boundary:] == (
        ["quiesce"] + ([] if role_login == "none" else ["clean"]) + ["upload", "launch"]
    )
    assert trial._credential_ledger == [str(auth)]
    await trial.disconnect()
    assert not auth.exists()


@pytest.mark.parametrize("api_slot", [None, "", "dummy-unadmitted-api"])
@pytest.mark.parametrize("admitted", [False, True])
async def test_inline_upload_uses_admission_not_competing_slot_presence(
    tmp_path, monkeypatch, api_slot, admitted
):
    """Guards 5e77deca: launch-only slots cannot reselect the recorded login."""
    Sandbox(monkeypatch)
    values = resolve_agent_env("codex-acp", None, {"CODEX_AUTH_JSON": INLINE})
    if api_slot is not None:
        values["OPENAI_API_KEY"] = api_slot
    if not admitted:
        values[CREDENTIAL_POLICY_MARKER] = '{"route":"none"}'
    uploads = AsyncMock()
    monkeypatch.setattr(credentials, "upload_credential", uploads)
    ledger = []
    paths = await credentials.write_credential_files(
        SimpleNamespace(),
        "codex-acp",
        values,
        AGENTS["codex-acp"],
        None,
        str(tmp_path),
        record=ledger.append,
    )
    expected = [str(tmp_path / ".codex/auth.json")] if admitted else []
    assert paths == ledger == expected
    assert uploads.await_count == int(admitted)
    if admitted:
        assert uploads.await_args.args[2] == INLINE
