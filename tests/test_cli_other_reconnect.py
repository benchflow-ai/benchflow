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


MODES = ["strict", "audit", "off"]


def file_trial(tmp_path, monkeypatch, backend, passed, integrity="strict"):
    sandbox = Sandbox(monkeypatch)
    trial = _rollout(tmp_path, sandbox, "codex-acp", None, passed, integrity=integrity)
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


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entry", ["primary", "role"])
@pytest.mark.parametrize("explicit_disconnect", [False, True])
@pytest.mark.parametrize("login", ["inline", "api"])
async def test_repeated_connections_stage_the_selected_login(
    tmp_path, monkeypatch, backend, entry, explicit_disconnect, login, integrity
):
    """5e77deca drops inline auth on primary reconnect; API/role are controls.
    The same role with the same logins: strict stops and cleans at each
    turn's end; off and audit stop nothing, remove the file and keep the
    ledger's entry until a quiesce (the lead's 18:40Z and 19:23Z
    rulings)."""
    strict = integrity == "strict"
    passed = (
        {"CODEX_AUTH_JSON": INLINE}
        if login == "inline"
        else {"OPENAI_API_KEY": API_KEY, "CODEX_AUTH_JSON": INLINE}
    )
    expected = json.loads(INLINE) if login == "inline" else {"OPENAI_API_KEY": API_KEY}
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, backend, passed, integrity
    )
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
            assert trial._credential_ledger == ([] if strict else [str(auth)])
    await trial.disconnect()
    assert not auth.exists()
    upload = [] if login == "api" else ["upload"]
    turn = ["clean"] if explicit_disconnect else []
    assert events == (
        [*upload, "launch", "quiesce", "clean"] * 3
        if strict
        else [*upload, "launch", *turn] * 3 + ["clean"]
    )


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("role_login", ["inline", "api", "none"])
async def test_primary_reconnect_replaces_every_previous_role_login(
    tmp_path, monkeypatch, backend, role_login, integrity
):
    """5e77deca must restore only primary B after quiescing/cleaning role A,
    in every mode (the lead's 18:40Z ruling)."""
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, backend, {"CODEX_AUTH_JSON": INLINE}, integrity
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


# -- a role change in every mode (the lead's 18:40Z ruling, R330D) ----


def _leftover(trial, events):
    """A process the connection left, alive until a quiesce: it re-creates
    the login it read, as Codex's writer does."""
    state = {"alive": False}

    async def quiesce(*args, **kwargs):
        events.append("quiesce")
        state["alive"] = False

    trial._planes.quiesce_agent = quiesce
    return state


NO_LOGIN = {"CODEX_AUTH_JSON": "", "OPENAI_API_KEY": ""}


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("explicit_disconnect", [False, True])
async def test_a_second_role_never_launches_with_the_first_roles_login(
    tmp_path, monkeypatch, backend, explicit_disconnect, integrity
):
    """R330D's probe: an inline login, a process of that connection that
    re-creates it after the turn, then a second role with no login. The
    second role's launch must find no login in every mode."""
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, backend, {}, integrity
    )
    leftover = _leftover(trial, events)
    first = {"CODEX_AUTH_JSON": INLINE}
    await trial.connect_as(
        Role(name="solver", agent="codex-acp", model=None, env=first)
    )
    assert logins[-1] == json.loads(INLINE)
    leftover["alive"] = True
    if explicit_disconnect:
        await trial.disconnect()
    if leftover["alive"]:
        auth.parent.mkdir(parents=True, exist_ok=True)
        auth.write_text(INLINE)

    await trial.connect_as(
        Role(name="reviewer", agent="codex-acp", model=None, env=NO_LOGIN)
    )

    assert logins[-1] is None, "the second role launched with the first's login"
    assert not leftover["alive"]


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_an_earlier_role_never_returns_to_a_later_roles_login(
    tmp_path, monkeypatch, backend, integrity
):
    """The other way: a role with no login, then a role with one whose
    process re-creates it after its turn, then the first role again: it must
    find no login."""
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, backend, {}, integrity
    )
    leftover = _leftover(trial, events)
    first = Role(name="solver", agent="codex-acp", model=None, env=NO_LOGIN)
    await trial.connect_as(first)
    assert logins[-1] is None
    await trial.connect_as(
        Role(
            name="reviewer",
            agent="codex-acp",
            model=None,
            env={"CODEX_AUTH_JSON": OTHER_INLINE},
        )
    )
    assert logins[-1] == json.loads(OTHER_INLINE)
    leftover["alive"] = True
    await trial.disconnect()
    if leftover["alive"]:
        auth.write_text(OTHER_INLINE)

    await trial.connect_as(first)

    assert logins[-1] is None, "the first role launched with the second's login"


@pytest.mark.parametrize("integrity", MODES)
async def test_a_leftover_process_is_stopped_before_the_next_roles_login(
    tmp_path, monkeypatch, integrity
):
    """A process the first role left must not live to read the second role's
    login: the quiesce comes before the second role's upload, in every mode,
    and the transition is recorded host-side."""
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, "docker", {}, integrity
    )
    leftover = _leftover(trial, events)
    seen_alive = []
    upload = credentials.upload_credential

    async def watched(env, path, content, **kwargs):
        seen_alive.append(leftover["alive"])
        await upload(env, path, content, **kwargs)

    monkeypatch.setattr(credentials, "upload_credential", watched)
    await trial.connect_as(
        Role(
            name="solver",
            agent="codex-acp",
            model=None,
            env={"CODEX_AUTH_JSON": INLINE},
        )
    )
    leftover["alive"] = True
    await trial.disconnect()
    await trial.connect_as(
        Role(
            name="reviewer",
            agent="codex-acp",
            model=None,
            env={"CODEX_AUTH_JSON": OTHER_INLINE},
        )
    )

    assert logins[-1] == json.loads(OTHER_INLINE)
    assert seen_alive[-1] is False, "the first role's process saw the second's login"
    transitions = tmp_path / "role-transitions.jsonl"
    record = (
        [json.loads(line) for line in transitions.read_text().splitlines()]
        if transitions.exists()
        else []
    )
    if integrity == "strict":
        # The turn's end stopped it already: no transition quiesce to record.
        assert record == []
    else:
        [line] = record
        assert line["from"] == ["codex-acp", None] and line["to"] == ["codex-acp", None]
        assert line["logins_differ"] is True and line["quiesced"] is True
        assert line["logins_removed"] == [str(auth)]
        assert "dummy" not in json.dumps(line)


# -- the trial's end and grading (the lead's 19:23Z ruling; R330D) -----------


INLINE_ROLE = Role(
    name="solver", agent="codex-acp", model=None, env={"CODEX_AUTH_JSON": INLINE}
)


@pytest.mark.parametrize("integrity", MODES)
async def test_no_login_file_is_left_for_test_sh(tmp_path, monkeypatch, integrity):
    """(c) After the turn the login file is gone before grading starts, in
    every mode; off and audit keep the ledger's entry for the next quiesce."""
    trial, auth, _events, _logins = file_trial(
        tmp_path, monkeypatch, "docker", {}, integrity
    )
    await trial.connect_as(INLINE_ROLE)
    assert auth.exists()

    await trial.disconnect()

    assert not auth.exists()
    assert trial._credential_ledger == ([] if integrity == "strict" else [str(auth)])


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_trial_that_never_grades_ends_with_no_login(
    tmp_path, monkeypatch, backend, integrity
):
    """(a) An agent error, a cancel or a lost sandbox: no grading, so no kill
    after it. The trial's end stops what the turn left and removes the
    login and the ledger's entry."""
    trial, auth, events, _logins = file_trial(
        tmp_path, monkeypatch, backend, {}, integrity
    )
    leftover = _leftover(trial, events)
    await trial.connect_as(INLINE_ROLE)
    leftover["alive"] = True
    await trial.disconnect()
    if leftover["alive"]:
        auth.write_text(INLINE)

    await trial._disconnect_at_trial_end()

    assert not auth.exists()
    assert trial._credential_ledger == []
    assert not leftover["alive"]


@pytest.mark.parametrize("integrity", MODES)
async def test_a_login_written_back_during_grading_is_removed_after_it(
    tmp_path, monkeypatch, integrity
):
    """(d) A process the turn left writes its login back while test.sh runs
    (off and audit keep it alive through grading); the kill after grading and
    the removal after it catch it."""
    trial, auth, events, _logins = file_trial(
        tmp_path, monkeypatch, "docker", {}, integrity
    )
    leftover = _leftover(trial, events)
    await trial.connect_as(INLINE_ROLE)
    leftover["alive"] = True
    await trial.disconnect()
    if leftover["alive"]:
        auth.write_text(INLINE)  # during grading
    # S2's kill after grading (rollout._setup._verify_rollout).
    await trial._planes.quiesce_agent(trial._env, "agent")
    trial._env._benchflow_quiesced_after_grading = True

    await trial._disconnect_at_trial_end()

    assert not auth.exists()
    assert trial._credential_ledger == []


@pytest.mark.parametrize("integrity", MODES)
async def test_a_kept_sandbox_holds_no_login(tmp_path, monkeypatch, integrity):
    """(b) ``--keep-sandbox``: the sandbox is kept after the trial's last
    disconnect, which has stopped what the turn left and removed its login."""
    trial, auth, events, _logins = file_trial(
        tmp_path, monkeypatch, "docker", {}, integrity
    )
    _leftover(trial, events)
    await trial.connect_as(INLINE_ROLE)
    await trial.disconnect()
    seen = []

    async def teardown():
        # Where the sandbox is kept or stopped.
        seen.append((auth.exists(), list(trial._credential_ledger)))

    monkeypatch.setattr(
        rollout_mod, "collect_unverified_rollout_artifacts", AsyncMock()
    )
    monkeypatch.setattr(rollout_mod, "_record_reported_models", lambda *a: None)
    trial._steps_before_agent_stop = AsyncMock(return_value=True)
    trial._enforce_required_usage_tracking = lambda: None
    trial._record_main_metadata_at_end = AsyncMock()
    trial._discard_verifier_warmup = AsyncMock()
    trial._release_host_proxy = lambda: None
    trial._teardown_sandbox = teardown

    await trial._cleanup_steps()

    assert seen == [(False, [])]
