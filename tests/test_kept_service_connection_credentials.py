"""Surviving service processes must not receive a later login.

Use the existing real credential writer and file cleanup fixture. The fake ACP
client reads the exact file a native client would receive; no provider is called.
"""

import json

import pytest

from benchflow.rollout import Role
from benchflow.task.keep_services import KeepService, KeptServices
from tests.test_cli_other_reconnect import BACKENDS, INLINE, OTHER_INLINE, file_trial
from tests.test_sweep_credential_lifecycle import (
    host,  # noqa: F401 -- isolate host auth and register dummy agents
    no_kept_services,  # noqa: F401
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entry", ["primary", "role"])
@pytest.mark.parametrize("explicit_disconnect", [False, True])
@pytest.mark.parametrize("kept", [False, True])
async def test_reconnect_cannot_publish_credentials_to_surviving_service_processes(
    tmp_path, monkeypatch, backend, entry, explicit_disconnect, kept
):
    """Reconnect at 82c2e4269 exposes the second native login."""
    trial, auth, events, logins = file_trial(
        tmp_path, monkeypatch, backend, {"CODEX_AUTH_JSON": OTHER_INLINE}
    )
    services = KeptServices((KeepService(port=5000),), source="the task")
    monkeypatch.setattr(
        "benchflow.task.keep_services.trial_keep",
        lambda *_: services if kept else None,
    )

    await trial.connect()
    assert logins == [json.loads(OTHER_INLINE)]
    assert auth.exists()
    if explicit_disconnect:
        await trial.disconnect()

    # An unchanged role is also a new connection: its native login can rotate.
    trial._agent_env["CODEX_AUTH_JSON"] = INLINE
    trial._config.agent_env = {"CODEX_AUTH_JSON": INLINE}
    if entry == "primary":
        reconnect = trial.connect
    else:

        async def reconnect():
            await trial.connect_as(
                Role(
                    name="next",
                    agent="codex-acp",
                    model=None,
                    env={"CODEX_AUTH_JSON": INLINE},
                )
            )

    consumed = []
    original_runtime = trial._planes.ensure_litellm_runtime

    async def runtime(**kwargs):
        consumed.append("runtime")
        return await original_runtime(**kwargs)

    trial._planes.ensure_litellm_runtime = runtime
    if kept:
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match=r"Cannot reconnect.*x-keep-services"
            ):
                await reconnect()
        assert logins == [json.loads(OTHER_INLINE)]
        assert not auth.exists()
        assert consumed == []
        assert "quiesce" not in events
    else:
        await reconnect()
        assert logins == [json.loads(OTHER_INLINE), json.loads(INLINE)]
        assert json.loads(auth.read_text()) == json.loads(INLINE)
        assert "quiesce" in events

    await trial.disconnect()
