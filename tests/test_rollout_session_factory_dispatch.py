"""Rollout-level session-factory dispatch (the PUBLIC kernel surface).

These exercise the seams where ``Rollout`` decides between the ACP path and the
non-ACP session-factory path, driven through the public ``execute`` /
``connect_as`` / ``on_ask_user`` / ``disconnect`` entry points rather than the
``session_factory_runtime`` helpers in isolation (those are covered by
``test_session_factory_runtime.py``). A fake module-level session-factory agent
(returning a fake protocol ``Session``) keeps the path free of omnigent / sandbox
deps while still walking real ``Rollout`` state — tree growth, trajectory
commit, sticky on_ask_user rebinding, and the ``_is_session_factory`` lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow._types import Role
from benchflow.acp.types import StopReason
from benchflow.agents.registry import (
    AGENT_ALIASES,
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    register_agent,
)
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.rollout.session_factory_runtime import (
    execute_prompts_session_factory as _real_sf_drive,
)


class _FakeSession:
    """Steps-only protocol ``Session`` (no ACP client / adapter / transport)."""

    def __init__(self) -> None:
        self.on_change = None
        self._steps: list[dict] = []
        self.prompts_seen: list[str] = []
        self.ask_user_handler = None

    async def prompt(self, text: str) -> StopReason:
        self.prompts_seen.append(text)
        self._steps.append({"type": "user_message", "text": text})
        self._steps.append({"type": "agent_message", "text": f"did:{text}"})
        if self.on_change:
            self.on_change(self)
        return StopReason.END_TURN

    async def cancel(self) -> None:
        pass

    def on_ask_user(self, handler) -> None:
        self.ask_user_handler = handler

    @property
    def steps(self) -> list[dict]:
        return self._steps


def _rollout(tmp_path: Path) -> Rollout:
    return Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )


# --------------------------------------------------------------------------- #
# (1) execute() end-to-end SF dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_execute_routes_session_factory_agent_through_sf_drive(
    tmp_path: Path, monkeypatch
):
    """The public ``Rollout.execute`` must route a session-factory rollout to
    ``execute_prompts_session_factory`` (NOT the ACP ``execute_prompts``) and
    commit the fake Session's steps as the rollout trajectory + grow the tree.

    Guards PR #825 line 1088: ``if getattr(self, "_is_session_factory", False)``.
    A session-factory rollout has ``_acp_client is None`` — taking the ACP branch
    would drive a ``None`` client, so the conditional is load-bearing.
    """
    rollout = _rollout(tmp_path)
    session = _FakeSession()
    rollout._is_session_factory = True
    rollout._session = session
    rollout._timeout = 30  # per-prompt wall-clock budget for the SF drive

    # The ACP drive must never run for a session-factory rollout.
    async def _acp_must_not_run(*_a, **_k):
        raise AssertionError("ACP execute_prompts ran for a session-factory rollout")

    monkeypatch.setattr(rollout._planes, "execute_prompts", _acp_must_not_run)

    # Spy that proves dispatch AND drives the REAL session-factory loop so the
    # fake Session's steps actually flow back through execute().
    seen: dict = {}

    async def _sf_spy(sess, prompts, timeout, idle_timeout=None):
        seen["session"] = sess
        seen["prompts"] = list(prompts)
        return await _real_sf_drive(sess, prompts, timeout, idle_timeout=idle_timeout)

    monkeypatch.setattr(rollout._planes, "execute_prompts_session_factory", _sf_spy)

    trajectory, n_tool_calls = await rollout.execute(["do the thing"])

    # routed to the SF drive with the live session + our prompt
    assert seen["session"] is session
    assert seen["prompts"] == ["do the thing"]
    # the fake Session's steps are the produced trajectory
    assert trajectory == [
        {"type": "user_message", "text": "do the thing"},
        {"type": "agent_message", "text": "did:do the thing"},
    ]
    assert n_tool_calls == 0
    assert session.prompts_seen == ["do the thing"]
    # committed into rollout state on the clean (non-partial) SF path
    assert rollout._trajectory == trajectory
    assert rollout._partial_trajectory is False
    assert rollout._trajectory_source == "acp"
    assert rollout._n_tool_calls == 0
    # the tree advanced one Step per event (no longer the bare root)
    assert rollout._cursor is not rollout.tree.root


# --------------------------------------------------------------------------- #
# (5) execute -> disconnect flag lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_execute_then_disconnect_clears_session_factory_state(
    tmp_path: Path, monkeypatch
):
    """A full SF execute()->disconnect() cycle through the public surface must
    flip ``_is_session_factory`` True during the session and back to False (with
    the session torn down) after disconnect — so a subsequent ACP connect is not
    poisoned by a stale session-factory flag.

    Guards disconnect() line 879: ``self._is_session_factory = False``.
    """
    rollout = _rollout(tmp_path)
    session = _FakeSession()
    rollout._is_session_factory = True
    rollout._session = session
    rollout._timeout = 30  # per-prompt wall-clock budget for the SF drive
    rollout._env = None  # no sandbox: skip the pkill cleanup branch

    monkeypatch.setattr(
        rollout._planes,
        "execute_prompts_session_factory",
        AsyncMock(side_effect=_real_sf_drive),
    )

    await rollout.execute(["go"])
    assert rollout._is_session_factory is True
    assert rollout._session is session

    await rollout.disconnect()

    assert rollout._is_session_factory is False
    assert rollout._session is None
    assert rollout._session_adapter is None
    assert rollout._session_tool_count == 0
    assert rollout._session_traj_count == 0
    assert rollout._phase == "installed"


# --------------------------------------------------------------------------- #
# (2) connect_as(role) SF reconnect re-binds the sticky on_ask_user handler
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_connect_as_session_factory_rebinds_sticky_on_ask_user(
    tmp_path: Path, monkeypatch
):
    """Switching to a session-factory role via ``connect_as`` must (a) take the
    session-factory connect path, (b) set ``_is_session_factory`` True, and
    (c) re-bind a previously-registered sticky ``on_ask_user`` handler onto the
    fresh Session — the role-switch re-application that keeps a handler from
    silently reverting to auto-approve (pre-#382 bug, re-asserted for #825).

    Guards connect_as: the ``sf_entrypoint`` dispatch + the trailing
    ``self._reapply_ask_user_handler()`` after a session-factory reconnect.
    """
    register_agent(
        "fake-sf-connect-as",
        "echo install",
        "echo launch",
        protocol="session-factory",
        session_factory="fake_mod:build_agent",
    )
    try:
        rollout = _rollout(tmp_path)
        rollout._rollout_dir = tmp_path
        rollout._env = None

        fresh = _FakeSession()
        # Stub the heavy connect_as plane work (no real provider/sandbox/install).
        monkeypatch.setattr(rollout._planes, "resolve_agent_env", lambda *a, **k: {})
        monkeypatch.setattr(
            rollout._planes,
            "ensure_litellm_runtime",
            AsyncMock(return_value=({}, None)),
        )
        monkeypatch.setattr(
            rollout._planes, "install_agent", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            rollout._planes, "write_credential_files", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            rollout._planes, "apply_web_tool_policy", AsyncMock(return_value=None)
        )
        connect_sf = AsyncMock(return_value=(None, fresh, None, "fake-sf-connect-as"))
        monkeypatch.setattr(rollout._planes, "connect_session_factory", connect_sf)
        # connect_acp must NOT be taken for a session-factory role.
        monkeypatch.setattr(
            rollout._planes,
            "connect_acp",
            AsyncMock(side_effect=AssertionError("ACP connect taken for SF role")),
        )

        # Register the sticky handler BEFORE the role switch (no live session yet).
        async def handler(request):
            return "approved"

        rollout.on_ask_user(handler)

        role = Role(name="reviewer", agent="fake-sf-connect-as", model="m")
        await rollout.connect_as(role)

        # dispatched to the session-factory connect path
        connect_sf.assert_awaited_once()
        assert rollout._is_session_factory is True
        assert rollout._session is fresh
        assert rollout._acp_client is None
        assert rollout._active_role is role
        # the sticky handler was re-bound onto the fresh session-factory Session
        assert fresh.ask_user_handler is handler
    finally:
        AGENTS.pop("fake-sf-connect-as", None)
        AGENT_ALIASES.pop("fake-sf-connect-as", None)
        AGENT_INSTALLERS.pop("fake-sf-connect-as", None)
        AGENT_LAUNCH.pop("fake-sf-connect-as", None)


# --------------------------------------------------------------------------- #
# (3) on_ask_user full SF lifecycle: register -> fire -> survive reconnect -> clear
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_on_ask_user_session_factory_lifecycle(tmp_path: Path):
    """Full sticky-handler lifecycle for a session-factory rollout, through the
    PUBLIC ``on_ask_user`` setter:

    1. register on a live SF session -> bound immediately (setter calls reapply);
    2. the bound handler actually FIRES when the session emits a request;
    3. survives a reconnect to a *different* SF session (re-bound, not lost);
    4. clear (``on_ask_user(None)``) -> a subsequent fresh SF session is left
       on the default auto-approve path (no handler bound).

    Guards on_ask_user() line 908 (the setter's ``_reapply_ask_user_handler()``
    call) + the session-factory branch of ``_reapply_ask_user_handler``.
    """
    rollout = Rollout.__new__(Rollout)  # only the on_ask_user surface is exercised
    rollout._is_session_factory = True
    rollout._session_adapter = None
    rollout._acp_client = None
    rollout._ask_user_handler = None
    rollout._ask_user_handler_set = False

    session1 = _FakeSession()
    rollout._session = session1

    fired: list = []

    async def handler(request):
        fired.append(request)
        return "approved"

    # (1) public setter binds onto the already-live session via reapply.
    rollout.on_ask_user(handler)
    assert rollout._ask_user_handler_set is True
    assert session1.ask_user_handler is handler

    # (2) the bound handler actually runs when the SF session emits a request.
    result = await session1.ask_user_handler({"q": "rm -rf?"})
    assert result == "approved"
    assert fired == [{"q": "rm -rf?"}]

    # (3) reconnect to a different SF session -> handler survives (re-bound).
    session2 = _FakeSession()
    rollout._session = session2
    rollout._reapply_ask_user_handler()
    assert session2.ask_user_handler is handler

    # (4) clear, then a fresh session must be left unbound (auto-approve default).
    rollout.on_ask_user(None)
    session3 = _FakeSession()
    rollout._session = session3
    rollout._reapply_ask_user_handler()
    assert session3.ask_user_handler is None
