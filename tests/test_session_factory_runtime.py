"""Session-factory connect + drive (the non-ACP kernel path).

Drives the runtime with a FAKE session-factory agent (a module:callable that
returns a fake Agent → fake Session implementing the Session protocol) so the
kernel path is exercised without omnigent/sandbox deps.
"""

from __future__ import annotations

import asyncio

import pytest

from benchflow.acp.runtime import AgentPromptTimeoutError
from benchflow.acp.types import StopReason
from benchflow.rollout.session_factory_runtime import (
    _load_session_factory,
    connect_session_factory,
    execute_prompts_session_factory,
)


class _FakeSession:
    def __init__(self) -> None:
        self.on_change = None
        self._steps: list[dict] = []
        self.prompts_seen: list[str] = []
        # The Session protocol requires on_ask_user(handler) (protocol.py).
        # Tracking the last-bound handler makes the session-factory ask-user
        # rebinding observable instead of masking the gap with a missing method.
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


class _FakeAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.connected: tuple | None = None

    async def connect(self, sandbox, role):
        self.connected = (sandbox, role)
        return _FakeSession()


class _SlowConnectAgent:
    async def connect(self, sandbox, role):
        await asyncio.sleep(10)
        return _FakeSession()


_BUILT: dict = {}


def build_fake_agent(**kwargs) -> _FakeAgent:
    agent = _FakeAgent(**kwargs)
    _BUILT["agent"] = agent
    return agent


def build_slow_connect_agent(**kwargs) -> _SlowConnectAgent:
    return _SlowConnectAgent()


@pytest.mark.asyncio
async def test_connect_builds_agent_and_connects_with_proxy_env():
    """Guards PR #825: session-factory connect uses Agent.connect(sandbox, role)."""
    _BUILT.clear()
    client, session, adapter, name = await connect_session_factory(
        env="SANDBOX-HANDLE",
        agent="fake-sf",
        session_factory=f"{__name__}:build_fake_agent",
        agent_env={"BENCHFLOW_PROVIDER_MODEL": "deepseek-v4-flash"},
        sandbox_user="root",
        model="deepseek-v4-flash",
        rollout_dir=None,
        timeout=30,
        agent_cwd="/workspace",
    )
    # shape-compatible with connect_acp: no client/adapter for session-factory
    assert client is None and adapter is None and name == "fake-sf"
    assert isinstance(session, _FakeSession)
    # the agent was connected with the sandbox wrapper + proxy-routing agent_env
    connect_sandbox, role = _BUILT["agent"].connected
    assert connect_sandbox.sandbox == "SANDBOX-HANDLE"
    assert role == "agent"
    assert connect_sandbox.agent_env == {
        "BENCHFLOW_PROVIDER_MODEL": "deepseek-v4-flash",
        "BENCHFLOW_AGENT_CWD": "/workspace",
    }
    # sandbox_user forwarded as exec_user (credential store + exec stay in lockstep)
    assert _BUILT["agent"].kwargs == {"exec_user": "root"}


@pytest.mark.asyncio
async def test_connect_timeout_raises():
    """Guards PR #825: hung session-factory connect is bounded by the agent budget."""
    with pytest.raises(TimeoutError, match=r"session-factory connect exceeded 0\.01s"):
        await connect_session_factory(
            env="SANDBOX-HANDLE",
            agent="fake-sf",
            session_factory=f"{__name__}:build_slow_connect_agent",
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=None,
            timeout=0.01,
        )


@pytest.mark.asyncio
async def test_drive_runs_each_prompt_and_captures_steps():
    session = _FakeSession()
    streamed: list = []
    session.on_change = lambda s: streamed.append(len(s.steps))
    trajectory, n_tool_calls = await execute_prompts_session_factory(
        session, ["hello", "again"], timeout=30
    )
    assert n_tool_calls == 0  # no per-tool-call stream for session-factory
    assert session.prompts_seen == ["hello", "again"]
    assert {"type": "agent_message", "text": "did:again"} in trajectory
    assert streamed  # on_change fired (kernel wires the trajectory writer onto it)


@pytest.mark.asyncio
async def test_drive_timeout_raises_with_partial_trajectory():
    class _HangSession(_FakeSession):
        async def prompt(self, text: str) -> StopReason:
            self._steps.append({"type": "user_message", "text": text})
            import asyncio

            await asyncio.sleep(10)
            return StopReason.END_TURN

    session = _HangSession()
    with pytest.raises(AgentPromptTimeoutError) as ei:
        await execute_prompts_session_factory(session, ["slow"], timeout=1)
    assert ei.value.n_tool_calls == 0
    assert ei.value.executed_prompts == ["slow"]
    assert ei.value.trajectory  # the user_message step captured before timeout
    # a wall-clock timeout snapshot is terminal-complete (no pending tool calls)
    assert ei.value.terminal_trajectory_complete is True


@pytest.mark.asyncio
async def test_session_factory_partial_trajectory_on_non_timeout_failure():
    """A NON-timeout prompt failure must still preserve the partial trajectory.

    A session-factory agent that appends steps then blows up mid-run (e.g. the
    one-shot CLI raises ``RuntimeError``) must not lose the steps captured so
    far. ``execute_prompts_session_factory`` wraps the failure in an
    ``AgentPromptTimeoutError`` carrying the partial trajectory (marked partial,
    not terminal-complete) so the kernel's existing handler commits it — instead
    of the bare exception bubbling up and discarding the trajectory entirely.
    """

    class _FailSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        async def prompt(self, text: str) -> StopReason:
            self._calls += 1
            self._steps.append({"type": "user_message", "text": text})
            if self._calls >= 2:
                raise RuntimeError("provider exploded")
            self._steps.append({"type": "agent_message", "text": f"did:{text}"})
            return StopReason.END_TURN

    session = _FailSession()
    with pytest.raises(AgentPromptTimeoutError) as ei:
        await execute_prompts_session_factory(session, ["one", "two"], timeout=30)

    # the partial trajectory captured before the RuntimeError survives
    assert ei.value.trajectory
    assert {"type": "user_message", "text": "two"} in ei.value.trajectory
    assert ei.value.n_tool_calls == 0
    assert ei.value.executed_prompts == ["one", "two"]
    # a non-timeout failure is partial, not a clean terminal snapshot
    assert ei.value.terminal_trajectory_complete is False
    # the real cause is preserved on the exception chain (not masked as a timeout)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_session_factory_on_ask_user_handler_wired():
    """``on_ask_user`` must rebind onto the live session even when adapter=None.

    A session-factory agent returns ``adapter=None`` from connect (the session
    IS the protocol-conformant object). ``_reapply_ask_user_handler`` returned
    early whenever the adapter was None, so the sticky on_ask_user handler never
    reached a session-factory agent. The handler must instead bind onto
    ``self._session.on_ask_user`` for the session-factory path (#825).
    """
    from benchflow.rollout import Rollout

    session = _FakeSession()
    r = Rollout.__new__(Rollout)  # bypass __init__ (only the method is exercised)
    r._is_session_factory = True
    r._session = session
    r._session_adapter = None
    r._ask_user_handler_set = True

    async def handler(request):
        return "ok"

    r._ask_user_handler = handler
    r._reapply_ask_user_handler()
    assert session.ask_user_handler is handler


@pytest.mark.asyncio
async def test_session_factory_partial_trajectory_captured_on_disconnect():
    """disconnect() must capture the session-factory session's partial tail.

    For a session-factory agent ``self._acp_client`` is None, so the ACP partial
    capture is a no-op; the live trajectory lives on ``self._session.steps``. If
    ``execute`` raised before its normal extend, ``disconnect`` must still flush
    those uncommitted steps into ``self._trajectory`` (marked partial) before it
    tears the session down (#825).
    """
    from benchflow.rollout import Rollout

    session = _FakeSession()
    session._steps = [
        {"type": "user_message", "text": "task"},
        {"type": "agent_message", "text": "partial work"},
    ]

    r = Rollout.__new__(Rollout)
    r._acp_client = None
    r._session = session
    r._session_adapter = None
    r._is_session_factory = True
    r._agent_launch = ""
    r._env = None
    r._active_role = None
    r._trajectory = []
    r._session_traj_count = 0
    r._session_tool_count = 0
    r._partial_trajectory = False
    r._trajectory_source = None
    r._phase = "connected"

    await r.disconnect()

    assert r._trajectory == [
        {"type": "user_message", "text": "task"},
        {"type": "agent_message", "text": "partial work"},
    ]
    assert r._partial_trajectory is True
    assert r._trajectory_source == "partial_acp"
    # state is still torn down after the capture
    assert r._session is None
    assert r._is_session_factory is False


def test_load_session_factory_rejects_malformed():
    with pytest.raises(ValueError):
        _load_session_factory("no-colon")
    with pytest.raises(ValueError):
        _load_session_factory(":missing-module")


def test_session_factory_entrypoint_is_the_dispatch_key():
    """Rollout._session_factory_entrypoint returns the entrypoint for a
    session-factory agent (→ the non-ACP path) and None for an ACP agent (→ the
    ACP path). This is the single key the connect + drive dispatch branches on."""
    from benchflow.agents.registry import (
        AGENT_ALIASES,
        AGENT_INSTALLERS,
        AGENT_LAUNCH,
        AGENTS,
        register_agent,
    )
    from benchflow.rollout import Rollout

    register_agent(
        "fake-sf-agent",
        "echo install",
        "echo launch",
        protocol="session-factory",
        session_factory="mymod:build_agent",
    )
    try:
        r = Rollout.__new__(Rollout)  # bypass __init__ (only the method is exercised)
        assert r._session_factory_entrypoint("fake-sf-agent") == "mymod:build_agent"
        # a real ACP agent → None (stays on the ACP path)
        assert r._session_factory_entrypoint("opencode") is None
        # unknown agent → None (degrade to ACP rather than raise)
        assert r._session_factory_entrypoint("no-such-agent-xyz") is None
    finally:
        AGENTS.pop("fake-sf-agent", None)
        AGENT_ALIASES.pop("fake-sf-agent", None)
        AGENT_INSTALLERS.pop("fake-sf-agent", None)
        AGENT_LAUNCH.pop("fake-sf-agent", None)


@pytest.mark.asyncio
async def test_disconnect_clears_session_factory_state():
    """Guards PR #825 against reusing a stale session-factory session after disconnect."""
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._acp_client = None
    rollout._session = _FakeSession()
    rollout._session_adapter = object()
    rollout._is_session_factory = True
    rollout._agent_launch = ""
    rollout._env = None
    rollout._active_role = object()
    rollout._session_tool_count = 7
    rollout._session_traj_count = 11
    rollout._phase = "connected"

    await rollout.disconnect()

    assert rollout._session is None
    assert rollout._session_adapter is None
    assert rollout._is_session_factory is False
    assert rollout._active_role is None
    assert rollout._session_tool_count == 0
    assert rollout._session_traj_count == 0
    assert rollout._phase == "installed"


@pytest.mark.asyncio
async def test_steps_only_session_trajectory_sink_writes_steps(tmp_path):
    """The real streaming sink must work on a steps-only session-factory Session.

    ``Rollout._attach_trajectory_writer`` wires
    ``make_trajectory_sink(TrajectoryWriter(...))`` onto ``self._session.on_change``
    for BOTH the ACP and session-factory paths. A session-factory ``self._session``
    is the steps-only protocol ``Session`` (``benchflow.agents.protocol.Session``):
    it exposes ``.steps`` + ``.on_change`` but NOT the ACP streaming bookkeeping
    (``_events_active`` / ``_pending_text`` live only on ``ACPSession``). The sinks
    ``_snapshot_session_trajectory`` must therefore duck-type off ``.steps`` — else
    every streaming ``on_change`` raises ``AttributeError`` and
    ``acp_trajectory.jsonl`` is never written (#825 BLOCKER 8).
    """
    import json
    from pathlib import Path

    from benchflow.trajectories._capture import (
        TrajectoryWriter,
        make_trajectory_sink,
    )

    session = _FakeSession()  # steps-only: no _events_active / _pending_text
    assert not hasattr(session, "_events_active")
    assert not hasattr(session, "_pending_text")

    traj_path: Path = tmp_path / "trajectory" / "acp_trajectory.jsonl"
    # Wire the REAL sink exactly as Rollout._attach_trajectory_writer does.
    session.on_change = make_trajectory_sink(TrajectoryWriter(traj_path), [])

    # Drive one turn: prompt() appends steps then fires on_change (the sink).
    # (a) no AttributeError must leak out of the sink.
    await session.prompt("do the thing")

    # (b) the streamed file content equals the sessions steps.
    written = [
        json.loads(line) for line in traj_path.read_text().splitlines() if line.strip()
    ]
    assert written == session.steps
