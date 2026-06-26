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

    async def prompt(self, text: str) -> StopReason:
        self.prompts_seen.append(text)
        self._steps.append({"type": "user_message", "text": text})
        self._steps.append({"type": "agent_message", "text": f"did:{text}"})
        if self.on_change:
            self.on_change(self)
        return StopReason.END_TURN

    async def cancel(self) -> None:
        pass

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
