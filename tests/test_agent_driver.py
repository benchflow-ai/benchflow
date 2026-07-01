"""Protocol-branched seat driver: import_callable + connect/prompt dispatch."""

from __future__ import annotations

import os.path

import pytest

from benchflow.agents.registry import AgentConfig
from benchflow.arena import agent_driver
from benchflow.arena.agent_driver import (
    close_seat,
    connect_seat,
    import_callable,
    prompt_seat,
)


def test_import_callable_colon_and_dotted():
    assert import_callable("os.path:join") is os.path.join
    assert import_callable("os.path.join") is os.path.join


def test_import_callable_rejects_bad_refs():
    with pytest.raises(ValueError, match="module:callable"):
        import_callable("nocolonnomodule")
    with pytest.raises(TypeError, match="non-callable"):
        import_callable("os:sep")  # os.sep is a str, not callable


@pytest.mark.asyncio
async def test_acp_branch_calls_connect_acp_and_execute_prompts(monkeypatch):
    seen = {}

    async def fake_connect_acp(**kw):
        seen["connect"] = kw
        return ("CLIENT", "SESSION", "ADAPTER", "the-agent")

    async def fake_execute_prompts(*, acp_client, session, prompts, timeout, idle_timeout):
        seen["exec"] = dict(client=acp_client, prompts=prompts, timeout=timeout)
        return ([{"type": "tool_call"}], 1)

    monkeypatch.setattr(agent_driver, "connect_acp", fake_connect_acp)
    monkeypatch.setattr(agent_driver, "execute_prompts", fake_execute_prompts)

    cfg = AgentConfig(name="codex-acp", install_cmd="x", launch_cmd="codex --acp")
    conn = await connect_seat(
        cfg, env="SANDBOX", agent_cwd="/work/cx", agent_env={"K": "v"},
        model="gpt-5.5", rollout_dir="/tmp/out", environment="docker", seat_id="cx",
    )
    assert conn.protocol == "acp"
    assert conn.client == "CLIENT" and conn.session == "SESSION"
    assert seen["connect"]["agent_cwd"] == "/work/cx"
    assert seen["connect"]["agent"] == "codex-acp"
    assert seen["connect"]["agent_launch"] == "codex --acp"

    traj, n = await prompt_seat(conn, "play", timeout=10, idle_timeout=5)
    assert (traj, n) == ([{"type": "tool_call"}], 1)
    assert seen["exec"]["prompts"] == ["play"]


# --- session-factory branch, exercised with a fake Agent/Session (no sandbox) ---

class _FakeSession:
    on_change = None

    def __init__(self):
        self.prompts: list[str] = []
        self.steps = [{"type": "agent_message", "text": "hi"}, {"type": "tool_call"}]

    async def prompt(self, text):
        self.prompts.append(text)
        return "end_turn"


class _FakeAgent:
    last: _FakeSession | None = None

    async def connect(self, sandbox, role):
        _FakeAgent.last = _FakeSession()
        _FakeAgent.last.role = role
        return _FakeAgent.last


def build_fake_agent():  # the session_factory "module:callable" target
    return _FakeAgent()


@pytest.mark.asyncio
async def test_session_factory_branch_uses_agent_connect_and_prompt():
    cfg = AgentConfig(
        name="omni",
        install_cmd="x",
        launch_cmd="unused",
        protocol="session-factory",
        session_factory="tests.test_agent_driver:build_fake_agent",
    )
    conn = await connect_seat(
        cfg, env="SANDBOX", agent_cwd="/work/omni", agent_env={},
        model=None, rollout_dir="/tmp/out", environment="docker", seat_id="omni",
    )
    assert conn.protocol == "session-factory"
    assert conn.session.role == "omni"

    traj, n = await prompt_seat(conn, "go", timeout=10)
    assert conn.session.prompts == ["go"]
    assert n == 1  # one tool_call in steps
    assert len(traj) == 2
    await close_seat(conn)  # no client → no-op


@pytest.mark.asyncio
async def test_session_factory_requires_factory():
    cfg = AgentConfig(
        name="bad", install_cmd="x", launch_cmd="y", protocol="session-factory"
    )
    with pytest.raises(ValueError, match="session_factory"):
        await connect_seat(
            cfg, env="S", agent_cwd="/work/bad", agent_env={}, model=None,
            rollout_dir="/tmp", environment="docker", seat_id="bad",
        )


@pytest.mark.asyncio
async def test_unsupported_protocol_raises():
    cfg = AgentConfig(name="weird", install_cmd="x", launch_cmd="y", protocol="cli")
    with pytest.raises(ValueError, match="unsupported protocol"):
        await connect_seat(
            cfg, env="S", agent_cwd="/work/w", agent_env={}, model=None,
            rollout_dir="/tmp", environment="docker", seat_id="w",
        )
