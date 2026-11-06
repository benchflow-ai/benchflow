"""Guard FF-sw-agents-C3 at 5679911d: Muse text survives ACP filtering."""

import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.session import ACPSession
from benchflow.agents import antigravity_acp_shim, muse_acp_shim
from benchflow.rollout._setup import final_agent_reply
from benchflow.sandbox.process._acp_lines import line_limit, shrink_line


@pytest.fixture(
    params=[muse_acp_shim, antigravity_acp_shim], ids=["muse", "antigravity"]
)
def shim(request):
    return request.param


def _session(shim, session_id):
    if shim is muse_acp_shim:
        return shim.MuseSession(session_id, "/tmp/workspace", "fixture", ())
    return shim.AgySession(session_id, "/tmp/workspace")


def _commit(shim, session, kind, text):
    if shim is muse_acp_shim:
        session._handle_event({"kind": kind, "text": text})
    else:
        step = "agent_response" if kind == "assistant_message_committed" else "thinking"
        session._handle_step({"step_type": step, "text_delta": text}, set())


@pytest.mark.parametrize(
    "kind,update",
    [
        ("assistant_message_committed", "agent_message_chunk"),
        ("reasoning_summary_committed", "agent_thought_chunk"),
    ],
)
@pytest.mark.parametrize(
    "limit", [None, "8192", "1024", "0", "off", "none", "false", "-1", "invalid"]
)
@pytest.mark.parametrize(
    "text",
    ["A" * 9000, "A" * 1048576, "λ😀" * 9000, '\\"\n\t\x00' * 9000],
    ids=["reported-answer", "default-overflow", "unicode", "json-escaping"],
)
async def test_muse_text_survives_both_filters(
    shim, kind, update, limit, text, monkeypatch
):
    if limit is None:
        monkeypatch.delenv("BENCHFLOW_ACP_LINE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", limit)
    output = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", output)
    session_id = 'session-"\\😀'
    session = _session(shim, session_id)
    _commit(shim, session, kind, text)
    lines = output.getvalue().encode().splitlines(keepends=True)
    effective = line_limit()
    # The complete JSON envelope, escaping and newline count toward the budget.
    assert lines and all(len(line) <= min(effective or 65536, 65536) for line in lines)
    if effective is not None:
        lines = [shrink_line(line, effective) for line in lines]
    process = SimpleNamespace(readline=AsyncMock(side_effect=lines))
    transport = ContainerTransport(process, "unused")
    received = [await transport.receive() for _ in lines]
    assert all(message["params"]["sessionId"] == session_id for message in received)
    updates = [message["params"]["update"] for message in received]
    assert all(message["sessionUpdate"] == update for message in updates)
    assert "".join(message["content"]["text"] for message in updates) == text
    acp = ACPSession(session_id)
    for message in updates:
        acp.handle_update(message)
    acp._flush_agent_text()
    assert final_agent_reply(acp.events) == (
        text if kind == "assistant_message_committed" else None
    )


@pytest.mark.parametrize(
    "kind", ["assistant_message_committed", "reasoning_summary_committed"]
)
@pytest.mark.parametrize("limit", ["1", "200"])
def test_muse_impossible_budget_fails_prompt(shim, kind, limit, monkeypatch):
    """An oversized session envelope must not become a successful end_turn."""
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", limit)
    output = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", output)
    session = _session(shim, 'session-"\\😀' * 30)

    def prompt(text):
        _commit(shim, session, kind, "A")
        if shim is muse_acp_shim:
            return "end_turn", shim.Usage()
        return {"success": True}

    monkeypatch.setattr(session, "prompt", prompt)
    shim.Server()._run_prompt(7, session, "prompt")
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(messages) == 1
    assert messages[0]["id"] == 7
    assert "ACP line limit" in messages[0]["error"]["message"]
    assert not session.prompt_active


def test_muse_preserves_order_and_ignores_empty_text(shim, monkeypatch):
    """C3 chunking keeps separate commits ordered and thoughts out of answers."""
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "8192")
    output = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", output)
    session = _session(shim, "fixture")
    for kind, text in [
        ("assistant_message_committed", "first" * 3000),
        ("reasoning_summary_committed", "thought" * 3000),
        ("assistant_message_committed", ""),
        ("assistant_message_committed", None),
        ("assistant_message_committed", 7),
        ("assistant_message_committed", "second" * 3000),
    ]:
        _commit(shim, session, kind, text)
    updates = [
        json.loads(shrink_line(line, 8192))["params"]["update"]
        for line in output.getvalue().encode().splitlines(keepends=True)
    ]
    assert (
        "".join(
            message["content"]["text"]
            for message in updates
            if message["sessionUpdate"] == "agent_message_chunk"
        )
        == "first" * 3000 + "second" * 3000
    )


@pytest.mark.parametrize("update", ["agent_message_chunk", "agent_thought_chunk"])
@pytest.mark.parametrize(
    "text,extra,accepted", [("A", 1, True), ("😀", 12, True), ("😀", 11, False)]
)
def test_muse_exact_envelope_boundary(shim, update, text, extra, accepted, monkeypatch):
    """C3 includes JSON escaping and the newline even at the last byte."""
    output = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", output)
    shim._notify(
        "fixture", {"sessionUpdate": update, "content": {"type": "text", "text": ""}}
    )
    envelope = len(output.getvalue().encode())
    output.seek(0)
    output.truncate()
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", str(envelope + extra))
    if accepted:
        shim._emit_text("fixture", text, update)
        assert len(output.getvalue().encode()) == envelope + extra
        assert (
            json.loads(output.getvalue())["params"]["update"]["content"]["text"] == text
        )
    else:
        with pytest.raises(ValueError, match="ACP line limit"):
            shim._emit_text("fixture", text, update)
        assert output.getvalue() == ""


@pytest.mark.parametrize("agent", ["muse", "muse-code"])
@pytest.mark.parametrize("saved_drop", [False, True])
async def test_muse_host_budget_survives_login_shell(
    agent, saved_drop, monkeypatch, tmp_path
):
    """Guards C3's budget and the S3 merge at 147e3505 for both spellings."""
    from benchflow.acp import runtime

    class StopBeforeConnect(Exception):
        pass

    commands = []

    real_priv_drop = runtime.build_priv_drop_cmd

    def capture(command, user):
        # S2 makes this unconditional, even without an S3 audit record.
        wrapped = real_priv_drop(command, user)
        for option in (
            "--no-new-privs",
            "--bounding-set=-net_admin,-net_raw",
            "--inh-caps=-all",
            "--ambient-caps=-all",
        ):
            assert wrapped.count(option) == 2
        commands.append(command)
        return wrapped

    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "8192")
    monkeypatch.setattr(runtime, "build_priv_drop_cmd", capture)
    env = SimpleNamespace(live_process=AsyncMock(side_effect=StopBeforeConnect))
    if saved_drop:
        env._benchflow_net_admin_dropped_for = "agent"
    with pytest.raises(StopBeforeConnect):
        await runtime.connect_acp(
            env=env,
            agent=agent,
            agent_launch='printf %s "$BENCHFLOW_ACP_LINE_LIMIT"',
            agent_env={"BENCHFLOW_ACP_LINE_LIMIT": "9999999"},
            sandbox_user="agent",
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/tmp/workspace",
        )
    result = subprocess.run(
        ["/bin/sh", "-c", commands[0]],
        env={},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "8192"


@pytest.mark.parametrize(
    "step_type", ["thinking", "thought", "planning", "reasoning", "new-step"]
)
def test_antigravity_all_thought_paths_fit(step_type, monkeypatch):
    """C3's sibling scan covers known and fallback Antigravity thought steps."""
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "8192")
    output = io.StringIO()
    monkeypatch.setattr(antigravity_acp_shim.sys, "stdout", output)
    session = antigravity_acp_shim.AgySession("fixture", "/tmp/workspace")
    text = "λ😀" * 9000
    session._handle_step({"step_type": step_type, "text_delta": text}, set())
    lines = output.getvalue().encode().splitlines(keepends=True)
    assert lines and all(len(line) <= 8192 for line in lines)
    updates = [
        json.loads(shrink_line(line, 8192))["params"]["update"] for line in lines
    ]
    assert all(update["sessionUpdate"] == "agent_thought_chunk" for update in updates)
    assert "".join(update["content"]["text"] for update in updates) == (
        f"[{step_type}] {text}" if step_type == "new-step" else text
    )
