"""Guard FF-sw-agents-C2 at 1f93158d across the actual ACP line filter."""

import io
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.session import ACPSession
from benchflow.agents import deepagents_acp_shim, harvey_lab_acp_shim
from benchflow.rollout._setup import final_agent_reply
from benchflow.sandbox.process._acp_lines import line_limit, shrink_line


@pytest.mark.parametrize("shim", [deepagents_acp_shim, harvey_lab_acp_shim])
@pytest.mark.parametrize("limit", [None, "8192", "1024", "0", "off", "invalid"])
@pytest.mark.parametrize(
    "answer",
    ["A" * 1048576, "λ😀" * 90000, '\\"\n\t\x00' * 90000, ""],
    ids=["base64-like", "unicode", "json-escaping", "empty"],
)
async def test_answer_survives_both_transport_filters(shim, limit, answer, monkeypatch):
    if limit is None:
        monkeypatch.delenv("BENCHFLOW_ACP_LINE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", limit)
    output = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", output)
    session_id = 'session-"\\😀'
    shim._emit_text(session_id, answer)
    lines = output.getvalue().encode().splitlines(keepends=True)
    effective = line_limit()
    if effective is not None:
        # Include the newline and escaped session envelope in the budget.
        assert all(len(line) <= effective for line in lines)
        lines = [shrink_line(line, effective) for line in lines]
    process = SimpleNamespace(readline=AsyncMock(side_effect=lines))
    transport = ContainerTransport(process, "unused")
    received = [await transport.receive() for _ in lines]
    assert all(message["params"]["sessionId"] == session_id for message in received)
    assert "".join(m["params"]["update"]["content"]["text"] for m in received) == answer
    session = ACPSession(session_id)
    for message in received:
        session.handle_update(message["params"]["update"])
    session._flush_agent_text()
    assert final_agent_reply(session.events) == (answer or None)


@pytest.mark.parametrize(
    "limit,expected", [(None, "1048576"), ("512", "512"), ("off", "0")]
)
async def test_transport_passes_effective_host_limit_to_shims(
    limit, expected, monkeypatch
):
    if limit is None:
        monkeypatch.delenv("BENCHFLOW_ACP_LINE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", limit)
    process = SimpleNamespace(start=AsyncMock())
    supplied = {"BENCHFLOW_ACP_LINE_LIMIT": "9999999", "OTHER": "kept"}
    transport = ContainerTransport(process, "shim", env=supplied)
    await transport.start()
    assert process.start.await_args.kwargs["env"] == {
        "BENCHFLOW_ACP_LINE_LIMIT": expected,
        "OTHER": "kept",
    }
    assert supplied["BENCHFLOW_ACP_LINE_LIMIT"] == "9999999"


@pytest.mark.parametrize("shim", [deepagents_acp_shim, harvey_lab_acp_shim])
def test_impossible_envelope_limit_fails_before_sending(shim, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "1")
    output = io.StringIO()
    monkeypatch.setattr(shim.sys, "stdout", output)
    with pytest.raises(ValueError, match=r"ACP.*limit"):
        shim._emit_text("session", "answer")
    assert output.getvalue() == ""


@pytest.mark.parametrize(
    "agent", ["deepagents", "harvey-lab-harness", "harvey-lab", "acp/deepagents"]
)
@pytest.mark.parametrize("limit,expected", [("2048", "2048"), ("off", "0")])
@pytest.mark.parametrize("saved_drop", [False, True])
async def test_shim_budget_survives_a_login_shell(
    agent, limit, expected, saved_drop, monkeypatch, tmp_path
):
    """Guards the S3 merge at 147e3505 and the post-login budget together."""
    from benchflow.acp import runtime

    class StopBeforeConnect(Exception):
        pass

    inner_commands = []

    real_priv_drop = runtime.build_priv_drop_cmd

    def capture_inner(command, user):
        # S2 makes this unconditional, even without an S3 audit record.
        wrapped = real_priv_drop(command, user)
        for option in (
            "--no-new-privs",
            "--bounding-set=-net_admin,-net_raw",
            "--inh-caps=-all",
            "--ambient-caps=-all",
        ):
            assert wrapped.count(option) == 2
        inner_commands.append(command)
        return wrapped

    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", limit)
    monkeypatch.setattr(runtime, "build_priv_drop_cmd", capture_inner)
    env = SimpleNamespace(live_process=AsyncMock(side_effect=StopBeforeConnect))
    if saved_drop:
        env._benchflow_net_admin_dropped_for = "agent"
    with pytest.raises(StopBeforeConnect):
        await runtime.connect_acp(
            env=env,
            agent=agent,
            agent_launch='printf %s "$BENCHFLOW_ACP_LINE_LIMIT"',
            agent_env={},
            sandbox_user="agent",
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/tmp/workspace",
        )
    # Replay only the inner shell with no inherited environment, as su -l does.
    result = subprocess.run(
        ["/bin/sh", "-c", inner_commands[0]],
        env={},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == expected
