"""Tool-call content and raw I/O capture (issue #1099).

codex-acp reports a command as ``rawInput`` on the opening ``tool_call``, its
output as ``rawOutput`` on the completing ``tool_call_update``, and file-change
diffs as ``content`` on the opening ``tool_call``. Before the fix every one of
those was dropped and the trajectory kept only the title.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import _events_to_trajectory
from benchflow.trajectories.viewer.payload import _build_acp_payload, _raw_io_texts


def _codex_exec_session() -> ACPSession:
    session = ACPSession("s")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call_1",
            "title": "ls /root",
            "kind": "execute",
            "status": "in_progress",
            "rawInput": {"command": "ls /root", "cwd": "/root"},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call_1",
            "status": "completed",
            "rawOutput": {"formatted_output": "a.py\nb.py\n", "exit_code": 0},
        }
    )
    return session


def test_session_keeps_raw_input_and_output() -> None:
    """Guards the fix for issue #1099: rawInput from the opening tool_call and
    rawOutput from the completing update both survive on the record."""
    record = _codex_exec_session().tool_calls[0]
    assert record.raw_input == {"command": "ls /root", "cwd": "/root"}
    assert record.raw_output == {"formatted_output": "a.py\nb.py\n", "exit_code": 0}


def test_session_keeps_content_sent_on_the_opening_tool_call() -> None:
    """Guards the fix for issue #1099: codex-acp file-change diffs arrive as
    content on the opening tool_call, not on an update."""
    session = ACPSession("s")
    diff = {"type": "diff", "path": "/root/a.py", "oldText": None, "newText": "x = 1\n"}
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call_2",
            "title": "Editing files",
            "kind": "edit",
            "status": "completed",
            "content": [diff],
        }
    )
    assert session.tool_calls[0].content == [diff]


def test_opening_call_status_is_honored() -> None:
    """Guards the follow-up on #1099 review: a file edit that codex-acp reports
    completed in its opening tool_call (no later update) is terminal, so it
    never lingers in pending_tool_call_ids(); an opening call without a status
    stays pending as before."""
    session = ACPSession("s")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "done",
            "title": "Editing files",
            "kind": "edit",
            "status": "completed",
            "content": [
                {"type": "diff", "path": "/a", "oldText": None, "newText": "x"}
            ],
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "open",
            "title": "ls",
            "kind": "execute",
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "odd",
            "title": "t",
            "kind": "other",
            "status": "not-a-status",
        }
    )
    assert session.pending_tool_call_ids() == ["open", "odd"]
    assert session.tool_calls[0].finished_at is not None


def test_trajectory_emits_raw_fields_only_when_present() -> None:
    """Guards the fix for issue #1099 without changing the shape agents that
    send no raw I/O produce: the keys appear only when set."""
    with_raw = _events_to_trajectory(_codex_exec_session().events)[0]
    assert with_raw["raw_input"] == {"command": "ls /root", "cwd": "/root"}
    assert with_raw["raw_output"]["exit_code"] == 0

    bare = ACPSession("s")
    bare.handle_update(
        {"sessionUpdate": "tool_call", "toolCallId": "t", "title": "t", "kind": "read"}
    )
    assert set(_events_to_trajectory(bare.events)[0]) == {
        "type",
        "tool_call_id",
        "kind",
        "title",
        "status",
        "content",
    }


def test_raw_io_texts_render_commands_outputs_and_exit_codes() -> None:
    """Execute calls read as command then output; a nonzero exit is stated;
    other shapes fall back to JSON so nothing recorded is hidden."""
    texts = _raw_io_texts(
        "execute",
        {"command": ["python", "-c", "print(1)"], "cwd": "/"},
        {"formatted_output": "boom\n", "exit_code": 2},
    )
    assert texts == ["python -c print(1)", "boom\n[exit code 2]"]
    assert _raw_io_texts("other", {"server": "mcp", "tool": "t"}, None) == [
        '{"server": "mcp", "tool": "t"}'
    ]
    assert _raw_io_texts("execute", None, None) == []


def test_payload_falls_back_to_raw_io_when_content_is_empty(tmp_path: Path) -> None:
    """Guards the viewer side of issue #1099: a codex-style tool call with no
    content blocks still renders its command and output; content wins when
    both exist."""
    traj = tmp_path / "trajectory"
    traj.mkdir()
    events = [
        {
            "type": "tool_call",
            "tool_call_id": "c1",
            "kind": "execute",
            "title": "ls /root",
            "status": "completed",
            "content": [],
            "raw_input": {"command": "ls /root", "cwd": "/root"},
            "raw_output": {"formatted_output": "a.py\n", "exit_code": 0},
        },
        {
            "type": "tool_call",
            "tool_call_id": "c2",
            "kind": "execute",
            "title": "cat x",
            "status": "completed",
            "content": [{"type": "content", "content": {"type": "text", "text": "x"}}],
            "raw_output": {"formatted_output": "ignored", "exit_code": 0},
        },
    ]
    (traj / "acp_trajectory.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "result.json").write_text("{}")
    steps = _build_acp_payload(tmp_path, None).to_payload()["steps"]
    tools = [s["tool"] for s in steps if s["kind"] == "tool"]
    assert tools[0]["content"] == ["ls /root", "a.py\n"]
    assert tools[1]["content"] == ["x"]
