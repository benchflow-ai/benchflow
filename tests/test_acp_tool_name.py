"""The agent's own tool name is kept on each recorded tool call.

Branch-tree training rows used to name tool calls by
their ACP display title ("Write hello.txt", "cp /app/draft.txt …") because
the trajectory never recorded the tool's name. Claude's ACP adapter sends it
as ``_meta.claudeCode.toolName``; the session now keeps it as ``tool_name``
on the trajectory event (absent when the agent does not send it).
"""

from __future__ import annotations

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import _capture_session_trajectory


def _call(meta=None, **extra):
    update = {
        "sessionUpdate": "tool_call",
        "toolCallId": "t1",
        "title": "Write hello.txt",
        "kind": "edit",
        "status": "completed",
        "rawInput": {"file_path": "/app/hello.txt", "content": "hi"},
        **extra,
    }
    if meta is not None:
        update["_meta"] = meta
    return update


def test_claude_tool_name_is_recorded():
    session = ACPSession("s")
    session.handle_update(_call({"claudeCode": {"toolName": "Write"}}))
    [event] = [
        e for e in _capture_session_trajectory(session) if e["type"] == "tool_call"
    ]
    assert event["tool_name"] == "Write"
    assert event["title"] == "Write hello.txt"


def test_a_later_update_can_supply_it():
    session = ACPSession("s")
    session.handle_update(_call(status="in_progress"))
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "t1",
            "status": "completed",
            "_meta": {"claudeCode": {"toolName": "Write"}},
        }
    )
    [event] = [
        e for e in _capture_session_trajectory(session) if e["type"] == "tool_call"
    ]
    assert event["tool_name"] == "Write"


def test_without_it_there_is_no_field():
    session = ACPSession("s")
    session.handle_update(_call())
    [event] = [
        e for e in _capture_session_trajectory(session) if e["type"] == "tool_call"
    ]
    assert "tool_name" not in event
