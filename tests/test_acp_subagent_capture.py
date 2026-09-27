"""Received child-tool attribution, distinct from PR #1057 provider capture.

Fixtures match claude-agent-acp 0.73.0 ensureToolCallEmitted/toAcpNotifications.
No transcript/native-session capability is enabled by this change.
"""

import json

import pytest

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _snapshot_session_trajectory,
)


@pytest.mark.parametrize("first_type", ["tool_call", "tool_call_update"])
def test_received_child_parent_survives_live_final_and_disk(first_type, tmp_path):
    """Child-trace fix: retain producer attribution; separate from PR #1057."""
    session = ACPSession("root")
    session.handle_update(
        {
            "sessionUpdate": first_type,
            "toolCallId": "child-read",
            "title": "Read file",
            "kind": "read",
            "status": "in_progress",
            "_meta": {
                "claudeCode": {"parentToolUseId": "spawn-agent"},
                "unrelated_private_extension": "do-not-export",
            },
        }
    )
    live = _snapshot_session_trajectory(session)
    assert live[0]["parent_tool_call_id"] == "spawn-agent"
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "child-read",
            "status": "completed",
        }
    )
    final = _capture_session_trajectory(session)
    assert final[0]["parent_tool_call_id"] == "spawn-agent"
    assert final[0]["status"] == "completed"
    writer = TrajectoryWriter(tmp_path / "acp.jsonl")
    writer.write_events(final)
    persisted = json.loads(writer.path.read_text())
    assert persisted == final[0]
    assert "do-not-export" not in writer.path.read_text()
    assert live[0]["status"] == "in_progress"


def test_late_parent_metadata_and_conflict_keep_first_attribution():
    """Fix beside PR #1057: late metadata is retained, not silently reassigned."""
    session = ACPSession("root")
    session.handle_update({"sessionUpdate": "tool_call", "toolCallId": "child"})
    for parent in ("parent-a", "parent-b"):
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "child",
                "_meta": {"claudeCode": {"parentToolUseId": parent}},
            }
        )
    assert _capture_session_trajectory(session)[0]["parent_tool_call_id"] == "parent-a"


@pytest.mark.parametrize(
    "meta",
    [
        None,
        [],
        "bad",
        {"claudeCode": None},
        {"claudeCode": {"parentToolUseId": 42}},
        {"claudeCode": {"parentToolUseId": ""}},
    ],
)
def test_malformed_or_absent_attribution_does_not_invent_parent(meta):
    """Fix beside PR #1057: ordinary calls retain unknown parentage."""
    session = ACPSession("root")
    session.handle_update(
        {"sessionUpdate": "tool_call", "toolCallId": "call", "_meta": meta}
    )
    assert "parent_tool_call_id" not in _capture_session_trajectory(session)[0]
