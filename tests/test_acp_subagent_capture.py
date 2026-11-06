"""Received child-tool attribution, distinct from PR #1057 provider capture.

Fixtures match claude-agent-acp ensureToolCallEmitted/toAcpNotifications and
its tool_progress heartbeat, whose shapes are the same from 0.73.0 through
0.81.2. No transcript/native-session capability is enabled by this change.
"""

import json

import pytest

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _snapshot_session_trajectory,
)
from benchflow.trajectories.export_atif import trajectory_to_atif_record


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


def test_heartbeat_naming_a_call_its_own_parent_keeps_the_subagent_reachable():
    """claude-agent-acp's tool_progress beat for an Agent call whose subagent is
    live reports the call with its own id as parentToolUseId (the beat's
    tool_use_id is a synthetic ``<id>-heartbeat-<n>``, so the adapter falls back
    to parent_tool_use_id for both fields). Recorded as parentage, the Agent
    call would sit in its own scope, and the ATIF export would leave out the
    call and its whole subagent as unreachable."""
    session = ACPSession("root")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "spawn",
            "title": "Task",
            "kind": "think",
            "status": "pending",
            "_meta": {"claudeCode": {"toolName": "Agent"}},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "spawn",
            "status": "in_progress",
            "_meta": {
                "claudeCode": {
                    "toolName": "Agent",
                    "parentToolUseId": "spawn",
                    "toolResponse": {"elapsedTimeSeconds": 31},
                }
            },
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "child-read",
            "title": "Read notes.md",
            "kind": "read",
            "status": "completed",
            "_meta": {"claudeCode": {"parentToolUseId": "spawn"}},
        }
    )
    events = _capture_session_trajectory(session)
    assert "parent_tool_call_id" not in events[0]
    assert events[1]["parent_tool_call_id"] == "spawn"
    record = trajectory_to_atif_record(
        session_id="root", agent_name="claude-agent-acp", events=events
    )
    assert [sub["trajectory_id"] for sub in record["subagent_trajectories"]] == [
        "subagent:spawn"
    ]
