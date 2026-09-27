"""Transcript fix after the child attribution fix: root trainer projections cannot mix children."""

import copy
import json

from benchflow.trajectories.export import (
    acp_events_to_messages,
    write_rollout_verifiers_jsonl,
)
from benchflow.trajectories.export_adp import trajectory_to_adp_record
from benchflow.trajectories.export_atif import (
    trajectory_to_atif_record,
    write_rollout_atif_json,
)


def interleaved():
    return [
        {"type": "agent_thought", "text": "root-reason"},
        {"type": "agent_thought", "text": "CHILD1-thought", "parent_tool_call_id": "a"},
        {
            "type": "tool_call",
            "tool_call_id": "c1",
            "title": "CHILD1-tool",
            "parent_tool_call_id": "a",
        },
        {"type": "agent_message", "text": "CHILD2-answer", "parent_tool_call_id": "b"},
        {"type": "tool_call", "tool_call_id": "root-call", "title": "root-tool"},
        {"type": "agent_thought", "text": "CHILD2-thought", "parent_tool_call_id": "b"},
        {"type": "agent_message", "text": "root-answer"},
    ]


def test_root_exports_declare_omissions_and_never_attach_child_thoughts(tmp_path):
    """Guards the fix after the child attribution fix: root-only scope explicit, raw ACP unmodified.

    ATIF left the root-only group when it began embedding children as
    ATIF-v1.7 subagent trajectories; its root steps must still carry no
    child text (see tests/trajectories/test_atif_subagents.py).
    """
    events = interleaved()
    before = copy.deepcopy(events)
    atif = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    assert "CHILD" not in json.dumps(atif["steps"])
    assert atif["extra"]["acp_projection"]["excluded_attributed_child_events"] == 0
    adp = trajectory_to_adp_record(trajectory_id="s", events=events)
    vf = write_rollout_verifiers_jsonl(
        tmp_path,
        task_id="t",
        prompts=[],
        trajectory=events,
        rewards={"reward": 1},
        model=None,
        environment="docker",
    )
    for record, metadata in [
        (adp, adp["details"]),
        (vf, vf["info"]),
    ]:
        assert "CHILD" not in json.dumps(record)
        coverage = metadata["acp_projection"]
        assert coverage["scope"] == "root_agent_projection"
        assert coverage["excluded_attributed_child_events"] == 4
        assert coverage["child_transcript_completeness"] == "unknown"
        assert coverage["source_artifact"] == "trajectory/acp_trajectory.jsonl"
    assert atif["steps"][0]["reasoning_content"] == "root-reason"
    assert adp["content"][0]["reasoning_content"] == "root-reason"
    assert events == before
    assert "CHILD" not in json.dumps(acp_events_to_messages(events))
    persisted = json.loads((tmp_path / "trainer/verifiers.jsonl").read_text())
    assert persisted["info"]["acp_projection"] == vf["info"]["acp_projection"]


def test_unattributed_legacy_events_keep_existing_record_shape():
    """Guards the fix after the child attribution fix: never infer parentage from tool name or title."""
    events = [{"type": "agent_message", "text": "legacy child maybe"}]
    atif = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    adp = trajectory_to_adp_record(trajectory_id="s", events=events)
    assert "extra" not in atif
    assert "acp_projection" not in adp["details"]
    assert atif["steps"][0]["message"] == "legacy child maybe"


def test_child_only_atif_has_discoverable_coverage_without_fake_step(tmp_path):
    """Guards the fix after the child attribution fix: empty root ATIF must not fabricate a step."""
    events = [
        {"type": "agent_message", "text": "child-only", "parent_tool_call_id": "a"}
    ]
    result = write_rollout_atif_json(
        tmp_path, session_id="s", agent_name="a", prompts=[], trajectory=events
    )
    assert result is None
    assert not (tmp_path / "trainer/atif.json").exists()
    coverage = json.loads((tmp_path / "trainer/atif.coverage.json").read_text())
    assert coverage["acp_projection"]["excluded_attributed_child_events"] == 1
    assert coverage["artifact_status"] == "omitted_empty_root_projection"


def test_atif_empty_child_coverage_is_removed_when_projection_changes(tmp_path):
    """Guards the fix after the child attribution fix: stale omission receipts must not describe later exports."""
    child = [{"type": "agent_message", "text": "child", "parent_tool_call_id": "a"}]
    write_rollout_atif_json(
        tmp_path, session_id="s", agent_name="a", prompts=[], trajectory=child
    )
    receipt = tmp_path / "trainer/atif.coverage.json"
    assert receipt.exists()
    write_rollout_atif_json(
        tmp_path, session_id="s", agent_name="a", prompts=[], trajectory=[]
    )
    assert not receipt.exists()


def test_empty_atif_projection_removes_previous_root_artifact(tmp_path):
    """Transcript artifact integrity: an empty replacement must not retain old steps."""
    write_rollout_atif_json(
        tmp_path,
        session_id="s",
        agent_name="a",
        prompts=[],
        trajectory=[{"type": "agent_message", "text": "previous answer"}],
    )
    artifact = tmp_path / "trainer/atif.json"
    assert artifact.exists()
    result = write_rollout_atif_json(
        tmp_path,
        session_id="s",
        agent_name="a",
        prompts=[],
        trajectory=[],
    )
    assert result is None
    assert not artifact.exists()
    assert not (tmp_path / "trainer/atif.coverage.json").exists()
