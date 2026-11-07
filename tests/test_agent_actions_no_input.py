"""A call of an unknown kind stays in the coverage summary, input or not.

The action records decide by what a call does, not by which program named
it. An unknown kind with no input has no command, path or URL to witness, so
it gives no record, but it is still counted as unmapped: a tool with no
arguments can still act (a reset, a restart), so the reported coverage gap
keeps it. The same kind carrying a path is witnessed.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchflow.integrity.agent_actions import action_records_from_trajectory


def _call(kind: str, call_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "kind": kind,
        "status": "completed",
        **extra,
    }


@pytest.mark.parametrize(
    "event",
    [
        _call("environment_observation", "a"),
        _call("mcp__db__reset_database", "b", arguments={}),
        _call("restart_service", "c", raw_input={"note": "", "data": None}),
        _call("custom_report", "d", arguments={"items": []}),
    ],
    ids=["no-arguments", "empty-arguments", "empty-values", "empty-list"],
)
def test_a_call_with_no_input_gives_no_record_and_stays_unmapped(event):
    lane = action_records_from_trajectory([event], agent_cwd="/app")
    assert lane.records == []
    assert lane.summary["unmapped_kinds"] == {event["kind"]: 1}
    assert lane.summary["kind_counts"] == {event["kind"]: 1}


def test_the_same_kind_with_a_path_is_witnessed():
    event = _call(
        "environment_observation", "p", arguments={"path": "/app/tests/test.py"}
    )
    lane = action_records_from_trajectory(
        [event], agent_cwd="/app", manifest_resources=[("/app", "AgentWritable")]
    )
    assert len(lane.records) == 1
    assert lane.records[0]["source_path"] == "/app/tests/test.py"
    assert lane.summary["unmapped_kinds"] == {}


def test_the_same_kind_with_other_input_stays_unmapped():
    event = _call("environment_observation", "o", arguments={"payload": "reset"})
    lane = action_records_from_trajectory([event], agent_cwd="/app")
    assert lane.records == []
    assert lane.summary["unmapped_kinds"] == {"environment_observation": 1}


def test_a_known_command_without_its_input_is_still_a_gap():
    """Known kinds keep their own handling: a Bash call with no command is a
    coverage gap (dropped_missing_input)."""
    lane = action_records_from_trajectory([_call("Bash", "x")], agent_cwd="/app")
    assert lane.records == []
    assert lane.summary["dropped_missing_input"] == {"Bash": 1}
