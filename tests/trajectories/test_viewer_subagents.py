"""Subagent steps in the trajectory viewer payload.

Claude captures attribute a child agent's events to the tool call that spawned
it through ``parent_tool_call_id``. The viewer nests those
events under the spawning tool card instead of mixing them into the main
timeline, using the same scope and depth rules as the ATIF export.
"""

import json
from pathlib import Path

from benchflow.trajectories.export_atif import (
    MAX_SUBAGENT_DEPTH,
    trajectory_to_atif_record,
)
from benchflow.trajectories.viewer import render_jsonl_file
from benchflow.trajectories.viewer.models import iter_event_steps
from benchflow.trajectories.viewer.payload import _build_acp_payload, _normalize_steps


def _tool(call_id: str, parent: str | None = None, **extra) -> dict:
    event = {
        "type": "tool_call",
        "tool_call_id": call_id,
        "kind": "think",
        "title": "Task",
        "status": "completed",
        "content": [],
        **extra,
    }
    if parent is not None:
        event["parent_tool_call_id"] = parent
    return event


def _say(text: str, parent: str | None = None) -> dict:
    event = {"type": "agent_message", "text": text}
    if parent is not None:
        event["parent_tool_call_id"] = parent
    return event


def _payload(tmp_path: Path, events: list[dict]) -> dict:
    (tmp_path / "trajectory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    return _build_acp_payload(tmp_path, None).to_payload()


CLAUDE_CHILD = [
    {"type": "user_message", "text": "use one child agent"},
    _tool(
        "toolu_task",
        raw_input={
            "description": "Create child.txt",
            "prompt": "create /app/child.txt",
            "subagent_type": "general-purpose",
        },
    ),
    {
        "type": "agent_thought",
        "text": "child thinking",
        "parent_tool_call_id": "toolu_task",
    },
    _tool("toolu_bash", "toolu_task", kind="execute", title="printf x > f"),
    _tool("toolu_xxd", "toolu_task", kind="execute", title="xxd f", status="failed"),
    _say("child done", "toolu_task"),
    _say("root continues"),
]


def test_child_events_nest_under_the_spawning_tool_call(tmp_path):
    """Guards the viewer against mixing the child attribution fix's attributed child events
    into the main timeline: they ride on the spawning tool step instead."""
    payload = _payload(tmp_path, CLAUDE_CHILD)
    steps = payload["steps"]
    assert [step["kind"] for step in steps] == ["prompt", "tool", "message"]
    subagent = steps[1]["subagent"]
    assert subagent["parent_tool_call_id"] == "toolu_task"
    assert subagent["depth"] == 1
    assert subagent["subagent_type"] == "general-purpose"
    assert subagent["description"] == "Create child.txt"
    # Capture order is kept: nested steps keep their global event numbers.
    assert [step["i"] for step in subagent["steps"]] == [3, 4, 5, 6]
    assert steps[2]["i"] == 7
    assert "subagent" not in steps[0] and "subagent" not in steps[2]


def test_counts_include_subagent_events_and_report_them(tmp_path):
    """Guards the header/Metrics counts against dropping nested events."""
    counts = _payload(tmp_path, CLAUDE_CHILD)["meta"]["counts"]
    assert counts == {
        "prompts": 1,
        "messages": 2,
        "thoughts": 1,
        "tools": 3,
        "subagents": 1,
        "subagent_events": 4,
    }


def test_runs_without_subagents_keep_their_exact_wire_shape(tmp_path):
    """Guards PR #1034's payload shape for the common, subagent-free run."""
    payload = _payload(tmp_path, [_say("hi"), _tool("c1")])
    assert payload["meta"]["counts"] == {
        "prompts": 0,
        "messages": 1,
        "thoughts": 0,
        "tools": 1,
    }
    assert all("subagent" not in step for step in payload["steps"])


def test_nesting_follows_the_atif_depth_cap(tmp_path):
    """Guards parity with the ATIF subagent export: subagents nest exactly as deep as the
    ATIF export embeds them; deeper scopes become a labelled group."""
    events = [_tool("a0")]
    for level in range(1, MAX_SUBAGENT_DEPTH + 2):
        events.append(_tool(f"a{level}", f"a{level - 1}"))
    events.append(_say("deepest", f"a{MAX_SUBAGENT_DEPTH + 1}"))
    steps = _payload(tmp_path, events)["steps"]

    depth = 0
    node = steps[0]
    while "subagent" in node:
        depth += 1
        assert node["subagent"]["depth"] == depth
        node = node["subagent"]["steps"][0]
    assert depth == MAX_SUBAGENT_DEPTH

    atif_depth = 0
    record = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    while record.get("subagent_trajectories"):
        atif_depth += 1
        (record,) = record["subagent_trajectories"]
    assert atif_depth == depth

    # The export drops the scopes past the cap; the viewer keeps them, labelled.
    groups = {
        step["subagent"]["parent_tool_call_id"]: step
        for step in steps
        if step["kind"] == "subagent"
    }
    too_deep = {f"a{MAX_SUBAGENT_DEPTH}", f"a{MAX_SUBAGENT_DEPTH + 1}"}
    assert set(groups) == too_deep
    assert {group["reason"] for group in groups.values()} == {"too_deep"}
    deepest = groups[f"a{MAX_SUBAGENT_DEPTH + 1}"]["subagent"]["steps"]
    assert [step["text"] for step in deepest] == ["deepest"]


def test_unattributed_children_render_as_a_labelled_group_in_place(tmp_path):
    """Guards the viewer against hiding events whose spawning call was not
    captured (the ATIF export embeds these at the root without a reference)."""
    events = [
        _say("root first"),
        _say("orphan one", "toolu_missing"),
        _tool("child_call", "toolu_missing"),
        _say("grandchild", "child_call"),
        _say("root last"),
    ]
    steps = _payload(tmp_path, events)["steps"]
    assert [step["kind"] for step in steps] == ["message", "subagent", "message"]
    group = steps[1]
    assert group["reason"] == "parent_not_captured"
    assert group["gid"] == "g1"
    inner = group["subagent"]["steps"]
    assert [step["i"] for step in inner] == [2, 3]
    # A captured spawn inside the unattributed scope still nests normally.
    assert inner[1]["subagent"]["steps"][0]["text"] == "grandchild"
    assert inner[1]["subagent"]["depth"] == 2


def test_cyclic_attribution_is_shown_not_dropped(tmp_path):
    """Guards against self/cyclic parent attribution recursing or vanishing."""
    events = [
        _say("root"),
        _tool("x", "y"),
        _tool("y", "x"),
        _tool("self", "self"),
    ]
    steps = _payload(tmp_path, events)["steps"]
    groups = [step for step in steps if step["kind"] == "subagent"]
    assert {group["reason"] for group in groups} == {"cyclic"}
    shown = {
        step["tool"]["id"] for group in groups for step in group["subagent"]["steps"]
    }
    assert shown == {"x", "y", "self"}


def test_every_event_appears_exactly_once(tmp_path):
    """Guards the nesting against losing or duplicating any captured event."""
    events = [*CLAUDE_CHILD, _say("orphan", "nope"), _tool("z", "z")]
    steps = _normalize_steps(events, None)
    flat = iter_event_steps(steps)
    assert [step.i for step in flat] == list(range(1, len(events) + 1))


def test_legacy_session_page_stays_flat_and_complete(tmp_path):
    """Guards the inline ACP session renderer (PR #1034) against losing nested
    subagent events: it renders every step in capture order."""
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in CLAUDE_CHILD))
    page = render_jsonl_file(path)
    positions = [page.index(text) for text in ("child done", "root continues")]
    assert positions == sorted(positions)
    assert "xxd f" in page


def test_subagent_spawn_is_labelled_agent_not_think(tmp_path):
    """Regression test: the ``Task`` call that spawns a subagent
    showed the badge ``think``, the ACP kind claude-agent-acp (Task/Agent)
    and opencode (task) report for it. The viewer labels it ``agent``; a
    genuine think call keeps its kind."""
    spawn = {"type": "tool_call", "kind": "think", "status": "completed"}
    events = [
        {**spawn, "tool_call_id": "claude", "title": "Task"},
        {
            **spawn,
            "tool_call_id": "claude-described",
            "title": "Create child.txt",
            "raw_input": {"description": "Create child.txt", "prompt": "go"},
        },
        {**spawn, "tool_call_id": "opencode", "title": "task"},
        {
            **spawn,
            "tool_call_id": "opencode-args",
            "title": 'task {"description": "Read the 2024 filings"}',
        },
        {**spawn, "tool_call_id": "plan", "title": "Plan the next step"},
    ]
    tools = [step["tool"] for step in _payload(tmp_path, events)["steps"]]
    assert [tool["kind"] for tool in tools] == ["agent"] * 4 + ["think"]
    # Same agent palette as before; a title word such as "Read" never
    # reclassifies the spawn.
    assert [tool["hue"] for tool in tools] == ["skill"] * 4 + ["think"]
