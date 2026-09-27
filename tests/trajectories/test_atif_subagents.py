"""ATIF-v1.7 native subagent export.

The child transcript filter stopped attributed Claude child events from leaking into the
root ATIF steps (earlier releases, since 2b6fed8b, interleaved them inline) by
dropping them and keeping only a count. These tests pin the replacement: each
child becomes an embedded ``subagent_trajectories`` entry referenced from the
spawning tool call's ``observation.results[].subagent_trajectory_ref``.

``_atif_problems`` mirrors the Harbor pydantic validators
(``harbor-framework/harbor`` ``src/harbor/models/trajectories/*.py``) and adds
the reference resolution Harbor leaves to consumers. The Harbor-model test at
the bottom runs only where ``harbor`` is importable, e.g.
``uv run --with harbor==0.23.0 pytest tests/trajectories/test_atif_subagents.py``.
"""

import json

import pytest

from benchflow.trajectories.export_atif import (
    MAX_SUBAGENT_DEPTH,
    trajectory_to_atif_record,
    write_rollout_atif_json,
)

_AGENT_ONLY = ("reasoning_content", "tool_calls", "metrics", "model_name")


def _atif_problems(doc, where="root"):
    """Structural ATIF-v1.7 problems of *doc* and its embedded subagents."""
    problems = []
    steps = doc.get("steps") or []
    if not steps:
        problems.append(f"{where}: no steps")
    embedded = {
        sub.get("trajectory_id"): sub for sub in doc.get("subagent_trajectories") or []
    }
    if None in embedded or len(embedded) != len(doc.get("subagent_trajectories") or []):
        problems.append(f"{where}: embedded trajectory_id missing or duplicated")
    for index, step in enumerate(steps, start=1):
        if step.get("step_id") != index:
            problems.append(f"{where}: step_id {step.get('step_id')} != {index}")
        if step.get("source") != "agent":
            problems += [
                f"{where}: {key} on non-agent step"
                for key in _AGENT_ONLY
                if key in step
            ]
        call_ids = {call["tool_call_id"] for call in step.get("tool_calls") or []}
        for result in (step.get("observation") or {}).get("results", []):
            if result.get("source_call_id") not in call_ids:
                problems.append(f"{where}: dangling source_call_id")
            for ref in result.get("subagent_trajectory_ref") or []:
                if ref.get("trajectory_id") not in embedded:
                    problems.append(f"{where}: unresolved ref {ref}")
    for trajectory_id, sub in embedded.items():
        problems += _atif_problems(sub, where=f"{where}>{trajectory_id}")
    return problems


def _refs(doc):
    """``{tool_call_id: [trajectory_id, ...]}`` for *doc*'s own steps."""
    out = {}
    for step in doc["steps"]:
        for result in (step.get("observation") or {}).get("results", []):
            if result.get("subagent_trajectory_ref"):
                out[result["source_call_id"]] = [
                    ref["trajectory_id"] for ref in result["subagent_trajectory_ref"]
                ]
    return out


def _spawn(call_id, prompt, *, parent=None, output="done"):
    event = {
        "type": "tool_call",
        "tool_call_id": call_id,
        "kind": "think",
        "title": "Task",
        "status": "completed",
        "raw_input": {
            "description": "delegate",
            "prompt": prompt,
            "subagent_type": "general-purpose",
        },
        "content": [{"type": "content", "content": {"type": "text", "text": output}}],
    }
    if parent is not None:
        event["parent_tool_call_id"] = parent
    return event


def _child(parent, event_type, text):
    return {"type": event_type, "text": text, "parent_tool_call_id": parent}


def claude_task_rollout():
    """A root that delegates twice; the two children interleave with the root."""
    return [
        {"type": "agent_thought", "text": "root-reason"},
        _spawn("toolu_01A", "Find the config loader.", output="loader is cfg.py"),
        _child("toolu_01A", "agent_thought", "CHILD-A-thought"),
        {
            "type": "tool_call",
            "tool_call_id": "toolu_01A1",
            "kind": "search",
            "title": "grep load_config",
            "status": "completed",
            "content": [{"text": "cfg.py:12"}],
            "parent_tool_call_id": "toolu_01A",
        },
        _spawn("toolu_01B", "Summarise the tests."),
        _child("toolu_01B", "agent_message", "CHILD-B-answer"),
        _child("toolu_01A", "agent_message", "CHILD-A-answer"),
        {"type": "agent_message", "text": "root-answer"},
    ]


def test_children_embed_as_referenced_subagent_trajectories():
    """Guards ATIF subagent export against the child-event drop of the child transcript filter."""
    record = trajectory_to_atif_record(
        session_id="task__1",
        agent_name="claude-agent-acp",
        events=claude_task_rollout(),
        prompts=["Solve."],
    )
    assert _atif_problems(record) == []
    assert "CHILD" not in json.dumps(record["steps"])
    assert record["steps"][1]["reasoning_content"] == "root-reason"
    assert _refs(record) == {
        "toolu_01A": ["subagent:toolu_01A"],
        "toolu_01B": ["subagent:toolu_01B"],
    }
    spawn_result = record["steps"][1]["observation"]["results"][0]
    assert spawn_result["content"] == "loader is cfg.py"

    child_a, child_b = record["subagent_trajectories"]
    assert child_a["trajectory_id"] == "subagent:toolu_01A"
    assert "session_id" not in child_a
    assert child_a["agent"] == {
        "name": "claude-agent-acp",
        "version": "unknown",
        "extra": {"subagent_type": "general-purpose"},
    }
    assert [(s["source"], s["message"]) for s in child_a["steps"]] == [
        ("user", "Find the config loader."),
        ("agent", ""),
        ("agent", "CHILD-A-answer"),
    ]
    assert child_a["steps"][1]["reasoning_content"] == "CHILD-A-thought"
    assert child_a["steps"][1]["tool_calls"][0]["tool_call_id"] == "toolu_01A1"
    assert child_a["extra"] == {
        "parent_tool_call_id": "toolu_01A",
        "prompt_source": "parent_tool_call.raw_input.prompt",
    }
    assert [s["message"] for s in child_b["steps"]] == [
        "Summarise the tests.",
        "CHILD-B-answer",
    ]
    assert record["extra"]["acp_projection"] == {
        "scope": "root_with_embedded_subagents",
        "attributed_child_events": 4,
        "embedded_subagent_trajectories": 2,
        "unlinked_subagent_trajectories": 0,
        "excluded_attributed_child_events": 0,
        "max_subagent_depth": MAX_SUBAGENT_DEPTH,
        "source_artifact": "trajectory/acp_trajectory.jsonl",
        "child_transcript_completeness": "unknown",
        "outcome_and_usage_scope": "rollout_not_partitioned_by_agent",
    }


def test_spawn_without_output_still_gets_an_observation_ref():
    """Guards ATIF subagent export (after the child transcript filter): the ref needs no tool output."""
    spawn = _spawn("toolu_01A", "Look around.")
    spawn["content"] = []
    record = trajectory_to_atif_record(
        session_id="s",
        agent_name="a",
        events=[spawn, _child("toolu_01A", "agent_message", "child answer")],
    )
    (result,) = record["steps"][0]["observation"]["results"]
    assert result == {
        "source_call_id": "toolu_01A",
        "subagent_trajectory_ref": [{"trajectory_id": "subagent:toolu_01A"}],
    }
    assert _atif_problems(record) == []


def test_nested_child_embeds_under_its_own_parent():
    """Guards ATIF subagent export (after the child transcript filter): grandchildren nest, never flatten."""
    events = [
        _spawn("t1", "level one"),
        _spawn("t2", "level two", parent="t1"),
        _child("t2", "agent_message", "grandchild answer"),
        _child("t1", "agent_message", "child answer"),
    ]
    record = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    assert _atif_problems(record) == []
    (child,) = record["subagent_trajectories"]
    assert child["trajectory_id"] == "subagent:t1"
    assert _refs(child) == {"t2": ["subagent:t2"]}
    (grandchild,) = child["subagent_trajectories"]
    assert [s["message"] for s in grandchild["steps"]] == [
        "level two",
        "grandchild answer",
    ]
    assert record["extra"]["acp_projection"]["embedded_subagent_trajectories"] == 2


def test_uncaptured_spawn_embeds_unreferenced_at_root():
    """Guards ATIF subagent export (after the child transcript filter): orphans stay visible, unreferenced."""
    events = [
        {"type": "agent_message", "text": "root"},
        _child("lost", "agent_message", "orphan answer"),
    ]
    record = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    assert _atif_problems(record) == []
    assert _refs(record) == {}
    (orphan,) = record["subagent_trajectories"]
    assert orphan["extra"] == {
        "parent_tool_call_id": "lost",
        "parent_tool_call_captured": False,
    }
    assert orphan["steps"] == [
        {"step_id": 1, "source": "agent", "message": "orphan answer"}
    ]
    coverage = record["extra"]["acp_projection"]
    assert coverage["unlinked_subagent_trajectories"] == 1
    assert coverage["excluded_attributed_child_events"] == 0


def test_cyclic_attribution_terminates_and_is_counted():
    """Guards ATIF subagent export (after the child transcript filter): malformed parentage cannot loop."""
    events = [
        {"type": "agent_message", "text": "root"},
        _spawn("x", "x prompt", parent="y"),
        _spawn("y", "y prompt", parent="x"),
        _spawn("z", "self", parent="z"),
    ]
    record = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    assert "subagent_trajectories" not in record
    coverage = record["extra"]["acp_projection"]
    assert coverage["attributed_child_events"] == 3
    assert coverage["excluded_attributed_child_events"] == 3
    assert coverage["embedded_subagent_trajectories"] == 0


def test_nesting_beyond_the_depth_cap_is_left_out_and_counted():
    """Guards ATIF subagent export (after the child transcript filter): the nesting cap is explicit."""
    depth = MAX_SUBAGENT_DEPTH + 2
    events = [_spawn("t1", "p1")]
    for level in range(2, depth + 1):
        events.append(_spawn(f"t{level}", f"p{level}", parent=f"t{level - 1}"))
    events.append(_child(f"t{depth}", "agent_message", "deepest"))
    record = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    assert _atif_problems(record) == []
    chain, node = [], record
    while node.get("subagent_trajectories"):
        (node,) = node["subagent_trajectories"]
        chain.append(node["trajectory_id"])
    assert chain == [f"subagent:t{level}" for level in range(1, MAX_SUBAGENT_DEPTH + 1)]
    # The deepest embedded child still shows its spawn call, just unreferenced.
    assert _refs(node) == {}
    assert "deepest" not in json.dumps(record)
    coverage = record["extra"]["acp_projection"]
    assert coverage["embedded_subagent_trajectories"] == MAX_SUBAGENT_DEPTH
    assert coverage["excluded_attributed_child_events"] == depth - MAX_SUBAGENT_DEPTH


def test_stepless_child_is_neither_embedded_nor_referenced():
    """Guards ATIF subagent export (after the child transcript filter): no empty subagent trajectories."""
    spawn = _spawn("t1", "unused")
    spawn["raw_input"] = None
    events = [spawn, _child("t1", "agent_message", "")]
    record = trajectory_to_atif_record(session_id="s", agent_name="a", events=events)
    assert _atif_problems(record) == []
    assert "subagent_trajectories" not in record
    assert _refs(record) == {}
    assert record["extra"]["acp_projection"]["excluded_attributed_child_events"] == 1


def test_written_artifact_keeps_refs_resolvable_and_redacts_children(tmp_path):
    """Guards ATIF subagent export (after the child transcript filter) through redaction and disk."""
    events = claude_task_rollout()
    events.append(
        _child("toolu_01B", "agent_message", "OPENAI_API_KEY=sk-abc123def456ghi789")
    )
    record = write_rollout_atif_json(
        tmp_path,
        session_id="task__1",
        agent_name="claude-agent-acp",
        prompts=["Solve."],
        trajectory=events,
    )
    raw = (tmp_path / "trainer" / "atif.json").read_text()
    assert "sk-abc123def456ghi789" not in raw
    parsed = json.loads(raw)
    assert parsed == record
    assert _atif_problems(parsed) == []
    assert _refs(parsed) == {
        "toolu_01A": ["subagent:toolu_01A"],
        "toolu_01B": ["subagent:toolu_01B"],
    }
    assert not (tmp_path / "trainer" / "atif.coverage.json").exists()


def test_records_validate_against_harbor_trajectory_model():
    """Guards ATIF subagent export (after the child transcript filter) against Harbor's own models."""
    trajectory_model = pytest.importorskip("harbor.models.trajectories").Trajectory
    records = [
        trajectory_to_atif_record(
            session_id="task__1",
            agent_name="claude-agent-acp",
            events=claude_task_rollout(),
            prompts=["Solve."],
        ),
        trajectory_to_atif_record(
            session_id="s",
            agent_name="a",
            events=[
                _spawn("t1", "level one"),
                _spawn("t2", "level two", parent="t1"),
                _child("t2", "agent_message", "grandchild"),
                _child("orphan", "agent_thought", "orphan thought"),
            ],
        ),
    ]
    for record in records:
        model = trajectory_model.model_validate(record)
        assert model.to_json_dict() == record
