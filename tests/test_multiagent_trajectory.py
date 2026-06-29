from __future__ import annotations

import json
from pathlib import Path

from benchflow.trajectories.multiagent import RealAgentTraceRecorder


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_real_agent_trace_recorder_writes_isolated_session_artifacts(
    tmp_path: Path,
) -> None:
    recorder = RealAgentTraceRecorder(tmp_path, rollout_id="rollout-1")

    planner = recorder.start_session(
        agent_id="planner",
        agent_type="claude-agent-acp",
        model="claude-sonnet-4-6",
        driver="acp",
        scene="plan",
        scene_index=0,
        turn_index=0,
    )
    recorder.finish_session(
        planner,
        [
            {"type": "agent_message", "text": "plan"},
            {"type": "tool_call", "tool_call_id": "tool-1"},
        ],
    )

    implementer = recorder.start_session(
        agent_id="implementer",
        agent_type="codex-acp",
        model="gpt-5.5",
        driver="acp",
        scene="implement",
        scene_index=1,
        turn_index=0,
    )
    recorder.finish_session(implementer, [{"type": "agent_message", "text": "done"}])

    sessions = _read_jsonl(tmp_path / "trajectory" / "sessions.jsonl")
    handoffs = _read_jsonl(tmp_path / "trajectory" / "handoffs.jsonl")
    events = _read_jsonl(tmp_path / "trajectory" / "multiagent_events.jsonl")
    graph = json.loads((tmp_path / "trajectory" / "agent_graph.json").read_text())

    assert [session["agent_id"] for session in sessions] == ["planner", "implementer"]
    assert (
        sessions[0]["trajectory_path"]
        == "trajectory/agents/planner/sess_planner_001/acp.jsonl"
    )
    assert sessions[0]["n_tool_calls"] == 1
    assert handoffs == [
        {
            "artifacts": [],
            "created_at": handoffs[0]["created_at"],
            "from_agent_id": "planner",
            "from_session_id": "sess_planner_001",
            "handoff_id": "handoff_001",
            "relation": "handoff_to",
            "scene": "implement",
            "to_agent_id": "implementer",
            "to_session_id": "sess_implementer_001",
        }
    ]
    assert graph["schema_version"] == "benchflow.real_agents.graph.v0"
    assert graph["metrics"] == {
        "n_events": len(events),
        "n_handoffs": 1,
        "n_sessions": 2,
    }
    assert graph["edges"][0]["relation"] == "handoff_to"
    assert _read_jsonl(tmp_path / sessions[0]["trajectory_path"])[0] == {
        "type": "agent_message",
        "text": "plan",
    }


def test_real_agent_trace_recorder_links_sessions_to_shared_environment(
    tmp_path: Path,
) -> None:
    recorder = RealAgentTraceRecorder(
        tmp_path,
        rollout_id="rollout-1",
        shared_environment_id="casinobench",
    )

    seat0 = recorder.start_session(
        agent_id="seat0",
        agent_type="claude-agent-acp",
        model="claude-sonnet-4-6",
        driver="acp",
        scene="casino-floor",
        scene_index=0,
        turn_index=0,
    )
    recorder.finish_session(seat0, [])
    seat1 = recorder.start_session(
        agent_id="seat1",
        agent_type="codex-acp",
        model="gpt-5.5",
        driver="acp",
        scene="casino-floor",
        scene_index=0,
        turn_index=1,
    )
    recorder.finish_session(seat1, [])

    graph = json.loads((tmp_path / "trajectory" / "agent_graph.json").read_text())

    assert graph["shared_environment_id"] == "casinobench"
    assert {
        "id": "casinobench",
        "kind": "shared_environment",
        "name": "casinobench",
    } in graph["nodes"]
    assert [
        edge["relation"] for edge in graph["edges"] if edge["relation"] == "plays_in"
    ] == ["plays_in", "plays_in"]
