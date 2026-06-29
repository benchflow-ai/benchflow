from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from benchflow.rollout import Role, Rollout, RolloutConfig, Scene, Turn
from benchflow.scenes import compile_scenes_to_steps


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def test_scene_steps_emit_isolated_real_agent_trajectories(tmp_path: Path) -> None:
    scene = Scene(
        name="handoff",
        roles=[
            Role("planner", "claude-agent-acp", "sonnet"),
            Role("implementer", "codex-acp", "gpt-5.5"),
        ],
        turns=[
            Turn("planner", "Write /app/plan.md"),
            Turn("implementer", "Use /app/plan.md"),
        ],
    )
    rollout = Rollout(
        RolloutConfig(
            task_path=Path("tasks/fake"),
            scenes=[scene],
            environment="docker",
        )
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout-1"
    rollout._resolved_prompts = ["Solve"]
    rollout._trajectory = []

    active_role = ""

    async def fake_connect_as(role: Role) -> None:
        nonlocal active_role
        active_role = role.name

    async def fake_execute(prompts=None):
        event = {"type": "agent_message", "text": f"{active_role}: {prompts[0]}"}
        rollout._trajectory.append(event)
        return [event], 0

    rollout.connect_as = fake_connect_as  # type: ignore[method-assign]
    rollout.disconnect = AsyncMock()

    rollout.execute = fake_execute  # type: ignore[method-assign,assignment]

    await rollout._run_steps(compile_scenes_to_steps([scene], default_prompt="Solve"))

    sessions = _read_jsonl(tmp_path / "trajectory" / "sessions.jsonl")
    handoffs = _read_jsonl(tmp_path / "trajectory" / "handoffs.jsonl")
    graph = json.loads((tmp_path / "trajectory" / "agent_graph.json").read_text())

    assert [session["agent_id"] for session in sessions] == ["planner", "implementer"]
    assert sessions[0]["trajectory_path"] == "trajectory/agents/planner/sess_planner_001/acp.jsonl"
    assert sessions[1]["trajectory_path"] == "trajectory/agents/implementer/sess_implementer_001/acp.jsonl"
    assert _read_jsonl(tmp_path / sessions[0]["trajectory_path"]) == [
        {"type": "agent_message", "text": "planner: Write /app/plan.md"}
    ]
    assert _read_jsonl(tmp_path / sessions[1]["trajectory_path"]) == [
        {"type": "agent_message", "text": "implementer: Use /app/plan.md"}
    ]
    assert handoffs[0]["from_agent_id"] == "planner"
    assert handoffs[0]["to_agent_id"] == "implementer"
    assert graph["edges"][0]["relation"] == "handoff_to"
