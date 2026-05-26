from __future__ import annotations

import json
from datetime import datetime

from benchflow._utils.evaluation_results import (
    rollout_result_payload,
    skill_invocation_summary,
)
from benchflow.models import RolloutResult
from benchflow.rollout import _build_rollout_result
from benchflow.trajectories.metrics import count_skill_invocations


def test_skill_invocation_count_uses_structured_kind_only() -> None:
    """Guards issue #507: skill counts must not come from display-text matching."""
    trajectory = [
        {
            "type": "tool_call",
            "kind": "bash",
            "title": "Use the data-cleaning skill",
        },
        {"type": "agent_message", "text": "Invoking Skill(data-cleaning)"},
        {"type": "tool_call", "kind": "skill", "title": "data-cleaning"},
    ]

    assert count_skill_invocations(trajectory) == 1


def test_build_rollout_result_writes_skill_invocation_metric(tmp_path) -> None:
    """Guards issue #507: result.json exposes structured skill invocation counts."""
    trajectory = [
        {"type": "user_message", "text": "solve"},
        {"type": "tool_call", "kind": "skill", "title": "calculator"},
        {"type": "tool_call", "kind": "bash", "title": "python solve.py"},
    ]

    result = _build_rollout_result(
        tmp_path,
        task_name="task-a",
        rollout_name="task-a__abc123",
        agent="agentA",
        agent_name="agentA",
        model="test-model",
        n_tool_calls=2,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=trajectory,
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={"agent": 1.0},
    )

    payload = json.loads((tmp_path / "result.json").read_text())
    assert result.n_skill_invocations == 1
    assert payload["n_skill_invocations"] == 1
    assert payload["agent_result"]["n_skill_invocations"] == 1


def test_evaluation_payload_and_summary_include_skill_invocations(tmp_path) -> None:
    """Guards issue #507: evaluation artifacts aggregate the canonical metric."""
    result = RolloutResult(
        task_name="task-a",
        rewards={"reward": 1.0},
        trajectory=[
            {"type": "tool_call", "kind": "skill", "title": "calculator"},
            {"type": "tool_call", "kind": "skill", "title": "spreadsheet"},
            {"type": "tool_call", "kind": "bash", "title": "pytest"},
        ],
        n_tool_calls=3,
    )

    payload = rollout_result_payload(
        result,
        source_provenance=None,
        tasks_dir=tmp_path,
        task_name="task-a",
    )
    summary = skill_invocation_summary({"task-a": payload})

    assert payload["n_skill_invocations"] == 2
    assert payload["agent_result"]["n_skill_invocations"] == 2
    assert summary == {"total_skill_invocations": 2, "avg_skill_invocations": 2.0}
