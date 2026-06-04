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


def test_skill_invocation_count_uses_structured_tool_calls_only() -> None:
    """Guards issue #507: skill counts must not come from agent display text."""
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


def test_skill_invocation_count_accepts_openhands_invoke_skill_content() -> None:
    """Guards issue #507: OpenHands invoke_skill ACP calls count as skills."""
    trajectory = [
        {
            "type": "tool_call",
            "kind": "other",
            "title": "Load PDF skill for processing",
            "status": "completed",
            "content": [
                {
                    "content": {
                        "type": "text",
                        "text": "Tool: invoke_skill\nResult:\n[skill: pdf]\n# PDF Guide",
                    },
                    "type": "content",
                }
            ],
        }
    ]

    assert count_skill_invocations(trajectory) == 1


def test_skill_invocation_count_ignores_non_skill_tool_output_mentions() -> None:
    """Guards issue #507: ordinary tool output is not a skill invocation."""
    trajectory = [
        {
            "type": "tool_call",
            "kind": "bash",
            "title": "cat log.txt",
            "content": [
                {
                    "content": {
                        "type": "text",
                        "text": "Tool: invoke_skill\nResult:\n[skill: pdf]",
                    },
                    "type": "content",
                }
            ],
        },
        {
            "type": "agent_message",
            "text": "Tool: invoke_skill\nResult:\n[skill: marker]",
        },
    ]

    assert count_skill_invocations(trajectory) == 0


def test_skill_invocation_count_ignores_mid_output_skill_marker() -> None:
    """Guards #507: an unclassified tool whose output merely mentions the
    invoke_skill marker mid-stream is not counted; only a result whose text
    *begins* with the tool header is a legacy skill invocation."""
    trajectory = [
        {
            "type": "tool_call",
            "kind": "other",
            "title": "grep invoke_skill logs/",
            "content": [
                {
                    "content": {
                        "type": "text",
                        "text": "logs/run.txt:42:Tool: invoke_skill\n[skill: pdf]",
                    },
                    "type": "content",
                }
            ],
        }
    ]

    assert count_skill_invocations(trajectory) == 0


def test_skill_invocation_count_ignores_marker_in_nested_metadata() -> None:
    """Guards #507: marker text buried in non-text tool-call metadata (diffs,
    locations, raw inputs) is ignored — only structured text result blocks
    are inspected."""
    trajectory = [
        {
            "type": "tool_call",
            "kind": "other",
            "title": "edit notes.md",
            "content": [
                {
                    "type": "diff",
                    "path": "notes.md",
                    "oldText": "",
                    "newText": "Tool: invoke_skill\nResult:\n[skill: pdf]",
                },
                {
                    "type": "content",
                    "content": {
                        "type": "text",
                        "text": "Applied edit to notes.md",
                    },
                },
            ],
        }
    ]

    assert count_skill_invocations(trajectory) == 0


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
