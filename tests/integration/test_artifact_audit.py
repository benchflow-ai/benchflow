"""Tests for deterministic E2E artifact auditing."""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.integration.artifact_audit import audit_trial_result


def test_artifact_audit_accepts_fixture_result(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "agent").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "trial_name": "task-a__abc",
                "rewards": {"reward": 1.0},
                "agent": "gemini",
                "agent_name": "gemini",
                "model": "gemini-3.1-flash-lite-preview",
                "n_tool_calls": 2,
                "n_prompts": 1,
                "error": None,
                "verifier_error": None,
                "partial_trajectory": False,
                "trajectory_source": "acp",
                "started_at": "2026-05-15 00:00:00",
                "finished_at": "2026-05-15 00:01:00",
                "timing": {"total": 60.0},
            }
        )
    )
    (trial / "config.json").write_text("{}")
    (trial / "timing.json").write_text("{}")
    (trial / "prompts.json").write_text("[]")
    (trial / "agent" / "install-stdout.txt").write_text("ok")
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "tool_call"}) + "\n"
    )

    report = audit_trial_result(trial / "result.json")

    assert report["ok"] is True
    assert report["issues"] == []
    assert report["files"]["trajectory/acp_trajectory.jsonl"]["valid"] is True
