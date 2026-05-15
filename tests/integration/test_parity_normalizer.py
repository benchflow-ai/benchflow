"""Tests for E2E result-schema parity normalization."""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.integration.parity import normalize_result, write_parity_report


def test_parity_normalizes_current_benchflow_result() -> None:
    record = normalize_result(
        {
            "task_name": "task-a",
            "trial_name": "task-a__abc",
            "rewards": {"reward": 0.5},
            "agent": "gemini",
            "agent_name": "Gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "n_tool_calls": 3,
            "error": None,
            "verifier_error": None,
            "trajectory_source": "acp",
            "partial_trajectory": False,
            "timing": {"total": 1.2},
        }
    )

    assert record["schema"] == "benchflow"
    assert record["task_name"] == "task-a"
    assert record["reward"] == 0.5
    assert record["n_tool_calls"] == 3


def test_parity_normalizes_historical_skillsbench_result() -> None:
    record = normalize_result(
        {
            "task_name": "task-a",
            "trial_name": "task-a__abc",
            "config": {
                "agent": {
                    "name": "claude-code",
                    "model_name": "minimax/MiniMax-M2.1",
                },
                "environment": {"type": "docker"},
            },
            "agent_info": {"name": "claude-code"},
            "agent_result": {
                "n_input_tokens": 100,
                "n_output_tokens": 20,
                "n_cache_tokens": 5,
                "cost_usd": 0.01,
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        }
    )

    assert record["schema"] == "skillsbench-historical"
    assert record["agent"] == "claude-code"
    assert record["model"] == "minimax/MiniMax-M2.1"
    assert record["environment"] == "docker"
    assert record["reward"] == 1.0
    assert record["n_input_tokens"] == 100


def test_write_parity_report_records_baseline_error_and_jsonl(tmp_path: Path) -> None:
    trial = tmp_path / "run" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "trial_name": "task-a__abc",
                "rewards": {"reward": 1.0},
                "agent": "gemini",
                "model": "gemini-3.1-flash-lite-preview",
                "n_tool_calls": 1,
            }
        )
    )

    report = write_parity_report(tmp_path / "run", baseline_error="clone failed")

    assert report["baseline_error"] == "clone failed"
    assert (tmp_path / "run" / "parity_report.json").exists()
    normalized = (tmp_path / "run" / "normalized_results.jsonl").read_text()
    assert '"schema": "benchflow"' in normalized
