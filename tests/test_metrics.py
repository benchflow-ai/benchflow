"""Tests for metrics collection and aggregation."""

import json
from pathlib import Path

import pytest

from benchflow.metrics import collect_metrics


@pytest.fixture
def results_dir(tmp_path):
    """Create a mock results directory with result.json files."""
    # Task A: passed
    trial_a = tmp_path / "job1" / "task-a__abc123"
    trial_a.mkdir(parents=True)
    (trial_a / "result.json").write_text(json.dumps({
        "task_name": "task-a",
        "rewards": {"reward": 1.0},
        "error": None,
        "n_tool_calls": 10,
        "started_at": "2026-03-24 10:00:00.000000",
        "finished_at": "2026-03-24 10:01:00.000000",
    }))

    # Task B: failed
    trial_b = tmp_path / "job1" / "task-b__def456"
    trial_b.mkdir(parents=True)
    (trial_b / "result.json").write_text(json.dumps({
        "task_name": "task-b",
        "rewards": {"reward": 0.0},
        "error": None,
        "n_tool_calls": 25,
        "started_at": "2026-03-24 10:00:00.000000",
        "finished_at": "2026-03-24 10:02:00.000000",
    }))

    # Task C: errored
    trial_c = tmp_path / "job1" / "task-c__ghi789"
    trial_c.mkdir(parents=True)
    (trial_c / "result.json").write_text(json.dumps({
        "task_name": "task-c",
        "rewards": None,
        "error": "Agent timed out after 900.0s",
        "n_tool_calls": 0,
        "started_at": "2026-03-24 10:00:00.000000",
        "finished_at": "2026-03-24 10:15:00.000000",
    }))

    return tmp_path


@pytest.fixture
def results_dir_with_retries(tmp_path):
    """Results dir where task-a failed first, then passed on retry."""
    # Task A: first attempt failed
    trial_a1 = tmp_path / "attempt1" / "task-a__first"
    trial_a1.mkdir(parents=True)
    (trial_a1 / "result.json").write_text(json.dumps({
        "task_name": "task-a",
        "rewards": {"reward": 0.0},
        "error": None,
        "n_tool_calls": 5,
        "started_at": "2026-03-24 10:00:00.000000",
        "finished_at": "2026-03-24 10:01:00.000000",
    }))

    # Task A: retry passed
    trial_a2 = tmp_path / "attempt2" / "task-a__second"
    trial_a2.mkdir(parents=True)
    (trial_a2 / "result.json").write_text(json.dumps({
        "task_name": "task-a",
        "rewards": {"reward": 1.0},
        "error": None,
        "n_tool_calls": 15,
        "started_at": "2026-03-24 10:02:00.000000",
        "finished_at": "2026-03-24 10:03:00.000000",
    }))

    return tmp_path


def test_collect_metrics_basic(results_dir):
    """Test basic pass/fail/error counting."""
    metrics = collect_metrics(str(results_dir))
    s = metrics.summary()

    assert s["total"] == 3
    assert s["passed"] == 1
    assert s["failed"] == 1
    assert s["errored"] == 1
    assert s["score"] == "33.3%"
    assert "task-a" in s["passed_tasks"]
    assert "task-b" in s["failed_tasks"]
    assert "task-c" in s["errored_tasks"]


def test_collect_metrics_tool_calls(results_dir):
    """Test average tool call calculation (excludes errored tasks)."""
    metrics = collect_metrics(str(results_dir))
    s = metrics.summary()
    # (10 + 25) / 2 = 17.5 — errored task-c excluded
    assert abs(s["avg_tool_calls"] - 17.5) < 0.1


def test_collect_metrics_duration(results_dir):
    """Test average duration calculation (excludes errored tasks)."""
    metrics = collect_metrics(str(results_dir))
    s = metrics.summary()
    # (60 + 120) / 2 = 90 — errored task-c excluded
    assert abs(s["avg_duration_sec"] - 90.0) < 1.0


def test_collect_metrics_best_result_picking(results_dir_with_retries):
    """Test that best result per task is picked (higher reward wins)."""
    metrics = collect_metrics(str(results_dir_with_retries))
    s = metrics.summary()

    assert s["total"] == 1
    assert s["passed"] == 1
    assert s["failed"] == 0
    assert "task-a" in s["passed_tasks"]


def test_collect_metrics_empty_dir(tmp_path):
    """Test with no results."""
    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()

    assert s["total"] == 0
    assert s["passed"] == 0


def test_collect_metrics_corrupt_file(tmp_path):
    """Test that corrupt result.json files are skipped."""
    trial = tmp_path / "job" / "task__abc"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("not json")

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()
    assert s["total"] == 0


def test_collect_metrics_metadata(results_dir):
    """Test that benchmark/agent/model metadata is passed through."""
    metrics = collect_metrics(
        str(results_dir), benchmark="TB2", agent="claude", model="haiku"
    )
    s = metrics.summary()

    assert s["benchmark"] == "TB2"
    assert s["agent"] == "claude"
    assert s["model"] == "haiku"


def test_collect_metrics_partial_reward(tmp_path):
    """Test handling of partial rewards (not 0 or 1)."""
    trial = tmp_path / "job" / "task-a__abc"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "task_name": "task-a",
        "rewards": {"reward": 0.5},
        "error": None,
        "n_tool_calls": 10,
        "started_at": "2026-03-24 10:00:00.000000",
        "finished_at": "2026-03-24 10:01:00.000000",
    }))

    metrics = collect_metrics(str(tmp_path))
    s = metrics.summary()
    assert s["total"] == 1
    # 0.5 is not 1.0, so not passed
    assert s["passed"] == 0
    assert s["failed"] == 1
