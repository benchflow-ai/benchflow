from __future__ import annotations

import json

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.models import RunResult


def test_benchflow_version_flag() -> None:
    from benchflow import __version__

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"benchflow {__version__}"


def test_benchflow_run_exits_nonzero_when_verifier_errors(tmp_path, monkeypatch):
    """Guards the v0.5 reward-output regression at the CLI boundary."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('version = "1.0"\n[verifier]\n')

    class FakeSDK:
        async def run(self, **_kwargs):
            return RunResult(
                task_name="task",
                agent_name="oracle",
                rewards=None,
                verifier_error="verifier crashed: No reward file found",
            )

    monkeypatch.setattr("benchflow.sdk.SDK", FakeSDK)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(task_dir),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 1
    assert "Rewards:" in result.output
    assert "Verifier error:" in result.output
    assert "No reward file found" in result.output


def test_benchflow_metrics_pretty_prints_memory_score(tmp_path):
    """Guards OPEN-3 memory-space metrics at the CLI boundary."""
    rollout = tmp_path / "jobs" / "run" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {"reward": 1.0},
                "memory_score": 0.5,
                "error": None,
                "verifier_error": None,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )

    result = CliRunner().invoke(app, ["metrics", str(tmp_path / "jobs")])

    assert result.exit_code == 0
    assert "Memory score" in result.output
    assert "50.0%" in result.output


def test_benchflow_metrics_json_includes_memory_score_without_changing_score(tmp_path):
    """Guards memory score as additive JSON, not output pass/fail score."""
    rollout = tmp_path / "jobs" / "run" / "task__abc123"
    rollout.mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": {"reward": 0.0},
                "memory_score": 1.0,
                "error": None,
                "verifier_error": None,
                "started_at": "2026-03-24 10:00:00.000000",
                "finished_at": "2026-03-24 10:01:00.000000",
            }
        )
    )

    result = CliRunner().invoke(app, ["metrics", str(tmp_path / "jobs"), "--json"])

    assert result.exit_code == 0
    payload_start = result.output.index("{")
    payload = json.loads(result.output[payload_start:])
    assert payload["score"] == "0.0%"
    assert payload["failed"] == 1
    assert payload["memory_score"] == 1.0
    assert payload["memory_score_coverage"] == 1.0


def test_benchflow_eval_list_surfaces_root_summary_memory_score(tmp_path):
    """Guards OPEN-3 summary visibility in eval list."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 2,
                "passed": 1,
                "score": "50.0%",
                "memory_score": 0.75,
            }
        )
    )

    result = CliRunner().invoke(app, ["eval", "list", str(jobs_dir)])

    assert result.exit_code == 0
    assert "1/2" in result.output
    assert "50.0%" in result.output
    assert "75.0%" in result.output


def test_eval_create_reports_runtime_config_errors_without_traceback(
    tmp_path, monkeypatch
):
    """Guards PR #587: required usage preflight failures stay user-facing."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    class FakeEvaluation:
        def __init__(self, **_kwargs):
            pass

        async def run(self):
            raise RuntimeError("Token usage tracking is required")

    monkeypatch.setattr("benchflow.evaluation.Evaluation", FakeEvaluation)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(tasks_dir),
            "--agent",
            "openhands",
            "--model",
            "aws-bedrock/example-model",
            "--usage-tracking",
            "required",
        ],
    )

    assert result.exit_code == 1
    assert "Token usage tracking is required" in result.output
    assert "Traceback" not in result.output
