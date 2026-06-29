"""CLI coverage for eval/train artifact workflow surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.eval_artifacts import build_canonical_selection, build_health_summary
from benchflow.evaluation import Evaluation

runner = CliRunner()


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")


def _write_llm_trajectory(rollout_dir: Path, *, tool_calls: bool = True) -> None:
    trajectory = rollout_dir / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    message: dict[str, Any] = {"role": "assistant", "content": "done"}
    if tool_calls:
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "finish", "arguments": "{}"},
                }
            ],
        }
    (trajectory / "llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "body": {
                        "model": "m",
                        "messages": [{"role": "user", "content": "do it"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                    }
                },
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": message}]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rollout(rollout_dir: Path, task_name: str = "task-a") -> None:
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "rewards": {"reward": 1.0},
                "n_tool_calls": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_llm_trajectory(rollout_dir)


def test_eval_run_writes_manifest_health_and_canonical_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")

    async def fake_run(self):
        job_dir = self._jobs_dir / self._job_name
        _write_rollout(job_dir / "task-a__abc", "task-a")
        (job_dir / "summary.json").write_text(
            json.dumps({"total": 1, "passed": 1}), encoding="utf-8"
        )
        return SimpleNamespace(
            job_name=self._job_name,
            passed=1,
            failed=0,
            errored=0,
            verifier_errored=0,
            total=1,
            score=1.0,
            score_excl_errors=1.0,
        )

    monkeypatch.setattr(Evaluation, "run", fake_run)
    manifest = tmp_path / "task-manifest.json"
    health = tmp_path / "health.json"
    selection = tmp_path / "canonical-selection.json"
    canonical_jobs = tmp_path / "canonical-jobs"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--task-manifest-out",
            str(manifest),
            "--health-summary-out",
            str(health),
            "--expected-tasks",
            "1",
            "--canonicalize",
            "one-healthy-per-task",
            "--canonical-selection-out",
            str(selection),
            "--canonical-jobs-dir",
            str(canonical_jobs),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(manifest.read_text())["total"] == 1
    assert json.loads(health.read_text())["rows_with_tool_calls"] == 1
    assert json.loads(selection.read_text())["selected_count"] == 1
    assert (canonical_jobs / "task-a__abc" / "result.json").is_file()


def test_sharded_health_and_selection_discover_worker_shards(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    advertised_job_dir = jobs_dir / "worker-sharded"
    shard_job_dir = (
        jobs_dir / "worker-shards" / "shard-000" / "jobs" / "2026-06-29__15-02-55"
    )
    _write_rollout(shard_job_dir / "task-a__abc", "task-a")
    (jobs_dir / "worker-shards" / "plan.json").write_text(
        json.dumps(
            {
                "total_concurrency": 10,
                "worker_concurrency": 2,
                "shards": [{"index": 0, "concurrency": 2, "task_names": ["task-a"]}],
            }
        ),
        encoding="utf-8",
    )

    health = build_health_summary(advertised_job_dir)
    selection = build_canonical_selection(
        advertised_job_dir,
        policy="one-healthy-per-task",
        expected_tasks=1,
    )

    assert health["total_rows"] == 1
    assert health["rows"][0]["task_id"] == "task-a"
    assert selection["selected_count"] == 1
    assert selection["selected"][0]["task_id"] == "task-a"


def test_eval_run_matrix_runs_each_alias_and_trial(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        """
models:
  base: openai/base
  sft:
    model: openai/sft
    agent_env:
      OPENAI_BASE_URL: https://example.invalid/v1
""",
        encoding="utf-8",
    )

    async def fake_run(self):
        job_dir = self._jobs_dir / self._job_name
        _write_rollout(job_dir / "task-a__abc", "task-a")
        return SimpleNamespace(
            job_name=self._job_name,
            passed=1,
            failed=0,
            errored=0,
            verifier_errored=0,
            total=1,
            score=1.0,
            score_excl_errors=1.0,
        )

    monkeypatch.setattr(Evaluation, "run", fake_run)
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--matrix",
            str(matrix),
            "--trials",
            "2",
            "--task-manifest-out",
            str(tmp_path / "task-manifest.json"),
            "--run-config-out",
            str(tmp_path / "run-config.json"),
            "--health-summary-out",
            str(tmp_path / "health.json"),
            "--expected-tasks",
            "1",
            "--canonicalize",
            "one-healthy-per-task",
            "--canonical-selection-out",
            str(tmp_path / "canonical-selection.json"),
            "--canonical-jobs-dir",
            str(tmp_path / "canonical-jobs"),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "jobs" / "matrix-summary.json").read_text())
    assert len(summary["runs"]) == 4
    assert {row["alias"] for row in summary["runs"]} == {"base", "sft"}
    assert json.loads((tmp_path / "task-manifest.json").read_text())["total"] == 1
    assert (tmp_path / "base" / "trial-01" / "run-config.json").is_file()
    assert (tmp_path / "base" / "trial-01" / "health.json").is_file()
    assert (tmp_path / "base" / "trial-01" / "canonical-selection.json").is_file()
    assert (
        tmp_path
        / "canonical-jobs"
        / "base"
        / "trial-01"
        / "task-a__abc"
        / "result.json"
    ).is_file()


def test_tasks_overlap_reports_exact_overlap(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "a", "digest": "sha256:1"},
                    {"task_id": "b", "digest": "sha256:2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "b", "digest": "sha256:2"},
                    {"task_id": "c", "digest": "sha256:3"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["tasks", "overlap", str(left), str(right)])

    assert result.exit_code == 0, result.output
    assert "Task-id overlap: 1" in result.output
    assert "Digest overlap: 1" in result.output
