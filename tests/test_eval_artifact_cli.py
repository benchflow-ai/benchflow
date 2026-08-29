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


def test_health_rejects_empty_terminal_llm_capture_manifest(tmp_path: Path) -> None:
    """Guards PR #1057 against validating no-call artifacts as healthy traces."""

    job = tmp_path / "job"
    rollout = job / "task-a__abc"
    _write_rollout(rollout)
    trajectory = rollout / "trajectory"
    (trajectory / "llm_trajectory.jsonl").write_text("")
    (trajectory / "llm_trajectory.manifest.json").write_text(
        json.dumps(
            {
                "status": "no_model_call",
                "capture_fidelity": "none",
                "exchange_count": 0,
                "request_complete": False,
                "response_complete": False,
            }
        )
    )

    health = build_health_summary(job)

    assert health["missing_llm_trajectory"] == 0
    assert health["malformed_llm_trajectory"] == 1
    assert health["rows"][0]["has_llm_trajectory"] is True
    assert health["rows"][0]["valid_llm_trajectory"] is False
    assert health["rows"][0]["llm_trajectory_rows"] == 0


def test_health_rejects_empty_llm_capture_without_manifest(tmp_path: Path) -> None:
    """Guards PR #1057 against accepting an interrupted empty capture artifact."""

    job = tmp_path / "job"
    rollout = job / "task-a__abc"
    _write_rollout(rollout)
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text("")

    health = build_health_summary(job)

    assert health["missing_llm_trajectory"] == 0
    assert health["malformed_llm_trajectory"] == 1
    assert health["rows"][0]["has_llm_trajectory"] is True
    assert health["rows"][0]["valid_llm_trajectory"] is False
    assert health["rows"][0]["llm_trajectory_rows"] == 0


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


def test_eval_run_publish_bucket_writes_readme_summary(
    tmp_path: Path, monkeypatch
) -> None:
    """--publish-bucket writes a README.md run summary into job_dir before upload."""
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")

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

    captured: dict[str, Path] = {}

    def fake_publish_folder_to_bucket(
        folder, *, bucket_id, path_in_repo="", private=False
    ):
        captured["folder"] = folder
        return SimpleNamespace(url=f"https://huggingface.co/buckets/{bucket_id}")

    import benchflow.publish.huggingface as hf_publish

    monkeypatch.setattr(
        hf_publish, "publish_folder_to_bucket", fake_publish_folder_to_bucket
    )

    jobs_dir = tmp_path / "jobs"
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
            str(jobs_dir),
            "--publish-bucket",
            "org/some-bucket",
        ],
    )

    assert result.exit_code == 0, result.output
    readme = captured["folder"] / "README.md"
    assert readme.is_file()
    text = readme.read_text()
    assert "task-a" in text
    assert "Mean reward: 1.000" in text
    assert "`oracle`" in text


def test_eval_run_publish_bucket_readme_dedupes_retry_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    """Retry rollout dirs for the same task must count as one task, best-scored.

    Guards against README.md (and the --eval-results-model reward mean)
    treating each retry attempt as a distinct task.
    """
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")

    async def fake_run(self):
        job_dir = self._jobs_dir / self._job_name
        _write_rollout(job_dir / "task-a__attempt1", "task-a")
        (job_dir / "task-a__attempt1" / "result.json").write_text(
            json.dumps({"task_name": "task-a", "rewards": {"reward": 0.0}}),
            encoding="utf-8",
        )
        _write_rollout(job_dir / "task-a__attempt2", "task-a")
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

    captured: dict[str, Path] = {}

    def fake_publish_folder_to_bucket(
        folder, *, bucket_id, path_in_repo="", private=False
    ):
        captured["folder"] = folder
        return SimpleNamespace(url=f"https://huggingface.co/buckets/{bucket_id}")

    import benchflow.publish.huggingface as hf_publish

    monkeypatch.setattr(
        hf_publish, "publish_folder_to_bucket", fake_publish_folder_to_bucket
    )

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
            "--publish-bucket",
            "org/some-bucket",
        ],
    )

    assert result.exit_code == 0, result.output
    text = (captured["folder"] / "README.md").read_text()
    assert "- Tasks: 1" in text
    assert "- Mean reward: 1.000" in text
    assert text.count("| task-a |") == 1


def test_eval_run_eval_results_dedupes_retry_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    """The --eval-results-model reward mean must not double-count retries."""
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")

    async def fake_run(self):
        job_dir = self._jobs_dir / self._job_name
        _write_rollout(job_dir / "task-a__attempt1", "task-a")
        (job_dir / "task-a__attempt1" / "result.json").write_text(
            json.dumps({"task_name": "task-a", "rewards": {"reward": 0.0}}),
            encoding="utf-8",
        )
        _write_rollout(job_dir / "task-a__attempt2", "task-a")
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

    captured: dict[str, float] = {}

    def fake_open_eval_results_pr(
        *, model_repo, dataset_id, task_id, value, source_url=None, notes=None
    ):
        captured["value"] = value
        return "https://huggingface.co/org/model/discussions/1"

    import benchflow.publish.huggingface as hf_publish

    monkeypatch.setattr(hf_publish, "open_eval_results_pr", fake_open_eval_results_pr)

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
            "--eval-results-model",
            "org/model",
            "--eval-results-dataset",
            "org/bench",
            "--eval-results-task",
            "t1",
        ],
    )

    assert result.exit_code == 0, result.output
    # Best-of-retries reward is 1.0 -> 100.0, not the naive two-attempt mean (50.0).
    assert captured["value"] == 100.0


def test_eval_run_eval_results_warns_when_nothing_scored(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")

    async def fake_run(self):
        job_dir = self._jobs_dir / self._job_name
        rollout = job_dir / "task-a__abc"
        rollout.mkdir(parents=True)
        (rollout / "result.json").write_text(
            json.dumps({"task_name": "task-a", "rewards": None}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            job_name=self._job_name,
            passed=0,
            failed=1,
            errored=0,
            verifier_errored=0,
            total=1,
            score=0.0,
            score_excl_errors=0.0,
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
            "--eval-results-model",
            "org/model",
            "--eval-results-dataset",
            "org/bench",
            "--eval-results-task",
            "t1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "no scored rollouts" in result.output


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
