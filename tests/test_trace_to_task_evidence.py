from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tests.integration import check_trace_to_task_evidence
from tests.integration.check_trace_to_task_evidence import generate_trace_evidence, main

SUITE_PATH = Path("tests/integration/suites/release.yaml")


def test_trace_evidence_generates_checked_tasks(tmp_path: Path) -> None:
    """Guards ENG-93 trace evidence covers both JSONL and opentraces sources."""
    summary = generate_trace_evidence(
        suite_path=SUITE_PATH,
        repo_root=Path.cwd(),
        evidence_dir=tmp_path,
        run_eval=False,
        sandbox="docker",
    )

    assert summary["status"] == "pass"
    assert summary["lane_id"] == "trace-to-task-e2e"
    assert [source["generated_count"] for source in summary["sources"]] == [1, 1]
    assert [source["status"] for source in summary["sources"]] == ["pass", "pass"]
    for source in summary["sources"]:
        task = source["generated_tasks"][0]
        task_path = Path(task["path"])
        assert task["check_status"] == "pass"
        assert task["issues"] == []
        assert (task_path / "solution" / "solve.sh").exists()

    summary_path = tmp_path / "trace-evidence.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text())["status"] == "pass"


def test_trace_evidence_main_writes_summary(tmp_path: Path) -> None:
    """Guards ENG-93 trace evidence CLI produces a durable summary artifact."""
    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--repo-root",
            str(Path.cwd()),
            "--evidence-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert (tmp_path / "trace-evidence.json").exists()


def test_trace_evidence_eval_uses_unique_oracle_run_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    """Guards trace evidence evals from intentionally resuming prior jobs."""
    captured_jobs_dirs: list[Path] = []

    def fake_run_oracle_eval(task_dir: Path, jobs_dir: Path, sandbox: str) -> dict:
        captured_jobs_dirs.append(jobs_dir)
        return {
            "task_path": str(task_dir),
            "jobs_dir": str(jobs_dir),
            "returncode": 0,
            "result_path": str(jobs_dir / "result.json"),
            "status": "pass",
            "agent": "oracle",
            "reward": 1.0,
            "error": None,
            "verifier_error": None,
        }

    monkeypatch.setattr(
        check_trace_to_task_evidence, "_run_oracle_eval", fake_run_oracle_eval
    )

    summary = generate_trace_evidence(
        suite_path=SUITE_PATH,
        repo_root=Path.cwd(),
        evidence_dir=tmp_path,
        run_eval=True,
        sandbox="docker",
    )

    run_id = summary["oracle_run_id"]
    assert run_id
    assert summary["status"] == "pass"
    assert [path.relative_to(tmp_path).parts[:2] for path in captured_jobs_dirs] == [
        ("oracle-runs", run_id),
        ("oracle-runs", run_id),
    ]


def test_trace_evidence_oracle_eval_requires_fresh_result(
    monkeypatch, tmp_path: Path
) -> None:
    """Guards trace evidence from passing on stale resumed oracle results."""
    jobs_dir = tmp_path / "jobs-oracle-jsonl-fixture"
    old_result = jobs_dir / "2026-05-22__00-00-00" / "task__old" / "result.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text(
        json.dumps(
            {
                "task_name": "task",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
            }
        )
    )

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="resumed old job", stderr="")

    monkeypatch.setattr(check_trace_to_task_evidence.subprocess, "run", fake_run)

    record = check_trace_to_task_evidence._run_oracle_eval(
        tmp_path / "task", jobs_dir, "docker"
    )

    assert record["status"] == "fail"
    assert record["result_path"] is None
    assert "fresh result.json" in record["message"]


def test_trace_evidence_oracle_eval_does_not_persist_success_logs(
    monkeypatch, tmp_path: Path
) -> None:
    """Guards release evidence summaries from retaining full subprocess logs."""
    jobs_dir = tmp_path / "jobs-oracle-jsonl-fixture"

    def fake_run(command, **kwargs):
        result_path = jobs_dir / "2026-05-22__00-00-00" / "task__new" / "result.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "task_name": "task",
                    "agent": "oracle",
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                }
            )
        )
        return SimpleNamespace(
            returncode=0,
            stdout="large stdout that should not be persisted",
            stderr="large stderr that should not be persisted",
        )

    monkeypatch.setattr(check_trace_to_task_evidence.subprocess, "run", fake_run)

    record = check_trace_to_task_evidence._run_oracle_eval(
        tmp_path / "task", jobs_dir, "docker"
    )

    assert record["status"] == "pass"
    assert "stdout" not in record
    assert "stderr" not in record
