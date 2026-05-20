from __future__ import annotations

import json
from pathlib import Path

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
