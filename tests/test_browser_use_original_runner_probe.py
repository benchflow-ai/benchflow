"""Browser Use official original-runner probe plumbing."""

from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path
from types import ModuleType


def _load_probe_script() -> ModuleType:
    script = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "browser-use-smoke"
        / "original_runner_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "browser_use_original_runner_probe", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_runner(upstream: Path, body: str) -> None:
    (upstream / "run_framework_eval.py").write_text(textwrap.dedent(body))


def test_browser_use_original_runner_probe_records_completed_summary(
    tmp_path: Path,
) -> None:
    module = _load_probe_script()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _write_fake_runner(
        upstream,
        """
        import json
        from pathlib import Path

        root = Path.cwd()
        results = root / "results" / "summary.json"
        run_data = root / "run_data" / "probe"
        task_results = run_data / "_task_results"
        task_results.mkdir(parents=True)
        results.parent.mkdir(parents=True)
        entry = {
            "run_start": "20260612_000000",
            "benchmark": "BU_Bench_V1",
            "framework": "browser-use",
            "browser": "local_headless",
            "model": "gemini-2.5-flash",
            "task_indices": [0],
            "tasks_completed": 1,
            "tasks_successful": 1,
            "total_steps": 4,
            "total_duration": 12.5,
            "total_cost": 0.01,
            "task_results": [
                {
                    "task_id": "task-0",
                    "task_index": 0,
                    "score": 1,
                    "steps": 4,
                    "duration": 12.5,
                    "cost": 0.01,
                }
            ],
        }
        results.write_text(json.dumps([entry]))
        (task_results / "task_0.json").write_text(json.dumps(entry["task_results"][0]))
        (run_data / "task-0.json").write_text("{}")
        print(f"Summary: {results}")
        print(f"Trace artifacts: {run_data}")
        """,
    )

    report = module.probe_original_runner(
        upstream_repo=upstream,
        task_indices=[0],
        runner_timeout_sec=5,
    )

    assert report["schema"] == module.SCHEMA
    assert report["status"] == "completed"
    assert "failure_class" not in report
    assert report["checks"]["score_recorded"] is True
    assert report["checks"]["trace_complete"] is True
    assert report["artifacts"]["task_result_count"] == 1
    assert report["artifacts"]["raw_trace_file_count"] == 1
    assert "confirmed_task" not in json.dumps(report)


def test_browser_use_original_runner_probe_classifies_timeout_without_trace(
    tmp_path: Path,
) -> None:
    module = _load_probe_script()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _write_fake_runner(
        upstream,
        """
        import json
        from pathlib import Path

        root = Path.cwd()
        results = root / "results" / "summary.json"
        run_data = root / "run_data" / "probe"
        task_results = run_data / "_task_results"
        task_results.mkdir(parents=True)
        results.parent.mkdir(parents=True)
        entry = {
            "benchmark": "BU_Bench_V1",
            "framework": "browser-use",
            "browser": "local_headless",
            "model": "gemini-2.5-flash",
            "task_indices": [0],
            "tasks_completed": 1,
            "tasks_successful": 0,
            "total_steps": 0,
            "total_duration": 1,
            "total_cost": 0,
            "task_results": [
                {
                    "task_id": "task-0",
                    "task_index": 0,
                    "score": 0,
                    "steps": 0,
                    "duration": 1,
                    "cost": 0,
                    "error": "Local browser startup timed out after 1s",
                }
            ],
        }
        results.write_text(json.dumps([entry]))
        (task_results / "task_0.json").write_text(json.dumps(entry["task_results"][0]))
        print(f"Summary: {results}")
        print(f"Trace artifacts: {run_data}")
        """,
    )

    report = module.probe_original_runner(
        upstream_repo=upstream,
        task_indices=[0],
        runner_timeout_sec=5,
    )

    assert report["status"] == "blocked"
    assert report["failure_class"] == "host-local-browser-startup-timeout"
    assert report["checks"]["score_recorded"] is True
    assert report["checks"]["trace_complete"] is False
    assert report["artifacts"]["raw_trace_file_count"] == 0


def test_browser_use_original_runner_probe_classifies_process_timeout(
    tmp_path: Path,
) -> None:
    module = _load_probe_script()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _write_fake_runner(
        upstream,
        """
        import time
        time.sleep(10)
        """,
    )

    report = module.probe_original_runner(
        upstream_repo=upstream,
        task_indices=[0],
        runner_timeout_sec=1,
    )

    assert report["status"] == "blocked"
    assert report["failure_class"] == "original-runner-process-timeout"
    assert report["checks"]["score_recorded"] is False
    assert report["checks"]["trace_complete"] is False
