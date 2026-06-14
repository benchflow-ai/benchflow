"""Official Browser Use adoption driver evidence wiring."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_driver() -> ModuleType:
    script_dir = Path(__file__).parents[1] / "benchmarks" / "browser-use-smoke"
    script = script_dir / "official_adoption_driver.py"
    spec = importlib.util.spec_from_file_location(
        "browser_use_official_adoption_driver", script
    )
    assert spec is not None
    assert spec.loader is not None
    previous_path = list(sys.path)
    sys.path.insert(0, str(script_dir))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    return module


def test_official_adoption_driver_writes_not_ready_loop_for_blocked_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_driver()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    work_dir = tmp_path / "work"
    parity_out = tmp_path / "evidence" / "parity_experiment.json"

    def fake_import_tasks(**kwargs):
        task_dir = kwargs["out_dir"] / "000-task-0"
        task_dir.mkdir(parents=True)
        return [task_dir]

    def fake_run_task_check(task_dir: Path, *, sandbox: str) -> None:
        assert task_dir.name == "000-task-0"
        assert sandbox == "docker"

    def fake_benchflow_eval(tasks_dir: Path, jobs_dir: Path, **kwargs):
        result_dir = jobs_dir / "browser-use-task-0" / "trial-0"
        artifact_dir = result_dir / "artifacts"
        artifact_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "browser-use/task-0",
                    "agent": "browser-use-agent",
                    "rewards": {"reward": 1.0},
                    "trajectory_summary": {
                        "steps": 3,
                        "tool_call_steps": 1,
                    },
                    "n_tool_calls": 1,
                    "timing": {"total": 9.0},
                    "elapsed_sec": 9.0,
                    "error": None,
                }
            )
        )
        (artifact_dir / "browser-use-smoke-trace.json").write_text(
            json.dumps(
                {
                    "framework": "benchflow-browser-use-agent",
                    "steps": [{"action": "open"}],
                    "screenshots_b64": ["abc"],
                    "final_result": "done",
                    "duration_sec": 8.0,
                }
            )
        )
        return {
            "status": "completed",
            "ok": True,
            "summary_path": str(jobs_dir / "summary.json"),
            "result": {
                "total": 1,
                "errored": 0,
                "verifier_errored": 0,
                "elapsed_sec": 10.0,
            },
            "summary": {
                "total": 1,
                "agent": "browser-use-agent",
                "environment": "docker",
                "errored": 0,
                "verifier_errored": 0,
                "total_trajectory_steps": 3,
                "elapsed_sec": 10.0,
            },
        }

    def fake_probe(**kwargs):
        return {
            "schema": "benchflow.browser-use-original-runner-probe.v1",
            "status": "blocked",
            "failure_class": "original-runner-process-timeout",
            "runner": {
                "framework": "browser-use",
                "browser": "local_headless",
                "model": "gemini-2.5-flash",
            },
            "checks": {
                "score_recorded": False,
                "trace_complete": False,
                "result_count": 0,
                "expected_result_count": 1,
            },
            "task_results": [],
        }

    monkeypatch.setattr(module, "import_tasks", fake_import_tasks)
    monkeypatch.setattr(module, "_run_task_check", fake_run_task_check)
    monkeypatch.setattr(module, "_run_benchflow_eval", fake_benchflow_eval)
    monkeypatch.setattr(module, "probe_original_runner", fake_probe)
    monkeypatch.setattr(
        module,
        "_benchflow_docker_resources",
        lambda: {"available": True, "containers": [], "networks": []},
    )
    monkeypatch.setattr(
        module,
        "_wait_for_benchflow_docker_cleanup",
        lambda *, expected: expected,
    )

    result = module.run_official_adoption(
        upstream_repo=upstream,
        work_dir=work_dir,
        task_indices=[0],
        parity_out=parity_out,
        overwrite=True,
    )

    assert result["ok"] is False
    assert result["status"] == "parity-recorded"
    assert result["loop_status"] == "not-ready"
    assert result["original_runner"]["failure_class"] == (
        "original-runner-process-timeout"
    )
    assert parity_out.is_file()
    assert parity_out.with_name("adoption_report.json").is_file()
    assert parity_out.with_name("original_runner_probe.json").is_file()

    loop_state = json.loads(parity_out.with_name("loop_state.json").read_text())
    roles = {item["name"]: item for item in loop_state["roles"]}
    assert roles["original-runner"]["status"] == "blocked"
    assert roles["benchflow-runner"]["status"] == "passed"
    assert roles["verifier"]["status"] == "pending"
    assert loop_state["artifacts"]["original_runner_probe"].endswith(
        "original_runner_probe.json"
    )
    assert loop_state["queue"][0]["status"] == "blocked"


def test_official_adoption_driver_rejects_multi_task_side_by_side_loop(
    tmp_path: Path,
) -> None:
    module = _load_driver()

    try:
        module.run_official_adoption(
            upstream_repo=tmp_path,
            work_dir=tmp_path / "work",
            task_indices=[0, 1],
        )
    except ValueError as exc:
        assert "one task index" in str(exc)
    else:
        raise AssertionError("expected multi-task official adoption run to fail")


def test_official_adoption_driver_leaked_resources_scopes_to_this_run() -> None:
    module = _load_driver()

    before = {
        "available": True,
        "containers": ["hello-world-task-main-1", "skillsbench-citation-main-1"],
        "networks": [],
    }

    # Pre-existing/concurrent benchflow containers survive the run -> no leak.
    clean = module._leaked_benchflow_resources(before=before, after=dict(before))
    assert clean["containers"] == []
    assert clean["networks"] == []

    # A container/network THIS run created and failed to reap IS a leak.
    after = {
        "available": True,
        "containers": [*before["containers"], "benchflow-bu-task-0-main-1"],
        "networks": ["benchflow-bu-task-0_default"],
    }
    leaked = module._leaked_benchflow_resources(before=before, after=after)
    assert leaked["containers"] == ["benchflow-bu-task-0-main-1"]
    assert leaked["networks"] == ["benchflow-bu-task-0_default"]


def test_official_adoption_driver_tolerates_preexisting_container(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Regression: an unrelated benchflow container that exists before AND after
    # the run must not abort the whole run with a false cleanup failure before
    # parity is even attempted.
    module = _load_driver()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    work_dir = tmp_path / "work"
    parity_out = tmp_path / "evidence" / "parity_experiment.json"

    def fake_import_tasks(**kwargs):
        task_dir = kwargs["out_dir"] / "000-task-0"
        task_dir.mkdir(parents=True)
        return [task_dir]

    def fake_benchflow_eval(tasks_dir: Path, jobs_dir: Path, **kwargs):
        result_dir = jobs_dir / "browser-use-task-0" / "trial-0"
        artifact_dir = result_dir / "artifacts"
        artifact_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "browser-use/task-0",
                    "agent": "browser-use-agent",
                    "rewards": {"reward": 1.0},
                    "trajectory_summary": {"steps": 3, "tool_call_steps": 1},
                    "n_tool_calls": 1,
                    "timing": {"total": 9.0},
                    "elapsed_sec": 9.0,
                    "error": None,
                }
            )
        )
        (artifact_dir / "browser-use-smoke-trace.json").write_text(
            json.dumps(
                {
                    "framework": "benchflow-browser-use-agent",
                    "steps": [{"action": "open"}],
                    "screenshots_b64": ["abc"],
                    "final_result": "done",
                    "duration_sec": 8.0,
                }
            )
        )
        return {
            "status": "completed",
            "ok": True,
            "summary_path": str(jobs_dir / "summary.json"),
            "result": {
                "total": 1,
                "errored": 0,
                "verifier_errored": 0,
                "elapsed_sec": 10.0,
            },
            "summary": {
                "total": 1,
                "agent": "browser-use-agent",
                "environment": "docker",
                "errored": 0,
                "verifier_errored": 0,
                "total_trajectory_steps": 3,
                "elapsed_sec": 10.0,
            },
        }

    def fake_probe(**kwargs):
        return {
            "schema": "benchflow.browser-use-original-runner-probe.v1",
            "status": "blocked",
            "failure_class": "original-runner-process-timeout",
            "runner": {
                "framework": "browser-use",
                "browser": "local_headless",
                "model": "gemini-2.5-flash",
            },
            "checks": {
                "score_recorded": False,
                "trace_complete": False,
                "result_count": 0,
                "expected_result_count": 1,
            },
            "task_results": [],
        }

    preexisting = {
        "available": True,
        "containers": ["hello-world-task-main-1", "skillsbench-citation-main-1"],
        "networks": [],
    }

    monkeypatch.setattr(module, "import_tasks", fake_import_tasks)
    monkeypatch.setattr(module, "_run_task_check", lambda *a, **k: None)
    monkeypatch.setattr(module, "_run_benchflow_eval", fake_benchflow_eval)
    monkeypatch.setattr(module, "probe_original_runner", fake_probe)
    # The unrelated containers are present both before and after the run.
    monkeypatch.setattr(
        module, "_benchflow_docker_resources", lambda: dict(preexisting)
    )
    monkeypatch.setattr(
        module,
        "_wait_for_benchflow_docker_cleanup",
        lambda *, expected: dict(preexisting),
    )

    # Must not raise even though benchflow-owned containers exist globally.
    result = module.run_official_adoption(
        upstream_repo=upstream,
        work_dir=work_dir,
        task_indices=[0],
        parity_out=parity_out,
        overwrite=True,
    )

    assert result["cleanup"]["docker_containers"] == 0
    assert result["cleanup"]["docker_networks"] == 0
