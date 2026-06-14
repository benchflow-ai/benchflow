"""Browser-use smoke parity script plumbing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _eval_json() -> str:
    return (
        '{"status": "completed", "ok": true, "summary_path": "/tmp/jobs/summary.json", '
        '"result": {"total": 1, "errored": 0, "verifier_errored": 0}, '
        '"summary": {"total": 1, "agent": "browser-use-smoke", '
        '"environment": "docker", "errored": 0, "verifier_errored": 0, '
        '"total_trajectory_steps": 3}}'
    )


def _load_parity_script() -> ModuleType:
    script_dir = Path(__file__).parents[1] / "benchmarks" / "browser-use-smoke"
    script = script_dir / "parity_test.py"
    spec = importlib.util.spec_from_file_location(
        "browser_use_smoke_parity_test", script
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


def test_browser_use_parity_run_passes_model_to_bench_eval(monkeypatch) -> None:
    module = _load_parity_script()
    recorded: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=_eval_json(), stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_benchflow(
        Path("/tmp/task"),
        Path("/tmp/jobs"),
        agent="stagehand-agent",
        model="google/gemini-3.5-flash",
    )

    assert recorded["cmd"][-2:] == ["--model", "google/gemini-3.5-flash"]
    assert "--json" in recorded["cmd"]
    assert recorded["kwargs"]["cwd"] == module.REPO_ROOT
    assert recorded["kwargs"]["capture_output"] is True


def test_browser_use_parity_run_omits_model_when_not_requested(monkeypatch) -> None:
    module = _load_parity_script()
    recorded: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=_eval_json(), stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_benchflow(
        Path("/tmp/task"),
        Path("/tmp/jobs"),
        agent="browser-use-smoke",
        model=None,
    )

    assert "--model" not in recorded["cmd"]
    assert "--json" in recorded["cmd"]
    assert recorded["kwargs"]["capture_output"] is True


def test_leaked_benchflow_resources_tolerates_preexisting_containers() -> None:
    # Per-run scoping: pre-existing/concurrent benchflow-owned containers that
    # this run did not create must NOT count as a leak. Only resources present
    # after the run that were absent before it are this run's leak.
    module = _load_parity_script()

    before = {
        "available": True,
        "containers": ["hello-world-task-main-1", "skillsbench-citation-main-1"],
        "networks": ["benchflow-other_default"],
    }
    after = {
        "available": True,
        # Both pre-existing containers survive; a concurrent run's container
        # appeared. None of these were created by THIS run.
        "containers": [
            "hello-world-task-main-1",
            "skillsbench-citation-main-1",
            "concurrent-run-main-1",
        ],
        "networks": ["benchflow-other_default"],
    }

    # The concurrent container was not in ``before``, so the raw diff would flag
    # it. That is the documented residual; the realistic false-fail (pre-existing
    # survivors) is gone: those are excluded.
    leaked = module._leaked_benchflow_resources(before=before, after=after)
    assert "hello-world-task-main-1" not in leaked["containers"]
    assert "skillsbench-citation-main-1" not in leaked["containers"]
    assert leaked["networks"] == []


def test_leaked_benchflow_resources_clean_run_reports_no_leak() -> None:
    # This run created and cleaned up its own container/network; even though
    # unrelated benchflow containers persist, the leak set is empty.
    module = _load_parity_script()

    before = {
        "available": True,
        "containers": ["hello-world-task-main-1"],
        "networks": [],
    }
    after = {
        "available": True,
        "containers": ["hello-world-task-main-1"],
        "networks": [],
    }

    leaked = module._leaked_benchflow_resources(before=before, after=after)
    assert leaked["containers"] == []
    assert leaked["networks"] == []


def test_leaked_benchflow_resources_detects_real_run_leak() -> None:
    # This run created a container/network and failed to clean them up: they are
    # present after but were absent before, so they ARE flagged as a real leak.
    module = _load_parity_script()

    before = {
        "available": True,
        "containers": ["hello-world-task-main-1"],
        "networks": [],
    }
    after = {
        "available": True,
        "containers": ["hello-world-task-main-1", "benchflow-open-local-page-main-1"],
        "networks": ["benchflow-open-local-page_default"],
    }

    leaked = module._leaked_benchflow_resources(before=before, after=after)
    assert leaked["containers"] == ["benchflow-open-local-page-main-1"]
    assert leaked["networks"] == ["benchflow-open-local-page_default"]
