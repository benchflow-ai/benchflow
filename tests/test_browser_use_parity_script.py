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
