from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_benchflow_module():
    path = Path("benchmarks/clbench/benchflow.py").resolve()
    spec = importlib.util.spec_from_file_location("clbench_converter_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_parity_module():
    path = Path("benchmarks/clbench/parity_test.py").resolve()
    spec = importlib.util.spec_from_file_location("clbench_parity_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_clbench_runner_accepts_plan_flags() -> None:
    """Guards ENG-103 dogfood commands from being ignored by the runner."""
    result = subprocess.run(
        [sys.executable, "benchmarks/clbench/run_clbench.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "--limit" in result.stdout
    assert "--overwrite" in result.stdout


def test_clbench_converter_emits_driver_mediated_task(tmp_path: Path) -> None:
    """Guards ENG-103 tasks from asking agents to hand-write verifier results."""
    converter = _load_benchflow_module()

    generated = converter.generate_all(tmp_path / "clbench", tmp_path / "out", limit=1)

    task_dir = generated[0]
    instruction = (task_dir / "instruction.md").read_text()
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()

    assert "/opt/run_task.py" in instruction
    assert "/opt/agent_responses.jsonl" in instruction
    assert "Do not create or edit `/opt/results.json`" in instruction
    assert "manually" in instruction
    assert "chmod 666 /opt/agent_responses.jsonl /opt/results.json" in dockerfile


def test_clbench_parity_accepts_limited_generated_subset(tmp_path: Path) -> None:
    """Guards ENG-103 `--limit 1` dogfood from requiring every CLBench task."""
    converter = _load_benchflow_module()
    parity = _load_parity_module()
    out = tmp_path / "out"
    converter.generate_all(tmp_path / "clbench", out, limit=1)

    structural = parity.run_structural_parity(out)
    eval_results = parity.run_eval_parity(out)

    assert structural["tasks_tested"] == 1
    assert structural["passed"] == 1
    assert eval_results["tasks_tested"] == 1
    assert eval_results["passed"] == 1
