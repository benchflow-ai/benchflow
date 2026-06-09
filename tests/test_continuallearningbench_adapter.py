from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.task.document import TaskDocument


def _load_benchflow_module():
    path = Path("benchmarks/continuallearningbench/benchflow.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "continuallearningbench_converter_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_parity_module():
    path = Path("benchmarks/continuallearningbench/parity_test.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "continuallearningbench_parity_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    path = Path(
        "benchmarks/continuallearningbench/run_continuallearningbench.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "continuallearningbench_runner_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_continuallearningbench_runner_accepts_plan_flags() -> None:
    """Guards ENG-103 dogfood commands from being ignored by the runner."""
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/continuallearningbench/run_continuallearningbench.py",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "--limit" in result.stdout
    assert "--task-format" in result.stdout
    assert "--overwrite" in result.stdout


def test_continuallearningbench_runner_uses_format_specific_default_output_dir() -> (
    None
):
    """Guards PR #1 against default legacy/task-md output-dir contamination."""
    runner = _load_runner_module()

    assert runner._default_output_dir("legacy") == Path(
        "/tmp/continuallearningbench-tasks"
    )
    assert runner._default_output_dir("task-md") == Path(
        "/tmp/continuallearningbench-tasks-task-md"
    )


def test_continuallearningbench_converter_emits_driver_mediated_task(
    tmp_path: Path,
) -> None:
    """Guards ENG-103 tasks from asking agents to hand-write verifier results."""
    converter = _load_benchflow_module()

    generated = converter.generate_all(
        tmp_path / "continuallearningbench",
        tmp_path / "out",
        limit=1,
        task_format="legacy",
    )

    task_dir = generated[0]
    instruction = (task_dir / "instruction.md").read_text()
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()

    assert "/opt/run_task.py" in instruction
    assert "/opt/agent_responses.jsonl" in instruction
    assert "Do not create or edit `/opt/results.json`" in instruction
    assert "manually" in instruction
    assert "chmod 666 /opt/agent_responses.jsonl /opt/results.json" in dockerfile


def test_continuallearningbench_parity_accepts_limited_generated_subset(
    tmp_path: Path,
) -> None:
    """Guards ENG-103 `--limit 1` dogfood from requiring every ContinualLearningBench task."""
    converter = _load_benchflow_module()
    parity = _load_parity_module()
    out = tmp_path / "out"
    converter.generate_all(
        tmp_path / "continuallearningbench", out, limit=1, task_format="legacy"
    )

    structural = parity.run_structural_parity(out)
    eval_results = parity.run_eval_parity(out)

    assert structural["tasks_tested"] == 1
    assert structural["passed"] == 1
    assert eval_results["tasks_tested"] == 1
    assert eval_results["passed"] == 1


def test_continuallearningbench_converter_defaults_to_task_md(
    tmp_path: Path,
) -> None:
    """The converter default is native task.md; legacy is emitted only on request."""
    converter = _load_benchflow_module()
    out = tmp_path / "out"
    generated = converter.generate_all(
        tmp_path / "continuallearningbench", out, limit=1
    )

    task_dir = generated[0]
    assert (task_dir / "task.md").exists()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()


def test_continuallearningbench_converter_emits_native_task_md(
    tmp_path: Path,
) -> None:
    """Guards PR #1's ContinualLearningBench native no-static-oracle path."""
    converter = _load_benchflow_module()
    parity = _load_parity_module()

    out = tmp_path / "out"
    generated = converter.generate_all(
        tmp_path / "continuallearningbench",
        out,
        limit=1,
        task_format="task-md",
    )

    task_dir = generated[0]
    assert (task_dir / "task.md").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "environment" / "run_task.py").exists()
    assert (task_dir / "environment" / "schedule.json").exists()
    assert (task_dir / "verifier" / "test.sh").exists()
    assert (task_dir / "verifier" / "evaluate.py").exists()
    assert (task_dir / "verifier" / "verifier.md").exists()
    assert (task_dir / "verifier" / "rubrics" / "verifier.md").exists()
    assert (task_dir / "oracle" / "README.md").exists()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert not (task_dir / "tests").exists()
    assert not (task_dir / "solution").exists()
    assert check_task(task_dir, validation_level="publication-grade") == []

    document = TaskDocument.from_path(task_dir / "task.md")
    assert document.config.schema_version == "1.3"
    assert document.config.task is not None
    assert document.config.task.name.startswith("continuallearningbench/")
    assert "/opt/run_task.py" in document.instruction

    structural = parity.run_structural_parity(out, task_format="task-md")
    eval_results = parity.run_eval_parity(out, task_format="task-md")
    assert structural["tasks_tested"] == 1
    assert structural["passed"] == 1
    assert eval_results["tasks_tested"] == 1
    assert eval_results["passed"] == 1

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps({"score": 0.0}))
    log_dir = tmp_path / "logs" / "verifier"
    smoke = subprocess.run(
        ["bash", str(task_dir / "verifier" / "test.sh")],
        env={
            **os.environ,
            "BENCHFLOW_VERIFIER_DIR": str(task_dir / "verifier"),
            "BENCHFLOW_RESULTS_JSON": str(results_file),
            "BENCHFLOW_VERIFIER_LOG": str(log_dir / "verifier.log"),
            "BENCHFLOW_REWARD_TEXT": str(log_dir / "reward.txt"),
            "BENCHFLOW_REWARD_JSON": str(log_dir / "reward.json"),
            "BENCHFLOW_REWARD_DETAILS_JSON": str(log_dir / "reward-details.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert (log_dir / "reward.txt").read_text().strip() == "0.0"
    assert json.loads((log_dir / "reward.json").read_text()) == {"reward": 0.0}


def test_continuallearningbench_converter_rejects_mismatched_existing_layout(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against direct ContinualLearningBench output contamination."""
    converter = _load_benchflow_module()
    out = tmp_path / "out"
    converter.generate_all(
        tmp_path / "continuallearningbench", out, limit=1, task_format="legacy"
    )

    with pytest.raises(ValueError, match=r"already contains task\.toml"):
        converter.generate_all(
            tmp_path / "continuallearningbench",
            out,
            limit=1,
            task_format="task-md",
        )


def test_continuallearningbench_converter_rejects_stale_same_format_task(
    tmp_path: Path,
) -> None:
    """Guards PR #1 against stale skipped evaluate.py parity evidence."""
    converter = _load_benchflow_module()
    out = tmp_path / "out"
    generated = converter.generate_all(
        tmp_path / "continuallearningbench",
        out,
        limit=1,
        task_format="task-md",
    )
    stale_evaluate = generated[0] / "verifier" / "evaluate.py"
    stale_evaluate.write_text(
        "#!/usr/bin/env python3\n"
        "RESULTS_FILE = '/opt/results.json'\n"
        "REWARD_FILE = '/logs/verifier/reward.txt'\n"
    )

    with pytest.raises(ValueError, match="older ContinualLearningBench converter"):
        converter.generate_all(
            tmp_path / "continuallearningbench",
            out,
            limit=1,
            task_format="task-md",
        )
