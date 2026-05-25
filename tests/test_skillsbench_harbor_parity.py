from __future__ import annotations

import json
from pathlib import Path

from tests.integration.check_skillsbench_harbor_parity import (
    DEFAULT_HARBOR_BASELINE_REF,
    PIN_FILE,
    main,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_harbor_result(root: Path, task: str, reward: float, trial: str) -> None:
    trial_dir = root / f"{task}__{trial}"
    _write_json(
        trial_dir / "result.json",
        {
            "task_name": task,
            "trial_name": f"{task}__{trial}",
            "config": {
                "agent": {
                    "name": "gemini-cli",
                    "model_name": "google/gemini-3-flash-preview",
                },
                "environment": {"type": "docker"},
            },
            "agent_info": {
                "name": "gemini-cli",
                "model_info": {
                    "name": "gemini-3-flash-preview",
                    "provider": "google",
                },
            },
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": None,
        },
    )
    _write_json(
        trial_dir / "agent" / "trajectory.json",
        {
            "schema_version": "ATIF-v1.2",
            "steps": [
                {"source": "user", "message": "solve"},
                {
                    "source": "agent",
                    "message": "done",
                    "tool_calls": [{"function_name": "run_shell_command"}],
                },
            ],
        },
    )


def _write_benchflow_result(root: Path, task: str, reward: float, rollout: str) -> None:
    rollout_dir = root / "2026-05-24__00-00-00" / f"{task}__{rollout}"
    _write_json(
        rollout_dir / "result.json",
        {
            "task_name": task,
            "rollout_name": f"{task}__{rollout}",
            "rewards": {"reward": reward},
            "agent": "gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "error": None,
            "verifier_error": None,
        },
    )
    trajectory = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text(
        json.dumps({"type": "user_message", "text": "solve"}) + "\n"
        + json.dumps({"type": "assistant_message", "text": "done"})
        + "\n"
    )


def test_skillsbench_harbor_parity_normalizes_expected_schema_differences(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against comparing Harbor and BenchFlow artifacts literally."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_harbor_result(harbor, "python-scala-translation", 0.0, "harbor-b")
    _write_benchflow_result(benchflow, "jax-computing-basics", 1.0, "bf-a")
    _write_benchflow_result(benchflow, "python-scala-translation", 0.0, "bf-b")

    rc = main(
        [
            "--benchflow-root",
            str(benchflow),
            "--harbor-baseline-root",
            str(harbor),
            "--task",
            "jax-computing-basics",
            "--task",
            "python-scala-translation",
            "--max-outcome-rate-delta",
            "0",
            "--max-mean-reward-delta",
            "0",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "BenchFlow: total=2 passed=1 failed=1" in out
    assert "Harbor: total=2 passed=1 failed=1" in out


def test_skillsbench_harbor_parity_fails_on_meaningful_reward_drift(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against accepting reward distribution drift."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(benchflow, "jax-computing-basics", 0.0, "bf-a")

    rc = main(
        [
            "--benchflow-root",
            str(benchflow),
            "--harbor-baseline-root",
            str(harbor),
            "--task",
            "jax-computing-basics",
            "--max-outcome-rate-delta",
            "0",
            "--max-mean-reward-delta",
            "0",
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "outcome 'failed' is not present" in out
    assert "mean reward drift" in out


def test_skillsbench_harbor_parity_requires_a_pinned_baseline(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against unpinned Harbor baseline evidence."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(benchflow, "jax-computing-basics", 1.0, "bf-a")

    rc = main(
        [
            "--benchflow-root",
            str(benchflow),
            "--harbor-baseline-root",
            str(harbor),
            "--task",
            "jax-computing-basics",
        ]
    )

    assert rc == 1
    assert f"no {PIN_FILE} pin file" in capsys.readouterr().out
