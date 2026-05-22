from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.integration.check_results import check_agent


def _write_result_tree(
    tmp_path: Path,
    *,
    reward: float,
    summary: dict,
) -> Path:
    agent_dir = tmp_path / "agentA"
    run_dir = agent_dir / "2026-05-18__00-00-00" / "task-a"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "agentA",
                "rewards": {"reward": reward},
                "error": None,
                "verifier_error": None,
            }
        )
    )
    (agent_dir / "summary.json").write_text(json.dumps(summary))
    return agent_dir


def test_check_results_treats_partial_reward_as_failure(tmp_path: Path) -> None:
    """Guards ENG-91 P1 integration checker partial-reward regression."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=0.5,
        summary={"total": 1, "passed": 0, "failed": 1, "errored": 0, "score": 0.0},
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is True
    assert findings["passed"] == 0
    assert findings["failed"] == 1


def test_check_results_reconciles_summary_counts(tmp_path: Path) -> None:
    """Guards ENG-91 P1 integration checker summary reconciliation."""
    agent_dir = _write_result_tree(
        tmp_path,
        reward=0.5,
        summary={"total": 1, "passed": 1, "failed": 0, "errored": 0, "score": 1.0},
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is False
    assert any("summary.json passed=1" in issue for issue in findings["issues"])


def test_check_results_dedupes_retried_task_results(tmp_path: Path) -> None:
    """Guards the 2026-05-19 Gemini integration retry accounting bug."""
    agent_dir = tmp_path / "gemini"
    run_dir = agent_dir / "2026-05-19__00-00-00"
    first = run_dir / "weighted-gdp-calc__first"
    second = run_dir / "weighted-gdp-calc__second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "result.json").write_text(
        json.dumps(
            {
                "task_name": "weighted-gdp-calc",
                "agent": "gemini",
                "rewards": None,
                "error": "ACP error 400",
                "verifier_error": None,
            }
        )
    )
    (second / "result.json").write_text(
        json.dumps(
            {
                "task_name": "weighted-gdp-calc",
                "agent": "gemini",
                "rewards": {"reward": 0.0},
                "error": None,
                "verifier_error": None,
            }
        )
    )
    (agent_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 0,
                "failed": 1,
                "errored": 0,
                "verifier_errored": 0,
                "score": "0.0%",
            }
        )
    )

    findings = check_agent(agent_dir)

    assert findings["ok"] is True
    assert findings["total"] == 1
    assert findings["errored"] == 0
    assert findings["failed"] == 1


def test_check_results_cli_accepts_single_rollout_artifact_root(
    tmp_path: Path,
) -> None:
    """Guards v0.5 rollout audit command against direct artifact-root failures."""
    rollout_root = tmp_path / "codex-feature-rollouts-20260522-021530"
    run_dir = rollout_root / "2026-05-22__02-15-31" / "task-a__abc"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "oracle",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
            }
        )
    )
    (rollout_root / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "score": "100.0%",
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tests/integration/check_results.py",
            str(rollout_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "All checks passed." in completed.stdout
