"""Unit coverage for the runtime-results parity checker.

``tests/integration/check_skillsbench_harbor_parity.py`` compares completed
BenchFlow rollouts against a pinned Harbor baseline (rc codes, schema
normalization, reward drift, provenance, skill lane, and source-sha rules).
Conversion round-trip parity lives separately in
``tests/test_skillsbench_conversion_conformance.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.integration.check_skillsbench_harbor_parity import (
    DEFAULT_HARBOR_BASELINE_REF,
    PIN_FILE,
    main,
)

TASK_MD_HASH = "sha256:" + ("1" * 64)
TASK_TOML_HASH = "sha256:" + ("2" * 64)
SOURCE_SHA = "a" * 40


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


def _write_benchflow_result(
    root: Path,
    task: str,
    reward: float,
    rollout: str,
    *,
    skill_mode: str = "no-skill",
    skill_source: str = "none",
    source_repo: str = "benchflow-ai/skillsbench",
    source_sha: str = SOURCE_SHA,
    source_path: str | None = None,
    file_hashes: dict[str, str] | None = None,
    config_skill_mode: str | None = None,
    config_skill_source: str | None = None,
) -> None:
    rollout_dir = root / "2026-05-24__00-00-00" / f"{task}__{rollout}"
    task_source_path = source_path or f"tasks/{task}"
    task_local_path = root / "tasks" / task
    source = {
        "type": "github",
        "repo": source_repo,
        "requested_ref": "main",
        "resolved_sha": source_sha,
        "path": task_source_path,
        "local_path": str(task_local_path),
        "dirty": False,
        "file_hashes": file_hashes or {"task.md": TASK_MD_HASH},
    }
    _write_json(
        rollout_dir / "result.json",
        {
            "task_name": task,
            "rollout_name": f"{task}__{rollout}",
            "rewards": {"reward": reward},
            "agent": "gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "skill_mode": skill_mode,
            "skill_source": skill_source,
            "error": None,
            "verifier_error": None,
            "source": source,
        },
    )
    _write_json(
        rollout_dir / "config.json",
        {
            "task_path": str(task_local_path),
            "agent": "gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "environment": "daytona",
            "skill_mode": config_skill_mode or skill_mode,
            "skill_source": config_skill_source or skill_source,
            "source": {
                "type": "github",
                "repo": source_repo,
                "requested_ref": "main",
                "resolved_sha": source_sha,
                "path": "tasks",
                "local_path": str(root / "tasks"),
                "dirty": False,
                "file_hashes": {},
            },
        },
    )
    trajectory = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text(
        json.dumps({"type": "user_message", "text": "solve"})
        + "\n"
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


def test_skillsbench_harbor_parity_requires_native_task_md_provenance(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against accepting legacy SkillsBench task artifacts."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(
        benchflow,
        "jax-computing-basics",
        1.0,
        "bf-a",
        file_hashes={"task.toml": TASK_TOML_HASH},
    )

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
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "source.file_hashes must include native task.md" in out


def test_skillsbench_harbor_parity_requires_expected_skill_lane(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against comparing with-skills runs to no-skills Harbor."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(
        benchflow,
        "jax-computing-basics",
        1.0,
        "bf-a",
        skill_mode="with-skill",
        skill_source="task_bundled",
    )

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
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "skill_mode 'with-skill' does not match expected 'no-skill'" in out
    assert "skill_source 'task_bundled' does not match expected 'none'" in out


def test_skillsbench_harbor_parity_requires_config_skill_metadata_match(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against stitched result/config artifacts."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(
        benchflow,
        "jax-computing-basics",
        1.0,
        "bf-a",
        config_skill_mode="with-skill",
        config_skill_source="task_bundled",
    )

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
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "config.skill_mode 'with-skill' does not match result" in out
    assert "config.skill_source 'task_bundled' does not match result" in out


def test_skillsbench_harbor_parity_can_pin_exact_benchflow_source_sha(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against accepting stale source checkouts."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(
        benchflow,
        "jax-computing-basics",
        1.0,
        "bf-a",
        source_sha="b" * 40,
    )

    rc = main(
        [
            "--benchflow-root",
            str(benchflow),
            "--harbor-baseline-root",
            str(harbor),
            "--task",
            "jax-computing-basics",
            "--expected-benchflow-source-sha",
            SOURCE_SHA,
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert f"source.resolved_sha {('b' * 40)!r} does not match" in out


def test_skillsbench_harbor_parity_rejects_mismatched_source_task(
    tmp_path: Path,
    capsys,
) -> None:
    """Guards issue #508 against copied result files with the wrong task source."""
    harbor = tmp_path / "harbor"
    benchflow = tmp_path / "benchflow"
    harbor.mkdir()
    (harbor / PIN_FILE).write_text(DEFAULT_HARBOR_BASELINE_REF + "\n")

    _write_harbor_result(harbor, "jax-computing-basics", 1.0, "harbor-a")
    _write_benchflow_result(
        benchflow,
        "jax-computing-basics",
        1.0,
        "bf-a",
        source_path="tasks/python-scala-translation",
    )

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
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "source.path task segment 'python-scala-translation'" in out


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
