from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.run_suite import (
    main,
    run_skillsbench_harbor_parity,
)

SUITE_PATH = Path("tests/integration/suites/release.yaml")


def test_skillsbench_harbor_parity_execution_invokes_checker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guards issue #508 SkillsBench-vs-Harbor execution plumbing."""
    captured = {}

    def fake_checker(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "tests.integration.run_suite._run_skillsbench_harbor_parity_checker",
        fake_checker,
    )

    rc = main(
        [
            "--suite",
            str(SUITE_PATH),
            "--lane",
            "skillsbench-harbor-parity",
            "--execute-skillsbench-harbor-parity",
            "--skillsbench-harbor-benchflow-root",
            str(tmp_path / "benchflow"),
            "--skillsbench-harbor-baseline-root",
            str(tmp_path / "harbor"),
            "--skillsbench-harbor-task",
            "jax-computing-basics",
            "--skillsbench-harbor-max-outcome-rate-delta",
            "0",
            "--skillsbench-harbor-max-mean-reward-delta",
            "0",
            "--skillsbench-harbor-max-task-reward-delta",
            "0",
        ]
    )

    assert rc == 0
    assert captured["argv"] == [
        "--benchflow-root",
        str(tmp_path / "benchflow"),
        "--harbor-baseline-root",
        str(tmp_path / "harbor"),
        "--harbor-baseline-ref",
        "2d86fe82f6a06f7c7b3a22a3ae90d554d0e9655c",
        "--max-outcome-rate-delta",
        "0.0",
        "--max-mean-reward-delta",
        "0.0",
        "--max-task-reward-delta",
        "0.0",
        "--expected-benchflow-source-repo",
        "benchflow-ai/skillsbench",
        "--expected-benchflow-source-path-prefix",
        "tasks",
        "--expected-benchflow-task-entrypoint",
        "task.md",
        "--expected-skill-mode",
        "no-skill",
        "--expected-skill-source",
        "none",
        "--task",
        "jax-computing-basics",
    ]


def test_run_skillsbench_harbor_parity_rejects_empty_selection() -> None:
    """Guards issue #508 SkillsBench-vs-Harbor execution rejects missing lane."""
    args = SimpleNamespace(
        skillsbench_harbor_benchflow_root=Path.cwd(),
        skillsbench_harbor_baseline_root=Path.cwd(),
        skillsbench_harbor_baseline_ref="2d86fe82f6a06f7c7b3a22a3ae90d554d0e9655c",
        skillsbench_harbor_max_outcome_rate_delta=0.25,
        skillsbench_harbor_max_mean_reward_delta=0.25,
        skillsbench_harbor_max_task_reward_delta=0.0,
        skillsbench_harbor_expected_source_repo="benchflow-ai/skillsbench",
        skillsbench_harbor_expected_source_path_prefix="tasks",
        skillsbench_harbor_expected_task_entrypoint="task.md",
        skillsbench_harbor_expected_source_sha=None,
        skillsbench_harbor_expected_skill_mode="no-skill",
        skillsbench_harbor_expected_skill_source="none",
        skillsbench_harbor_task=[],
        skillsbench_harbor_no_require_trajectories=False,
    )

    with pytest.raises(ValueError, match="requires lane skillsbench-harbor-parity"):
        run_skillsbench_harbor_parity([], args)
