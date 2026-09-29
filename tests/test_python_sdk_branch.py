"""bf.branch: one call for what `bench eval branch` does.

Branching from Python needed the manual lifecycle (setup, start, install_agent,
connect, execute, branch, connect, execute, verify, finalize) and a hand-written
child runner. ``bf.branch`` / ``await bf.abranch`` build the same
``BranchPlan`` the CLI builds, run the same ``run_branch_trial`` and write the
same job folder, and return a typed ``BranchResult``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import benchflow as bf
from benchflow import branch_api, branch_run
from benchflow.branch_run import BranchPlan, BranchTrialOutcome

TASK = Path(__file__).parent / "examples" / "hello-world-task"


@pytest.fixture
def fake_trial(monkeypatch, tmp_path: Path) -> list[BranchPlan]:
    plans: list[BranchPlan] = []

    async def fake(plan: BranchPlan, task_path: Path, **_: Any) -> BranchTrialOutcome:
        plans.append(plan)
        await asyncio.sleep(0)
        rollout_dir = plan.jobs_dir / plan.job_name / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": rollout_dir.name,
                    "rewards": {"reward": 1.0},
                    "agent": plan.agent,
                }
            )
        )
        return BranchTrialOutcome(
            task=task_path.name,
            rollout_dir=str(rollout_dir),
            value=0.5,
            fork_status="completed",
            parent_restore="restored",
            parent_reward=1.0,
            children=[
                {
                    "label": "a",
                    "node_id": "n1",
                    "status": "scored",
                    "reward": 1.0,
                    "reward_source": "verifier",
                    "path": "branches/f/children/n1",
                    "fork_id": "f",
                    "parent_label": None,
                },
                {
                    "label": "b",
                    "node_id": "n2",
                    "status": "scored",
                    "reward": 0.0,
                    "reward_source": "verifier",
                    "path": "branches/f/children/n2",
                    "fork_id": "f",
                    "parent_label": None,
                },
            ],
        )

    monkeypatch.setattr(branch_api, "run_branch_trial", fake)
    return plans


def test_public_names() -> None:
    for name in ("branch", "abranch", "BranchResult", "BranchChildResult", "ChildSpec"):
        assert name in bf.__all__


def test_branch_builds_the_cli_plan_and_returns_a_typed_result(
    fake_trial, tmp_path: Path
) -> None:
    result = bf.branch(
        TASK,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        prompts=["Write draft.txt", "@instruction"],
        children={"baseline": None, "hint": "Rename draft.txt to hello.txt."},
        sandbox="daytona",
        jobs_dir=tmp_path / "jobs",
    )
    (plan,) = fake_trial
    assert plan.agent == "claude-agent-acp" and plan.sandbox == "daytona"
    assert plan.checkpoint_after == 1 and plan.parent_mode == "continue"
    assert [(c.label, c.prompt) for c in plan.children] == [
        ("baseline", None),
        ("hint", "Rename draft.txt to hello.txt."),
    ]
    assert plan.job_name.startswith("branch-")

    assert isinstance(result, bf.BranchResult) and result.ok
    assert result.value == 0.5 and result.parent_reward == 1.0
    assert [c.label for c in result.children] == ["a", "b"]
    assert result.child("b").reward == 0.0
    assert isinstance(result.parent, bf.RolloutResult) and result.parent.reward == 1.0
    summary = json.loads((result.job_dir / "summary.json").read_text())
    assert summary["kind"] == "benchflow-branch-job" and summary["total"] == 2


def test_abranch_is_the_async_form(fake_trial, tmp_path: Path) -> None:
    result = asyncio.run(
        bf.abranch(
            TASK,
            agent="oracle",
            children=["label=x", "label=y"],
            jobs_dir=tmp_path / "jobs",
        )
    )
    assert fake_trial[0].checkpoint_after == 0  # the oracle's default
    assert result.value == 0.5


def test_invalid_requests_fail_before_anything_runs(fake_trial) -> None:
    with pytest.raises(bf.BranchPlanError, match="two top-level"):
        bf.branch(TASK, agent="oracle", children={"only": None})
    with pytest.raises(ValueError, match="did you mean"):
        bf.branch(TASK, agent="claud-agent-acp", children={"a": None, "b": None})
    with pytest.raises(bf.BranchPlanError, match="empty"):
        bf.branch(TASK, agent="oracle", children={"a": " ", "b": None})
    with pytest.raises(TypeError, match="ChildSpec"):
        bf.branch(TASK, agent="oracle", children=[1, 2])  # type: ignore[list-item]
    assert fake_trial == []


def test_records_and_exports(fake_trial, tmp_path: Path) -> None:
    result = bf.branch(
        TASK, agent="oracle", children={"a": None, "b": None}, jobs_dir=tmp_path / "j"
    )
    rows = result.to_records()
    assert [r["label"] for r in rows] == ["a", "b"] and rows[0]["value"] == 0.5
    assert result.to_csv(tmp_path / "c.csv").read_text().startswith("task,value,label")
    assert len(result.to_jsonl(tmp_path / "c.jsonl").read_text().splitlines()) == 2
    with pytest.raises(KeyError, match="children"):
        result.child("zzz")


def test_same_driver_as_the_cli() -> None:
    """No second implementation: bf.branch calls the CLI's own driver."""
    assert branch_api.run_branch_trial is branch_run.run_branch_trial
    assert branch_api.write_branch_job is branch_run.write_branch_job


def test_docker_not_ready_raises_before_any_trial(fake_trial, monkeypatch, tmp_path):
    """bf.run, bf.run_batch, Evaluation.run and both CLI commands check the
    host before starting; bf.branch skipped it and failed inside the trial."""
    from benchflow import doctor as doctor_mod
    from benchflow.doctor import Check

    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(
        doctor_mod,
        "check_docker",
        lambda probes, *, required: [
            Check(
                "docker",
                "sandbox",
                "docker",
                "fail",
                "daemon unreachable",
                "colima start",
            )
        ],
    )
    with pytest.raises(RuntimeError, match="colima start"):
        bf.branch(
            TASK,
            agent="oracle",
            children=["label=a", "label=b"],
            sandbox="docker",
            jobs_dir=tmp_path / "jobs",
        )
    assert fake_trial == []
