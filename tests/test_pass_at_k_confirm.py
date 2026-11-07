"""Guard P4 confirmation findings against the reviewed commit 68365e70."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from benchflow.metrics import collect_metrics
from benchflow.review import automatic
from benchflow.review.resume import resume_review
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.rollout._review import prepare_terminal_result
from tests.test_automatic_review_scoring import _score
from tests.test_python_sdk_load_job import _trial
from tests.test_review_lifecycle import reviewed_rollout as _reviewed_rollout
from tests.test_verifier_recovery import _lease

reviewed_rollout = _reviewed_rollout


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_reward", [1, None])
async def test_setup_deadline_and_acp_retry_share_one_trial(
    tmp_path, monkeypatch, retry_reward
):
    """Guards P4-CONF-1 at 68365e70: ACP display names are not retry identity."""
    task = tmp_path / "deadline"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Solve the task.")
    cfg = RolloutConfig(
        task_path=task,
        jobs_dir=tmp_path,
        job_name="job",
        rollout_name="deadline__first",
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
    )
    first = Rollout(cfg)

    async def wedged_setup():
        await asyncio.sleep(3600)

    first._run_lifecycle = wedged_setup
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.01")
    result = await first.run()
    assert result.error_category == "infra_failure"
    job = tmp_path / "job"
    (job / "evaluation.json").write_text("{}")
    retry = _trial(
        job,
        "deadline",
        reward=retry_reward,
        error="agent crashed" if retry_reward is None else None,
        suffix="retry",
    )
    path = retry / "result.json"
    payload = json.loads(path.read_text())
    payload["agent_name"] = "@agentclientprotocol/claude-agent-acp"
    path.write_text(json.dumps(payload))
    output = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert output.exit_code == 0, output.output
    summary = json.loads(output.stdout)
    assert summary["total"] == summary["solve_rates"]["trials"] == 1
    assert summary["mean_reward_n"] == 1
    assert summary["mean_reward"] == (retry_reward or 0)
    assert summary["score"] == ("100.0%" if retry_reward else "0.0%")
    loaded = bf.load_job(job)
    assert len(loaded.trials) == 1
    assert loaded.trials[0].path == retry
    assert collect_metrics(job).tasks[0].error == payload["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["capture", "cleanup"])
@pytest.mark.parametrize("solver_saved", [False, True])
async def test_reviewed_post_solver_deadline_stays_in_readback(
    reviewed_rollout, tmp_path, monkeypatch, stage, solver_saved
):
    """Guards P4-CONF-2 at 68365e70: terminal scoring errors stay in n."""
    trial = reviewed_rollout
    task = trial._config.task_path
    (task / "task.md").write_text(
        "---\nverifier:\n  workspace_recovery: true\nsandbox:\n"
        + '  docker_image: "python@sha256:'
        + "a" * 64
        + '"\n---\nProduce a file.\n'
    )
    from benchflow.task import Task

    trial._task = Task(task)
    trial._docker_recovery_baseline = _lease()
    released = AsyncMock(return_value=True)
    monkeypatch.setattr(recovery, "release_baseline", released)
    assert recovery.recovery_ineligible_reason(trial) is None
    trial._review_plan = automatic.prepare_review(task, trial._config.reviewer)
    trial._config.task_digest = trial._review_plan.task_digest
    root = trial._require_rollout_dir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_digest": trial._config.task_digest,
                "review": trial._review_plan.metadata(),
                "verifier_recovery": {"eligible": True},
            }
        )
    )
    review = AsyncMock()
    monkeypatch.setattr("benchflow.rollout._review.finish_review", review)
    before = {}

    async def lifecycle():
        recovery.mark_solver_complete(trial)
        before["solver-complete.json"] = (root / "solver-complete.json").read_bytes()
        trial._phase = "cleaning" if stage == "cleanup" else "executed"
        trial._rewards = {"reward": 1}
        if solver_saved:
            prepare_terminal_result(trial)
            before["solver.json"] = (root / "solver.json").read_bytes()
        await asyncio.sleep(3600)

    trial._run_lifecycle = lifecycle
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.01")
    result = await trial.run()
    released.assert_not_awaited()
    assert recovery.needs_verifier_recovery(result.verifier_error)
    assert result.rewards is None and result.error is None
    _trial(root.parent, "pass", reward=1)
    job = bf.load_job(root.parent)
    assert len(job.trials) == 2
    assert job.solve_rates().solve_rate == 0.5
    saved = json.loads((root / "result.json").read_text())
    assert saved["rewards"] is None
    assert saved["scoring"]["status"] == "error"
    assert saved["verifier_error_category"] == "infra_failure"
    assert result.scoring.status == "error"
    assert collect_metrics(root.parent).summary()["mean_reward"] == 0.5
    assert job.to_json_dict()["solve_rates"]["trials"] == 2
    for name, content in before.items():
        assert (root / name).read_bytes() == content
    assert (root / "solver.json").is_file()
    review.assert_not_awaited()
    # Scoring admission can still finish later from the preserved solver;
    # a stale deadline must never overwrite that completed review.
    verifier_resume = AsyncMock()
    monkeypatch.setattr(recovery, "resume_verification", verifier_resume)
    resumed_review = AsyncMock(return_value=_score())
    monkeypatch.setattr(automatic, "finish_review", resumed_review)
    resumed = await resume_review(root, tasks_root=task)
    assert resumed["scoring"]["status"] == "complete"
    verifier_resume.assert_awaited_once()
    resumed_review.assert_awaited_once()
    admitted = (root / "result.json").read_bytes()
    recovered = recovery.interrupted_solver_result(trial, "stale deadline")
    assert recovered.scoring.status == "complete"
    assert (root / "result.json").read_bytes() == admitted
