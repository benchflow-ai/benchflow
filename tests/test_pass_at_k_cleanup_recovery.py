"""Guard P4-CONF-2's cancellation cleanup regression at 3d473f78."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.review import automatic
from benchflow.review.evidence import EvidenceManifest
from benchflow.review.resume import resume_review
from benchflow.rollout import Rollout
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.sandbox._recovery_baseline import config_digest, validate_baseline
from benchflow.task import Task
from tests.test_automatic_review_scoring import _score
from tests.test_review_lifecycle import reviewed_rollout as _reviewed_rollout
from tests.test_verifier_recovery import _lease
from tests.test_verifier_recovery import child as _child

reviewed_rollout = _reviewed_rollout
child = _child


@pytest.fixture
def leased_trial(reviewed_rollout, monkeypatch):
    trial = reviewed_rollout
    task = trial._config.task_path
    (task / "task.md").write_text(
        "---\nverifier:\n  workspace_recovery: true\nsandbox:\n"
        + '  docker_image: "python@sha256:'
        + "a" * 64
        + '"\n---\nProduce a file.\n'
    )
    trial._task = Task(task)
    trial._review_plan = automatic.prepare_review(task, trial._config.reviewer)
    trial._config.task_digest = trial._review_plan.task_digest
    trial._config.sandbox_user = None
    trial._docker_recovery_baseline = _lease().model_copy(
        update={
            "task_digest": trial._config.task_digest,
            "effective_config_digest": config_digest(trial._task.config),
            "sandbox_config_digest": config_digest(trial._task.config.sandbox),
        }
    )
    trial._recovery_lease_owned = True
    root = trial._require_rollout_dir()
    (root / "docker-recovery-baseline.json").write_text(
        trial._docker_recovery_baseline.model_dump_json()
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_digest": trial._config.task_digest,
                "review": trial._review_plan.metadata(),
                "verifier_recovery": {"eligible": True},
                "sandbox_user": None,
            }
        )
    )
    # Model only the Docker tag table. No real daemon or image is touched.
    tags = {trial._docker_recovery_baseline.lease_tag}

    async def release(baseline):
        tags.discard(baseline.lease_tag)
        return True

    released = AsyncMock(side_effect=release)
    monkeypatch.setattr(recovery, "release_baseline", released)
    trial._env = SimpleNamespace(stop=AsyncMock())
    trial._artifacts_collected = True
    # Keep the real cleanup, including release_lease_at_teardown, and the
    # real _run_lifecycle finally. Only sandbox operations are substituted.
    trial.cleanup = Rollout.cleanup.__get__(trial)
    return trial, tags, released


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["deadline", "external"])
async def test_cancelled_verifier_cleanup_keeps_lease_for_scoring_resume(
    leased_trial, monkeypatch, request, cancellation
):
    """Guards P4-CONF-2 at 3d473f78 through real cancellation/finally cleanup."""
    trial, tags, released = leased_trial
    root = trial._require_rollout_dir()
    verifying = asyncio.Event()

    async def verify():
        trial._phase = "verifying"
        bundle = root / "evidence"
        (bundle / "workspace").mkdir(parents=True)
        (bundle / "manifest.json").write_text(
            EvidenceManifest(
                workspace="/app", archive_sha256="0" * 64, entries=()
            ).model_dump_json()
        )
        verifying.set()
        await asyncio.Event().wait()

    trial.verify = AsyncMock(side_effect=verify)
    # Leave scoring to its separate queue (`bench eval score` below).
    trial._finish_scoring = AsyncMock(side_effect=lambda result: result)
    monkeypatch.setenv(
        "BENCHFLOW_ROLLOUT_HARD_DEADLINE",
        "0.05" if cancellation == "deadline" else "60",
    )
    running = asyncio.create_task(trial.run())
    await asyncio.wait_for(verifying.wait(), timeout=5)
    if cancellation == "external":
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not (root / "result.json").exists()
        assert not (root / "solver.json").exists()
        assert (root / "solver-complete.json").is_file()
    else:
        result = await running
        assert result.rewards is None
        assert result.scoring.status == "error"
        assert recovery.needs_verifier_recovery(result.verifier_error)
    assert trial._phase == "cleaned"
    trial._env.stop.assert_awaited_once_with(delete=True)
    released.assert_not_awaited()
    assert trial._docker_recovery_baseline.lease_tag in tags
    if cancellation == "external":
        return

    original = (root / "solver.json").read_bytes()
    evidence = (root / "evidence/manifest.json").read_bytes()
    # Run the actual resume/recovery code. The sandbox boundary is fake,
    # but original task identity and lease validation remain real.
    child = request.getfixturevalue("child")
    child._task = trial._task
    child._env.task_env_config = trial._task.config.sandbox
    child._env.use_recovery_baseline = Mock()
    monkeypatch.setattr(recovery, "DockerSandbox", SimpleNamespace)
    monkeypatch.setattr(recovery, "prepare_recovery_network", AsyncMock())

    async def docker_cli(args, *, raw_output=False):
        assert raw_output
        value = (
            trial._docker_recovery_baseline.daemon_id
            if args[0] == "info"
            else trial._docker_recovery_baseline.image_id
            if args[-1] in tags
            else ""
        )
        return SimpleNamespace(stdout=value)

    child._env._docker_cli = AsyncMock(side_effect=docker_cli)

    async def start():
        await validate_baseline(child._env, trial._docker_recovery_baseline)

    child.start.side_effect = start
    verify_again = AsyncMock(return_value=({"reward": 1.0}, None, None))
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify_again)
    review = AsyncMock(return_value=_score())
    monkeypatch.setattr(automatic, "finish_review", review)
    resumed = await resume_review(root, tasks_root=trial._config.task_path)
    assert resumed["scoring"]["status"] == "complete"
    assert resumed["rewards"]["reward"] == 0.8
    assert resumed["rewards"]["verifier_reward"] == 1.0
    verify_again.assert_awaited_once()
    review.assert_awaited_once()
    child.run.assert_not_awaited()
    trial._run_steps.assert_awaited_once()
    assert (root / "solver.json").read_bytes() == original
    assert (root / "evidence/manifest.json").read_bytes() == evidence
    released.assert_awaited_once()
    assert tags == set()


@pytest.mark.asyncio
async def test_completed_solver_releases_lease_after_successful_finalization(
    leased_trial, monkeypatch
):
    """Guards P4-CONF-2 at 3d473f78 without leaking successful solver leases."""
    trial, tags, released = leased_trial
    monkeypatch.setattr(
        "benchflow.rollout._review.finish_review", AsyncMock(return_value=_score())
    )
    result = await trial.run()
    assert result.scoring.status == "complete"
    released.assert_awaited_once_with(trial._docker_recovery_baseline)
    assert tags == set()
