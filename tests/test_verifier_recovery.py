"""Guards issue #1136 recovery without replaying completed solver work."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._utils.result_paths import load_task_results
from benchflow._utils.task_authoring import task_digest
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult
from benchflow.review.evidence import EvidenceError, EvidenceManifest
from benchflow.review.evidence_runtime import ensure_evidence_python
from benchflow.review.persistence import scoring_lock
from benchflow.review.resume import ReviewResumeError, resume_review
from benchflow.rollout import Rollout as SolverRollout
from benchflow.rollout import RolloutConfig, TaskRuntime, TaskRuntimeConfig, _deadline
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.rollout._review import prepare_terminal_result
from benchflow.sandbox._recovery_baseline import DockerRecoveryBaseline
from benchflow.sandbox.docker import _sanitize_docker_compose_project_name
from benchflow.task import RolloutPaths, Task
from benchflow.task.config import NetworkMode, SetupCommandConfig
from tests.test_automatic_review_scoring import _score


@pytest.fixture
def trial(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\nworkspace_recovery = true\n[sandbox]\ndocker_image = "python@sha256:'
        + "a" * 64
        + '"\n'
    )
    (task / "instruction.md").write_text("Produce a file.")
    root = tmp_path / "trial"
    bundle = root / "evidence"
    (bundle / "workspace").mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        EvidenceManifest(
            workspace="/app",
            archive_sha256="0" * 64,
            entries=(),
        ).model_dump_json()
    )
    return SimpleNamespace(
        _config=RolloutConfig(task_path=task, task_digest=task_digest(task)),
        _task=Task(task),
        _verifier_error="verifier_wedge: no receipt",
        _trajectory=[{"type": "agent_message", "content": "done"}],
        _require_rollout_dir=lambda: root,
        _planes=SimpleNamespace(setup_sandbox_user=AsyncMock()),
    )


@pytest.fixture
def child(monkeypatch):
    env = SimpleNamespace(
        upload_file=AsyncMock(),
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
        ),
    )
    child = SimpleNamespace(
        _env=env,
        _task=Mock(),
        _agent_cwd="/app",
        setup=AsyncMock(),
        start=AsyncMock(),
        install_agent=AsyncMock(),
        cleanup=AsyncMock(),
        run=AsyncMock(),
    )

    def make_child(cfg):
        child._rollout_paths = RolloutPaths(
            rollout_dir=cfg.jobs_dir / cfg.job_name / cfg.rollout_name
        )
        return child

    factory = Mock(side_effect=make_child)
    monkeypatch.setattr("benchflow.rollout.Rollout", factory)
    monkeypatch.setattr(recovery, "ensure_evidence_python", AsyncMock())
    monkeypatch.setattr(recovery, "install_uploaded_workspace", AsyncMock())
    monkeypatch.setattr(
        "benchflow.rollout._setup._publish_trajectory_for_verifier", AsyncMock()
    )
    child.factory = factory
    return child


def _receipt(root):
    return json.loads(next(root.glob("verifier-recovery/*/recovery.json")).read_text())


@pytest.mark.asyncio
async def test_fresh_verifier_scores_without_solver_replay(trial, child, monkeypatch):
    """Guards #1136: recovery starts no solver, keeps immutable evidence."""
    original = (trial._require_rollout_dir() / "evidence/manifest.json").read_bytes()
    verify = AsyncMock(return_value=({"reward": 1.0}, None, None))
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards == {"reward": 1.0}
    assert error is None
    child.run.assert_not_awaited()
    child.cleanup.assert_awaited_once()
    assert child.factory.call_args.args[0].primary_agent == "oracle"
    assert verify.call_args.args[0] is child._env
    assert (
        trial._require_rollout_dir() / "evidence/manifest.json"
    ).read_bytes() == original
    record = _receipt(trial._require_rollout_dir())
    assert record["status"] == "complete"
    assert record["solver_replayed"] is False


@pytest.mark.asyncio
async def test_recovered_mounted_reward_is_read_and_published(
    trial, child, monkeypatch
):
    """Real Docker writes rewards into the child mount, not the receipt directory."""

    async def start():
        mounted = child._rollout_paths.verifier_dir
        mounted.mkdir(parents=True)
        (mounted / "reward.json").write_text('{"reward": 1}')

    async def verify(env, task, paths, *args, **kwargs):
        return json.loads(paths.reward_json_path.read_text()), None, None

    child.start.side_effect = start
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    rewards, error = await recovery.recover_verifier(trial)
    assert (rewards, error) == ({"reward": 1}, None)
    root = trial._require_rollout_dir()
    pointer = json.loads((root / "verification.json").read_text())
    assert (
        json.loads((root / pointer["attempt"] / "verifier/reward.json").read_text())
        == rewards
    )
    assert json.loads((root / "verifier/reward.json").read_text()) == rewards
    child.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "task_changed", "external"])
async def test_unavailable_recovery_never_starts_fresh_solver(trial, child, failure):
    """Guards #1136: reject lost/incompatible evidence instead of replaying."""
    if failure == "missing":
        (trial._require_rollout_dir() / "evidence/manifest.json").unlink()
    elif failure == "task_changed":
        (trial._config.task_path / "instruction.md").write_text("Changed task")
    else:
        trial._config.services = ["database"]
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None
    assert error.startswith(recovery.PRESERVED_SOLVER)
    assert not RetryConfig().should_retry_verifier_error(error)
    child.factory.assert_not_called()


@pytest.mark.asyncio
async def test_recovery_failure_is_terminal_and_retains_receipt(
    trial, child, monkeypatch
):
    """Guards #1136: a second transport failure cannot replay the solver."""
    monkeypatch.setattr(
        "benchflow.rollout._setup._verify_rollout",
        AsyncMock(
            return_value=(None, "verifier_wedge: transport still unavailable", None)
        ),
    )
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None
    assert not RetryConfig().should_retry_verifier_error(error)
    child.run.assert_not_awaited()
    record = _receipt(trial._require_rollout_dir())
    assert record["status"] == "failed"
    assert record["original_error"] == trial._verifier_error


@pytest.mark.asyncio
async def test_cleanup_error_does_not_discard_recovered_score(
    trial, child, monkeypatch
):
    """Guards #1136: cleanup cannot turn a recovered score into solver retry."""
    child.cleanup.side_effect = RuntimeError("cleanup transport failed")
    monkeypatch.setattr(
        "benchflow.rollout._setup._verify_rollout",
        AsyncMock(return_value=({"reward": 1.0}, None, None)),
    )
    assert await recovery.recover_verifier(trial) == ({"reward": 1.0}, None)
    record = _receipt(trial._require_rollout_dir())
    assert record["cleanup_error"] == "cleanup transport failed"


@pytest.mark.asyncio
async def test_undeclared_workspace_contract_fails_explicitly(trial, child):
    """Guards #1136: a workspace archive is not proof of full VM state."""
    trial._task.config.verifier.workspace_recovery = False
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None
    assert "workspace_recovery=true" in error
    child.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ineligible", ["undeclared", "services"])
async def test_ineligible_recovery_leaves_no_attempt_artifacts(
    trial, child, ineligible
):
    """Guards the fix for solver-evidence preservation's stray attempts on ineligible tasks.

    Recovery created ``verifier-recovery/<id>/`` before its eligibility check
    and pointed ``verification.json`` at the "unavailable" attempt.
    """
    if ineligible == "undeclared":
        trial._task.config.verifier.workspace_recovery = False
    else:
        trial._config.services = ["database"]
    root = trial._require_rollout_dir()
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None
    assert error.startswith(
        recovery.PRESERVED_SOLVER + " verifier recovery unavailable"
    )
    assert not (root / "verifier-recovery").exists()
    assert not (root / "verification.json").exists()
    child.factory.assert_not_called()


def _save_solver(trial):
    root = trial._require_rollout_dir()
    source = {
        "task_name": "task",
        "task_digest": trial._config.task_digest,
        "purpose": "task",
        "rollout_name": "task__a",
        "rewards": None,
        "verifier_error": trial._verifier_error,
        "started_at": "2026-01-01 00:00:00",
        "finished_at": "2026-01-01 00:01:00",
        "agent": "opencode",
        "model": "solver-model",
        "agent_result": {"total_tokens": 123},
    }
    (root / "solver.json").write_text(json.dumps(source))
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_digest": trial._config.task_digest,
                "verifier_recovery": {"eligible": True},
            }
        )
    )
    (root / "prompts.json").write_text('["Produce a file"]')
    (root / "trajectory").mkdir()
    (root / "trajectory/acp_trajectory.jsonl").write_text("")
    return (root / "solver.json").read_bytes()


@pytest.mark.asyncio
async def test_interrupted_recovery_resumes_verifier_only(trial, child, monkeypatch):
    """Guards #1136: an interrupted recovery resumes without changing solver."""
    before = _save_solver(trial)
    verify = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    with pytest.raises(asyncio.CancelledError):
        await recovery.recover_verifier(trial)
    root = trial._require_rollout_dir()
    assert not (root / "verification.json").exists()
    record = _receipt(root)
    assert record["status"] == "interrupted"
    verify.side_effect = None
    verify.return_value = ({"reward": 1.0}, None, None)
    result = await resume_review(root, tasks_root=trial._config.task_path)
    assert result["rewards"] == {"reward": 1.0}
    assert result["verifier_error"] is None
    assert (root / "solver.json").read_bytes() == before
    assert result["agent_result"] == {"total_tokens": 123}
    child.run.assert_not_awaited()
    assert len(list(root.glob("verifier-recovery/*/recovery.json"))) == 2
    # A further resume uses the successful revision, without rerunning tests.
    await resume_review(root, tasks_root=trial._config.task_path)
    assert verify.await_count == 2


@pytest.mark.asyncio
async def test_resume_retains_original_ineligibility(trial, child, monkeypatch):
    """Guards #1136: omitted live hooks/services must not become eligible on resume.

    Since the fix for the solver-evidence preservation regression, an ineligible trial also keeps its
    original (retryable) verifier error and gets no recovery attempt artifacts;
    without a rubric, resume refuses it as before instead of
    rewriting its result.
    """
    _save_solver(trial)
    root = trial._require_rollout_dir()
    config = json.loads((root / "config.json").read_text())
    config["verifier_recovery"] = {
        "eligible": False,
        "reason": "external database state",
    }
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ReviewResumeError, match="no automatic review rubric"):
        await resume_review(root, tasks_root=trial._config.task_path)
    assert not (root / "result.json").exists()
    assert not (root / "verifier-recovery").exists()
    assert not (root / "verification.json").exists()
    child.factory.assert_not_called()


@pytest.mark.asyncio
async def test_trajectory_publication_failure_stays_on_verifier_channel(
    trial, monkeypatch
):
    """Guards #1136/#948: post-solver publication failure cannot trigger solver retry."""
    trial._env = SimpleNamespace()
    trial._rollout_paths = RolloutPaths(rollout_dir=trial._require_rollout_dir())
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._error = None
    trial._solver_execution_complete = True  # publication follows the stage checkpoint
    monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", AsyncMock())
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier",
        AsyncMock(
            side_effect=RuntimeError(
                "Failed to execute session command: channel closed"
            )
        ),
    )
    await SolverRollout.verify(trial)
    assert trial._error is None
    assert trial._verifier_error.startswith(recovery.PRESERVED_SOLVER)
    assert recovery.needs_verifier_recovery(trial._verifier_error)
    assert not RetryConfig().should_retry_verifier_error(trial._verifier_error)


@pytest.mark.asyncio
async def test_manual_task_runtime_uses_same_recovery_finalization(
    trial, child, monkeypatch
):
    """Guards #1136: manually driven TaskRuntime retains its one solver episode."""
    parent = SolverRollout(trial._config)
    parent.__dict__.update(vars(trial))
    parent._rollout_dir = trial._require_rollout_dir()
    parent._phase = "verified"
    parent.verify = AsyncMock()

    async def cleanup():
        parent._phase = "cleaned"

    parent.cleanup = AsyncMock(side_effect=cleanup)
    snapshots = []

    def build_result(*, result_filename="result.json"):
        if result_filename == "solver.json":
            snapshots.append(_save_solver(trial))
        return RolloutResult(
            task_name="task",
            rollout_name="task__a",
            rewards=parent._rewards,
            verifier_error=parent._verifier_error,
        )

    parent._build_result = Mock(side_effect=build_result)
    verify = AsyncMock(return_value=({"reward": 1.0}, None, None))
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    runtime = TaskRuntime(TaskRuntimeConfig(task_path=trial._config.task_path))
    runtime._rollout = parent
    runtime._started = True
    result = await runtime.verify()
    assert result.reward == 1.0
    assert len(snapshots) == 1
    assert (parent._rollout_dir / "solver.json").read_bytes() == snapshots[0]
    child.run.assert_not_awaited()
    child.cleanup.assert_awaited_once()
    # Idempotent finalization cannot execute the verifier or rewrite solver again.
    assert await parent.finalize() is result.result
    assert verify.await_count == 1
    parent.cleanup.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", ["planes", "context", "mutable_image", "setup"])
async def test_recovery_rejects_unreconstructible_baselines(trial, child, unsupported):
    """Guards #1136 review: immutable task files alone do not pin runtime state."""
    if unsupported == "planes":
        trial._config.planes = Mock()
    elif unsupported == "context":
        trial._config.context_root = trial._config.task_path.parent
    elif unsupported == "mutable_image":
        trial._task.config.sandbox.docker_image = "python:3.12-slim"
    else:
        trial._task.config.sandbox.setup_commands = [
            SetupCommandConfig(command="apt-get update")
        ]
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None and recovery.PRESERVED_SOLVER in error
    child.factory.assert_not_called()


@pytest.mark.asyncio
async def test_initial_recovery_and_cli_resume_share_lock(trial, monkeypatch):
    """Guards #1136 review: concurrent resume cannot replace an active attempt."""
    before = _save_solver(trial)
    parent = SolverRollout(trial._config)
    parent.__dict__.update(vars(trial))
    parent._rollout_dir = trial._require_rollout_dir()
    parent._phase = "cleaned"
    original = RolloutResult(
        task_name="task", rollout_name="task__a", verifier_error=trial._verifier_error
    )
    recovered = RolloutResult(
        task_name="task", rollout_name="task__a", rewards={"reward": 1.0}
    )
    parent._build_result = Mock(return_value=recovered)
    started, release = asyncio.Event(), asyncio.Event()

    async def recover(_):
        started.set()
        await release.wait()
        return {"reward": 1.0}, None

    run_recovery = AsyncMock(side_effect=recover)
    monkeypatch.setattr(recovery, "recover_verifier", run_recovery)
    active = asyncio.create_task(parent._finish_scoring(original))
    await started.wait()
    try:
        with pytest.raises(ValueError, match="Scoring is already running"):
            await resume_review(parent._rollout_dir, tasks_root=trial._config.task_path)
    finally:
        release.set()
    assert await active is recovered
    run_recovery.assert_awaited_once()
    assert (parent._rollout_dir / "solver.json").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["capture", "hardening", "cleanup"])
async def test_post_solver_capture_deadline_preserves_completion(
    trial, monkeypatch, stage
):
    """Guards #1136 review: a capture wedge cannot make the solver retryable."""
    parent = SolverRollout(trial._config)
    parent.__dict__.update(vars(trial))
    parent._rollout_dir = trial._require_rollout_dir()
    parent._rollout_name = "task__a"
    parent._rollout_paths = RolloutPaths(rollout_dir=parent._rollout_dir)
    parent._phase = "executed"
    parent._verifier_error = None

    def build_result(*, result_filename="result.json"):
        source = {
            "task_name": "task",
            "task_digest": trial._config.task_digest,
            "rollout_name": "task__a",
            "rewards": parent._rewards,
            "verifier_error": parent._verifier_error,
            "error": None,
            "agent_result": {"total_tokens": 321},
        }
        (parent._rollout_dir / result_filename).write_text(json.dumps(source))
        return RolloutResult(task_name="task", rollout_name="task__a", total_tokens=321)

    parent._build_result = Mock(side_effect=build_result)

    async def capture(_):
        assert (parent._rollout_dir / "solver-complete.json").is_file()
        await asyncio.sleep(3600)

    monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", capture)
    parent._run_lifecycle = parent.verify
    if stage == "hardening":
        parent._env = SimpleNamespace()
        parent._timing = {}
        parent._agent_cwd = "/app"

        async def harden(*args, **kwargs):
            await asyncio.sleep(3600)

        parent._planes.harden_before_verify = harden
        monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", AsyncMock())
        monkeypatch.setattr(
            "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
        )
    elif stage == "cleanup":

        async def cleanup_lifecycle():
            recovery.mark_solver_complete(parent)
            parent._phase = "cleaning"
            await asyncio.sleep(3600)

        parent._run_lifecycle = cleanup_lifecycle
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.01")
    result = await parent.run()
    assert result.rollout_name == "task__a"
    assert result.total_tokens == 321
    assert result.error is None
    assert not RetryConfig().should_retry_verifier_error(result.verifier_error)
    persisted = load_task_results(parent._rollout_dir.parent)["task"]
    assert recovery.PRESERVED_SOLVER in persisted["verifier_error"]
    assert persisted["agent_result"]["total_tokens"] == 321
    assert persisted["telemetry_finalized"] is False


def test_process_exit_after_solver_checkpoint_cannot_rerun_solver(trial):
    """Guards #1136 review: pre-capture process exit retains completed work."""
    root = trial._require_rollout_dir()
    (root / "solver-complete.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "purpose": "task",
                "rewards": None,
                "verifier_error": recovery.PRESERVED_SOLVER + " verifier pending",
                "execution_stage": "solver_complete",
                "telemetry_finalized": False,
            }
        )
    )
    job = Evaluation.__new__(Evaluation)
    job._jobs_dir = root.parent
    job._job_name = ""
    job._config = EvaluationConfig()
    assert "task" in job._get_completed_tasks()
    assert not (root / "result.json").exists()


@pytest.mark.asyncio
async def test_pinned_recovery_runtime_does_not_install_packages():
    """Guards #1136 review: capture setup cannot mutate an immutable baseline."""
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=1, stdout="", stderr="no python")
        )
    )
    with pytest.raises(
        EvidenceError, match="Could not prepare workspace capture Python"
    ):
        await ensure_evidence_python(env, allow_install=False)
    command = env.exec.call_args.args[0]
    assert "apt-get" not in command and "apk" not in command
    assert command.startswith("python3 -I")


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["start", "verify"])
async def test_abandoned_recovery_cannot_publish_after_lock_release(
    trial, child, monkeypatch, blocked_phase
):
    """Guards #1136: cancelled attempts lose admission even when awaits swallow cancellation."""
    root = trial._require_rollout_dir()
    (root / "verifier").mkdir()
    (root / "verifier/reward.txt").write_text("original")
    started, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def resistant():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    if blocked_phase == "start":
        child.start = AsyncMock(side_effect=resistant)

    async def verify(env, task, paths, *args, **kwargs):
        if blocked_phase == "verify":
            await resistant()
        paths.verifier_dir.mkdir(parents=True)
        (paths.verifier_dir / "reward.txt").write_text("late stale verdict")
        return {"reward": 1.0}, None, None

    child.cleanup = AsyncMock(side_effect=lambda: cleaned.set())
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.01)
    finished = asyncio.Event()
    original_callback = _deadline._swallow_abandoned_outcome

    def observe_finish(task):
        original_callback(task)
        finished.set()

    monkeypatch.setattr(_deadline, "_swallow_abandoned_outcome", observe_finish)
    with scoring_lock(root):
        active = asyncio.create_task(recovery.recover_verifier(trial))
        await started.wait()
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
    with scoring_lock(root):
        (root / "verifier/reward.txt").write_text("new admitted verdict")
        (root / "verification.json").write_text('{"attempt":"new-admitted"}')
    release.set()
    await asyncio.wait_for(cleaned.wait(), 1)
    await asyncio.wait_for(finished.wait(), 1)
    assert (root / "verifier/reward.txt").read_text() == "new admitted verdict"
    assert (root / "verification.json").read_text() == '{"attempt":"new-admitted"}'
    receipt = _receipt(root)
    assert receipt["status"] == "interrupted"
    assert receipt["admission"] == "revoked"
    child.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_does_not_publish_over_admitted_completed_scoring(tmp_path):
    """Guards #1136: finalize checks admission under lock before any result writer."""
    parent = SolverRollout(RolloutConfig(task_path=tmp_path))
    parent._phase = "cleaned"
    parent._rollout_dir = tmp_path
    score = _score()
    original = {
        "task_name": "task",
        "rollout_name": "task__a",
        "rewards": score.numeric_rewards(),
        "scoring": score.to_dict(),
        "agent_result": {"total_tokens": 41},
    }
    (tmp_path / "result.json").write_text(json.dumps(original))
    (tmp_path / "solver.json").write_text('{"immutable":true}')
    before_result = (tmp_path / "result.json").read_bytes()
    before_solver = (tmp_path / "solver.json").read_bytes()
    parent._build_result = Mock(
        side_effect=AssertionError("must not publish stale result")
    )
    result = await parent.finalize()
    assert isinstance(result, RolloutResult)
    assert result.rewards == score.numeric_rewards()
    assert result.total_tokens == 41
    assert result.scoring.status == "complete"
    assert (tmp_path / "result.json").read_bytes() == before_result
    assert (tmp_path / "solver.json").read_bytes() == before_solver
    parent._build_result.assert_not_called()


def test_terminal_publication_rejects_concurrent_scoring_before_any_write(tmp_path):
    """Guards #1136: lifecycle publication cannot bypass another scorer's admission lock."""
    parent = SolverRollout(RolloutConfig(task_path=tmp_path))
    parent._rollout_dir = tmp_path
    parent._build_result = Mock(
        side_effect=AssertionError("must not publish without lock")
    )
    with (
        scoring_lock(tmp_path),
        pytest.raises(ValueError, match="Scoring is already running"),
    ):
        prepare_terminal_result(parent)
    parent._build_result.assert_not_called()


@pytest.mark.asyncio
async def test_denylist_recovery_rejected_before_unrestricted_child_starts(
    trial, child
):
    """Guards #1136: oracle setup alone does not recreate connect-time denylist enforcement."""
    trial._task.config.sandbox.network_mode = NetworkMode.DENYLIST
    trial._task.config.sandbox.blocked_hosts = ["example.test"]
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None
    assert "policy bootstrap" in error
    child.factory.assert_not_called()


@pytest.mark.asyncio
async def test_lifecycle_tail_does_not_overwrite_previously_admitted_score(tmp_path):
    """Guards #1136: real lifecycle tail publication honors an existing scoring admission."""
    parent = SolverRollout(RolloutConfig(task_path=tmp_path))
    parent._rollout_dir = tmp_path
    score = _score()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rollout_name": "task__a",
                "rewards": score.numeric_rewards(),
                "scoring": score.to_dict(),
            }
        )
    )
    before = (tmp_path / "result.json").read_bytes()
    parent.setup = AsyncMock(side_effect=RuntimeError("stale lifecycle setup"))

    async def cleanup():
        parent._phase = "cleaned"

    parent.cleanup = AsyncMock(side_effect=cleanup)
    parent._build_result = Mock(
        side_effect=AssertionError("must not overwrite admitted result")
    )
    result = await parent._run_lifecycle()
    assert result.rewards == score.numeric_rewards()
    assert result.scoring.status == "complete"
    assert (tmp_path / "result.json").read_bytes() == before
    parent._build_result.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_recovery_attempts_use_distinct_compose_projects(
    trial, child, monkeypatch
):
    """GH #1136: concurrent recovery must not share/delete a verifier Compose project."""
    first_root = trial._require_rollout_dir()
    second_root = first_root.parent / "second-trial"
    shutil.copytree(first_root, second_root)
    other = SimpleNamespace(**vars(trial))
    other._require_rollout_dir = lambda: second_root
    both_started = asyncio.Event()
    starts = 0

    async def start():
        nonlocal starts
        starts += 1
        if starts == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=3)

    child.start.side_effect = start
    monkeypatch.setattr(
        "benchflow.rollout._setup._verify_rollout",
        AsyncMock(return_value=({"reward": 1.0}, None, None)),
    )
    results = await asyncio.gather(
        recovery.recover_verifier(trial), recovery.recover_verifier(other)
    )
    assert results == [({"reward": 1.0}, None), ({"reward": 1.0}, None)]
    configs = [call.args[0] for call in child.factory.call_args_list]
    assert len(configs) == 2
    # create_environment passes rollout_name as DockerSandbox.session_id;
    # Compose's actual normalization must retain distinct project identities.
    projects = [
        _sanitize_docker_compose_project_name(cfg.rollout_name) for cfg in configs
    ]
    assert len(set(projects)) == 2
    for cfg in configs:
        assert cfg.jobs_dir.name in cfg.rollout_name
        assert cfg.rollout_name.startswith("verifier-")


@pytest.mark.asyncio
async def test_pending_recovery_receipt_names_explicit_submission_contract(
    trial, monkeypatch
):
    """GH #1136: even interrupted/pending receipts must identify the requested file scope.

    Stops after the eligibility check: since the fix for solver-evidence preservation's stray
    attempts, an ineligible trial writes no receipt at all.
    """
    trial._task.config.verifier.workspace_recovery = False
    trial._task.config.verifier.submission_files = [
        "/root/result.csv",
        "/root/summary.json",
        "/root/report.md",
    ]

    def stop(_):
        raise EvidenceError("test stop before startup")

    monkeypatch.setattr(recovery, "admitted_submission", stop)
    rewards, error = await recovery.recover_verifier(trial)
    assert rewards is None and "test stop before startup" in error
    assert _receipt(trial._require_rollout_dir())["evidence"] == "submission-files-v1"


def _lease():
    return DockerRecoveryBaseline(
        image_id="sha256:" + "a" * 64,
        lease_tag="benchflow-recovery-lease:" + "1" * 32,
        effective_allow_internet=False,
        daemon_id="daemon-one",
        task_digest="task-digest",
        effective_config_digest="b" * 64,
        sandbox_config_digest="c" * 64,
    )


@pytest.fixture
def released(monkeypatch):
    release = AsyncMock(return_value=True)
    monkeypatch.setattr(recovery, "release_baseline", release)
    return release


def _leased_rollout(trial, verifier_error, *, owned=True):
    rollout = SolverRollout(RolloutConfig(task_path=trial._config.task_path))
    rollout._task = trial._task
    rollout._rollout_dir = trial._require_rollout_dir()
    rollout._env = SimpleNamespace(stop=AsyncMock())
    rollout._verifier_error = verifier_error
    rollout._docker_recovery_baseline = _lease()
    rollout._recovery_lease_owned = owned
    return rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verifier_error", "owned", "expected"),
    [
        (None, True, 1),  # scored: recovery not needed
        ("[solver-preserved] verifier crashed: verifier_wedge: lost", True, 0),
        (None, False, 0),  # a recovery child borrows the parent's lease
    ],
)
async def test_teardown_releases_lease_only_when_recovery_is_not_needed(
    trial, released, verifier_error, owned, expected
):
    """Guards the fix for image leases leaked by leased-image verifier recovery at rollout teardown."""
    rollout = _leased_rollout(trial, verifier_error, owned=owned)
    await rollout.cleanup()
    await rollout.cleanup()  # idempotent across repeated teardown
    assert released.await_count == expected
    rollout._env.stop.assert_awaited()


@pytest.mark.asyncio
async def test_finalization_without_recovery_releases_lease(trial, released):
    """Guards the same fix: finalization frees a lease no recovery will use."""
    rollout = _leased_rollout(trial, None)
    rollout._phase = "cleaned"
    result = RolloutResult(task_name="task", rollout_name="task__a")
    assert await rollout._finish_scoring(result) is result
    released.assert_awaited_once_with(rollout._docker_recovery_baseline)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "failed", "interrupted"])
async def test_recovery_attempt_releases_lease_once_finished(
    trial, child, monkeypatch, released, outcome
):
    """Guards the same fix: success or failure frees the lease; interruption keeps it for resume."""
    trial._docker_recovery_baseline = _lease()
    monkeypatch.setattr(recovery, "validate_task_identity", Mock())
    monkeypatch.setattr(recovery, "DockerSandbox", SimpleNamespace)
    child._env.use_recovery_baseline = Mock()
    verify = AsyncMock(
        side_effect=asyncio.CancelledError() if outcome == "interrupted" else None,
        return_value=({"reward": 1.0}, None, None)
        if outcome == "complete"
        else (None, "verifier_wedge: still lost", None),
    )
    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    if outcome == "interrupted":
        with pytest.raises(asyncio.CancelledError):
            await recovery.recover_verifier(trial)
        released.assert_not_awaited()
    else:
        await recovery.recover_verifier(trial)
        released.assert_awaited_once_with(trial._docker_recovery_baseline)
    child.cleanup.assert_awaited_once()
    assert _receipt(trial._require_rollout_dir())["status"] == outcome


@pytest.mark.asyncio
async def test_failed_publication_keeps_original_verifier_dir_and_verdict(
    trial, child, monkeypatch
):
    """Guards the fix for solver-evidence preservation's non-atomic publish of recovered outputs.

    It moved verifier/ away and then copied into place, so a mid-copy failure
    left a half-written verifier/ and skipped the verdict pointer.
    """
    root = trial._require_rollout_dir()
    (root / "verifier").mkdir()
    (root / "verifier/reward.txt").write_text("original")

    async def verify(env, task, paths, *args, **kwargs):
        paths.verifier_dir.mkdir(parents=True)
        (paths.verifier_dir / "a.txt").write_text("recovered")
        (paths.verifier_dir / "reward.txt").write_text("1")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    real_copy2 = shutil.copy2
    copied = []

    def flaky_copy2(src, dst, **kwargs):
        # Receipt-local copies succeed; the canonical publish fails mid-copy.
        if (root / "verifier-recovery") not in Path(dst).parents:
            copied.append(dst)
            if len(copied) == 2:
                raise OSError(28, "No space left on device")
        return real_copy2(src, dst, **kwargs)

    real_copytree = shutil.copytree

    def copytree(src, dst, **kwargs):
        return real_copytree(src, dst, **{"copy_function": flaky_copy2, **kwargs})

    monkeypatch.setattr(shutil, "copytree", copytree)
    rewards, error = await recovery.recover_verifier(trial)
    assert (rewards, error) == ({"reward": 1.0}, None)
    assert copied, "publication must have been attempted"
    assert [p.name for p in (root / "verifier").iterdir()] == ["reward.txt"]
    assert (root / "verifier/reward.txt").read_text() == "original"
    assert sorted(p.name for p in root.iterdir()) == [
        "evidence",
        "verification.json",
        "verifier",
        "verifier-recovery",
    ]
    pointer = json.loads((root / "verification.json").read_text())["attempt"]
    record = json.loads((root / pointer / "recovery.json").read_text())
    assert record["status"] == "complete" and record["rewards"] == {"reward": 1.0}
    assert "No space left on device" in record["publication_error"]
    assert (root / pointer / "verifier/a.txt").read_text() == "recovered"
    assert not (root / pointer / "previous-verifier").exists()
