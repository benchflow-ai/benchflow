"""Scoring-stage resume for automatic review introduced after PR #1126.

Guards the rubric integration against the solver replay behavior at commit
PR #1126: a reviewer failure must never spend on a second solver trajectory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from typer.testing import CliRunner

from benchflow._utils.task_authoring import task_digest
from benchflow.diagnostics import (
    IdleTimeoutDiagnostic,
    ProviderApiErrorDiagnostic,
    RolloutDiagnostics,
)
from benchflow.review import automatic, persistence
from benchflow.review.options import ReviewerConfig
from benchflow.review.outcome import ScoringResult, scoring_error
from benchflow.review.resume import (
    ReviewResumeError,
    resume_pending_reviews,
    resume_review,
)
from benchflow.trajectories.results import (
    _record_to_redacted_json_line,
    _structured_diagnostics_info,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def saved_trial(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "tasks" / "physics"
    _write(task / "rubric.json", [{"name": "quality"}])
    rollout = tmp_path / "job" / "physics__a"
    _write(
        rollout / "solver.json",
        {
            "task_name": "physics",
            "task_digest": task_digest(task),
            "purpose": "task",
            "rollout_name": rollout.name,
            "rewards": {"reward": 1.0},
            "started_at": "2026-09-14 00:00:00",
            "finished_at": "2026-09-14 00:01:00",
            "agent": "opencode",
            "model": "solver-model",
            "agent_result": {"total_tokens": 123},
        },
    )
    _write(
        rollout / "config.json",
        {
            "task_digest": task_digest(task),
            "review": {
                "reviewer": {
                    "agent": "opencode",
                    "model": "original-review-model",
                    "environment": "daytona",
                    "timeout_sec": 1234,
                    "agent_env_keys": ["AZURE_API_KEY"],
                },
            },
        },
    )
    _write(rollout / "prompts.json", ["Solve physics"])
    trajectory = rollout / "trajectory" / "acp_trajectory.jsonl"
    trajectory.parent.mkdir()
    trajectory.write_text("")
    return rollout, task


def _complete(*, passed: bool = True) -> ScoringResult:
    return ScoringResult(
        status="complete",
        passed=passed,
        tests_pass=True,
        all_blockers_pass=passed,
        failed_blockers=[] if passed else ["correctness"],
        verifier_reward=1.0,
        rubric_reward=0.8,
        reviewer_run="reviews/attempt-001/reviewer",
    )


def _parent(rollout: Path, scoring: ScoringResult) -> dict:
    source = json.loads((rollout / "solver.json").read_text())
    return {
        **source,
        "scoring": scoring.to_dict(),
        "rewards": scoring.numeric_rewards(),
    }


@pytest.mark.asyncio
async def test_resume_only_reviews_and_preserves_solver_identity(
    saved_trial, monkeypatch
):
    rollout, task = saved_trial
    original = (rollout / "solver.json").read_bytes()
    verdict = _complete()
    prepare = Mock(return_value=SimpleNamespace(config=ReviewerConfig()))
    finish = AsyncMock(return_value=verdict)
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    result = await resume_review(rollout, tasks_root=task.parent)

    finish.assert_awaited_once_with(prepare.return_value, rollout)
    assert (rollout / "solver.json").read_bytes() == original
    assert result["agent_result"]["total_tokens"] == 123
    assert result["scoring"]["passed"] is True
    assert result["rewards"]["reward"] == 0.8
    saved_config = prepare.call_args.args[1]
    assert saved_config.model == "original-review-model"
    assert saved_config.environment == "daytona"
    assert saved_config.agent_env == {}


@pytest.fixture
def other_mount():
    """A private folder on /dev/shm when it is another filesystem than the
    cwd and TMPDIR (BenchFlow's pinned roots besides /)."""
    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("no writable /dev/shm")
    devices = {os.stat(p).st_dev for p in (Path.cwd(), Path(tempfile.gettempdir()))}
    if shm.stat().st_dev in devices:
        pytest.skip("/dev/shm is not another mount here")
    folder = Path(tempfile.mkdtemp(dir=shm))
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.asyncio
async def test_a_rollout_on_another_mount_is_reviewed(
    saved_trial, other_mount, monkeypatch
):
    """FP patch 95: a resume pins the rollout as the run did, so the review
    reads its evidence through BenchFlow's no-mount-crossing reader."""
    from benchflow.review import evidence

    rollout, task = saved_trial
    moved = other_mount / "job" / rollout.name
    shutil.copytree(rollout.parent, moved.parent)
    verdict = _complete()
    read: list[str] = []

    async def finish(plan, rollout_dir):
        read.append(evidence._digest(rollout_dir / "solver.json"))
        return verdict

    prepare = Mock(return_value=SimpleNamespace(config=ReviewerConfig()))
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    result = await resume_review(moved, tasks_root=task.parent)

    assert result["scoring"]["passed"] is True
    assert len(read) == 1


@pytest.mark.asyncio
async def test_valid_negative_is_not_rejudged(saved_trial, monkeypatch):
    rollout, task = saved_trial
    negative = _parent(rollout, _complete(passed=False))
    _write(rollout / "result.json", negative)
    prepare = Mock(side_effect=AssertionError("must not prepare another reviewer"))
    monkeypatch.setattr(automatic, "prepare_review", prepare)

    assert await resume_review(rollout, tasks_root=task.parent) == negative
    prepare.assert_not_called()


@pytest.mark.asyncio
async def test_force_rejudges_valid_negative_with_explicit_overrides(
    saved_trial, monkeypatch
):
    rollout, task = saved_trial
    _write(rollout / "result.json", _parent(rollout, _complete(passed=False)))
    prepare = Mock(return_value=object())
    finish = AsyncMock(return_value=_complete())
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    await resume_review(
        rollout,
        tasks_root=task,
        reviewer=ReviewerConfig(model="new-review-model"),
        force=True,
    )

    config = prepare.call_args.args[1]
    assert config.model == "new-review-model"
    assert config.environment == "daytona"
    assert config.timeout_sec == 1234
    finish.assert_awaited_once()


@pytest.mark.asyncio
async def test_changed_task_rejected_before_reviewer(saved_trial, monkeypatch):
    rollout, task = saved_trial
    (task / "rubric.json").write_text("changed")
    finish = AsyncMock()
    monkeypatch.setattr(automatic, "finish_review", finish)
    with pytest.raises(ReviewResumeError, match="Task digest mismatch"):
        await resume_review(rollout, tasks_root=task.parent)
    finish.assert_not_awaited()


def _link(entry: Path, target: Path) -> None:
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.symlink_to(target, target_is_directory=target.is_dir())


@pytest.mark.asyncio
async def test_a_task_linked_into_the_root_is_reviewed(
    saved_trial, tmp_path, monkeypatch
):
    """#1126's guard, narrowed (FP patch 32).

    #1126 (edbf30cab) refused every task whose real folder lies outside
    --tasks-root, so its test refused a link sitting directly in the root
    with the task's own name and digest. ``bench run`` runs such a root (a
    task list made of links into a task store), so every review resume of
    that run failed and a reviewer that died in the run left the trial
    unscored. A review resume now takes a root entry by bench run's rule,
    ``tasks_root_entry_refusal``: a link directly in the root to a folder of
    the same name is followed, and solver.json's task_digest pins the files
    it reaches. The tests below keep the rest of #1126's guard: a link to a
    folder of another name, to a file or to another link, a link not
    directly in the root, and changed task files are still refused, and
    links inside the task still go through copy_tree's rules.
    """
    rollout, task = saved_trial
    listed = tmp_path / "listed"
    _link(listed / "physics", task)
    prepare = Mock(return_value=SimpleNamespace(config=ReviewerConfig()))
    finish = AsyncMock(return_value=_complete())
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    result = await resume_review(rollout, tasks_root=listed)

    assert prepare.call_args.args[0] == task.resolve()
    finish.assert_awaited_once_with(prepare.return_value, rollout)
    assert result["scoring"]["passed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        ("other-name", "a linked task keeps its folder's name"),
        ("file", "which is not a folder"),
        ("link-to-link", "which is a link too"),
        ("nested", "has no folder physics"),
        ("parent", "is not the name of a folder"),
    ],
)
async def test_a_root_entry_bench_run_would_not_take_is_refused(
    saved_trial, tmp_path, monkeypatch, layout, reason
):
    """Each layout reaches files with the solver's digest; only the link is wrong.

    other-name: a link to a copy of the task in a folder of another name;
    file: a link to a file; link-to-link: a link to a link to the task;
    nested: the task linked one folder down (task_name ``group/physics``);
    parent: task_name ``..``, the folder that holds the root.
    """
    rollout, task = saved_trial
    root = tmp_path / "listed"
    root.mkdir()
    name = "physics"
    if layout == "other-name":
        renamed = tmp_path / "store" / "chemistry"
        shutil.copytree(task, renamed)
        _link(root / name, renamed)
    elif layout == "file":
        (tmp_path / "store").mkdir()
        (tmp_path / "store" / name).write_text("")
        _link(root / name, tmp_path / "store" / name)
    elif layout == "link-to-link":
        _link(tmp_path / "middle" / name, task)
        _link(root / name, tmp_path / "middle" / name)
    elif layout == "nested":
        _link(root / "group" / name, task)
        name = "group/physics"
    else:
        root = task / "inner"
        root.mkdir()
        name = ".."
    solver = json.loads((rollout / "solver.json").read_text())
    _write(rollout / "solver.json", {**solver, "task_name": name})
    prepare = Mock(side_effect=AssertionError("must not prepare a reviewer"))
    monkeypatch.setattr(automatic, "prepare_review", prepare)

    # The refusal names the rule's own reason (a review): a link to a file
    # is refused by the rule, not only by the folder check after it.
    refusal = rf"in --tasks-root .+ is refused: .*{re.escape(reason)}"
    with pytest.raises(ReviewResumeError, match=refusal):
        await resume_review(rollout, tasks_root=root)
    prepare.assert_not_called()


@pytest.mark.parametrize("caller", ["resume", "bench review"])
@pytest.mark.parametrize(
    ("swapped_to", "reason"),
    [
        ("chemistry", "a linked task keeps its folder's name"),
        ("linked-env/physics", "environment/extra"),
    ],
)
def test_a_link_swapped_after_the_entry_check_is_judged_on_the_folder_used(
    tmp_path, monkeypatch, swapped_to, reason, caller
):
    """a review: the review checked the root's entry, then resolved
    it a second time, so a link swapped between the two took a copy of the
    task (same files, same digest) in a folder of another name, or with a
    link bench run refuses in a folder it uploads. The folder the review
    resolves once is the one judged, by a resume and by ``bench review
    --tasks-root`` alike (one helper, ``tasks_root_entry_folder``)."""
    from benchflow._utils import task_authoring
    from benchflow.review import resume
    from benchflow.review.runner import _source_task_dir

    task = tmp_path / "store" / "physics"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "task.md").write_text('---\nschema_version: "1.3"\n---\n\nDo nothing.\n')
    root = tmp_path / "tasks"
    _link(root / "physics", task)
    other = tmp_path / "store" / swapped_to
    shutil.copytree(task, other)
    if swapped_to.startswith("linked-env"):
        (other / "environment" / "extra").symlink_to("Dockerfile")
    solver = {"task_name": "physics", "task_digest": task_digest(task)}
    assert task_digest(other) == solver["task_digest"]
    check = task_authoring.tasks_root_entry_refusal

    def check_then_swap(root_dir: Path, name: str) -> str | None:
        refusal = check(root_dir, name)
        (root / "physics").unlink()
        _link(root / "physics", other)
        return refusal

    monkeypatch.setattr(task_authoring, "tasks_root_entry_refusal", check_then_swap)
    if caller == "resume":
        with pytest.raises(ReviewResumeError, match=reason):
            resume._trusted_task(tmp_path / "rollout", solver, root)
        return
    rollout = tmp_path / "rollout"
    _write(
        rollout / "config.json",
        {"task_path": "/elsewhere/physics", "task_digest": solver["task_digest"]},
    )
    folder, skip = _source_task_dir(rollout, tasks_root=root)
    assert folder is None
    assert skip is not None and skip.startswith("task evidence skipped: ")
    assert reason in skip


@pytest.mark.asyncio
async def test_a_linked_task_still_pins_its_digest(saved_trial, tmp_path):
    rollout, task = saved_trial
    listed = tmp_path / "listed"
    _link(listed / "physics", task)
    (task / "rubric.json").write_text("changed")
    with pytest.raises(ReviewResumeError, match="Task digest mismatch"):
        await resume_review(rollout, tasks_root=listed)


def test_links_inside_a_linked_task_keep_copy_tree_rules(saved_trial, tmp_path):
    """Following the root's link does not widen what is copied from the task.

    copy_tree judges each link inside the task against the task's real
    folder, whether it is given the root's link or the folder the resume
    returns: a link to a file beside the task is refused, and a link inside
    the task is made again as a relative link.
    """
    from benchflow.review.resume import _trusted_task
    from benchflow.task.staging import StagingRefused, copy_tree

    rollout, task = saved_trial
    listed = tmp_path / "listed"
    _link(listed / "physics", task)
    (task / "alias.json").symlink_to("rubric.json")
    (task.parent / "operator.txt").write_text("not a task file")
    (task / "notes").symlink_to(os.path.join("..", "operator.txt"))
    solver = json.loads((rollout / "solver.json").read_text())

    accepted = _trusted_task(rollout, solver, listed)

    assert accepted == task.resolve()
    for index, source in enumerate((accepted, listed / "physics")):
        with pytest.raises(StagingRefused, match="notes is a link to") as refused:
            copy_tree(source, tmp_path / f"copy-{index}", label="physics")
        assert "outside the task's folder" in str(refused.value)
    (task / "notes").unlink()
    copy_tree(listed / "physics", tmp_path / "kept", label="physics")
    assert os.readlink(tmp_path / "kept" / "alias.json") == "rubric.json"


@pytest.mark.asyncio
async def test_batch_retries_saved_solver_even_without_final_result(
    saved_trial, monkeypatch
):
    from benchflow.review import resume

    rollout, task = saved_trial
    retry = AsyncMock()
    monkeypatch.setattr(resume, "resume_review", retry)
    # A child review with an identical task name must not become a second trial.
    _write(rollout / "reviews" / "child" / "solver.json", {"task_name": "physics"})

    config = ReviewerConfig()
    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=config,
        task_names={"physics"},
    )
    retry.assert_awaited_once_with(
        rollout, tasks_root=task.parent, reviewer=config, budget_guard=None
    )


@pytest.mark.asyncio
async def test_batch_completed_result_wins_over_incomplete_retry(
    saved_trial, monkeypatch
):
    from benchflow.review import resume

    rollout, task = saved_trial
    _write(rollout / "result.json", _parent(rollout, _complete(passed=False)))
    orphan = rollout.parent / "physics__newer"
    _write(orphan / "solver.json", json.loads((rollout / "solver.json").read_text()))
    _write(orphan / "result.json", _parent(rollout, scoring_error("timeout")))
    retry = AsyncMock()
    monkeypatch.setattr(resume, "resume_review", retry)

    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=ReviewerConfig(),
        task_names={"physics"},
    )
    retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_resumes_a_reviewed_trial_whose_result_has_no_scoring(
    saved_trial, monkeypatch
):
    """Guards main's completeness rule for reviewed trials: a result.json with
    rewards but no scoring block holds the unreviewed verifier reward (earlier
    builds wrote one before the review ran), so the review is still pending."""
    from benchflow.review import resume

    rollout, task = saved_trial
    _write(rollout / "result.json", json.loads((rollout / "solver.json").read_text()))
    retry = AsyncMock()
    monkeypatch.setattr(resume, "resume_review", retry)

    config = ReviewerConfig()
    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=config,
        task_names={"physics"},
    )
    retry.assert_awaited_once_with(
        rollout, tasks_root=task.parent, reviewer=config, budget_guard=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verifier_error", "resumed"),
    [(None, False), ("verifier_wedge: no receipt", True)],
)
async def test_batch_keeps_a_recovery_verdict_without_review_as_final(
    saved_trial, monkeypatch, verifier_error, resumed
):
    """Guards verifier recovery without a review, which commits no scoring
    block: its result.json is final once it has rewards and no verifier error,
    and a verifier error that needs recovery is still resumed."""
    from benchflow.review import resume

    rollout, task = saved_trial
    config = json.loads((rollout / "config.json").read_text())
    del config["review"]
    config["verifier_recovery"] = {"eligible": True, "reason": None}
    _write(rollout / "config.json", config)
    source = json.loads((rollout / "solver.json").read_text())
    _write(
        rollout / "result.json",
        {
            **source,
            "rewards": None if verifier_error else source["rewards"],
            "verifier_error": verifier_error,
        },
    )
    retry = AsyncMock()
    monkeypatch.setattr(resume, "resume_review", retry)

    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=ReviewerConfig(),
        task_names={"physics"},
    )
    assert retry.await_count == int(resumed)


def test_pending_review_keeps_its_solver_when_resume_cannot_finish_it(saved_trial):
    """Guards #1134's "a dead review must not cost the solver result" now that
    a reviewed trial has no result.json until its scoring commits. A review
    that resume_pending_reviews could not finish leaves solver.json alone;
    evaluation resume keeps that solver, unscored, instead of replaying it."""
    from benchflow._utils.scoring import classify_score_outcome
    from benchflow.evaluation import Evaluation, RetryConfig
    from benchflow.rollout._verifier_recovery import PRESERVED_SOLVER

    rollout, task = saved_trial
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )

    kept = job._get_completed_tasks()["physics"]
    assert kept["rewards"] is None
    assert PRESERVED_SOLVER in kept["verifier_error"]
    assert not RetryConfig().should_retry_verifier_error(kept["verifier_error"])
    assert classify_score_outcome(kept) == "verifier_errored"
    assert kept["agent_result"] == {"total_tokens": 123}
    assert not (rollout / "result.json").exists()


INFRA_FAILURE = "Sandbox startup failed: daytona returned 503"


@pytest.mark.parametrize(
    ("error", "verifier_reward", "rerun"),
    [
        (INFRA_FAILURE, None, True),
        # The verifier judged the output: the scoring error is the verdict.
        (INFRA_FAILURE, 1.0, False),
        # A solver error that is not retryable infrastructure.
        ("the agent gave up", None, False),
    ],
)
def test_resume_reruns_a_rubric_solver_that_failed_on_infrastructure(
    saved_trial, error, verifier_reward, rerun
):
    """Guards #1059 for rubric tasks. A rubric trial commits a scoring block
    even when its sandbox or transport failed (a scoring error with no
    verifier reward), and any result with a scoring block was reused on
    resume, so a solver that failed on infrastructure never ran again."""
    from benchflow.evaluation import Evaluation, EvaluationConfig

    rollout, task = saved_trial
    result = _parent(
        rollout,
        scoring_error(
            "Deterministic verifier produced no reward",
            tests_pass=None if verifier_reward is None else True,
            verifier_reward=verifier_reward,
        ),
    )
    result.update(error=error, rewards=None)
    _write(rollout / "result.json", result)

    def completed(**config):
        return Evaluation(
            tasks_dir=task.parent,
            jobs_dir=rollout.parent.parent,
            job_name=rollout.parent.name,
            config=EvaluationConfig(**config),
        )._get_completed_tasks()

    assert ("physics" not in completed()) is rerun
    # A sequential-shared job reuses errored results, as before.
    assert "physics" in completed(job_mode="sequential-shared")


def test_a_pending_review_whose_solver_failed_on_infrastructure_reruns(saved_trial):
    """The solver.json-only case of the test above: nothing to review."""
    from benchflow.evaluation import Evaluation

    rollout, task = saved_trial
    solver = json.loads((rollout / "solver.json").read_text())
    _write(rollout / "solver.json", {**solver, "rewards": None, "error": INFRA_FAILURE})
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )
    assert "physics" not in job._get_completed_tasks()


@pytest.mark.asyncio
async def test_in_run_retry_reruns_a_rubric_solver_that_failed_on_infrastructure(
    job_factory,
):
    """The in-run half of #1059 for rubric tasks: a scoring block stopped the
    retry loop even when the solver failed on infrastructure unjudged."""
    from benchflow.models import RolloutResult

    job, tasks_dir = job_factory(n_tasks=1, max_retries=1)
    job._config.retry.min_wait_sec = 0.0
    job._prune_docker = lambda: None  # never touch a real Docker daemon
    unjudged = RolloutResult(
        task_name="task-0",
        rollout_name="task-0__a",
        error=INFRA_FAILURE,
        scoring=scoring_error("Deterministic verifier produced no reward"),
    )
    judged = RolloutResult(
        task_name="task-0",
        rollout_name="task-0__a",
        error=INFRA_FAILURE,
        scoring=scoring_error(
            "reviewer timed out", tests_pass=True, verifier_reward=1.0
        ),
    )
    ok = RolloutResult(task_name="task-0", rewards={"reward": 1.0})

    job._run_single_task = AsyncMock(side_effect=[unjudged, ok])
    assert await job._run_task(tasks_dir / "task-0") is ok
    job._run_single_task = AsyncMock(side_effect=[judged, ok])
    assert await job._run_task(tasks_dir / "task-0") is judged


@pytest.mark.asyncio
async def test_batch_isolates_a_failing_resumed_review(
    saved_trial, monkeypatch, caplog
):
    """Guards the #1134 fix on top of PR #1126: one trial's resume failure
    used to escape the TaskGroup as an ExceptionGroup, cancel every sibling
    review and abort the evaluation. It is now logged and skipped."""
    from benchflow.review import resume

    rollout, task = saved_trial
    other = rollout.parent / "chemistry__a"
    solver = json.loads((rollout / "solver.json").read_text())
    _write(other / "solver.json", {**solver, "task_name": "chemistry"})
    _write(other / "config.json", json.loads((rollout / "config.json").read_text()))
    reviewed: list[Path] = []

    async def retry(path, **_):
        if path == rollout:
            raise ReviewResumeError(
                "Task digest mismatch: restore the exact solver task"
            )
        await asyncio.sleep(0.05)  # still reviewing when the sibling fails
        reviewed.append(path)

    monkeypatch.setattr(resume, "resume_review", retry)
    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=ReviewerConfig(),
        task_names={"physics", "chemistry"},
    )
    assert reviewed == [other]
    assert rollout.name in caplog.text
    assert "Task digest mismatch" in caplog.text


def test_eval_score_cli_uses_shared_reviewer_options(saved_trial, monkeypatch):
    from benchflow.cli import rescore
    from benchflow.cli.main import app

    rollout, task = saved_trial
    retry = AsyncMock(return_value=_parent(rollout, _complete()))
    monkeypatch.setattr(rescore, "resume_review", retry)
    monkeypatch.setattr(rescore, "_apply_dotenv_to_process_env", lambda: None)
    monkeypatch.setattr(rescore, "write_job_results_jsonl", lambda _: None)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "score",
            str(rollout),
            "--tasks-root",
            str(task.parent),
            "--reviewer-model",
            "gpt-5.6-terra",
            "--reviewer-sandbox",
            "daytona",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "passed=True, reward=0.800" in result.output
    config = retry.call_args.kwargs["reviewer"]
    assert config.model == "gpt-5.6-terra"
    assert config.environment == "daytona"


def test_job_summary_refresh_preserves_solver_usage_and_deduplicates(saved_trial):
    from benchflow.cli.rescore import _refresh_job_artifacts

    rollout, _ = saved_trial
    _write(rollout / "result.json", _parent(rollout, _complete()))
    orphan = rollout.parent / "physics__orphan"
    _write(
        orphan / "result.json", _parent(rollout, scoring_error("reviewer timed out"))
    )
    original = {
        "job_name": "job",
        "total": 1,
        "passed": 0,
        "verifier_errored": 1,
        "n_input_tokens": 500,
        "model": "solver-model",
    }
    _write(rollout.parent / "summary.json", original)
    _write(rollout.parent.parent / "summary.json", original)

    _refresh_job_artifacts(rollout.parent)

    summary = json.loads((rollout.parent / "summary.json").read_text())
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["verifier_errored"] == 0
    assert summary["mean_reward"] == 0.8
    assert summary["n_input_tokens"] == 500
    assert summary["model"] == "solver-model"
    assert json.loads((rollout.parent.parent / "summary.json").read_text()) == summary


def test_failed_review_is_retained_for_review_only_resume(saved_trial):
    from benchflow.evaluation import Evaluation

    rollout, task = saved_trial
    result = _parent(rollout, scoring_error("verifier timed out after 900s"))
    result["verifier_error"] = "verifier timed out after 900s"
    _write(rollout / "result.json", result)
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )

    assert job._get_completed_tasks()["physics"]["scoring"]["status"] == "error"


def test_committed_scoring_result_writes_every_time_with_its_timezone(saved_trial):
    """A rubric review's commit rewrites result.json; every time in it carries a timezone, as the rollout's do."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(
        started_at="2026-11-19T08:20:31.160828+00:00",
        finished_at="2026-11-19T08:30:00.000000+00:00",
    )
    _write(rollout / "solver.json", source)

    committed = persistence.commit_scoring_result(rollout, _complete())

    written = json.loads((rollout / "result.json").read_text())
    times = {k: v for k, v in written.items() if k.endswith("_at") and v is not None}
    assert {"finished_at", "scoring_finished_at"} <= set(times)
    naive = {k: v for k, v in times.items() if datetime.fromisoformat(v).tzinfo is None}
    assert not naive, naive
    assert committed["finished_at"] == written["finished_at"]


def test_deferred_score_uses_execution_time_and_preserves_solver_timestamps(
    saved_trial,
):
    """Guards delayed rubric retries against elapsed-time inflation after PR #1126."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(
        started_at="2020-01-01 00:00:00",
        finished_at="2020-01-01 00:01:00",
        timing={"agent": 50.0, "verify": 10.0, "total": 60.0},
    )
    _write(rollout / "solver.json", source)
    original = (rollout / "solver.json").read_bytes()
    _write(
        rollout / "reviews/attempt-001/reviewer/result.json",
        {"timing": {"total": 12.4}},
    )

    committed = persistence.commit_scoring_result(rollout, _complete())

    assert committed["timing"]["review"] == 12.4
    assert committed["timing"]["total"] == 72.4
    assert committed["solver_started_at"] == source["started_at"]
    assert committed["solver_finished_at"] == source["finished_at"]
    assert committed["scoring_finished_at"] == committed["finished_at"]
    assert committed["finished_at"] > source["finished_at"]
    assert (rollout / "solver.json").read_bytes() == original


def test_force_commit_retains_prior_revision_exports(saved_trial):
    """Guards immutable scoring exports when revising a verdict after PR #1126."""
    rollout, _ = saved_trial
    first = _complete().model_copy(update={"revision": "scoring/first.json"})
    second = _complete(passed=False).model_copy(
        update={"revision": "scoring/second.json"}
    )
    _write(rollout / "scoring/first.json", {"scoring": first.to_dict()})
    _write(rollout / "scoring/second.json", {"scoring": second.to_dict()})

    persistence.commit_scoring_result(rollout, first)
    old_exports = (rollout / "scoring/first/results.jsonl").read_bytes()
    persistence.commit_scoring_result(rollout, second)

    assert (rollout / "scoring/first/results.jsonl").read_bytes() == old_exports
    first_parent = json.loads((rollout / "scoring/first/parent.json").read_text())
    assert first_parent["rewards"]["reward"] == 0.8
    second_parent = json.loads((rollout / "result.json").read_text())
    assert second_parent["rewards"]["reward"] == 0.0
    assert second_parent["scoring"]["revision"] == "scoring/second.json"


def test_trainer_export_error_does_not_publish_a_new_final_result(
    saved_trial, monkeypatch
):
    """Guards required export failures being swallowed after PR #1126."""
    rollout, _ = saved_trial
    original = _parent(rollout, _complete())
    _write(rollout / "result.json", original)
    monkeypatch.setattr(
        "benchflow.trajectories.export.write_rollout_verifiers_jsonl",
        Mock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        persistence.commit_scoring_result(rollout, _complete(passed=False))

    assert json.loads((rollout / "result.json").read_text()) == original


@pytest.mark.asyncio
async def test_concurrent_scoring_is_rejected_before_second_reviewer(
    saved_trial, monkeypatch
):
    rollout, task = saved_trial
    prepare = Mock(side_effect=AssertionError("must not prepare a second reviewer"))
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    with (
        persistence.scoring_lock(rollout),
        pytest.raises(ValueError, match="already running"),
    ):
        await resume_review(rollout, tasks_root=task.parent)
    prepare.assert_not_called()
    # Releasing an earlier attempt leaves no stale lock to clean up.
    with persistence.scoring_lock(rollout):
        pass


def test_scoring_commit_preserves_safe_diagnostics_from_solver(saved_trial):
    """Guards PR #1038 adaptation: scoring publication retains typed evidence."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(
        error="idle timeout",
        error_category="idle_timeout",
        verifier_error=None,
        idle_timeout_info={
            "idle_timeout_sec": 120,
            "idle_duration_sec": 121,
            "last_activity_at": "private timestamp",
            "future_field": "private future value",
        },
        unknown_diagnostic_info={"secret": "private unknown event"},
    )
    _write(rollout / "solver.json", source)
    persistence.commit_scoring_result(rollout, _complete())
    row = json.loads((rollout / "results.jsonl").read_text())
    block = row["info"]["diagnostics"]
    assert block["error_category"] == "idle_timeout"
    assert "verifier_error_category" not in block
    assert set(block["events"]) == {"idle_timeout_info"}
    assert block["events"]["idle_timeout_info"]["details"]["idle_duration_sec"] == 121
    assert "n_tool_calls" not in block["events"]["idle_timeout_info"]["details"]
    assert "private" not in json.dumps(block)
    assert json.loads((rollout / "solver.json").read_text()) == source


@pytest.mark.parametrize(
    "field,details",
    [
        (
            "transport_error_info",
            {"transport_diagnosis": [], "raw_message": "private secret"},
        ),
        ("api_error_info", {"subcategory": {}, "status_counts": {"inf": 2}}),
        ("verifier_timeout_info", {"elapsed_sec": 10**1000}),
        ("agent_timeout_info", {"pending_tool_call_ids": None}),
    ],
)
def test_scoring_commit_omits_malformed_saved_diagnostic_details(
    saved_trial, field, details
):
    """Guards PR #1038 adaptation: historical malformed details cannot break scoring."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(verifier_error=None)
    source[field] = details
    _write(rollout / "solver.json", source)
    result = persistence.commit_scoring_result(rollout, _complete())
    assert result["scoring"]["status"] == "complete"
    row = json.loads((rollout / "results.jsonl").read_text())
    block = row["info"].get("diagnostics", {})
    assert "private secret" not in json.dumps(block)
    event = block["events"][field]
    assert "elapsed_sec" not in event.get("details", {})
    assert "pending_tool_call_count" not in event.get("details", {})


def test_completed_verification_clears_category_but_retains_observed_event(saved_trial):
    """Guards PR #1038 adaptation: recovered status differs from historical evidence."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(
        rewards=None,
        verifier_error="verifier timed out",
        verifier_error_category="verifier_timeout",
        verifier_timeout_info={"timeout_budget_sec": 60, "elapsed_sec": 61},
    )
    _write(rollout / "solver.json", source)
    attempt = "verifier-recovery/abc123"
    _write(rollout / "verification.json", {"attempt": attempt})
    _write(
        rollout / attempt / "recovery.json",
        {
            "task_digest": source["task_digest"],
            "status": "complete",
            "rewards": {"reward": 1.0},
            "timing": {"verifier": 2},
        },
    )
    result = persistence.commit_scoring_result(rollout, _complete())
    assert result["verifier_error"] is None
    assert result["verifier_error_category"] is None
    block = json.loads((rollout / "results.jsonl").read_text())["info"]["diagnostics"]
    assert "verifier_error_category" not in block
    event = block["events"]["verifier_timeout_info"]
    assert event["category"] == "verifier_timeout"
    assert event["details"]["elapsed_sec"] == 61
    assert json.loads((rollout / "solver.json").read_text()) == source


@pytest.mark.parametrize(
    "category",
    [
        {"path": "/private/other"},
        ["private submission"],
        "private freeform category",
    ],
)
def test_scoring_publication_omits_invalid_saved_categories(saved_trial, category):
    """Guards PR #1038 review: category fields are fixed labels, not arbitrary data."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(
        verifier_error=None, error_category=category, verifier_error_category=category
    )
    _write(rollout / "solver.json", source)
    persistence.commit_scoring_result(rollout, _complete())
    row = json.loads((rollout / "results.jsonl").read_text())
    assert "diagnostics" not in row["info"]


def test_huge_programmatic_diagnostic_numbers_are_omitted(tmp_path):
    """Guards PR #1038 review: finite interchange bounds avoid JSON integer failure."""
    diagnostics = RolloutDiagnostics()
    diagnostics.set(
        IdleTimeoutDiagnostic(n_tool_calls=10**5000, idle_duration_sec=2**64)
    )
    diagnostics.set(ProviderApiErrorDiagnostic(status_counts={"429": 10**5000}))
    info = _structured_diagnostics_info(
        error_category="idle_timeout",
        verifier_error_category=None,
        diagnostics=diagnostics,
    )
    events = json.loads(_record_to_redacted_json_line(info))["diagnostics"]["events"]
    details = events["idle_timeout_info"]["details"]
    assert "n_tool_calls" not in details
    # Retained exactly, never float-rounded.
    assert details["idle_duration_sec"] == 2**64
    assert "status_counts" not in events["api_error_info"]["details"]


@pytest.mark.asyncio
async def test_an_expected_resume_refusal_logs_one_line_without_a_traceback(
    saved_trial, monkeypatch, caplog
):
    """Guards the #1134 isolation above: a task edited since the solver ran is
    an expected refusal (ReviewResumeError), so resuming the job logs one
    warning naming the reason instead of a full Python traceback."""
    from benchflow.review import resume

    rollout, task = saved_trial

    async def refuse(path, **_):
        raise ReviewResumeError("Task digest mismatch: restore the exact solver task")

    monkeypatch.setattr(resume, "resume_review", refuse)
    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=ReviewerConfig(),
        task_names={"physics"},
    )
    [record] = [r for r in caplog.records if rollout.name in r.getMessage()]
    assert "Task digest mismatch" in record.getMessage()
    assert record.exc_info is None


USAGE_LIMIT_FAILURE = (
    "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (environment): 7-day "
    "window, resets 2026-10-03 19:00 UTC (You've hit your weekly limit · resets "
    "Oct 3, 7pm (UTC))"
)


def test_resume_reruns_a_rubric_solver_that_hit_a_usage_limit(saved_trial):
    """dx/errors: a trial that ended on its login's usage limit is never retried
    within the run (the same login would hit it again), but resuming the job,
    on another login or after the reset, must run it again. A rubric trial
    commits an unjudged scoring block for it, which resume kept as final."""
    from benchflow.evaluation import Evaluation, RetryConfig

    rollout, task = saved_trial
    scoring = scoring_error(
        "Deterministic verifier produced no reward",
        tests_pass=None,
        verifier_reward=None,
    )
    result = _parent(rollout, scoring)
    result.update(error=USAGE_LIMIT_FAILURE, error_category="usage_limit", rewards=None)
    _write(rollout / "result.json", result)
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )
    assert "physics" not in job._get_completed_tasks()
    # Within the run it still does not rerun.
    retry = RetryConfig()
    assert not retry.reruns_unjudged_solver(
        scoring, USAGE_LIMIT_FAILURE, category="usage_limit"
    )
    assert retry.reruns_unjudged_solver(
        scoring, USAGE_LIMIT_FAILURE, category="usage_limit", on_resume=True
    )


def test_a_pending_review_whose_solver_hit_a_usage_limit_reruns(saved_trial):
    """The solver.json-only case of the test above: it was kept as a solver to
    finish with `bench eval score`, but there is nothing to review."""
    from benchflow.evaluation import Evaluation

    rollout, task = saved_trial
    solver = json.loads((rollout / "solver.json").read_text())
    _write(
        rollout / "solver.json",
        {
            **solver,
            "rewards": None,
            "error": USAGE_LIMIT_FAILURE,
            "error_category": "usage_limit",
        },
    )
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )
    assert "physics" not in job._get_completed_tasks()


# Follow-up to PR #141, 2026-11-13 (the readers of reruns_on_resume): a model
# refusal is never run again, as within the run (Evaluation._run_task).
REFUSAL = "ACP error -32603: Internal error"
MODEL_REFUSAL = {"by": "stop_reason", "stop_reason": "refusal", "call": None}


def test_resume_keeps_a_rubric_solver_whose_model_refused(saved_trial):
    """Follow-up to PR #141, 2026-11-13: a rubric trial whose model refused,
    with a scoring block that judged nothing. The run does not retry it (the
    refusal is the model's answer), but reruns_on_resume says yes to its ACP
    error, so a resume ran the solver again and re-rolled the refusal. It is
    kept."""
    from benchflow.evaluation import Evaluation, RetryConfig

    rollout, task = saved_trial
    scoring = scoring_error(
        "Deterministic verifier produced no reward",
        tests_pass=None,
        verifier_reward=None,
    )
    result = _parent(rollout, scoring)
    result.update(
        error=REFUSAL,
        error_category="acp_error",
        rewards=None,
        model_refusal=MODEL_REFUSAL,
    )
    _write(rollout / "result.json", result)
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )
    assert "physics" in job._get_completed_tasks()
    # What the retry policy alone says of its error.
    assert RetryConfig().reruns_unjudged_solver(
        scoring, REFUSAL, category="acp_error", on_resume=True
    )


def test_a_pending_review_whose_model_refused_is_kept(saved_trial):
    """Follow-up to PR #141, 2026-11-13: the solver.json-only case of the
    test above. The solver is kept for `bench eval score`, never run again."""
    from benchflow.evaluation import Evaluation

    rollout, task = saved_trial
    solver = json.loads((rollout / "solver.json").read_text())
    _write(
        rollout / "solver.json",
        {
            **solver,
            "rewards": None,
            "error": REFUSAL,
            "error_category": "acp_error",
            "model_refusal": MODEL_REFUSAL,
        },
    )
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )
    assert "physics" in job._get_completed_tasks()
