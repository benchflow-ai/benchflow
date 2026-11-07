"""Guards FF-sw-tests-ci-confirm-5: 79c3b89e bypassed budgets in eval score."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from typer.testing import CliRunner

from benchflow import budget as mod
from benchflow import evaluation as ev
from benchflow.cli import rescore
from benchflow.review import automatic
from benchflow.review.resume import ReviewResumeError, resume_review
from benchflow.sandbox.providers import SANDBOX_PROVIDERS
from tests.test_review_resume import _complete, _write
from tests.test_review_resume import saved_trial as saved_trial


def _job_record(path, budget):
    _write(
        path.parent / "evaluation.json",
        {"schema_version": 1, "config": {"budget": budget}},
    )


def _receipt(path, seconds):
    _write(
        path / "budget-cancelled.json",
        {"version": 1, "timing": {"total": seconds, "verifier_sandbox_total": 0}},
    )


def _task_receipt(path, seconds):
    name = path.name.rsplit("__", 1)[0]
    target = (
        path.parent
        / ".budget-spend"
        / f"{hashlib.sha256(name.encode()).hexdigest()}.json"
    )
    _write(target, {"version": 1, "task_name": name, "sandbox_seconds": seconds})
    return target


@pytest.fixture
def scoring(saved_trial, monkeypatch):
    path, task = saved_trial
    source = json.loads((path / "solver.json").read_text())
    source["timing"] = {"total": 20, "verifier_sandbox_total": 10}
    _write(path / "solver.json", source)
    prepare = Mock(return_value=SimpleNamespace(config=None))
    finish = AsyncMock(return_value=_complete())
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(rescore, "_apply_dotenv_to_process_env", lambda: None)
    return path, task, finish


@pytest.mark.parametrize("backend", SANDBOX_PROVIDERS)
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("receipt", ["cancelled", "task", "other-task"])
def test_cli_score_refuses_exhausted_job(scoring, backend, force, receipt):
    """Every backend and --force obey the enclosing job, including sibling spend."""
    from benchflow.cli.main import app

    path, task, finish = scoring
    _job_record(path, {"max_sandbox_seconds": 60})
    if receipt == "cancelled":
        _receipt(path, 60)
    elif receipt == "task":
        _task_receipt(path, 60)
    else:
        other = path.parent / "other__a"
        _receipt(other, 30)
    args = [
        "eval",
        "score",
        str(path),
        "--tasks-root",
        str(task),
        "--reviewer-sandbox",
        backend,
    ]
    if force:
        args.append("--force")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "budget reached" in result.output
    finish.assert_not_awaited()
    assert not (path / "result.json").exists()
    assert not (path.parent / ev.JOB_LOCK).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cap", [{"max_sandbox_seconds": 30}, {"max_tokens": 123}, {"max_cost_usd": 1}]
)
async def test_public_score_api_obeys_all_spend_caps(scoring, cap):
    path, task, finish = scoring
    source = json.loads((path / "solver.json").read_text())
    source["agent_result"]["cost_usd"] = 1
    _write(path / "solver.json", source)
    _job_record(path, cap)
    with pytest.raises(ReviewResumeError, match="budget reached"):
        await resume_review(path, tasks_root=task)
    finish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["recovery", "review"])
@pytest.mark.parametrize(
    "end", ["complete", "error", "external-cancel", "budget-cancel"]
)
async def test_score_spend_is_live_and_persisted(scoring, monkeypatch, stage, end):
    """Both scoring phases retain teardown on every exit and survive repeated restarts."""
    from benchflow.rollout import _verifier_recovery as recovery

    path, task, finish = scoring
    _job_record(path, {"max_sandbox_seconds": 60, "max_rollouts": 1})
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    entered, release = asyncio.Event(), asyncio.Event()

    async def phase(*args, **kwargs):
        charge = mod.start_verifier_charge() if stage == "recovery" else None
        entered.set()
        try:
            await release.wait()
            if end == "error":
                raise RuntimeError("dummy scoring failure")
            return _complete()
        finally:
            clock[0] += 2
            if charge:
                charge.finish()

    if stage == "recovery":
        monkeypatch.setattr(recovery, "resume_verification", phase)
    else:
        finish.side_effect = phase
    running = asyncio.create_task(resume_review(path, tasks_root=task))
    await asyncio.wait_for(entered.wait(), 5)
    clock[0] += 40
    if end == "external-cancel":
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    elif end == "budget-cancel":
        with pytest.raises(ReviewResumeError, match="budget reached"):
            await asyncio.wait_for(running, 2)
    else:
        release.set()
        if end == "error":
            with pytest.raises(RuntimeError, match="dummy scoring failure"):
                await running
        else:
            await running
    expected = 30 + 42 * (2 if stage == "recovery" else 1)
    for _ in range(2):
        restored = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=60))
        rows, interrupted = ev._recorded_attempts(path.parent)
        restored.seed(rows, interrupted)
        restored.restore(path.parent)
        assert restored.spent()["sandbox_seconds"] == expected
        assert restored.spent()["rollouts"] == 1
        with pytest.raises(ReviewResumeError, match="budget reached"):
            await resume_review(path, tasks_root=task, force=True)
    assert not (path.parent / ev.JOB_LOCK).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", ["{", "[]", '{"config":[]}', '{"config":{"budget":{"max_usd":5}}}']
)
async def test_invalid_saved_budget_fails_closed(scoring, body):
    path, task, finish = scoring
    (path.parent / "evaluation.json").write_text(body)
    with pytest.raises(ValueError):
        await resume_review(path, tasks_root=task)
    finish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cancelled", "task"])
async def test_invalid_spend_fails_closed(scoring, source):
    path, task, finish = scoring
    _job_record(path, {"max_sandbox_seconds": 60})
    target = (
        path / "budget-cancelled.json"
        if source == "cancelled"
        else _task_receipt(path, 20)
    )
    target.write_text("{")
    with pytest.raises(ValueError):
        await resume_review(path, tasks_root=task)
    finish.assert_not_awaited()


@pytest.mark.asyncio
async def test_spend_without_job_policy_fails_closed(scoring):
    path, task, finish = scoring
    _task_receipt(path, 30)
    with pytest.raises(ReviewResumeError, match=r"without evaluation.json"):
        await resume_review(path, tasks_root=task)
    finish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_completed_verdict_at_cap_is_read_only(scoring, passed):
    from tests.test_review_resume import _parent

    path, task, finish = scoring
    _job_record(path, {"max_sandbox_seconds": 30})
    completed = _parent(path, _complete(passed=passed))
    _write(path / "result.json", completed)
    assert await resume_review(path, tasks_root=task) == completed
    with pytest.raises(ReviewResumeError, match="budget reached"):
        await resume_review(path, tasks_root=task, force=True)
    finish.assert_not_awaited()


@pytest.mark.asyncio
async def test_independent_batch_shares_one_job_budget(scoring, monkeypatch):
    """The public batch API restores once, so overlapping reviewers share admission."""
    import shutil

    from benchflow.review.options import ReviewerConfig
    from benchflow.review.resume import resume_pending_reviews

    path, task, finish = scoring
    other_task = task.parent / "math"
    shutil.copytree(task, other_task)
    other = path.parent / "math__a"
    shutil.copytree(path, other)
    source = json.loads((other / "solver.json").read_text())
    source.update(task_name="math", rollout_name=other.name)
    _write(other / "solver.json", source)
    _job_record(path, {"max_sandbox_seconds": 120})
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    entered, release = asyncio.Event(), asyncio.Event()

    async def review(*args):
        # Both enter before either completes, under the same job lock/guard.
        if finish.await_count == 2:
            entered.set()
        await release.wait()
        return _complete()

    finish.side_effect = review
    running = asyncio.create_task(
        resume_pending_reviews(
            path.parent,
            tasks_root=task.parent,
            reviewer=ReviewerConfig(),
            task_names={"physics", "math"},
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    clock[0] += 5
    release.set()
    await running
    assert (path / "result.json").is_file()
    assert (other / "result.json").is_file()
    restored = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=120))
    rows, interrupted = ev._recorded_attempts(path.parent)
    restored.seed(rows, interrupted)
    restored.restore(path.parent)
    assert restored.spent()["sandbox_seconds"] == 70
    assert restored.spent()["rollouts"] == 2


@pytest.mark.asyncio
async def test_score_respects_job_lock(scoring):
    """An independent scorer cannot race an evaluation or another scoring process."""
    import os
    import socket

    path, task, finish = scoring
    _job_record(path, {"max_sandbox_seconds": 60})
    _write(
        path.parent / ev.JOB_LOCK, {"pid": os.getpid(), "host": socket.gethostname()}
    )
    with pytest.raises(RuntimeError, match="already running"):
        await resume_review(path, tasks_root=task)
    finish.assert_not_awaited()
    assert (path.parent / ev.JOB_LOCK).exists()


@pytest.mark.asyncio
async def test_explicitly_raised_saved_cap_allows_scoring(scoring, monkeypatch):
    path, task, finish = scoring
    _receipt(path, 50)  # Plus the solver's 10 separate-verifier seconds.
    _job_record(path, {"max_sandbox_seconds": 60})
    with pytest.raises(ReviewResumeError, match="budget reached"):
        await resume_review(path, tasks_root=task)
    _job_record(path, {"max_sandbox_seconds": 120, "max_rollouts": 1})
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def review(*args):
        clock[0] += 5
        return _complete()

    finish.side_effect = review
    await resume_review(path, tasks_root=task)
    finish.assert_awaited_once()
    restored = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=120))
    rows, interrupted = ev._recorded_attempts(path.parent)
    restored.seed(rows, interrupted)
    restored.restore(path.parent)
    assert restored.spent()["sandbox_seconds"] == 65
    assert restored.spent()["rollouts"] == 1
