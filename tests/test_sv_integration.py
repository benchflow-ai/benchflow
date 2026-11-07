"""Guards py8/sv-all's reconciliation of 2f538971 with a844a8ad."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchflow import budget
from benchflow.rollout import _separate_verifier as sv
from tests.test_separate_verifier import (
    FakeVerifierSandbox,
    _agent_trial,
    _fake_rollout,
    _task_dir,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [True, False])
async def test_overlap_charges_live_and_cancelled_attempts_until_agent_release(
    tmp_path, monkeypatch, stopped
):
    """Guards the 2f538971 merge: live admission and receipts count overlap once."""
    now = [0.0]
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(sv.timing_spans, "_clock", lambda: now[0])
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))

    class Box(FakeVerifierSandbox):
        async def stop(self, **kwargs):
            now[0] += 2
            await super().stop(**kwargs)

    box = Box(tmp_path / "verifier-box")
    guard = budget.BudgetGuard(budget.Budget(max_sandbox_seconds=1000))
    guard.start("trial")
    with guard.track_trial("trial"):
        run = sv.VerifierSandboxRun(
            rollout, order=sv.OVERLAP, create_environment=lambda *a: box
        )
        run.begin_start()
        await run.start_task
        now[0] = 5
        assert guard.spent()["sandbox_seconds"] == 10
        run.agent_stopped(None if stopped else "delete failed")
        now[0] = 20
        assert guard.spent()["sandbox_seconds"] == (25 if stopped else 40)
        await run.discard(cancelled=True)
    guard.finish("trial")
    assert box.stopped == [True]
    assert not guard._verifier_running
    assert guard.spent()["sandbox_seconds"] == (27 if stopped else 44)
    assert run.record["sandbox_seconds"] == 22
    assert rollout._timing["verifier_sandbox_overlap"] == (5 if stopped else 22)


@pytest.mark.asyncio
async def test_repeated_cancel_while_discard_settles_prestart_still_deletes(
    tmp_path, monkeypatch
):
    """Guards the 2f538971 merge: discarded prestarts cannot escape teardown."""
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    started, release = asyncio.Event(), asyncio.Event()

    class Box(FakeVerifierSandbox):
        async def start(self, **kwargs):
            await super().start(**kwargs)
            started.set()
            await release.wait()

    box = Box(tmp_path / "verifier-box")
    run = sv.VerifierSandboxRun(
        rollout, order=sv.OVERLAP, create_environment=lambda *a: box
    )
    run.begin_start()
    await asyncio.wait_for(started.wait(), 5)
    discarding = asyncio.create_task(run.discard(cancelled=False))
    await asyncio.sleep(0)
    discarding.cancel()
    await asyncio.sleep(0)
    discarding.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(discarding, 5)
    assert run.start_task.done()
    assert box.stopped == [True]
    assert run.record["status"] == "interrupted"


@pytest.mark.parametrize("path", ["/opt/runtime/rt/bin/sh", "/opt/runtime", "/opt"])
@pytest.mark.parametrize(
    "fields", [{"verifier_sandbox": "separate"}, {"integrity": "strict"}]
)
def test_protected_runtime_is_refused_before_a_job_or_sandbox(
    tmp_path, monkeypatch, path, fields
):
    """Guards py8/sv-all's PT contract: transfers cannot replace the root runtime."""
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from tests.test_sv_switch import _no_sandbox, _task

    task = _task(tmp_path, "protected-runtime", artifacts=f'artifacts = ["{path}"]\n')
    started = _no_sandbox(monkeypatch)
    jobs = tmp_path / "jobs"
    job = Evaluation(
        tasks_dir=task,
        jobs_dir=jobs,
        config=EvaluationConfig(agent="oracle", environment="docker", **fields),
        preflight=False,
    )
    with pytest.raises(ValueError, match=r"verifier.sandbox.transfer.*opt/runtime"):
        asyncio.run(job.run())
    assert not started
    assert not jobs.exists()


@pytest.mark.asyncio
async def test_agent_stop_confirmation_requests_unmodified_docker_output(tmp_path):
    """Guards the 2f538971 merge against parsing a diagnostic Docker listing."""
    from benchflow.sandbox.protocol import ExecResult
    from tests.test_sv_overlap import _docker_agent

    agent = _docker_agent(tmp_path)

    async def listing(args, check=True, *, raw_output=False):
        # Diagnostic mode may merge stderr or redact the stdout listing.
        return ExecResult(0, "" if raw_output else "transport warning", "warning")

    agent._docker_cli = listing
    assert await sv.agent_sandbox_left_running(agent) is None
