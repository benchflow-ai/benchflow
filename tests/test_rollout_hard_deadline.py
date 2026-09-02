"""Host-side hard deadline per rollout attempt.

Every phase inside a rollout has its own timeout, but an await stuck BELOW
that instrumentation (a Daytona PTY kill on a dead websocket, a wedged session
exec in the post-verify export path) used to freeze the whole job: one hung
bike-rebalance rollout wedged a 25-task eval for 11+ hours after its verifier
had already finished (2026-08-07). ``Rollout.run()`` now enforces a computed
hard deadline around the lifecycle (covering every caller — Evaluation,
Runtime, bf.run, continue_run, acceptance) and converts a trip into a normal
infra-retryable error result.

The enforcement deliberately avoids a bare ``asyncio.wait_for``: cancelling
the lifecycle runs its ``finally: cleanup()``, and when the teardown is
itself the wedged path, ``wait_for`` would block past its own deadline
waiting for that cleanup — so cancellation gets its own bounded grace before
the task is abandoned outright.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow._utils.scoring import INFRA_ERROR
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.rollout import Rollout, RolloutConfig, _deadline
from benchflow.rollout._deadline import hard_deadline_sec


def _write_task(task_dir: Path) -> None:
    # A minimal *valid* legacy task (task.toml + instruction.md): the deadline
    # tests must exercise the computed-budget path, not the unreadable-task
    # fallback.
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do the task.\n")


class TestDeadlineComputation:
    def test_env_disables(self, tmp_path, monkeypatch):
        _write_task(tmp_path / "t")
        for raw in ("off", "none", "0", "0.0", "-5"):
            monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raw)
            assert hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t")) is None

    def test_env_numeric_overrides(self, tmp_path, monkeypatch):
        _write_task(tmp_path / "t")
        monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "123.5")
        assert hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t")) == 123.5

    def test_computed_covers_all_phase_budgets(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        deadline = hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t"))
        # agent 60 + verifier 60 + build/install defaults + fixed margin: the
        # backstop must strictly dominate the sum of the declared phase budgets.
        assert deadline is not None
        assert deadline > 60 + 60 + 1800
        # ... and stay below the unreadable-task fallback: this proves the
        # computed path ran (a fallback value would also satisfy the bounds
        # above, silently masking a budget-read regression).
        assert deadline < _deadline._FALLBACK_SEC

    def test_caller_timeout_override_dominates_task_budget(self, tmp_path, monkeypatch):
        """RolloutConfig.timeout (Runtime's wall-clock seam, #378) is the
        enforced agent budget — the backstop must be derived from it, not the
        smaller task default, or it would fire during a legitimate long run."""
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        base = hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t"))
        raised = hard_deadline_sec(
            RolloutConfig(task_path=tmp_path / "t", timeout=7200)
        )
        assert base is not None and raised is not None
        assert raised >= base + 7200 - 60

    def test_user_loop_rounds_dominate(self, tmp_path, monkeypatch):
        """A user-loop run legitimately spends max_user_rounds x (agent +
        soft-verify); the backstop must cover it, not fire mid-loop."""
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        cfg = RolloutConfig(task_path=tmp_path / "t", max_user_rounds=10)
        cfg.user = SimpleNamespace()  # any non-None user engages the loop
        deadline = hard_deadline_sec(cfg)
        assert deadline is not None
        assert deadline >= 10 * (60 + 60) + 1800

    def test_unreadable_task_falls_back_conservative(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        deadline = hard_deadline_sec(RolloutConfig(task_path=tmp_path / "missing"))
        assert deadline is not None
        assert deadline >= 3600


def _wedged_rollout(task_dir: Path, *, cleanup_wedged: bool = False) -> Rollout:
    """A real Rollout whose lifecycle hangs; optionally its cancellation
    cleanup hangs too (the re-wedge case a bare wait_for cannot break)."""
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=task_dir)

    async def _lifecycle():
        try:
            await asyncio.sleep(3600)  # wedged transport await
        finally:
            if cleanup_wedged:
                # Mirrors run()'s `finally: await self.cleanup()` hanging on a
                # dead connection: swallow the cancel and keep blocking.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(3600)

    rollout._run_lifecycle = _lifecycle
    return rollout


@pytest.mark.asyncio
async def test_wedged_lifecycle_becomes_infra_error(tmp_path, monkeypatch):
    """A lifecycle that never returns must yield an error result, not a hang,
    and the standard retry policy must treat it as retryable."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")

    rollout = _wedged_rollout(task_dir)
    result = await asyncio.wait_for(rollout.run(), timeout=15)

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR
    assert RetryConfig().should_retry(result.error, category=result.error_category)


@pytest.mark.asyncio
async def test_wedged_cleanup_cannot_re_wedge(tmp_path, monkeypatch):
    """Even when the cancellation-triggered cleanup also hangs (the observed
    incident shape: teardown wedged on a dead websocket), the deadline must
    still surface a result — a bare wait_for would block here forever."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.5)

    rollout = _wedged_rollout(task_dir, cleanup_wedged=True)
    result = await asyncio.wait_for(rollout.run(), timeout=15)

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR


@pytest.mark.asyncio
async def test_caller_cancellation_is_bounded(tmp_path, monkeypatch):
    """Cancelling run() from outside (job shutdown) must forward the cancel to
    the lifecycle and not wait unbounded for a wedged teardown."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "60")
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.5)

    rollout = _wedged_rollout(task_dir, cleanup_wedged=True)
    run_task = asyncio.create_task(rollout.run())
    await asyncio.sleep(0.05)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=15)


@pytest.mark.asyncio
async def test_evaluation_path_surfaces_infra_error(tmp_path, monkeypatch):
    """End-to-end through Evaluation._run_single_task: the deadline inside
    Rollout.run() protects the eval job without any caller-side wrap."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")

    job = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="wedge-run",
    )

    with patch(
        "benchflow.rollout.Rollout.create",
        AsyncMock(return_value=_wedged_rollout(task_dir)),
    ):
        result = await asyncio.wait_for(
            job._run_single_task(task_dir, job._config), timeout=15
        )

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR


class _StopTransport:
    def __init__(
        self,
        events: list[str],
        *,
        graceful: bool,
        forced: bool,
        stopped: bool,
        liveness_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.termination_status = SimpleNamespace(
            graceful_termination=graceful,
            force_kill_required=forced,
            process_tree_stopped=stopped,
        )
        self._stopped = stopped
        self._liveness_error = liveness_error

    async def process_tree_stopped(self) -> bool:
        self.events.append("liveness")
        if self._liveness_error is not None:
            raise self._liveness_error
        return self._stopped


class _StopClient:
    session = None

    def __init__(
        self,
        transport: _StopTransport,
        events: list[str],
        *,
        cancel_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._transport = transport
        self._events = events
        self._cancel_error = cancel_error
        self._close_error = close_error

    async def cancel(self) -> None:
        self._events.append("cancel")
        if self._cancel_error is not None:
            raise self._cancel_error

    async def close(self) -> None:
        self._events.append("close")
        if self._close_error is not None:
            raise self._close_error


def _rollout_at_stop_boundary(client: _StopClient) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._acp_client = client
    rollout._session = None
    rollout._session_adapter = None
    rollout._is_session_factory = False
    rollout._active_role = None
    rollout._session_tool_count = 0
    rollout._session_traj_count = 0
    rollout._phase = "executed"
    rollout._termination_receipt = None
    rollout._acp_session_observation = None
    return rollout


@pytest.mark.asyncio
async def test_stop_agent_orders_evidence_and_is_idempotent() -> None:
    events: list[str] = []
    transport = _StopTransport(
        events,
        graceful=False,
        forced=True,
        stopped=True,
    )
    rollout = _rollout_at_stop_boundary(_StopClient(transport, events))

    first = await rollout.stop_agent(cancel_requested=True)
    second = await rollout.stop_agent(cancel_requested=True)

    assert events == ["cancel", "close", "liveness"]
    assert second is first
    assert first.cancel_requested is True
    assert first.cancel_acknowledged is True
    assert first.session_closed is True
    assert first.graceful_termination is False
    assert first.force_kill_required is True
    assert first.process_tree_stopped is True
    assert first.capture_safe is True


@pytest.mark.asyncio
async def test_stop_agent_preserves_independent_failure_evidence() -> None:
    events: list[str] = []
    transport = _StopTransport(
        events,
        graceful=False,
        forced=False,
        stopped=True,
        liveness_error=RuntimeError("cannot prove group death"),
    )
    rollout = _rollout_at_stop_boundary(
        _StopClient(
            transport,
            events,
            cancel_error=RuntimeError("cancel failed"),
            close_error=RuntimeError("close failed"),
        )
    )

    receipt = await rollout.stop_agent(cancel_requested=True)

    assert events == ["cancel", "close", "liveness"]
    assert receipt.cancel_acknowledged is False
    assert receipt.session_closed is False
    assert receipt.graceful_termination is False
    assert receipt.force_kill_required is False
    assert receipt.process_tree_stopped is False
    assert receipt.capture_safe is False


class _StagedStopTransport(_StopTransport):
    def __init__(self, events: list[str]) -> None:
        super().__init__(
            events,
            graceful=False,
            forced=False,
            stopped=False,
        )
        self.session_closed = False

    async def terminate_process_tree(self):
        self.events.append("terminate")
        await asyncio.sleep(0.02)
        self.termination_status = SimpleNamespace(
            graceful_termination=False,
            force_kill_required=True,
            process_tree_stopped=True,
        )
        self._stopped = True
        return self.termination_status


class _StagedStopClient(_StopClient):
    async def close_session(self) -> bool:
        self._events.append("close_session")
        self._transport.session_closed = True
        return True

    async def close(self) -> None:
        self._events.append("legacy_close")
        raise AssertionError("stop_agent must use independently bounded stages")


@pytest.mark.asyncio
async def test_stop_agent_does_not_preempt_transport_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    transport = _StagedStopTransport(events)
    client = _StagedStopClient(transport, events)
    rollout = _rollout_at_stop_boundary(client)
    monkeypatch.setattr(
        "benchflow.rollout._ACP_STOP_CLOSE_TIMEOUT_SEC",
        0.001,
        raising=False,
    )

    receipt = await rollout.stop_agent(cancel_requested=True)

    assert events == ["cancel", "close_session", "terminate", "liveness"]
    assert receipt.session_closed is True
    assert receipt.force_kill_required is True
    assert receipt.process_tree_stopped is True
    assert receipt.capture_safe is True


@pytest.mark.asyncio
async def test_stop_agent_preserves_session_close_evidence_when_termination_fails() -> (
    None
):
    events: list[str] = []
    transport = _StagedStopTransport(events)

    async def fail_termination():
        events.append("terminate")
        raise RuntimeError("termination control failed")

    transport.terminate_process_tree = fail_termination
    rollout = _rollout_at_stop_boundary(_StagedStopClient(transport, events))

    receipt = await rollout.stop_agent(cancel_requested=True)

    assert events == ["cancel", "close_session", "terminate", "liveness"]
    assert receipt.session_closed is True
    assert receipt.process_tree_stopped is False
    assert receipt.capture_safe is False


class _BlockingStopTransport(_StopTransport):
    def __init__(
        self,
        events: list[str],
        *,
        started: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        super().__init__(
            events,
            graceful=False,
            forced=False,
            stopped=False,
        )
        self.session_closed = False
        self._started = started
        self._release = release
        self.termination_calls = 0

    async def terminate_process_tree(self):
        self.events.append("terminate")
        self.termination_calls += 1
        self._started.set()
        await self._release.wait()
        self.termination_status = SimpleNamespace(
            graceful_termination=False,
            force_kill_required=True,
            process_tree_stopped=True,
        )
        self._stopped = True
        return self.termination_status


@pytest.mark.asyncio
async def test_concurrent_stop_agent_callers_share_one_teardown_and_receipt() -> None:
    events: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    transport = _BlockingStopTransport(
        events,
        started=started,
        release=release,
    )
    rollout = _rollout_at_stop_boundary(_StagedStopClient(transport, events))

    first_call = asyncio.create_task(rollout.stop_agent(cancel_requested=True))
    await started.wait()
    second_call = asyncio.create_task(rollout.stop_agent(cancel_requested=True))
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_call, second_call)

    assert first is second
    assert transport.termination_calls == 1
    assert events.count("cancel") == 1
    assert events.count("close_session") == 1
    assert events.count("liveness") == 1


@pytest.mark.asyncio
async def test_stop_agent_caller_cancellation_waits_for_shared_safe_teardown() -> None:
    events: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    transport = _BlockingStopTransport(
        events,
        started=started,
        release=release,
    )
    rollout = _rollout_at_stop_boundary(_StagedStopClient(transport, events))

    caller = asyncio.create_task(rollout.stop_agent(cancel_requested=True))
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    cancellation_propagated_before_teardown = caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    receipt = await rollout.stop_agent(cancel_requested=True)

    assert cancellation_propagated_before_teardown is False
    assert transport.termination_calls == 1
    assert receipt.process_tree_stopped is True
    assert receipt.capture_safe is True
