"""Unit tests for the eval live-progress dashboard (cli/_live_progress.py).

State math + the TTY/quiet-logging gates are tested directly; the Rich render is
exercised for "doesn't raise" rather than pixel-asserted.
"""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace

from rich.console import Console

from benchflow.cli._live_progress import (
    LiveEvalProgress,
    progress_enabled,
    quiet_root_logging,
)


def _result(reward, *, tokens=0, cost=None, src="unavailable"):
    return SimpleNamespace(
        rewards={"reward": reward} if reward is not None else None,
        total_tokens=tokens,
        cost_usd=cost,
        usage_source=src,
    )


def _dash() -> LiveEvalProgress:
    return LiveEvalProgress(
        Console(), label="skillsbench", agent="gemini", model="flash", sandbox="docker"
    )


def test_counts_classify_like_the_engine():
    d = _dash()
    d.on_plan(total=4, done=0, remaining=4)
    for name in ("a", "b", "c", "d"):
        d.on_task_start(name)
    d.on_result("a", _result(1.0, tokens=1000, cost=0.02, src="agent_native_acp"))
    d.on_result("b", _result(0.0))  # reward present but not 1 -> failed
    d.on_result("c", _result(None))  # no reward -> errored
    assert (d._passed, d._failed, d._errored) == (1, 1, 1)
    assert len(d._running) == 1  # "d" still running
    # render must not raise mid-run
    d.__rich__()


def test_resume_seeds_outcomes_so_counts_cover_whole_job():
    # On resume the counts + pass-rate must include the resumed tasks' outcomes,
    # not just this process's new tasks (Bugbot #726 medium).
    d = _dash()
    d.on_plan(total=10, done=6, remaining=4, resumed_outcomes=(5, 1, 0))
    d.on_task_start("x")
    d.on_result("x", _result(1.0))  # one new pass on top of the resumed 5/1/0
    assert (d._passed, d._failed, d._errored) == (6, 1, 0)
    assert d._resumed == 6
    assert d._completed == 1  # this-run only — drives the ETA rate, not the bar
    d.__rich__()


def test_classify_completed_outcomes_mirrors_engine():
    from benchflow.evaluation import _classify_completed_outcomes

    completed = {
        "a": {"rewards": {"reward": 1.0}},
        "b": {"rewards": {"reward": 0.0}},
        "c": {"rewards": None, "verifier_error": "boom"},
        "d": {},
    }
    assert _classify_completed_outcomes(completed) == (1, 1, 2)


def test_footer_no_telemetry_is_dash_not_zero():
    # A coverage-0 run must read as undecidable ("—"), never "$0.00 / 0 tokens".
    d = _dash()
    d.on_plan(total=1, done=0, remaining=1)
    d.on_result("a", _result(1.0, tokens=0, cost=None, src="unavailable"))
    assert d._covered == 0 and d._tokens == 0
    text = d.__rich__()  # builds the Group; tokens shown as "—"
    assert text is not None


def test_trusted_telemetry_accumulates():
    d = _dash()
    d.on_plan(total=2, done=0, remaining=2)
    d.on_result("a", _result(1.0, tokens=1500, cost=0.03, src="agent_native_acp"))
    d.on_result("b", _result(1.0, tokens=2500, cost=0.05, src="provider_response"))
    assert d._tokens == 4000
    assert round(d._cost, 2) == 0.08
    assert d._covered == 2


class _FakeSession:
    def progress_snapshot(self):
        return 38, "file_editor"

    def latest_usage_totals(self):
        return {"total_tokens": 1500}


def test_activity_cell_polls_live_session_counters():
    # Per-task activity in the running-now table (fresh-user dogfood
    # 2026-08-09): the heartbeat's counters must reach the dashboard, since
    # the Live mutes the logged heartbeat line itself.
    from benchflow._utils import live_activity
    from benchflow._utils.live_activity import ActivitySnapshot
    from benchflow.cli._live_progress import _activity_cell

    live_activity.register(
        "edit-pdf",
        SimpleNamespace(
            activity_snapshot=lambda: ActivitySnapshot(
                38, "file_editor", 1500, "connected"
            )
        ),
    )
    try:
        assert _activity_cell("edit-pdf") == "38 calls · 1.5k tok · last: file_editor"
    finally:
        live_activity.unregister("edit-pdf")
    # Unregistered tasks (pre-register, teardown races) degrade to the
    # fallback label — a live row's cell must never be blank, and never raise.
    assert _activity_cell("edit-pdf") == "starting…"


def test_rollout_activity_snapshot_reads_acp_session():
    # The client/session dig lives on Rollout (typed, owner-side) so a rename
    # of session counters breaks HERE instead of silently blanking the cell.
    from benchflow._utils.live_activity import ActivitySnapshot
    from benchflow.rollout import Rollout

    connected = SimpleNamespace(
        _acp_client=SimpleNamespace(session=_FakeSession()), _phase="connected"
    )
    assert Rollout.activity_snapshot(connected) == ActivitySnapshot(
        38, "file_editor", 1500, "connected"
    )
    # Pre-connect (and session-factory) rollouts have no client: counters are
    # None but the lifecycle phase still rides out so the cell can label it.
    assert Rollout.activity_snapshot(
        SimpleNamespace(_acp_client=None, _phase="setup")
    ) == ActivitySnapshot(None, None, None, "setup")


def test_dashboard_renders_activity_for_registered_running_task():
    # End-to-end through __rich__: a registered running task's activity must
    # appear in the rendered panel — reverting the table wiring fails this.
    import io

    from benchflow._utils import live_activity
    from benchflow._utils.live_activity import ActivitySnapshot

    live_activity.register(
        "edit-pdf",
        SimpleNamespace(
            activity_snapshot=lambda: ActivitySnapshot(
                38, "file_editor", None, "connected"
            )
        ),
    )
    try:
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("edit-pdf")
        out = Console(file=io.StringIO(), width=120)
        out.print(d.__rich__())
        text = out.file.getvalue()
        assert "38 calls" in text
        assert "file_editor" in text
    finally:
        live_activity.unregister("edit-pdf")


def test_activity_cell_shows_phase_label_before_session_exists():
    # Fresh-user dogfood follow-up: ~1.5min of sandbox create / agent install
    # (and the whole verifier) used to render a blank cell — indistinguishable
    # from a hang. Counter-less snapshots must surface the lifecycle phase.
    from benchflow._utils import live_activity
    from benchflow._utils.live_activity import ActivitySnapshot
    from benchflow.cli._live_progress import _activity_cell

    phase = "setup"
    live_activity.register(
        "warming-up",
        SimpleNamespace(
            activity_snapshot=lambda: ActivitySnapshot(None, None, None, phase)
        ),
    )
    try:
        assert _activity_cell("warming-up") == "creating sandbox…"
        phase = "started"
        assert _activity_cell("warming-up") == "installing agent…"
        phase = "verifying"
        assert _activity_cell("warming-up") == "verifying…"
        # Unknown phases (e.g. "branched") still never blank the cell.
        phase = "branched"
        assert _activity_cell("warming-up") == "starting…"
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("warming-up")
        d.__rich__()  # builds the running-now table row; must not raise
    finally:
        live_activity.unregister("warming-up")


def test_dashboard_renders_phase_label_for_counterless_task():
    # End-to-end through __rich__: the phase label must reach the rendered
    # panel, not just the cell helper — reverting the table wiring fails this.
    import io

    from benchflow._utils import live_activity
    from benchflow._utils.live_activity import ActivitySnapshot

    live_activity.register(
        "edit-pdf",
        SimpleNamespace(
            activity_snapshot=lambda: ActivitySnapshot(None, None, None, "started")
        ),
    )
    try:
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("edit-pdf")
        out = Console(file=io.StringIO(), width=120)
        out.print(d.__rich__())
        assert "installing agent…" in out.file.getvalue()
    finally:
        live_activity.unregister("edit-pdf")


def test_rollout_verify_marks_verifying_phase():
    # verify() must flip _phase to "verifying" at ENTRY (other transitions
    # mark completion): disconnect() has already reset the phase to
    # "installed" by then, and the activity cell keys off this value for the
    # minutes-long verifier stretch.
    import asyncio

    from benchflow.rollout import Rollout

    rollout = SimpleNamespace(
        _config=SimpleNamespace(primary_agent="x"),
        _trajectory=[{"type": "tool_call"}],  # non-empty: skip the scrape path
        _phase="installed",
    )

    async def run() -> None:
        # The full verify flow needs a sandbox; stop at the first await and
        # assert the phase already transitioned.
        with contextlib.suppress(AttributeError):
            await Rollout.verify(rollout)

    asyncio.run(run())
    assert rollout._phase == "verifying"


def test_progress_enabled_respects_tty_and_optout(monkeypatch):
    tty = SimpleNamespace(is_terminal=True)
    notty = SimpleNamespace(is_terminal=False)
    monkeypatch.delenv("BENCHFLOW_NO_PROGRESS", raising=False)
    assert progress_enabled(tty) is True
    assert progress_enabled(notty) is False
    monkeypatch.setenv("BENCHFLOW_NO_PROGRESS", "1")
    assert progress_enabled(tty) is False


def test_quiet_root_logging_buffers_then_restores():
    from benchflow.cli._live_progress import _WarningBuffer

    root = logging.getLogger()
    before = root.handlers[:]
    with quiet_root_logging():
        # INFO is dropped (would shred the Live), WARNING+ is buffered, not a NullHandler.
        assert all(isinstance(h, _WarningBuffer) for h in root.handlers)
    assert root.handlers == before


def test_quiet_root_logging_replays_warnings_not_info(monkeypatch):
    # B5 regression: batch-level reliability WARNING/ERROR must survive the Live
    # (be replayed after), while INFO chatter stays suppressed.
    import benchflow.cli._live_progress as lp

    printed: list[str] = []
    monkeypatch.setattr(
        lp.console, "print", lambda msg, *a, **k: printed.append(str(msg))
    )
    log = logging.getLogger("benchflow.test")
    with quiet_root_logging():
        log.info("per-task chatter that would shred the Live")
        log.warning(">20% verifier errors — results may be unreliable")
        log.error("circuit breaker tripped")
    blob = "\n".join(printed)
    assert "unreliable" in blob and "circuit breaker" in blob
    assert "per-task chatter" not in blob


def test_quiet_root_logging_restores_on_exception():
    root = logging.getLogger()
    before = root.handlers[:]
    try:
        with quiet_root_logging():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert root.handlers == before


def test_report_eval_result_surfaces_verifier_errors(monkeypatch):
    # B-1 regression: a verifier-error-only run is NOT "errors=0" — the displayed
    # count must agree with the red colour (which keys off total errors).
    import io

    from rich.console import Console

    import benchflow.cli._shared as shared

    rec = Console(file=io.StringIO(), width=200)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(
        SimpleNamespace(
            passed=0, total=3, errored=0, verifier_errored=3, score=0.0, job_name="j"
        )
    )
    out = rec.file.getvalue()
    assert "errors=0 verifier-errors=3" in out
    assert "Score: 0/3" in out


def _reported(result) -> str:
    """Render _report_eval_result through a captured Console, return the text."""
    import io

    from rich.console import Console

    import benchflow.cli._shared as shared

    rec = Console(file=io.StringIO(), width=200)
    original = shared.console
    shared.console = rec
    try:
        shared._report_eval_result(result)
    finally:
        shared.console = original
    return rec.file.getvalue()


def _failed_result(task_failures):
    return SimpleNamespace(
        passed=0,
        total=len(task_failures),
        errored=0,
        verifier_errored=0,
        score=0.0,
        job_name="j",
        task_failures=task_failures,
    )


def test_report_eval_result_prints_failure_reason_lines():
    # Dogfood follow-up: "✗ Score: 0/1" alone forces a dig into summary.json.
    # Each FAILED task gets one dim reason line — verifier_error first, else a
    # compact reward/metric breakdown (zero metrics first), else the reward.
    from benchflow.evaluation import TaskFailure

    out = _reported(
        _failed_result(
            [
                TaskFailure(
                    task_name="edit-pdf",
                    rewards={"reward": 0.0},
                    verifier_error="AssertionError:\n  output.pdf missing",
                ),
                TaskFailure(
                    task_name="plan-meeting",
                    rewards={
                        "reward": 0.3,
                        "decisions_found": 0.0,
                        "deadlines_found": 0.0,
                        "sections": 1.0,
                        "extra_metric": 1.0,
                    },
                    verifier_error=None,
                ),
                TaskFailure(
                    task_name="sum-csv", rewards={"reward": 0.0}, verifier_error=None
                ),
            ]
        )
    )
    # verifier_error wins and is collapsed to one line.
    assert "✗ edit-pdf: AssertionError: output.pdf missing" in out
    # Metric breakdown: zero/failed metrics first, capped at 3.
    assert (
        "✗ plan-meeting: reward 0.3 — decisions_found 0.0, deadlines_found 0.0, "
        "sections 1.0" in out
    )
    assert "extra_metric" not in out
    # No named metrics: just the reward.
    assert "✗ sum-csv: reward 0.0" in out


def test_report_eval_result_caps_failure_lines_at_five():
    from benchflow.evaluation import TaskFailure

    failures = [
        TaskFailure(task_name=f"task-{i}", rewards={"reward": 0.0}, verifier_error=None)
        for i in range(7)
    ]
    out = _reported(_failed_result(failures))
    assert out.count("✗ task-") == 5
    assert "… and 2 more" in out


def test_report_eval_result_truncates_failure_lines():
    from benchflow.evaluation import TaskFailure

    out = _reported(
        _failed_result(
            [
                TaskFailure(
                    task_name="edit-pdf",
                    rewards=None,
                    verifier_error="boom " * 60,
                )
            ]
        )
    )
    (line,) = [ln for ln in out.splitlines() if "✗ edit-pdf" in ln]
    assert len(line.rstrip()) <= 100
    assert line.rstrip().endswith("…")


def test_fire_progress_swallows_callback_errors():
    # The feature's core safety contract: a raising display hook must never
    # propagate out of the engine (a render bug can't abort a run).
    from benchflow.evaluation import Evaluation

    seen = []

    def boom(*args):
        seen.append(args)
        raise RuntimeError("display bug")

    Evaluation._fire_progress(boom, "task-x")  # must NOT raise
    Evaluation._fire_progress(None)  # None callback is a no-op
    assert seen == [("task-x",)]
