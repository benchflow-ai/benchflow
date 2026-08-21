"""Unit tests for the eval live-progress dashboard (cli/_live_progress.py).

State math + the TTY/quiet-logging gates are tested directly; the Rich render is
exercised for "doesn't raise" rather than pixel-asserted.
"""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace

import pytest
from rich.console import Console

from benchflow._utils import live_activity
from benchflow._utils.live_activity import ActivitySnapshot, SessionCounters
from benchflow.cli._live_progress import (
    LiveEvalProgress,
    _activity_cell,
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


@contextlib.contextmanager
def _registered_rollouts(rollouts: dict):
    """Register fake rollouts and guarantee unregistration.

    Values are the ActivitySnapshot to serve — or an Exception instance to
    raise from ``activity_snapshot()`` (a teardown-racing rollout).
    """

    def _fake(snap):
        if isinstance(snap, Exception):

            def _boom():
                raise snap

            return SimpleNamespace(activity_snapshot=_boom)
        return SimpleNamespace(activity_snapshot=lambda: snap)

    for name, snap in rollouts.items():
        live_activity.register(name, _fake(snap))
    try:
        yield
    finally:
        for name in rollouts:
            live_activity.unregister(name)


def _counters_snap(tokens, *, calls=3, last="file_editor", distinct=2):
    return ActivitySnapshot("connected", SessionCounters(calls, last, tokens, distinct))


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


def _rendered(d: LiveEvalProgress) -> str:
    import io

    out = Console(file=io.StringIO(), width=200)
    out.print(d.__rich__())
    return out.file.getvalue()


def test_footer_sums_live_running_usage_with_completed():
    # Mid-run spend visibility (51-min single-task dogfood: footer read
    # "— tokens · $—" for the whole agent phase): the footer must sum the
    # completed tasks' trusted telemetry PLUS every running session's live
    # usage. Cost stays completed-only — $ exists only in the scoring-time
    # LiteLLM gateway import.
    with _registered_rollouts(
        {"task-a": _counters_snap(1_000_000), "task-b": _counters_snap(500_000)}
    ):
        d = _dash()
        d.on_plan(total=3, done=0, remaining=3)
        for name in ("task-a", "task-b", "task-c"):
            d.on_task_start(name)
        d.on_result(
            "task-c", _result(1.0, tokens=2_500_000, cost=0.02, src="provider_response")
        )
        text = _rendered(d)
        assert "4.00M tokens" in text
        assert "$0.02" in text


def test_footer_shows_live_usage_before_any_completion():
    # The single-task common case: no task has finished, but the running
    # session has per-completed-prompt usage — real tokens render, and $
    # stays "—" (no live price signal exists).
    with _registered_rollouts({"task-a": _counters_snap(750_000)}):
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("task-a")
        text = _rendered(d)
        assert "750.0k tokens" in text
        assert "$—" in text
        assert "— tokens" not in text


def test_footer_degrades_to_dash_without_any_usage_signal():
    # No completed telemetry and no live counters (counter-less snapshot, a
    # teardown-racing rollout whose snapshot raises, an unregistered task):
    # the footer keeps the "—" contract — a coverage-0 run reads broken, not
    # free — and never raises.
    with _registered_rollouts(
        {
            "warming": ActivitySnapshot("setup", None),
            "racing": RuntimeError("teardown race"),
        }
    ):
        d = _dash()
        d.on_plan(total=3, done=0, remaining=3)
        for name in ("warming", "racing", "unregistered"):
            d.on_task_start(name)
        text = _rendered(d)
        assert "— tokens" in text
        assert "$—" in text


def test_footer_live_usage_ignores_sessions_without_usage():
    # A running session before its first completed prompt reports
    # total_tokens=None — it must contribute 0, not poison the sum.
    with _registered_rollouts(
        {"task-a": _counters_snap(None), "task-b": _counters_snap(250_000)}
    ):
        d = _dash()
        d.on_plan(total=2, done=0, remaining=2)
        d.on_task_start("task-a")
        d.on_task_start("task-b")
        assert "250.0k tokens" in _rendered(d)


class _FakeSession:
    distinct_tool_titles = 4

    def progress_snapshot(self):
        return 38, "file_editor"

    def latest_usage_totals(self):
        return {"total_tokens": 1500}


def test_activity_cell_formats_counters_and_registry_miss():
    # Pure formatter over one polled snapshot — __rich__ polls the registry
    # once per running task per frame and shares snapshots between the cells
    # and the footer sum. distinct_tools=None (unknown, e.g. an old-style
    # producer) must degrade to the full last:-suffixed cell, never mislabel.
    snap = ActivitySnapshot("connected", SessionCounters(38, "file_editor", 1500))
    assert _activity_cell(snap) == "38 calls · 1.5k tok · last: file_editor"
    # Registry misses (pre-register, teardown races) hand the formatter None —
    # a live row's cell must never be blank, and never raise.
    assert _activity_cell(None) == "starting…"


@pytest.mark.parametrize(
    ("counters", "expected"),
    [
        # Single-tool agents (prime-agent funnels everything through one
        # IPython tool): "last: IPython cell" is a constant across all N
        # calls — once tokens are available they carry the information.
        (
            SessionCounters(38, "IPython cell", 412_000, 1),
            "38 calls · 412.0k tok",
        ),
        # Varied tool names: last: still carries information.
        (
            SessionCounters(38, "file_editor", 412_000, 5),
            "38 calls · 412.0k tok · last: file_editor",
        ),
        # Constant tool but no tokens yet (first prompt still running): the
        # name, shown once, is still the only signal — it stays.
        (
            SessionCounters(5, "IPython cell", None, 1),
            "5 calls · last: IPython cell",
        ),
        # A lone first call is trivially "constant" but the name is still
        # news; the drop only pays off once repetition makes it redundant.
        (
            SessionCounters(1, "IPython cell", 90_000, 1),
            "1 calls · 90.0k tok · last: IPython cell",
        ),
    ],
)
def test_activity_cell_constant_vs_varied_tools(counters, expected):
    assert _activity_cell(ActivitySnapshot("connected", counters)) == expected


def test_rollout_activity_snapshot_reads_acp_session():
    # The client/session dig lives on Rollout (typed, owner-side) so a rename
    # of session counters breaks HERE instead of silently blanking the cell.
    from benchflow._utils.live_activity import ActivitySnapshot, SessionCounters
    from benchflow.rollout import Rollout

    connected = SimpleNamespace(
        _acp_client=SimpleNamespace(session=_FakeSession()),
        _phase="connected",
        _usage_runtime=None,
    )
    assert Rollout.activity_snapshot(connected) == ActivitySnapshot(
        "connected", SessionCounters(38, "file_editor", 1500, 4)
    )
    # Pre-connect (and session-factory) rollouts have no client: counters are
    # None but the lifecycle phase still rides out so the cell can label it.
    assert Rollout.activity_snapshot(
        SimpleNamespace(_acp_client=None, _phase="setup")
    ) == ActivitySnapshot("setup", None)


class _NoUsageSession(_FakeSession):
    """An ACP session mid-first-prompt: counters exist, usage doesn't yet."""

    def latest_usage_totals(self):
        return None


def _gateway_server(tokens):
    if isinstance(tokens, Exception):

        def _boom():
            raise tokens

        return SimpleNamespace(live_usage_tokens=_boom)
    return SimpleNamespace(live_usage_tokens=lambda: tokens)


def _rollout_ns(session, gateway):
    return SimpleNamespace(
        _acp_client=SimpleNamespace(session=session),
        _phase="connected",
        _usage_runtime=SimpleNamespace(server=_gateway_server(gateway)),
    )


@pytest.mark.parametrize(
    ("session", "gateway", "expected_tokens"),
    [
        # The round-7 failure case: a single-prompt rollout has NO ACP usage
        # snapshot until the prompt completes — the gateway live capture is
        # the only mid-run signal and must carry the counters.
        (_NoUsageSession(), 412_000, 412_000),
        # Both signals: max() — whichever cumulative counter leads wins, so
        # the footer never steps down mid-run (documented at
        # Rollout.activity_snapshot).
        (_FakeSession(), 900, 1500),
        (_FakeSession(), 2_000, 2_000),
        # Gateway signal dead (no usage-bearing record yet / reader raising /
        # garbage): degrade to exactly #963's ACP-only behavior.
        (_FakeSession(), None, 1500),
        (_FakeSession(), RuntimeError("teardown race"), 1500),
        (_FakeSession(), "not-an-int", 1500),
        (_NoUsageSession(), None, None),
        (_NoUsageSession(), RuntimeError("teardown race"), None),
    ],
)
def test_rollout_activity_snapshot_reconciles_acp_and_gateway_usage(
    session, gateway, expected_tokens
):
    from benchflow.rollout import Rollout

    snap = Rollout.activity_snapshot(_rollout_ns(session, gateway))
    assert snap.counters is not None
    assert snap.counters.total_tokens == expected_tokens


def test_rollout_activity_snapshot_without_usage_runtime_matches_963():
    # A rollout whose proxy never started (oracle, native-subscription auth)
    # has no _usage_runtime server — the ACP-only path must be untouched.
    from benchflow.rollout import Rollout

    ns = SimpleNamespace(
        _acp_client=SimpleNamespace(session=_NoUsageSession()),
        _phase="connected",
        _usage_runtime=None,
    )
    snap = Rollout.activity_snapshot(ns)
    assert snap.counters is not None
    assert snap.counters.total_tokens is None


def test_footer_and_cell_render_gateway_live_tokens_mid_prompt():
    # End-to-end through __rich__ with the REAL Rollout.activity_snapshot dig:
    # a single-prompt run mid-prompt (no ACP usage) whose gateway capture has
    # tokens must light BOTH the footer sum and the activity cell — the
    # e2e acceptance criterion, unit-shaped.
    from benchflow.rollout import Rollout

    session = _NoUsageSession()
    session.distinct_tool_titles = 1  # single-tool agent (prime-agent shape)
    ns = _rollout_ns(session, 412_000)
    rollout = SimpleNamespace(activity_snapshot=lambda: Rollout.activity_snapshot(ns))
    live_activity.register("dialogue-parser", rollout)
    try:
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("dialogue-parser")
        text = _rendered(d)
    finally:
        live_activity.unregister("dialogue-parser")
    assert "412.0k tokens" in text  # footer live sum
    assert "38 calls · 412.0k tok" in text  # constant-tool activity cell
    assert "— tokens" not in text


def test_dashboard_renders_activity_for_registered_running_task():
    # End-to-end through __rich__: a registered running task's activity must
    # appear in the rendered panel — reverting the table wiring fails this.
    with _registered_rollouts(
        {"edit-pdf": _counters_snap(None, calls=38, last="file_editor")}
    ):
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("edit-pdf")
        text = _rendered(d)
        assert "38 calls" in text
        assert "file_editor" in text


def test_activity_cell_shows_phase_label_before_session_exists():
    # Fresh-user dogfood follow-up: ~1.5min of sandbox create / agent install
    # (and the whole verifier) used to render a blank cell — indistinguishable
    # from a hang. Counter-less snapshots must surface the lifecycle phase.
    assert _activity_cell(ActivitySnapshot("setup", None)) == "creating sandbox…"
    assert _activity_cell(ActivitySnapshot("started", None)) == "installing agent…"
    assert _activity_cell(ActivitySnapshot("verifying", None)) == "verifying…"
    # Unknown phases (e.g. "branched") still never blank the cell.
    assert _activity_cell(ActivitySnapshot("branched", None)) == "starting…"


def test_dashboard_renders_phase_label_for_counterless_task():
    # End-to-end through __rich__: the phase label must reach the rendered
    # panel, not just the cell helper — reverting the table wiring fails this.
    with _registered_rollouts({"edit-pdf": ActivitySnapshot("started", None)}):
        d = _dash()
        d.on_plan(total=1, done=0, remaining=1)
        d.on_task_start("edit-pdf")
        assert "installing agent…" in _rendered(d)


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


def test_phase_labels_never_walk_backwards_through_a_run():
    # Fresh-user dogfood follow-up: the row showed "verifying…" and then
    # "running agent…" again for the last ~90s of a 26-minute run. The label is
    # a progress indicator — an earlier stage reappearing reads as the run
    # having restarted, so the label sequence a rollout can produce must be
    # monotonic.
    from benchflow.cli._live_progress import _PHASE_LABELS

    order = [
        "creating sandbox…",
        "installing agent…",
        "running agent…",
        "verifying…",
        "cleaning up…",
    ]
    lifecycle = [
        "created",
        "setup",
        "started",
        "installed",
        "connected",
        "executed",
        "verifying",
        "verified",
        "cleaned",
    ]
    ranks = [order.index(_PHASE_LABELS[phase]) for phase in lifecycle]
    assert ranks == sorted(ranks)
    # The specific inversion that shipped: verify() marks "verifying" at entry,
    # so "executed" only ever renders inside disconnect() — agent teardown, not
    # verification.
    assert _PHASE_LABELS["executed"] == "running agent…"


def test_rollout_disconnect_does_not_rewind_a_terminal_phase():
    # cleanup() calls disconnect() *after* verify(), and the unguarded rewind
    # to "installed" relabelled the whole teardown stretch "running agent…"
    # (and transiently blanked Rollout.result, which is gated on the same
    # phases). Between scenes the rewind is still correct: another agent turn
    # follows, and connect_as() re-marks "connected".
    import asyncio

    from benchflow.rollout import Rollout

    def _rollout(phase: str) -> SimpleNamespace:
        return SimpleNamespace(
            _phase=phase,
            _is_session_factory=False,
            _capture_partial_acp_trajectory=lambda: None,
            _acp_client=None,
            _session=None,
            _session_adapter=None,
            _agent_launch="",
            _env=None,
            _active_role=None,
            _session_tool_count=0,
            _session_traj_count=0,
        )

    mid_run = _rollout("executed")
    asyncio.run(Rollout.disconnect(mid_run))
    assert mid_run._phase == "installed"

    for terminal in ("verifying", "verified", "cleaned"):
        done = _rollout(terminal)
        asyncio.run(Rollout.disconnect(done))
        assert done._phase == terminal


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


def _reported(result, job_dir=None) -> str:
    """Render _report_eval_result through a captured Console, return the text."""
    import io

    from rich.console import Console

    import benchflow.cli._shared as shared

    rec = Console(file=io.StringIO(), width=200)
    original = shared.console
    shared.console = rec
    try:
        shared._report_eval_result(result, job_dir)
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


def test_report_eval_result_shows_mean_reward():
    # Partial credit must be visible next to the binarized counts: pass/fail
    # thresholds at reward==1, so "0/1 (0.0%)" alone can't distinguish a 0.3
    # rubric score from a flat 0.
    out = _reported(
        SimpleNamespace(
            passed=0,
            total=1,
            errored=0,
            verifier_errored=0,
            score=0.0,
            job_name="j",
            mean_reward=0.3,
        )
    )
    assert "Score: 0/1" in out
    assert "mean reward 0.30" in out


def test_report_eval_result_omits_mean_reward_when_unavailable():
    # Sharded aggregation and older callers don't carry mean_reward — the line
    # must render without it rather than showing a misleading 0.00.
    out = _reported(
        SimpleNamespace(
            passed=0,
            total=1,
            errored=1,
            verifier_errored=0,
            score=0.0,
            job_name="j",
        )
    )
    assert "Score: 0/1" in out
    assert "mean reward" not in out


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


def _env0_rewards(reward=0.8):
    """env0-shaped rewards: numeric evidence nested one level under "metrics",
    with the matching totals under "details" (observed on gdoc-extract-content)."""
    return {
        "reward": reward,
        "metrics": {
            "summary_doc_exists": 1,
            "decisions_found": 5,
            "deadlines_found": 1,
            "originals_preserved": 1,
            "agent_acted": 1,
        },
        "details": {"decisions_total": 5, "deadlines_total": 5},
    }


def _failure(
    task_name: str, rollout_name: str | None = None, *, rewards: dict | None = None
):
    """A FAILED (scored) task; ``rewards`` defaults to the bare `reward 0.0`
    fallback shape."""
    from benchflow.evaluation import TaskFailure

    return TaskFailure(
        task_name=task_name,
        rewards={"reward": 0.0} if rewards is None else rewards,
        verifier_error=None,
        rollout_name=rollout_name,
    )


def test_failure_reason_flattens_nested_metrics():
    # env0 dogfood (gdoc-extract-content, reward 0.8): the decisive numbers sat
    # one level under "metrics" and the console printed the bare `reward 0.8`
    # fallback. The breakdown tier must flatten one level, pair <name>_found
    # with a positive <name>_total (env0 keeps totals under "details"), and
    # lead with the furthest-from-max metric — deadlines 1/5, not the
    # insertion-ordered all-ones.
    out = _reported(
        _failed_result([_failure("gdoc-extract-content", rewards=_env0_rewards())])
    )
    assert (
        "✗ gdoc-extract-content: reward 0.8 — deadlines 1/5, "
        "summary_doc_exists 1, decisions 5/5" in out
    )
    # The 3-metric cap still applies after flattening.
    assert "originals_preserved" not in out
    assert "agent_acted" not in out


def test_failure_reason_nested_details_when_no_metrics():
    # No "metrics" sub-dict: "details" numerics carry the breakdown, and
    # unpaired values keep the zero-first rule.
    out = _reported(
        _failed_result(
            [
                _failure(
                    "report-task",
                    rewards={"reward": 0.5, "details": {"sections": 1, "tables": 0}},
                )
            ]
        )
    )
    assert "✗ report-task: reward 0.5 — tables 0, sections 1" in out


def test_failure_reason_flat_metrics_win_over_nested():
    # Flat numeric keys keep working exactly as before — a nested sub-dict
    # never overrides them.
    out = _reported(
        _failed_result(
            [
                _failure(
                    "flat-task",
                    rewards={"reward": 0.2, "checks": 0, "metrics": {"x": 1}},
                )
            ]
        )
    )
    assert "✗ flat-task: reward 0.2 — checks 0" in out
    assert "x 1" not in out


def test_failure_reason_pairing_consumes_total_key():
    # found/total living in the SAME dict: the pair renders as a fraction and
    # the consumed *_total key never renders as its own metric.
    out = _reported(
        _failed_result(
            [
                _failure(
                    "pair-task",
                    rewards={
                        "reward": 0.4,
                        "metrics": {
                            "deadlines_found": 1,
                            "deadlines_total": 5,
                            "notes": 1,
                        },
                    },
                )
            ]
        )
    )
    assert "✗ pair-task: reward 0.4 — deadlines 1/5, notes 1" in out
    assert "deadlines_total" not in out


def test_failure_reason_zero_total_never_pairs():
    # A non-positive total can't pair (no division): both keys render as
    # plain zero-first values, and nothing raises.
    out = _reported(
        _failed_result(
            [
                _failure(
                    "zero-task",
                    rewards={"reward": 0.0, "metrics": {"a_found": 0, "a_total": 0}},
                )
            ]
        )
    )
    assert "✗ zero-task: reward 0.0 — a_found 0, a_total 0" in out


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


def _write_ctrf_tests(verifier_dir, tests: list[dict]) -> None:
    """Write a CTRF report (pytest-json-ctrf shape) with the given tests."""
    import json

    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "reportFormat": "CTRF",
                "results": {"tool": {"name": "pytest"}, "tests": tests},
            }
        )
    )


def _write_ctrf(verifier_dir, *, name: str | None, trace: str | None = None) -> None:
    """Write a minimal CTRF report.

    ``name`` is the failed test's node id; ``None`` writes an all-passed report.
    """
    tests = [{"name": "tests/test_x.py::test_ok", "status": "passed"}]
    if name is not None:
        test: dict = {"name": name, "status": "failed"}
        if trace is not None:
            test["trace"] = trace
        tests.append(test)
    _write_ctrf_tests(verifier_dir, tests)


def _write_stdout(verifier_dir, text: str) -> None:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "test-stdout.txt").write_text(text)


def _write_reward_json(verifier_dir, rewards: dict) -> None:
    """Write the verifier's reward.json (env0 shape: reward + metrics/details)."""
    import json

    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.json").write_text(json.dumps(rewards))


def test_failure_reason_artifact_tier_reads_ctrf(tmp_path):
    # Dogfood follow-up: `✗ fix-build: reward 0.0` while verifier/ctrf.json held
    # the real answer. A bare-reward reason is upgraded from the rollout dir the
    # engine recorded (rollout_name — no globbing), plus one (details: …) line.
    # The redundant "AssertionError: " prefix is stripped from the assertion.
    _write_ctrf(
        tmp_path / "fix-build__ab12cd34" / "verifier",
        name="tests/test_build.py::test_build_success",
        trace=(
            "def test_build_success():\n"
            ">       assert result.ok, 'Build failed for py312'\n"
            "E       AssertionError: Build failed for py312\n"
        ),
    )
    out = _reported(
        _failed_result([_failure("fix-build", "fix-build__ab12cd34")]), tmp_path
    )
    assert (
        "✗ fix-build: reward 0.0 — test_build_success failed: "
        "Build failed for py312" in out
    )
    assert f"(details: {tmp_path / 'fix-build__ab12cd34' / 'verifier'})" in out
    # A single failed test needs no roll-up count.
    assert "(+" not in out


def test_ctrf_multi_failure_appends_count_suffix(tmp_path):
    # Dogfood follow-up (skillsbench dialogue-parser, reward 0.667): the CTRF
    # held 2 failed / 4 passed but the console named only the first failure —
    # a reader stopping at the console under-counted what was broken. With >1
    # failed test the line gains a count suffix, and a param id carried by the
    # report's test name is preserved end-to-end. (pytest-json-ctrf, as of
    # 0.5.x, strips the param id at generation time — nodeid.split('[')[0] —
    # so this fixture uses the spec-conformant full name other producers emit.)
    _write_ctrf_tests(
        tmp_path / "dialogue-parser__ab12cd34" / "verifier",
        [
            {"name": "test_outputs.py::test_system_basics", "status": "passed"},
            {"name": "test_outputs.py::test_narrative_content", "status": "passed"},
            {
                "name": "test_outputs.py::test_graph_logic[reachability]",
                "status": "failed",
                "trace": (
                    ">           assert not unreachable\n"
                    "E           AssertionError: Unreachable nodes found: "
                    "['End']...\n"
                ),
            },
            {
                "name": "test_outputs.py::test_visualization_validity",
                "status": "failed",
            },
            {"name": "test_outputs.py::test_content_integrity", "status": "passed"},
            {"name": "test_outputs.py::test_structural_integrity", "status": "passed"},
        ],
    )
    from benchflow.evaluation import TaskFailure

    out = _reported(
        _failed_result(
            [
                TaskFailure(
                    task_name="dialogue-parser",
                    rewards={"reward": 0.667},
                    verifier_error=None,
                    rollout_name="dialogue-parser__ab12cd34",
                )
            ]
        ),
        tmp_path,
    )
    # The body hits the 100-char budget and truncates; the suffix rides after
    # it whole. Counts are report-wide: 1 extra failure, 4 of 6 checks passed.
    assert (
        "✗ dialogue-parser: reward 0.667 — test_graph_logic[reachability] "
        "failed: Unreachable nodes found:… (+1 more failure; 4/6 checks passed)" in out
    )


def test_ctrf_count_suffix_plural_and_param_id_with_colons(tmp_path):
    # 3 failed tests pluralize the suffix, and a param id containing "::" must
    # not be mangled by the node-id segment split (`test_foo[a::b]`, not `b]`).
    _write_ctrf_tests(
        tmp_path / "multi__ab12cd34" / "verifier",
        [
            {"name": "tests/test_x.py::test_foo[a::b]", "status": "failed"},
            {"name": "tests/test_x.py::test_bar", "status": "failed"},
            {"name": "tests/test_x.py::test_baz", "status": "failed"},
            {"name": "tests/test_x.py::test_ok", "status": "passed"},
        ],
    )
    out = _reported(_failed_result([_failure("multi", "multi__ab12cd34")]), tmp_path)
    assert (
        "✗ multi: reward 0.0 — test_foo[a::b] failed "
        "(+2 more failures; 1/4 checks passed)" in out
    )


def test_ctrf_count_suffix_survives_truncation(tmp_path):
    # A long assertion must not push the count suffix off the line: the body
    # alone is truncated to the 100-char budget, then the suffix is appended
    # whole — so the "more than this is broken" signal always survives.
    _write_ctrf_tests(
        tmp_path / "long__ab12cd34" / "verifier",
        [
            {
                "name": "tests/test_x.py::test_long",
                "status": "failed",
                "trace": "E   AssertionError: " + "verbose diagnosis " * 20 + "\n",
            },
            {"name": "tests/test_x.py::test_other", "status": "failed"},
            {"name": "tests/test_x.py::test_ok", "status": "passed"},
        ],
    )
    out = _reported(_failed_result([_failure("long", "long__ab12cd34")]), tmp_path)
    suffix = " (+1 more failure; 1/3 checks passed)"
    (line,) = [ln for ln in out.splitlines() if "✗ long" in ln]
    assert line.rstrip().endswith(f"…{suffix}")
    assert len(line.rstrip()) <= 100 + len(suffix)


def test_stdout_tail_tier_never_gets_count_suffix(tmp_path):
    # The suffix is a CTRF-tier concept: the stdout tail's pytest summary line
    # already carries the full counts, so nothing is appended there.
    _write_stdout(
        tmp_path / "fix-build__ab12cd34" / "verifier",
        "FAILED tests/test_build.py::test_a - AssertionError: boom\n"
        "FAILED tests/test_build.py::test_b - AssertionError: boom\n"
        "=========== 2 failed, 1 passed in 4.20s ===========\n",
    )
    out = _reported(
        _failed_result([_failure("fix-build", "fix-build__ab12cd34")]), tmp_path
    )
    assert "✗ fix-build: reward 0.0 — 2 failed, 1 passed in 4.20s" in out
    assert "(+" not in out


def test_failure_reason_artifact_tier_requires_rollout_name(tmp_path):
    # No rollout_name (old result.json predating the key, sharded aggregation):
    # nothing is resolved — even when a plausible dir exists on disk — and a
    # stale rollout_name whose dir is gone degrades the same way.
    _write_ctrf(tmp_path / "fix-build__ab12cd34" / "verifier", name="t::test_x")
    out = _reported(
        _failed_result(
            [
                _failure("fix-build", None),
                _failure("other-task", "other-task__99999999"),
            ]
        ),
        tmp_path,
    )
    assert "✗ fix-build: reward 0.0" in out
    assert "✗ other-task: reward 0.0" in out
    assert "test_x" not in out
    assert "details:" not in out


def test_failure_reason_artifact_tier_stdout_tail_summary(tmp_path):
    # No CTRF report: fall back to the tail of test-stdout.txt — the pytest
    # summary line wins over earlier FAILED lines, decoration stripped.
    _write_stdout(
        tmp_path / "fix-build__ab12cd34" / "verifier",
        "collected 3 items\n"
        "FAILED tests/test_build.py::test_build_success - AssertionError: boom\n"
        "=========== 1 failed, 2 passed in 40.86s ===========\n",
    )
    out = _reported(
        _failed_result([_failure("fix-build", "fix-build__ab12cd34")]), tmp_path
    )
    assert "✗ fix-build: reward 0.0 — 1 failed, 2 passed in 40.86s" in out


def test_failure_reason_artifact_tier_stdout_failed_line(tmp_path):
    # No summary line in the tail: the last FAILED line is the evidence.
    _write_stdout(
        tmp_path / "fix-build__ab12cd34" / "verifier",
        "collected 1 item\n"
        "FAILED tests/test_build.py::test_build - AssertionError: boom\n",
    )
    out = _reported(
        _failed_result([_failure("fix-build", "fix-build__ab12cd34")]), tmp_path
    )
    assert (
        "✗ fix-build: reward 0.0 — FAILED tests/test_build.py::test_build - "
        "AssertionError: boom" in out
    )


def test_failure_reason_artifact_tier_tries_sources_in_order(tmp_path):
    # Try-in-order, first non-None wins: a CTRF report that parses but carries
    # no failed test (reward zeroed by something else) yields to the stdout
    # tail, as does an oversized (> 64KB) report that can't be tail-parsed.
    no_failed = tmp_path / "task-a__11111111" / "verifier"
    _write_ctrf(no_failed, name=None)
    _write_stdout(no_failed, "=== 1 failed, 2 passed in 1.00s ===\n")
    oversized = tmp_path / "task-b__22222222" / "verifier"
    oversized.mkdir(parents=True)
    (oversized / "ctrf.json").write_text('{"pad": "%s"}' % ("x" * 70_000))
    _write_stdout(oversized, "=== 2 failed, 1 passed in 2.00s ===\n")
    out = _reported(
        _failed_result(
            [
                _failure("task-a", "task-a__11111111"),
                _failure("task-b", "task-b__22222222"),
            ]
        ),
        tmp_path,
    )
    assert "✗ task-a: reward 0.0 — 1 failed, 2 passed in 1.00s" in out
    assert "✗ task-b: reward 0.0 — 2 failed, 1 passed in 2.00s" in out


def test_failure_reason_artifact_tier_never_raises(tmp_path):
    # The tier's safety contract: a corrupt CTRF report falls through to the
    # stdout tail; with no other evidence it degrades to the bare `reward 0.0`
    # — no exception. The block's single pointer survives (first displayed
    # failure with an existing verifier dir — evidence or not).
    corrupt = tmp_path / "corrupt-task__11111111" / "verifier"
    corrupt.mkdir(parents=True)
    (corrupt / "ctrf.json").write_text("{not json")
    salvaged = tmp_path / "salvaged-task__33333333" / "verifier"
    salvaged.mkdir(parents=True)
    (salvaged / "ctrf.json").write_text("{not json")
    _write_stdout(salvaged, "=== 1 failed in 3.00s ===\n")
    (tmp_path / "empty-task__22222222" / "verifier").mkdir(parents=True)
    out = _reported(
        _failed_result(
            [
                _failure("corrupt-task", "corrupt-task__11111111"),
                _failure("empty-task", "empty-task__22222222"),
                _failure("salvaged-task", "salvaged-task__33333333"),
            ]
        ),
        tmp_path,
    )
    assert "✗ corrupt-task: reward 0.0" in out
    assert "✗ empty-task: reward 0.0" in out
    assert "✗ salvaged-task: reward 0.0 — 1 failed in 3.00s" in out
    assert out.count("(details:") == 1


def test_failure_reason_artifact_tier_reads_reward_json(tmp_path):
    # env0-style verifiers write reward.json + non-pytest stdout (no CTRF):
    # when the result's rewards dict was stripped to the bare aggregate, the
    # probe mines the same flattened low-metrics breakdown from disk — and the
    # (details:) pointer prints.
    _write_reward_json(tmp_path / "gdoc__ab12cd34" / "verifier", _env0_rewards())
    out = _reported(
        _failed_result([_failure("gdoc", "gdoc__ab12cd34", rewards={"reward": 0.8})]),
        tmp_path,
    )
    assert (
        "✗ gdoc: reward 0.8 — deadlines 1/5, summary_doc_exists 1, decisions 5/5" in out
    )
    assert f"(details: {tmp_path / 'gdoc__ab12cd34' / 'verifier'})" in out


def test_reward_json_yields_to_ctrf(tmp_path):
    # Probe order: the CTRF report (named failing checks) stays first;
    # reward.json only fills in when CTRF yields nothing.
    verifier = tmp_path / "both__ab12cd34" / "verifier"
    _write_ctrf(verifier, name="t::test_x")
    _write_reward_json(verifier, _env0_rewards())
    out = _reported(_failed_result([_failure("both", "both__ab12cd34")]), tmp_path)
    assert "✗ both: reward 0.0 — test_x failed" in out
    assert "deadlines" not in out


def test_reward_json_wins_over_stdout_tail(tmp_path):
    # And the other side of the probe order: with BOTH reward.json and a
    # pytest-shaped stdout tail present, the named metric breakdown beats the
    # anonymous counts — reordering those two probes fails this.
    verifier = tmp_path / "order__ab12cd34" / "verifier"
    _write_reward_json(verifier, _env0_rewards())
    _write_stdout(verifier, "=== 1 failed, 4 passed in 2.00s ===\n")
    out = _reported(
        _failed_result([_failure("order", "order__ab12cd34", rewards={"reward": 0.8})]),
        tmp_path,
    )
    assert "✗ order: reward 0.8 — deadlines 1/5" in out
    assert "1 failed, 4 passed" not in out


def test_reward_json_corrupt_falls_through(tmp_path):
    # The probe keeps the tier's never-raise contract: corrupt JSON falls
    # through to the stdout tail.
    verifier = tmp_path / "task__ab12cd34" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text("{not json")
    _write_stdout(verifier, "=== 1 failed in 2.00s ===\n")
    out = _reported(_failed_result([_failure("task", "task__ab12cd34")]), tmp_path)
    assert "✗ task: reward 0.0 — 1 failed in 2.00s" in out


def test_reward_json_without_named_metrics_yields(tmp_path):
    # A bare {"reward": …} reward.json repeats nothing the line doesn't
    # already say — the probe yields to the stdout tail.
    verifier = tmp_path / "bare__ab12cd34" / "verifier"
    _write_reward_json(verifier, {"reward": 0.0})
    _write_stdout(verifier, "=== 2 failed, 1 passed in 1.50s ===\n")
    out = _reported(_failed_result([_failure("bare", "bare__ab12cd34")]), tmp_path)
    assert "✗ bare: reward 0.0 — 2 failed, 1 passed in 1.50s" in out


def test_pointer_prints_when_every_probe_misses(tmp_path):
    # The third stacked miss from the env0 dogfood: exactly when evidence
    # extraction fails, the user also lost the pointer to find it themselves.
    # A failed task whose verifier dir exists gets the (details:) pointer even
    # beside a bare reward reason (here: non-pytest stdout, no CTRF, no
    # reward.json — every probe misses).
    verifier = tmp_path / "gdoc__ab12cd34" / "verifier"
    _write_stdout(verifier, "verifier ran custom checks; see reward artifacts\n")
    out = _reported(_failed_result([_failure("gdoc", "gdoc__ab12cd34")]), tmp_path)
    assert "✗ gdoc: reward 0.0" in out
    assert f"(details: {verifier})" in out


def test_failure_reason_artifact_tier_yields_to_metric_breakdown(tmp_path):
    # Named metrics already explain the miss — artifacts are only for reasons
    # that would otherwise be a bare `reward X`. The (details:) pointer is
    # decided separately, by on-disk artifacts alone: it prints here even
    # though the reason came from the in-memory metrics (an identical line
    # mined from reward.json must not differ on provenance the console can't
    # show).
    _write_ctrf(tmp_path / "plan-meeting__ab12cd34" / "verifier", name="t::test_x")
    out = _reported(
        _failed_result(
            [
                _failure(
                    "plan-meeting",
                    "plan-meeting__ab12cd34",
                    rewards={"reward": 0.3, "sections": 0.0},
                )
            ]
        ),
        tmp_path,
    )
    assert "✗ plan-meeting: reward 0.3 — sections 0.0" in out
    assert "test_x" not in out
    assert f"(details: {tmp_path / 'plan-meeting__ab12cd34' / 'verifier'})" in out


def test_report_eval_result_artifact_reads_stay_capped(tmp_path):
    # The display cap also bounds artifact reads: 7 bare failures with readable
    # artifacts still render 5 lines + "… and 2 more", and the pointer prints
    # once per block, not per line.
    for i in range(7):
        _write_ctrf(tmp_path / f"task-{i}__ab12cd34" / "verifier", name=f"t::test_{i}")
    failures = [_failure(f"task-{i}", f"task-{i}__ab12cd34") for i in range(7)]
    out = _reported(_failed_result(failures), tmp_path)
    assert out.count("✗ task-") == 5
    assert "… and 2 more" in out
    assert out.count("(details:") == 1


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


def test_ctrf_outcome_map_reports_every_test_not_only_the_failures(tmp_path):
    """Guards "feat(ablate): per-test attribution so a scalar tie cannot hide a
    behavioral difference": the per-test reading of the same CTRF report the
    failure line mines. Passes are evidence too — an ablation arm that flips a
    test *to* passing is exactly the difference a 0.00/0.00 scalar tie hides —
    and CTRF statuses pass through verbatim rather than being re-classified.
    """
    from benchflow.cli._failure_evidence import artifact_test_outcomes

    rollout = tmp_path / "lake__ab12cd34"
    _write_ctrf_tests(
        rollout / "verifier",
        [
            {"name": "::TestLake::test_trend_result", "status": "passed"},
            {"name": "::TestLake::test_dominant_factor", "status": "failed"},
            {"name": "::TestLake::test_optional", "status": "skipped"},
        ],
    )

    assert artifact_test_outcomes(rollout) == {
        "test_dominant_factor": "failed",
        "test_optional": "skipped",
        "test_trend_result": "passed",
    }
    # No CTRF report at all is "not observed", never an empty map: the caller
    # has to be able to say "scalar-only attribution" instead of "all tie".
    assert artifact_test_outcomes(tmp_path / "missing__ab12cd34") is None


def test_ctrf_outcome_map_keeps_raw_node_ids_when_short_names_collide(tmp_path):
    """Same-named tests in two files must not collapse into one row.

    Guards "feat(ablate): per-test attribution …": the display shortening that
    keeps the failure line inside its char budget is only safe while it stays
    injective — collapsing ``a.py::test_x`` and ``b.py::test_x`` would hide one
    arm's outcome behind the other's.
    """
    from benchflow.cli._failure_evidence import artifact_test_outcomes

    rollout = tmp_path / "collide__ab12cd34"
    _write_ctrf_tests(
        rollout / "verifier",
        [
            {"name": "tests/test_a.py::test_x", "status": "passed"},
            {"name": "tests/test_b.py::test_x", "status": "failed"},
        ],
    )

    assert artifact_test_outcomes(rollout) == {
        "tests/test_a.py::test_x": "passed",
        "tests/test_b.py::test_x": "failed",
    }
