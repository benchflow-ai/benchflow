"""Unit tests for the eval live-progress dashboard (cli/_live_progress.py).

State math + the TTY/quiet-logging gates are tested directly; the Rich render is
exercised for "doesn't raise" rather than pixel-asserted.
"""

from __future__ import annotations

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


def test_resume_denominator_and_progress():
    d = _dash()
    d.on_plan(total=10, done=6, remaining=4)  # 6 resumed-complete
    d.on_task_start("x")
    d.on_result("x", _result(1.0))
    # done = resumed(6) + finished(1) = 7 of 10; render reflects it
    assert d._resumed == 6 and d._passed == 1
    d.__rich__()


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


def test_progress_enabled_respects_tty_and_optout(monkeypatch):
    tty = SimpleNamespace(is_terminal=True)
    notty = SimpleNamespace(is_terminal=False)
    monkeypatch.delenv("BENCHFLOW_NO_PROGRESS", raising=False)
    assert progress_enabled(tty) is True
    assert progress_enabled(notty) is False
    monkeypatch.setenv("BENCHFLOW_NO_PROGRESS", "1")
    assert progress_enabled(tty) is False


def test_quiet_root_logging_restores_handlers():
    root = logging.getLogger()
    before = root.handlers[:]
    with quiet_root_logging():
        assert all(isinstance(h, logging.NullHandler) for h in root.handlers)
    assert root.handlers == before


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
