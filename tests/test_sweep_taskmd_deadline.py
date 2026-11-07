"""Guards the session deadline against PY8 base commit 363d815b."""

import asyncio
from dataclasses import replace

import pytest

from benchflow.taskmd import judging
from tests.test_taskmd_judging import _Script, _spec, _submission, _use


@pytest.mark.parametrize("late_at", ["provider", "run", "read"])
async def test_expired_sessions_never_accept_a_later_review(
    tmp_path, monkeypatch, late_at
):
    """A deadline applies at every tool boundary, including the final review."""
    clock = [0.0]
    monkeypatch.setattr(judging.time, "monotonic", lambda: clock[0])
    review = _use("submit_review", _submission(), "final")
    tools = (
        [review]
        if late_at == "provider"
        else [_use(late_at, {"command": "work", "path": "/work/report.md"}), review]
    )

    async def call(*args):
        if late_at == "provider":
            clock[0] = 2
        return {"content": tools}

    async def runner(command, timeout):
        assert timeout <= 1
        clock[0] = 2
        return 0, False, b"done"

    def read(*args):
        clock[0] = 2
        return "done", [], {}

    monkeypatch.setattr(judging, "read_result", read)
    result = await judging.run_session(
        replace(_spec(), seconds=1),
        call=call,
        credentials=judging.Credentials("api-key", "dummy"),
        files=judging.JudgeFiles(root=tmp_path, kept=["/work"]),
        evidence_text=None,
        runner=runner,
    )
    assert result.end == "timeout"
    assert result.accepted is None


async def test_timeout_awaits_runner_cancellation_cleanup(tmp_path):
    """A timed-out tool completes its finally block before the session returns."""
    cleaned = []

    async def runner(command, timeout):
        try:
            await asyncio.sleep(0.2)
        finally:
            await asyncio.sleep(0)
            cleaned.append(True)

    result = await judging.run_session(
        replace(_spec(), seconds=0.02),
        call=_Script(
            [
                [
                    _use("run", {"command": "work", "timeout": 60}),
                    _use("submit_review", _submission(), "final"),
                ]
            ]
        ),
        credentials=judging.Credentials("api-key", "dummy"),
        files=judging.JudgeFiles(root=tmp_path, kept=["/work"]),
        evidence_text=None,
        runner=runner,
    )
    assert cleaned == [True]
    assert result.end == "timeout" and result.accepted is None


async def test_sandbox_runner_sweeps_before_propagating_cancellation(monkeypatch):
    """Cancelling a backend exec also reaps the runner's processes."""
    from types import SimpleNamespace

    from benchflow.taskmd import verify

    calls = []
    completed = []

    async def execute(sandbox, script, **kwargs):
        calls.append(script)
        if len(calls) == 1:
            await asyncio.sleep(0.05)
            completed.append(True)
        else:
            assert completed == [True], "a cancelled client is not remote completion"
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(verify, "root_exec", execute)
    runner = verify.SandboxRunner(sandbox=SimpleNamespace(), kept=[], views={})
    runner.ready = True
    runner.drop = ("setpriv", "/usr/bin/setpriv", "")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner("work", 60), timeout=0.02)
    assert len(calls) == 2 and calls[-1].endswith("bf_sweep")
