"""A stop in flight is the trial's ending, even when its cleanup fails.

Review of PR #122 (F2b-R122, 2026-11-13, M2; the lead's ruling: a
cancellation in flight wins). Ctrl-C during a trial whose cleanup then
raised recorded ``error: "trial cleanup failed: ..."`` with
``error_category: infra_failure`` beside ``interrupted: true``: the stop
read as an infrastructure failure. The record now says it was interrupted,
as a plain Ctrl-C does, and keeps the cleanup's failure beside it in
``post_turn_failure_info``.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.test_en374_salvage import _rollout

CLEANUP_FAILURE = "sandbox teardown failed: container is not running"


def _prepare(tmp_path, monkeypatch, *, cleanup_fails: bool):
    rollout, trial = _rollout(
        tmp_path,
        monkeypatch,
        agent="claude-agent-acp",
        harness="acp",
        home=tmp_path / "home",
    )
    (trial / "config.json").write_text(json.dumps({"agent": "claude-agent-acp"}))
    entered = asyncio.Event()

    async def setup():
        entered.set()
        await asyncio.sleep(60)

    async def cleanup():
        if cleanup_fails:
            raise RuntimeError(CLEANUP_FAILURE)

    monkeypatch.setattr(rollout, "_setup_and_start", setup)
    monkeypatch.setattr(rollout, "cleanup", cleanup)
    return rollout, trial, entered


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["plain", "cleanup-fails"])
async def test_a_cancel_is_recorded_as_interrupted(
    tmp_path, monkeypatch, cleanup_fails
):
    rollout, trial, entered = _prepare(
        tmp_path, monkeypatch, cleanup_fails=cleanup_fails
    )
    task = asyncio.create_task(rollout.run())
    await asyncio.wait_for(entered.wait(), 10)
    task.cancel()  # what Ctrl-C, SIGTERM or a budget stop does to the trial
    with pytest.raises(asyncio.CancelledError):
        await task

    data = json.loads((trial / "result.json").read_text())
    assert data["interrupted"] is True
    assert data["error"].startswith("Interrupted")
    assert data["error_category"] != "infra_failure"
    failure = data.get("post_turn_failure_info")
    if cleanup_fails:
        assert failure is not None
        assert failure["step"] == "cleanup"
        assert CLEANUP_FAILURE in failure["detail"]
    else:
        assert failure is None
    # A resume runs the stopped trial again: the failure beside the stop is
    # no kept ending (result_paths.kept_ending_is_final is never asked).
    from benchflow._utils.result_paths import load_task_results

    assert load_task_results(trial.parent) == {}


SCRIPT = textwrap.dedent(
    """
    import asyncio, json, signal, sys
    from pathlib import Path

    # A suite started in the background (setsid nohup ... &) ignores SIGINT,
    # and a child inherits that: the trial must get the Ctrl-C regardless.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.path.insert(0, sys.argv[2])
    from tests.test_en374_cancel_cleanup import CLEANUP_FAILURE
    from tests.test_en374_salvage import _rollout

    class Patch:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    tmp = Path(sys.argv[1])
    rollout, trial = _rollout(
        tmp, Patch(), agent="claude-agent-acp", harness="acp", home=tmp / "home"
    )
    (trial / "config.json").write_text(json.dumps({"agent": "claude-agent-acp"}))

    async def setup():
        print("ENTERED", flush=True)
        await asyncio.sleep(60)

    async def cleanup():
        raise RuntimeError(CLEANUP_FAILURE)

    rollout._setup_and_start = setup
    rollout.cleanup = cleanup
    print("TRIAL", trial, flush=True)
    try:
        asyncio.run(rollout.run())
    except KeyboardInterrupt:
        print("STOPPED", flush=True)
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT to a child process")
def test_ctrl_c_during_a_failing_cleanup_is_recorded_as_interrupted(tmp_path):
    """A real Ctrl-C: SIGINT to a process running the trial under
    asyncio.run, whose cleanup then fails."""
    root = Path(__file__).resolve().parent.parent
    script = tmp_path / "run_trial.py"
    script.write_text(SCRIPT)
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "HARBOR_TELEMETRY": "off"}
    child = subprocess.Popen(
        [sys.executable, str(script), str(work), str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=root,
        env=env,
    )
    assert child.stdout is not None
    trial = None
    try:
        for line in child.stdout:
            if line.startswith("TRIAL "):
                trial = Path(line.split(" ", 1)[1].strip())
            if line.startswith("ENTERED"):
                break
        else:
            child.wait(30)
            pytest.fail("the trial never started")
        child.send_signal(signal.SIGINT)
        rest, _ = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(10)
    assert "STOPPED" in rest, rest[-2000:]
    assert trial is not None
    data = json.loads((trial / "result.json").read_text())
    assert data["interrupted"] is True
    assert data["error"].startswith("Interrupted"), data["error"]
    assert data["error_category"] != "infra_failure"
    assert data["post_turn_failure_info"]["step"] == "cleanup"
    assert CLEANUP_FAILURE in data["post_turn_failure_info"]["detail"]
