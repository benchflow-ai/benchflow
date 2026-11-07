"""Guards the Docker client lifetime at 363d815b."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import docker as d
from tests.test_docker_build_identity import sandbox


@pytest.mark.skipif(os.name != "posix", reason="POSIX client process groups")
@pytest.mark.parametrize("ending", ["timeout", "cancel"])
@pytest.mark.parametrize("child_pipes", ["inherited", "closed"])
async def test_client_children_are_reaped_before_timeout_or_cancel_returns(
    tmp_path, monkeypatch, ending, child_pipes
):
    """Guards 363d815b: no buildx child can outlive the budgeted client."""
    env = sandbox(tmp_path)
    marker = tmp_path / "dummy-pids"
    script = (
        "import os,signal,time,sys\n"
        "child=os.fork()\n"
        "if not child:\n"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        + (" os.close(1); os.close(2)\n" if child_pipes == "closed" else "")
        + "if child:\n"
        " with open(sys.argv[1],'w') as f: f.write(f'{os.getpid()} {child}')\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(d, "_TIMEOUT_TEARDOWN_GRACE_SEC", 0.02)
    task = asyncio.create_task(
        env._run_docker_process(
            [sys.executable, "-c", script, str(marker)],
            check=True,
            timeout_sec=0.3 if ending == "timeout" else 30,
            what="dummy build",
        )
    )
    owned = []

    def running(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False

    try:
        async with asyncio.timeout(3):
            while not marker.exists() or len(marker.read_text().split()) != 2:
                await asyncio.sleep(0.01)
        owned = [int(v) for v in marker.read_text().split()]
        if ending == "cancel":
            task.cancel()
        with pytest.raises(
            RuntimeError if ending == "timeout" else asyncio.CancelledError
        ):
            await asyncio.wait_for(asyncio.shield(task), 2)
        for _ in range(50):
            if not any(running(pid) for pid in owned):
                break
            await asyncio.sleep(0.01)
        assert not any(running(pid) for pid in owned)
    finally:
        # Only dummy PIDs created by this test, in the throwaway container.
        for pid in owned:
            if running(pid):
                os.kill(pid, signal.SIGKILL)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_single_client_timeout_does_not_require_posix_signals(monkeypatch):
    """Guards 6c649624: retain Windows terminate/kill without SIGKILL support."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    process = SimpleNamespace(
        terminate=Mock(),
        kill=Mock(),
        communicate=AsyncMock(side_effect=[TimeoutError, (b"stopped", None)]),
    )
    monkeypatch.setattr(d, "signal", SimpleNamespace(SIGTERM=15), raising=False)
    assert await d._drain_timed_out_process(process) == (b"stopped", None)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


@pytest.mark.skipif(os.name != "posix", reason="POSIX client process groups")
@pytest.mark.parametrize(
    "pid",
    [None, False, True, -1, 0, 1, 1.0, "12", os.getpgrp() if os.name == "posix" else 0],
)
async def test_group_cleanup_refuses_non_child_process_identities(monkeypatch, pid):
    """Guards 6525eea1: an invalid PID must never become a broadcast signal."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    killpg = Mock()
    monkeypatch.setattr(d.os, "killpg", killpg)
    process = SimpleNamespace(pid=pid, communicate=AsyncMock(return_value=(b"", b"")))
    with pytest.raises(RuntimeError, match=r"invalid.*process.*identity"):
        await d._drain_timed_out_process(process, process_group=True)
    killpg.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="POSIX client process groups")
async def test_mock_pid_cannot_be_coerced_into_a_broadcast_signal(monkeypatch):
    """Guards 6525eea1: MagicMock.__index__ is 1, not its wrapped child's PID."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, Mock

    killpg = Mock()
    monkeypatch.setattr(d.os, "killpg", killpg)
    process = SimpleNamespace(
        pid=MagicMock(), communicate=AsyncMock(return_value=(b"", b""))
    )
    with pytest.raises(RuntimeError, match=r"invalid.*process.*identity"):
        await d._drain_timed_out_process(process, process_group=True)
    killpg.assert_not_called()
