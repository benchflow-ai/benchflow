"""Regressions for GH #1078 and the PR #1088 quiescence approach."""

import os
import shlex
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.lockdown import (
    _ASSERT_SANDBOX_USER_QUIESCENT_CMD_TEMPLATE,
    _SAFE_VERIFIER_PATH,
    _kill_sandbox_user_procs,
)


@pytest.mark.asyncio
async def test_failed_final_probe_prevents_verification():
    """GH #1078: an unsuccessful final process observation cannot admit scoring."""
    env = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(
                    return_code=1, stdout="1234\n", stderr="process still running"
                ),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="quiescence"):
        await _kill_sandbox_user_procs(env, "agent")


# How each fake /proc entry behaves. ``dies_on`` removes the entry when that
# signal arrives; ``vanished`` is a dangling link, i.e. a process that exited
# between the directory scan and the status read.
_PROCESSES = {
    "live": ("State:\tS (sleeping)\nUid:\t1234\t1234\t1234\t1234\n", None),
    "dies_on_term": ("State:\tS (sleeping)\nUid:\t1234\t1234\t1234\t1234\n", "TERM"),
    "dies_on_kill": ("State:\tR (running)\nUid:\t1234\t1234\t1234\t1234\n", "KILL"),
    "zombie": ("State:\tZ (zombie)\nUid:\t1234\t1234\t1234\t1234\n", None),
    "other_effective_uid": (
        "State:\tS (sleeping)\nUid:\t1234\t5678\t1234\t1234\n",
        None,
    ),
    "no_state": ("Name:\tunknown\nUid:\t1234\t1234\t1234\t1234\n", None),
    "unreadable": (None, None),
    "vanished": (None, None),
}


def run_quiescence(tmp_path, processes, *, process_table=True):
    """Run the real quiescence script under ``/bin/sh`` against a fake /proc.

    ``id``, ``sleep`` and ``kill`` are shell functions so the host's own
    processes are never signalled; everything else is the shipped script.
    """
    proc = tmp_path / "proc"
    proc.mkdir()
    if process_table:
        (proc / "self").mkdir()
        (proc / "self/status").write_text("State:\tR (running)\nUid:\t0\t0\t0\t0\n")
    for pid, kind in enumerate(processes, start=100):
        entry = proc / str(pid)
        status, dies_on = _PROCESSES[kind]
        if kind == "vanished":
            entry.symlink_to(tmp_path / "exited")
            continue
        entry.mkdir()
        if kind == "unreadable":
            (entry / "status").mkdir()
        else:
            (entry / "status").write_text(status)
        if dies_on:
            (entry / f"dies_on_{dies_on}").touch()
    signals = tmp_path / "signals"
    prelude = f"""
id() {{ echo 1234; }}
sleep() {{ :; }}
kill() {{
    echo "$2 $3" >> {shlex.quote(str(signals))}
    if [ -e {shlex.quote(str(proc))}/"$3"/dies_on_"$2" ]; then
        rm -rf {shlex.quote(str(proc))}/"$3"
    fi
}}
"""
    script = _ASSERT_SANDBOX_USER_QUIESCENT_CMD_TEMPLATE.replace("__USER__", "agent")
    script = script.replace("/proc", str(proc))
    result = subprocess.run(
        ["/bin/sh", "-c", prelude + script],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    sent = signals.read_text().split("\n") if signals.exists() else []
    return result, [line for line in sent if line]


@pytest.mark.parametrize(
    "processes,expected,signalled",
    [
        ([], 0, []),  # no process of the sandbox user
        (["dies_on_term"], 0, ["TERM 100"]),
        (["dies_on_kill"], 0, ["TERM 100", "KILL 100"]),  # second signal
        (["live"], 1, ["TERM 100", "KILL 100"]),  # writer survived
        (["zombie"], 0, []),  # a zombie cannot write
        (["other_effective_uid"], 0, []),  # effective UID decides
        (["vanished", "zombie"], 0, []),  # exited before inspection
        (["no_state"], 2, []),  # existing entry without a state
        (["unreadable", "dies_on_term"], 2, []),  # existing but unreadable
    ],
)
def test_real_shell_distinguishes_absence_failure_and_live_writers(
    tmp_path, processes, expected, signalled
):
    """GH #1078/PR #1088 semantics, now without procps or python3.

    Guards the fix for the process quiescence check, whose check needed pgrep or python3
    and failed with rc=127 on images that ship neither. A failed observation
    must still never pass as quiescence.
    """
    result, sent = run_quiescence(tmp_path, processes)
    assert result.returncode == expected, (result.stdout, result.stderr)
    assert sent == signalled


def test_missing_process_table_is_not_quiescence(tmp_path):
    """GH #1078: no readable /proc is an unknown state, not an empty one."""
    result, sent = run_quiescence(tmp_path, ["live"], process_table=False)
    assert result.returncode == 2
    assert sent == []


@pytest.mark.asyncio
async def test_quiescence_needs_neither_procps_nor_python(tmp_path):
    """Guards the fix for the process quiescence check: busybox/slim images without pgrep,
    pkill or python3 must be observed as quiescent, not fail with rc=127."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in {"id": "echo 1234", "sleep": "exit 0"}.items():
        (bindir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bindir / name).chmod(0o755)
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/status").write_text("State:\tR (running)\nUid:\t0\t0\t0\t0\n")
    (proc / "100").mkdir()
    (proc / "100/status").write_text(
        "State:\tZ (zombie)\nUid:\t1234\t1234\t1234\t1234\n"
    )
    commands = []

    class ShellEnv:
        async def exec(self, command, **kwargs):
            commands.append(command)
            command = command.replace(
                f"export PATH={shlex.quote(_SAFE_VERIFIER_PATH)}",
                f"export PATH={shlex.quote(str(bindir))}",
            ).replace("/proc", str(proc))
            result = subprocess.run(
                ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=10
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    await _kill_sandbox_user_procs(ShellEnv(), "agent")
    assert len(commands) == 2
    assert "python3" not in commands[1] and "pgrep" not in commands[1]


@pytest.mark.skipif(
    not os.path.exists("/proc/self/status") or os.geteuid() == 0,
    reason="needs Linux /proc and a non-root test user",
)
def test_real_process_is_stopped_through_real_proc(tmp_path):
    """Guards the fix for the process quiescence check against real kernel status files."""
    child = subprocess.Popen(["sleep", "60"])
    try:
        proc = tmp_path / "proc"
        proc.mkdir()
        (proc / "self").symlink_to("/proc/self", target_is_directory=True)
        (proc / str(child.pid)).symlink_to(
            f"/proc/{child.pid}", target_is_directory=True
        )
        script = _ASSERT_SANDBOX_USER_QUIESCENT_CMD_TEMPLATE.replace(
            "__USER__", "agent"
        ).replace("/proc", str(proc))
        result = subprocess.run(
            ["/bin/sh", "-c", f"id() {{ echo {os.getuid()}; }}\n" + script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert child.wait(timeout=10) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.asyncio
async def test_controller_uid_rejected_before_any_signal(tmp_path):
    """GH #1078: even a root alias must not kill the verifier/controller UID."""

    class ShellEnv:
        async def exec(self, command, **kwargs):
            # The real local root identity resolves to zero. No pkill may run.
            # Replace both signal commands with a marker to keep the host untouched.
            sentinel = tmp_path / "signalled"
            command = command.replace("pkill", f"touch {sentinel}; false")
            result = subprocess.run(
                ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=3
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    with pytest.raises(RuntimeError, match="identity check"):
        await _kill_sandbox_user_procs(ShellEnv(), "root")
    assert not (tmp_path / "signalled").exists()
