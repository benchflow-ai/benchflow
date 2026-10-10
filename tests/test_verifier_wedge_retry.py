"""Verifier retry on the zero-output timeout (exec-layer wedge) signature.

Observed on Daytona (2026-08-07/08): verifiers that complete in under a second
when they run sometimes burned their whole timeout budget with an EMPTY
test-stdout.txt — the exec session wedged before test.sh ever started. Those
rollouts scored ``reward=None`` (lost data) even though the workspace held
scoreable work. ``_verify_rollout`` now retries the verifier exactly once when
a timeout carries the no-output signature; a timeout WITH output, or with the
verifier's processes still running (a genuinely slow or hung verifier), is
never retried.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout._setup import _verify_rollout


def _mk_task():
    return SimpleNamespace(
        name="wedge-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=0.2, reward_range=None, env={})
        ),
    )


def _mk_paths(tmp_path):
    return SimpleNamespace(verifier_dir=tmp_path / "verifier")


def _mk_planes(verifier_stub):
    planes = SimpleNamespace()
    planes.harden_before_verify = AsyncMock()
    planes.verifier = lambda **_: verifier_stub
    return planes


def _env_with_probe_output(stdout: str):
    env = SimpleNamespace()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(stdout=stdout, stderr="", return_code=0)
    )
    return env


@pytest.mark.asyncio
async def test_zero_output_timeout_retries_once_and_scores(tmp_path):
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)  # wedge: never returns
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")  # probe: empty test-stdout

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert verr is None
    assert vtimeout is None


@pytest.mark.asyncio
async def test_timeout_with_output_is_not_retried(tmp_path):
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("4242\n")  # probe: real output was produced

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 1
    assert rewards is None
    assert verr is not None and "timed out" in verr
    assert vtimeout is not None


@pytest.mark.asyncio
async def test_wedged_retry_that_also_times_out_reports_timeout(tmp_path):
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 2  # exactly one retry, never more
    assert rewards is None
    assert verr is not None and "timed out" in verr
    assert vtimeout is not None


# Signal stub for ``_ProcSandbox``: ``kill``/``pkill`` act on the fake table.
_SIGNAL_STUB = r"""
import os, shutil, sys
from pathlib import Path

proc, log = Path(os.environ["FAKE_PROC"]), Path(os.environ["KILL_LOG"])


def parent(p):
    for line in (p / "status").read_text().splitlines():
        if line.startswith("PPid:"):
            return line.split()[1]


def signal(pid, sig):
    with log.open("a") as f:
        f.write(f"{sig} {pid}\n")
    forked = proc / ".forks" / pid
    if forked.is_dir():  # a process starts as this signal lands
        for child in forked.iterdir():
            child.rename(proc / child.name)
        forked.rmdir()
    if sig == "STOP" and (proc / ".exits" / pid).exists():
        sig = "KILL"  # the process exits on its own before the stop lands
    if sig in ("KILL", "TERM") and (proc / pid).is_dir():
        shutil.rmtree(proc / pid)
        for child in proc.iterdir():  # orphans are reparented to PID 1
            if child.name.isdigit() and parent(child) == pid:
                (child / "stat").write_text(f"{child.name} (x) S 1 0 0\n")
                (child / "status").write_text(
                    f"Name:\tx\nPid:\t{child.name}\nPPid:\t1\n"
                )


mode, args = sys.argv[1], sys.argv[2:]
if mode == "kill":
    sig = args.pop(1) if args[0] == "-s" else args[0].lstrip("-")
    for pid in args[1:]:
        signal(pid, sig)
else:  # pkill -f PATTERN
    for p in list(proc.iterdir()):
        if not p.name.isdigit() or not (p / "cmdline").exists():
            continue
        if args[-1] in (p / "cmdline").read_bytes().replace(b"\0", b" ").decode():
            signal(p.name, "TERM")
"""


class _ProcSandbox:
    """Sandbox double whose ``exec`` runs a real ``/bin/sh`` over a fake /proc.

    ``/proc/`` and ``/logs/`` in each command resolve under ``root``, where
    every fake process is a ``<pid>/stat``, ``status`` and ``cmdline`` set. ``kill``
    and ``pkill`` are stubs acting on that table, so nothing on the test host
    is signalled. stderr is merged into stdout, as the Docker sandbox does.
    While a command runs, its own ``sh -c`` shows up in the table, as it
    would in a real sandbox. ``forks`` maps a PID to a ``(pid, ppid, argv)``
    process that starts just as the first signal reaches it (a child it
    forks, or anything else); PIDs in ``exits`` exit on their own just
    before a stop lands; PIDs in ``vanishing`` are listed in /proc but gone
    by the time their cmdline is opened; ``comms`` gives a PID a command
    name other than the default.
    """

    SELF = 999

    def __init__(self, root, procs, *, forks=(), exits=(), vanishing=(), comms=None):
        self.root = root
        self.proc = root / "proc"
        (root / "logs" / "verifier").mkdir(parents=True)
        (root / "logs" / "verifier" / "test-stdout.txt").write_text("")
        for pid, (ppid, argv) in procs.items():
            self._add(
                self.proc,
                pid,
                ppid,
                argv,
                **({"comm": comms[pid]} if comms and pid in comms else {}),
            )
        for trigger, (pid, ppid, argv) in dict(forks).items():
            self._add(self.proc / ".forks" / str(trigger), pid, ppid, argv)
        for pid in exits:
            (self.proc / ".exits").mkdir(exist_ok=True)
            (self.proc / ".exits" / str(pid)).touch()
        for pid in vanishing:
            self._add(self.proc, pid, 1, [])
            (self.proc / str(pid) / "cmdline").unlink()
            (self.proc / str(pid) / "cmdline").symlink_to("gone")
        self.kill_log = root / "signals.log"
        self.kill_log.write_text("")
        stub = root / "signal_stub.py"
        stub.write_text(_SIGNAL_STUB)
        self.bin = root / "bin"
        self.bin.mkdir()
        (self.bin / "pkill").write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{stub}" pkill "$@"\n'
        )
        (self.bin / "pkill").chmod(0o755)
        self._kill_fn = f'kill() {{ "{sys.executable}" "{stub}" kill "$@"; }}\n'
        self._fail_next = False

    @staticmethod
    def _add(where, pid, ppid, argv, comm="x) (y"):
        d = where / str(pid)
        d.mkdir(parents=True)
        # comm may hold spaces, parens and even newlines. ``stat`` prints it
        # raw; ``status`` escapes a newline as ``\n``, as the kernel does.
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1\n")
        name = comm.replace("\\", "\\\\").replace("\n", "\\n")
        (d / "status").write_text(f"Name:\t{name}\nPid:\t{pid}\nPPid:\t{ppid}\n")
        (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))

    def fail_next_exec(self):
        self._fail_next = True
        return self

    def alive(self):
        return {int(p.name) for p in self.proc.iterdir() if p.name.isdigit()}

    def signals(self):
        return self.kill_log.read_text().split("\n")[:-1]

    async def exec(self, command, **_):
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError("exec channel down")
        command = command.replace("/proc/", f"{self.proc}/").replace(
            "/logs/", f"{self.root}/logs/"
        )
        self._add(self.proc, self.SELF, 0, ["sh", "-c", command])
        try:
            r = subprocess.run(
                ["/bin/sh", "-c", self._kill_fn + command],
                env={
                    "PATH": f"{self.bin}:/usr/bin:/bin",
                    "FAKE_PROC": str(self.proc),
                    "KILL_LOG": str(self.kill_log),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
        finally:
            shutil.rmtree(self.proc / str(self.SELF), ignore_errors=True)
        return SimpleNamespace(stdout=r.stdout, stderr=None, return_code=r.returncode)


async def _verify_in(sandbox, tmp_path, verify):
    return await _verify_rollout(
        sandbox,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(SimpleNamespace(verify=verify)),
    )


def _slow_verify(attempts):
    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    return verify


def _wedged_then_scored_verify(attempts):
    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)
        return SimpleNamespace(rewards={"reward": 1.0})

    return verify


def _sandbox_with_verifier_tree(tmp_path, **kwargs):
    """A sandbox whose first verifier attempt is still running: the exec
    wrapper, test.sh, pytest (logging to its own file, so test-stdout.txt
    is empty) and a worker child, next to PID 1 and an unrelated service."""
    root = tmp_path / "sandbox"
    capture = f"{root}/logs/verifier/test-stdout.txt"
    return _ProcSandbox(
        root,
        {
            1: (0, ["sleep", "infinity"]),
            7: (0, ["python3", "/app/tests/server.py"]),
            40: (
                0,
                [
                    "sh",
                    "-c",
                    "trap 'rm -f /tmp/.env' EXIT && . /tmp/.env && "
                    f"/verifier/test.sh > {capture} 2>&1",
                ],
            ),
            41: (40, ["/bin/bash", "/verifier/test.sh"]),
            42: (41, ["python3", "-m", "pytest", "/verifier/test_outputs.py"]),
            43: (42, ["python3", "worker.py", "--steps", "1000000"]),
        },
        **kwargs,
    )


@pytest.mark.asyncio
async def test_slow_verifier_with_empty_stdout_is_not_retried(tmp_path):
    """Guards the zero-output retry from PR #949 against a slow verifier.

    test.sh is still running and pytest writes to its own log, so
    test-stdout.txt is empty. That is a slow verifier, not a wedge: it must
    time out once, not get a second verifier started beside it.
    """
    attempts = []
    sandbox = _sandbox_with_verifier_tree(tmp_path)

    rewards, verr, vtimeout = await _verify_in(
        sandbox, tmp_path, _slow_verify(attempts)
    )

    assert len(attempts) == 1
    assert rewards is None
    assert verr is not None and "timed out" in verr
    assert vtimeout is not None
    assert sandbox.alive() == {1, 7, 40, 41, 42, 43}
    assert sandbox.signals() == []


@pytest.mark.asyncio
async def test_probe_output_survives_a_process_exiting_mid_scan(tmp_path):
    """Guards the zero-output retry from PR #949 against shell noise.

    A process that exits between the /proc listing and the read makes the
    shell report an error, and the Docker sandbox merges stderr into the
    stdout the probe parses. That must not turn a slow verifier into a wedge.
    """
    attempts = []
    sandbox = _sandbox_with_verifier_tree(tmp_path, vanishing=(30,))

    *_, vtimeout = await _verify_in(sandbox, tmp_path, _slow_verify(attempts))

    assert len(attempts) == 1
    assert vtimeout is not None
    assert sandbox.signals() == []


@pytest.mark.asyncio
async def test_wedge_retry_first_kills_the_whole_verifier_tree(tmp_path):
    """Guards the orphan kill from PR #949 against leaving test.sh's children.

    When the probe cannot tell (its exec fails) the retry goes ahead, so the
    first attempt must go down with all of its descendants, including a
    child forked while the kill is scanning, and nothing else.
    """
    attempts = []
    sandbox = _sandbox_with_verifier_tree(
        tmp_path, forks={42: (44, 42, ["python3", "-c", "worker"])}
    ).fail_next_exec()

    rewards, verr, vtimeout = await _verify_in(
        sandbox, tmp_path, _wedged_then_scored_verify(attempts)
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert verr is None
    assert vtimeout is None
    assert sandbox.alive() == {1, 7}


@pytest.mark.asyncio
async def test_wedge_retry_kills_children_whose_parent_exits_mid_kill(tmp_path):
    """Guards the orphan kill from PR #949 against stopped orphans.

    The exec wrapper exits on its own just before the stop reaches it, so
    test.sh is reparented to PID 1. It and its children were already found
    and must still be killed, not left stopped.
    """
    attempts = []
    sandbox = _sandbox_with_verifier_tree(tmp_path, exits=(40,)).fail_next_exec()

    await _verify_in(sandbox, tmp_path, _wedged_then_scored_verify(attempts))

    assert len(attempts) == 2
    assert sandbox.alive() == {1, 7}


@pytest.mark.asyncio
async def test_wedge_kill_leaves_a_verifier_started_after_its_scan(tmp_path):
    """Guards the orphan kill from PR #949 against killing the retry.

    The kill only follows the process tree it found on its first scan. A new
    verifier wrapper that starts while it works (the retry, if the kill's
    exec outlived its budget) is not part of that tree and must survive.
    """
    attempts = []
    root = tmp_path / "sandbox"
    retry = ["sh", "-c", f"/verifier/test.sh > {root}/logs/verifier/test-stdout.txt"]
    sandbox = _sandbox_with_verifier_tree(
        tmp_path, forks={42: (60, 0, retry)}
    ).fail_next_exec()

    await _verify_in(sandbox, tmp_path, _wedged_then_scored_verify(attempts))

    assert len(attempts) == 2
    assert sandbox.alive() == {1, 7, 60}


@pytest.mark.asyncio
async def test_wedge_with_no_verifier_process_left_still_retries(tmp_path):
    """Guards the zero-output retry from PR #949: a real wedge still retries.

    No verifier process and an empty test-stdout.txt is the wedge signature.
    The probe's own shell is in the process table and must not count.
    """
    attempts = []
    sandbox = _ProcSandbox(
        tmp_path / "sandbox",
        {
            1: (0, ["sleep", "infinity"]),
            7: (0, ["python3", "/app/tests/server.py"]),
        },
    )

    rewards, verr, vtimeout = await _verify_in(
        sandbox, tmp_path, _wedged_then_scored_verify(attempts)
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert verr is None
    assert vtimeout is None
    assert sandbox.alive() == {1, 7}


def _busybox_verifier_tree(tmp_path, **kwargs):
    """The same running verifier under busybox ``sh -c``, which runs a single
    redirected command in place of itself: no wrapper names the capture
    file, and test.sh is the exec layer's direct child."""
    return _ProcSandbox(
        tmp_path / "sandbox",
        {
            1: (0, ["sleep", "infinity"]),
            7: (0, ["python3", "/app/tests/server.py"]),
            8: (0, ["/bin/sh", "/app/tests/test.sh"]),
            41: (0, ["/bin/sh", "/verifier/test.sh"]),
            42: (41, ["python3", "-m", "pytest", "/verifier/test_outputs.py"]),
            43: (42, ["python3", "worker.py", "--steps", "1000000"]),
        },
        **kwargs,
    )


@pytest.mark.asyncio
async def test_slow_verifier_without_a_wrapper_is_not_retried(tmp_path):
    """Guards the zero-output retry from PR #949 on shells that exec in place.

    With no wrapper naming the capture file, the probe falls back to test.sh
    itself, so a slow verifier is still told apart from a wedge.
    """
    attempts = []
    sandbox = _busybox_verifier_tree(tmp_path)

    *_, vtimeout = await _verify_in(sandbox, tmp_path, _slow_verify(attempts))

    assert len(attempts) == 1
    assert vtimeout is not None
    assert sandbox.signals() == []


@pytest.mark.asyncio
async def test_wedge_kill_without_a_wrapper_takes_the_test_sh_tree(tmp_path):
    """Guards the orphan kill from PR #949 on shells that exec in place.

    main's ``pkill -f '/verifier/test.sh'`` killed test.sh there; the tree
    kill must find it too and take its children with it, but leave a task
    service that merely has a test.sh of its own.
    """
    attempts = []
    sandbox = _busybox_verifier_tree(tmp_path).fail_next_exec()

    await _verify_in(sandbox, tmp_path, _wedged_then_scored_verify(attempts))

    assert len(attempts) == 2
    assert sandbox.alive() == {1, 7, 8}


@pytest.mark.asyncio
async def test_wedge_kill_ignores_a_command_name_posing_as_a_record(tmp_path):
    """Guards the orphan kill from PR #949 against a crafted command name.

    A process in the verifier tree names itself so that its raw ``stat``
    line reads as a second record making PID 2 (an unrelated service) its
    child. Parents must come from ``status``, so PID 2 survives.
    """
    attempts = []
    root = tmp_path / "sandbox"
    capture = f"{root}/logs/verifier/test-stdout.txt"
    sandbox = _ProcSandbox(
        root,
        {
            1: (0, ["sleep", "infinity"]),
            2: (0, ["python3", "/srv/service.py"]),
            40: (0, ["sh", "-c", f"/verifier/test.sh > {capture} 2>&1"]),
            41: (40, ["/bin/bash", "/verifier/test.sh"]),
            43: (41, ["python3", "worker.py"]),
        },
        comms={43: "w\n2 (svc) S 41"},
    ).fail_next_exec()

    await _verify_in(sandbox, tmp_path, _wedged_then_scored_verify(attempts))

    assert len(attempts) == 2
    assert sandbox.alive() == {1, 2}
