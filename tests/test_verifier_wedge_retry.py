"""Verifier retry on the zero-output timeout (exec-layer wedge) signature.

Verifiers that complete in under a second when they run could burn their whole timeout budget with an EMPTY
test-stdout.txt — the exec session wedged before test.sh ever started. Those
rollouts scored ``reward=None`` (lost data) even though the workspace held
scoreable work. ``_verify_rollout`` now retries the verifier exactly once when
a timeout carries the no-output signature; a timeout WITH output, or with the
verifier's processes still running (a genuinely slow or hung verifier), is
never retried.

That in-place retry (PR #949, 66054d5b) still applies to every task without a
verifier-only recovery contract; solver-evidence preservation had removed it for all tasks. Tasks
with an eligible recovery contract (#1136) instead use command start receipts
and recover in a fresh sandbox, never replaying in place.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.task_authoring import task_digest
from benchflow.rollout import Rollout, RolloutConfig, _setup
from benchflow.rollout._setup import _verify_rollout
from benchflow.task import RolloutPaths, Task


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
    planes.quiesce_agent = AsyncMock()
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
    """Guards PR #949 against the solver-evidence preservation removal of the in-place retry."""
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
async def test_timeout_with_output_is_not_retried(tmp_path, caplog):
    """Guards PR #949: a slow verifier that produced output is never replayed."""
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
    assert rewards == {"reward": 0.0}
    assert verr is None
    assert "timed out" in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert vtimeout is not None


@pytest.mark.asyncio
async def test_wedged_retry_that_also_times_out_reports_timeout(tmp_path, caplog):
    """Guards PR #949: exactly one in-place retry, never more."""
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
    assert rewards == {"reward": 0.0}
    assert verr is None
    assert "timed out" in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert vtimeout is not None


def _rollout(tmp_path, verify, *, recovery: bool):
    task = tmp_path / "task"
    task.mkdir()
    toml = 'version = "1.0"\n[verifier]\ntimeout_sec = 0.2\n'
    if recovery:
        toml += "workspace_recovery = true\n"
        toml += '[sandbox]\ndocker_image = "python@sha256:' + "a" * 64 + '"\n'
    (task / "task.toml").write_text(toml)
    (task / "instruction.md").write_text("Produce a file.")
    rollout = Rollout(RolloutConfig(task_path=task, task_digest=task_digest(task)))
    rollout._task = Task(task)
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=rollout._rollout_dir)
    rollout._env = _env_with_probe_output("0\n")
    # The real verifier records a start receipt on the recovery path.
    rollout._planes = _mk_planes(
        SimpleNamespace(verify=verify, execution_receipt=("/logs/verifier/s", "main"))
    )
    rollout._agent_cwd = "/app"
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]
    return rollout


@pytest.mark.asyncio
async def test_rollout_without_recovery_contract_retries_wedge_in_place(
    tmp_path, monkeypatch
):
    """Guards PR #949 through Rollout.verify against the solver-evidence preservation regression."""
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)
        return SimpleNamespace(rewards={"reward": 1.0})

    rollout = _rollout(tmp_path, verify, recovery=False)
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )

    assert await rollout.verify() == {"reward": 1.0}
    assert len(attempts) == 2
    assert rollout._verifier_error is None


@pytest.mark.asyncio
async def test_rollout_with_recovery_contract_leaves_wedge_to_recovery(
    tmp_path, monkeypatch, caplog
):
    """Guards #1136: an eligible task reports the wedge instead of replaying."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    rollout = _rollout(tmp_path, verify, recovery=True)
    rollout._env.exec.return_value.return_code = 1  # no start receipt
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )
    monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", AsyncMock())

    assert await rollout.verify() == {"reward": 0.0}
    assert len(attempts) == 1
    assert rollout._verifier_error is None
    assert "Verifier scored 0" in caplog.text


@pytest.mark.asyncio
async def test_timeout_without_start_receipt_is_not_replayed(tmp_path, caplog):
    """Guards #1136: an eligible task never replays its verifier in place."""
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)  # wedge: never returns
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")

    rewards, verr, vtimeout = await _verify_rollout(
        env,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )

    assert len(attempts) == 1
    assert rewards == {"reward": 0.0}
    assert verr is None
    assert "timed out" in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert vtimeout is not None
    env.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_command_receipt_reports_wedge_without_in_place_retry(
    tmp_path, caplog
):
    """Guards #1136: an eligible task reports startup loss for fresh recovery."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start-unique", "main")
    )
    env = _env_with_probe_output("0\n")

    rewards, verr, vtimeout = await _verify_rollout(
        env,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )

    assert len(attempts) == 1
    assert rewards == {"reward": 0.0}
    assert verr is None
    assert "Verifier scored 0" in caplog.text
    assert vtimeout is None


@pytest.mark.asyncio
async def test_quiet_started_verifier_is_a_real_timeout(tmp_path, caplog):
    """Guards #1136: a start receipt beats lack of stdout."""

    async def verify():
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/unique", "main")
    )
    rewards, error, diagnostic = await _verify_rollout(
        _env_with_probe_output("started"),
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "Verifier scored 0" in caplog.text
    assert diagnostic is not None


@pytest.mark.asyncio
async def test_startup_wedge_uses_short_grace_not_full_budget(
    tmp_path, monkeypatch, caplog
):
    """Guards #1136: startup loss does not consume two verifier budgets."""
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)
    task = _mk_task()
    task.config.verifier.timeout_sec = 60

    async def verify():
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start", "main")
    )
    rewards, error, diagnostic = await asyncio.wait_for(
        _verify_rollout(
            _env_with_probe_output(""),
            task,
            _mk_paths(tmp_path),
            {},
            _mk_planes(verifier),
            recovery_eligible=True,
        ),
        timeout=1,
    )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "Verifier scored 0" in caplog.text
    assert diagnostic is None


@pytest.mark.asyncio
async def test_receipt_probe_error_is_unknown_not_a_wedge(tmp_path, monkeypatch):
    """Guards the fix for solver-evidence preservation's false wedge verdict on a transient probe error.

    A failed start-receipt probe says nothing about the verifier; treating it as
    "not started" cancelled a running verifier after one grace interval.
    """
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)
    task = _mk_task()
    task.config.verifier.timeout_sec = 5
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(0.2)
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start", "main")
    )
    env = SimpleNamespace(
        exec=AsyncMock(side_effect=RuntimeError("Failed to execute session command"))
    )
    rewards, error, diagnostic = await _verify_rollout(
        env,
        task,
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )
    assert (rewards, error, diagnostic) == ({"reward": 1.0}, None, None)
    assert len(attempts) == 1
    assert env.exec.await_count >= 1


@pytest.mark.asyncio
async def test_unknown_start_until_deadline_is_a_timeout(tmp_path, monkeypatch, caplog):
    """Guards the same fix: probe errors until the budget ends are a timeout."""
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)

    async def verify():
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start", "main")
    )
    env = SimpleNamespace(exec=AsyncMock(side_effect=TimeoutError()))
    rewards, error, diagnostic = await _verify_rollout(
        env,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "Verifier scored 0" in caplog.text
    assert diagnostic is not None


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
    every fake process is a ``<pid>/stat``, ``status`` and ``cmdline`` set.
    ``kill`` and ``pkill`` are stubs acting on that table, so nothing on the
    test host is signalled. ``envs`` gives a PID an ``environ`` (``NAME=value``
    entries); every other one's is empty. stderr is merged into stdout, as the Docker
    sandbox does. While a command runs, its own ``sh -c`` shows up in the
    table, as it would in a real sandbox. ``forks`` maps a PID to a
    ``(pid, ppid, argv)`` process that starts just as the first signal
    reaches it; PIDs in ``exits`` exit on their own just before a stop lands;
    PIDs in ``vanishing`` are listed in /proc but gone by the time their
    cmdline is opened; ``comms`` gives a PID a command name of its own.
    """

    SELF = 999

    def __init__(
        self, root, procs, *, forks=(), exits=(), vanishing=(), comms=None, envs=None
    ):
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
                env=(envs or {}).get(pid, ()),
            )
        for trigger, (pid, ppid, argv) in dict(forks).items():
            self._add(self.proc / ".forks" / str(trigger), pid, ppid, argv)
        for pid in exits:
            (self.proc / ".exits").mkdir(exist_ok=True)
            (self.proc / ".exits" / str(pid)).touch()
        for pid in vanishing:
            self._add(self.proc, pid, 1, [])
            for name in ("cmdline", "environ"):
                (self.proc / str(pid) / name).unlink()
                (self.proc / str(pid) / name).symlink_to("gone")
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
    def _add(where, pid, ppid, argv, comm="x) (y", env=()):
        d = where / str(pid)
        d.mkdir(parents=True)
        # comm may hold spaces, parens and even newlines. ``stat`` prints it
        # raw; ``status`` escapes a newline as ``\n``, as the kernel does.
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1\n")
        name = comm.replace("\\", "\\\\").replace("\n", "\\n")
        (d / "status").write_text(f"Name:\t{name}\nPid:\t{pid}\nPPid:\t{ppid}\n")
        (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
        (d / "environ").write_bytes(b"".join(e.encode() + b"\0" for e in env))

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
    """Guards #1164's probe: a slow verifier is not a wedge.

    test.sh is still running and pytest writes to its own log, so
    test-stdout.txt is empty. It must time out once, not get a second
    verifier started beside it.
    """
    attempts = []
    sandbox = _sandbox_with_verifier_tree(tmp_path)

    rewards, verr, vtimeout = await _verify_in(
        sandbox, tmp_path, _slow_verify(attempts)
    )

    assert len(attempts) == 1
    assert rewards == {"reward": 0.0}
    assert verr is None
    assert vtimeout is not None
    assert sandbox.alive() == {1, 7, 40, 41, 42, 43}
    assert sandbox.signals() == []


@pytest.mark.asyncio
async def test_probe_output_survives_a_process_exiting_mid_scan(tmp_path):
    """Guards #1164's probe against shell noise.

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
    """Guards #1164's kill: the retry takes the first attempt's whole tree.

    When the probe cannot tell (its exec fails) the retry goes ahead, so the
    first attempt must go down with all of its descendants, including a
    child forked while the kill is scanning, and nothing else. ``pkill -f``
    took the exec shell and test.sh only, and nothing on an image without
    procps.
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
    """Guards #1164's kill against stopped orphans.

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
    """Guards #1164's kill against killing the retry.

    The kill only follows the process tree it found on its first scan. A new
    verifier wrapper that starts while it works is not part of that tree and
    must survive.
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
    """Guards #949's retry: a real wedge still retries.

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
    """Guards #1164's probe on shells that exec in place.

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
    """Guards #1164's kill on shells that exec in place.

    The tree kill must find test.sh and take its children with it, but leave
    a task service that merely has a test.sh of its own.
    """
    attempts = []
    sandbox = _busybox_verifier_tree(tmp_path).fail_next_exec()

    await _verify_in(sandbox, tmp_path, _wedged_then_scored_verify(attempts))

    assert len(attempts) == 2
    assert sandbox.alive() == {1, 7, 8}


@pytest.mark.asyncio
async def test_wedge_kill_ignores_a_command_name_posing_as_a_record(tmp_path):
    """Guards #1164's kill against a crafted command name.

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


@pytest.mark.asyncio
async def test_wedge_kill_with_a_start_receipt_takes_only_its_command(tmp_path):
    """Guards the kill's roots when the attempt has a start receipt.

    A command that never confirmed its start is retried in place; the kill
    before the retry takes only the tree of the command naming this
    attempt's receipt, never another test command beside it (another
    attempt's, or another sandbox's on a shared host).
    """
    attempts = []
    root = tmp_path / "sandbox"
    capture = f"{root}/logs/verifier/test-stdout.txt"
    mine = "/logs/verifier/verifier-started-" + "a" * 32
    other = "/logs/verifier/verifier-started-" + "b" * 32

    def command(receipt):
        started = f"printf started > {receipt}"
        return ["sh", "-c", f"{started} && /verifier/test.sh > {capture}"]

    sandbox = _ProcSandbox(
        root,
        {
            1: (0, ["sleep", "infinity"]),
            40: (0, command(mine)),
            41: (40, ["/bin/bash", "/verifier/test.sh"]),
            50: (0, command(other)),
            51: (50, ["/bin/bash", "/verifier/test.sh"]),
        },
    )

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(verify=verify, execution_receipt=(mine, "main"))
    planes = _mk_planes(verifier)
    rewards, *_ = await _verify_rollout(
        sandbox, _mk_task(), _mk_paths(tmp_path), {}, planes
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert sandbox.alive() == {1, 50, 51}


@pytest.mark.asyncio
async def test_wedge_kill_finds_a_command_sent_as_data_by_its_environment(tmp_path):
    """Guards the receipt-scoped kill when no command line names the receipt.

    A test command too long for a command line is sent as data (``sh -s``),
    so only the environment of the processes it starts carries the attempt
    (``BENCHFLOW_VERIFIER_ATTEMPT``). The kill takes this attempt's tests by
    it, and leaves another attempt's test.sh and one with no attempt at all:
    a kill that arrives late, once a retry's tests run, must not take them.
    """
    attempts = []
    mine = "verifier-started-" + "c" * 32
    other = "verifier-started-" + "d" * 32
    marked = {"BENCHFLOW_VERIFIER_ATTEMPT=" + mine, "PATH=/usr/bin:/bin"}
    sandbox = _ProcSandbox(
        tmp_path / "sandbox",
        {
            1: (0, ["sleep", "infinity"]),
            7: (0, ["python3", "/app/tests/server.py"]),
            40: (0, ["/bin/sh", "-s"]),
            41: (40, ["/bin/sh", "/verifier/test.sh"]),
            42: (41, ["python3", "-m", "pytest", "/verifier/test_outputs.py"]),
            43: (42, ["python3", "worker.py"]),
            60: (0, ["/bin/sh", "-s"]),
            61: (60, ["/bin/sh", "/verifier/test.sh"]),
            70: (0, ["/bin/sh", "/tests/test.sh"]),
        },
        envs={
            41: sorted(marked),
            42: sorted(marked),
            43: [],  # its own environment, but a child of this attempt's tests
            61: ["BENCHFLOW_VERIFIER_ATTEMPT=" + other],
            70: ["BENCHFLOW_VERIFIER_ATTEMPT=" + mine + "x"],
        },
    )

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)
        return SimpleNamespace(rewards={"reward": 1.0})

    receipt = (f"/run/runtime/{mine}", "main")
    verifier = SimpleNamespace(verify=verify, execution_receipt=receipt)
    rewards, *_ = await _verify_rollout(
        sandbox, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert sandbox.alive() == {1, 7, 40, 60, 61, 70}
