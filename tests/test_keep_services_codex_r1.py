"""K2's fixes for the first cross-model review.

The kills run against K1's file-backed stand-in ``/proc`` (``id``, ``sleep``,
``kill``, ``setpriv``, ``su`` and ``readlink`` are shell functions or act on
the stand-in only), so no real process is ever signalled. Each test fails on
9accdaab, the reviewed tip, for the behaviour it names.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import kill_record
from benchflow.rollout import _review
from benchflow.rollout._setup import _verify_rollout
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.lockdown import (
    _kill_sandbox_user_procs,
    _kill_sandbox_user_procs_keeping,
    end_kept_services,
    harden_before_verify,
    release_kept_services,
)
from tests.test_keep_services_kill import _env_on, _NetProc, _task
from tests.test_keep_services_verify import _Planes, _verifier
from tests.test_verifier_process_quiescence import _LIVE, UID, _FakeProc

# A setuid program the agent started: its real uid is the agent's, its
# effective and saved uids root's.
_SETUID_AGENT = f"State:\tS (sleeping)\nUid:\t{UID}\t0\t0\t0\nThreads:\t1\n"


class _Helpers(_FakeProc):
    """K1's stand-in, with ``setpriv`` and ``su`` replaced by *overrides*
    (shell function definitions that come after the stand-in's own)."""

    def __init__(self, tmp_path, overrides: str):
        super().__init__(tmp_path)
        self._overrides = overrides

    def _prelude(self):
        return super()._prelude() + self._overrides + "\n"


# setpriv says it can drop to a uid, but the drop fails (no CAP_SETUID).
_SETPRIV_FAILS = (
    'setpriv() { [ "$1" = --help ] && { echo "--reuid"; return 0; }; return 1; }'
)
_SU_FAILS = "su() { return 1; }"


# Helpers that run the kill as the user


@pytest.mark.asyncio
async def test_a_setpriv_that_cannot_drop_falls_back_to_su(tmp_path):
    """setpriv advertises --reuid but fails to run anything as the user: the
    kill uses su, so kill(-1) as the user still reaches a pid hopper."""
    proc = _Helpers(tmp_path, _SETPRIV_FAILS)
    proc.add_hopper()
    await _kill_sandbox_user_procs(proc, "agent")
    assert proc.hoppers() == []
    assert "KILL -1" in proc.signalled() or "STOP -1" in proc.signalled()


@pytest.mark.asyncio
async def test_no_helper_that_runs_as_the_user_fails_the_kill(tmp_path):
    """Neither setpriv nor su runs a command as the user: the kill fails
    rather than report a kill(-1) it never sent (a hopper would live on)."""
    proc = _Helpers(tmp_path, _SETPRIV_FAILS + "\n" + _SU_FAILS)
    proc.add_hopper()
    with pytest.raises(RuntimeError, match="rc=2"):
        await _kill_sandbox_user_procs(proc, "agent")


@pytest.mark.asyncio
async def test_a_broadcast_that_stops_working_fails_the_kill(tmp_path):
    """setpriv passes the first check and fails afterwards: the kill(-1)
    that did not run as the user fails the kill (exit 2)."""
    count = tmp_path / "setpriv-calls"
    proc = _Helpers(
        tmp_path,
        'setpriv() { [ "$1" = --help ] && { echo "--reuid"; return 0; }; '
        f"n=$(cat {count} 2>/dev/null || echo 0); echo $((n + 1)) > {count}; "
        '[ "$n" = 0 ] || return 1; '
        'while [ $# -gt 0 ]; do [ "$1" = -c ] && { eval "$2"; return; }; shift; done; }\n'
        + _SU_FAILS,
    )
    proc.add(_LIVE)
    with pytest.raises(RuntimeError, match="rc=2"):
        await _kill_sandbox_user_procs(proc, "agent")


# Setuid programs the agent started


@pytest.mark.asyncio
async def test_the_kill_counts_and_ends_a_setuid_program_the_agent_started(tmp_path):
    """A process whose real uid is the agent's (a setuid program it started)
    is one kill(-1) reaches, so the walk lists it too: it is killed,
    counted, and the check that nothing is left waits for it."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_SETUID_AGENT, cmdline="setuid-helper")
    killed = await _kill_sandbox_user_procs(proc, "agent")
    assert str(pid) not in proc.live_pids()
    assert killed == 1


@pytest.mark.asyncio
async def test_a_setuid_program_that_does_not_die_fails_the_kill(tmp_path):
    """A setuid program of the agent's that outlives SIGKILL (stuck in the
    kernel) is a survivor, so the folders are not opened after it."""
    proc = _FakeProc(tmp_path)
    proc.add(_SETUID_AGENT, stubborn=True, cmdline="stuck-helper")
    with pytest.raises(RuntimeError, match="writers remain"):
        await end_kept_services(proc, "agent", (("/logs/agent", "0:0", "777"),))


# Each round's frozen pids before kill -1


@pytest.mark.asyncio
async def test_each_round_signals_its_frozen_pids_before_kill_minus_one(tmp_path):
    """The walk's own pids get SIGKILL (and SIGCONT) while they are still
    frozen, before kill(-1) as the user, which would let them exit and their
    pids go to other processes first."""
    proc = _FakeProc(tmp_path)
    pid = proc.add(_LIVE)
    await _kill_sandbox_user_procs(proc, "agent")
    sent = proc.signalled()
    assert sent.index(f"KILL {pid}") < sent.index("KILL -1"), sent
    assert sent.index(f"CONT {pid}") < sent.index("CONT -1"), sent


# A missing or cut receipt


class _Silent:
    """A backend that reports exit status 0 and loses the output."""

    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_a_keeping_kill_with_no_receipt_is_not_a_kill_that_kept_nothing():
    with pytest.raises(RuntimeError, match="receipt"):
        await _kill_sandbox_user_procs_keeping(_Silent(), "agent", (5000,), wait=True)


@pytest.mark.asyncio
async def test_a_keeping_kill_whose_kept_rows_were_cut_fails():
    """The kept count does not match the kept rows: some were lost."""

    class _Cut(_Silent):
        async def exec(self, command, **kwargs):
            return SimpleNamespace(
                return_code=0,
                stdout="BF waited 0\nBF kept count 1\nBF killed 2 end\n",
                stderr="",
            )

    with pytest.raises(RuntimeError, match="receipt"):
        await _kill_sandbox_user_procs_keeping(_Cut(), "agent", (5000,), wait=True)


@pytest.mark.asyncio
async def test_a_release_with_no_receipt_fails():
    with pytest.raises(RuntimeError, match="receipt"):
        await release_kept_services(_Silent(), "agent", (4242,), ("/tests",))


@pytest.mark.asyncio
async def test_the_end_of_kept_services_with_no_receipt_opens_nothing():
    env = _Silent()
    with pytest.raises(RuntimeError, match="receipt"):
        await end_kept_services(env, "agent", (("/logs/agent", "0:0", "777"),))
    assert not any("chmod 777" in c for c in env.commands), env.commands


@pytest.mark.asyncio
async def test_the_keeping_kill_prints_the_kept_count(tmp_path):
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="python3 app.py")
    proc.hold(pid, proc.listen(5000))
    outcome = await _kill_sandbox_user_procs_keeping(
        _AsyncProc(proc), "agent", (5000,), wait=True
    )
    assert [k.pid for k in outcome.kept] == [pid]


class _AsyncProc:
    def __init__(self, proc):
        self._proc = proc

    async def exec(self, command, **kwargs):
        return self._proc.run(command)


# A listener in a thread's own descriptor table


@pytest.mark.asyncio
async def test_a_listener_in_a_threads_own_descriptor_table_is_kept(tmp_path):
    """A worker thread that unshared its descriptor table holds the listener
    only under task/<tid>/fd: the service is kept, not killed."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="c-service")
    inode = proc.listen(5000)
    fd = proc.root / str(pid) / "task" / "777" / "fd"
    fd.mkdir(parents=True)
    (fd / "4").symlink_to(f"socket:[{inode}]")
    other = proc.add(_LIVE)
    record: dict = {}
    with kill_record.recording_into(record):
        await harden_before_verify(
            _env_on(proc), _task([{"port": 5000}]), "agent", workspace="/app"
        )
    assert str(pid) in proc.live_pids()
    assert str(other) not in proc.live_pids()
    assert [r["pid"] for r in record["kept_services"]] == [pid]
    assert record["kept_services_no_listener"] == []


# The record of the turn-end kill


@pytest.mark.asyncio
async def test_the_record_keeps_the_turn_end_kill_and_a_service_lost_after_it(
    tmp_path, monkeypatch
):
    """The turn-end kill waited for, kept and killed; the kill before grading
    finds nothing to kill and the kept service gone. The record still shows
    the turn-end kill and names the service lost between the two."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="python3 app.py")
    proc.hold(pid, proc.listen(5000))
    launcher = proc.add(_LIVE, cmdline="launcher")

    async def idle(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_review, "capture_task_evidence", idle)
    task = _task([{"port": 5000}])
    task.config.verifier.submission_files = []
    task.config.artifacts = []
    rollout = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=object(),
        _config=SimpleNamespace(purpose="task", sandbox_user="agent"),
        _env=proc,
        _agent_env={},
        _planes=DefaultRolloutPlanes(),
        _task=task,
        _agent_cwd="/app",
        disconnect=idle,
        _require_rollout_dir=lambda: tmp_path,
    )
    await _review.capture_terminal_workspace(rollout)
    assert str(launcher) not in proc.live_pids()
    assert str(pid) in proc.live_pids()
    # The kept service is gone before the kill before grading.
    import shutil

    shutil.rmtree(proc.root / str(pid))

    record: dict = {}
    with kill_record.recording_into(record):
        await harden_before_verify(_env_on(proc), task, "agent", workspace="/app")
    assert record["killed"] == 0
    assert record["kept_services"] == []
    turn_end = record["kept_services_turn_end"]
    assert turn_end["killed"] == 1
    assert turn_end["kept"] == [{"port": 5000, "pid": pid}]
    assert record["kept_services_lost_before_grading"] == [pid]


# The scope while kept services run


class _ScopedEnv:
    """An env with the sandbox's services_kept scope."""

    _persistent_env: ClassVar[dict] = {}

    # getattr: the module still imports on the reviewed tip, where the scope
    # does not exist and the test below fails on what it checks.
    services_kept = getattr(BaseSandbox, "services_kept", None)
    _services_are_kept = getattr(BaseSandbox, "_services_are_kept", False)


@pytest.mark.asyncio
async def test_the_verifier_runs_inside_the_services_kept_scope(tmp_path):
    """From the verifier's start to the end of the kept services, the
    sandbox knows they run (so exec and copy-out act on it)."""
    planes = _Planes()
    planes.order = []
    env = _ScopedEnv()
    seen: list = []
    make = planes.verifier

    def verifier(**kwargs):
        v = make(**kwargs)
        inner = v.verify

        async def verify():
            seen.append(("verify", env._services_are_kept))
            return await inner()

        v.verify = verify
        return v

    planes.verifier = verifier
    end = planes.end_kept_services

    async def end_kept(*args, **kwargs):
        seen.append(("end", env._services_are_kept))
        return await end(*args, **kwargs)

    planes.end_kept_services = end_kept
    task = SimpleNamespace(
        name="svc",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                timeout_sec=30, type="test-script", reward_range=None
            )
        ),
    )
    rollout_paths = SimpleNamespace(
        verifier_dir=tmp_path / "verifier", rollout_dir=tmp_path
    )
    _rewards, error, _ = await _verify_rollout(
        env, task, rollout_paths, {}, planes, sandbox_user="agent", workspace="/app"
    )
    assert error is None, error
    assert seen == [("verify", True), ("end", True)]
    assert env._services_are_kept is False
    assert (
        json.loads((tmp_path / "kill-before-grading.json").read_text())[
            "kept_services_ended"
        ]
        == 2
    )


@pytest.mark.asyncio
async def test_docker_copy_out_changes_no_owner_while_services_are_kept():
    """download_dir and download_file chown what they copy to the
    host user first; while services are kept that would hand /logs/verifier
    (and the reward) to a uid that can be the agent user's."""
    calls: list = []

    async def exec_(command, **kwargs):
        calls.append(command)

    kept = SimpleNamespace(_services_are_kept=True, exec=exec_)
    await DockerSandbox._chown_to_host_user(kept, "/logs/verifier", recursive=True)
    assert calls == []
    free = SimpleNamespace(_services_are_kept=False, exec=exec_)
    await DockerSandbox._chown_to_host_user(free, "/logs/verifier", recursive=True)
    assert len(calls) == 1 and calls[0].startswith("chown -R ")


@pytest.mark.asyncio
async def test_the_persistent_env_reaches_the_test_through_the_root_file(tmp_path):
    """The sandbox's own env (a non-compose task's [environment]
    env) goes into the root-only file with the verifier's, so no exec while
    services are kept needs the backend's wrapper, which writes its file
    under /tmp."""
    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._sandbox._persistent_env = {"LANG": "C"}
    test_exec = verifier._sandbox.exec.side_effect
    calls: list = []
    uploads: list = []

    async def exec_(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        calls.append((cmd, kwargs.get("env")))
        return await test_exec(*args, **kwargs)

    async def upload_file(source_path, target_path, *, mode=None, **_kwargs):
        uploads.append((target_path, Path(source_path).read_text()))

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)
    verifier._sandbox.upload_file = AsyncMock(side_effect=upload_file)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))

    verifier.before_test = release
    await verifier._verify_test_script()
    ((target, body),) = uploads
    assert "export LANG=C\n" in body
    test_cmd = order[-1][1]
    assert test_cmd.startswith(f"set -- benchflow-env-pending; if . {target};"), (
        test_cmd
    )
    assert all(env is None for _cmd, env in calls), calls


def test_exec_adds_no_persistent_env_while_services_are_kept():
    """Every backend's exec gets its env from _merge_env: inside the scope it
    is only what the caller passed."""
    sandbox = SimpleNamespace(_persistent_env={"LANG": "C"})
    with contextlib.ExitStack() as stack:
        stack.enter_context(BaseSandbox.services_kept(sandbox))
        sandbox._services_are_kept = BaseSandbox._services_are_kept.fget(sandbox)
        assert BaseSandbox._merge_env(sandbox, None) is None
        assert BaseSandbox._merge_env(sandbox, {"A": "1"}) == {"A": "1"}
    sandbox._services_are_kept = BaseSandbox._services_are_kept.fget(sandbox)
    assert BaseSandbox._merge_env(sandbox, None) == {"LANG": "C"}
