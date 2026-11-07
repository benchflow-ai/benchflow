"""Guards K2's 3af6d457 confirmation fixes (-3 and -7).

Signals and privilege helpers operate on the file-backed process table from
the K1/K2 tests. No real process is signalled by these tests.
"""

from __future__ import annotations

import shlex

import pytest

from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox.lockdown import (
    _kill_sandbox_user_procs,
    _kill_sandbox_user_procs_keeping,
    end_kept_services,
    release_kept_services,
)
from tests.test_keep_services_kill import _Async, _logs, _NetProc
from tests.test_verifier_process_quiescence import _LIVE, _FakeProc


class _EnvProc(_NetProc):
    _services_are_kept = BaseSandbox._services_are_kept

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self._persistent_env = {"LANG": "C"}
        self.environments = []

    async def exec(self, command, **kwargs):
        self.environments.append(BaseSandbox._merge_env(self, kwargs.get("env")))
        return await super().exec(command, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("keeping", [False, True])
async def test_first_quiesce_suppresses_persistent_environment(tmp_path, keeping):
    """Guards the fix at 3af6d457, including the turn-end kill primitive."""
    proc = _EnvProc(tmp_path)
    proc.add(_LIVE)
    if keeping:
        # No listener is needed to expose the first root exec's environment.
        await _kill_sandbox_user_procs_keeping(proc, "agent", (5000,), wait=False)
    else:
        await _kill_sandbox_user_procs(proc, "agent")
    assert proc.environments == [None]
    assert BaseSandbox._merge_env(proc, None) == {"LANG": "C"}


class _DeniedBroadcast(_NetProc):
    def __init__(self, tmp_path, helper, denied):
        super().__init__(tmp_path)
        self.helper = helper
        self.denied = denied

    def _prelude(self):
        base = super()._prelude().replace("kill() {", "allowed_kill() {", 1)
        base = base.replace(
            'eval "$2"; AS_USER=; return;',
            'eval "$2"; helper_rc=$?; AS_USER=; return "$helper_rc";',
        )
        if self.helper == "su":
            base += "\nsetpriv() { return 1; }\n"
        return base + (
            '\nkill() { if [ "$2" = -1 ] && '
            f'[ "$1" = -{self.denied} ]; then return 1; fi; '
            'allowed_kill "$@"; }\n'
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["setpriv", "su"])
@pytest.mark.parametrize("phase", ["plain", "keeping", "finalize"])
async def test_denied_broadcast_fails_even_when_uid_drop_succeeds(
    tmp_path, helper, phase
):
    """EPERM cannot certify quiescence."""
    proc = _DeniedBroadcast(tmp_path, helper, "STOP")
    pid = proc.add(_LIVE)
    proc.hold(pid, proc.listen(5000))
    proc.add_hopper()
    logs = _logs(tmp_path)
    env = _Async(proc)
    with pytest.raises(RuntimeError, match="rc=2"):
        if phase == "plain":
            await _kill_sandbox_user_procs(env, "agent")
        elif phase == "keeping":
            await _kill_sandbox_user_procs_keeping(env, "agent", (5000,), wait=False)
        else:
            await end_kept_services(env, "agent", ((str(logs), "0:0", "777"),))
    assert logs.stat().st_mode & 0o777 == 0o755


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["setpriv", "su"])
@pytest.mark.parametrize("denied", ["CONT", "KILL"])
async def test_later_denied_broadcast_fails(tmp_path, helper, denied):
    """Guards the fix at 3af6d457 on broadcasts after the first freeze."""
    proc = _DeniedBroadcast(tmp_path, helper, denied)
    proc.add(_LIVE)
    with pytest.raises(RuntimeError, match="rc=2"):
        await _kill_sandbox_user_procs(_Async(proc), "agent")


@pytest.mark.asyncio
@pytest.mark.parametrize("fd_state", ["empty", "missing", "unreadable", "broken"])
async def test_worker_listener_selection_through_release(
    tmp_path, fd_state, monkeypatch
):
    """Empty is inspectable; unknown is unsafe."""
    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE, cmdline="worker-service")
    proc.place(pid, tasks=(pid, pid + 1000))
    main_fd = proc.root / str(pid) / "task" / str(pid) / "fd"
    (main_fd / "3").unlink()
    if fd_state == "missing":
        main_fd.rmdir()
    elif fd_state == "broken":
        (main_fd / "4").write_text("not a readable proc link")
    elif fd_state == "unreadable":
        # 7a2387ac: chmod(0) does not deny container root. Model the procfs
        # listing failure at the same shell boundary as the fake readlink,
        # so both root and ordinary users exercise the unknown-fd branch.
        prelude = proc._prelude()
        denied = shlex.quote(str(main_fd))
        marker = tmp_path / "listing-denied"
        monkeypatch.setattr(
            proc,
            "_prelude",
            lambda: (
                prelude
                + f'\nls() {{ if [ "$1" = -A ] && [ "$2" = {denied} ]; then '
                + f': > {shlex.quote(str(marker))}; return 2; fi; command ls "$@"; }}\n'
            ),
        )
    worker_fd = proc.root / str(pid) / "task" / str(pid + 1000) / "fd"
    (worker_fd / "3").unlink()
    (worker_fd / "4").symlink_to(f"socket:[{proc.listen(5000)}]")
    outcome = await _kill_sandbox_user_procs_keeping(
        _Async(proc), "agent", (5000,), wait=False
    )
    assert [p.pid for p in outcome.kept] == [pid]
    released = await release_kept_services(
        _Async(proc), "agent", (pid,), (), logs_dir=str(_logs(tmp_path))
    )
    if fd_state == "unreadable":
        assert marker.exists(), "the fixture must reach the denied listing"
    if fd_state == "empty":
        assert released.released == 1 and released.dropped == ()
        assert str(pid) in proc.live_pids() and not proc.stopped(pid)
    else:
        assert released.released == 0 and released.dropped
        assert str(pid) not in proc.live_pids()


@pytest.mark.asyncio
async def test_broadcast_with_no_other_process_is_successful(tmp_path):
    """Guards the fix at 3af6d457 without treating ESRCH as an error."""
    assert await _kill_sandbox_user_procs(_FakeProc(tmp_path), "agent") == 0
