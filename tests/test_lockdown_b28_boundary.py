"""Guard CONFIRM-lockdown-5 at bd23e190 through the grading boundary."""

from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize("char", ["\u00a0", "\x0b", "\r", "\x85", "\u2028", "\u2029"])
@pytest.mark.parametrize("backend", ["overlay", "ext4", "tmpfs"])
@pytest.mark.parametrize("separate", [False, True])
async def test_grading_retains_literal_mount_aliases(
    monkeypatch, char, backend, separate
):
    alias = "/opt/plugin" + char + "view"
    mounts = (
        "1 0 0:1 / / rw - overlay dummy rw\n"
        f"2 1 0:2 / /app/plugins rw - {backend} dummy rw\n"
        f"3 1 0:2 / {alias} rw - {backend} dummy rw\n"
    )

    def execute(command, **kwargs):
        return ExecResult(
            stdout=mounts + "# end\n\0" if "/proc/self/mountinfo" in command else "",
            stderr="",
            return_code=0,
        )

    for name in ("_kill_sandbox_user_procs", "_freeze_workspace", "_reclaim_disk"):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    build = AsyncMock(return_value={})
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    await lockdown.harden_before_verify(
        _make_env(execute),
        _make_task(),
        sandbox_user=None if separate else "agent",
        workspace="/app",
        agent_paths=("/app",) if separate else None,
    )
    captured = build.call_args.kwargs["blocked"]
    assert isinstance(captured, lockdown.PathExclusions)
    assert ("0:2", "/") in captured.coordinates
    assert alias in captured.paths()
