"""Guard CONFIRM-lockdown-6 at bd23e190 through the grading boundary."""

import subprocess
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.task import keep_services
from benchflow.task.keep_services import KeepService, KeptServices
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize("unavailable", [False, True], ids=["mutable", "no-python"])
async def test_mutable_or_unknown_ancestry_refuses_grading(
    tmp_path, monkeypatch, unavailable
):
    workspace = tmp_path / "job" / "submission"
    workspace.mkdir(parents=True)
    workspace.parent.chmod(0o777)
    (workspace / "answer.txt").write_text("dummy answer\n")

    def execute(command, **kwargs):
        if "Cannot protect workspace resolution" in command:
            if unavailable:
                return ExecResult(
                    stdout="", stderr="python3: not found", return_code=127
                )
            result = subprocess.run(
                ["sh", "-c", command], capture_output=True, text=True
            )
            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        return ExecResult(stdout="", stderr="", return_code=0)

    for name in ("_freeze_workspace", "_reclaim_disk"):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(lockdown, "_record_kept", lambda *a, **kw: None)
    monkeypatch.setattr(
        keep_services,
        "trial_keep",
        lambda *a: KeptServices((KeepService(5000),), "the task"),
    )
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        lockdown,
        "_kill_sandbox_user_procs_keeping",
        AsyncMock(return_value=lockdown.KeepingKill(0, (), 0.0)),
    )
    build = AsyncMock(return_value={})
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    with pytest.raises(RuntimeError, match="workspace ancestors"):
        await lockdown.harden_before_verify(
            _make_env(execute),
            _make_task(),
            sandbox_user="agent",
            workspace=str(workspace),
        )
    build.assert_not_awaited()
    assert (workspace / "answer.txt").read_text() == "dummy answer\n"
