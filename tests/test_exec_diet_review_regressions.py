"""Guards two PY8 U12 regressions present at commit 71f4d05f.

The same faults reach the original separate calls and U12's start helper;
no sandbox command is executed by these tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.task import verifier_core
from benchflow.task.verifier_core import Verifier
from benchflow.task.verifier_errors import (
    VerifierExecutionError,
    VerifierOutputParseError,
)
from tests.test_exec_diet_units import _daytona, _SyncProcess


@pytest.mark.parametrize("fault", ["lost", "nonzero", "slow"])
@pytest.mark.parametrize("failures", [0, 1, 2, 3])
async def test_start_keeps_the_separate_calls_retry_and_deadline_contract(
    monkeypatch, fault, failures
):
    """Guards 71f4d05f: a lost first exec is attempt one, not a free try."""
    chmod_command = "chmod +x /tests/test.sh"

    async def run(separate):
        calls = []
        waits = []
        attempts = 0

        async def sleep(delay):
            waits.append(delay)

        async def execute(command, *, user, service, timeout_sec):
            nonlocal attempts
            calls.append((user, service, timeout_sec))
            if chmod_command in command:
                attempts += 1
                if fault == "slow" and timeout_sec < 60:
                    raise TimeoutError("chmod needs 12 seconds")
                if attempts <= failures:
                    if fault == "nonzero":
                        return SimpleNamespace(return_code=7, stdout="")
                    raise ConnectionResetError("lost setup exec")
            return SimpleNamespace(return_code=0, stdout="")

        monkeypatch.setattr(verifier_core.asyncio, "sleep", sleep)
        verifier = Verifier(MagicMock(), MagicMock(), SimpleNamespace(exec=execute))
        verifier._write_execution_receipt = True
        verifier.execution_receipt = ("/run/runtime/verifier-started-old", "main")
        try:
            if separate:
                await verifier._setup_step(
                    "chmod",
                    lambda: execute(
                        chmod_command, user="root", service="main", timeout_sec=60
                    ),
                )
                receipt = await verifier._prepare_execution_receipt("main")
            else:
                receipt = await verifier._chmod_and_prepare_receipt(
                    chmod_command, "main"
                )
        except (VerifierOutputParseError, VerifierExecutionError) as exc:
            if fault != "nonzero":
                assert isinstance(exc, VerifierExecutionError)
                assert isinstance(exc.__cause__, ConnectionResetError)
            outcome = (type(exc).__name__, str(exc), verifier.execution_receipt)
        else:
            outcome = (
                "ok",
                receipt is not None,
                verifier.execution_receipt is not None,
            )
        return outcome, calls, waits, attempts

    base = await run(separate=True)
    current = await run(separate=False)
    assert current == base
    assert base[3] == min(failures + 1, 3)
    assert base[2] == [5.0, 15.0][: min(failures, 2)]
    assert base[1] == [("root", "main", 60)] * base[3] + (
        [("root", "main", 10)] if failures < 3 else []
    )
    if failures == 3 and fault != "nonzero":
        assert base[0][1].startswith("Verifier setup failed: chmod did not run:")


class _DelayedProcess(_SyncProcess):
    async def get_session_command(self, session_id, command_id):
        self.requests.append("status")
        return SimpleNamespace(id=command_id, exit_code=124)

    async def get_session_command_logs(self, session_id, command_id):
        self.requests.append("logs")
        return SimpleNamespace(stdout="early", stderr="")


@pytest.mark.parametrize("sync_limit", [None, 0, 30])
async def test_delayed_delivery_preserves_a_completed_exit_124(monkeypatch, sync_limit):
    """Guards 71f4d05f: HTTP latency is not evidence of command timeout."""
    import asyncio

    from benchflow.sandbox import daytona

    if sync_limit is not None:
        monkeypatch.setattr(daytona, "_DAYTONA_SYNC_EXEC_MAX_SEC", sync_limit)

    async def execute(timeout):
        # The command exited immediately; only delivery is delayed.
        await asyncio.sleep(timeout + 0.1)
        return SimpleNamespace(cmd_id="c", exit_code=124, stdout="early", stderr="")

    process = _DelayedProcess(execute)
    result = await _daytona(process).exec("printf early; exit 124", timeout_sec=1)
    assert (result.return_code, result.stdout, result.stderr) == (124, "early", "")
    assert sum(r.startswith("execute:") for r in process.requests) == 1
    if sync_limit in (None, 0):
        assert process.requests == ["create", "execute:True", "status", "logs"]


async def test_delayed_sync_transport_recovery_keeps_completed_124(monkeypatch):
    """Guards 71f4d05f's second 124 inference, after a failed request."""
    import asyncio

    from benchflow.sandbox import daytona

    monkeypatch.setattr(daytona, "_DAYTONA_SYNC_EXEC_MAX_SEC", 30)

    async def execute(timeout):
        await asyncio.sleep(timeout + 0.1)
        raise TimeoutError("response lost")

    process = _DelayedProcess(
        execute, SimpleNamespace(commands=[SimpleNamespace(id="c", exit_code=124)])
    )
    result = await _daytona(process).exec("printf early; exit 124", timeout_sec=1)
    assert (result.return_code, result.stdout, result.stderr) == (124, "early", "")
    assert process.requests == ["create", "execute:False", "session", "logs"]


@pytest.mark.parametrize("delay", [0, 1.1])
async def test_default_transport_errors_do_not_recover_or_replay(delay):
    """Guards 71f4d05f: keep the base transport contract even after the budget."""
    import asyncio

    from benchflow.diagnostics import TransientSandboxTransportError

    async def execute(timeout):
        await asyncio.sleep(delay)
        raise TimeoutError("response lost")

    process = _DelayedProcess(execute)
    with pytest.raises(TransientSandboxTransportError):
        await _daytona(process).exec("true", timeout_sec=1)
    assert process.requests == ["create", "execute:True"]
