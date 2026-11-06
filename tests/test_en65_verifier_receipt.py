"""Guard receipt parsing in scoring fix a2ed23f7c."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout._setup import _TestEndUnknown, _verifier_finished
from benchflow.sandbox._base import ExecResult
from tests.test_verifier_lost_exec import _score, _setup


@pytest.mark.parametrize("response", ["", "ended ", "ended 256", "ended 999", "noise"])
async def test_invalid_receipt_retries_then_stays_unknown(response):
    """Guards a2ed23f7c: truncated/malformed receipt is not proof of running tests."""
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(stdout=response, return_code=0))
    )
    verifier = SimpleNamespace(
        execution_receipt=("/run/benchflow/receipt", "main"),
        test_command_finishable=True,
    )
    with pytest.raises(_TestEndUnknown, match="incomplete or invalid"):
        await _verifier_finished(sandbox, verifier)
    assert sandbox.exec.await_count == 2


@pytest.mark.parametrize(
    "response,expected", [("running", None), ("ended 0", 0), ("ended 255", 255)]
)
async def test_valid_receipt_preserves_running_and_exit_states(response, expected):
    """Guards a2ed23f7c: known-running and valid exit statuses remain distinct."""
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(stdout=response, return_code=0))
    )
    verifier = SimpleNamespace(
        execution_receipt=("/run/benchflow/receipt", "main"),
        test_command_finishable=True,
    )
    assert await _verifier_finished(sandbox, verifier) == expected
    assert sandbox.exec.await_count == 1


async def test_truncated_receipt_cannot_score_a_lost_exec_timeout(tmp_path):
    """Guards a2ed23f7c through the real verifier's lost-exec scoring path."""
    task, paths, sandbox = _setup(
        tmp_path, "echo 1 > /logs/verifier/reward.txt\n", lose_exec=True
    )
    original_exec = sandbox.exec

    async def truncated_probe(command, *args, **kwargs):
        if "printf 'ended '" in command:
            return ExecResult(stdout="ended ", return_code=0)
        return await original_exec(command, *args, **kwargs)

    sandbox.exec = truncated_probe
    rewards, error, diagnostic = await _score(task, paths, sandbox)
    assert rewards is None
    assert "did not answer whether the test command had ended" in error
    assert diagnostic is not None and diagnostic.scored is False
