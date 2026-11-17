"""The verifier outputs folder's owner rule holds the folder's real
state: once it is root's, the host never writes test-stdout.txt there.

On an Alpine image the rule's one command failed after its chown (BusyBox's
find has no -uid), and the host then wrote into a folder it no longer owned:
PermissionError, a verifier crash, in every integrity mode. A process of
another user still writing in the folder (the agent's, alive through grading
in audit and off) could also keep the sweep from finishing: that is the
verifier's failure (verifier_failed), never infrastructure, never retried."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox.protocol import ExecTimeoutError
from benchflow.task.verifier_core import (
    _OUTPUTS_CLOSED,
    _OUTPUTS_HELD,
    Verifier,
    VerifierOutputsHeldError,
)
from benchflow.task.verifier_errors import VerifierSetupError


def _verifier(stdout: str | None, return_code: int, stderr: str = "") -> Verifier:
    from benchflow.task import TaskConfig

    task = MagicMock()
    task.config = TaskConfig.model_validate({"verifier": {}})
    sandbox = MagicMock()
    sandbox.exec = AsyncMock(
        return_value=SimpleNamespace(
            stdout=stdout, stderr=stderr, return_code=return_code, exit_code=return_code
        )
    )
    verifier = Verifier(task=task, rollout_paths=MagicMock(), sandbox=sandbox)
    verifier._verifier_uid_read = "0"
    verifier._host_writes_outputs = lambda: False  # type: ignore[method-assign]
    return verifier


def _command(verifier: Verifier) -> str:
    return str(verifier._sandbox.exec.await_args.args[0])


@pytest.mark.asyncio
async def test_the_sweep_is_one_busybox_takes_and_takes_entries_from_their_owner():
    verifier = _verifier(f"{_OUTPUTS_CLOSED}\n", 0)
    await verifier._close_output_dir("main", mounted=True)
    command = _command(verifier)
    assert "! -user 0" in command and "-uid" not in command, command
    for step in ("-prune", "chown -R -h 0:0", "chmod -R go-w", "rm -rf --"):
        assert step in command, (step, command)
    assert verifier._closed_to_host is True


@pytest.mark.asyncio
async def test_a_sweep_that_fails_after_the_close_is_the_verifiers_failure():
    # Before: rc 1 left _closed_to_host False with the folder already root's,
    # and the host's own test-stdout.txt write raised PermissionError.
    verifier = _verifier(f"{_OUTPUTS_CLOSED}\n", 1, "rm: Directory not empty")
    with pytest.raises(VerifierOutputsHeldError, match=_OUTPUTS_HELD) as caught:
        await verifier._close_output_dir("main", mounted=True)
    assert not isinstance(caught.value, VerifierSetupError)


@pytest.mark.asyncio
async def test_a_close_that_does_not_finish_is_the_verifiers_failure():
    verifier = _verifier(None, 0)
    verifier._sandbox.exec = AsyncMock(
        side_effect=ExecTimeoutError("command timed out after 60s")
    )
    with pytest.raises(VerifierOutputsHeldError, match=_OUTPUTS_HELD) as caught:
        await verifier._close_output_dir("main", mounted=True)
    assert "timed out" not in str(caught.value)


@pytest.mark.asyncio
async def test_a_chown_that_failed_fails_closed():
    """The folder never became root's: test.sh does not run with it open to
    the agent's processes (read-back #6, the lead's 20:31Z ruling),
    the same ending as a sweep that failed, and the reward's provenance
    records why."""
    from benchflow.task.verifier_core import VerifierOutputsHeldError

    verifier = _verifier(None, 1, "chown: Operation not permitted")
    with pytest.raises(VerifierOutputsHeldError, match="could not be made root's"):
        await verifier._close_output_dir("main", mounted=True)
    assert verifier._closed_to_host is False
    assert verifier.outputs_closed is False


@pytest.mark.asyncio
async def test_a_folder_that_is_not_the_hosts_mount_is_not_closed_to_it():
    verifier = _verifier(f"{_OUTPUTS_CLOSED}\n", 0)
    await verifier._close_output_dir("main", mounted=False)
    assert verifier._closed_to_host is False


@pytest.mark.parametrize(
    "why", ["(rc=1: rm: Directory not empty)", "(the close did not finish in 60 s)"]
)
@pytest.mark.asyncio
async def test_the_trial_ends_verifier_failed_and_is_never_retried(tmp_path, why):
    from benchflow._utils.scoring import VERIFIER_FAILED, classify_verifier_error
    from benchflow.evaluation import RetryConfig
    from benchflow.sdk import SDK

    task = MagicMock()
    task.config.verifier.timeout_sec = 5
    task.config.verifier.env = None
    task.config.verifier.reward_range = None
    tp = MagicMock()
    tp.verifier_dir = tmp_path / "verifier"
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    verifier = MagicMock()
    verifier.verify = AsyncMock(
        side_effect=VerifierOutputsHeldError(f"{_OUTPUTS_HELD} {why}")
    )
    with (
        patch("benchflow.task.Verifier", return_value=verifier),
        patch(
            "benchflow.sandbox.lockdown.harden_before_verify", new_callable=AsyncMock
        ),
    ):
        rewards, verifier_error, _ = await SDK()._verify(env, task, tp, {})
    assert rewards is None
    assert verifier_error is not None and _OUTPUTS_HELD in verifier_error, (
        verifier_error
    )
    assert classify_verifier_error(verifier_error) == VERIFIER_FAILED, verifier_error
    assert not RetryConfig().should_retry_verifier_error(verifier_error)
    assert not RetryConfig().reruns_verifier_error_on_resume(verifier_error)
