"""A verifier command that never starts is retried once, then counts as infrastructure.

Guards #1136. After a long agent phase a Daytona exec session can lose the
verifier's command: no output for the whole budget. Only tasks with a
verifier-only recovery contract asked the command for a start receipt; every
other task waited out its whole budget, and then, finding no output, a second
whole budget, and ended as ``verifier timed out`` (a verifier timeout, whose
retry replays the solver). Now every script verifier's command writes a start
receipt: a receipt still missing a short grace after the command was issued
abandons the attempt, the command runs once more, and a second miss is
``verifier_wedge:`` (verifier infrastructure). A command that did start is
never replayed, however quiet it is. The Daytona command poll also bounds each
status request, so one request on a dead connection cannot hold a finished
command until the verifier budget ends.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.rollout import _setup
from benchflow.rollout._setup import _verify_rollout


class ReceiptVerifier:
    """Like the real Verifier: each verify() names a fresh receipt when it
    issues the test command, after ``upload_sec`` of setup; a run takes
    ``run_sec`` (or hangs) before it returns its rewards."""

    def __init__(self, runs, *, upload_sec: float = 0.0, run_sec: float = 0.0):
        self.runs = list(runs)  # per attempt: "hang" or a rewards dict
        self.upload_sec = upload_sec
        self.run_sec = run_sec
        self.execution_receipt = None
        self.attempts = 0
        self.issued_at: dict[str, float] = {}

    async def verify(self):
        self.attempts += 1
        self.execution_receipt = None
        await asyncio.sleep(self.upload_sec)
        path = f"/run/benchflow/verifier-started-{self.attempts}"
        self.execution_receipt = (path, "main")
        self.issued_at[path] = asyncio.get_running_loop().time()
        run = self.runs[min(self.attempts, len(self.runs)) - 1]
        await asyncio.sleep(3600 if run == "hang" else self.run_sec)
        return SimpleNamespace(rewards=run)


def _task(timeout_sec: float):
    return SimpleNamespace(
        name="receipt-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=timeout_sec, reward_range=None, env={})
        ),
    )


def _planes(verifier, seen: list[dict] | None = None):
    def make(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        return verifier

    return SimpleNamespace(harden_before_verify=AsyncMock(), verifier=make)


def _env(started_paths):
    """``cat <receipt>`` answers "started" only for commands that ran:
    a set of receipt paths, or a predicate on the path."""
    started = started_paths if callable(started_paths) else started_paths.__contains__

    async def exec_(command, **kwargs):
        if command.startswith("if [ -e "):  # the receipt probe
            path = command.split()[3]
            if started(path):
                return SimpleNamespace(stdout="started", stderr="", return_code=0)
            abandoned.add(path)
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        return SimpleNamespace(stdout="0\n", stderr="", return_code=0)

    abandoned: set[str] = set()

    return SimpleNamespace(exec=AsyncMock(side_effect=exec_), abandoned=abandoned)


@pytest.fixture
def short_grace(monkeypatch):
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.05)
    monkeypatch.setattr(_setup, "_VERIFIER_RECEIPT_POLL_SEC", 0.01)


async def test_every_script_verifier_is_asked_for_a_start_receipt(tmp_path):
    seen: list[dict] = []
    verifier = ReceiptVerifier([{"reward": 1.0}])
    rewards, error, _ = await _verify_rollout(
        _env(set()),
        _task(60),
        SimpleNamespace(verifier_dir=tmp_path / "verifier"),
        {},
        _planes(verifier, seen),
    )
    assert (rewards, error) == ({"reward": 1.0}, None)
    assert seen[0]["execution_receipt"] is True


@pytest.mark.usefixtures("short_grace")
async def test_a_command_that_never_starts_is_retried_once_then_a_wedge(tmp_path):
    verifier = ReceiptVerifier(["hang", "hang"])
    env = _env(set())  # neither command wrote its receipt
    t0 = time.monotonic()
    rewards, error, diagnostic = await _verify_rollout(
        env,
        _task(60),
        SimpleNamespace(verifier_dir=tmp_path / "v"),
        {},
        _planes(verifier),
    )
    assert time.monotonic() - t0 < 10  # not two 60 s budgets
    assert verifier.attempts == 2
    assert rewards is None and diagnostic is None
    assert error.startswith("verifier crashed: RuntimeError: verifier_wedge:")
    assert "did not start in two attempts" in error
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert env.abandoned == {
        "/run/benchflow/verifier-started-1",
        "/run/benchflow/verifier-started-2",
    }
    assert any("pkill" in c.args[0] for c in env.exec.await_args_list)


@pytest.mark.usefixtures("short_grace")
async def test_a_retried_command_that_starts_is_scored(tmp_path):
    verifier = ReceiptVerifier(["hang", {"reward": 1.0}])
    rewards, error, _ = await _verify_rollout(
        _env(set()),
        _task(60),
        SimpleNamespace(verifier_dir=tmp_path / "v"),
        {},
        _planes(verifier),
    )
    assert (rewards, error, verifier.attempts) == ({"reward": 1.0}, None, 2)


@pytest.mark.usefixtures("short_grace")
async def test_a_started_quiet_verifier_is_a_timeout_and_never_replayed(tmp_path):
    """Before, a timeout with an empty test-stdout.txt was replayed for a whole
    second budget; a receipt proves the command ran."""
    verifier = ReceiptVerifier(["hang"])
    env = _env({"/run/benchflow/verifier-started-1"})
    rewards, error, diagnostic = await _verify_rollout(
        env,
        _task(0.3),
        SimpleNamespace(verifier_dir=tmp_path / "v"),
        {},
        _planes(verifier),
    )
    assert verifier.attempts == 1
    assert rewards is None
    assert error == "verifier timed out after 0.3s"
    assert diagnostic is not None


@pytest.mark.usefixtures("short_grace")
async def test_the_grace_starts_when_the_command_is_issued(tmp_path):
    """The receipt is probed a grace after the command was issued, not a grace
    after the verifier started: a tests upload that ended just before the
    grace left the command no time to write its receipt (a false wedge)."""
    verifier = ReceiptVerifier([{"reward": 0.5}], upload_sec=0.04, run_sec=0.3)
    loop = asyncio.get_running_loop()

    def written(path: str) -> bool:  # the command writes it 20 ms after issue
        return loop.time() >= verifier.issued_at.get(path, float("inf")) + 0.02

    env = _env(written)
    rewards, error, _ = await _verify_rollout(
        env,
        _task(60),
        SimpleNamespace(verifier_dir=tmp_path / "v"),
        {},
        _planes(verifier),
    )
    assert (rewards, error, verifier.attempts) == ({"reward": 0.5}, None, 1)
    assert any(c.args[0].startswith("if [ -e ") for c in env.exec.await_args_list)
    assert env.abandoned == set()


@pytest.mark.usefixtures("short_grace")
async def test_a_recovery_task_leaves_the_first_lost_start_to_recovery(tmp_path):
    verifier = ReceiptVerifier(["hang", {"reward": 1.0}])
    rewards, error, _ = await _verify_rollout(
        _env(set()),
        _task(60),
        SimpleNamespace(verifier_dir=tmp_path / "v"),
        {},
        _planes(verifier),
        recovery_eligible=True,
    )
    assert verifier.attempts == 1
    assert rewards is None and "verifier_wedge:" in error


async def test_a_hung_status_request_is_retried_not_waited_out(monkeypatch):
    """The Daytona poll: a status request on a dead connection times out and
    is retried, instead of holding a finished command for the whole budget."""
    from benchflow.sandbox import daytona as daytona_module
    from benchflow.sandbox.daytona import DaytonaSandbox

    monkeypatch.setattr(daytona_module, "_DAYTONA_POLL_STATUS_TIMEOUT_SEC", 0.05)
    calls = {"status": 0}

    class Process:
        async def get_session_command(self, session_id, command_id):
            calls["status"] += 1
            if calls["status"] == 1:
                await asyncio.sleep(3600)  # a request that never returns
            return SimpleNamespace(id=command_id, exit_code=0)

        async def get_session_command_logs(self, session_id, command_id):
            return SimpleNamespace(stdout="done", stderr="")

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(process=Process())
    result = await asyncio.wait_for(
        sandbox._poll_response("session", "cmd", timeout_sec=900), timeout=10
    )
    assert (result.return_code, result.stdout, calls["status"]) == (0, "done", 2)


async def test_a_command_starting_after_the_probe_gave_up_runs_no_tests(
    tmp_path, monkeypatch
):
    """Real shell: the probe marks a missing receipt abandoned, and a test
    command the exec layer delivers only afterwards exits before test.sh, so
    it cannot race the retry on the verifier's outputs."""
    from types import SimpleNamespace as NS

    from benchflow.rollout._setup import _verifier_started
    from benchflow.task import verifier_core
    from benchflow.task.paths import RolloutPaths
    from benchflow.task.verifier_core import Verifier
    from benchflow.task.verifier_errors import RewardFileNotFoundError
    from tests.test_verifier_environment_checks import (
        _receipt_sandbox,
        _receipt_task,
    )

    task = _receipt_task(tmp_path)
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    sandbox = _receipt_sandbox(tmp_path, paths)
    (tmp_path / "sandbox-run-benchflow").mkdir()
    receipt = "/run/benchflow/verifier-started-late"
    # The probe runs before the command: no receipt yet, so it gives up.
    assert (
        await _verifier_started(sandbox, NS(execution_receipt=(receipt, "main")))
        is False
    )
    assert (tmp_path / "sandbox-run-benchflow/verifier-started-late.abandoned").exists()

    monkeypatch.setattr(verifier_core, "uuid4", lambda: NS(hex="late"))
    verifier = Verifier(task, paths, sandbox, execution_receipt=True)
    with pytest.raises(RewardFileNotFoundError):  # test.sh never ran
        await verifier.verify()
    assert not (paths.verifier_dir / "reward.txt").exists()
    assert (tmp_path / "sandbox-run-benchflow/verifier-started-late").exists()
