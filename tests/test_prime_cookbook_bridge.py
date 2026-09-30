"""The Prime cookbook's bridge (the BenchFlow side of benchflow-taskset), offline.

bridge.py runs in a BenchFlow venv and owns each episode's sandbox through
TaskRuntime. These tests use BenchFlow's real task loading and reward helper
(benchflow.integrations.rewards) with a scripted TaskRuntime, and run the
bridge as a process to check that every way a session can end closes the
sandbox: a close request, end of input, SIGTERM, and the idle timeout.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "docs"
    / "examples"
    / "rl"
    / "prime"
    / "benchflow_taskset"
    / "benchflow_taskset"
)
sys.path.insert(0, str(PACKAGE))
import bridge  # noqa: E402

import benchflow.rollout  # noqa: E402
from benchflow.sandbox.daytona_pty import TransientSandboxTransportError  # noqa: E402

DEMO_TASK = ROOT / "src" / "benchflow" / "demo_task"


class FakeRuntime:
    """A TaskRuntime whose sandbox is a script."""

    instances: ClassVar[list[FakeRuntime]] = []
    start_error: Exception | None = None
    bash_outcome: object = None
    verify_outcome: object = None

    def __init__(self, config) -> None:
        self.config = config
        self.closed = 0
        self.commands: list[tuple[str, int]] = []
        FakeRuntime.instances.append(self)

    async def start(self) -> None:
        if FakeRuntime.start_error is not None:
            raise FakeRuntime.start_error

    @property
    def workspace(self) -> str:
        return "/workdir"

    @property
    def rollout_dir(self) -> Path:
        return Path("/jobs/r")

    async def bash(self, command: str, *, timeout_sec: int = 30):
        self.commands.append((command, timeout_sec))
        outcome = FakeRuntime.bash_outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome or SimpleNamespace(
            return_code=0, stdout="ok\n", stderr="", elapsed_sec=0.1
        )

    async def verify(self):
        outcome = FakeRuntime.verify_outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    FakeRuntime.instances = []
    FakeRuntime.start_error = None
    FakeRuntime.bash_outcome = None
    FakeRuntime.verify_outcome = None
    monkeypatch.setattr(benchflow.rollout, "TaskRuntime", FakeRuntime)
    return FakeRuntime


def verified(reward=None, *, verifier_error=None, error=None, error_category=None):
    run = SimpleNamespace(
        rewards={"reward": reward} if reward is not None else None,
        reward=None,
        verifier_error=verifier_error,
        error=error,
        error_category=error_category,
        verifier_error_category=None,
    )
    return SimpleNamespace(
        reward=reward,
        rewards=run.rewards,
        verifier_error=verifier_error,
        error=error,
        rollout_dir=Path("/jobs/r"),
        result=run,
    )


def test_tasks_are_listed_with_benchflows_own_loader() -> None:
    (row,) = bridge.list_tasks(DEMO_TASK, [], [])
    assert row["name"] == "demo_task"
    assert "hello.txt" in row["prompt"]
    assert row["task_dir"] == str(DEMO_TASK.resolve())
    assert bridge.list_tasks(DEMO_TASK, [], ["demo_task"]) == []


async def test_a_sandbox_that_never_starts_is_a_sandbox_start_drop(
    fake_runtime,
) -> None:
    fake_runtime.start_error = RuntimeError("image build failed")
    session = bridge.Session()
    reply = await session.start({"task_dir": str(DEMO_TASK)})
    assert reply["ok"] is False
    assert reply["decision"]["dropped"] is True
    assert reply["decision"]["reason"] == "sandbox_start"


async def test_commands_run_under_their_own_timeout(fake_runtime) -> None:
    session = bridge.Session()
    assert (await session.start({"task_dir": str(DEMO_TASK)}))["ok"]
    reply = await session.bash({"command": "echo 'hi there'", "timeout_sec": 30})
    assert reply == {
        "ok": True,
        "return_code": 0,
        "stdout": "ok\n",
        "stderr": "",
        "timed_out": False,
        "elapsed_sec": 0.1,
    }
    command, wait = fake_runtime.instances[0].commands[0]
    assert command == "timeout -k 5 30 bash -c 'echo '\"'\"'hi there'\"'\"''"
    assert wait == 30 + bridge.EXEC_GRACE_SEC, (
        "the exec waits past the command's own limit"
    )


async def test_a_command_that_runs_out_of_time_is_a_normal_result(fake_runtime) -> None:
    session = bridge.Session()
    await session.start({"task_dir": str(DEMO_TASK)})
    fake_runtime.bash_outcome = RuntimeError("Command timed out after 45 seconds")
    reply = await session.bash({"command": "sleep 99", "timeout_sec": 30})
    assert (
        reply["ok"] is True
        and reply["timed_out"] is True
        and reply["return_code"] == 124
    )
    fake_runtime.bash_outcome = SimpleNamespace(
        return_code=124, stdout="partial", stderr="", elapsed_sec=30.2
    )
    assert (await session.bash({"command": "sleep 99", "timeout_sec": 30}))[
        "timed_out"
    ] is True


async def test_a_transport_blip_is_not_a_command_timeout(fake_runtime) -> None:
    session = bridge.Session()
    await session.start({"task_dir": str(DEMO_TASK)})
    fake_runtime.bash_outcome = TransientSandboxTransportError(
        "DaytonaTimeoutError: Request timed out"
    )
    reply = await session.bash({"command": "ls", "timeout_sec": 30})
    assert reply["ok"] is False and reply["transient"] is True


async def test_verifier_rewards_and_failures_follow_the_training_rule(
    fake_runtime,
) -> None:
    cases = [
        # (verify outcome, policy acted, expected reward, expected reason)
        (verified(1.0), True, 1.0, "scored"),
        (verified(0.0), True, 0.0, "scored"),
        (verified(None, verifier_error="pytest crashed"), True, 0.0, "verifier_error"),
        (
            verified(None, verifier_error="pytest crashed"),
            False,
            None,
            "verifier_crash_clean_run",
        ),
        (RuntimeError("sandbox gone"), True, 0.0, "verifier_error"),
        (RuntimeError("sandbox gone"), False, None, "verifier_crash_clean_run"),
    ]
    for outcome, acted, reward, reason in cases:
        session = bridge.Session()
        await session.start({"task_dir": str(DEMO_TASK)})
        fake_runtime.verify_outcome = outcome
        reply = await session.verify({"policy_acted": acted})
        assert reply["ok"] is True
        decision = reply["decision"]
        assert (decision["reward"], decision["reason"]) == (reward, reason), (
            outcome,
            acted,
        )
        assert decision["dropped"] is (reward is None)
        assert (await session.verify({"policy_acted": acted}))["ok"] is False, (
            "verify runs once"
        )


async def test_close_is_idempotent_and_a_verified_runtime_still_closes(
    fake_runtime,
) -> None:
    session = bridge.Session()
    await session.start({"task_dir": str(DEMO_TASK)})
    fake_runtime.verify_outcome = verified(1.0)
    await session.verify({"policy_acted": True})
    assert (await session.close())["ok"] and (await session.close())["ok"]
    assert fake_runtime.instances[0].closed == 1


WRAPPER = """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {package!r})
import benchflow.rollout, bridge
log = Path({log!r})
class Runtime:
    def __init__(self, config): pass
    async def start(self):
        # BenchFlow or its dependencies may print; replies must stay the only stdout lines.
        print("noise on stdout from the sandbox layer", flush=True)
        log.open("a").write("started\\n")
    workspace = "/workdir"
    rollout_dir = Path("/jobs/r")
    async def bash(self, command, *, timeout_sec=30):
        log.open("a").write("bash\\n")
        await asyncio.sleep(60)
    async def close(self): log.open("a").write("closed\\n")
benchflow.rollout.TaskRuntime = Runtime
sys.exit(bridge.main(sys.argv[1:]))
"""


def start_bridge(tmp_path: Path, *args: str) -> tuple[subprocess.Popen, Path]:
    log = tmp_path / "runtime.log"
    script = tmp_path / "wrapper.py"
    script.write_text(WRAPPER.format(package=str(PACKAGE), log=str(log)))
    process = subprocess.Popen(
        [sys.executable, str(script), "session", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.stdin.write(json.dumps({"op": "start", "task_dir": str(DEMO_TASK)}) + "\n")
    process.stdin.flush()
    reply = json.loads(process.stdout.readline())
    assert reply["ok"] is True, reply
    return process, log


def wait_closed(process: subprocess.Popen, log: Path) -> list[str]:
    process.wait(timeout=30)
    return log.read_text().split()


def test_end_of_input_closes_the_sandbox(tmp_path) -> None:
    process, log = start_bridge(tmp_path)
    process.stdin.close()
    assert wait_closed(process, log) == ["started", "closed"]
    assert "noise on stdout" in process.stderr.read()


def test_sigterm_during_a_command_closes_the_sandbox(tmp_path) -> None:
    process, log = start_bridge(tmp_path)
    process.stdin.write(json.dumps({"op": "bash", "command": "sleep 60"}) + "\n")
    process.stdin.flush()
    deadline = time.monotonic() + 20
    while (
        "bash" not in (log.read_text() if log.exists() else "")
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)
    os.kill(process.pid, signal.SIGTERM)
    assert wait_closed(process, log) == ["started", "bash", "closed"]


def test_an_idle_bridge_closes_the_sandbox(tmp_path) -> None:
    process, log = start_bridge(tmp_path, "--idle-timeout", "1")
    assert wait_closed(process, log) == ["started", "closed"]


def test_a_close_request_is_answered_then_the_bridge_exits(tmp_path) -> None:
    process, log = start_bridge(tmp_path)
    process.stdin.write(json.dumps({"op": "close"}) + "\n")
    process.stdin.flush()
    assert json.loads(process.stdout.readline()) == {"ok": True}
    assert wait_closed(process, log) == ["started", "closed"]


def test_stray_output_goes_to_stderr_and_unknown_ops_are_answered(tmp_path) -> None:
    # start_bridge already read the start reply as the first stdout line, although the
    # scripted runtime printed to stdout first; that print went to stderr.
    process, log = start_bridge(tmp_path)
    process.stdin.write(json.dumps({"op": "nonsense"}) + "\n")
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["ok"] is False
    process.stdin.close()
    assert wait_closed(process, log) == ["started", "closed"]
