"""Separate mode stops the agent's sandbox before the transfer, and starts the
verifier's meanwhile (PY8 SV3).

Each trial here is a real ``Rollout`` whose two sandboxes are fakes writing one
ordered event log: the agent's (capture, its teardown steps, its stop) and the
verifier's (create, start, uploads, execs, stop). The frozen workspace and
``/logs/artifacts`` are real captures of a local folder, so the transfer, the
install script and the publish run as in a trial. ``run()`` sets
``_release_agent_before_verify`` around its own verify(); the tests set it
the same way, then call ``cleanup()`` as ``run()``'s ``finally`` does.

The serial order (``BENCHFLOW_VERIFIER_PREPARE=serial``) is the base order, so
several tests run a scenario in both orders and compare the trial's outcome.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import benchflow.rollout as rollout_mod
from benchflow._utils import timing_spans
from benchflow._utils.reachable_write import ReachablePath
from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.budget import Budget, BudgetGuard
from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout import _separate_verifier as sv
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    SolutionTransferRefused,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import Task, TaskConfig
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from benchflow.trajectories.types import redact_acp_trajectory_jsonl

RECORD = Path("verifier-sandbox") / "verifier-sandbox.json"
PREPARE_ENV = "BENCHFLOW_VERIFIER_PREPARE"
# The teardown steps that read the agent's sandbox, in today's order.
PRE_STOP = (
    "skill_export",
    "usage_runtime_stop",
    "reconcile",
    "egress_stop",
    "finalize_usage",
    "price",
)
NEW_SPANS = (
    "verifier_sandbox_ready_wait",
    "agent_release",
    "agent_sandbox_stop_before_verify",
    "verifier_sandbox_overlap",
)
TOOL_CALL_ID = "call-ls-1"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class LocalTransport:
    """The agent's sandbox for capture is this machine's folder."""

    is_mounted = False

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30, **_):
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(process.returncode or 0, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        shutil.copyfile(src, dst)


class Scenario:
    """One trial: the rollout, its two fake sandboxes and the event log."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.log: list[str] = []
        # A step named here waits (until cancelled or ``unblock`` is set)
        # after logging "<step>.blocked"; one named in ``fail`` raises.
        self.block: str | None = None
        self.unblock = asyncio.Event()
        self.fail: dict[str, BaseException] = {}
        # Seconds the fake clock advances in each step (timing tests).
        self.seconds: dict[str, float] = {}
        self.clock: FakeClock | None = None
        self.pack_gate = threading.Event()
        self.verifier_started = asyncio.Event()
        self.capture_waits_for_verifier = False
        self.agent_stop_errors: list[BaseException] = []
        self.verifier_start_error: BaseException | None = None
        self.verifier: FakeVerifier | None = None
        self.verifier_uploads: dict[str, bytes] = {}
        self.scraped: list[dict] = []
        # The agent sandbox's faked remote side, where a test has one.
        self.remote: Any = None
        self.rollout: Rollout

    def tick(self, name: str) -> None:
        if self.clock is not None:
            self.clock.now += self.seconds.get(name, 0.0)

    async def step(self, name: str) -> None:
        self.log.append(name)
        self.tick(name)
        if name in self.fail:
            raise self.fail[name]
        if self.block == name:
            self.log.append(f"{name}.blocked")
            await self.unblock.wait()

    def sync_step(self, name: str) -> None:
        self.log.append(name)
        self.tick(name)
        if name in self.fail:
            raise self.fail[name]

    def index(self, event: str) -> int:
        return self.log.index(event)

    def count(self, event: str) -> int:
        return self.log.count(event)

    def record(self) -> dict[str, Any]:
        return json.loads((self.rollout._rollout_dir / RECORD).read_text())


class FakeAgent:
    """The agent's sandbox: only its stop matters here."""

    is_mounted = False
    sandbox_id = "agent-box"

    def __init__(self, sc: Scenario) -> None:
        self.sc = sc
        self.stops: list[bool] = []

    async def exec(self, cmd: str, **_: Any) -> ExecResult:
        self.sc.log.append("agent.exec")
        return ExecResult(0, "", "")

    async def stop(self, delete: bool = True) -> None:
        self.stops.append(delete)
        await self.sc.step("agent.stop")
        if self.sc.agent_stop_errors:
            raise self.sc.agent_stop_errors.pop(0)
        self.sc.log.append("agent.stopped")


class FakeVerifier:
    """The verifier's sandbox: a local folder, each call logged."""

    is_mounted = False
    sandbox_id = "verifier-box"

    def __init__(self, sc: Scenario, root: Path) -> None:
        self.sc = sc
        self.root = root
        self.stops: list[bool] = []
        self.uploads: dict[str, Path] = {}

    async def start(self, force_build: bool = False) -> None:
        self.sc.verifier_started.set()
        await self.sc.step("verifier.start")
        if self.sc.verifier_start_error is not None:
            raise self.sc.verifier_start_error

    async def stop(self, delete: bool = True) -> None:
        # "verifier.stop" is the provider's own await before its delete (a
        # log chown on Docker); "verifier.deleted" is the delete done.
        self.stops.append(delete)
        await self.sc.step("verifier.stop")
        self.sc.log.append("verifier.deleted")

    async def upload_file(self, source: Path | str, target: str, **_kwargs) -> None:
        self.sc.verifier_uploads[target] = Path(source).read_bytes()
        kind = "tar" if target.endswith(".tar") else target
        await self.sc.step(f"verifier.upload {kind}")
        dest = self.root / target.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        self.uploads[target] = dest

    async def exec(self, cmd: str, *, user: str = "root", **_: Any) -> ExecResult:
        if "runtime-verifier-destinations-" in cmd:
            # Match the current custody probe, without carrying agent bytes.
            process = await asyncio.create_subprocess_exec(
                "sh",
                str(next(p for t, p in self.uploads.items() if "destinations" in t)),
                str(self.root.resolve()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await process.communicate()
            return ExecResult(process.returncode, out.decode(), err.decode())
        if "tar -xf" in cmd:
            await self.sc.step("verifier.exec unpack")
            archive = next(p for t, p in self.uploads.items() if t.endswith(".tar"))
            with tarfile.open(archive) as tar:
                tar.extractall(self.root, filter="tar")
        else:
            self.sc.log.append(f"verifier.exec {cmd.split()[0]}")
        return ExecResult(0, "", "")


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "tests" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nCOPY test.sh /tests/test.sh\n"
    )
    (task / "instruction.md").write_text("write 42\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\nenvironment_mode = "separate"\n'
    )
    return task


async def _captured(tmp_path: Path) -> tuple[Path, Path]:
    """The agent's outputs, frozen and collected into a trial folder."""
    box = tmp_path / "box"
    workspace = box / "app"
    workspace.mkdir(parents=True)
    (workspace / "answer.txt").write_text("42\n")
    logs = box / "logs" / "artifacts"
    logs.mkdir(parents=True)
    (logs / "mode.txt").write_text("separate\n")
    trial = tmp_path / "trial"
    (trial / "artifacts").mkdir(parents=True)
    env = LocalTransport()
    await capture_task_evidence(env, str(workspace), trial / "evidence", artifacts=[])
    await collect_artifacts(
        env,
        artifacts=[],
        workspace=str(workspace),
        artifacts_dir=ReachablePath.directory(trial / "artifacts"),
        manifest_path=trial / MANIFEST_NAME,
        mounted=False,
        logs_source=str(logs),
    )
    return workspace, trial


def _acp_trajectory() -> list[dict]:
    """An ACP trajectory whose one tool call lost its output (lossy ACP)."""
    return [
        {"type": "user_message", "text": "write 42"},
        {
            "type": "tool_call",
            "tool_call_id": TOOL_CALL_ID,
            "title": "Bash",
            "kind": "execute",
            "status": "completed",
            "content": [],
        },
        {"type": "agent_message", "text": "done"},
    ]


def _usage_runtime() -> Any:
    """A usage proxy whose capture holds that tool call's output."""
    exchange = SimpleNamespace(
        request=SimpleNamespace(
            body={
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": TOOL_CALL_ID,
                                "function": {
                                    "name": "Bash",
                                    "arguments": '{"command": "ls"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": TOOL_CALL_ID,
                        "content": "answer.txt",
                    },
                ]
            }
        ),
        response=SimpleNamespace(status_code=200),
    )
    return SimpleNamespace(
        server=SimpleNamespace(trajectory=SimpleNamespace(exchanges=[exchange]))
    )


async def _scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    environment: str = "docker",
    trajectory: list[dict] | None = None,
    agent_env: Any = None,
    clock: bool = False,
) -> Scenario:
    sc = Scenario(tmp_path)
    if clock:
        sc.clock = FakeClock()
        monkeypatch.setattr(timing_spans, "_clock", sc.clock)
    monkeypatch.delenv(PREPARE_ENV, raising=False)
    task = _task(tmp_path)
    workspace, trial = await _captured(tmp_path)
    config = RolloutConfig(
        task_path=task,
        agent="claude-agent-acp",
        environment=environment,
        export_generated_skills_to=tmp_path / "skills",
    )
    rollout = Rollout(config)
    sc.rollout = rollout
    rollout._task = Task(task)
    rollout._rollout_dir = trial
    rollout._rollout_paths = RolloutPaths(rollout_dir=trial)
    rollout._rollout_paths.mkdir()
    rollout._rollout_name = "trial-1"
    rollout._agent_cwd = str(workspace)
    rollout._env = agent_env if agent_env is not None else FakeAgent(sc)
    rollout._trajectory = list(trajectory or [])
    rollout._usage_runtime = _usage_runtime()
    rollout._egress_denylist = object()

    def create(environment, vtask, context, name, paths, **_):
        sc.log.append("verifier.create")
        sc.verifier = FakeVerifier(sc, tmp_path / "verifier-root")
        return sc.verifier

    async def stop_provider_runtime(runtime):
        await sc.step("usage_runtime_stop")

    async def stop_egress(env, rollout_dir):
        await sc.step("egress_stop")

    async def prepare_runtime(env, *, task, workspace):
        # V2's protected verifier runtime: nothing to build in these fakes.
        return None

    async def snapshot_build_config(env, workspace=None):
        sc.log.append("seal")

    rollout._planes = SimpleNamespace(
        create_environment=create,
        snapshot_build_config=snapshot_build_config,
        stop_provider_runtime=stop_provider_runtime,
        extract_usage=lambda runtime: {},
        stop_egress_denylist=stop_egress,
        prepare_separate_verifier_runtime=prepare_runtime,
    )

    async def capture(r):
        await sc.step("capture")
        if sc.capture_waits_for_verifier:
            await asyncio.wait_for(sc.verifier_started.wait(), timeout=5)
        sc.log.append("capture.end")

    async def collect(r):
        r._artifacts_collected = True
        await sc.step("collect")

    async def scrape(env, agent, user):
        await sc.step("scrape")
        return list(sc.scraped)

    async def price(r):
        await sc.step("price")

    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", capture)
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", collect)
    monkeypatch.setattr(rollout_mod, "_scrape_agent_trajectory", scrape)
    monkeypatch.setattr(rollout_mod, "price_from_session_log", price)

    async def export_skills():
        await sc.step("skill_export")

    def reconcile(runtime):
        sc.sync_step("reconcile")
        Rollout._reconcile_acp_tool_evidence(rollout, runtime)

    rollout._export_generated_skills = export_skills
    rollout._write_llm_trajectory = lambda runtime: None
    rollout._reconcile_acp_tool_evidence = reconcile
    rollout._finalize_usage_metrics = lambda: sc.sync_step("finalize_usage")
    rollout._enforce_required_usage_tracking = lambda: sc.sync_step("enforce")

    real_pack = sv.build_transfer_payload

    def pack(*args, **kwargs):
        sc.sync_step("pack")
        if sc.block == "pack":
            sc.log.append("pack.blocked")
            sc.pack_gate.wait(10)
        return real_pack(*args, **kwargs)

    monkeypatch.setattr(sv, "build_transfer_payload", pack)

    async def verify_rollout(env, task, paths, timing, planes, *args, **kwargs):
        await sc.step("verifier.test_sh")
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (paths.verifier_dir / "reward.txt").write_text("1\n")
        timing["verifier"] = 1.0
        return {"reward": 1.0}, None, None

    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify_rollout)
    return sc


async def _verify(sc: Scenario) -> dict | None:
    """verify() as run() calls it."""
    sc.rollout._release_agent_before_verify = True
    try:
        rewards = await sc.rollout.verify()
    finally:
        sc.rollout._release_agent_before_verify = False
    sc.log.append("verify.returned")
    return rewards


async def _trial(sc: Scenario) -> dict | None:
    """verify() and cleanup(), as run() runs them."""
    try:
        return await _verify(sc)
    finally:
        await sc.rollout.cleanup()


def _outcome(sc: Scenario) -> tuple[Any, Any, Any]:
    r = sc.rollout
    return r._rewards, r._verifier_error, r._error


def _first_agent_bytes(sc: Scenario) -> int:
    """The first verifier event that carries the agent's bytes."""
    carriers = [
        i
        for i, event in enumerate(sc.log)
        if event
        in (
            "verifier.upload tar",
            "verifier.exec unpack",
            "verifier.upload /logs/agent/acp_trajectory.jsonl",
        )
    ]
    assert carriers, sc.log
    return carriers[0]


# --- the order ---------------------------------------------------------------


@pytest.mark.parametrize("environment", ["docker", "remote-docker", "daytona", "modal"])
async def test_the_agent_stop_returns_before_the_verifier_gets_agent_bytes(
    tmp_path, monkeypatch, environment
):
    sc = await _scenario(
        tmp_path, monkeypatch, environment=environment, trajectory=_acp_trajectory()
    )

    rewards = await _trial(sc)

    assert rewards == {"reward": 1.0}
    assert sc.index("agent.stopped") < _first_agent_bytes(sc), sc.log
    # cleanup() does not stop it a second time.
    assert sc.rollout._env.stops == [True]
    record = sc.record()
    assert record["order"] == "overlap"
    assert record["agent_sandbox_alive_during_verify"] is False
    assert record["custody"]["agent_stop_error"] is None


@pytest.mark.parametrize("environment", ["docker", "remote-docker"])
async def test_on_docker_the_verifier_starts_before_capture_returns(
    tmp_path, monkeypatch, environment
):
    sc = await _scenario(
        tmp_path, monkeypatch, environment=environment, trajectory=_acp_trajectory()
    )
    sc.capture_waits_for_verifier = True

    await _trial(sc)

    assert sc.index("verifier.start") < sc.index("capture.end"), sc.log
    # Before the transfer the verifier sandbox only got created and started.
    first = _first_agent_bytes(sc)
    before = [e for e in sc.log[:first] if e.startswith("verifier.")]
    assert before[:2] == ["verifier.create", "verifier.start"], sc.log
    assert all("runtime-verifier-destinations-" in e for e in before[2:]), sc.log


@pytest.mark.parametrize("environment", ["daytona", "modal"])
async def test_on_daytona_the_verifier_starts_after_the_agent_stop(
    tmp_path, monkeypatch, environment
):
    """Modal too: its agent sandbox (a GPU) is gone first."""
    sc = await _scenario(
        tmp_path, monkeypatch, environment=environment, trajectory=_acp_trajectory()
    )

    await _trial(sc)

    assert sc.index("agent.stopped") < sc.index("verifier.create"), sc.log


async def test_the_scrape_runs_before_the_agent_stop(tmp_path, monkeypatch):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    sc.scraped = [{"type": "agent_message", "text": "scraped"}]

    await _trial(sc)

    assert sc.index("scrape") < sc.index("agent.stop"), sc.log
    assert sc.index("agent.stopped") < _first_agent_bytes(sc), sc.log
    assert sc.rollout._trajectory_source == "scraped"
    # The verifier reads the scraped trajectory.
    published = next(
        data
        for target, data in sc.verifier_uploads.items()
        if target.endswith("/acp_trajectory.jsonl")
    )
    assert b"scraped" in published


async def test_the_steps_that_read_the_agent_sandbox_run_once_before_its_stop(
    tmp_path, monkeypatch
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=_acp_trajectory())

    await _trial(sc)

    for step in (*PRE_STOP, "enforce"):
        assert sc.count(step) == 1, (step, sc.log)
    order = [sc.index(step) for step in PRE_STOP]
    assert order == sorted(order), sc.log
    assert order[-1] < sc.index("agent.stop"), sc.log
    # All of it inside verify(), before the verifier gets anything.
    assert sc.index("agent.stopped") < _first_agent_bytes(sc), sc.log
    # Enforcement stays in cleanup(), after the trial was graded.
    assert sc.index("verify.returned") < sc.index("enforce"), sc.log


async def test_a_rollout_set_up_again_gets_its_whole_teardown_again(
    tmp_path, monkeypatch
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    await _trial(sc)
    first = list(sc.log)

    # The same Rollout with a new sandbox and usage proxy, cleaned up again.
    sc.rollout._env = second = FakeAgent(sc)
    sc.rollout._usage_runtime = _usage_runtime()
    await sc.rollout.cleanup()

    again = sc.log[len(first) :]
    for step in (*PRE_STOP, "enforce", "agent.stopped"):
        assert again.count(step) == 1, (step, again)
    assert second.stops == [True]


async def test_the_verifier_reads_the_trajectory_from_before_the_reconcile(
    tmp_path, monkeypatch
):
    trajectory = _acp_trajectory()
    expected = (redact_acp_trajectory_jsonl(_acp_trajectory()) + "\n").encode()
    sc = await _scenario(tmp_path, monkeypatch, trajectory=trajectory)

    await _trial(sc)

    # The reconcile repaired the trial's own record before the verifier ran...
    assert sc.index("reconcile") < sc.index("verifier.test_sh"), sc.log
    own = (sc.rollout._rollout_dir / "trajectory" / "acp_trajectory.jsonl").read_text()
    assert "answer.txt" in own
    # ...and the verifier still got the bytes it got before (byte-identical).
    published = next(
        data
        for target, data in sc.verifier_uploads.items()
        if target.endswith("/acp_trajectory.jsonl")
    )
    assert published == expected
    host_copy = sc.rollout._rollout_dir / "verifier-sandbox" / "agent"
    assert (host_copy / "acp_trajectory.jsonl").read_bytes() == expected
    custody = sc.record()["custody"]["trajectory"]
    assert custody["bytes"] == len(expected)


# --- failures and cancellation ----------------------------------------------

CANCEL_POINTS = (
    "capture",
    "collect",
    "scrape",
    "skill_export",
    "usage_runtime_stop",
    "egress_stop",
    "price",
    "agent.stop",
    "pack",
    "verifier.start",
    "verifier.upload tar",
    "verifier.test_sh",
    # The verifier sandbox's own deletion, before its delete went through.
    "verifier.stop",
)


@pytest.mark.parametrize("point", CANCEL_POINTS)
async def test_a_cancellation_at_each_await_deletes_both_sandboxes(
    tmp_path, monkeypatch, point
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    sc.block = point
    task = asyncio.create_task(_verify(sc))
    # Docker: the verifier sandbox was prestarted before any of these.
    for _ in range(500):
        if f"{point}.blocked" in sc.log and "verifier.start" in sc.log:
            break
        await asyncio.sleep(0.01)
    assert f"{point}.blocked" in sc.log and "verifier.start" in sc.log, sc.log

    task.cancel()
    sc.pack_gate.set()
    sc.unblock.set()  # a deletion under way goes on to its end
    with pytest.raises(asyncio.CancelledError):
        await task
    sc.block = None
    await sc.rollout.cleanup()  # run()'s finally

    # The agent's sandbox ends stopped and deleted (a stop cut short by the
    # cancel is done again by cleanup()).
    assert sc.rollout._env.stops and sc.rollout._env.stops[-1] is True
    assert sc.log.count("agent.stopped") == 1, sc.log
    # The prestarted verifier sandbox is deleted exactly once.
    assert sc.verifier is not None
    assert sc.verifier.stops == [True], sc.log
    assert sc.count("verifier.deleted") == 1, sc.log
    record = sc.record()
    assert record["status"] == "interrupted"
    assert "verifier_sandbox_teardown" in sc.rollout.timing
    assert "verifier_sandbox_teardown" in record["timing"]
    assert record["sandbox_seconds"] is not None


def _boom(step: str) -> BaseException:
    return RuntimeError(f"{step} broke")


async def _both_orders(tmp_path, monkeypatch, arrange) -> tuple[Scenario, Scenario]:
    """The same failing trial in the serial (base) order and the overlap order."""
    outcomes = []
    for order in ("serial", "overlap"):
        sub = tmp_path / order
        sub.mkdir()
        sc = await _scenario(sub, monkeypatch, trajectory=[])
        if order == "serial":
            monkeypatch.setenv(PREPARE_ENV, "serial")
        arrange(sc)
        try:
            await _trial(sc)
        except Exception as exc:
            sc.log.append(f"raised {type(exc).__name__}: {exc}")
        monkeypatch.delenv(PREPARE_ENV, raising=False)
        outcomes.append(sc)
    return outcomes[0], outcomes[1]


@pytest.mark.parametrize("step", ["capture", "collect", "scrape"])
async def test_an_exception_before_the_release_deletes_the_verifier_once(
    tmp_path, monkeypatch, step
):
    def arrange(sc: Scenario) -> None:
        sc.fail[step] = _boom(step)

    serial, overlap = await _both_orders(tmp_path, monkeypatch, arrange)

    # Today's outcome: the exception leaves verify() and run() reports it.
    assert f"raised RuntimeError: {step} broke" in serial.log
    assert f"raised RuntimeError: {step} broke" in overlap.log
    assert _outcome(overlap) == _outcome(serial)
    # The verifier sandbox started for the trial is deleted, once.
    assert overlap.verifier is not None and overlap.verifier.stops == [True]
    assert overlap.record()["status"] == "abandoned"
    assert "verifier_sandbox_teardown" in overlap.rollout.timing
    # The agent's sandbox is stopped by cleanup(), as today.
    assert overlap.rollout._env.stops == [True]


@pytest.mark.parametrize(
    "step",
    ["skill_export", "usage_runtime_stop", "egress_stop", "finalize_usage", "price"],
)
async def test_an_exception_in_a_step_before_the_stop_keeps_todays_outcome(
    tmp_path, monkeypatch, step
):
    def arrange(sc: Scenario) -> None:
        sc.fail[step] = _boom(step)

    serial, overlap = await _both_orders(tmp_path, monkeypatch, arrange)

    assert _outcome(overlap) == _outcome(serial)
    raised = [e for e in serial.log if e.startswith("raised ")]
    assert [e for e in overlap.log if e.startswith("raised ")] == raised
    assert overlap.verifier is not None and overlap.verifier.stops == [True]
    # The agent's sandbox is stopped as often as today (a failure that
    # leaves cleanup() stops it before the stop, both ways).
    assert overlap.rollout._env.stops == serial.rollout._env.stops
    if step in ("skill_export", "usage_runtime_stop", "egress_stop"):
        # Those steps catch their own failures: the stop still comes first.
        assert overlap.index("agent.stopped") < _first_agent_bytes(overlap)
    else:
        # A failure that leaves its step: the agent's sandbox stays up
        # until cleanup(), which runs the step again, as today.
        assert overlap.record()["custody"]["agent_stop_error"].startswith(
            "not stopped before the verifier"
        )
        assert overlap.count(step) == serial.count(step) + 1


async def test_a_transfer_failure_keeps_todays_error_and_deletes_the_verifier_once(
    tmp_path, monkeypatch
):
    def arrange(sc: Scenario) -> None:
        sc.fail["pack"] = SeparateVerifierError("manifest mismatch")

    serial, overlap = await _both_orders(tmp_path, monkeypatch, arrange)

    assert _outcome(overlap) == _outcome(serial)
    rewards, error, _ = _outcome(overlap)
    assert rewards is None
    assert error == "separate verifier transfer failed: manifest mismatch"
    assert overlap.record()["status"] == serial.record()["status"] == "transfer_failed"
    assert overlap.verifier is not None and overlap.verifier.stops == [True]


async def test_a_transfer_failure_still_comes_before_a_failed_start(
    tmp_path, monkeypatch
):
    """Both fail: the serial order never got to the start, so its error and
    status are the transfer's; the overlap order keeps them."""

    def arrange(sc: Scenario) -> None:
        sc.fail["pack"] = SeparateVerifierError("manifest mismatch")
        sc.verifier_start_error = RuntimeError("image build failed")

    serial, overlap = await _both_orders(tmp_path, monkeypatch, arrange)

    assert _outcome(overlap) == _outcome(serial)
    assert overlap.record()["status"] == serial.record()["status"] == "transfer_failed"
    assert overlap.verifier is not None and overlap.verifier.stops == [True]


async def test_a_failed_agent_stop_is_recorded_and_retried_by_cleanup(
    tmp_path, monkeypatch
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=_acp_trajectory())
    sc.agent_stop_errors = [RuntimeError("daemon said no")]

    rewards = await _trial(sc)

    assert rewards == {"reward": 1.0}
    assert sc.rollout._verifier_error is None and sc.rollout._error is None
    record = sc.record()
    assert record["status"] == "complete"
    assert "daemon said no" in record["custody"]["agent_stop_error"]
    assert record["agent_sandbox_alive_during_verify"] is True
    # The trial is verified as usual, then cleanup() stops the sandbox again.
    assert sc.rollout._env.stops == [True, True]
    last_stop = len(sc.log) - 1 - sc.log[::-1].index("agent.stop")
    assert sc.index("verify.returned") < last_stop, sc.log
    assert sc.log.count("agent.stopped") == 1
    # Billed together the whole time the verifier sandbox lived.
    timing = sc.rollout.timing
    assert timing["verifier_sandbox_overlap"] == pytest.approx(
        timing["verifier_sandbox_total"]
    )


async def test_a_wedged_agent_stop_is_bounded_and_recorded(tmp_path, monkeypatch):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=_acp_trajectory())
    monkeypatch.setattr(rollout_mod, "_AGENT_STOP_TIMEOUT_SEC", 0.2, raising=False)
    sc.block = "agent.stop"

    task = asyncio.create_task(_verify(sc))
    rewards = await asyncio.wait_for(task, timeout=10)
    sc.block = None
    await sc.rollout.cleanup()

    assert rewards == {"reward": 1.0}
    assert "TimeoutError" in sc.record()["custody"]["agent_stop_error"]
    assert sc.rollout._env.stops == [True, True]


# The real providers do not raise when their delete fails: Daytona logs the
# sandbox as left running and drops its handle, Docker's forced kill only
# logs. These trials stop real provider objects whose remote side is faked.


class DaytonaNotFoundError(Exception):
    status_code = 404


DaytonaNotFoundError.__module__ = "daytona.common.errors"


class DaytonaRemote:
    """The agent's Daytona sandbox as the SDK's ``AsyncSandbox`` shows it:
    ``delete()`` refreshes ``state``, as the SDK does after a delete."""

    def __init__(
        self,
        sc: Scenario,
        *errors: BaseException,
        block_first: bool = False,
        gone: bool = False,
        block_refresh: bool = False,
    ) -> None:
        self.sc = sc
        self.id = "agent-daytona-box"
        self.state = "started"
        self.desired_state = "started"
        self.errors = list(errors)
        self.block_first = block_first
        # Deleted on Daytona's side already: every call answers 404.
        self.gone = gone
        self.deletes = 0
        self.refreshes = 0
        # The first refresh (the stop's check) waits forever, as a stalled
        # lookup would.
        self.block_refresh = block_refresh

    async def delete(self, timeout=60, wait=False) -> None:
        self.deletes += 1
        self.sc.log.append("agent.delete")
        if self.block_first:
            self.block_first = False
            await asyncio.Event().wait()
        if self.gone:
            raise DaytonaNotFoundError("Sandbox not found")
        if self.errors:
            raise self.errors.pop(0)
        self.state = self.desired_state = "destroyed"

    async def refresh_data(self) -> None:
        self.refreshes += 1
        self.sc.log.append("agent.refresh")
        if self.block_refresh:
            self.block_refresh = False
            await asyncio.Event().wait()
        if self.gone:
            raise DaytonaNotFoundError("Sandbox not found")


def _daytona_agent(remote: DaytonaRemote) -> Any:
    """A real ``DaytonaSandbox`` and stop strategy around a faked remote."""
    import logging

    pytest.importorskip("tenacity")  # the delete retry ships with sandbox-daytona
    from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect

    agent = DaytonaSandbox.__new__(DaytonaSandbox)
    agent._sandbox = remote
    agent._client_manager = None
    agent.logger = logging.getLogger("test.sv_overlap.daytona")
    agent._strategy = _DaytonaDirect(agent)
    return agent


async def _daytona_scenario(tmp_path, monkeypatch, **remote_kwargs) -> Scenario:
    sc = await _scenario(tmp_path, monkeypatch, environment="daytona", trajectory=[])
    errors = remote_kwargs.pop("errors", ())
    remote = DaytonaRemote(sc, *errors, **remote_kwargs)
    sc.rollout._env = _daytona_agent(remote)
    sc.remote = remote
    return sc


async def test_a_daytona_delete_that_fails_inside_the_provider_is_recorded_and_retried(
    tmp_path, monkeypatch
):
    """Guards ac3bedaa1: after_stop cannot verify while release is unconfirmed.

    The earlier expectation awarded a score while the provider still owned
    the agent sandbox, contradicting the declared after_stop lifecycle.
    """
    sc = await _daytona_scenario(
        tmp_path, monkeypatch, errors=(RuntimeError("delete refused"),)
    )
    remote = sc.remote

    rewards = await _trial(sc)

    assert rewards is None
    assert "agent sandbox release" in sc.rollout._verifier_error
    assert sc.rollout._error is None
    record = sc.record()
    assert record["status"] == "abandoned"
    assert record["custody"]["agent_stop_error"] == (
        "Daytona sandbox agent-daytona-box is still started after its delete"
    )
    assert record["agent_sandbox_alive_during_verify"] is True
    assert "verifier.create" not in sc.log
    # cleanup() deleted it again, after verification was refused.
    assert remote.deletes == 2 and remote.state == "destroyed"
    delete_at = [i for i, e in enumerate(sc.log) if e == "agent.delete"]
    assert sc.index("verify.returned") < delete_at[-1], sc.log
    timing = sc.rollout.timing
    assert timing["verifier_sandbox_overlap"] == 0.0
    assert "verifier_sandbox_total" not in timing


async def test_a_wedged_daytona_stop_leaves_the_sandbox_for_cleanup(
    tmp_path, monkeypatch
):
    """Guards ac3bedaa1: an after_stop timeout retains custody and no score."""
    monkeypatch.setattr(rollout_mod, "_AGENT_STOP_TIMEOUT_SEC", 0.2, raising=False)
    sc = await _daytona_scenario(tmp_path, monkeypatch, block_first=True)
    remote = sc.remote

    rewards = await asyncio.wait_for(_trial(sc), timeout=10)

    assert rewards is None
    assert "agent sandbox release" in sc.rollout._verifier_error
    assert "verifier.create" not in sc.log
    record = sc.record()
    assert "TimeoutError" in record["custody"]["agent_stop_error"]
    assert record["agent_sandbox_alive_during_verify"] is True
    # The cancelled stop dropped its handle; cleanup() still deletes the box.
    assert remote.deletes == 2 and remote.state == "destroyed"
    assert sc.rollout._env._sandbox is None


@pytest.mark.parametrize("gone", [False, True], ids=["deleted", "already-gone"])
async def test_a_daytona_delete_that_went_through_counts_as_stopped(
    tmp_path, monkeypatch, gone
):
    sc = await _daytona_scenario(tmp_path, monkeypatch, gone=gone)
    remote = sc.remote

    await _trial(sc)

    record = sc.record()
    assert record["custody"]["agent_stop_error"] is None
    assert record["agent_sandbox_alive_during_verify"] is False
    assert remote.deletes == 1
    # A delete the SDK saw through costs no lookup; a 404 costs one.
    assert remote.refreshes == (1 if gone else 0)
    assert sc.rollout.timing["verifier_sandbox_overlap"] == 0.0


# Each await of the stop before the verifier on Daytona: the delete itself
# (still pending when the cancel lands), and the check that looks the sandbox
# up after a delete that did not go through.
DAYTONA_STOP_CANCEL_POINTS = {
    "delete": {"block_first": True},
    "check": {"errors": (RuntimeError("delete refused"),), "block_refresh": True},
}


@pytest.mark.parametrize("point", sorted(DAYTONA_STOP_CANCEL_POINTS))
async def test_a_cancel_during_the_daytona_stop_leaves_no_sandbox_behind(
    tmp_path, monkeypatch, point
):
    """Guards SV-ALL-1 (fails on 91e4806c: Daytona's stop cleared the handle in
    its finally, the cancel skipped the restore, and cleanup found no sandbox:
    the box stayed started with one delete)."""
    sc = await _daytona_scenario(
        tmp_path, monkeypatch, **dict(DAYTONA_STOP_CANCEL_POINTS[point])
    )
    remote = sc.remote
    task = asyncio.create_task(_verify(sc))
    awaited = "agent.delete" if point == "delete" else "agent.refresh"
    for _ in range(500):
        if awaited in sc.log:
            break
        await asyncio.sleep(0.01)
    assert awaited in sc.log, sc.log
    await asyncio.sleep(0.05)  # it is waiting there now

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await sc.rollout.cleanup()  # run()'s finally

    # cleanup() had the handle back and deleted the box.
    assert remote.state == "destroyed", (remote.deletes, sc.log)
    assert remote.deletes == 2
    assert sc.rollout._env._sandbox is None
    record = sc.record()
    assert record["status"] == "interrupted"
    assert record["custody"]["agent_stop_error"] == (
        "the stop before the verifier was interrupted (CancelledError); "
        "cleanup stops it again"
    )
    assert record["agent_sandbox_alive_during_verify"] is not False
    # On Daytona the verifier sandbox starts only after the stop: none was made.
    assert sc.verifier is None or sc.verifier.stops == [True]


async def test_an_error_during_the_daytona_stop_check_hands_the_sandbox_to_cleanup(
    tmp_path, monkeypatch
):
    """An exception that leaves the stop (not one the stop catches) also gives
    the handle back; the trial's own error handling is unchanged."""
    sc = await _daytona_scenario(
        tmp_path, monkeypatch, errors=(RuntimeError("delete refused"),)
    )
    remote = sc.remote

    class Interrupted(BaseException):
        """Not an Exception: nothing on the way catches it."""

    async def broken_check(env, handle=None):
        raise Interrupted

    monkeypatch.setattr(sv, "agent_sandbox_left_running", broken_check)
    with pytest.raises(Interrupted):
        await _verify(sc)
    await sc.rollout.cleanup()

    assert remote.state == "destroyed" and remote.deletes == 2
    record = sc.record()
    assert record["custody"]["agent_stop_error"] == (
        "the stop before the verifier was interrupted (Interrupted); "
        "cleanup stops it again"
    )


def _docker_agent(tmp_path: Path, *, keep_containers: bool = False) -> DockerSandbox:
    environment = tmp_path / "agent-env" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "agent-env" / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="task",
        session_id="trial-1",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
        keep_containers=keep_containers,
    )


async def test_a_docker_teardown_that_fails_inside_the_provider_is_recorded(
    tmp_path, monkeypatch
):
    agent = _docker_agent(tmp_path)
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[], agent_env=agent)
    containers = ["c0ffee"]
    downs: list[str] = []

    async def compose(command, check=True, timeout_sec=None):
        sc.log.append("agent.compose " + " ".join(command))
        if command[0] == "down":
            downs.append(command[0])
            if len(downs) == 1:
                raise RuntimeError("compose down failed")
            containers.clear()
        return ExecResult(stdout="", stderr="", return_code=0)

    async def force_kill():
        # The real one logs its own failure and returns.
        sc.log.append("agent.force_kill")

    async def docker_cli(args, check=True, *, raw_output=False):
        sc.log.append("agent.docker " + " ".join(args[:2]))
        return ExecResult(stdout="\n".join(containers), stderr="", return_code=0)

    agent._run_docker_compose_command = compose
    agent._force_kill_project = force_kill
    agent._docker_cli = docker_cli

    rewards = await _trial(sc)

    assert rewards == {"reward": 1.0}
    record = sc.record()
    assert record["custody"]["agent_stop_error"] == (
        f"1 container(s) of compose project {agent.compose_project_name} "
        "still there after its stop"
    )
    assert record["agent_sandbox_alive_during_verify"] is True
    # cleanup() ran the teardown again, after the verifier.
    assert len(downs) == 2
    down_at = [i for i, e in enumerate(sc.log) if e.startswith("agent.compose down")]
    assert sc.index("verify.returned") < down_at[-1], sc.log
    timing = sc.rollout.timing
    assert timing["verifier_sandbox_overlap"] == pytest.approx(
        timing["verifier_sandbox_total"]
    )


async def test_a_refused_transfer_still_scores_zero_and_deletes_the_verifier_once(
    tmp_path, monkeypatch
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    sc.fail["pack"] = SolutionTransferRefused("workspace over the capture limits")

    rewards = await _trial(sc)

    assert rewards == {"reward": 0.0}
    assert sc.rollout._verifier_error is None
    record = sc.record()
    assert record["status"] == "refused"
    assert "workspace over the capture limits" in record["refusal"]
    assert sc.verifier is not None and sc.verifier.stops == [True]
    assert "verifier.upload tar" not in sc.log


async def test_a_cancel_during_a_refused_transfers_deletion_still_deletes_the_verifier(
    tmp_path, monkeypatch
):
    """Before SV3 a refused transfer had no verifier sandbox yet;
    now its prestarted one is deleted even when the trial is cancelled while
    that deletion awaits (Docker's log chown, before ``compose down``)."""
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    sc.fail["pack"] = SolutionTransferRefused("workspace over the capture limits")
    sc.block = "verifier.stop"
    task = asyncio.create_task(_verify(sc))
    for _ in range(500):
        if "verifier.stop.blocked" in sc.log:
            break
        await asyncio.sleep(0.01)
    assert "verifier.stop.blocked" in sc.log, sc.log

    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0)
    # The cancel waits for the deletion it interrupted...
    assert not task.done()
    sc.unblock.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await sc.rollout.cleanup()  # run()'s finally

    # ...which went through, once, and was recorded.
    assert sc.verifier is not None and sc.verifier.stops == [True], sc.log
    assert sc.count("verifier.deleted") == 1, sc.log
    record = sc.record()
    assert record["status"] == "interrupted"
    assert "workspace over the capture limits" in record["refusal"]
    assert record["sandbox_seconds"] is not None
    assert "verifier_sandbox_teardown" in record["timing"]
    assert "verifier_sandbox_overlap" in record["timing"]
    assert sc.log.count("agent.stopped") == 1, sc.log


async def test_a_verifier_start_failure_after_the_agent_stop_is_unscored(
    tmp_path, monkeypatch
):
    from benchflow.rollout._verifier_recovery import recovery_ineligible_reason

    def arrange(sc: Scenario) -> None:
        sc.verifier_start_error = RuntimeError("image build failed")

    serial, overlap = await _both_orders(tmp_path, monkeypatch, arrange)

    assert _outcome(overlap) == _outcome(serial)
    rewards, error, _ = _outcome(overlap)
    assert rewards is None
    assert error == (
        "separate verifier sandbox failed to start: RuntimeError: image build failed"
    )
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert overlap.record()["status"] == "sandbox_failed"
    assert overlap.index("agent.stopped") < overlap.index("verify.returned")
    # Not re-graded by workspace recovery (SV1).
    assert recovery_ineligible_reason(overlap.rollout) is not None
    assert overlap.verifier is not None and overlap.verifier.stops == [True]


async def test_kept_containers_are_stopped_not_deleted_before_the_transfer(
    tmp_path, monkeypatch
):
    agent = _docker_agent(tmp_path, keep_containers=True)
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[], agent_env=agent)
    listed: list[list[str]] = []

    async def compose(command, check=True, timeout_sec=None):
        sc.log.append("agent.compose " + " ".join(command))
        return ExecResult(stdout="", stderr="", return_code=0)

    async def docker_cli(args, check=True, *, raw_output=False):
        listed.append(list(args))
        # The kept containers exist, stopped: `docker ps -q` lists none.
        return ExecResult(stdout="", stderr="", return_code=0)

    agent._run_docker_compose_command = compose
    agent._docker_cli = docker_cli

    await _trial(sc)

    composed = [e for e in sc.log if e.startswith("agent.compose")]
    assert composed and all(" stop" in e for e in composed), composed
    assert not any(" down" in e or " rm" in e or " kill" in e for e in composed)
    assert sc.index(composed[0]) < _first_agent_bytes(sc), sc.log
    # The check after the stop asks only for running containers.
    assert listed and listed[0][:2] == ["ps", "-q"], listed
    assert sc.record()["custody"]["agent_stop_error"] is None


# --- where the base order stays ----------------------------------------------


def _assert_base_order(sc: Scenario) -> None:
    assert sc.index("scrape") < sc.index("pack") < sc.index("verifier.create")
    assert "agent.stop" not in sc.log[: sc.index("verify.returned")], sc.log
    assert sc.index("verify.returned") < sc.index("skill_export"), sc.log
    assert sc.index("skill_export") < sc.index("agent.stop"), sc.log
    record = sc.record()
    assert record["order"] == "serial"
    assert record["agent_sandbox_alive_during_verify"] is True
    for key in NEW_SPANS:
        assert key not in sc.rollout.timing, key


async def test_an_externally_owned_sandbox_keeps_the_base_order(tmp_path, monkeypatch):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    sc.rollout._env_externally_owned = True

    await _trial(sc)

    assert sc.index("scrape") < sc.index("pack") < sc.index("verifier.create")
    assert "agent.stop" not in sc.log, sc.log  # the caller's to stop
    record = sc.record()
    assert record["order"] == "serial"
    assert record["agent_sandbox_alive_during_verify"] is True


async def test_the_serial_switch_gives_the_base_order(tmp_path, monkeypatch):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])
    monkeypatch.setenv(PREPARE_ENV, "serial")

    await _trial(sc)

    _assert_base_order(sc)


async def test_a_verify_outside_run_keeps_the_base_order(tmp_path, monkeypatch):
    """An SDK caller or a branch child may still use the sandbox after verify()."""
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[])

    await sc.rollout.verify()
    sc.log.append("verify.returned")
    await sc.rollout.cleanup()

    _assert_base_order(sc)


async def test_run_lets_only_its_own_verify_release_the_agent(tmp_path, monkeypatch):
    rollout = Rollout(RolloutConfig(task_path=tmp_path / "task", agent="nop"))
    seen: list[bool] = []

    async def nothing(*args, **kwargs):
        return None

    async def verify():
        seen.append(rollout._release_agent_before_verify)
        return None

    for name in ("setup", "start", "install_agent", "cleanup"):
        monkeypatch.setattr(rollout, name, nothing)
    monkeypatch.setattr(rollout, "verify", verify)

    await rollout._run_lifecycle()

    assert seen == [True]
    assert rollout._release_agent_before_verify is False


# --- timing and billing --------------------------------------------------------


async def test_agent_stop_to_reward_is_timed_to_the_millisecond_in_both_modes(
    tmp_path, monkeypatch
):
    # Shared mode: verify() runs the verifier in the agent's sandbox.
    shared = Rollout(RolloutConfig(task_path=tmp_path / "task", agent="oracle"))
    clock = FakeClock()
    monkeypatch.setattr(timing_spans, "_clock", clock)
    shared._task = SimpleNamespace(config=TaskConfig())
    shared._env = FakeAgent(Scenario(tmp_path))
    shared._rollout_paths = RolloutPaths(rollout_dir=tmp_path / "shared")
    # K1 writes the kill record into the rollout's own folder.
    shared._rollout_dir = tmp_path / "shared"
    shared._agent_cwd = "/app"

    async def noop(*args, **kwargs):
        return None

    async def graded(*args, **kwargs):
        clock.now += 2.3456
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", noop)
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", noop)
    monkeypatch.setattr(rollout_mod, "_publish_trajectory_for_verifier", noop)
    monkeypatch.setattr(rollout_mod, "_verify_rollout", graded)

    assert await shared.verify() == {"reward": 1.0}
    timing = timing_spans.rounded(shared.timing)
    assert timing["agent_stop_to_reward"] == 2.346
    # The separate verifier's spans are absent in shared mode, not 0.
    for key in NEW_SPANS:
        assert key not in timing, key

    # Separate mode, the agent released first.
    (tmp_path / "separate").mkdir()
    sc = await _scenario(tmp_path / "separate", monkeypatch, trajectory=[], clock=True)
    sc.seconds = {"capture": 0.5, "agent.stop": 0.25, "verifier.test_sh": 1.0004}
    await _trial(sc)
    timing = timing_spans.rounded(sc.rollout.timing)
    assert timing["agent_stop_to_reward"] == 1.75
    assert timing["agent_sandbox_stop_before_verify"] == 0.25
    for key in NEW_SPANS:
        assert key in timing, key


async def test_cleanup_holds_only_what_is_left_after_the_release(tmp_path, monkeypatch):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[], clock=True)
    sc.seconds = {
        "skill_export": 1.0,
        "usage_runtime_stop": 2.0,
        "egress_stop": 4.0,
        "price": 8.0,
        "agent.stop": 16.0,
        "enforce": 32.0,
    }

    await _trial(sc)

    timing = sc.rollout.timing
    assert timing["agent_release"] == 31.0
    assert timing["agent_sandbox_stop_before_verify"] == 16.0
    # cleanup() now holds the enforcement only (its other steps take no time
    # here); the release moved out of it.
    assert timing["cleanup"] == 32.0


async def test_the_record_bills_the_verifier_sandbox_from_start_to_deletion(
    tmp_path, monkeypatch
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[], clock=True)
    sc.capture_waits_for_verifier = True
    sc.seconds = {
        "pack": 0.125,
        "verifier.start": 3.0,
        "agent.stop": 2.0,
        "verifier.upload tar": 0.25,
        "verifier.test_sh": 4.0,
        "verifier.stop": 1.0,
    }

    await _trial(sc)

    # Life: its start (3) + the agent's stop (2) + pack (0.125) + install
    # (0.25) + test.sh (4) + its own stop (1).
    record = sc.record()
    assert record["sandbox_seconds"] == 10.375
    timing = sc.rollout.timing
    assert timing["verifier_sandbox_total"] == 10.375
    assert timing["verifier_sandbox_teardown"] == 1.0
    # Both existed from the verifier's start to the agent stop's return.
    assert timing["verifier_sandbox_overlap"] == 5.0
    assert record["timing"]["verifier_sandbox_overlap"] == 5.0
    # The start was done before the transfer was packed: no wait.
    assert timing["verifier_sandbox_ready_wait"] == 0.0


async def test_on_daytona_the_sandboxes_never_overlap(tmp_path, monkeypatch):
    sc = await _scenario(
        tmp_path, monkeypatch, environment="daytona", trajectory=[], clock=True
    )
    sc.seconds = {"verifier.start": 3.0, "agent.stop": 2.0, "verifier.stop": 1.0}

    await _trial(sc)

    timing = sc.rollout.timing
    assert sc.index("agent.stopped") < sc.index("verifier.create"), sc.log
    assert timing["verifier_sandbox_overlap"] == 0.0
    # The start ran after the stop, beside the packing; scoring waited for
    # whatever of it was left.
    assert 0.0 <= timing["verifier_sandbox_ready_wait"] <= 3.0
    assert sc.record()["sandbox_seconds"] == 4.0


def test_the_budget_adds_only_the_overlap_when_it_is_recorded():
    def spent(timing: dict) -> float:
        guard = BudgetGuard(Budget(max_sandbox_seconds=10_000))
        guard.seed([{"timing": timing}])
        return guard.spent()["sandbox_seconds"]

    # Docker: both existed for 5.5 s of the verifier sandbox's 12 s.
    assert spent(
        {
            "total": 100.0,
            "verifier_sandbox_total": 12.0,
            "verifier_sandbox_overlap": 5.5,
        }
    ) == pytest.approx(105.5)
    # Daytona: the verifier sandbox started after the agent's stopped.
    assert spent(
        {
            "total": 100.0,
            "verifier_sandbox_total": 12.0,
            "verifier_sandbox_overlap": 0.0,
        }
    ) == pytest.approx(100.0)
    # The serial order and older trials: the whole verifier sandbox is extra.
    assert spent({"total": 100.0, "verifier_sandbox_total": 12.0}) == pytest.approx(
        112.0
    )


async def test_the_budget_reads_the_overlap_from_a_finished_trial(
    tmp_path, monkeypatch
):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[], clock=True)
    sc.capture_waits_for_verifier = True
    sc.seconds = {"verifier.start": 3.0, "agent.stop": 2.0, "verifier.test_sh": 4.0}
    await _trial(sc)
    trial = sc.rollout._rollout_dir
    (trial / "timing.json").write_text(
        json.dumps(timing_spans.rounded(sc.rollout.timing))
    )

    guard = BudgetGuard(Budget(max_sandbox_seconds=10_000))
    guard.start("t")
    guard.attempt_done("t", SimpleNamespace(rollout_dir=trial))

    assert guard.spent()["sandbox_seconds"] == pytest.approx(5.0, abs=0.5)


async def test_the_verifier_teardown_is_timed_after_an_exception(tmp_path, monkeypatch):
    sc = await _scenario(tmp_path, monkeypatch, trajectory=[], clock=True)
    sc.seconds = {"verifier.stop": 0.75}
    sc.fail["collect"] = RuntimeError("collect broke")

    with pytest.raises(RuntimeError, match="collect broke"):
        await _trial(sc)

    assert sc.rollout.timing["verifier_sandbox_teardown"] == 0.75
    assert sc.record()["timing"]["verifier_sandbox_teardown"] == 0.75


async def test_the_verifier_seals_its_inspector_copies_before_any_transfer(
    tmp_path, monkeypatch
):
    """V3's protected inspector copies are made in the pristine verifier
    sandbox, before the destination check and before any agent byte."""
    sc = await _scenario(
        tmp_path, monkeypatch, environment="docker", trajectory=_acp_trajectory()
    )

    await _trial(sc)

    destinations = next(
        i for i, e in enumerate(sc.log) if "runtime-verifier-destinations-" in e
    )
    assert sc.index("seal") < destinations < _first_agent_bytes(sc), sc.log
    assert sc.count("seal") == 1, sc.log
