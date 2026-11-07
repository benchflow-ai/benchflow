"""Which sandbox asks for the kept-service freezer, and when.

The freezer (writable-cgroups on main, docker-compose-kept-freezer.yaml) is
decided once per start, before Compose creates anything, for ordinary
startup and recovery alike, by ``_compose.decide_kept_service_freezer``:

- never for a sandbox that runs no agent code of a task that keeps services
  (nop, a verifier-only sandbox): ``kept_service_runner`` is None, the
  Engine is not even asked;
- for an oracle or agent sandbox only when the Engine's own report (``docker
  info``: version 28 or later, cgroup v2, not rootless, no userns remapping)
  says it can give it, never by retrying a failed start;
- where it cannot: the sandbox starts without it and records the Engine's
  reason; once the trial's identities are measured, a non-root agent that
  keeps services is refused with it and an oracle meets the freezer probe
  (tests/test_kept_freezer_probe_start.py).

Review findings: K1-R13-1 verifier-only sandboxes
asked for it; K1-R13-2 recovery skipped the old Engine handling; K1-R13-3 a
retry keyed on error text could restart a failed entrypoint; K1-R13-4 the
rootless and userns refusals were not recognised. The tests marked "behaviour"
run on 67589e1e unchanged and fail there.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task


def _compose():
    # Imported late: importing benchflow.sandbox._compose before the rest of
    # the package runs into the package's own import cycle.
    from benchflow.sandbox import _compose as module

    return module


def compose_needs_kept_freezer(env) -> bool:
    return _compose().compose_needs_kept_freezer(env)


_SECURITY = ["name=apparmor", "name=seccomp,profile=builtin", "name=cgroupns"]
ENGINES = {
    "able": {
        "ServerVersion": "29.1.3",
        "CgroupVersion": "2",
        "SecurityOptions": _SECURITY,
    },
    "engine-27": {
        "ServerVersion": "27.5.1",
        "CgroupVersion": "2",
        "SecurityOptions": _SECURITY,
    },
    "rootless": {
        "ServerVersion": "28.3.3",
        "CgroupVersion": "2",
        "SecurityOptions": [*_SECURITY, "name=rootless"],
    },
    "userns": {
        "ServerVersion": "28.3.3",
        "CgroupVersion": "2",
        "SecurityOptions": [*_SECURITY, "name=userns"],
    },
    "cgroup-v1": {
        "ServerVersion": "28.3.3",
        "CgroupVersion": "1",
        "SecurityOptions": _SECURITY,
    },
    "unreadable": None,
}


def _info(engine: str) -> str:
    value = ENGINES[engine]
    return "" if value is None else json.dumps(value)


def _task(root: Path, *, compose: bool = False) -> Path:
    task_dir = root / "keeps"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        '[metadata]\n"x-keep-services" = [{ port = 5000, name = "api" }]\n'
        "[agent]\ntimeout_sec = 1\n[verifier]\ntimeout_sec = 1\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose:
        (task_dir / "environment" / "compose.yaml").write_text(
            "services:\n  main: {}\n"
        )
    return task_dir


def _sandbox(tmp_path: Path, kind: str, monkeypatch):
    """A sandbox of *kind* for a task that keeps services, from the factory,
    with its own Engine-info reader."""
    if kind == "remote-docker":
        monkeypatch.setenv("DOCKER_HOST", "ssh://k1@example.invalid")
    backend = "daytona" if kind == "daytona-dind" else kind
    task_dir = _task(tmp_path, compose=kind == "daytona-dind")
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        patch("benchflow.sandbox._sdk_ops.apply"),
    ):
        return _create_sandbox_environment(
            backend,
            Task(task_dir),
            task_dir,
            "rollout",
            RolloutPaths(rollout_dir=tmp_path / "rollout"),
        )


def _uses_freezer(sandbox, kind: str) -> bool:
    if kind == "daytona-dind":
        flags = sandbox._strategy._compose_file_flags()
        return any(flag.endswith("docker-compose-kept-freezer.yaml") for flag in flags)
    return sandbox._DOCKER_COMPOSE_KEPT_FREEZER_PATH in sandbox._docker_compose_paths


# --- the table: (runner, sandbox kind, Engine) -> asked, refused -----------


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("kind", ["docker", "remote-docker", "daytona-dind"])
@pytest.mark.parametrize("runner", [None, "oracle", "agent"])
async def test_the_freezer_is_asked_for_exactly_when(
    tmp_path, monkeypatch, runner, kind, engine
):
    sandbox = _sandbox(tmp_path, kind, monkeypatch)
    sandbox.kept_service_runner = runner
    reader = AsyncMock(return_value=_info(engine))
    able = engine == "able"
    # Nothing is refused at the start: whether the trial keeps
    # anything is known once its identities are measured, and the rollout
    # refuses a kept non-root agent then, with the reason recorded here.
    await _compose().decide_kept_service_freezer(sandbox, reader)
    asked = runner is not None and able
    assert sandbox.kept_service_freezer is asked
    assert _uses_freezer(sandbox, kind) is asked
    refusal = sandbox.kept_service_freezer_refusal
    if runner is not None and not able:
        assert refusal == _compose().engine_freezer_refusal(_info(engine))
    else:
        assert refusal is None
    # A sandbox that runs no agent code does not even ask the Engine.
    assert reader.await_count == (0 if runner is None else 1)


@pytest.mark.parametrize(
    "engine, reason",
    [
        ("engine-27", "predates writable-cgroups"),
        ("rootless", "rootless mode"),
        ("userns", "userns mode"),
        ("cgroup-v1", "cgroup v1"),
        ("unreadable", "could not be read"),
    ],
)
def test_the_engine_report_says_why_not(engine, reason):
    assert reason in (_compose().engine_freezer_refusal(_info(engine)) or "")
    assert _compose().engine_freezer_refusal(_info("able")) is None


@pytest.mark.asyncio
async def test_each_docker_start_decides_before_anything_else(tmp_path, monkeypatch):
    """Ordinary startup and recovery go through the same decision first."""

    class Decided(Exception):
        pass

    async def decide(env, reader):
        raise Decided

    monkeypatch.setattr("benchflow.sandbox.docker.decide_kept_service_freezer", decide)
    sandbox = _sandbox(tmp_path, "docker", monkeypatch)
    with pytest.raises(Decided):
        await sandbox.start(force_build=False)


def test_the_dind_start_decides_before_compose_starts_anything():
    source = (
        Path(__file__).parents[1] / "src/benchflow/sandbox/daytona_dind.py"
    ).read_text()
    start = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "start"
    )
    calls = [
        ast.unparse(node.func).rsplit(".", 1)[-1]
        for node in ast.walk(start)
        if isinstance(node, ast.Call)
    ]
    assert calls.index("decide_kept_service_freezer") < calls.index(
        "_compose_up_with_retry"
    )


# --- one per finding, behaviour (they run on 67589e1e unchanged) ------------


async def _start_decision(sandbox, engine: str) -> None:
    """The start's decision where it exists (absent at 67589e1e)."""
    decide = getattr(_compose(), "decide_kept_service_freezer", None)
    if decide is not None:
        await decide(sandbox, AsyncMock(return_value=_info(engine)))


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["able", "engine-27"])
async def test_k1_r13_1_a_verifier_only_sandbox_asks_for_no_freezer(
    tmp_path, monkeypatch, engine
):
    """A separate verifier's sandbox comes from the same factory and runs no
    agent code: it starts as without K1."""
    sandbox = _sandbox(tmp_path, "docker", monkeypatch)
    await _start_decision(sandbox, engine)
    assert not compose_needs_kept_freezer(sandbox)


@pytest.mark.asyncio
async def test_k1_r13_2_recovery_decides_like_an_ordinary_start(tmp_path, monkeypatch):
    """A recovery start (a baseline image) meets the same decision first."""

    class Decided(Exception):
        pass

    async def decide(env, reader):
        raise Decided

    monkeypatch.setattr(
        "benchflow.sandbox.docker.decide_kept_service_freezer", decide, raising=False
    )
    sandbox = _sandbox(tmp_path, "docker", monkeypatch)
    sandbox._recovery_baseline = MagicMock()
    with pytest.raises(Decided):
        await sandbox.start(force_build=False)


def _compose_up_sandbox(name: str, refusal: str) -> DockerSandbox:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.environment_name = name
    sandbox.logger = MagicMock()
    sandbox.kept_service_freezer = True
    sandbox.kept_service_freezer_optional = True  # 67589e1e's oracle flag
    sandbox.kept_service_runner = "oracle"

    async def compose(command, **kwargs):
        sandbox.calls.append(compose_needs_kept_freezer(sandbox))
        if refusal:
            raise RuntimeError(refusal)

    sandbox.calls = []
    sandbox._run_docker_compose_command = AsyncMock(side_effect=compose)
    return sandbox


@pytest.mark.asyncio
async def test_k1_r13_3_a_failed_start_is_never_retried_without_the_freezer():
    """A task named writable-cgroups whose entrypoint fails: the failure
    propagates after one start, as on integ, never a second start."""
    sandbox = _compose_up_sandbox(
        "writable-cgroups",
        "docker compose up failed for writable-cgroups: container main exited (1)",
    )
    with pytest.raises(RuntimeError, match="exited"):
        await sandbox._run_docker_compose_up()
    assert sandbox.calls == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["rootless", "userns"])
async def test_k1_r13_4_an_oracle_starts_on_a_rootless_or_userns_engine(engine):
    """Docker refuses writable-cgroups there; the oracle's sandbox starts
    without the freezer, decided from the Engine's report, not its error."""
    refusal = "option WritableCgroups conflicts with user namespaces and rootless mode"
    sandbox = _compose_up_sandbox("keeps", "")

    async def compose(command, **kwargs):
        if compose_needs_kept_freezer(sandbox):
            raise RuntimeError(f"Error response from daemon: {refusal}")

    sandbox._run_docker_compose_command = AsyncMock(side_effect=compose)
    await _start_decision(sandbox, engine)
    await sandbox._run_docker_compose_up()
    assert not compose_needs_kept_freezer(sandbox)
