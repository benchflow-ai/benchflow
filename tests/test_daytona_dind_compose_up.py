"""Daytona DinD ``compose up`` timeout + network-race retry (REAP-04).

The DinD ``up -d`` previously used a hardcoded 120s regardless of the task's
``build_timeout_sec`` (which the host docker path honours) and had no
network create/attach-race retry. These tests pin the derived timeout and the
single-retry-then-succeed behaviour so a regression to the bare 120s, or a lost
retry, fails CI.
"""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona import _DaytonaDinD
from benchflow.sandbox.daytona_dind import (
    _is_daytona_sdk_error,
    _is_sandbox_disappeared_error,
    _positive_int_env,
    _raise_if_startup_provider_error,
    _with_long_exec_heartbeat,
)
from benchflow.sandbox.protocol import SandboxStartupError


def _strategy(build_timeout_sec: float):
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    env = SimpleNamespace(
        logger=logging.getLogger("test.daytona.dind.up"),
        task_env_config=SimpleNamespace(build_timeout_sec=build_timeout_sec),
    )
    strategy._env = env
    return strategy, env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "build_timeout_sec,expected_timeout",
    [(600, 600), (3600, 3600), (60, 120)],  # 60 -> floored at 120
)
async def test_up_timeout_derived_from_build_budget(
    build_timeout_sec, expected_timeout
):
    strategy, env = _strategy(build_timeout_sec)
    recorded: list[tuple[list[str], int | None]] = []

    async def compose_exec(subcommand, timeout_sec=None):
        recorded.append((subcommand, timeout_sec))
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._compose_exec = compose_exec  # type: ignore[method-assign]

    await strategy._compose_up_with_retry(env)

    assert recorded == [(["up", "-d"], expected_timeout)]


@pytest.mark.asyncio
async def test_network_race_is_retried_then_succeeds(monkeypatch):
    strategy, env = _strategy(600)
    monkeypatch.setattr("benchflow.sandbox.daytona_dind.asyncio.sleep", AsyncNoop())
    attempts: list[int] = []

    async def compose_exec(subcommand, timeout_sec=None):
        attempts.append(1)
        if len(attempts) == 1:
            return ExecResult(
                stdout="",
                stderr=("Error response from daemon: network abc_default not found"),
                return_code=1,
            )
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._compose_exec = compose_exec  # type: ignore[method-assign]

    await strategy._compose_up_with_retry(env)

    assert len(attempts) == 2  # one race, retried once, then success


@pytest.mark.asyncio
async def test_non_race_error_is_not_retried(monkeypatch):
    strategy, env = _strategy(600)
    monkeypatch.setattr("benchflow.sandbox.daytona_dind.asyncio.sleep", AsyncNoop())
    attempts: list[int] = []

    async def compose_exec(subcommand, timeout_sec=None):
        attempts.append(1)
        return ExecResult(stdout="", stderr="image pull failed", return_code=1)

    strategy._compose_exec = compose_exec  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="docker compose up failed"):
        await strategy._compose_up_with_retry(env)
    assert len(attempts) == 1  # non-race errors fail immediately, no retry


def test_docker_daemon_timeout_env_accepts_only_positive_ints(monkeypatch):
    """Guards the 2026-07-01 native-adapter hardening change."""

    monkeypatch.delenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", raising=False)
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 600) == 600

    monkeypatch.setenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", "240")
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 600) == 240

    monkeypatch.setenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", "0")
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 600) == 600

    monkeypatch.setenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", "not-int")
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 600) == 600


def test_sandbox_disappeared_marker_is_structured_startup_error():
    exc = RuntimeError(
        "Failed to get session command: not found: sandbox container not found: "
        "failed to inspect sandbox container sb-123: No such container"
    )
    assert _is_sandbox_disappeared_error(exc)
    env = SimpleNamespace(
        _sandbox=SimpleNamespace(id="sb-123"),
        task_env_config=SimpleNamespace(build_timeout_sec=2400),
    )
    with pytest.raises(SandboxStartupError) as raised:
        _raise_if_startup_provider_error(env, exc)
    assert raised.value.diagnostic.sandbox_id == "sb-123"
    assert raised.value.diagnostic.sandbox_state == "not_found"


def test_daytona_sdk_error_during_dind_start_is_structured_startup_error():
    daytona_error_type = type(
        "DaytonaError",
        (Exception,),
        {"__module__": "daytona.common.errors"},
    )
    exc = daytona_error_type("Failed to get session command: ")
    assert _is_daytona_sdk_error(exc)
    env = SimpleNamespace(
        _sandbox=SimpleNamespace(id="sb-456"),
        task_env_config=SimpleNamespace(build_timeout_sec=1800),
    )
    with pytest.raises(SandboxStartupError) as raised:
        _raise_if_startup_provider_error(env, exc)
    assert raised.value.diagnostic.sandbox_id == "sb-456"
    assert "Daytona DinD startup failed" in str(raised.value)


def test_long_exec_heartbeat_only_wraps_long_commands():
    command = "python preprocess.py"
    assert _with_long_exec_heartbeat(command, 599) == command
    wrapped = _with_long_exec_heartbeat(command, 600)
    assert command in wrapped
    assert "Daytona DinD command still running" in wrapped
    assert 'exit "$bf_rc"' in wrapped


def test_long_exec_heartbeat_returns_promptly_after_fast_command():
    """Guards the heartbeat cleanup race introduced by PR #894."""
    wrapped = _with_long_exec_heartbeat("printf done", 600)
    assert 'kill "$bf_heartbeat_pid"' not in wrapped
    assert ': > "$bf_stop"' in wrapped
    result = subprocess.run(
        ["bash", "-lc", wrapped],
        capture_output=True,
        check=True,
        text=True,
        timeout=2,
    )

    assert result.stdout == "done"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_pre_compose_hook_runs_inside_uploaded_environment():
    strategy, _env = _strategy(3600)
    captured: list[dict[str, object]] = []

    strategy._compose_env_vars = lambda: {"MAIN_IMAGE_NAME": "bf_task"}  # type: ignore[method-assign]

    async def vm_exec(command, cwd=None, env=None, timeout_sec=None):
        captured.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
            }
        )
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._vm_exec = vm_exec  # type: ignore[method-assign]

    await strategy._run_pre_compose_hook()

    assert captured == [
        {
            "command": (
                "if [ -f /benchflow/environment/benchflow-pre-compose.sh ]; then "
                "chmod +x /benchflow/environment/benchflow-pre-compose.sh && "
                "/benchflow/environment/benchflow-pre-compose.sh; fi"
            ),
            "cwd": "/benchflow/environment",
            "env": {"MAIN_IMAGE_NAME": "bf_task"},
            "timeout_sec": 3600,
        }
    ]


@pytest.mark.asyncio
async def test_pre_compose_hook_failure_raises():
    strategy, _env = _strategy(60)
    strategy._compose_env_vars = lambda: {}  # type: ignore[method-assign]

    async def vm_exec(command, cwd=None, env=None, timeout_sec=None):
        return ExecResult(stdout="setup failed", stderr="", return_code=42)

    strategy._vm_exec = vm_exec  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="setup failed"):
        await strategy._run_pre_compose_hook()


@pytest.mark.asyncio
async def test_denylist_build_uploads_every_referenced_compose_file(
    monkeypatch, tmp_path
):
    """Guards the DinD overlay fix against the regression introduced by f396c355."""
    from benchflow.sandbox import daytona as daytona_module

    class FakeClientManager:
        @classmethod
        async def get_instance(cls):
            return object()

    monkeypatch.setattr(daytona_module, "DaytonaClientManager", FakeClientManager)
    monkeypatch.setattr(
        daytona_module,
        "Resources",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        daytona_module,
        "Image",
        SimpleNamespace(base=lambda image: SimpleNamespace(image=image)),
    )
    monkeypatch.setattr(
        daytona_module,
        "CreateSandboxFromImageParams",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    available_paths: set[str] = set()

    async def upload_file(_source, destination):
        available_paths.add(destination)

    async def upload_dir(_source, destination):
        available_paths.add(f"{destination}/docker-compose.yaml")

    task_config = SimpleNamespace(
        allow_internet=True,
        build_timeout_sec=1200,
        cpus=4,
        docker_image=None,
        memory_mb=8192,
        network_mode="denylist",
        storage_mb=10240,
    )
    env = SimpleNamespace(
        _auto_delete_interval=0,
        _auto_stop_interval=0,
        _create_sandbox=AsyncMock(),
        _kwargs={},
        _sdk_upload_dir=upload_dir,
        _sdk_upload_file=upload_file,
        environment_dir=tmp_path,
        environment_name="task",
        logger=logging.getLogger("test.daytona.dind.overlays"),
        task_env_config=task_config,
    )
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    strategy._env = env
    strategy._vm_exec = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    strategy._wait_for_docker_daemon = AsyncMock()
    strategy._run_pre_compose_hook = AsyncMock()
    strategy._compose_up_with_retry = AsyncMock()
    strategy._wait_for_main_container = AsyncMock()

    referenced_at_build: list[str] = []

    async def compose_exec(subcommand, timeout_sec=None):
        del timeout_sec
        if subcommand == ["build"]:
            flags = strategy._compose_file_flags()
            referenced_at_build.extend(
                flags[index + 1] for index, item in enumerate(flags) if item == "-f"
            )
            assert set(referenced_at_build) <= available_paths
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._compose_exec = compose_exec

    await strategy.start(force_build=False)

    assert f"{strategy._COMPOSE_DIR}/docker-compose-net-admin.yaml" in (
        referenced_at_build
    )


class AsyncNoop:
    """A no-op async callable to replace ``asyncio.sleep`` in retry tests."""

    async def __call__(self, *_args, **_kwargs) -> None:
        return None
