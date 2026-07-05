"""Daytona DinD ``compose up`` timeout + network-race retry (REAP-04).

The DinD ``up -d`` previously used a hardcoded 120s regardless of the task's
``build_timeout_sec`` (which the host docker path honours) and had no
network create/attach-race retry. These tests pin the derived timeout and the
single-retry-then-succeed behaviour so a regression to the bare 120s, or a lost
retry, fails CI.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona import _DaytonaDinD
from benchflow.sandbox.daytona_dind import _positive_int_env


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
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 180) == 180

    monkeypatch.setenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", "240")
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 180) == 240

    monkeypatch.setenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", "0")
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 180) == 180

    monkeypatch.setenv("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", "not-int")
    assert _positive_int_env("BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 180) == 180


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


class AsyncNoop:
    """A no-op async callable to replace ``asyncio.sleep`` in retry tests."""

    async def __call__(self, *_args, **_kwargs) -> None:
        return None
