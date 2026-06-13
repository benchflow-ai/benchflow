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


class AsyncNoop:
    """A no-op async callable to replace ``asyncio.sleep`` in retry tests."""

    async def __call__(self, *_args, **_kwargs) -> None:
        return None
