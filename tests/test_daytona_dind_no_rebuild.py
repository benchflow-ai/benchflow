"""Guards FF-sw-docker C4 at e6151456 against unbudgeted DinD rebuilds."""

from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from tests.test_daytona_dind_followup import strategy


@pytest.mark.parametrize("retry", [False, True])
async def test_dind_up_cannot_spend_a_second_build_budget(tmp_path, monkeypatch, retry):
    env, _vm = strategy(tmp_path, monkeypatch, False)
    impl = _DaytonaDinD(env)
    calls = []

    async def execute(args, **kwargs):
        calls.append((args, kwargs))
        if retry and len(calls) == 1:
            return ExecResult(
                return_code=1,
                stderr="Error response from daemon: network abc_default not found",
            )
        return ExecResult(return_code=0)

    impl._compose_exec = execute
    monkeypatch.setattr("benchflow.sandbox.daytona_dind.asyncio.sleep", AsyncMock())
    await impl._compose_up_with_retry(env)
    assert all("--no-build" in args for args, _ in calls), (
        "up can rebuild outside the five-second budget"
    )


@pytest.mark.parametrize("budget", [0.005, 0.01])
async def test_fractional_build_budget_is_enforced(tmp_path, monkeypatch, budget):
    import asyncio

    env, _ = strategy(tmp_path, monkeypatch, False)
    env.task_env_config.build_timeout_sec = budget
    impl = _DaytonaDinD(env)
    impl._compose_up_with_retry = AsyncMock()
    original = impl._compose_exec

    async def execute(args, **kwargs):
        if args == ["build"]:
            await asyncio.sleep(0.05)
            return ExecResult(return_code=0)
        return await original(args, **kwargs)

    impl._compose_exec = execute
    with pytest.raises(RuntimeError, match="build timed out"):
        await impl.start(False)
    impl._compose_up_with_retry.assert_not_awaited()
