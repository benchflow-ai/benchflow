"""Fallback eligibility is not evidence of a failed build."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tenacity import wait_none

pytest.importorskip("daytona")

from daytona._utils.timeout import with_timeout
from daytona.common.errors import DaytonaError, DaytonaTimeoutError

from benchflow.sandbox import daytona as mod
from benchflow.sandbox import daytona_layers as layers
from benchflow.sandbox import daytona_start
from benchflow.sandbox.protocol import SandboxStartupError
from tests.test_daytona_install_layers import (
    AGENT,
    AGENT_ENV,
    MODEL,
    FakeDaytona,
    _configure,
    _sandbox,
)
from tests.test_dfx_scale_r3 import _fake_client

SCENARIOS = (
    "gateway",
    "create-timeout",
    "toolbox",
    "toolbox-build-failed",
    "recreate-build-failed",
    "start-error-build-text",
    "build-failed",
    "late-success",
    "late-success-sdk",
)


def _build_error():
    return DaytonaError(
        "Sandbox sb-failed entered error state: SandboxState.BUILD_FAILED, "
        "error reason: install failed"
    )


async def exercise_fallback(
    tmp_path, monkeypatch, options, scenario, reuse, *, confirmed_cleanup=True
):
    monkeypatch.setattr(layers, "_BUILT", set())
    monkeypatch.setattr(layers, "_BUILDING", set())
    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1" if reuse else "0")
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOTS", raising=False)
    client = _fake_client(monkeypatch, FakeDaytona)
    original_create = client.create
    requests = []
    deleted = []
    late_success = scenario in {"late-success", "late-success-sdk"}

    async def delete(sandbox_id, **kwargs):
        assert kwargs["wait"] is True
        if not confirmed_cleanup:
            raise RuntimeError("deletion not confirmed")
        deleted.append(sandbox_id)

    def handle(sandbox_id):
        async def remove(**kwargs):
            await delete(sandbox_id, **kwargs)

        return SimpleNamespace(id=sandbox_id, delete=AsyncMock(side_effect=remove))

    async def create(params, timeout=None):
        requests.append("RUN " in params.image.dockerfile())
        attempt = len(requests)
        if late_success and attempt == 1:
            time.sleep(0.03)  # Return a handle just after the outer deadline.
        if late_success and attempt == 2:
            raise _build_error()
        if scenario == "gateway" and attempt <= 6:
            error = DaytonaTimeoutError("provider gateway timeout", status_code=504)
            assert daytona_start.error_kind(error) == "transient"
            raise error
        if scenario == "create-timeout" and attempt <= 3:
            raise TimeoutError("provider read timed out")
        if scenario == "start-error-build-text" and attempt == 1:
            raise DaytonaError(
                "Sandbox sb-failed entered error state: ERROR, "
                "error reason: prior event entered error state: BUILD_FAILED"
            )
        if (scenario == "build-failed" and attempt == 1) or (
            scenario == "recreate-build-failed" and attempt == 2
        ):
            raise _build_error()
        result = await original_create(params, timeout)
        if late_success and attempt == 1:
            result.id = "late-success"
        result.delete = handle(result.id).delete
        return result

    async def listed(query=None, **kwargs):
        yield handle(f"late-{len(requests)}")

    async def get(sandbox_id):
        return handle(sandbox_id)

    monkeypatch.setattr(client, "create", create)
    monkeypatch.setattr(client, "list", listed)
    monkeypatch.setattr(client, "get", get)

    if late_success:
        original_wait_for = asyncio.wait_for
        shortened = False

        async def wait_for(awaitable, timeout):
            nonlocal shortened
            if not shortened and timeout > 60:
                shortened = True
                timeout = 0.01
            return await original_wait_for(awaitable, timeout=timeout)

        monkeypatch.setattr(mod.asyncio, "wait_for", wait_for)
        if scenario == "late-success-sdk":

            @with_timeout()
            async def provider_create(params, timeout=None):
                return await create(params, timeout)

            async def sdk_create(params, timeout=None):
                return await provider_create(
                    params, timeout=0.005 if not requests else timeout
                )

            monkeypatch.setattr(client, "create", sdk_create)

    async def start(label):
        directory = tmp_path / label
        directory.mkdir()
        sandbox = _sandbox(client, directory)
        _configure(sandbox, **options)
        sandbox._create_sandbox = mod.DaytonaSandbox._create_sandbox.retry_with(
            wait=wait_none()
        ).__get__(sandbox)
        empty = dict.fromkeys(mod.SYSTEM_SHELL_EMPTY_VARS, "")
        if label == "first":
            if scenario in {"toolbox", "toolbox-build-failed"}:
                error = (
                    TimeoutError("toolbox read timed out")
                    if scenario == "toolbox"
                    else _build_error()
                )
                sandbox._toolbox_startup_env = AsyncMock(side_effect=[error, empty])
            elif scenario == "recreate-build-failed":
                sandbox._toolbox_startup_env = AsyncMock(
                    side_effect=[{**empty, "BASH_ENV": "/image/startup.sh"}, empty]
                )
        if not confirmed_cleanup:
            with pytest.raises(SandboxStartupError):
                await sandbox._strategy.start(force_build=False)
            assert sandbox._create_outcome_unknown
            assert len(requests) == 1 and not deleted and not layers._FAILED
            if scenario == "toolbox":
                assert sandbox._sandbox is not None
            return sandbox
        await sandbox._strategy.start(force_build=False)
        return sandbox

    first = await start("first")
    if not confirmed_cleanup:
        return
    attempts = {
        "gateway": 6,
        "create-timeout": 3,
        "recreate-build-failed": 2,
        "late-success": 2,
        "late-success-sdk": 2,
    }.get(scenario, 1)
    assert requests == [True] * attempts + [False]
    assert len(deleted) == attempts
    if late_success:
        assert deleted == ["late-success", "sb-failed"]
        assert not first._create_outcome_unknown
    assert first._install_layer_keys == {}
    assert first._install_layers_record["fallback"] == "layer build failed"
    failed_build = scenario == "build-failed"
    assert bool(layers._FAILED) is failed_build
    assert not layers._BUILDING

    second = await start("second")
    assert requests[-1] is not failed_build
    assert bool(second._install_layer_keys) is not failed_build
    assert (second._layer_build_allowance_sec > 0) is not failed_build
    assert len(deleted) == attempts
    if failed_build:
        assert "earlier in this process" in second._install_layers_record["fallback"]
    else:
        assert second._install_layers_record.get("fallback") is None
        assert not layers._FAILED


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("kind", ["guard", "proxy-agent"])
async def test_only_confirmed_build_failures_disable_later_layers(
    tmp_path, monkeypatch, scenario, reuse, kind
):
    """Reviews 2/3 against 8952b9d3 and 9d50e323 failure memory."""
    options = (
        dict(agent=None, model=None, guard_daemon=True, only_if_reused=True)
        if kind == "guard"
        else dict(agent=AGENT, model=MODEL, agent_env=AGENT_ENV, only_if_reused=reuse)
    )
    await exercise_fallback(tmp_path, monkeypatch, options, scenario, reuse)


@pytest.mark.parametrize("scenario", ["gateway", "toolbox", "build-failed"])
async def test_unconfirmed_cleanup_still_refuses_fallback(
    tmp_path, monkeypatch, scenario
):
    """Review 2: failed deletion cannot allocate a plain fallback."""
    await exercise_fallback(
        tmp_path,
        monkeypatch,
        dict(agent=None, model=None, guard_daemon=True),
        scenario,
        False,
        confirmed_cleanup=False,
    )
