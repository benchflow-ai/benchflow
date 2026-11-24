"""A run's start serves the metadata block the same way whether this
process's helper image is built yet or not (warm against cold).

Every run's Docker sandbox comes from the factory, which asks for every rule
from the helper (setup.py: ``_wants_rules_by_helper``). Its start then waits
for this process's helper build (``_decide_rules_by_helper``) before it
decides main's capabilities, so a cold call whose build is still running and
a warm call whose image is built already start main alike and record alike.
Only a sandbox constructed directly keeps the start that does not wait
(test_metadata_helper.test_a_build_still_running_is_not_waited_for).
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from tests.test_metadata_helper import (
    _helper_aware_docker,
    _net_admin_overlay,
    _settle,
)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(helper_mod, "_READY", {})
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})
    monkeypatch.setattr(helper_mod, "_PREBUILDS", {}, raising=False)
    monkeypatch.delenv(helper_mod.HELPER_ENV, raising=False)


async def _start(tmp_path, monkeypatch, *, build_still_running: bool):
    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    if build_still_running:
        docker.build_gate = asyncio.Event()
        asyncio.get_running_loop().call_later(0.3, docker.build_gate.set)
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    tmp_path.mkdir()
    sandbox = cast(Any, direct._sandbox(tmp_path))
    sandbox.task_env_config.arm_cloud_metadata_block()
    # As the factory asks of every run's sandbox (setup.py).
    sandbox._wants_rules_by_helper = True
    try:
        await direct._started(sandbox)
    finally:
        if docker.build_gate is not None:
            docker.build_gate.set()
        await _settle()
    return (
        sandbox.rules_by_helper,
        getattr(sandbox.task_env_config, "cloud_metadata_by_helper", False),
        _net_admin_overlay(docker)[-1],
    )


@pytest.mark.asyncio
async def test_a_cold_start_whose_build_is_running_starts_main_as_a_warm_one(
    tmp_path, monkeypatch
):
    cold = await _start(tmp_path / "cold", monkeypatch, build_still_running=True)
    # The same process, its image now built: a warm call.
    warm = await _start(tmp_path / "warm", monkeypatch, build_still_running=False)

    assert cold == warm == (True, True, False)
