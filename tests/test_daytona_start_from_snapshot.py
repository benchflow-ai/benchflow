"""A Daytona direct sandbox can start straight from a branch snapshot.

Supports isolated branch children (``branch(isolate_children=True)``): each
child needs its own sandbox holding the checkpoint. Without this, a child
sandbox is first created from the task image and then replaced by one
created from the snapshot, which costs a second sandbox creation per child.
``start_from_snapshot`` makes ``start()`` create the sandbox from the
snapshot directly, and never falls back to the task image (a child that
silently started from the task image would not be at the checkpoint).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.protocol import SandboxImage


def _sandbox(strategy_cls=_DaytonaDirect) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox.environment_name = "hello-world-task"
    sandbox.task_env_config = SimpleNamespace(
        cpus=1, memory_mb=2048, storage_mb=10240, docker_image=None
    )
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._snapshot_template_name = None
    sandbox._start_snapshot_ref = None
    sandbox._strategy = strategy_cls(sandbox)
    return sandbox


async def test_direct_start_creates_the_sandbox_from_the_snapshot(monkeypatch):
    created = []

    class Params(SimpleNamespace):
        pass

    monkeypatch.setattr(daytona_mod, "CreateSandboxFromSnapshotParams", Params)
    monkeypatch.setattr(
        daytona_mod, "CreateSandboxFromImageParams", lambda **_: pytest.fail("image")
    )
    monkeypatch.setattr(daytona_mod, "Resources", lambda **kw: kw)
    manager = SimpleNamespace(get_client=AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    sandbox = _sandbox()
    sandbox._create_sandbox = AsyncMock(
        side_effect=lambda params: created.append(params)
    )
    sandbox._sandbox_exec = AsyncMock()
    sandbox._create_attempts = 0

    assert sandbox.start_from_snapshot(SandboxImage("daytona", "bf-snap-x")) is True
    await sandbox._strategy.start(force_build=False)
    assert [params.snapshot for params in created] == ["bf-snap-x"]
    assert created[0].labels  # owner labels as for every BenchFlow sandbox


def test_other_providers_and_dind_do_not_offer_it():
    assert _sandbox().start_from_snapshot(SandboxImage("docker", "img")) is False

    class NoSnapshots(_DaytonaDirect):
        supports_snapshot = False

    assert (
        _sandbox(NoSnapshots).start_from_snapshot(SandboxImage("daytona", "x")) is False
    )
