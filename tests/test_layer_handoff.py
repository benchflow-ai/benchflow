"""Guards the fix at integration commit 363d815b."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import env_layers as e
from tests.test_env_layers_agent import BASE


@pytest.mark.parametrize("kind", [e.AGENT, e.PROXY])
@pytest.mark.parametrize("alias", [None, "trial:owned"])
async def test_cancelled_prune_rolls_back_unreturned_acquisition(
    monkeypatch, kind, alias
):
    """Guards 363d815b: no returned result means no claim or trial alias."""
    key = "f" * 64
    layers = e.DockerLayers()
    monkeypatch.setattr(layers, "daemon_id", AsyncMock(return_value="test-daemon"))
    # A different trial's existing claim must survive rollback.
    monkeypatch.setattr(e.DockerLayers, "_claims", {("test-daemon", key): 1})
    monkeypatch.setattr(
        layers,
        "_lock_key",
        AsyncMock(
            return_value=(
                SimpleNamespace(release=lambda: None),
                SimpleNamespace(release=lambda: None),
                0,
            )
        ),
    )
    info = {"Id": "sha256:" + "a" * 64, "RootFS": {"Layers": []}, "Config": {}}
    monkeypatch.setattr(layers, "_layer_info", AsyncMock(side_effect=[None, info]))
    docker = AsyncMock(return_value=e._Run(stdout="", stderr="", code=0))
    monkeypatch.setattr(layers, "_docker", docker)
    entered, release = asyncio.Event(), asyncio.Event()

    async def prune(*args, **kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr(layers, "prune", prune)
    task = asyncio.create_task(
        layers._ensure(
            kind,
            key,
            BASE,
            AsyncMock(),
            started=time.monotonic(),
            alias=alias,
            prune=True,
            keep=1,
            max_bytes=None,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert e.DockerLayers._claims == {("test-daemon", key): 1}
    removed = [
        call.args for call in docker.await_args_list if call.args[:2] == ("image", "rm")
    ]
    assert not removed  # Nothing was acquired before the cancelled prune.


@pytest.mark.parametrize("kind", [e.AGENT, e.PROXY])
async def test_alias_rollback_finishes_under_repeated_cancellation(monkeypatch, kind):
    """Guards 363d815b: cancellation after Docker tags cannot orphan an alias."""
    key, alias = "e" * 64, "trial:owned"
    monkeypatch.setattr(e.DockerLayers, "_claims", {})
    layers = e.DockerLayers()
    monkeypatch.setattr(layers, "daemon_id", AsyncMock(return_value="test-daemon"))
    locks = [SimpleNamespace(release=AsyncMock()) for _ in range(2)]
    # Synchronous lock release, recorded for the ordering assertion.
    released = []
    for lock in locks:
        lock.release = lambda: released.append(True)
    monkeypatch.setattr(layers, "_lock_key", AsyncMock(return_value=(*locks, 0)))
    info = {"Id": "sha256:" + "b" * 64, "RootFS": {"Layers": []}, "Config": {}}
    monkeypatch.setattr(layers, "_layer_info", AsyncMock(return_value=info))
    tagged, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    aliases = set()

    async def docker(*args):
        if args[0] == "tag":
            aliases.add(alias)
            tagged.set()
            await asyncio.Event().wait()
        elif args[:2] == ("image", "rm"):
            cleaning.set()
            await finish.wait()
            aliases.remove(alias)
        return e._Run(stdout="", stderr="", code=0)

    monkeypatch.setattr(layers, "_docker", docker)
    task = asyncio.create_task(
        layers._ensure(
            kind,
            key,
            BASE,
            AsyncMock(),
            started=time.monotonic(),
            alias=alias,
            prune=False,
            keep=1,
            max_bytes=None,
        )
    )
    await tagged.wait()
    task.cancel()
    try:
        async with asyncio.timeout(1):
            await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not released
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
    assert not aliases and not e.DockerLayers._claims
    assert task.cancelled() and len(released) == 2
