"""What a trial and the prune do when a layer's host lock is busy or unusable (PY8 U13).

Against the in-memory stand-in ``docker`` of ``tests/test_env_layers_agent.py``.
A trial that cannot take the lock in time (``lock: timeout``) or at all
(``lock: process``) installs fresh and never reuses, builds or tags the shared
layer (before the lock only the task image's libc is read from a layer's labels); the prune skips a key whose lock it cannot take and holds the lock
through every removal it makes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _layer_lock, env_layers
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import (
    AGENT,
    AgentLayerRecipe,
    DockerLayers,
    LayerError,
    agent_layer_key,
    layer_tag,
)
from tests.test_env_layers_agent import FakeDocker

DAEMON = "fake-daemon-1"
RECIPE = AgentLayerRecipe(
    agent="fake-agent", version="1.2.3", install_cmd="true", timeout_sec=0
)


class LockCheckingDocker(FakeDocker):
    """Notes, at each removing command, whether its key's lock was held."""

    def __init__(self) -> None:
        super().__init__()
        self.held_at_removal: list[tuple[str, bool]] = []

    def _held(self, key: str) -> bool:
        probe = _layer_lock.HostLock(DAEMON, AGENT, key)
        if probe.try_acquire():
            probe.release()
            return False
        return True

    async def run(self, *args, stdin=None, timeout=0):
        argv = list(args)
        if argv[:1] == ["info"] and "{{.ID}}" in argv:
            self.calls.append(argv)
            return self._ok(DAEMON + "\n")
        if argv[:2] == ["image", "rm"] and argv[2].startswith("benchflow-layer:agent-"):
            key = argv[2].removeprefix("benchflow-layer:agent-")
            self.held_at_removal.append((f"image {key[:8]}", self._held(key)))
        if argv[:2] == ["builder", "prune"]:
            key = argv[-1].split("key=")[1].split(";")[0]
            self.held_at_removal.append((f"records {key[:8]}", self._held(key)))
        return await super().run(*args, stdin=stdin, timeout=timeout)


@pytest.fixture
def docker(monkeypatch, tmp_path_factory) -> LockCheckingDocker:
    monkeypatch.setenv(
        _layer_lock.LOCK_DIR_ENV, str(tmp_path_factory.mktemp("held") / "locks")
    )
    monkeypatch.setenv("BENCHFLOW_AGENT_LAYER_MIN_FREE_GB", "0")
    DockerLayers._claims.clear()
    DockerLayers._libc_cache.clear()
    _layer_lock.clear_process_locks()
    fake = LockCheckingDocker()

    async def run(self, *args, stdin=None, timeout=0):
        return await fake.run(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(DockerLayers, "_docker", run)
    yield fake
    DockerLayers._claims.clear()


def _fresh_record(error: LayerError) -> dict:
    sandbox = SimpleNamespace(
        logger=logging.getLogger("test"), environment_name="task", layer_records={}
    )
    DockerSandbox._layer_failed(sandbox, AGENT, "fake-agent", error)  # type: ignore[arg-type]
    return sandbox.layer_records[AGENT]


async def test_past_the_wait_bound_the_trial_installs_fresh_without_the_cache(
    docker, monkeypatch
):
    monkeypatch.setattr(env_layers, "LOCK_WAIT_GRACE_SEC", 0.3)
    docker.add_image("bf__task")
    layers = DockerLayers()
    base = await layers.inspect_base("bf__task")
    key = agent_layer_key(RECIPE, base)
    # Another process holds the key (building it) for longer than the bound.
    holder = _layer_lock.HostLock(DAEMON, AGENT, key)
    assert holder.try_acquire()
    try:
        docker.calls.clear()
        with pytest.raises(env_layers.LayerLockFallback) as caught:
            await layers.ensure_agent_layer(RECIPE, "bf__task", alias="bf__layer__t:1")
        during = list(docker.calls)
        # Meanwhile a cleanup of that key finds it held and leaves it.
        docker.add_image(
            layer_tag(AGENT, key),
            labels={"benchflow.layer": AGENT, "benchflow.layer.key": key},
        )
        report = await layers.prune(AGENT, keep=0, only=[key])
    finally:
        holder.release()
    error = caught.value
    assert error.lock == "timeout" and error.wait_sec >= 0.3
    # It never read, built or tagged the shared layer.
    assert not [c for c in during if c[:1] in (["build"], ["tag"])]
    assert not [c for c in during if any(layer_tag(AGENT, key) in a for a in c)]
    assert "bf__layer__t:1" not in docker.tags
    assert key in report.skipped and layer_tag(AGENT, key) in docker.tags
    record = _fresh_record(error)
    assert record["status"] == "fresh"
    assert record["lock"] == "timeout" and record["lock_wait_sec"] >= 0.3


async def test_an_unusable_lock_folder_gives_lock_process_and_a_fresh_install(
    docker, monkeypatch, tmp_path
):
    planted = tmp_path / "planted"
    planted.write_text("a file where the lock folder should be")
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(planted))
    docker.add_image("bf__task")
    layers = DockerLayers()
    docker.calls.clear()
    with pytest.raises(env_layers.LayerLockFallback) as caught:
        await layers.ensure_agent_layer(RECIPE, "bf__task", alias="bf__layer__t:1")
    assert caught.value.lock == "process"
    assert not [c for c in docker.calls if c[:1] in (["build"], ["tag"])]
    # Before the lock only the task image's libc is read from layers: a bare
    # inspect of this recipe's tag for each candidate libc, the labels the
    # filtered listing reads (SPEED-FIX). Nothing else names a layer tag.
    named = [c for c in docker.calls if any("benchflow-layer:" in a for a in c)]
    assert named, "nothing was inspected before lock fallback"
    base = await layers.inspect_base("bf__task")
    candidates = [
        [layer_tag(AGENT, agent_layer_key(RECIPE, dataclasses.replace(base, libc=c)))]
        for c in env_layers._LIBCS
    ]
    assert all(c[:2] == ["image", "inspect"] and c[2:] in candidates for c in named)
    record = _fresh_record(caught.value)
    assert record["lock"] == "process" and "cannot be used" in record["reason"]


async def test_cleanup_with_an_unusable_lock_keeps_everything(
    docker, monkeypatch, tmp_path
):
    docker.add_image("bf__task")
    first = await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
    DockerLayers.release(first.key, daemon=first.daemon_id)
    planted = tmp_path / "planted"
    planted.write_text("x")
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(planted))
    report = await DockerLayers().prune(AGENT, keep=0)
    assert layer_tag(AGENT, first.key) in docker.tags
    assert docker.records
    assert "cannot be used" in report.skipped[first.key]


async def test_the_prune_holds_each_keys_lock_through_its_removals(docker):
    docker.add_image("bf__task")
    layers = DockerLayers()
    built = await layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
    DockerLayers.release(built.key, daemon=built.daemon_id)
    # An orphan: records of a key whose image is gone.
    orphan = "e" * 64
    docker.records.append(
        {"id": "rec-orphan", "description": f"x {env_layers.marker(AGENT, orphan)} y"}
    )
    report = await layers.prune(AGENT, keep=0)
    assert built.key in report.removed and orphan in report.records_pruned
    assert docker.held_at_removal, "nothing was removed"
    assert all(held for _what, held in docker.held_at_removal), docker.held_at_removal
    # Released after: a trial takes the key at once.
    probe = _layer_lock.HostLock(DAEMON, AGENT, built.key)
    assert probe.try_acquire()
    probe.release()


async def test_a_record_names_the_lock_mode_and_wait(docker):
    docker.add_image("bf__task")
    layers = DockerLayers()
    miss = await layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
    hit = await layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
    for result in (miss, hit):
        record = result.record()
        assert record["lock"] == "host"
        assert isinstance(record["lock_wait_sec"], float)


async def test_a_waiter_rechecks_and_hits_after_the_holder_builds(docker):
    """In one process: the second trial waits on the key, then finds the layer."""
    docker.add_image("bf__task")
    gate = asyncio.Event()
    started = asyncio.Event()
    real = docker.run

    async def slow_build(*args, stdin=None, timeout=0):
        if args[:1] == ("build",):
            started.set()
            await gate.wait()
        return await real(*args, stdin=stdin, timeout=timeout)

    docker.run = slow_build  # type: ignore[method-assign]
    layers = DockerLayers()
    first = asyncio.create_task(
        layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
    )
    await started.wait()
    second = asyncio.create_task(
        layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
    )
    await asyncio.sleep(0.1)
    assert not second.done()
    gate.set()
    a, b = await asyncio.gather(first, second)
    assert (a.status, b.status) == ("miss", "hit")
    assert sum(1 for c in docker.calls if c[:1] == ["build"]) == 1
    assert b.lock_wait_sec > 0


@pytest.mark.parametrize("late_inspects", [False, True])
async def test_lock_fallback_guard_rejects_missing_inspects(
    docker, monkeypatch, tmp_path, late_inspects
):
    """Guard SPEED-FIX r2 against 2b5c678e6's vacuous / contaminated assertion."""
    ensure = DockerLayers.ensure_agent_layer
    inspect = DockerLayers.inspect_base
    removed = []

    async def without_inspects(self, *args, **kwargs):
        try:
            return await ensure(self, *args, **kwargs)
        finally:
            removed.extend(
                c for c in docker.calls if any("benchflow-layer:" in a for a in c)
            )
            docker.calls[:] = [c for c in docker.calls if c not in removed]

    async def inspect_after_fallback(self, *args, **kwargs):
        result = await inspect(self, *args, **kwargs)
        if late_inspects:
            docker.calls.extend(removed)
        return result

    monkeypatch.setattr(DockerLayers, "ensure_agent_layer", without_inspects)
    monkeypatch.setattr(DockerLayers, "inspect_base", inspect_after_fallback)
    with pytest.raises(AssertionError, match="nothing was inspected"):
        await test_an_unusable_lock_folder_gives_lock_process_and_a_fresh_install(
            docker, monkeypatch, tmp_path
        )
