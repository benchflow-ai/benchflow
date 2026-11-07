"""PERF-D: the daemon's networks are read beside start()'s Compose checks.

The subnet allocator reads every network of the daemon once (about 1 s on a
daemon with many); start() begins that read before its first Compose freeze,
and _choose_subnets uses it, instead of reading after the freezes and builds.
"""

from __future__ import annotations

from unittest.mock import AsyncMock


async def test_the_daemons_networks_are_read_beside_the_compose_checks(
    tmp_path, monkeypatch
):
    """The allocator's read of every network starts before the first Compose
    freeze of start() (it no longer waits for the freezes and the build)."""
    import threading

    from benchflow.sandbox import docker as docker_mod
    from benchflow.sandbox._docker_subnets import NetworkNeed, NetworkPlan
    from tests.test_docker_direct_exec import FakeDocker, _sandbox, _started

    fake = FakeDocker()
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake)
    events: list[str] = []
    read = threading.Event()

    async def allocator_for(key, docker, *, daemon_display="Docker daemon"):
        events.append("networks read")
        read.set()
        return None

    monkeypatch.setattr(docker_mod, "allocator_for", allocator_for)
    monkeypatch.setattr(
        docker_mod,
        "plan_networks",
        lambda *a, **k: NetworkPlan(needs=[NetworkNeed("default", 28)], pinned=[]),
    )
    sandbox = _sandbox(tmp_path)

    def freeze(*args):
        # A freeze runs in a thread: give a read already started a moment to
        # get there (2 s at most), then record.
        read.wait(timeout=2)
        events.append("compose freeze")

    sandbox.freeze_launch = freeze
    sandbox._docker_subnet_key = lambda: AsyncMock(identify=AsyncMock())
    await _started(sandbox)

    assert "compose freeze" in events
    assert events[0] == "networks read", events
