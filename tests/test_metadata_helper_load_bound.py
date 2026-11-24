"""The metadata helper's docker calls wait in step with the host's load.

VM 3 at a 1-minute load of 50-90 on 32 vCPUs: the helper's
``docker run --detach`` outlasted its fixed 60 s ("Command timed out after 60
seconds"), so the policy re-proof was "not proven" (runs k3, l3) and a
verifier block fell back to main and was refused (i4, j4). A busy daemon
answers later while it still works. The bound is 60 s while the load fits
the CPUs, and grows with the load per CPU beyond that, up to 4 times.
"""

from __future__ import annotations

import os

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from tests.test_metadata_helper import MAIN, FakeDocker, _helper

# Their own bounds: a build (_BUILD_SEC) and the script's exec.
_OWN_BOUND = {"build", "exec"}


class Timed(FakeDocker):
    """FakeDocker that keeps each call's bound."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bounds: list[tuple[list[str], float]] = []

    async def __call__(self, args, *, timeout, stdin=None):
        self.bounds.append((list(args), timeout))
        return await super().__call__(args, timeout=timeout, stdin=stdin)

    def start_bound(self) -> float:
        (bound,) = [t for args, t in self.bounds if "--detach" in args]
        return bound

    def call_bounds(self) -> set[float]:
        return {t for args, t in self.bounds if args[0] not in _OWN_BOUND}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(helper_mod, "_READY", {})
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})
    monkeypatch.setattr(helper_mod, "_PREBUILDS", {}, raising=False)
    monkeypatch.setattr(helper_mod, "_HOST_NETNS", {}, raising=False)
    monkeypatch.setattr(helper_mod, "_NETNS_IS_HOST", {}, raising=False)
    monkeypatch.delenv(helper_mod.HELPER_ENV, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)


def _load(monkeypatch, one_minute: float) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (one_minute, one_minute, one_minute))


@pytest.mark.asyncio
async def test_a_loaded_host_gives_the_helpers_start_longer(monkeypatch):
    _load(monkeypatch, 96.0)  # 3 per CPU, as on VM 3
    docker = Timed()

    assert await _helper(docker).installed(MAIN, "echo rules") is not None

    assert docker.start_bound() == 180.0


@pytest.mark.asyncio
async def test_every_docker_call_of_the_helper_follows_the_load(monkeypatch):
    _load(monkeypatch, 64.0)
    docker = Timed()

    await _helper(docker).installed(MAIN, "echo rules")

    assert docker.call_bounds() == {120.0}


@pytest.mark.asyncio
async def test_the_bound_stops_growing_at_four_times(monkeypatch):
    _load(monkeypatch, 3200.0)
    docker = Timed()

    await _helper(docker).installed(MAIN, "echo rules")

    assert docker.call_bounds() == {240.0}


@pytest.mark.asyncio
async def test_a_host_within_its_cpus_keeps_60_s(monkeypatch):
    _load(monkeypatch, 20.0)
    docker = Timed()

    await _helper(docker).installed(MAIN, "echo rules")

    assert docker.call_bounds() == {60.0}


@pytest.mark.asyncio
async def test_an_unreadable_load_keeps_60_s(monkeypatch):
    def unreadable():
        raise OSError("no load average here")

    monkeypatch.setattr(os, "getloadavg", unreadable)
    docker = Timed()

    await _helper(docker).installed(MAIN, "echo rules")

    assert docker.call_bounds() == {60.0}
