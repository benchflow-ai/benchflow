"""Tests for the Environment protocol surface."""

from benchflow.environment.protocol import (
    EnvHandle,
    Environment,
    EnvState,
    ReadinessProbe,
    StateSnapshot,
)


def test_envhandle_carries_endpoints():
    h = EnvHandle(name="clawsbench", endpoints={9001: "http://localhost:9001"})
    assert h.name == "clawsbench"
    assert h.endpoints[9001] == "http://localhost:9001"


def test_envhandle_endpoints_default_empty():
    h = EnvHandle(name="x")
    assert h.endpoints == {}


def test_readiness_probe_records_outcome():
    p = ReadinessProbe(ready=True, checked=["http://localhost:9001/health"])
    assert p.ready is True
    assert p.error is None
    assert p.checked == ["http://localhost:9001/health"]


def test_env_state_holds_query_payload():
    s = EnvState(data={"emails": 3})
    assert s.data["emails"] == 3


def test_state_snapshot_has_id():
    snap = StateSnapshot(id="snap-1")
    assert snap.id == "snap-1"


def test_environment_is_runtime_checkable_protocol():
    class Dummy:
        async def provision(self, ctx): ...
        async def readiness(self): ...
        async def query(self): ...
        async def teardown(self): ...
        async def reset(self): ...
        async def snapshot(self): ...
        async def restore(self, snap): ...

    assert isinstance(Dummy(), Environment)


def test_incomplete_class_is_not_an_environment():
    class Partial:
        async def provision(self, ctx): ...

    assert not isinstance(Partial(), Environment)
