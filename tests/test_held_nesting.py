"""When a held service's mount may hold another of its mounts
(``DockerSandbox._nesting_rule``): read-only as Docker made it, its source
written by nothing of the trial's, and for a bind a source holding no other
mount, on a daemon that sees this machine's files."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchflow.sandbox import _held_mounts as hm
from benchflow.sandbox import docker as docker_module
from benchflow.sandbox.docker import DockerSandbox


def _bind(destination: str, source: str, rw: bool) -> dict:
    return {"Type": "bind", "Destination": destination, "Source": source, "RW": rw}


def _volume(destination: str, name: str, rw: bool) -> dict:
    return {"Type": "volume", "Destination": destination, "Name": name, "RW": rw}


def _container(service: str, *mounts: dict) -> dict:
    return {
        "Config": {"Labels": {"com.docker.compose.service": service}},
        "Mounts": list(mounts),
    }


def _rule(monkeypatch, containers: dict, *, local=True, mounted=(), volumes=None):
    """The rule for *containers* (id to inspect), with *volumes* (name to
    its ``docker volume inspect`` record)."""
    monkeypatch.setattr(
        docker_module, "_daemon_sees_this_filesystem", lambda env: local
    )
    monkeypatch.setattr(docker_module, "_host_mountpoints", lambda: list(mounted))

    async def inspect(cid):
        return containers[cid]

    async def cli(args, check=True, *, raw_output=False):
        record = (volumes or {}).get(args[-1], {"Driver": "local", "Options": None})
        return SimpleNamespace(stdout=json.dumps([record]), return_code=0)

    env = cast(Any, SimpleNamespace(_inspect_container=inspect, _docker_cli=cli))
    env._volume_host_path = lambda name: DockerSandbox._volume_host_path(env, name)
    return asyncio.run(DockerSandbox._nesting_rule(env, list(containers)))


OUTER = _bind("/opt/task-src", "/srv/task", rw=False)
HIDDEN = (
    _volume("/opt/task-src/solution", "trial_a", True),
    _volume("/opt/task-src/tests", "trial_b", True),
)


def test_a_read_only_bind_nothing_writes_may_hold_other_mounts(monkeypatch):
    holds_none = _rule(monkeypatch, {"m": _container("main", OUTER, *HIDDEN)})
    assert holds_none(OUTER) is None


def test_a_bind_docker_made_writable_may_not(monkeypatch):
    writable = {**OUTER, "RW": True}
    holds_none = _rule(monkeypatch, {"m": _container("main", writable, *HIDDEN)})
    assert holds_none(writable) == "it is writable in its service"


@pytest.mark.parametrize(
    "writer",
    [
        _bind("/data", "/srv/task", rw=True),
        _bind("/data", "/srv/task/solution", rw=True),
        _bind("/data", "/srv", rw=True),
    ],
    ids=["same-source", "below", "above"],
)
def test_a_source_another_service_writes_at_above_or_below_may_not(monkeypatch, writer):
    containers = {
        "m": _container("main", OUTER, *HIDDEN),
        "s": _container("side", writer),
    }
    reason = _rule(monkeypatch, containers)(OUTER)
    assert reason and "written through the mount at /data of side" in reason, reason


def test_a_local_volume_bound_to_the_source_counts_as_that_path(monkeypatch):
    volumes = {
        "shared": {
            "Driver": "local",
            "Options": {"type": "none", "o": "bind", "device": "/srv/task"},
        }
    }
    containers = {
        "m": _container("main", OUTER, *HIDDEN),
        "s": _container("side", _volume("/data", "shared", True)),
    }
    reason = _rule(monkeypatch, containers, volumes=volumes)(OUTER)
    assert reason and "written through the mount at /data of side" in reason, reason


def test_benchflow_s_own_writable_mounts_count_too(monkeypatch):
    # The trial's log folder sits inside the outer's source.
    logs = _bind("/logs/verifier", "/srv/task/logs/verifier", rw=True)
    reason = _rule(monkeypatch, {"m": _container("main", OUTER, logs)})(OUTER)
    assert reason and "/logs/verifier" in reason, reason


def test_a_bind_whose_source_holds_another_mount_may_not(monkeypatch):
    holds_none = _rule(
        monkeypatch,
        {"m": _container("main", OUTER, *HIDDEN)},
        mounted=["/srv/task/sub"],
    )
    assert "holds another mount" in str(holds_none(OUTER))


def test_on_a_remote_daemon_a_bind_may_not_but_a_named_volume_may(monkeypatch):
    volume = _volume("/opt/data", "dataset", False)
    containers = {"m": _container("main", OUTER, volume, *HIDDEN)}
    holds_none = _rule(monkeypatch, containers, local=False)
    assert "doesn't share this machine's files" in str(holds_none(OUTER))
    assert holds_none(volume) is None


def test_a_named_volume_another_service_writes_may_not(monkeypatch):
    volume = _volume("/opt/data", "dataset", False)
    containers = {
        "m": _container("main", volume),
        "s": _container("side", _volume("/w", "dataset", True)),
    }
    assert "written through" in str(_rule(monkeypatch, containers)(volume))


def test_a_tmpfs_or_device_may_not_hold_another(monkeypatch):
    holds_none = _rule(monkeypatch, {"m": _container("main", OUTER)})
    assert holds_none(None) == "a tmpfs or a device may not hold another mount"


def _read(tree):
    async def read(path):
        return tree.get(path, ("missing", ""))

    return read


def _inspected(*mounts: dict) -> dict:
    return {**_container("main", *mounts), "HostConfig": {}}


def test_check_lets_a_mount_that_may_hold_others_hold_them():
    tree = {"/.bfx": ("dir", ""), "/opt": ("dir", ""), "/opt/task-src": ("dir", "")}
    inspected = _inspected(_bind("/.bfx", "/bfx", False), OUTER, *HIDDEN)
    asyncio.run(
        hm.check(
            inspected,
            _read(tree),
            own=lambda m: m.get("Destination") == "/.bfx",
            holds_none=lambda m: (
                None if m is not None and m["Destination"] == "/opt/task-src" else "no"
            ),
        )
    )


def test_check_names_why_the_outer_may_not_hold_them():
    tree = {"/.bfx": ("dir", ""), "/opt": ("dir", ""), "/opt/task-src": ("dir", "")}
    inspected = _inspected(_bind("/.bfx", "/bfx", False), OUTER, *HIDDEN)
    with pytest.raises(
        hm.HeldStartRefused, match=r"inside one another \(it is writable"
    ):
        asyncio.run(
            hm.check(
                inspected,
                _read(tree),
                own=lambda m: m.get("Destination") == "/.bfx",
                holds_none=lambda m: "it is writable in its service",
            )
        )


def test_a_link_inside_a_read_only_outer_to_bfx_is_still_refused():
    tree = {
        "/.bfx": ("dir", ""),
        "/opt": ("dir", ""),
        "/opt/task-src": ("dir", ""),
        "/opt/task-src/solution": ("link", "/.bfx"),
    }
    inspected = _inspected(_bind("/.bfx", "/bfx", False), OUTER, *HIDDEN)
    with pytest.raises(hm.HeldStartRefused, match=r"lands on /\.bfx"):
        asyncio.run(
            hm.check(
                inspected,
                _read(tree),
                own=lambda m: m.get("Destination") == "/.bfx",
                holds_none=lambda m: None,
            )
        )
