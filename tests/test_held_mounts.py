"""Where a held container's mounts land (sandbox._held_mounts), and when the
task's own restart policy has BenchFlow start a held service again
(_inert_bootstrap._restarts_after): the parts that need no Docker."""

from __future__ import annotations

import asyncio
import io
import tarfile

import pytest

from benchflow.sandbox import _held_mounts as hm
from benchflow.sandbox._inert_bootstrap import _restarts_after


def _archive(member: tarfile.TarInfo, data: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(member, io.BytesIO(data) if data else None)
    return buffer.getvalue()


def _member(name: str, kind: bytes, link: str = "", size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = link
    info.size = size
    return info


def test_the_first_member_says_link_folder_or_file():
    assert hm.first_member(_archive(_member("x", tarfile.SYMTYPE, "/.bfx"))) == (
        "link",
        "/.bfx",
    )
    assert hm.first_member(_archive(_member("lib", tarfile.DIRTYPE))) == ("dir", "")
    assert hm.first_member(_archive(_member("f", tarfile.REGTYPE, size=3), b"abc")) == (
        "file",
        "",
    )


def test_a_long_link_target_in_a_pax_header_is_read():
    target = "/" + "a" * 150 + "/.bfx"
    assert hm.first_member(_archive(_member("x", tarfile.SYMTYPE, target))) == (
        "link",
        target,
    )


def test_a_cut_off_header_is_not_yet_a_member():
    data = _archive(_member("lib", tarfile.DIRTYPE))
    assert hm.first_member(data[:300]) is None


def _reader(tree: dict[str, hm.Entry]):
    seen: list[str] = []

    async def read(path: str) -> hm.Entry:
        seen.append(path)
        return tree.get(path, ("missing", ""))

    return read, seen


@pytest.mark.parametrize(
    "tree, path, lands",
    [
        ({"/data": ("dir", "")}, "/data/bfx", "/data/bfx"),
        ({"/x": ("link", "/.bfx"), "/.bfx": ("dir", "")}, "/x/bfx", "/.bfx/bfx"),
        ({"/x": ("link", ".bfx"), "/.bfx": ("dir", "")}, "/x/bfx", "/.bfx/bfx"),
        ({"/x": ("link", "../../.bfx"), "/.bfx": ("dir", "")}, "/x/bfx", "/.bfx/bfx"),
        ({"/var": ("dir", ""), "/var/run": ("link", "/run")}, "/var/run/x", "/run/x"),
        ({}, "/new/a/../b", "/new/b"),
        ({"/a": ("dir", "")}, "/a/../../b", "/b"),
    ],
    ids=[
        "plain",
        "absolute-link",
        "relative-link",
        "link-past-the-root",
        "var-run",
        "missing",
        "dotdot-at-root",
    ],
)
def test_a_destination_is_followed_as_docker_follows_it(tree, path, lands):
    read, _ = _reader(tree)
    assert asyncio.run(hm.resolve(path, read, {})) == lands


def test_a_link_loop_is_refused():
    read, _ = _reader({"/a": ("link", "/b"), "/b": ("link", "/a")})
    with pytest.raises(hm.HeldStartRefused, match="too many links"):
        asyncio.run(hm.resolve("/a/x", read, {}))


def _inspected(*destinations: str, tmpfs=(), devices=()) -> dict:
    return {
        "Config": {"Labels": {"com.docker.compose.service": "side"}},
        "Mounts": [{"Destination": d, "Type": "bind"} for d in destinations],
        "HostConfig": {
            "Tmpfs": dict.fromkeys(tmpfs, ""),
            "Devices": [{"PathInContainer": d} for d in devices],
        },
    }


def _own(mount: dict) -> bool:
    return mount.get("Destination") == "/.bfx"


BASE = {"/.bfx": ("dir", ""), "/x": ("link", "/.bfx"), "/data": ("dir", "")}


def _check(inspected: dict, tree: dict | None = None) -> None:
    read, _ = _reader({**BASE, **(tree or {})})
    asyncio.run(hm.check(inspected, read, own=_own))


def test_bfx_alone_and_ordinary_mounts_start():
    _check(_inspected("/.bfx", "/data/bfx", "/logs/agent", "/var/run/app"))


@pytest.mark.parametrize(
    "inspected",
    [
        _inspected("/.bfx", "/x/bfx"),
        _inspected("/.bfx", "/x"),
        _inspected("/.bfx", "/x/sub/bfx"),
        _inspected("/.bfx", tmpfs=["/x/bfx"]),
        _inspected("/.bfx", devices=["/x/bfx"]),
        _inspected("/.bfx", "/"),
    ],
    ids=[
        "file-through-link",
        "folder-through-link",
        "beneath",
        "tmpfs",
        "device",
        "root",
    ],
)
def test_a_destination_landing_on_beneath_or_over_bfx_is_refused(inspected):
    with pytest.raises(hm.HeldStartRefused, match="nothing may be mounted"):
        _check(inspected)


def test_a_destination_beneath_another_is_refused():
    with pytest.raises(hm.HeldStartRefused, match="may not sit inside one another"):
        _check(_inspected("/.bfx", "/app", "/app/cache"))


def _ours(*destinations: str):
    def own(mount: dict) -> bool:
        return mount.get("Destination") in destinations

    return own


def test_benchflow_s_own_mounts_may_sit_inside_one_another():
    read, _ = _reader(BASE)
    inspected = _inspected("/.bfx", "/logs", "/logs/verifier")
    asyncio.run(
        hm.check(inspected, read, own=_ours("/.bfx", "/logs", "/logs/verifier"))
    )


@pytest.mark.parametrize(
    "inner, outer",
    [("/logs/verifier/x", "/logs/verifier"), ("/logs/verifier", "/logs")],
    ids=["task-inside-benchflow-s", "benchflow-s-inside-the-task-s"],
)
def test_a_task_mount_nested_with_one_of_benchflow_s_is_refused(inner, outer):
    read, _ = _reader(BASE)
    own = _ours("/.bfx", "/logs/verifier")
    with pytest.raises(hm.HeldStartRefused, match="may not sit inside one another"):
        asyncio.run(hm.check(_inspected("/.bfx", outer, inner), read, own=own))


def test_a_destination_reaching_another_through_a_link_is_nested_too():
    tree = {"/app": ("dir", ""), "/cache": ("link", "/app/cache")}
    with pytest.raises(hm.HeldStartRefused, match="may not sit inside one another"):
        _check(_inspected("/.bfx", "/app", "/cache"), tree)


def test_bfx_that_is_a_link_is_refused():
    with pytest.raises(hm.HeldStartRefused, match="is not a folder"):
        _check(_inspected("/.bfx"), {"/.bfx": ("link", "/tmp/x")})


def test_a_relinked_parent_left_by_an_earlier_run_is_refused():
    # The container's own program renamed /data and left a link to /.bfx.
    with pytest.raises(hm.HeldStartRefused, match=r"lands on /\.bfx/bfx"):
        _check(_inspected("/.bfx", "/data/bfx"), {"/data": ("link", "/.bfx")})


def test_the_destinations_come_from_the_containers_inspect():
    inspected = _inspected("/a", "/b", tmpfs=["/t"], devices=["/dev/x"])
    assert hm.destinations(inspected) == ["/a", "/b", "/dev/x", "/t"]


@pytest.mark.parametrize(
    "policy, exit_code, starts, again",
    [
        ({"Name": "always"}, 0, 9, True),
        ({"Name": "unless-stopped"}, 1, 0, True),
        ({"Name": "on-failure", "MaximumRetryCount": 0}, 1, 4, True),
        ({"Name": "on-failure", "MaximumRetryCount": 2}, 1, 1, True),
        ({"Name": "on-failure", "MaximumRetryCount": 2}, 1, 2, False),
        ({"Name": "on-failure"}, 0, 0, False),
        ({"Name": "no"}, 1, 0, False),
        ({}, 1, 0, False),
        (None, 1, 0, False),
    ],
)
def test_the_tasks_policy_says_when_benchflow_starts_it_again(
    policy, exit_code, starts, again
):
    assert _restarts_after(policy, exit_code, starts) is again
