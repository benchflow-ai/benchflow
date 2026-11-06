"""PERF-SBX: what the kit cache keys on, when it may be used, and its files."""

import os

import pytest

from benchflow.sandbox import _kit_cache

IMAGE = "sha256:" + "1" * 64


@pytest.fixture(autouse=True)
def enable_kit_cache_mechanism(monkeypatch):
    """Keep the deferred 0.8.4 mechanism covered while production disables it."""
    monkeypatch.setattr(_kit_cache, "ENABLED", True)


def _container(**over):
    record = {
        "Image": IMAGE,
        "Platform": "linux",
        "State": {"Running": True, "Restarting": False},
        "RestartCount": 0,
        "Config": {
            "Env": ["PATH=/usr/bin"],
            "Entrypoint": ["/bin/sh", "-c", "x"],
            "Cmd": ["sleep"],
            "WorkingDir": "/app",
            "User": "",
        },
        "HostConfig": {"Privileged": False},
        "Mounts": [
            {"Type": "bind", "Destination": d}
            for d in ("/logs/agent", "/logs/verifier", "/logs/artifacts")
        ],
    }
    record.update(over)
    return record


def test_a_fresh_container_with_benchflows_mounts_may_use_the_cache():
    assert _kit_cache.eligible(_container(), IMAGE) is None


@pytest.mark.parametrize(
    "over, why",
    [
        ({"Image": "sha256:" + "2" * 64}, "not the task's image"),
        ({"State": {"Running": False}}, "not running"),
        ({"RestartCount": 1}, "restarted"),
        (
            {"Mounts": [{"Type": "bind", "Destination": "/usr/local/lib"}]},
            "a mount other than",
        ),
        (
            {"Mounts": [{"Type": "volume", "Destination": "/logs/agent"}]},
            "a mount other",
        ),
        ({"HostConfig": {"Tmpfs": {"/usr": ""}}}, "Tmpfs set"),
        ({"HostConfig": {"VolumesFrom": ["x"]}}, "VolumesFrom set"),
        (
            {"HostConfig": {"Mounts": [{"Type": "tmpfs", "Target": "/opt"}]}},
            "a mount other",
        ),
    ],
)
def test_anything_else_builds(over, why):
    assert why in (_kit_cache.eligible(_container(**over), IMAGE) or "")


@pytest.mark.parametrize(
    "changes, why",
    [
        ([], None),
        (
            [
                {"Path": "/instruction.md", "Kind": 1},
                {"Path": "/logs", "Kind": 1},
                {"Path": "/logs/agent", "Kind": 1},
                {"Path": "/solution", "Kind": 1},
                {"Path": "/solution/solve.sh", "Kind": 1},
            ],
            None,
        ),
        ([{"Path": "/usr", "Kind": 0}], "/usr changed"),
        ([{"Path": "/logsx", "Kind": 1}], "/logsx changed"),
        ([{"Path": "/solution.d/x", "Kind": 1}], "/solution.d/x changed"),
        ([{"Path": "/opt", "Kind": 0}], "/opt changed"),
        ([{"Kind": 0}], "an unreadable change"),
        (None, "the daemon gave no change list"),
    ],
)
def test_only_benchflows_uploads_may_differ_from_the_image(changes, why):
    assert _kit_cache.unchanged(changes) == why


def test_the_key_changes_with_everything_the_build_sees():
    daemon = {"ID": "d", "Driver": "overlayfs", "SecurityOptions": ["seccomp"]}
    base = _kit_cache.kit_key(_container(), daemon, "recipe")
    assert base == _kit_cache.kit_key(_container(), dict(daemon), "recipe")
    variants = [
        _kit_cache.kit_key(_container(Image="sha256:" + "3" * 64), daemon, "recipe"),
        _kit_cache.kit_key(_container(), daemon, "recipe 2"),
        _kit_cache.kit_key(_container(), {**daemon, "Driver": "vfs"}, "recipe"),
        _kit_cache.kit_key(
            _container(), {**daemon, "SecurityOptions": ["userns"]}, "recipe"
        ),
    ]
    for field, value in (
        ("Env", ["PATH=/opt/conda/bin:/usr/bin"]),
        ("Entrypoint", None),
        ("Cmd", ["bash"]),
        ("WorkingDir", "/srv"),
        ("User", "1000"),
    ):
        config = dict(_container()["Config"], **{field: value})
        variants.append(_kit_cache.kit_key(_container(Config=config), daemon, "recipe"))
    assert len({base, *variants}) == len(variants) + 1


def test_a_kept_kit_is_used_only_while_its_bytes_match(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = "a" * 64
    assert _kit_cache.load(key).tar is None
    _kit_cache.store(key, b"tar bytes")
    entry = _kit_cache.load(key)
    assert entry.tar is not None and entry.tar.read_bytes() == b"tar bytes"
    assert oct(os.stat(_kit_cache.cache_root()).st_mode & 0o777) == "0o700"
    entry.tar.write_bytes(b"tar bytes changed")
    assert _kit_cache.load(key).tar is None
    assert _kit_cache.load("../" + key).tar is None


def test_the_oldest_kits_go_past_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(_kit_cache, "MAX_ENTRIES", 2)
    keys = [c * 64 for c in "abc"]
    for i, key in enumerate(keys):
        _kit_cache.store(key, b"x")
        os.utime(_kit_cache.cache_root() / f"{key}.tar", (i, i))
    _kit_cache.store("d" * 64, b"x")
    kept = sorted(p.stem for p in _kit_cache.cache_root().glob("*.tar"))
    assert kept == ["c" * 64, "d" * 64]


def test_a_staged_tar_keeps_every_member_under_its_stage(tmp_path):
    import io
    import tarfile

    kept = tmp_path / "kept.tar"
    with tarfile.open(kept, "w:") as tar:
        for name, kind in (("rt", tarfile.DIRTYPE), ("rt/bin", tarfile.DIRTYPE)):
            info = tarfile.TarInfo(name)
            info.type, info.mode = kind, 0o755
            tar.addfile(info)
        body = b"#!/bin/sh\n"
        info = tarfile.TarInfo("rt/libexec_sh")
        info.size, info.mode = len(body), 0o755
        tar.addfile(info, io.BytesIO(body))
        link = tarfile.TarInfo("rt/bin/sh")
        link.type, link.linkname = tarfile.SYMTYPE, "../libexec_sh"
        tar.addfile(link)
        hard = tarfile.TarInfo("rt/bin/sh2")
        hard.type, hard.linkname = tarfile.LNKTYPE, "rt/libexec_sh"
        tar.addfile(hard)
    stage = ".benchflow-kit-" + "a" * 32
    data = io.BytesIO(_kit_cache.staged_tar(kept, stage))
    with tarfile.open(fileobj=data) as staged:
        members = {m.name: m for m in staged}
        assert members[stage].isdir() and members[stage].mode == 0o700
        assert members[stage].uid == 0
        assert members[f"{stage}/rt/bin/sh"].linkname == "../libexec_sh"
        assert members[f"{stage}/rt/bin/sh2"].linkname == f"{stage}/rt/libexec_sh"
        assert staged.extractfile(f"{stage}/rt/libexec_sh").read() == body
    with pytest.raises(ValueError):
        _kit_cache.staged_tar(kept, "../x")
    with tarfile.open(kept, "a:") as tar:
        tar.addfile(tarfile.TarInfo("etc-passwd"), io.BytesIO(b""))
    with pytest.raises(ValueError, match="holds"):
        _kit_cache.staged_tar(kept, stage)
