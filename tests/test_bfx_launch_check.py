"""The launch check admits bfx's own mounts and nothing more
(TRUSTED-RUNTIME.md §6): the verified host folder bound read-only at /.bfx,
or the pinned harness volume mounted read-only and uncopied at /.bfx. A
task's Compose that mounts either another way, or anything else there, is
still refused."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from benchflow.task.compose_env import project_check


@pytest.fixture
def check(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    harness = tmp_path / "bfx"
    harness.mkdir()
    staged = SimpleNamespace(environment_dir=task / "environment", task_dir=task)

    def run(
        volumes,
        *,
        remote=False,
        binds=((harness, "/.bfx"),),
        named=(),
        harness_volumes=(),
        services=None,
    ):
        checker = project_check(
            staged,
            [],
            names=frozenset(),
            environ={},
            remote=remote,
            agent=False,
            project="trial" if named else None,
            read_only_binds=binds,
            read_only_volumes=harness_volumes,
        )
        project = {"services": services or {"main": {"volumes": volumes}}}
        if named:
            project["volumes"] = {n: {"external": True, "name": n} for n in named}
        return checker.check(project)

    return run, harness, tmp_path


def _bind(source, target, read_only=True):
    return {
        "type": "bind",
        "source": str(source),
        "target": target,
        "read_only": read_only,
    }


def test_the_read_only_bfx_bind_is_admitted(check):
    run, harness, _ = check
    assert run([_bind(harness, "/.bfx")]) == []


@pytest.mark.parametrize(
    "volume",
    [
        lambda h, o: _bind(h, "/.bfx", read_only=False),
        lambda h, o: _bind(h, "/opt/bfx"),
        lambda h, o: _bind(o, "/.bfx"),
    ],
    ids=["writable", "other-target", "other-source"],
)
def test_anything_but_that_bind_is_refused(check, volume):
    run, harness, tmp_path = check
    other = tmp_path / "other"
    other.mkdir()
    refused = run([volume(harness, other)])
    assert refused and any(
        "outside the task's folder" in (r.shown or "")
        or "nothing may be mounted at or beneath it" in (r.shown or "")
        for r in refused
    )


def test_a_trailing_slash_on_the_target_is_the_same_target(check):
    run, harness, _ = check
    assert run([_bind(harness, "/.bfx/")]) == []


def test_a_link_to_the_folder_is_the_folder(check):
    run, harness, tmp_path = check
    link = tmp_path / "link"
    link.symlink_to(harness)
    assert run([_bind(link, "/.bfx")]) == []


def test_without_bfx_the_folder_is_outside_the_task(check):
    run, harness, _ = check
    refused = run([_bind(harness, "/.bfx")], binds=())
    assert refused


def test_a_remote_daemon_never_takes_the_bind(check):
    run, harness, _ = check
    assert run([_bind(harness, "/.bfx")], remote=True)


VOLUME = "benchflow-bfx-0123456789ab"


def _volume_mount(target="/.bfx", read_only=True, nocopy=True, source=VOLUME):
    return {
        "type": "volume",
        "source": source,
        "target": target,
        "read_only": read_only,
        "volume": {"nocopy": nocopy},
    }


def test_the_read_only_uncopied_bfx_volume_is_admitted(check):
    run, _, _ = check
    assert (
        run([_volume_mount()], named=[VOLUME], harness_volumes=[(VOLUME, "/.bfx")])
        == []
    )


@pytest.mark.parametrize(
    "mount",
    [
        _volume_mount(read_only=False),
        _volume_mount(nocopy=False),
        _volume_mount(target="/opt"),
    ],
    ids=["writable", "copied", "other-target"],
)
def test_the_bfx_volume_mounted_another_way_is_refused(check, mount):
    run, _, _ = check
    refused = run([mount], named=[VOLUME], harness_volumes=[(VOLUME, "/.bfx")])
    assert refused and "BenchFlow's volume" in (refused[0].shown or "")


def test_a_sidecar_mounting_the_bfx_volume_writable_is_refused(check):
    run, _, _ = check
    services = {
        "main": {"volumes": [_volume_mount()]},
        "side": {"volumes": [_volume_mount(target="/data", read_only=False)]},
    }
    refused = run(
        [], named=[VOLUME], harness_volumes=[(VOLUME, "/.bfx")], services=services
    )
    assert refused and "service side" in (refused[0].shown or "")


def test_other_external_volumes_stay_refused(check):
    run, _, _ = check
    other = "benchflow-bfx-ffffffffffff"
    refused = run(
        [_volume_mount(source=other)],
        named=[other],
        harness_volumes=[(VOLUME, "/.bfx")],
    )
    assert any("existing volume" in (r.shown or "") for r in refused)


def test_without_bfx_its_volume_is_an_external_volume(check):
    run, _, _ = check
    refused = run([_volume_mount()], named=[VOLUME])
    assert any("existing volume" in (r.shown or "") for r in refused)


# M-1: nothing but BenchFlow's own read-only helper mount sits at or beneath
# /.bfx, whatever the mount's kind and whoever declared it.

BENEATH = "nothing may be mounted at or beneath it"


def _beneath(refused):
    return [r for r in refused if BENEATH in (r.shown or "")]


@pytest.mark.parametrize("target", ["/.bfx/bfx", "/.bfx/sub", "/.bfx/./bfx", "/.bfx"])
def test_a_task_bind_from_its_own_folder_at_or_beneath_bfx_is_refused(check, target):
    run, harness, tmp_path = check
    tool = tmp_path / "task" / "environment" / "tool"
    tool.write_text("#!/bin/sh\n")
    refused = run([_bind(harness, "/.bfx"), _bind(tool, target)])
    assert _beneath(refused), refused


@pytest.mark.parametrize(
    "mount",
    [
        {"type": "volume", "source": "data", "target": "/.bfx/bfx"},
        {"type": "tmpfs", "target": "/.bfx/sub"},
        {"type": "image", "source": "busybox", "target": "/.bfx/bfx"},
    ],
    ids=["volume", "tmpfs", "image"],
)
def test_any_other_kind_of_mount_beneath_bfx_is_refused(check, mount):
    run, harness, _ = check
    assert _beneath(run([_bind(harness, "/.bfx"), mount]))


def test_the_short_tmpfs_form_beneath_bfx_is_refused(check):
    run, harness, _ = check
    services = {
        "main": {"volumes": [_bind(harness, "/.bfx")], "tmpfs": ["/.bfx/x:size=1m"]}
    }
    assert _beneath(run([], services=services))


def test_a_sidecar_mount_beneath_bfx_is_refused(check):
    run, harness, _ = check
    services = {
        "main": {"volumes": [_bind(harness, "/.bfx")]},
        "side": {"tmpfs": "/.bfx"},
    }
    assert _beneath(run([], services=services))


def test_without_bfx_a_mount_at_bfx_is_still_refused(check):
    run, _, tmp_path = check
    tool = tmp_path / "task" / "environment" / "tool"
    tool.write_text("#!/bin/sh\n")
    assert _beneath(run([_bind(tool, "/.bfx/bfx")], binds=()))


def test_a_volume_beneath_bfx_next_to_the_harness_volume_is_refused(check):
    run, _, _ = check
    refused = run(
        [_volume_mount(), _volume_mount(target="/.bfx/bfx", source="data")],
        named=[VOLUME],
        harness_volumes=[(VOLUME, "/.bfx")],
    )
    assert _beneath(refused)


def test_paths_that_only_share_the_prefix_are_not_beneath(check):
    run, harness, _ = check
    assert run([_bind(harness, "/.bfx"), {"type": "tmpfs", "target": "/.bfxx"}]) == []


# M-1, every start path: main as created (its inspect) is checked before any
# task process or bfx runs, so a mount the Compose check cannot see (an
# image's VOLUME, a replayed snapshot, the direct Engine path) is caught too.

from benchflow._bfx import delivery  # noqa: E402

FOLDER = "/home/u/.cache/benchflow/bfx/3f84ec2b06023eb4-x86_64"


def _mount(kind, source, destination, rw=False):
    key = "Name" if kind == "volume" else "Source"
    return {"Type": kind, key: source, "Destination": destination, "RW": rw}


def _inspected(*mounts, tmpfs=None):
    return {"Mounts": list(mounts), "HostConfig": {"Tmpfs": tmpfs or {}}}


def test_main_with_only_our_read_only_bind_passes():
    ours = _mount("bind", FOLDER, "/.bfx")
    other = _mount("bind", "/srv/logs", "/logs", rw=True)
    assert (
        delivery.mount_problems(_inspected(ours, other), mode="bind", source=FOLDER)
        == []
    )


def test_main_with_only_our_read_only_volume_passes():
    ours = _mount("volume", VOLUME, "/.bfx")
    assert delivery.mount_problems(_inspected(ours), mode="volume", source=VOLUME) == []


@pytest.mark.parametrize(
    "extra",
    [
        _mount("bind", "/tmp/stage/environment/tool", "/.bfx/bfx"),
        _mount("volume", "0123abcd", "/.bfx/sub", rw=True),
        _mount("tmpfs", "", "/.bfx/bfx"),
    ],
    ids=["task-bind", "image-volume", "tmpfs"],
)
def test_main_with_a_mount_beneath_ours_is_refused(extra):
    ours = _mount("bind", FOLDER, "/.bfx")
    assert delivery.mount_problems(_inspected(ours, extra), mode="bind", source=FOLDER)


def test_a_cli_tmpfs_beneath_bfx_is_refused():
    ours = _mount("bind", FOLDER, "/.bfx")
    assert delivery.mount_problems(
        _inspected(ours, tmpfs={"/.bfx/x": ""}), mode="bind", source=FOLDER
    )


@pytest.mark.parametrize(
    "mount",
    [
        _mount("bind", FOLDER, "/.bfx", rw=True),
        _mount("bind", "/other", "/.bfx"),
        _mount("volume", VOLUME, "/.bfx"),
    ],
    ids=["writable", "other-source", "other-kind"],
)
def test_main_whose_bfx_mount_is_not_ours_is_refused(mount):
    assert delivery.mount_problems(_inspected(mount), mode="bind", source=FOLDER)


def test_main_without_our_mount_is_refused_when_bfx_is_delivered():
    assert delivery.mount_problems(_inspected(), mode="bind", source=FOLDER)


def test_without_bfx_any_mount_there_is_reported():
    assert delivery.mount_problems(
        _inspected(_mount("bind", FOLDER, "/.bfx")), mode=None, source=None
    )
    assert delivery.mount_problems(_inspected(), mode=None, source=None) == []


@pytest.mark.asyncio
async def test_the_bind_proof_runs_the_check_on_mains_inspect(monkeypatch):
    from benchflow.sandbox import _bind_targets

    class Env:
        seen: ClassVar[list] = []

        async def _inspect_container(self, container):
            return {
                "Config": {"Labels": {"com.docker.compose.service": "main"}},
                **_inspected(_mount("bind", "/x", "/.bfx/bfx")),
            }

        def _check_bfx_mounts(self, inspected):
            self.seen.append(inspected)
            raise RuntimeError("refused")

    async def no_targets(env, container, inspected):
        return {}

    monkeypatch.setattr(_bind_targets, "inspect_targets", no_targets)
    with pytest.raises(RuntimeError, match="refused"):
        await _bind_targets.pin_project(Env(), "a" * 64, None)
    assert len(Env.seen) == 1


def test_the_docker_sandbox_check_refuses_a_mount_beneath_its_bind():
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.bfx_delivery = "bind"
    sandbox._bfx_source = FOLDER
    ours = _mount("bind", FOLDER, "/.bfx")
    sandbox._check_bfx_mounts(_inspected(ours))
    with pytest.raises(RuntimeError, match=r"beneath /\.bfx"):
        sandbox._check_bfx_mounts(
            _inspected(ours, _mount("bind", "/tmp/tool", "/.bfx/bfx"))
        )


def test_a_snapshot_with_a_mount_beneath_bfx_is_not_replayed():
    from benchflow.sandbox.docker import _replayed_run_args
    from benchflow.sandbox.protocol import SandboxRestoreHostConfigUnavailable

    container = {
        "HostConfig": {"NetworkMode": "proj_default"},
        "Mounts": [
            _mount("bind", FOLDER, "/.bfx"),
            _mount("bind", "/tmp/tool", "/.bfx/bfx"),
        ],
    }
    with pytest.raises(SandboxRestoreHostConfigUnavailable, match=r"beneath /\.bfx"):
        _replayed_run_args(container, default_network="proj_default")
    container["Mounts"] = container["Mounts"][:1]
    assert "--mount" in _replayed_run_args(container, default_network="proj_default")
