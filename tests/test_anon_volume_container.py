"""DOCKER-ANON-VOLUME-LEAK on a real daemon: a trial leaves no volume it made.

An image that declares a ``VOLUME``, and a Compose ``volumes: [/path]``
entry, would give each container Compose makes an anonymous volume, whose
owner no later inspect can prove (rounds 1 and 2). Now every such volume is
named and carries the trial's label from its creation
(:mod:`benchflow.sandbox._trial_volumes`), and teardown removes the
project's labelled volumes and nothing else (results/ANONVOL.md, round 3).
Each test collects the volumes the trial's containers mounted while they ran,
checks that each of the trial's own is named and labelled (none anonymous),
then checks every one is gone after the trial, whether it passed, failed,
errored at start, was cancelled, or was torn down by the force-kill fallback,
and that volumes the trial did not make (an operator's named and anonymous
ones, BenchFlow's ``benchflow-bfx-*`` helper volume) are untouched.

Skipped without a Docker daemon holding IMAGE; every image, volume and
container a test makes is named ``anonvol-*`` and removed by name.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _docker_sweep as sweep
from benchflow.sandbox._docker_sweep import release_project
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

IMAGE = "ubuntu:24.04"
_run = subprocess.run
# The label every volume a trial makes carries from its creation
# (benchflow.sandbox._trial_volumes.LABEL; written out so this file also
# runs against a tree from before it).
LABEL = "io.benchflow.trial-volume"
# The name Docker gives an anonymous volume.
_ANONYMOUS = re.compile(r"[0-9a-f]{64}")
# BenchFlow's helper volume (batch 45): shared by trials, never a trial's.
_HELPER = "benchflow-bfx-"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = _run(["docker", "image", "inspect", IMAGE], capture_output=True)
    except Exception:
        return False
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_ready(), reason=f"needs a Docker daemon holding {IMAGE}"
)


def _docker(*args: str, check: bool = True) -> str:
    done = _run(["docker", *args], capture_output=True, text=True)
    if check and done.returncode:
        raise AssertionError(f"docker {' '.join(args)}: {done.stderr}")
    return done.stdout


def _exists(volume: str) -> bool:
    return (
        _run(["docker", "volume", "inspect", volume], capture_output=True).returncode
        == 0
    )


def _mounted_volumes(label: str) -> set[str]:
    """The volumes the containers carrying *label* mount now."""
    ids = _docker("ps", "-aq", "--no-trunc", "--filter", f"label={label}").split()
    if not ids:
        return set()
    out = _docker("inspect", *ids, check=False)
    try:
        containers = json.loads(out or "[]")
    except ValueError:
        return set()
    return {
        mount["Name"]
        for container in containers
        for mount in container.get("Mounts", [])
        if mount.get("Type") == "volume"
    }


class _Watch:
    """Collects, every 0.2 s, the volumes the containers carrying *label*
    mount, for the whole trial (the names are not known in advance)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.seen: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.seen |= _mounted_volumes(self.label)
            self._stop.wait(0.2)

    def __enter__(self) -> _Watch:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


@pytest.fixture(scope="module")
def volume_image():
    """IMAGE with ``VOLUME /data`` holding a file, owned 1234:5678, mode
    0751; labelled, built once for the module."""
    tag = f"anonvol-image:{uuid.uuid4().hex[:8]}"
    _run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=(
            f"FROM {IMAGE}\n"
            "RUN mkdir -p /data && echo image-data > /data/f "
            "&& chown 1234:5678 /data && chmod 0751 /data\n"
            "VOLUME /data\nLABEL anonvol-test=1\n"
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    yield tag
    _run(["docker", "image", "rm", "--no-prune", tag], capture_output=True)


def _sandbox(
    tmp_path: Path, image: str, compose: dict | None = None, **kwargs
) -> DockerSandbox:
    name = f"anonvol-{uuid.uuid4().hex[:8]}"
    env = tmp_path / name / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text(f"FROM {image}\n")
    if compose is not None:
        (env / "docker-compose.yaml").write_text(json.dumps(compose))
    paths = RolloutPaths(rollout_dir=tmp_path / name / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=env,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512, docker_image=image),
        **kwargs,
    )


def _project_volumes(sandbox: DockerSandbox) -> set[str]:
    return _mounted_volumes(
        f"com.docker.compose.project={sandbox.compose_project_name}"
    )


def _assert_named(volumes: set[str], project: str | None = None) -> None:
    """None of the trial's volumes is anonymous; with *project*, each (still
    on the daemon) carries that project's trial label."""
    for name in volumes:
        if name.startswith(_HELPER):
            continue
        assert not _ANONYMOUS.fullmatch(name), f"Docker made {name} anonymous"
        if project is not None:
            labels = json.loads(_docker("volume", "inspect", name))[0]["Labels"]
            assert (labels or {}).get(LABEL) == project, (name, labels)


def _assert_gone(volumes: set[str]) -> None:
    left = sorted(v for v in volumes if _exists(v) and not v.startswith(_HELPER))
    assert not left, f"the trial left its volumes behind: {left}"
    gone = sorted(v for v in volumes if v.startswith(_HELPER) and not _exists(v))
    assert not gone, f"the trial removed BenchFlow's helper volume: {gone}"


def _down_fails(sandbox: DockerSandbox) -> None:
    real = sandbox._run_docker_compose_command

    async def down_fails(command, *args, **kwargs):
        if command and command[0] == "down":
            raise RuntimeError("anonvol: compose down hung")
        return await real(command, *args, **kwargs)

    sandbox._run_docker_compose_command = down_fails


@pytest.mark.asyncio
async def test_a_completed_start_leaves_no_volume(tmp_path, volume_image):
    sandbox = _sandbox(tmp_path, volume_image)
    await sandbox.start(force_build=False)
    try:
        made = _project_volumes(sandbox)
        assert made, "the image's VOLUME made no volume"
        _assert_named(made, sandbox.compose_project_name)
    finally:
        await sandbox.stop(delete=True)
    _assert_gone(made)


@pytest.mark.asyncio
async def test_a_named_volume_holds_what_the_image_put_there(tmp_path, volume_image):
    """Docker fills an empty named volume from the image path as it filled
    the anonymous one: the same file, owner and mode."""
    sandbox = _sandbox(tmp_path, volume_image)
    await sandbox.start(force_build=False)
    try:
        made = _project_volumes(sandbox)
        seen = await sandbox.exec("stat -c '%u:%g %a' /data; cat /data/f", user="root")
        assert seen.stdout.split() == ["1234:5678", "751", "image-data"], seen
    finally:
        await sandbox.stop(delete=True)
    _assert_gone(made)


@pytest.mark.asyncio
async def test_a_compose_anonymous_volume_is_named_too(tmp_path, volume_image):
    """``volumes: [/cache]`` (short form) and the long form without a
    source, on main and a sidecar, each get a labelled named volume."""
    compose = {
        "services": {
            "main": {"volumes": ["/cache"]},
            "side": {
                "image": IMAGE,
                "command": ["sleep", "infinity"],
                "volumes": [{"type": "volume", "target": "/sidecache"}],
            },
        }
    }
    sandbox = _sandbox(tmp_path, volume_image, compose)
    await sandbox.start(force_build=False)
    try:
        made = _project_volumes(sandbox)
        assert len(made) >= 3, made  # /data, /cache, /sidecache
        _assert_named(made, sandbox.compose_project_name)
    finally:
        await sandbox.stop(delete=True)
    _assert_gone(made)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step", ["_release_inert_bootstrap", "_prepare_log_dirs_after_up"]
)
@pytest.mark.parametrize("how", ["fails", "is_cancelled"])
async def test_a_start_that_ends_after_up_leaves_no_volume(
    tmp_path, volume_image, monkeypatch, step, how
):
    """The start ends after Compose made the containers: inside the bind
    proof and release (start's own ``down`` follows) or after it (teardown
    finds the containers)."""
    sandbox = _sandbox(tmp_path, volume_image)
    made: set[str] = set()
    reached = asyncio.Event()

    async def ends(self):
        made.update(_project_volumes(self))
        _assert_named(made, self.compose_project_name)
        reached.set()
        if how == "fails":
            raise RuntimeError("anonvol: the start fails here")
        await asyncio.Event().wait()

    monkeypatch.setattr(DockerSandbox, step, ends)
    start = asyncio.ensure_future(sandbox.start(force_build=False))
    try:
        if how == "fails":
            with pytest.raises(RuntimeError, match="anonvol"):
                await start
        else:
            # No short cap: on a loaded daemon the start can take minutes
            # to reach the step; a start that errors first ends the wait.
            waiter = asyncio.ensure_future(reached.wait())
            await asyncio.wait(
                {waiter, start}, timeout=900, return_when=asyncio.FIRST_COMPLETED
            )
            waiter.cancel()
            assert reached.is_set(), "the start never reached the step"
            start.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start
    finally:
        # Teardown only after the start has ended: a start still running
        # would make its containers after teardown looked for them.
        if not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)
        await sandbox.stop(delete=True)
    assert made, "the image's VOLUME made no volume"
    _assert_gone(made)


@pytest.mark.asyncio
async def test_the_force_kill_fallback_leaves_no_volume(tmp_path, volume_image):
    """``compose down`` fails: the force-kill removes the containers, and
    teardown then removes the labelled volumes."""
    sandbox = _sandbox(tmp_path, volume_image)
    await sandbox.start(force_build=False)
    made = _project_volumes(sandbox)
    _down_fails(sandbox)
    try:
        assert made
    finally:
        await sandbox.stop(delete=True)
    assert not _docker(
        "ps",
        "-aq",
        "--filter",
        f"label=com.docker.compose.project={sandbox.compose_project_name}",
    ).split()
    _assert_gone(made)


@pytest.mark.asyncio
async def test_a_recovery_sandbox_removes_its_own_volumes(tmp_path, volume_image):
    """A recovery start keeps its parent's leased image, never the volumes
    its own containers got."""
    sandbox = _sandbox(tmp_path, volume_image)
    await sandbox.start(force_build=False)
    made = _project_volumes(sandbox)
    # Teardown's view of a recovery sandbox: the lease is the image's.
    sandbox._recovery_baseline = SimpleNamespace(image_id="sha256:lease")
    try:
        assert made
    finally:
        await sandbox.stop(delete=True)
    _assert_gone(made)
    assert _docker("image", "inspect", volume_image, check=False), "the image went"


@pytest.mark.asyncio
async def test_a_second_start_gets_fresh_volumes_and_leaves_none(
    tmp_path, volume_image
):
    """A second start of the same sandbox recreates its containers on new
    volumes, filled from the image again, as anonymous ones were; teardown
    removes both starts' volumes."""
    sandbox = _sandbox(tmp_path, volume_image)
    await sandbox.start(force_build=False)
    try:
        first = _project_volumes(sandbox)
        await sandbox.exec("echo agent > /data/f", user="root")
        await sandbox.start(force_build=False)
        second = _project_volumes(sandbox)
        seen = await sandbox.exec("cat /data/f", user="root")
        assert seen.stdout.strip() == "image-data", seen
        assert first.isdisjoint(second - {v for v in first if v.startswith(_HELPER)})
        _assert_named(first | second, sandbox.compose_project_name)
    finally:
        await sandbox.stop(delete=True)
    _assert_gone(first | second)


@pytest.mark.asyncio
async def test_a_restore_makes_no_anonymous_volume(tmp_path, volume_image):
    """Snapshot restore replays main's mounts by name; main mounts every
    VOLUME of its image, so the replacement gets no new volume."""
    sandbox = _sandbox(tmp_path, volume_image)
    await sandbox.start(force_build=False)
    snapshot = None
    try:
        before = _project_volumes(sandbox)
        await sandbox.exec("echo kept > /data/f", user="root")
        snapshot = await sandbox.snapshot()
        await sandbox.restore(snapshot)
        after = _project_volumes(sandbox)
        assert after == before, (before, after)
        seen = await sandbox.exec("cat /data/f", user="root")
        assert seen.stdout.strip() == "kept", seen
    finally:
        await sandbox.stop(delete=True)
        if snapshot is not None:
            _run(["docker", "image", "rm", snapshot.ref], capture_output=True)
    _assert_gone(before)


@pytest.mark.asyncio
async def test_the_layer_builds_libc_probe_makes_no_anonymous_volume(volume_image):
    """The libc probe runs the task image once (``docker run --rm``); each
    VOLUME gets a labelled named volume, removed after the probe."""
    from benchflow.sandbox._docker_subnets import daemon_key
    from benchflow.sandbox.env_layers import DockerLayers

    layers = DockerLayers(daemon=daemon_key(None))
    probes: list[list[str]] = []
    real = layers._docker

    async def docker(*args, **kwargs):
        if args[:1] == ("run",):
            probes.append(list(args))
        return await real(*args, **kwargs)

    layers._docker = docker
    base = await layers.inspect_base(volume_image)
    assert base.libc == "glibc"
    (probe,) = probes
    mounts = [probe[i + 1] for i, arg in enumerate(probe) if arg == "--mount"]
    assert len(mounts) == 1 and ",dst=/data," in mounts[0], mounts
    assert f"volume-label={LABEL}=" in mounts[0]
    name = mounts[0].split("src=")[1].split(",")[0]
    assert not _exists(name), f"the probe left {name}"


@pytest.fixture
def operator_volumes():
    """Volumes the trial did not make: an operator's named one, BenchFlow's
    helper volume by its name, and an operator's anonymous one (Docker's
    anonymous label, left by a container removed without ``-v``)."""
    tag = uuid.uuid4().hex[:8]
    named = f"anonvol-operator-{tag}"
    helper = f"benchflow-bfx-anonvol-{tag}"
    _docker("volume", "create", named)
    _docker("volume", "create", helper)
    cid = _docker(
        "create", "-v", "/x", "--label", "anonvol-test=1", IMAGE, "true"
    ).strip()
    anonymous = next(
        m["Name"] for m in json.loads(_docker("inspect", cid))[0]["Mounts"]
    )
    _docker("rm", cid)  # without -v: the anonymous volume stays
    assert "com.docker.volume.anonymous" in (
        json.loads(_docker("volume", "inspect", anonymous))[0]["Labels"] or {}
    )
    volumes = {"named": named, "helper": helper, "anonymous": anonymous}
    yield volumes
    for name in volumes.values():
        _run(["docker", "volume", "rm", name], capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("use", ["unmounted", "external", "adopted"])
@pytest.mark.parametrize("teardown", ["normal", "force_kill"])
async def test_volumes_the_trial_did_not_make_are_untouched(
    tmp_path, volume_image, operator_volumes, monkeypatch, use, teardown
):
    """On the daemon, or mounted into main by the task's compose file,
    ``external`` or adopted (Compose uses an existing volume of that name,
    with a warning; either needs the operator's
    BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1), they stay after either teardown;
    the image's own volume goes."""
    compose = None
    if use != "unmounted":
        monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
        external = {"external": True} if use == "external" else {}
        compose = {
            "services": {
                "main": {"volumes": [f"{key}:/mnt/{key}" for key in operator_volumes]}
            },
            "volumes": {
                key: {"name": name, **external}
                for key, name in operator_volumes.items()
            },
        }
    sandbox = _sandbox(tmp_path, volume_image, compose)
    await sandbox.start(force_build=False)
    mounted = _project_volumes(sandbox)
    if teardown == "force_kill":
        _down_fails(sandbox)
    own = mounted - set(operator_volumes.values())
    try:
        assert own, mounted
        _assert_named(own, sandbox.compose_project_name)
        if use != "unmounted":
            assert set(operator_volumes.values()) <= mounted, mounted
    finally:
        await sandbox.stop(delete=True)
    for key, name in operator_volumes.items():
        assert _exists(name), f"the operator's {key} volume {name} was removed"
    _assert_gone(own)


@pytest.fixture
def operator_source():
    """An operator's container (``anonvol-src-*``) with an anonymous volume
    at /opdata, and that volume's name; both removed by name after."""
    name = f"anonvol-src-{uuid.uuid4().hex[:8]}"
    _docker(
        "create",
        "--name",
        name,
        "-v",
        "/opdata",
        "--label",
        "anonvol-test=1",
        IMAGE,
        "true",
    )
    volume = json.loads(_docker("inspect", name))[0]["Mounts"][0]["Name"]
    yield name, volume
    _run(["docker", "rm", "-f", name], capture_output=True)
    _run(["docker", "volume", "rm", volume], capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["_release_inert_bootstrap", "completed"])
async def test_a_volume_inherited_through_external_volumes_from_stays(
    tmp_path, volume_image, operator_source, monkeypatch, step
):
    """Codex review r1 M1: main inherits the operator's anonymous volume
    through ``volumes_from: [container:...]`` (operator opt-in
    BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1). The operator removes the source
    container without -v; the start fails, its own ``down`` removes main,
    then stop. The operator's volume stays, the image's own volume goes."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    source, theirs = operator_source
    compose = {"services": {"main": {"volumes_from": [f"container:{source}:ro"]}}}
    sandbox = _sandbox(tmp_path, volume_image, compose)
    made: set[str] = set()
    real = DockerSandbox._release_inert_bootstrap

    async def source_goes(self):
        made.update(_project_volumes(self))
        _docker("rm", source)  # without -v: the operator keeps the data
        if step == "completed":
            return await real(self)
        raise RuntimeError("anonvol: the start fails here")

    monkeypatch.setattr(DockerSandbox, "_release_inert_bootstrap", source_goes)
    try:
        if step == "completed":
            await sandbox.start(force_build=False)
        else:
            with pytest.raises(RuntimeError, match="anonvol"):
                await sandbox.start(force_build=False)
    finally:
        await sandbox.stop(delete=True)
    assert theirs in made, made
    assert _exists(theirs), f"the operator's volume {theirs} was removed"
    own = made - {theirs}
    assert own, "the image's VOLUME made no volume"
    _assert_gone(own)


@pytest.mark.asyncio
async def test_a_source_gone_before_up_leaves_no_trial_volume(
    tmp_path, volume_image, operator_source, monkeypatch
):
    """Codex review r2 R2-M1 (second reproducer): the operator removes the
    ``volumes_from`` source right after Compose made the stopped prototypes;
    Compose's recreate of main then fails ("No such container"). The
    trial's own volume was kept at db6096398 (its owner could no longer be
    proved); it is labelled from creation, so it goes. The operator's
    stays."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    source, theirs = operator_source
    compose = {"services": {"main": {"volumes_from": [f"container:{source}:ro"]}}}
    sandbox = _sandbox(tmp_path, volume_image, compose)
    made: set[str] = set()
    real = DockerSandbox._inspect_container

    async def inspected(self, cid):
        row = await real(self, cid)
        if not made and source in " ".join(
            (row.get("HostConfig") or {}).get("VolumesFrom") or []
        ):
            made.update(m["Name"] for m in row["Mounts"] if m["Type"] == "volume")
            _docker("rm", source)  # after Docker made main, before up
        return row

    monkeypatch.setattr(DockerSandbox, "_inspect_container", inspected)
    try:
        with pytest.raises(Exception):  # noqa: B017 - Docker's own error
            await sandbox.start(force_build=False)
    finally:
        made |= _project_volumes(sandbox)
        await sandbox.stop(delete=True)
    assert theirs in made, made
    assert _exists(theirs), f"the operator's volume {theirs} was removed"
    own = made - {theirs}
    assert own, "the image's VOLUME made no volume"
    _assert_gone(own)


@pytest.mark.asyncio
async def test_the_sweep_removes_a_dead_trials_volumes_only(
    tmp_path, volume_image, operator_source, monkeypatch
):
    """Codex review r2 R2-M1 (first reproducer): a trial whose teardown never
    ran (its project released, its main stopped) has its own volume at
    /data and the operator's at /opdata through an external
    ``volumes_from`` whose source still exists. The sweep removes the
    trial's labelled volumes, never by an inspect of the container, and the
    operator's stays. At db6096398 the trial's own volume was left."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    source, theirs = operator_source
    compose = {"services": {"main": {"volumes_from": [f"container:{source}:ro"]}}}
    sandbox = _sandbox(tmp_path, volume_image, compose)
    project = sandbox.compose_project_name
    await sandbox.start(force_build=False)
    try:
        made = _project_volumes(sandbox)
        assert theirs in made
        # The trial's process went before its teardown: nothing stops it.
        _docker("stop", "-t", "0", *_docker(
            "ps", "-q", "--filter", f"label=com.docker.compose.project={project}"
        ).split())  # fmt: skip
        release_project(project)
        real = sweep._docker

        def scoped(args):
            # This test's project only, leaving any other run's alone.
            if args[0] == "ps" or args[:2] == ["network", "ls"]:
                args = [
                    *args,
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                ]
            return real(args)

        monkeypatch.setattr(sweep, "_docker", scoped)
        result = sweep.sweep_leftovers()
        monkeypatch.setattr(sweep, "_docker", real)
        assert result.removed_containers
        assert _exists(theirs), f"the operator's volume {theirs} was removed"
        _assert_gone(made - {theirs})
    finally:
        await sandbox.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["fails", "completed"])
async def test_a_source_named_like_an_image_changes_nothing(
    tmp_path, volume_image, operator_source, monkeypatch, step
):
    """Claude review r2 M1: a sidecar (never recreated) inherits the
    operator's volume at its image's VOLUME path, and an image carries the
    source container's name. The source goes during release. Round 2's
    untyped inspect found the image and removed the operator's volume; no
    inspect decides ownership now."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    name = f"anonvol-srcimg-{uuid.uuid4().hex[:8]}"
    _docker("tag", volume_image, f"{name}:latest")
    _docker(
        "create",
        "--name",
        name,
        "-v",
        "/data",
        "--label",
        "anonvol-test=1",
        IMAGE,
        "true",
    )
    theirs = json.loads(_docker("inspect", "--type", "container", name))[0]["Mounts"][
        0
    ]["Name"]
    compose = {
        "services": {
            "side": {
                "image": volume_image,
                "command": ["sleep", "infinity"],
                "volumes_from": [f"container:{name}"],
            }
        }
    }
    sandbox = _sandbox(tmp_path, volume_image, compose)
    made: set[str] = set()
    real = DockerSandbox._release_inert_bootstrap

    async def source_goes(self):
        made.update(_project_volumes(self))
        _docker("rm", name)
        if step == "completed":
            return await real(self)
        raise RuntimeError("anonvol: the start fails here")

    monkeypatch.setattr(DockerSandbox, "_release_inert_bootstrap", source_goes)
    try:
        if step == "completed":
            await sandbox.start(force_build=False)
            made |= _project_volumes(sandbox)
        else:
            with pytest.raises(RuntimeError, match="anonvol"):
                await sandbox.start(force_build=False)
    finally:
        await sandbox.stop(delete=True)
        _run(
            ["docker", "image", "rm", "--no-prune", f"{name}:latest"],
            capture_output=True,
        )
        _run(["docker", "rm", "-f", name], capture_output=True)
    try:
        assert theirs in made, made
        assert _exists(theirs), f"the operator's volume {theirs} was removed"
        _assert_gone(made - {theirs})
    finally:
        _run(["docker", "volume", "rm", theirs], capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["_release_inert_bootstrap", "completed"])
async def test_a_project_volumes_from_leaves_no_volume(
    tmp_path, volume_image, monkeypatch, step
):
    """main takes a sidecar's volumes through ``volumes_from: [side]``: the
    sidecar's Compose anonymous volume and its image's VOLUME, both named
    for the trial; main sees the same volumes (the sharing is unchanged).
    They go after a completed start and after one that fails in release."""
    compose = {
        "services": {
            "main": {"volumes_from": ["side"]},
            "side": {
                "image": volume_image,
                "command": ["sleep", "infinity"],
                "volumes": ["/shared"],
            },
        }
    }
    sandbox = _sandbox(tmp_path, volume_image, compose)
    made: set[str] = set()
    real = DockerSandbox._release_inert_bootstrap

    async def release(self):
        made.update(_project_volumes(self))
        _assert_named(made, self.compose_project_name)
        if step == "completed":
            return await real(self)
        raise RuntimeError("anonvol: the start fails here")

    monkeypatch.setattr(DockerSandbox, "_release_inert_bootstrap", release)
    try:
        if step == "completed":
            await sandbox.start(force_build=False)
            made |= _project_volumes(sandbox)
            await sandbox.exec("echo main > /shared/m", user="root")
            seen = await sandbox.exec(
                "cat /shared/m /data/f", service="side", user="root"
            )
            assert seen.stdout.split() == ["main", "image-data"], seen
        else:
            with pytest.raises(RuntimeError, match="anonvol"):
                await sandbox.start(force_build=False)
    finally:
        await sandbox.stop(delete=True)
    assert len(made) >= 2, made
    _assert_gone(made)


@pytest.mark.asyncio
@pytest.mark.parametrize("side_image", ["plain", "volume"])
async def test_a_peers_tmpfs_is_not_passed_through_volumes_from(
    tmp_path, volume_image, side_image
):
    """Round 3 review A, M1: ``side`` mounts a tmpfs at /data (over its own
    image's VOLUME there, or not). Docker copies only mount points through
    ``volumes_from``, and a tmpfs is none, so main's image VOLUME /data gets
    a volume of its own. It is named and labelled, holds the image's file,
    and goes with the trial."""
    compose = {
        "services": {
            "main": {"volumes_from": ["side"]},
            "side": {
                "image": IMAGE if side_image == "plain" else volume_image,
                "command": ["sleep", "infinity"],
                "tmpfs": ["/data"],
            },
        }
    }
    sandbox = _sandbox(tmp_path, volume_image, compose)
    await sandbox.start(force_build=False)
    try:
        made = _project_volumes(sandbox)
        assert made, "main's VOLUME /data made no volume"
        _assert_named(made, sandbox.compose_project_name)
        seen = await sandbox.exec("cat /data/f", user="root")
        assert seen.stdout.split() == ["image-data"], seen
    finally:
        await sandbox.stop(delete=True)
    _assert_gone(made)


@pytest.mark.asyncio
async def test_three_trials_at_once_leave_only_what_they_did_not_make(
    tmp_path, volume_image, operator_volumes, operator_source, monkeypatch
):
    """Three sandboxes on one daemon at once: one completes, one fails in
    release, one inherits the operator's volume through an external
    ``volumes_from`` whose source goes during the start. Beside them: an
    operator named volume, a ``benchflow-bfx-*`` volume in use by a running
    container, and an operator's dangling anonymous volume. Each trial's
    volumes carry its own project; all three trials' go, nothing else."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    source, theirs = operator_source
    holder = f"anonvol-holder-{uuid.uuid4().hex[:8]}"
    _docker(
        "run", "-d", "--name", holder, "--label", "anonvol-test=1",
        "-v", f"{operator_volumes['helper']}:/h", IMAGE, "sleep", "infinity",
    )  # fmt: skip
    sandboxes = {
        "completes": _sandbox(tmp_path, volume_image),
        "fails": _sandbox(tmp_path, volume_image),
        "inherits": _sandbox(
            tmp_path,
            volume_image,
            {"services": {"main": {"volumes_from": [f"container:{source}:ro"]}}},
        ),
    }
    made: dict[str, set[str]] = {key: set() for key in sandboxes}
    real = DockerSandbox._release_inert_bootstrap

    async def release(self):
        key = next(k for k, s in sandboxes.items() if s is self)
        made[key] |= _project_volumes(self)
        _assert_named(made[key] - {theirs}, self.compose_project_name)
        if key == "fails":
            raise RuntimeError("anonvol: the start fails here")
        if key == "inherits":
            _docker("rm", source)
        return await real(self)

    monkeypatch.setattr(DockerSandbox, "_release_inert_bootstrap", release)

    async def trial(key: str) -> None:
        sandbox = sandboxes[key]
        try:
            await sandbox.start(force_build=False)
            made[key] |= _project_volumes(sandbox)
        except RuntimeError as exc:
            assert key == "fails" and "anonvol" in str(exc), exc
        finally:
            await sandbox.stop(delete=True)

    try:
        await asyncio.gather(*(trial(key) for key in sandboxes))
    finally:
        _run(["docker", "rm", "-f", holder], capture_output=True)
    for key in sandboxes:
        own = made[key] - {theirs}
        assert own, (key, made)
        _assert_gone(own)
    assert _exists(theirs)
    for key, name in operator_volumes.items():
        assert _exists(name), f"the operator's {key} volume {name} was removed"


# --- whole trials through `bench run` -----------------------------------------

_DOCKERFILE = (
    f"FROM {IMAGE}\nVOLUME /data\nWORKDIR /app\n"
    "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
)
_TEST = (
    "#!/bin/bash\n"
    "if grep -q hello /app/out.txt 2>/dev/null; then echo 1.0 > "
    "/logs/verifier/reward.txt; else echo 0.0 > /logs/verifier/reward.txt; fi\n"
)


def _task(
    root: Path, name: str, label: str, *, solve: str, compose: dict | None = None
) -> Path:
    task = root / "tasks" / name
    for folder in ("environment", "oracle", "verifier"):
        (task / folder).mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        _DOCKERFILE + f'LABEL anonvol-trial="{label}"\n'
    )
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(json.dumps(compose))
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\n" + solve)
    (task / "verifier" / "test.sh").write_text(_TEST)
    (task / "task.md").write_text(
        '---\nschema_version: "1.3"\nagent:\n  timeout_sec: 300\n'
        f"verifier:\n  timeout_sec: 120\n---\n# {name}\n\n"
        "## prompt\n\nWrite the word hello into the file /app/out.txt.\n"
    )
    return task


def _env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        NO_COLOR="1",
        COLUMNS="160",
        BENCHFLOW_NO_PROGRESS="1",
        BENCHFLOW_LOG_DIR=str(root / "logs"),
    )
    env.pop("BENCHFLOW_LOG_LEVEL", None)
    return env


def _argv(name: str) -> list[str]:
    return [
        sys.executable, "-m", "benchflow.cli.main", "run", f"tasks/{name}",
        "--jobs-dir", "jobs", "--job", name, "--retry-attempts", "0",
    ]  # fmt: skip


def _result(root: Path, name: str) -> dict:
    found = sorted((root / "jobs" / name).rglob("result.json"))
    assert found, f"no result.json for {name}"
    return json.loads(found[-1].read_text())


def _remove_built(label: str) -> None:
    for image in _docker(
        "images", "-q", "--filter", f"label=anonvol-trial={label}"
    ).split():
        _run(["docker", "image", "rm", "--no-prune", image], capture_output=True)


@pytest.mark.parametrize(
    "case,solve,compose,reward",
    [
        ("passes", "echo hello > /app/out.txt\n", None, 1.0),
        ("fails", "echo nothing\n", None, 0.0),
        # main never turns healthy: the start fails after Compose made it.
        (
            "errors-at-start",
            "echo hello > /app/out.txt\n",
            {
                "services": {
                    "main": {
                        "healthcheck": {
                            "test": ["CMD", "false"],
                            "interval": "1s",
                            "timeout": "5s",
                            "retries": 1,
                        }
                    }
                }
            },
            None,
        ),
    ],
)
def test_a_trial_leaves_no_volume(tmp_path, case, solve, compose, reward):
    label = f"anonvol-{uuid.uuid4().hex[:8]}"
    _task(tmp_path, case, label, solve=solve, compose=compose)
    try:
        with _Watch(f"anonvol-trial={label}") as watch:
            done = _run(
                _argv(case), cwd=tmp_path, env=_env(tmp_path),
                capture_output=True, text=True, timeout=600,
            )  # fmt: skip
        out = done.stdout + done.stderr
        result = _result(tmp_path, case)
        if reward is None:
            assert result.get("error"), out
        else:
            assert result.get("rewards", {}).get("reward") == reward, out
        assert watch.seen, f"the image's VOLUME made no volume\n{out}"
        _assert_named(watch.seen)
        _assert_gone(watch.seen)
    finally:
        _remove_built(label)


def _solving(label: str) -> bool:
    """Whether the oracle's ``sleep 120`` runs in the trial's container."""
    for cid in _docker("ps", "-q", "--filter", f"label=anonvol-trial={label}").split():
        if "sleep 120" in _docker("top", cid, check=False):
            return True
    return False


# Ctrl-C's signal with its default action, whatever the suite was started
# with (an ``&`` job starts with SIGINT ignored, CTRLC-1).
_INTERRUPTIBLE = (
    "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); "
    "os.execv(sys.argv[1], sys.argv[1:])"
)


def test_a_cancelled_trial_leaves_no_volume(tmp_path):
    """Ctrl-C (SIGINT) while the agent runs: the run stops the trial and
    removes its sandbox, its volumes included."""
    label = f"anonvol-{uuid.uuid4().hex[:8]}"
    _task(tmp_path, "cancelled", label, solve="sleep 120\necho hello > /app/out.txt\n")
    try:
        with _Watch(f"anonvol-trial={label}") as watch:
            run = subprocess.Popen(
                [sys.executable, "-c", _INTERRUPTIBLE, *_argv("cancelled")],
                cwd=tmp_path,
                env=_env(tmp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + 300
            while not _solving(label) and run.poll() is None:
                assert time.monotonic() < deadline, "the agent never started"
                time.sleep(0.5)
            assert run.poll() is None, run.communicate()[0]
            run.send_signal(signal.SIGINT)
            out, _ = run.communicate(timeout=300)
        assert run.returncode == 130, out
        assert watch.seen, out
        _assert_gone(watch.seen)
    finally:
        _remove_built(label)
