"""bfx's helper volume on a real second Docker daemon, as ``remote-docker``
reaches it (``tests/_bfx_daemons.py``: on a build VM, a
``docker:29.1.3-dind`` container over TLS). Skipped without one.

What must hold (TRUSTED-RUNTIME.md §6, v1 review's remote-Docker row):
- **Population is atomic under concurrent first uses.** Fillers that start
  together on an empty volume all finish, a reader polling meanwhile never
  sees a partial or other ``bfx``, and the volume ends holding exactly the
  pinned ``bfx``. Two processes with their own lock files (two machines, in
  effect) both get the volume.
- **The digest check refuses a volume whose content differs.** A labelled
  helper volume whose ``bfx`` is other bytes is refused and left as it is:
  never refilled, never mounted.
- **A task-image volume is never reused as the helper volume.** A volume of
  that name filled from an image's own ``/.bfx`` (Docker's copy into an
  empty volume) is refused whether or not it carries our label, and so is a
  Compose project's volume of that name. A sandbox whose task image has its
  own ``/.bfx`` gets only the pinned ``bfx`` there, and the helper volume
  holds nothing of the image's.

- **No volume step uses a task image** (M-2, M-3 of the Claude review). The
  check and the fill run in containers of a helper image made from the
  pinned bytes alone, so an image that plants ``/.bfx-fill/bfx`` cannot fill
  the volume, and a task image the daemon lacks is not pulled before ``up``.
  The helper image is removed at the end, on the classic image store and on
  Docker 29's containerd store, which lists the untagged helper image only
  with ``--all`` (R3-M1): after a fill, a check, and a step that timed out,
  on the local daemon and the remote one.

Every test uses a helper-volume name of its own (``own_volume_name``), so
the daemon's real helper volume is never touched; volumes, containers and
images are removed by name at the end.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid

import pytest

from benchflow._bfx import delivery, pins
from benchflow._bfx.locate import SANDBOX_DIR, SANDBOX_PATH, BfxUnavailable
from tests._bfx_daemons import (
    local_daemon,
    own_volume_name,
    point_remote_docker_at,
    remote_daemon,
)

BASE_IMAGE = "ubuntu:24.04"


@pytest.fixture
def remote():
    daemon = remote_daemon()
    daemon.ok(["pull", "-q", BASE_IMAGE], timeout=600)
    yield daemon
    daemon.cleanup()


@pytest.fixture
def arch(remote):
    from benchflow._bfx.locate import verified_binary

    name = remote.ok(["info", "--format", "{{.Architecture}}"]).decode().strip()
    try:
        verified_binary(name)
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx for {name}: {exc}")
    return delivery.normalize_arch(name)


def _ensure(remote, arch, *, daemon_key="test") -> str:
    return asyncio.run(
        delivery.ensure_volume(["docker"], remote.env, arch=arch, daemon_key=daemon_key)
    )


def _make_volume(remote, name: str, *labels: str) -> None:
    args = ["volume", "create"]
    for label in labels:
        args += ["--label", label]
    remote.ok([*args, name])


def _write_into(remote, volume: str, files: dict[str, bytes]) -> None:
    """Put *files* into *volume* through a throwaway container (tar in)."""
    import io
    import tarfile

    name = f"bfx-test-put-{uuid.uuid4().hex[:10]}"
    remote.containers.append(name)
    remote.ok(
        [
            "create", "--name", name, "--network", "none",
            "--mount", f"type=volume,src={volume},dst=/v,volume-nocopy",
            BASE_IMAGE, "true",
        ]
    )  # fmt: skip
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(data), 0o555
            tar.addfile(info, io.BytesIO(data))
    remote.ok(["cp", "-", f"{name}:/v"], input=buf.getvalue())
    remote.run(["rm", "-f", name])


def _image_with_own_bfx(remote, files: dict[str, bytes]) -> str:
    """A task image of the base whose own ``/.bfx`` holds *files*."""
    tag = f"bfx-test-task:{uuid.uuid4().hex[:10]}"
    remote.images.append(tag)
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        dockerfile = f"FROM {BASE_IMAGE}\nCOPY bfxdir/ {SANDBOX_DIR}/\n".encode()
        for path, data in {
            "Dockerfile": dockerfile,
            **{f"bfxdir/{k}": v for k, v in files.items()},
        }.items():
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    remote.ok(["build", "-q", "-t", tag, "-"], input=buf.getvalue(), timeout=600)
    return tag


def _copy_up(remote, volume: str, image: str) -> None:
    """Mount *volume* (empty) at the image's ``/.bfx`` the way a task's own
    volume would be mounted: without nocopy, so Docker copies the image's
    files into it."""
    name = f"bfx-test-copyup-{uuid.uuid4().hex[:10]}"
    remote.containers.append(name)
    remote.ok(
        [
            "run",
            "--name",
            name,
            "--network",
            "none",
            "-v",
            f"{volume}:{SANDBOX_DIR}",
            image,
            "true",
        ]
    )
    remote.run(["rm", "-f", name])


def _pinned_bytes(arch: str) -> bytes:
    from benchflow._bfx.locate import verified_binary

    return verified_binary(arch).read_bytes()


# ------------------------------------------------------ atomic population


def test_concurrent_fillers_on_an_empty_volume_leave_exactly_the_pin(
    remote, arch, monkeypatch
):
    volume = own_volume_name(monkeypatch, remote)
    want = pins.SHA256[arch]
    _make_volume(remote, volume, f"{delivery.VOLUME_LABEL}={want}")
    folder = delivery.host_dir(arch)
    seen: list[tuple[list[str], dict]] = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            seen.append(remote.volume_view(volume, BASE_IMAGE))

    watcher = threading.Thread(target=watch)
    watcher.start()

    async def fill_all(n: int):
        t = delivery.DockerTransport(["docker"], remote.env)
        nonce = uuid.uuid4().hex
        helper = await delivery._helper_image(t, folder, want, nonce)
        try:
            return await asyncio.gather(
                *(
                    delivery._populate(t, helper=helper, volume=volume, want=want)
                    for _ in range(n)
                ),
                return_exceptions=True,
            )
        finally:
            await delivery._remove_helper_image(t, nonce)

    try:
        results = asyncio.run(fill_all(4))
    finally:
        stop.set()
        watcher.join()
    assert results == [None] * 4, results
    names, sums = remote.volume_view(volume, BASE_IMAGE)
    assert names == ["bfx"] and sums == {"bfx": want}
    assert seen, "the watcher read the volume at least once"
    for names_then, sums_then in seen:
        # Only bfx (complete) and fillers' private temporary files, ever.
        assert sums_then.get("bfx") in (None, want), sums_then
        for n in names_then:
            assert n == "bfx" or delivery._FILL_TEMP.fullmatch(n), names_then


def test_two_processes_with_their_own_locks_both_get_the_volume(
    remote, arch, monkeypatch, tmp_path
):
    """Two first uses at once from processes that share no lock file (two
    controllers on one remote daemon): both succeed on the same volume, and
    it holds exactly the pin."""
    volume = own_volume_name(monkeypatch, remote)
    script = (
        "import asyncio, sys\n"
        "from benchflow._bfx import delivery\n"
        "delivery.volume_name = lambda arch: sys.argv[1]\n"
        "print(asyncio.run(delivery.ensure_volume(['docker'], None, "
        "arch=sys.argv[2], daemon_key='k')))\n"
    )
    procs = []
    for i in range(2):
        env = dict(remote.env or os.environ)
        env["XDG_CACHE_HOME"] = str(tmp_path / f"cache-{i}")
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", script, volume, arch],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    outs = [p.communicate(timeout=600) for p in procs]
    for p, (out, err) in zip(procs, outs, strict=True):
        assert p.returncode == 0, err.decode(errors="replace")[-2000:]
        assert out.decode().strip().splitlines()[-1] == volume
    names, sums = remote.volume_view(volume, BASE_IMAGE)
    assert names == ["bfx"] and sums == {"bfx": pins.SHA256[arch]}


def test_a_first_use_fills_and_a_second_reuses(remote, arch, monkeypatch):
    volume = own_volume_name(monkeypatch, remote)
    assert _ensure(remote, arch) == volume
    labels = remote.inspect("volume", volume)["Labels"]
    assert labels.get(delivery.VOLUME_LABEL) == pins.SHA256[arch]
    monkeypatch.setattr(delivery, "_verified_volumes", set())
    assert _ensure(remote, arch, daemon_key="again") == volume
    names, sums = remote.volume_view(volume, BASE_IMAGE)
    assert names == ["bfx"] and sums == {"bfx": pins.SHA256[arch]}


# ------------------------------------------------------- the digest check


def test_a_labelled_volume_whose_bfx_differs_is_refused_and_left_alone(
    remote, arch, monkeypatch
):
    volume = own_volume_name(monkeypatch, remote)
    _make_volume(remote, volume, f"{delivery.VOLUME_LABEL}={pins.SHA256[arch]}")
    pinned = _pinned_bytes(arch)
    # One byte changed: same size, same name, not the pin.
    other = pinned[:-1] + bytes([pinned[-1] ^ 1])
    _write_into(remote, volume, {"bfx": other})
    before = remote.volume_view(volume, BASE_IMAGE)
    with pytest.raises(BfxUnavailable, match="not the pinned"):
        _ensure(remote, arch)
    assert remote.volume_view(volume, BASE_IMAGE) == before


# ------------------------------------- a task image's volume is never reused


@pytest.mark.parametrize("labelled", [False, True], ids=["unlabelled", "labelled"])
def test_a_volume_filled_from_an_images_own_bfx_is_refused(
    remote, arch, monkeypatch, labelled
):
    """The image's ``/.bfx`` holds the pinned bytes (so a digest of ``bfx``
    alone would pass) and one file of its own."""
    volume = own_volume_name(monkeypatch, remote)
    image = _image_with_own_bfx(
        remote, {"bfx": _pinned_bytes(arch), "task-file": b"from the task image\n"}
    )
    if labelled:
        _make_volume(remote, volume, f"{delivery.VOLUME_LABEL}={pins.SHA256[arch]}")
    _copy_up(remote, volume, image)
    before = remote.volume_view(volume, BASE_IMAGE)
    assert before[0] == ["bfx", "task-file"], before
    with pytest.raises(BfxUnavailable, match=r"not BenchFlow's|not bfx's"):
        _ensure(remote, arch)
    assert remote.volume_view(volume, BASE_IMAGE) == before


def test_a_compose_projects_volume_of_that_name_is_refused(remote, arch, monkeypatch):
    volume = own_volume_name(monkeypatch, remote)
    _make_volume(
        remote,
        volume,
        f"{delivery.VOLUME_LABEL}={pins.SHA256[arch]}",
        "com.docker.compose.project=task",
        "com.docker.compose.volume=bfx",
    )
    _write_into(remote, volume, {"bfx": _pinned_bytes(arch)})
    with pytest.raises(BfxUnavailable, match="Compose project"):
        _ensure(remote, arch)


@pytest.mark.asyncio
async def test_a_task_image_with_its_own_bfx_gets_only_the_pinned_one(
    remote, arch, monkeypatch, tmp_path
):
    """A remote-docker sandbox whose task image has its own ``/.bfx``: main's
    ``/.bfx`` is the helper volume holding only the pinned ``bfx`` (nothing
    copied from the image), checked from the host."""
    from benchflow.sandbox._docker_sweep import release_project
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    point_remote_docker_at(monkeypatch)
    monkeypatch.delenv(delivery.DELIVERY_ENV, raising=False)
    volume = own_volume_name(monkeypatch, remote)
    image = _image_with_own_bfx(
        remote, {"bfx": b"the task's own bfx\n", "task-file": b"x\n"}
    )
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(f"FROM {image}\n")
    (environment / "docker-compose.yaml").write_text(
        json.dumps({"services": {"main": {}}})
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    name = f"bf-test-bfxtask-{uuid.uuid4().hex[:8]}"
    sandbox = RemoteDockerSandbox(
        environment_dir=environment,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image=image, cpus=1, memory_mb=256),
    )
    try:
        await sandbox.start(force_build=False)
        assert sandbox.bfx_delivery == delivery.VOLUME
        container = await sandbox._main_container_id()
        assert container
        [mount] = [
            m
            for m in remote.inspect("container", container)["Mounts"]
            if m["Destination"] == SANDBOX_DIR
        ]
        assert mount["Type"] == "volume" and mount["Name"] == volume
        assert mount["RW"] is False
        assert remote.folder_names(container, SANDBOX_DIR) == ["bfx"]
        assert remote.file_sha256(container, SANDBOX_PATH) == pins.SHA256[arch]
        names, sums = remote.volume_view(volume, BASE_IMAGE)
        assert names == ["bfx"] and sums == {"bfx": pins.SHA256[arch]}
    finally:
        try:
            await sandbox.stop(delete=True)
        finally:
            release_project(sandbox.compose_project_name)


# ---------------------------------- no volume step uses a task image (M-2/M-3)


def _remote_sandbox(tmp_path, image: str):
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(f"FROM {image}\n")
    (environment / "docker-compose.yaml").write_text(
        json.dumps({"services": {"main": {}}})
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    name = f"bf-test-bfxtask-{uuid.uuid4().hex[:8]}"
    return RemoteDockerSandbox(
        environment_dir=environment,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image=image, cpus=1, memory_mb=256),
    )


def _image_with_planted_filler(remote) -> str:
    """The review's ``fillimg``: a task image whose ``/.bfx-fill/bfx`` is a
    script that writes other bytes into ``/v/bfx``."""
    import io
    import tarfile

    tag = f"bfx-test-fillimg:{uuid.uuid4().hex[:10]}"
    remote.images.append(tag)
    script = b"#!/bin/sh\nprintf other-bytes > /v/bfx\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, data, mode in (
            (
                "Dockerfile",
                f"FROM {BASE_IMAGE}\nCOPY fill/ /.bfx-fill/\n".encode(),
                0o644,
            ),
            ("fill/bfx", script, 0o755),
        ):
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    remote.ok(["build", "-q", "-t", tag, "-"], input=buf.getvalue(), timeout=600)
    return tag


@pytest.mark.asyncio
async def test_a_task_image_with_a_planted_filler_cannot_fill_the_volume(
    remote, arch, monkeypatch, tmp_path
):
    """M-2: the first start on an empty helper volume, with a task image
    that has its own ``/.bfx-fill/bfx``, fills the volume with the pinned
    bytes only, and a second start reuses it."""
    from benchflow.sandbox._docker_sweep import release_project

    point_remote_docker_at(monkeypatch)
    monkeypatch.delenv(delivery.DELIVERY_ENV, raising=False)
    volume = own_volume_name(monkeypatch, remote)
    image = _image_with_planted_filler(remote)
    helpers_before = _helper_images(remote)
    for attempt in range(2):
        folder = tmp_path / f"start-{attempt}"
        folder.mkdir()
        sandbox = _remote_sandbox(folder, image)
        try:
            await sandbox.start(force_build=False)
            assert sandbox.bfx_delivery == delivery.VOLUME
            container = await sandbox._main_container_id()
            assert remote.file_sha256(container, SANDBOX_PATH) == pins.SHA256[arch]
        finally:
            try:
                await sandbox.stop(delete=True)
            finally:
                release_project(sandbox.compose_project_name)
        names, sums = remote.volume_view(volume, BASE_IMAGE)
        assert names == ["bfx"] and sums == {"bfx": pins.SHA256[arch]}
        monkeypatch.setattr(delivery, "_verified_volumes", set())
    # The helper images made here are gone (the daemon may hold others').
    assert _helper_images(remote) - helpers_before == set()


def _helper_images(daemon) -> set[str]:
    """Every helper image on *daemon*, listed with ``--all``: the
    containerd image store lists the untagged helper image only so."""
    return set(
        daemon.ok(
            [
                "image", "ls", "--all", "--quiet", "--no-trunc",
                "--filter", f"label={delivery.HELPER_LABEL}",
            ]
        ).split()
    )  # fmt: skip


PULLED_IMAGE = "alpine:3.20"


@pytest.mark.asyncio
async def test_the_volume_check_does_not_pull_the_task_image(
    remote, arch, monkeypatch, tmp_path
):
    """M-3: with main's prebuilt image absent from the daemon, the bfx
    volume steps finish without it (the start's own ``up`` pulls it,
    afterwards)."""
    from benchflow.sandbox._docker_sweep import release_project
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    point_remote_docker_at(monkeypatch)
    monkeypatch.delenv(delivery.DELIVERY_ENV, raising=False)
    own_volume_name(monkeypatch, remote)
    # This test's daemon is the throwaway DinD one: the image is removed
    # there by name so that the start must pull it.
    remote.run(["rmi", PULLED_IMAGE])
    assert remote.run(["image", "inspect", PULLED_IMAGE]).returncode != 0
    remote.images.append(PULLED_IMAGE)
    present_after_fill: list[bool] = []
    original = RemoteDockerSandbox._fill_bfx_volume

    async def fill_then_look(self):
        await original(self)
        present_after_fill.append(
            remote.run(["image", "inspect", PULLED_IMAGE]).returncode == 0
        )

    monkeypatch.setattr(RemoteDockerSandbox, "_fill_bfx_volume", fill_then_look)
    sandbox = _remote_sandbox(tmp_path, PULLED_IMAGE)
    try:
        await sandbox.start(force_build=False)
        assert sandbox.bfx_delivery == delivery.VOLUME
    finally:
        try:
            await sandbox.stop(delete=True)
        finally:
            release_project(sandbox.compose_project_name)
    assert present_after_fill == [False]
    left = remote.ok(["ps", "-aq", "--filter", "label=benchflow.bfx"])
    assert left.split() == []


class _RecordingTransport(delivery.DockerTransport):
    """The real ``docker`` CLI, recording each helper image's nonce and the
    ID ``docker import`` printed. With *hang_on*, a ``docker`` call whose
    first argument is that word waits past the step's bound first."""

    def __init__(self, env, *, hang_on: str | None = None) -> None:
        super().__init__(["docker"], env)
        self.hang_on = hang_on
        self.nonces: list[str] = []
        self.images: list[str] = []

    async def run(self, args, *, timeout=120.0):
        if args[:1] == [self.hang_on]:
            await asyncio.sleep(delivery.STEP_TIMEOUT + 30)
        return await super().run(args, timeout=timeout)

    async def import_image(self, archive, changes, *, timeout):
        prefix = f"LABEL {delivery.HELPER_LABEL}="
        self.nonces += [c[len(prefix) :] for c in changes if c.startswith(prefix)]
        code, out, err = await super().import_image(archive, changes, timeout=timeout)
        if code == 0:
            self.images += out.decode().split()[-1:]
        return code, out, err


@pytest.fixture(params=["local", "remote"])
def daemon(request):
    made = local_daemon() if request.param == "local" else remote_daemon()
    made.ok(["pull", "-q", BASE_IMAGE], timeout=600)
    yield made
    made.cleanup()


def test_no_helper_image_is_left_after_a_fill_a_check_or_a_timed_out_step(
    daemon, monkeypatch
):
    """R3-M1: ``docker import``'s helper image has no tag, so on Docker 29's
    containerd store only ``image ls --all`` lists it. Whatever the image
    store, each helper image made here is gone afterwards: by its ID and by
    its label, listed with ``--all``."""
    from benchflow._bfx.locate import verified_binary

    name = daemon.ok(["info", "--format", "{{.Architecture}}"]).decode().strip()
    try:
        verified_binary(name)
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx for {name}: {exc}")
    arch = delivery.normalize_arch(name)
    volume = own_volume_name(monkeypatch, daemon)
    monkeypatch.setattr(delivery, "STEP_TIMEOUT", 15.0)

    def ensure(t, key):
        return asyncio.run(
            delivery.ensure_volume(
                ["docker"], daemon.env, arch=arch, daemon_key=key, transport=t
            )
        )

    fill, check = _RecordingTransport(daemon.env), _RecordingTransport(daemon.env)
    assert ensure(fill, "fill") == volume
    assert ensure(check, "check") == volume
    # The check's ``create`` waits past the bound: the step times out.
    hung = _RecordingTransport(daemon.env, hang_on="create")
    with pytest.raises(BfxUnavailable, match="did not finish within"):
        ensure(hung, "timed-out")
    names, sums = daemon.volume_view(volume, BASE_IMAGE)
    assert names == ["bfx"] and sums == {"bfx": pins.SHA256[arch]}

    made = fill.images + check.images + hung.images
    assert len(made) == 3 and len(set(fill.nonces + check.nonces + hung.nonces)) == 3
    for image in made:
        assert daemon.run(["image", "inspect", image]).returncode != 0, image
    for nonce in fill.nonces + check.nonces + hung.nonces:
        left = daemon.ok(
            [
                "image", "ls", "--all", "--quiet", "--no-trunc",
                "--filter", f"label={delivery.HELPER_LABEL}={nonce}",
            ]
        )  # fmt: skip
        assert left.split() == [], nonce
