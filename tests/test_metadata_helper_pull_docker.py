"""The prebuilt metadata helper image pulled by its digest, on a real Docker daemon.

A local registry (``registry:2`` on 127.0.0.1) stands in for the release's. An image is built, pushed there and its
local copy removed; ``ensure_image`` then pulls it by digest, checks it (the digest among its RepoDigests, this recipe's
labels) and returns the ID it runs by. Every docker call goes through the sandbox's own helper runner
(``DockerSandbox._helper_docker``: its timeout, and the kill of the docker CLI's process group). The pull rows push a
stand-in (the pinned base, a marker file and the two labels, no package install), so a moved Alpine pin can't fail
them; the refusal row's build of the recipe is what serves there. The rows skip where no Docker daemon answers.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import uuid

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox.docker import DockerSandbox


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=30
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]


def _run(
    *args: str, stdin: bytes | None = None, timeout: float = 300
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], input=stdin, capture_output=True, timeout=timeout
    )


def _containerd_store() -> bool:
    info = _run("info", "--format", "{{json .DriverStatus}}")
    return "io.containerd.snapshotter" in info.stdout.decode()


@pytest.fixture()
def sandbox(monkeypatch):
    """A DockerSandbox with no task, for its helper runner and daemon key; the
    docker client's environment is this process's."""
    monkeypatch.setattr(
        DockerSandbox,
        "_docker_process_env",
        lambda self, environment=None: dict(os.environ),
    )
    return DockerSandbox.__new__(DockerSandbox)


@pytest.fixture()
def registry():
    """A registry on a free 127.0.0.1 port, named ``localhost:<port>`` (Docker
    trusts loopback registries without TLS; the name sorts after
    ``docker.io/library/benchflow-helper``, as ``ghcr.io`` does); removed after."""
    owner = uuid.uuid4().hex[:12]
    started = _run(
        "run", "-d", "--rm", "-p", "127.0.0.1::5000",
        "--label", f"bf.test.helper-pull={owner}", "registry:2",
    )  # fmt: skip
    assert started.returncode == 0, started.stderr.decode()[-400:]
    container = started.stdout.decode().strip()
    port = _run("port", container, "5000/tcp").stdout.decode().strip()
    try:
        yield f"localhost:{port.rsplit(':', 1)[-1]}"
    finally:
        _run("rm", "-f", container)


def _build(tag: str, recipe: str, dockerfile: str, *build_args: str) -> str:
    built = _run(
        "build", "--no-cache", "--quiet", *build_args, "--tag", tag,
        "--label", f"{helper_mod.HELPER_LABEL}=metadata",
        "--label", f"{helper_mod.HELPER_RECIPE_LABEL}={recipe}",
        "-", stdin=dockerfile.encode(),
    )  # fmt: skip
    assert built.returncode == 0, built.stderr.decode()[-400:]
    return built.stdout.decode().strip()


def _stand_in(marker: str) -> str:
    return f"FROM {helper_mod.HELPER_BASE}\nRUN echo {marker} > /marker\n"


def _push(
    registry: str, name: str, recipe: str, dockerfile: str, *build_args: str
) -> str:
    """``dockerfile`` built with ``recipe`` as its recipe label, pushed as
    ``<registry>/<name>``, its local copies removed: the pinned reference."""
    tag = f"{registry}/{name}:t"
    image = _build(tag, recipe, dockerfile, *build_args)
    pushed = _run("push", tag)
    assert pushed.returncode == 0, pushed.stderr.decode()[-400:]
    digest = re.findall(r"digest: (sha256:[0-9a-f]{64})", pushed.stdout.decode())
    assert digest, pushed.stdout.decode()[-400:]
    _run("image", "rm", "--force", tag)
    _run("image", "rm", "--force", image)
    ref = f"{registry}/{name}@{digest[-1]}"
    assert _run("image", "inspect", ref).returncode != 0  # gone: the pull is real
    return ref


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(helper_mod, "_READY", type(helper_mod._READY)())
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})
    monkeypatch.setattr(helper_mod, "_PULL_FAILED", {})
    monkeypatch.setattr(helper_mod, "_SOURCE", {})
    monkeypatch.setenv(helper_mod.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.delenv(helper_mod.PULL_ENV, raising=False)
    monkeypatch.setattr(helper_mod, "HELPER_IMAGE_RECIPE", helper_mod.HELPER_RECIPE)


def _cleanup(*images: str) -> None:
    for image in images:
        seen = (
            _run("image", "inspect", "--format", "{{json .RepoTags}}", image)
            if image
            else None
        )
        if seen is not None and seen.returncode == 0:
            for tag in json.loads(seen.stdout):
                _run("image", "rm", tag)
            _run("image", "rm", "--force", image)


def _marker(image: str) -> str:
    ran = _run("run", "--rm", "--network", "none", image, "cat", "/marker")
    assert ran.returncode == 0, ran.stderr.decode()[-300:]
    return ran.stdout.decode().strip()


@pytest.mark.asyncio
async def test_the_pinned_image_is_pulled_checked_and_run_by_its_id(
    registry, fresh, sandbox, monkeypatch, tmp_path
):
    """Pulled, checked and run by its ID. A later process with no record (another
    user, an unwritable state folder) pulls it again after the first kept its
    tag, which lists another name first among its RepoDigests, and still takes it."""
    ref = _push(registry, "helper", helper_mod.HELPER_RECIPE, _stand_in("real"))
    monkeypatch.setattr(helper_mod, "HELPER_IMAGE", ref)
    daemon = sandbox._helper_daemon()
    image = ""
    try:
        image = await helper_mod.ensure_image(sandbox._helper_docker, daemon)
        assert sandbox.metadata_rules_image == "pulled"
        assert _marker(image) == "real"
        seen = json.loads(
            _run("image", "inspect", "--format", "{{json .RepoDigests}}", ref).stdout
        )
        assert ref in seen, seen
        if _containerd_store():
            assert seen.index(ref) > 0, seen  # the kept tag's name sorts first

        monkeypatch.setattr(helper_mod, "_READY", type(helper_mod._READY)())
        monkeypatch.setattr(helper_mod, "_SOURCE", {})
        monkeypatch.setenv(helper_mod.STATE_DIR_ENV, str(tmp_path / "elsewhere"))
        assert await helper_mod.ensure_image(sandbox._helper_docker, daemon) == image
        assert sandbox.metadata_rules_image == "pulled"
    finally:
        _cleanup(image, ref)


@pytest.mark.asyncio
async def test_a_loaded_record_under_the_pinned_name_is_never_what_runs(
    registry, fresh, sandbox, monkeypatch, tmp_path
):
    """Another image loaded from an archive that names it by the pinned
    reference, with this recipe's labels: its RepoDigests then list that exact
    reference. The pull by digest and the inspect of that reference give the
    real image, which is what runs."""
    ref = _push(registry, "helper", helper_mod.HELPER_RECIPE, _stand_in("real"))
    evil = _build(
        f"evil-{uuid.uuid4().hex[:8]}:t", helper_mod.HELPER_RECIPE, _stand_in("evil")
    )
    saved = tmp_path / "evil.tar"
    assert _run("save", "-o", str(saved), evil).returncode == 0
    forged = tmp_path / "forged.tar"
    with tarfile.open(saved) as src, tarfile.open(forged, "w") as dst:
        for member in src.getmembers():
            file = src.extractfile(member) if member.isfile() else None
            data = file.read() if file is not None else None
            if member.name == "index.json" and data is not None:
                index = json.loads(data)
                for m in index["manifests"]:
                    m.setdefault("annotations", {})["io.containerd.image.name"] = ref
                data = json.dumps(index).encode()
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data) if data is not None else None)
    _run("image", "rm", "--force", evil)
    loaded = _run("load", "-i", str(forged))
    assert loaded.returncode == 0, loaded.stderr.decode()[-300:]
    if _containerd_store():
        listed = _run("image", "inspect", "--format", "{{json .RepoDigests}}", evil)
        assert ref in json.loads(listed.stdout), listed.stdout  # the forgery holds
    monkeypatch.setattr(helper_mod, "HELPER_IMAGE", ref)
    daemon = sandbox._helper_daemon()
    image = ""
    try:
        image = await helper_mod.ensure_image(sandbox._helper_docker, daemon)
        assert sandbox.metadata_rules_image == "pulled"
        assert image != evil and _marker(image) == "real"
    finally:
        _cleanup(image, ref, evil)


@pytest.mark.asyncio
async def test_a_registry_that_never_answers_costs_the_cap_then_the_build_serves(
    fresh, sandbox, monkeypatch
):
    """A registry that accepts the connection and never answers: the pull is
    given up at its cap, its docker process is gone, and the local build
    serves."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        listener.settimeout(0.5)
        while not stop.is_set():
            try:
                held.append(listener.accept()[0])
            except OSError:
                continue

    threading.Thread(target=accept, daemon=True).start()
    port = listener.getsockname()[1]
    ref = f"127.0.0.1:{port}/helper@sha256:" + "ab" * 32
    monkeypatch.setattr(helper_mod, "HELPER_IMAGE", ref)
    monkeypatch.setattr(helper_mod, "_PULL_SEC", 5)
    pulls: list[float] = []

    async def timed(args: list[str], *, timeout: float, stdin: bytes | None = None):
        started = time.monotonic()
        try:
            return await sandbox._helper_docker(args, timeout=timeout, stdin=stdin)
        finally:
            if args[:1] == ["pull"]:
                pulls.append(time.monotonic() - started)

    image = ""
    try:
        image = await helper_mod.ensure_image(timed, sandbox._helper_daemon())
        assert sandbox.metadata_rules_image == "built"
        # Given up at its cap (Docker alone takes about 90 s), not before.
        assert len(pulls) == 1 and 5 <= pulls[0] < 5 + 3, pulls
        left = subprocess.run(
            ["pgrep", "-f", f"pull --quiet 127.0.0.1:{port}/"], capture_output=True
        )
        assert left.returncode == 1, left.stdout
    finally:
        stop.set()
        for s in held:
            s.close()
        listener.close()
        _cleanup(image)


@pytest.mark.asyncio
async def test_a_pinned_image_with_another_recipes_label_is_refused_and_the_build_serves(
    registry, fresh, sandbox, monkeypatch
):
    ref = _push(registry, "other", "0" * 16, _stand_in("other"))
    monkeypatch.setattr(helper_mod, "HELPER_IMAGE", ref)
    image = ""
    try:
        image = await helper_mod.ensure_image(
            sandbox._helper_docker, sandbox._helper_daemon()
        )
        assert sandbox.metadata_rules_image == "built"
        labels = json.loads(
            _run(
                "image", "inspect", "--format", "{{json .Config.Labels}}", image
            ).stdout
        )
        assert labels[helper_mod.HELPER_RECIPE_LABEL] == helper_mod.HELPER_RECIPE
        got = _run("image", "inspect", "--format", "{{json .RepoDigests}}", image)
        assert ref not in json.loads(got.stdout or b"[]")
    finally:
        _cleanup(image, ref)


@pytest.mark.asyncio
async def test_a_pinned_image_for_another_platform_is_refused_and_the_build_serves(
    registry, fresh, sandbox, monkeypatch
):
    """A single-platform image of another architecture, with this recipe's
    labels: Docker pulls it, and it would start, then die at its first exec.
    It is refused, and the build serves. (Pushed as one manifest, no index: an
    index without the daemon's platform fails the pull itself.)"""
    arch = _run("version", "--format", "{{.Server.Arch}}").stdout.decode().strip()
    other = "linux/arm64" if arch == "amd64" else "linux/amd64"
    dockerfile = f"FROM {helper_mod.HELPER_BASE}\n"  # no RUN: nothing to emulate
    ref = _push(
        registry, "foreign", helper_mod.HELPER_RECIPE, dockerfile,
        "--platform", other, "--provenance=false",
    )  # fmt: skip
    monkeypatch.setattr(helper_mod, "HELPER_IMAGE", ref)
    image = ""
    try:
        image = await helper_mod.ensure_image(
            sandbox._helper_docker, sandbox._helper_daemon()
        )
        assert sandbox.metadata_rules_image == "built"
        pulled = _run(
            "image", "inspect", "--format", "{{.Id}} {{.Os}}/{{.Architecture}}", ref
        )
        assert pulled.returncode == 0, "the pull itself must succeed"
        foreign_id, platform = pulled.stdout.decode().split()
        assert platform == other and image != foreign_id
    finally:
        _cleanup(image, ref)
