"""The two Docker daemons bfx's delivery tests run against, from the host.

``local`` is the daemon the ``docker`` CLI reaches by default. ``remote`` is
a second, real daemon reached as ``remote-docker`` reaches one: a TLS
``tcp://`` address in ``BENCHFLOW_TEST_REMOTE_DOCKER_HOST`` with the client
certificates in ``BENCHFLOW_TEST_REMOTE_DOCKER_CERT_PATH`` (ca.pem, cert.pem,
key.pem). On a build VM that is a ``docker:29.1.3-dind`` container started
with ``--init``, its own DNS and its TLS port published on 127.0.0.1 (see
results/TR-BFX.md). Tests that need it skip without it.

Every helper here talks to a daemon with the ``docker`` CLI from the host; no
step runs inside a sandbox.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pytest

REMOTE_HOST_ENV = "BENCHFLOW_TEST_REMOTE_DOCKER_HOST"
REMOTE_CERTS_ENV = "BENCHFLOW_TEST_REMOTE_DOCKER_CERT_PATH"


@dataclass
class Daemon:
    name: str
    env: dict[str, str] | None = None
    #: Volumes and containers a test made here, removed by name at the end.
    volumes: list[str] = field(default_factory=list)
    containers: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def run(
        self, args: list[str], *, input: bytes | None = None, timeout: float = 300
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["docker", *args],
            input=input,
            capture_output=True,
            env=self.env,
            timeout=timeout,
        )

    def ok(self, args: list[str], **kw) -> bytes:
        done = self.run(args, **kw)
        assert done.returncode == 0, (args, done.stderr.decode(errors="replace"))
        return done.stdout

    def inspect(self, kind: str, name: str) -> dict:
        return json.loads(self.ok([kind, "inspect", name]))[0]

    def cp_out(self, container: str, path: str) -> tarfile.TarFile:
        """``docker cp`` of *path* out of *container*, as a tar archive."""
        return tarfile.open(
            fileobj=io.BytesIO(self.ok(["cp", f"{container}:{path}", "-"])), mode="r:"
        )

    def file_sha256(self, container: str, path: str) -> str:
        """The SHA-256 of the regular file *path*, copied out of *container*."""
        with self.cp_out(container, path) as tar:
            [member] = tar.getmembers()
            assert member.isreg(), member
            assert member.name == PurePosixPath(path).name, member.name
            f = tar.extractfile(member)
            assert f is not None
            return hashlib.sha256(f.read()).hexdigest()

    def folder_names(self, container: str, path: str) -> list[str]:
        """The names directly in the folder *path* of *container*."""
        top = PurePosixPath(path).name
        with self.cp_out(container, path) as tar:
            return sorted(
                {
                    PurePosixPath(m.name).parts[1]
                    for m in tar.getmembers()
                    if len(PurePosixPath(m.name).parts) > 1
                    and PurePosixPath(m.name).parts[0] == top
                }
            )

    def volume_view(self, volume: str, image: str) -> tuple[list[str], dict]:
        """What *volume* holds, read through a created and never started
        container: (names, {name: sha256 of each regular file})."""
        name = f"bfx-test-view-{uuid.uuid4().hex[:10]}"
        self.ok(
            [
                "create", "--name", name, "--network", "none",
                "--mount", f"type=volume,src={volume},dst=/v,readonly,volume-nocopy",
                image, "true",
            ]
        )  # fmt: skip
        try:
            sums = {}
            with self.cp_out(name, "/v") as tar:
                for m in tar.getmembers():
                    parts = PurePosixPath(m.name).parts
                    if len(parts) == 2 and m.isreg():
                        f = tar.extractfile(m)
                        assert f is not None
                        sums[parts[1]] = hashlib.sha256(f.read()).hexdigest()
            return self.folder_names(name, "/v"), sums
        finally:
            self.run(["rm", "-f", name])

    def cleanup(self) -> None:
        for c in self.containers:
            self.run(["rm", "-f", c])
        for v in self.volumes:
            self.run(["volume", "rm", "-f", v])
        for i in self.images:
            self.run(["image", "rm", "-f", i])


def _answers(env: dict[str, str] | None) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=30, env=env
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def local_daemon() -> Daemon:
    if not _answers(None):
        pytest.skip("no local Docker daemon answers")
    return Daemon("local")


def remote_settings() -> tuple[str, str] | None:
    host = os.environ.get(REMOTE_HOST_ENV, "").strip()
    certs = os.environ.get(REMOTE_CERTS_ENV, "").strip()
    return (host, certs) if host and certs else None


def remote_daemon() -> Daemon:
    settings = remote_settings()
    if settings is None:
        pytest.skip(f"no second Docker daemon ({REMOTE_HOST_ENV} unset)")
    host, certs = settings
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DOCKER_CONTEXT", "DOCKER_HOST", "DOCKER_TLS_VERIFY")
    }
    env.update(DOCKER_HOST=host, DOCKER_TLS_VERIFY="1", DOCKER_CERT_PATH=certs)
    if not _answers(env):
        pytest.skip(f"the second Docker daemon at {REMOTE_HOST_ENV} does not answer")
    return Daemon("remote", env=env)


def point_remote_docker_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure BenchFlow's ``remote-docker`` for the second daemon."""
    from benchflow.sandbox.remote_docker import REMOTE_DOCKER_HOST_ENV

    settings = remote_settings()
    assert settings is not None
    host, certs = settings
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, host)
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", certs)


def own_volume_name(monkeypatch: pytest.MonkeyPatch, daemon: Daemon) -> str:
    """A helper-volume name of this test's own, used in place of
    ``benchflow-bfx-<sha12>`` so no test touches a daemon's real helper
    volume; removed by name at the end."""
    from benchflow._bfx import delivery

    name = f"benchflow-bfx-test-{uuid.uuid4().hex[:12]}"
    monkeypatch.setattr(delivery, "volume_name", lambda arch: name)
    monkeypatch.setattr(delivery, "_verified_volumes", set())
    daemon.volumes.append(name)
    return name
