"""Guards FF-sw-launchers against warm context reads on the event loop (363d815b)."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox.agentcore_image import AgentCoreImagePublisher
from benchflow.task.config import SandboxConfig


def publisher(path):
    return AgentCoreImagePublisher(
        environment_dir=path,
        environment_name="task",
        task_env_config=SandboxConfig(),
        region="us-east-1",
        ecr_repository="repo",
        client_factory=MagicMock(),
        logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_identity_is_resolved_off_the_event_loop(tmp_path, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    image = publisher(tmp_path)
    main_thread = threading.get_ident()
    original = image.identity

    def identity():
        assert threading.get_ident() != main_thread
        return original()

    monkeypatch.setattr(image, "identity", identity)
    monkeypatch.setattr(image, "_registry", lambda: "registry")
    monkeypatch.setattr(image, "_ensure_ecr_repository", lambda: None)
    monkeypatch.setattr(image, "_image_exists", lambda tag: True)
    monkeypatch.setattr(image, "_resolve_image_digest", lambda tag: "repo@sha256:one")
    provisioning.reset_cache()
    await image.publish(force_build=False)
    provisioning.reset_cache()


def test_context_hashing_does_not_allocate_a_whole_file(tmp_path, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "data").write_bytes(b"data" * 1024)

    def forbidden(*args, **kwargs):
        raise AssertionError("whole-file allocation")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert len(publisher(tmp_path).identity()[0]) == 64


def test_warm_snapshot_reads_once_and_detects_preserved_mtime_edits(
    tmp_path, monkeypatch
):
    """Same trusted inode version shares bytes; ctime invalidates an edit."""
    import os

    from benchflow.sandbox import agentcore_provisioning as p

    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    data = tmp_path / "data"
    data.write_bytes(b"first")
    calls = []
    original = os.fdopen

    class Reader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def fileno(self):
            return self.source.fileno()

        def read(self, count):
            calls.append(count)
            return self.source.read(count)

    monkeypatch.setattr(
        p.os,
        "fdopen",
        lambda *args, **kwargs: (
            Reader(original(*args, **kwargs))
            if len(args) > 1 and args[1] == "rb"
            else original(*args, **kwargs)
        ),
    )
    p.reset_cache()
    first = publisher(tmp_path).identity()
    reads = len(calls)
    for _ in range(5):
        assert publisher(tmp_path).identity() == first
    assert len(calls) == reads
    before = data.stat()
    data.write_bytes(b"other")
    os.utime(data, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert publisher(tmp_path).identity() != first
    assert len(calls) > reads
    assert max(calls) <= 1024 * 1024
    p.reset_cache()
