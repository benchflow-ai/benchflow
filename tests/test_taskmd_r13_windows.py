"""Native-shape Windows streaming: no POSIX descriptors disguised as handles."""

import contextlib
import io
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

from benchflow._utils import reachable_write as rw
from benchflow.rewards import _safe_evidence


def test_streaming_uses_windows_handle_adapter(tmp_path, monkeypatch):
    """Guards R12-3 introduced by 7c24f4d88, with every POSIX dir_fd refused."""
    active = {}
    next_handle = 0

    class API:
        def open(self, path, *, directory, parent=None):
            nonlocal next_handle
            assert directory
            path = active[parent] / path if parent is not None else path
            assert path.is_dir() and not path.is_symlink()
            next_handle += 1
            active[next_handle] = path
            return next_handle

        def close(self, handle):
            active.pop(handle)

    api = API()

    @contextlib.contextmanager
    def open_directory(path, *, evidence_root):
        handle = api.open(path, directory=True)
        try:
            yield SimpleNamespace(api=api, handle=handle)
        finally:
            api.close(handle)

    def posix_forbidden(function):
        def call(*args, **kwargs):
            assert not any(key.endswith("dir_fd") for key in kwargs), (
                "native Windows has no POSIX dir_fd"
            )
            return function(*args, **kwargs)

        return call

    facade = vars(os) | {"name": "nt"}
    for name in ("open", "mkdir", "stat", "rename", "unlink"):
        facade[name] = posix_forbidden(getattr(os, name))
    original_replace = os.replace

    def replace(source, destination):
        assert Path(destination).parent in active.values()
        original_replace(source, destination)

    facade["replace"] = replace
    monkeypatch.setattr(rw, "os", SimpleNamespace(**facade))
    monkeypatch.setattr(rw, "_held_handles_available", lambda: False)
    monkeypatch.setattr(_safe_evidence, "open_directory", open_directory)
    target = rw.ReachablePath.directory(tmp_path) / "output.json"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("output.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    stream.seek(0)
    rw.publish_tar_stream(stream, target, directory=False)
    assert (tmp_path / "output.json").read_bytes() == b"{}"
    assert list(active.values()) == [tmp_path]
