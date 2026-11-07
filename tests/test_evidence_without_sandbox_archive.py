"""Guards the Daytona evidence capture against the archive it wrote on the
sandbox's own disk (review/evidence.py's capture script at e118a8d3): an
uncompressed tar of the whole workspace in /tmp, so a workspace larger than
the free space failed with ENOSPC, the trial lost its review and its score,
and every retry failed the same way. A benchmark's own infrastructure works
around it as a patch.

The sandbox here is this machine: commands run as the toolbox would run them
and the SDK's bulk download copies the files it is asked for.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchflow.sandbox.daytona as daytona_mod
from benchflow._utils.reachable_write import private_folder
from benchflow.review.evidence import (
    EvidenceError,
    capture_task_evidence,
    capture_workspace,
    validate_workspace,
)
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.sandbox.daytona_strategies import _DaytonaDirect
from benchflow.task.config import ArtifactConfig
from tests.test_review_evidence import LocalTransport

# Put before every Python program the sandbox runs: /tmp and /var/tmp hold
# 64 KiB, and a write past that fails with ENOSPC, as on a sandbox disk whose
# free space is smaller than the workspace.
_SMALL_TMP = """
import builtins, errno, os
_room = [64 * 1024]
_open = builtins.open
class _Capped:
    def __init__(self, stream):
        self._stream = stream
    def write(self, data):
        if len(data) > _room[0]:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        _room[0] -= len(data)
        return self._stream.write(data)
    def __getattr__(self, name):
        return getattr(self._stream, name)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self._stream.close()
def _capped_open(file, mode="r", *args, **kwargs):
    stream = _open(file, mode, *args, **kwargs)
    if isinstance(file, str) and set(mode) & set("wax") and (
        os.path.realpath(file).startswith(("/tmp/", "/var/tmp/"))
    ):
        return _Capped(stream)
    return stream
builtins.open = _capped_open
"""

# A process the agent left running appends to growing.log while the capture
# reads it.
_GROWING_LOG = """
import pathlib
_read = pathlib.Path.open
def _open(self, mode="r", *args, **kwargs):
    stream = _read(self, mode, *args, **kwargs)
    if mode == "rb" and self.name == "growing.log":
        with _read(self, "ab") as log:
            log.write(b"a line written while it was captured\\n")
    return stream
pathlib.Path.open = _open
"""


@dataclass
class _Request:
    source: str
    destination: str


class _Fs:
    """The SDK's ``sandbox.fs``: ``download_files`` streams each requested
    file to its destination and reports it, as SDK 0.207 does."""

    def __init__(self, change=None):
        self.change = change
        self.batches: list[int] = []
        self.in_flight = 0
        self.most_in_flight = 0

    async def download_files(self, *, files):
        self.batches.append(len(files))
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            for request in files:
                data = Path(request.source).read_bytes()
                if self.change is not None:
                    data = self.change(request.source, data)
                Path(request.destination).parent.mkdir(parents=True, exist_ok=True)
                Path(request.destination).write_bytes(data)
            return [
                SimpleNamespace(
                    source=request.source, result=request.destination, error=None
                )
                for request in files
            ]
        finally:
            self.in_flight -= 1


class _Daytona(DaytonaSandbox):
    """A started Daytona sandbox whose disk is this machine."""

    def __init__(self, fs=None, *, shim="", strategy=_DaytonaDirect, python=True):
        self._sandbox = SimpleNamespace(fs=fs or _Fs())
        self._strategy = object.__new__(strategy)
        self.logger = logging.getLogger("test.daytona")
        self.shim = shim
        self.python = python
        self.archives: list[str] = []

    async def runtime_kit_for_route(self, service="main"):
        return None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, **kwargs):
        argv = shlex.split(command)
        if argv[0] == "python3":
            if not self.python:
                return ExecResult(
                    return_code=127, stdout="", stderr="sh: 1: python3: not found"
                )
            argv[0] = sys.executable
            program = argv.index("-c") + 1
            argv[program] = self.shim + argv[program]
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(
            return_code=process.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    async def download_file(self, source_path, target_path):
        self.archives.append(source_path)
        shutil.copyfile(source_path, target_path)


@pytest.fixture(autouse=True)
def _sdk_request(monkeypatch):
    monkeypatch.setattr(daytona_mod, "FileDownloadRequest", _Request)


def _workspace(root: Path) -> Path:
    """Every kind of entry the capture keeps or records, a file larger than
    the sandbox's /tmp, and more files than one bulk request carries."""
    workspace = root / "box" / "app"
    (workspace / "nested" / "empty").mkdir(parents=True)
    payload = workspace / "nested" / "résult data.bin"
    payload.write_bytes(bytes(range(256)) * 4096)
    (workspace / "empty.txt").write_bytes(b"")
    (workspace / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (workspace / "run.sh").chmod(0o755)
    (workspace / "private.txt").write_text("solver notes\n")
    (workspace / "private.txt").chmod(0o600)
    (workspace / "shared").mkdir(mode=0o750)
    (workspace / "shared" / "result.json").write_text('{"answer": 42}\n')
    for index in range(300):
        many = workspace / "many" / f"{index:03d}.txt"
        many.parent.mkdir(exist_ok=True)
        many.write_text(f"{index}\n")
    (workspace / "relative-link").symlink_to("nested/résult data.bin")
    (workspace / "absolute-link").symlink_to(payload)
    (workspace / "directory-link").symlink_to("nested", target_is_directory=True)
    os.link(payload, workspace / "hardlink.bin")
    outside = root / "box" / "secret.txt"
    outside.write_text("never captured\n")
    (workspace / "escape-link").symlink_to(outside)
    os.mkfifo(workspace / "pipe")
    (workspace / ".cache" / "pip" / "wheels").mkdir(parents=True)
    (workspace / ".cache" / "pip" / "wheels" / "x.whl").write_bytes(b"cache")
    return workspace


def _tree(root: Path) -> dict[str, tuple]:
    found: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        name = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            found[name] = ("symlink", os.readlink(path))
        elif stat.S_ISDIR(info.st_mode):
            found[name] = ("directory", stat.S_IMODE(info.st_mode))
        else:
            found[name] = ("file", stat.S_IMODE(info.st_mode), path.read_bytes())
    return found


async def test_a_workspace_larger_than_the_sandbox_tmp_is_captured(tmp_path):
    """Red at e118a8d3: the archive write fails with ENOSPC. The bundle is
    the archive capture's, byte for byte: tree, modes, links, manifest."""
    workspace = _workspace(tmp_path)
    fs = _Fs()
    env = _Daytona(fs, shim=_SMALL_TMP)

    manifest = await capture_workspace(env, str(workspace), tmp_path / "listed")

    assert env.archives == []
    files = [e for e in manifest.entries if e.kind == "file" and e.size]
    assert sum(fs.batches) == len(files) and max(fs.batches) <= 256
    archived = await capture_workspace(
        LocalTransport(), str(workspace), tmp_path / "archived"
    )
    assert manifest == archived
    assert (tmp_path / "listed" / "manifest.json").read_bytes() == (
        tmp_path / "archived" / "manifest.json"
    ).read_bytes()
    assert _tree(tmp_path / "listed" / "workspace") == _tree(
        tmp_path / "archived" / "workspace"
    )
    assert {e.reason for e in manifest.exclusions} == {
        "package_cache",
        "special_file",
        "symlink_escape",
    }
    validate_workspace(tmp_path / "listed" / "workspace", manifest)


async def test_declared_artifacts_are_captured_without_an_archive_too(tmp_path):
    """Red at e118a8d3: the review's capture of the workspace and of each
    declared artifact outside it each wrote an archive in /tmp."""
    workspace = _workspace(tmp_path)
    outputs = tmp_path / "box" / "outputs"
    outputs.mkdir()
    (outputs / "figure.bin").write_bytes(b"\x89PNG" * 40_000)

    manifest = await capture_task_evidence(
        _Daytona(shim=_SMALL_TMP),
        str(workspace),
        tmp_path / "evidence",
        artifacts=[ArtifactConfig(source=str(outputs))],
    )

    (artifact,) = manifest.artifacts
    assert artifact.status == "captured" and artifact.bundle_path is not None
    captured = tmp_path / "evidence" / artifact.bundle_path / "workspace"
    assert (captured / "figure.bin").read_bytes() == b"\x89PNG" * 40_000


@pytest.mark.parametrize("change", ["grown", "rewritten"])
async def test_a_file_that_differs_from_its_listing_fails_the_capture(tmp_path, change):
    """The sandbox's /tmp has room, so an archive capture would succeed: a
    file that differs from its listing is an evidence error, never a reason
    to capture again another way."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42\n")
    (workspace / "notes.txt").write_text("checked\n")

    def alter(source: str, data: bytes) -> bytes:
        if not source.endswith("answer.txt"):
            return data
        return data + b"0" if change == "grown" else data[::-1]

    env = _Daytona(_Fs(change=alter))
    with pytest.raises(EvidenceError, match=r"differs from its listing: answer\.txt"):
        await capture_workspace(env, str(workspace), tmp_path / "evidence")

    assert env.archives == []
    assert not (tmp_path / "evidence").exists()
    assert not list(tmp_path.glob(".benchflow-private-*"))


async def test_a_file_that_changes_while_it_is_listed_fails_as_before(tmp_path):
    """A control: the archive capture fails the same way."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "growing.log").write_text("first line\n")

    with pytest.raises(EvidenceError, match="workspace changed during evidence"):
        await capture_workspace(
            _Daytona(shim=_GROWING_LOG), str(workspace), tmp_path / "evidence"
        )

    assert not (tmp_path / "evidence").exists()


async def test_sandboxes_without_bulk_download_keep_the_archive(tmp_path):
    """A control: Docker (here the local transport), a compose task's DinD
    sandbox and a sandbox without python3 capture through an archive."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42\n")
    local = LocalTransport()
    dind = _Daytona(strategy=_DaytonaDinD)
    shell_only = _Daytona(python=False)

    for name, env in (("local", local), ("dind", dind), ("shell", shell_only)):
        manifest = await capture_workspace(env, str(workspace), tmp_path / name)
        assert [entry.path for entry in manifest.entries] == ["answer.txt"]

    archives = [*local.downloaded, *dind.archives, *shell_only.archives]
    assert len(archives) == 3
    assert all(Path(path).name.startswith("runtime-evidence-") for path in archives)
    assert dind._sandbox.fs.batches == shell_only._sandbox.fs.batches == []


async def test_download_files_batches_and_bounds_requests(tmp_path):
    source = tmp_path / "source"
    names = [f"d{index % 7}/{index:04d}.txt" for index in range(1300)]
    for name in names:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(name)
    fs = _Fs()

    with private_folder(tmp_path / "evidence") as folder:
        target = Path(folder) / "files"
        await _Daytona(fs).download_files(str(source), names, target)
        assert all((target / name).read_text() == name for name in names)

    assert sorted(fs.batches) == [20] + [256] * 5
    assert fs.most_in_flight == 4


@pytest.mark.parametrize("name", ["../escape.txt", "/etc/passwd", "a/../../b"])
async def test_download_files_refuses_names_outside_the_target(tmp_path, name):
    fs = _Fs()

    with private_folder(tmp_path / "evidence") as folder, pytest.raises(ValueError):
        await _Daytona(fs).download_files(str(tmp_path), [name], Path(folder) / "f")

    assert fs.batches == []


async def test_download_files_needs_the_single_container_sandbox(tmp_path):
    assert _Daytona().can_download_files
    assert not _Daytona(strategy=_DaytonaDinD).can_download_files
    with pytest.raises(NotImplementedError):
        await _Daytona(strategy=_DaytonaDinD).download_files(
            str(tmp_path), ["a.txt"], tmp_path / "f"
        )
