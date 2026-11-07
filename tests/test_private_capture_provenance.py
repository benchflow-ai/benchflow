"""Private capture regressions at frozen 7bc81e6ba.

Local and remote adapters use their real metadata, streaming extraction and
final rollout handlers. Only subprocess transport and the failing I/O are fake.
"""

import asyncio
import base64
import errno
import io
import json
import os
import tarfile
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest

from benchflow._utils import reachable_write as rw
from benchflow._utils.scoring import NO_USABLE_REWARD
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox
from benchflow.task.verifier_errors import verifier_infrastructure_failure
from tests.test_r5_release_contract import observe, prime
from tests.test_verifier_own_failures import _assess, _task

SPOOL_FAULTS = [
    ("retained", "create", "EACCES"),
    ("retained", "create", "EIO"),
    ("retained", "create", "ENOSPC"),
    ("member", "create", "EACCES"),
    ("retained", "write", "EACCES"),
    ("member", "write", "EACCES"),
    ("retained", "flush", "EACCES"),
    ("member", "flush", "EACCES"),
    ("member", "read", "EACCES"),
    ("member", "read", "EIO"),
    ("member", "read", "ENOSPC"),
    ("member", "seek", "EACCES"),
    ("member", "tell", "EACCES"),
    ("retained", "close", "EACCES"),
]


def capture_transport(monkeypatch, remote, present, control="valid"):
    backend = object.__new__(RemoteDockerSandbox if remote else DockerSandbox)
    if remote:
        backend.docker_host = RemoteDockerHost("ssh://fixture.invalid")
    backend._main_container = "a" * 64
    backend.environment_name = "fixture"
    backend._docker_compose_global_args = lambda: (
        ["-H", "ssh://fixture.invalid"] if remote else []
    )
    backend._docker_compose_env = lambda: {}
    backend._compose_transport_env = lambda: {}
    backend._docker_process_env = lambda _environment=None: {}
    state = {"metadata": [], "copies": [], "active": False, "spools": 0, "faults": []}
    missing = "reward.txt" if present.endswith("json") else "reward.json"

    async def spawn(*argv, stdout=None, **kwargs):
        assert argv[0] == "docker"
        if remote:
            assert argv[1:3] == ("-H", "ssh://fixture.invalid")
        if argv[-2:] == ("system", "dial-stdio"):

            async def communicate(request):
                verb, endpoint, _ = request.decode().split("\r\n")[0].split()
                assert verb == "HEAD"
                source = parse_qs(urlsplit(endpoint).query)["path"][0]
                state["metadata"].append(source)
                name = PurePosixPath(source).name
                status, mode, link = 404, None, ""
                if name == present:
                    status, mode = 200, 0o644
                elif source in ("/logs/verifier", "/logs", "/"):
                    status, mode = 200, 0x800001ED
                elif name == missing and control == "unsafe-companion":
                    status, mode, link = 200, 0x080001FF, "/missing"
                elif name == missing and control == "conflict":
                    status, mode = 200, 0o644
                header = f"HTTP/1.1 {status} fixture\r\n"
                if mode is not None:
                    encoded = base64.b64encode(
                        json.dumps({"mode": mode, "linkTarget": link}).encode()
                    ).decode()
                    header += f"X-Docker-Container-Path-Stat: {encoded}\r\n"
                return (header + "\r\n").encode(), b""

            return SimpleNamespace(returncode=0, communicate=communicate)
        assert argv[-1] == "-" and "cp" in argv
        source = argv[-2].split(":", 1)[1]
        name = PurePosixPath(source).name
        state["copies"].append(source)
        stderr = asyncio.StreamReader()
        code = 1
        if name == present or control in ("unsafe-companion", "conflict"):
            if name == present:
                content = b'{"reward":1}' if present.endswith("json") else b"1"
                if control == "malformed":
                    content = b"not a reward"
            else:
                content = b'{"reward":0}' if missing.endswith("json") else b"0"
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as archive:
                entry = tarfile.TarInfo(name)
                if name == missing and control == "unsafe-companion":
                    entry.type, entry.linkname = tarfile.SYMTYPE, "/missing"
                    archive.addfile(entry)
                else:
                    entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
            # No earlier setup/logging spool may receive this injected fault.
            # The actual copy is now delivering the metadata-verified file.
            assert source in state["metadata"]
            state["active"] = True
            owned_fd = os.dup(stdout)
            os.set_blocking(owned_fd, False)
            stopped = [False]

            async def produce():
                remaining = memoryview(payload.getvalue())
                try:
                    while remaining and not stopped[0]:
                        try:
                            sent = os.write(owned_fd, remaining)
                        except BlockingIOError:
                            await asyncio.sleep(0.001)
                        except BrokenPipeError:
                            return 1
                        else:
                            remaining = remaining[sent:]
                    return -9 if stopped[0] else 0
                finally:
                    os.close(owned_fd)

            producer = asyncio.create_task(produce())

            async def wait():
                return await asyncio.shield(producer)

            def kill():
                stopped[0] = True

            stderr.feed_eof()
            return SimpleNamespace(stderr=stderr, wait=wait, kill=kill)
        else:
            stderr.feed_data(b"Could not find the file in container\n")
        stderr.feed_eof()
        return SimpleNamespace(
            stderr=stderr, wait=AsyncMock(return_value=code), kill=lambda: None
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return backend, state, missing


def fault_spool(monkeypatch, state, slot, phase, kind):
    original = rw.tempfile.TemporaryFile
    fault = OSError(getattr(errno, kind), "fixture private spool storage fault")

    class Stream:
        def __init__(self, stream, selected):
            self.stream = stream
            self.selected = selected

        def __getattr__(self, name):
            target = getattr(self.stream, name)
            if name == "seek" and phase == "flush" and self.selected:
                # Buffered seek flushes pending bytes in F too. Expose that
                # same private flush to injection before the underlying seek.

                def seek(*args, **kwargs):
                    self.flush()
                    return target(*args, **kwargs)

                return seek
            if name != phase or not self.selected:
                return target

            def operate(*args, **kwargs):
                if not state["faults"]:
                    state["faults"].append((slot, phase, kind))
                    if name == "close":
                        self.stream.close()
                    raise fault
                return target(*args, **kwargs)

            return operate

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def create(*args, **kwargs):
        selected = False
        if state["active"]:
            state["spools"] += 1
            selected = state["spools"] == (1 if slot == "retained" else 2)
        if selected and phase == "create":
            state["faults"].append((slot, phase, kind))
            raise fault
        return Stream(original(*args, **kwargs), selected)

    monkeypatch.setattr(rw.tempfile, "TemporaryFile", create)


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("present", ["reward.json", "reward.txt"])
@pytest.mark.parametrize("slot,phase,kind", SPOOL_FAULTS)
async def test_private_archive_spool_fault_is_saved_unscored(
    tmp_path, monkeypatch, prior, remote, present, slot, phase, kind
):
    sandbox, verifier = _task(tmp_path)
    await prime(tmp_path / "earlier", sandbox, prior)
    assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
    sandbox.is_mounted = False
    backend, state, _ = capture_transport(monkeypatch, remote, present)
    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk unavailable")
    )
    sandbox.download_file = backend.download_file
    with monkeypatch.context() as patch:
        fault_spool(patch, state, slot, phase, kind)
        rewards, error, warnings = await _assess(sandbox, verifier)
    assert state["faults"] == [(slot, phase, kind)]
    assert f"/logs/verifier/{present}" in state["metadata"]
    assert f"/logs/verifier/{present}" in state["copies"]
    observe(tmp_path, rewards, error, "unscored", "private spool storage fault")
    if kind == "EACCES":
        assert "PrivateSpoolError" in error
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("present", ["reward.json", "reward.txt"])
@pytest.mark.parametrize(
    "control", ["valid", "malformed", "source-read", "unsafe-companion", "conflict"]
)
async def test_capture_evidence_controls_keep_their_scores(
    tmp_path, monkeypatch, prior, remote, present, control
):
    sandbox, verifier = _task(tmp_path)
    await prime(tmp_path / "earlier", sandbox, prior)
    assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
    sandbox.is_mounted = False
    backend, state, missing = capture_transport(monkeypatch, remote, present, control)
    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk unavailable")
    )
    sandbox.download_file = backend.download_file
    source_reads = []
    if control == "source-read":

        def denied_read(self, *args, **kwargs):
            source_reads.append(True)
            raise PermissionError(errno.EACCES, "fixture unreadable captured source")

        monkeypatch.setattr(tarfile.ExFileObject, "read", denied_read)
    rewards, error, warnings = await _assess(sandbox, verifier)
    if control == "valid":
        observe(tmp_path, rewards, error, 1)
        assert not warnings
        assert f"/logs/verifier/{missing}" in state["metadata"]
        assert f"/logs/verifier/{missing}" not in state["copies"]
    elif control == "conflict":
        # Conflicting companions are no usable reward in either direction:
        # neither may be discarded to accept the other's higher reward. That
        # is now the verifier's error, a failure in every score.
        observe(tmp_path, rewards, error, "unscored", NO_USABLE_REWARD)
        assert f"/logs/verifier/{missing}" in state["copies"]
        assert f"/logs/verifier/{present}" in state["copies"]
    else:
        observe(tmp_path, rewards, error, "unscored", NO_USABLE_REWARD)
        if control == "source-read":
            assert source_reads


@pytest.mark.parametrize("kind", ["EACCES", "EIO", "ENOSPC"])
@pytest.mark.parametrize("operation", ["mkdir", "open"])
def test_private_staging_retains_original_causes(
    tmp_path, monkeypatch, kind, operation
):
    caps = [rw.pin_destination(tmp_path / name) for name in ("a", "b")]
    original = os.mkdir if operation == "mkdir" else rw._open_beneath
    faults = []

    def deny(name, *args, **kwargs):
        if os.path.basename(os.fspath(name)).startswith(".benchflow-private-"):
            fault = OSError(getattr(errno, kind), "fixture private root failure")
            faults.append(fault)
            raise fault
        return original(name, *args, **kwargs)

    if operation == "mkdir":
        monkeypatch.setattr(os, "mkdir", deny)
    else:

        def opened(parent, name, *args, **kwargs):
            if name.startswith(".benchflow-private-"):
                fault = OSError(getattr(errno, kind), "fixture private root failure")
                faults.append(fault)
                raise fault
            return original(parent, name, *args, **kwargs)

        monkeypatch.setattr(rw, "_open_beneath", opened)
    with (
        pytest.raises(OSError) as caught,
        rw._private_staging([cap._anchor.path for cap in caps], None, temp=False),
    ):
        pytest.fail("all private creation attempts should fail")
    assert len(faults) == 2
    assert verifier_infrastructure_failure(caught.value)
    assert caught.value.__cause__ is faults[0]
    assert caught.value.attempts == tuple(faults)
    assert not list(tmp_path.rglob(".benchflow-private-*"))


def test_private_staging_can_use_a_safe_fallback(tmp_path, monkeypatch):
    caps = [rw.pin_destination(tmp_path / name) for name in ("a", "b")]
    original = os.mkdir
    denied = []

    def mkdir(name, *args, **kwargs):
        if (
            os.path.basename(os.fspath(name)).startswith(".benchflow-private-")
            and not denied
        ):
            denied.append(True)
            raise PermissionError(errno.EACCES, "fixture first candidate unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", mkdir)
    with rw._private_staging(
        [cap._anchor.path for cap in caps], None, temp=False
    ) as folder:
        assert folder.parent == caps[1]._anchor.path
        assert folder.is_dir()
    assert denied and not folder.exists()


@pytest.mark.parametrize("refusal", ["mount", "device", "parent"])
def test_unsafe_staging_selection_does_not_acquire_private_provenance(
    tmp_path, monkeypatch, refusal
):
    cap = rw.pin_destination(tmp_path / "candidate")
    device = None
    if refusal == "mount":
        monkeypatch.setattr(rw, "_inside_declared_mount", lambda _: True)
    elif refusal == "device":
        device = cap._anchor.path.stat().st_dev + 1
    else:

        def unsafe(_):
            raise PermissionError(errno.EACCES, "fixture unsafe parent selection")

        monkeypatch.setattr(rw, "_staging_owner", unsafe)
    with (
        pytest.raises(rw.ReachableWriteError) as caught,
        rw._private_staging([cap._anchor.path], device, temp=False),
    ):
        pytest.fail("unsafe location must not be selected")
    assert not verifier_infrastructure_failure(caught.value)
    assert not list(tmp_path.rglob(".benchflow-private-*"))


@pytest.mark.parametrize("phase", ["read", "short-write"])
def test_retained_hardlink_spool_failure_keeps_private_provenance(
    tmp_path, monkeypatch, phase
):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("root/first")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"1"))
        link = tarfile.TarInfo("root/second")
        link.type, link.linkname = tarfile.LNKTYPE, "root/first"
        archive.addfile(link)
    payload.seek(0)
    target = rw.pin_destination(tmp_path / "private") / "captured"
    state = {"active": True, "spools": 0, "faults": []}
    if phase == "read":
        fault_spool(monkeypatch, state, "retained", "read", "EACCES")
    else:
        original = rw.tempfile.TemporaryFile

        class ShortWrite:
            def __init__(self, stream):
                self.stream = stream

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def write(self, data):
                state["faults"].append(True)
                return 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.stream.close()

        monkeypatch.setattr(
            rw.tempfile, "TemporaryFile", lambda *a, **k: ShortWrite(original(*a, **k))
        )
    with pytest.raises(OSError) as caught:
        rw.publish_tar_stream(payload, target, directory=True)
    assert state["faults"]
    assert verifier_infrastructure_failure(caught.value)
    assert not target.exists()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("present", ["reward.json", "reward.txt"])
@pytest.mark.parametrize(
    "boundary", ["staging-root", "archive-leaf", "archive-stat", "publication"]
)
async def test_adjacent_private_archive_storage_is_saved_unscored(
    tmp_path, monkeypatch, prior, remote, present, boundary
):
    sandbox, verifier = _task(tmp_path)
    await prime(tmp_path / "earlier", sandbox, prior)
    assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
    sandbox.is_mounted = False
    backend, state, _ = capture_transport(monkeypatch, remote, present)
    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk unavailable")
    )
    sandbox.download_file = backend.download_file
    mkdir, opened = os.mkdir, rw._open_beneath
    faults = []

    def create(name, *args, **kwargs):
        if state["active"] and os.path.basename(os.fspath(name)).startswith(
            ".benchflow-download-"
        ):
            faults.append(True)
            raise PermissionError(
                errno.EACCES, "fixture private archive creation denied"
            )
        return mkdir(name, *args, **kwargs)

    def open_leaf(parent, name, flags, *args, **kwargs):
        if state["active"] and name == "payload" and flags & os.O_CREAT:
            faults.append(True)
            raise PermissionError(
                errno.EACCES, "fixture private archive creation denied"
            )
        return opened(parent, name, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        if boundary == "staging-root":
            patch.setattr(os, "mkdir", create)
        elif boundary == "archive-leaf":
            patch.setattr(rw, "_open_beneath", open_leaf)
        elif boundary == "archive-stat":
            extract, stat = rw._extract_members, os.stat

            def denied_stat(name, *args, **kwargs):
                if name == "payload" and kwargs.get("dir_fd") is not None:
                    # Restore before exception cleanup can open another anchor;
                    # descriptor capability discovery must see the real OS API.
                    patch.setattr(os, "stat", stat)
                    faults.append(True)
                    raise PermissionError(
                        errno.EACCES, "fixture private archive metadata denied"
                    )
                return stat(name, *args, **kwargs)

            def extract_then_fault(*args, **kwargs):
                extract(*args, **kwargs)
                patch.setattr(os, "stat", denied_stat)

            patch.setattr(rw, "_extract_members", extract_then_fault)
        else:
            commit, rename = rw.StagedTar.commit, os.rename

            def denied_rename(source, *args, **kwargs):
                if source == "payload":
                    patch.setattr(os, "rename", rename)
                    faults.append(True)
                    raise PermissionError(
                        errno.EACCES, "fixture private archive publication denied"
                    )
                return rename(source, *args, **kwargs)

            def commit_with_fault(staged):
                patch.setattr(os, "rename", denied_rename)
                return commit(staged)

            patch.setattr(rw.StagedTar, "commit", commit_with_fault)
        rewards, error, warnings = await _assess(sandbox, verifier)
    assert faults == [True]
    observe(tmp_path, rewards, error, "unscored", "private archive")
    assert not warnings


@pytest.mark.parametrize("writer", ["write", "host-text", "log"])
def test_private_publication_source_read_failure_is_not_shared_evidence(
    tmp_path, monkeypatch, writer
):
    target = rw.pin_destination(tmp_path / "private") / "result"
    original = os.fdopen
    reads = []

    class Unreadable:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, *_):
            reads.append(True)
            raise PermissionError(errno.EACCES, "fixture private duplicate unreadable")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

    def opened(fd, mode="r", *args, **kwargs):
        stream = original(fd, mode, *args, **kwargs)
        return Unreadable(stream) if mode == "rb" else stream

    monkeypatch.setattr(os, "fdopen", opened)
    with pytest.raises(OSError) as caught:
        if writer == "write":
            target.write_text("1")
        elif writer == "host-text":
            rw.write_host_text(target, "1")
        else:
            with target.log() as log:
                log.write("1")
    assert reads
    assert verifier_infrastructure_failure(caught.value)
    assert caught.value.__cause__.errno == errno.EACCES
