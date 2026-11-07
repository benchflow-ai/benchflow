"""Guards capture fixes for the integrity sweep at commit 363d815b."""

from __future__ import annotations

import ast
import asyncio
import io
import os
import shlex
import shutil
import sys
import tarfile

import pytest

from benchflow.review.evidence import (
    EvidenceLimitError,
    _normalize_tar_capture,
    capture_task_evidence,
    capture_workspace,
    install_uploaded_workspace,
    prepare_workspace_upload,
    validate_uploaded_workspace,
)
from benchflow.review.evidence_runtime import ensure_capture_tools
from benchflow.sandbox.protocol import ExecResult


@pytest.mark.parametrize("hardlink", [False, True])
@pytest.mark.parametrize("cap", [128, 256 * 1024])
def test_tar_capture_streams_and_checks_resolved_size_first(
    tmp_path, monkeypatch, hardlink, cap
):
    """Guards the fix at 363d815b, including hard links."""
    raw = tmp_path / "raw.tar"
    payload = b"x" * (128 * 1024)
    with tarfile.open(raw, "w") as archive:
        member = tarfile.TarInfo("answer")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
        if hardlink:
            member = tarfile.TarInfo("alias")
            member.type, member.linkname = tarfile.LNKTYPE, "answer"
            archive.addfile(member)

    original = tarfile.ExFileObject.read
    reads = []

    def bounded_read(stream, size=-1):
        assert 0 <= size <= 16 * 1024, "capture allocated a whole artifact"
        reads.append(size)
        return original(stream, size)

    monkeypatch.setattr(tarfile.ExFileObject, "read", bounded_read)
    rules = {"exclude": [], "credentials": [], "paths": [], "sandbox_runtime": []}
    args = (raw, tmp_path / "normalized.tar", "/app", rules)
    if cap < len(payload) or (hardlink and cap < 2 * len(payload)):
        with pytest.raises(EvidenceLimitError):
            _normalize_tar_capture(*args, max_bytes=cap, max_entries=10)
        assert not reads
    else:
        _normalize_tar_capture(*args, max_bytes=cap, max_entries=10)
        with tarfile.open(args[1]) as archive:
            assert archive.getmember("answer").size == len(payload)
            if hardlink:
                alias = archive.getmember("alias")
                assert alias.isfile() and alias.size == len(payload)


class CwdTransport:
    def __init__(self, cwd):
        self.cwd = cwd

    async def exec(self, command, *, user="root", timeout_sec=30):
        argv = shlex.split(command)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            env={**os.environ, "PYTHONPATH": str(self.cwd)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, source, destination):
        shutil.copyfile(source, destination)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["capture", "admission", "unpack", "artifact"])
@pytest.mark.parametrize("module", ["json", "pathlib", "tarfile"])
async def test_privileged_evidence_helpers_ignore_workspace_imports(
    tmp_path, operation, module
):
    """Guards the fix at 363d815b across all helpers."""
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "answer").write_text("dummy answer")
    clean = tmp_path / "clean"
    clean.mkdir()
    bundle = tmp_path / "bundle"
    await capture_workspace(CwdTransport(clean), str(source), bundle)
    archive = tmp_path / "upload.tar"
    prepare_workspace_upload(bundle, archive)
    marker = tmp_path / "unexpected-import"
    (source / f"{module}.py").write_text(
        f"open({str(marker)!r}, 'w').write('untrusted code ran')\n"
        "raise RuntimeError('workspace module executed')\n"
    )
    transport = CwdTransport(source)
    if operation == "capture":
        await capture_workspace(transport, str(source), tmp_path / "captured")
    elif operation == "admission":
        await validate_uploaded_workspace(
            transport, str(bundle / "workspace"), str(bundle / "manifest.json")
        )
    elif operation == "unpack":
        await install_uploaded_workspace(
            transport,
            str(archive),
            str(tmp_path / "restored"),
            str(bundle / "manifest.json"),
        )
    else:
        await capture_task_evidence(
            transport, str(source), tmp_path / "captured", artifacts=["answer"]
        )
    assert not marker.exists()


@pytest.mark.asyncio
async def test_capture_and_preflight_choose_tar_for_older_python(tmp_path):
    """Guards the fix at 363d815b with old Python semantics."""
    from tests.test_review_evidence_runtime import ShellOnlyTransport

    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    for program in (
        "sh",
        "tar",
        "mktemp",
        "rm",
        "sha256sum",
        "cut",
        "readlink",
        "dirname",
        "basename",
    ):
        (binary_dir / program).symlink_to(shutil.which(program))
    python = binary_dir / "python3"
    prefix = (
        "import sys, pathlib; sys.version_info = (3, 8, 20); "
        "delattr(pathlib.PurePath, 'is_relative_to'); "
    )
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "index = sys.argv.index('-c') + 1\n"
        f"code = {prefix!r} + sys.argv[index]\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, '-I', '-c', code, *sys.argv[index + 1:]])\n"
    )
    python.chmod(0o755)
    transport = ShellOnlyTransport(binary_dir)
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "answer").write_text("dummy answer")
    await ensure_capture_tools(transport)
    manifest = await capture_workspace(transport, str(source), tmp_path / "captured")
    assert [entry.path for entry in manifest.entries] == ["answer"]


class _StopAtFirstExec(Exception):
    pass


class _RecordingTransport:
    def __init__(self):
        self.commands = []

    async def exec(self, command, *, user="root", timeout_sec=30):
        self.commands.append(command)
        raise _StopAtFirstExec


@pytest.mark.asyncio
async def test_capture_helper_compiles_only_after_the_version_probe(tmp_path):
    """Guards a4d88434's incomplete fix: the probe ran inside the same
    -c program as the helper, and Python compiles a whole program before it
    runs any of it. The capture helper's walrus is a SyntaxError on Python
    3.7 and older (exit 1, not the probe's 127), so capture never fell back to
    tar there. The program a root exec runs must parse as Python 3.7."""
    transport = _RecordingTransport()
    with pytest.raises(_StopAtFirstExec):
        await capture_workspace(transport, "/app", tmp_path / "captured")
    argv = shlex.split(transport.commands[0])
    program = argv[argv.index("-c") + 1]
    ast.parse(program, feature_version=(3, 7))
