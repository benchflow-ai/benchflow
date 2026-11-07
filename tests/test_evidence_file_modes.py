"""A frozen workspace must come back with the permissions the solver left.

Guards the frozen-workspace restore used by ``bench eval regrade`` and
verifier recovery. Capture
normalised every file to 0644/0755 and recorded no mode, so a solver that
wrote a private key with ``chmod 600`` inside the workspace passed its
verifier, and a regrade with the unchanged verifier then failed it
("Key permissions too open: 644") and reported ``pass->fail``. The reviewer's
copy of the evidence keeps the normalised modes.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import stat
import sys
from pathlib import Path

import pytest

from benchflow.review.evidence import (
    EvidenceManifest,
    capture_workspace,
    install_uploaded_workspace,
    prepare_workspace_upload,
)
from benchflow.sandbox.protocol import ExecResult


class LocalTransport:
    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        shutil.copyfile(src, dst)


async def _frozen(tmp_path: Path) -> Path:
    workspace = tmp_path / "app"
    (workspace / "ssl").mkdir(parents=True)
    key = workspace / "ssl" / "server.key"
    key.write_text("private\n")
    key.chmod(0o600)
    (workspace / "ssl").chmod(0o700)
    (workspace / "run.sh").write_text("#!/bin/sh\n")
    (workspace / "run.sh").chmod(0o750)
    bundle = tmp_path / "evidence"
    await capture_workspace(LocalTransport(), str(workspace), bundle)
    return bundle


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.asyncio
async def test_manifest_records_the_solvers_permissions(tmp_path: Path):
    bundle = await _frozen(tmp_path)
    manifest = EvidenceManifest.model_validate_json(
        (bundle / "manifest.json").read_text()
    )
    modes = {entry.path: entry.mode for entry in manifest.entries}
    assert modes["ssl/server.key"] == 0o600
    assert modes["ssl"] == 0o700
    assert modes["run.sh"] == 0o750


@pytest.mark.asyncio
async def test_a_restore_for_verification_keeps_the_permissions(tmp_path: Path):
    bundle = await _frozen(tmp_path)
    archive = tmp_path / "upload" / "workspace.tar"
    prepare_workspace_upload(bundle, archive)
    manifest = tmp_path / "upload" / "workspace.json"
    manifest.write_text((bundle / "manifest.json").read_text())

    restored = tmp_path / "restored"
    await install_uploaded_workspace(
        LocalTransport(),
        str(archive),
        str(restored),
        str(manifest),
        restore_modes=True,
    )
    assert _mode(restored / "ssl" / "server.key") == 0o600
    assert _mode(restored / "ssl") == 0o700
    assert _mode(restored / "run.sh") == 0o750

    reviewer_copy = tmp_path / "reviewer"
    await install_uploaded_workspace(
        LocalTransport(), str(archive), str(reviewer_copy), str(manifest)
    )
    assert _mode(reviewer_copy / "ssl" / "server.key") == 0o644


def test_an_older_manifest_without_modes_still_loads():
    old = {
        "workspace": "/app",
        "archive_sha256": "0" * 64,
        "entries": [
            {
                "path": "a.txt",
                "original_path": "/app/a.txt",
                "kind": "file",
                "size": 1,
                "sha256": "0" * 64,
            }
        ],
    }
    manifest = EvidenceManifest.model_validate_json(json.dumps(old))
    assert manifest.entries[0].mode is None


@pytest.mark.asyncio
async def test_regrade_restores_the_workspace_with_its_modes(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from benchflow import eval_regrade

    bundle = await _frozen(tmp_path)
    install = AsyncMock()
    monkeypatch.setattr("benchflow.review.evidence.install_uploaded_workspace", install)
    env = AsyncMock()
    staging = tmp_path / "staging"
    staging.mkdir()

    await eval_regrade._install_bundle(env, bundle, staging, "workspace")

    assert install.await_args.kwargs.get("restore_modes") is True
