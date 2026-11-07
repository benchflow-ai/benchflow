"""Guards FF-sw-launchers: compressed downloads have expanded budgets (363d815b)."""

import base64
import io
import tarfile
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import agentcore as module
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.task.config import SandboxConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "aggregate", "entries", "pax"])
async def test_rejects_expanded_archive_before_publication(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(module, "_MAX_EXTRACT_BYTES", 1024, raising=False)
    monkeypatch.setattr(module, "_MAX_EXTRACT_MEMBERS", 4, raising=False)
    monkeypatch.setattr(module, "_MAX_TAR_BYTES", 8192, raising=False)
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    env = AgentCoreSandbox(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="s",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for i in range(5 if kind == "entries" else 2 if kind == "aggregate" else 1):
            item = tarfile.TarInfo(f"file-{i}")
            item.size = 2048 if kind == "file" else 600 if kind == "aggregate" else 0
            if kind == "pax":
                item.pax_headers = {"comment": "a" * 20000}
            tar.addfile(item, io.BytesIO(b"\0" * item.size))
    env.exec = AsyncMock(
        return_value=ExecResult(
            stdout=base64.b64encode(buffer.getvalue()).decode(),
            stderr=None,
            return_code=0,
        )
    )
    target = tmp_path / "out"
    with pytest.raises(RuntimeError, match=r"(budget|cap|limit)"):
        await env.download_dir("/logs/artifacts", target)
    assert not target.exists() or not list(target.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.GNUTYPE_SPARSE,
        "duplicate",
    ],
)
async def test_archive_only_contains_unique_regular_files_and_directories(
    tmp_path, kind
):
    from benchflow.sandbox._bounded_tar import extract_download

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        member = tarfile.TarInfo("entry")
        if kind != "duplicate":
            member.type = kind
            member.linkname = "other"
        tar.addfile(member)
        if kind == "duplicate":
            tar.addfile(member)
    with pytest.raises(RuntimeError, match="unsafe"):
        extract_download(
            buffer.getvalue(),
            tmp_path / "out",
            tar_bytes=65536,
            content_bytes=1024,
            members=10,
        )
    assert not (tmp_path / "out").exists()
