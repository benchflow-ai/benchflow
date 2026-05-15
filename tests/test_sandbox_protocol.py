"""Tests for benchflow.sandbox — protocol types and Harbor adapters.

Validates ENG-48 Phase A: parallel Sandbox protocol alongside existing Harbor types.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult, ImageConfig, ImageRef, Sandbox

# ── Dataclass tests ───────────────────────────────────────────────────────────


class TestExecResult:
    def test_fields(self):
        r = ExecResult(return_code=0, stdout="ok", stderr="")
        assert r.return_code == 0
        assert r.stdout == "ok"
        assert r.stderr == ""

    def test_frozen(self):
        r = ExecResult(return_code=1, stdout="", stderr="err")
        with pytest.raises(AttributeError):
            r.return_code = 2  # type: ignore[misc]


class TestImageRef:
    def test_defaults(self):
        ref = ImageRef(tag="latest")
        assert ref.tag == "latest"
        assert ref.digest is None

    def test_with_digest(self):
        ref = ImageRef(tag="v1", digest="sha256:abc")
        assert ref.digest == "sha256:abc"

    def test_frozen(self):
        ref = ImageRef(tag="v1")
        with pytest.raises(AttributeError):
            ref.tag = "v2"  # type: ignore[misc]


class TestImageConfig:
    def test_required_fields(self):
        cfg = ImageConfig(dockerfile=Path("Dockerfile"), context_dir=Path("."))
        assert cfg.dockerfile == Path("Dockerfile")
        assert cfg.context_dir == Path(".")
        assert cfg.build_args is None
        assert cfg.cache_key is None

    def test_optional_fields(self):
        cfg = ImageConfig(
            dockerfile=Path("Dockerfile"),
            context_dir=Path("."),
            build_args={"FOO": "bar"},
            cache_key="my-key",
        )
        assert cfg.build_args == {"FOO": "bar"}
        assert cfg.cache_key == "my-key"

    def test_mutable(self):
        cfg = ImageConfig(dockerfile=Path("Dockerfile"), context_dir=Path("."))
        cfg.cache_key = "new-key"
        assert cfg.cache_key == "new-key"


# ── Protocol conformance ──────────────────────────────────────────────────────


def _make_harbor_mock():
    """Build a mock that looks like a Harbor BaseEnvironment subclass."""
    mock = AsyncMock()
    mock.exec = AsyncMock(
        return_value=MagicMock(return_code=0, stdout="hello", stderr="")
    )
    mock.upload_file = AsyncMock()
    mock.upload_dir = AsyncMock()
    mock.download_file = AsyncMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    return mock


class TestDockerSandboxProtocol:
    def test_isinstance_check(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        assert isinstance(adapter, Sandbox)

    @pytest.mark.asyncio
    async def test_exec_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        result = await adapter.exec("echo hi", user="root", timeout_sec=10)
        mock.exec.assert_awaited_once_with("echo hi", user="root", timeout_sec=10)
        assert isinstance(result, ExecResult)
        assert result.return_code == 0
        assert result.stdout == "hello"

    @pytest.mark.asyncio
    async def test_read_file_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        data = await adapter.read_file("/tmp/test.txt")
        mock.exec.assert_awaited_once_with("cat /tmp/test.txt", timeout_sec=30)
        assert isinstance(data, bytes)

    @pytest.mark.asyncio
    async def test_upload_file_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        await adapter.upload_file(Path("/local/file"), "/remote/file")
        mock.upload_file.assert_awaited_once_with(Path("/local/file"), "/remote/file")

    @pytest.mark.asyncio
    async def test_upload_dir_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        await adapter.upload_dir(Path("/local/dir"), "/remote/dir")
        mock.upload_dir.assert_awaited_once_with(Path("/local/dir"), "/remote/dir")

    @pytest.mark.asyncio
    async def test_download_file_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        await adapter.download_file("/remote/file", Path("/local/file"))
        mock.download_file.assert_awaited_once_with("/remote/file", Path("/local/file"))

    @pytest.mark.asyncio
    async def test_start_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        await adapter.start()
        mock.start.assert_awaited_once_with(force_build=False)

    @pytest.mark.asyncio
    async def test_stop_delegates(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        await adapter.stop(delete=False)
        mock.stop.assert_awaited_once_with(delete=False)

    def test_host_property(self):
        mock = _make_harbor_mock()
        adapter = DockerSandbox(mock)
        assert adapter.host == "localhost"


class TestDaytonaSandboxProtocol:
    def test_isinstance_check(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        assert isinstance(adapter, Sandbox)

    @pytest.mark.asyncio
    async def test_exec_delegates(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        result = await adapter.exec("ls -la", user="agent", timeout_sec=60)
        mock.exec.assert_awaited_once_with("ls -la", user="agent", timeout_sec=60)
        assert isinstance(result, ExecResult)
        assert result.return_code == 0

    @pytest.mark.asyncio
    async def test_read_file_delegates(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        data = await adapter.read_file("/etc/hostname")
        mock.exec.assert_awaited_once_with("cat /etc/hostname", timeout_sec=30)
        assert isinstance(data, bytes)

    @pytest.mark.asyncio
    async def test_upload_file_delegates(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        await adapter.upload_file(Path("/local/file"), "/remote/file")
        mock.upload_file.assert_awaited_once_with(Path("/local/file"), "/remote/file")

    @pytest.mark.asyncio
    async def test_start_delegates(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        await adapter.start()
        mock.start.assert_awaited_once_with(force_build=False)

    @pytest.mark.asyncio
    async def test_stop_delegates(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        await adapter.stop()
        mock.stop.assert_awaited_once_with(delete=True)

    def test_host_property(self):
        mock = _make_harbor_mock()
        adapter = DaytonaSandbox(mock)
        assert adapter.host == "localhost"


# ── ExecResult conversion: None → empty string ───────────────────────────────


class TestNoneToEmptyString:
    @pytest.mark.asyncio
    async def test_docker_none_stdout(self):
        mock = _make_harbor_mock()
        mock.exec.return_value = MagicMock(return_code=0, stdout=None, stderr=None)
        adapter = DockerSandbox(mock)
        result = await adapter.exec("true")
        assert result.stdout == ""
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_daytona_none_stdout(self):
        mock = _make_harbor_mock()
        mock.exec.return_value = MagicMock(return_code=0, stdout=None, stderr=None)
        adapter = DaytonaSandbox(mock)
        result = await adapter.exec("true")
        assert result.stdout == ""
        assert result.stderr == ""
