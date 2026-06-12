from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow.sandbox.daytona as daytona_mod
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD


@dataclass
class _FakeFileDownloadRequest:
    source: str
    destination: str


class _FakeFs:
    def __init__(self) -> None:
        self.download_files_calls = 0
        self.download_file_calls: list[tuple[str, str]] = []

    async def search_files(self, source_dir: str, pattern: str):
        assert source_dir == "/logs/verifier"
        assert pattern == "*"
        return SimpleNamespace(
            files=[
                "/logs/verifier/reward.txt",
                "/logs/verifier/ctrf.json",
            ]
        )

    async def get_file_info(self, file_path: str):
        return SimpleNamespace(is_dir=False)

    async def download_files(self, *, files):
        self.download_files_calls += 1
        raise RuntimeError("provider batch export failed")

    async def download_file(self, source: str, destination: str):
        self.download_file_calls.append((source, destination))
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.rsplit("/", 1)[-1])


class _FakeMetadataErrorFs(_FakeFs):
    async def get_file_info(self, file_path: str):
        raise ValueError("modified_at=None")

    async def download_files(self, *, files):
        self.download_files_calls += 1
        for request in files:
            path = Path(request.destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(request.source.rsplit("/", 1)[-1])


class _FakeMissingFs(_FakeFs):
    async def download_file(self, source: str, destination: str):
        raise FileNotFoundError(source)


def _fake_daytona_sandbox(fake_fs):
    sandbox = object.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(fs=fake_fs)
    sandbox.logger = logging.getLogger("test.daytona")
    return sandbox


@pytest.mark.asyncio
async def test_daytona_download_dir_falls_back_to_individual_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards SkillsBench paratransit-routing Daytona verifier export fallback."""
    fake_fs = _FakeFs()
    sandbox = _fake_daytona_sandbox(fake_fs)
    monkeypatch.setattr(daytona_mod, "FileDownloadRequest", _FakeFileDownloadRequest)
    monkeypatch.setattr(daytona_mod, "DaytonaNotFoundError", FileNotFoundError)

    await sandbox._sdk_download_dir("/logs/verifier", tmp_path)

    assert fake_fs.download_files_calls == 1
    assert fake_fs.download_file_calls == [
        ("/logs/verifier/reward.txt", str(tmp_path / "reward.txt")),
        ("/logs/verifier/ctrf.json", str(tmp_path / "ctrf.json")),
    ]
    assert (tmp_path / "reward.txt").read_text() == "reward.txt"
    assert (tmp_path / "ctrf.json").read_text() == "ctrf.json"


@pytest.mark.asyncio
async def test_daytona_download_dir_tolerates_broken_file_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards private PR #1's Daytona FileInfo.modified_at=None regression."""
    fake_fs = _FakeMetadataErrorFs()
    sandbox = _fake_daytona_sandbox(fake_fs)
    monkeypatch.setattr(daytona_mod, "FileDownloadRequest", _FakeFileDownloadRequest)
    monkeypatch.setattr(daytona_mod, "DaytonaNotFoundError", FileNotFoundError)

    await sandbox._sdk_download_dir("/logs/verifier", tmp_path)

    assert fake_fs.download_files_calls == 1
    assert (tmp_path / "reward.txt").read_text() == "reward.txt"
    assert (tmp_path / "ctrf.json").read_text() == "ctrf.json"


@pytest.mark.asyncio
async def test_daytona_download_dir_fails_when_individual_fallback_recovers_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards private PR #1's Daytona listed-but-undownloadable export case."""
    fake_fs = _FakeMissingFs()
    sandbox = _fake_daytona_sandbox(fake_fs)
    monkeypatch.setattr(daytona_mod, "FileDownloadRequest", _FakeFileDownloadRequest)
    monkeypatch.setattr(daytona_mod, "DaytonaNotFoundError", FileNotFoundError)

    with pytest.raises(RuntimeError, match="recovered no files"):
        await sandbox._sdk_download_dir("/logs/verifier", tmp_path)


def _fake_dind_strategy():
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    strategy._env = SimpleNamespace(
        logger=logging.getLogger("test.daytona.dind"),
        _sdk_download_dir=AsyncMock(),
        _sdk_download_file=AsyncMock(),
    )
    strategy._vm_exec = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    return strategy


@pytest.mark.asyncio
async def test_dind_log_download_dir_falls_back_to_compose_when_host_path_fails(
    tmp_path: Path,
) -> None:
    """Guards private PR #1's Daytona DinD verifier log export fallback."""
    strategy = _fake_dind_strategy()
    sdk_calls: list[tuple[str, Path | str]] = []
    compose_calls: list[list[str]] = []

    async def sdk_download_dir(source: str, target: Path | str) -> None:
        sdk_calls.append((source, target))
        if source == "/benchflow/logs/verifier":
            raise RuntimeError("host log export failed")
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "reward.txt").write_text("0")

    async def compose_exec(subcommand, timeout_sec=None):
        compose_calls.append(subcommand)
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._env._sdk_download_dir = AsyncMock(side_effect=sdk_download_dir)
    strategy._compose_exec = compose_exec  # type: ignore[method-assign]

    await strategy.download_dir("/logs/verifier", tmp_path, service="main")

    assert sdk_calls[0] == ("/benchflow/logs/verifier", tmp_path)
    assert sdk_calls[1][0].startswith("/tmp/benchflow_")
    assert sdk_calls[1][1] == tmp_path
    assert compose_calls == [["cp", "main:/logs/verifier/.", sdk_calls[1][0]]]
    assert (tmp_path / "reward.txt").read_text() == "0"


@pytest.mark.asyncio
async def test_dind_log_download_file_falls_back_to_compose_when_host_path_fails(
    tmp_path: Path,
) -> None:
    """Guards private PR #1's Daytona DinD verifier file recovery fallback."""
    strategy = _fake_dind_strategy()
    sdk_calls: list[tuple[str, Path | str]] = []
    compose_calls: list[list[str]] = []

    async def sdk_download_file(source: str, target: Path | str) -> None:
        sdk_calls.append((source, target))
        if source == "/benchflow/logs/verifier/reward.txt":
            raise RuntimeError("host log file export failed")
        Path(target).write_text("0")

    async def compose_exec(subcommand, timeout_sec=None):
        compose_calls.append(subcommand)
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._env._sdk_download_file = AsyncMock(side_effect=sdk_download_file)
    strategy._compose_exec = compose_exec  # type: ignore[method-assign]

    target = tmp_path / "reward.txt"
    await strategy.download_file("/logs/verifier/reward.txt", target)

    assert sdk_calls[0] == ("/benchflow/logs/verifier/reward.txt", target)
    assert sdk_calls[1][0].startswith("/tmp/benchflow_")
    assert sdk_calls[1][1] == target
    assert compose_calls == [["cp", "main:/logs/verifier/reward.txt", sdk_calls[1][0]]]
    assert target.read_text() == "0"
