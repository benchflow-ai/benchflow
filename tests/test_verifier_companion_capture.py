"""Guards FIX-49 r5's recovery gap inherited from dc509087a/c278f22a6.

Capture must preserve both rewards: unsafe evidence is zero, a lost transport
is unscored, and only a genuinely absent optional companion may be omitted.
"""

import io
import json
import logging
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from benchflow._utils.reachable_write import StagedTar, destination
from tests.test_verifier_own_failures import _assess, _task


@pytest.mark.parametrize("surviving", ["reward.json", "reward.txt"])
@pytest.mark.parametrize(
    "failure",
    ["unreadable", "directory", "stream-directory", "link", "transport", "absent"],
)
async def test_canonical_recovery_never_accepts_a_higher_partial_reward(
    tmp_path, surviving, failure
):
    sandbox, verifier = _task(tmp_path)
    sandbox.is_mounted = False
    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk export unavailable")
    )
    companion = "reward.txt" if surviving == "reward.json" else "reward.json"

    async def download(source, target):
        name = source.rsplit("/", 1)[-1]
        if name == surviving:
            target.write_text('{"reward":1}' if name.endswith("json") else "1")
        elif name == companion:
            if failure == "unreadable":
                raise PermissionError("existing companion holding zero is unreadable")
            if failure == "directory":
                target.mkdir()
            elif failure == "stream-directory":
                payload = io.BytesIO()
                with tarfile.open(fileobj=payload, mode="w") as archive:
                    folder = tarfile.TarInfo(name)
                    folder.type = tarfile.DIRTYPE
                    archive.addfile(folder)
                payload.seek(0)
                with StagedTar(destination(target), directory=False) as staged:
                    staged.extract(payload)
                    staged.commit()
            elif failure == "link":
                target.symlink_to(tmp_path / "absent-link-target")
            elif failure == "transport":
                raise httpx.ConnectError("fixture companion transport disconnected")
            else:
                raise FileNotFoundError(source)
        else:
            raise FileNotFoundError(source)

    sandbox.download_file = AsyncMock(side_effect=download)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert any(
        call.args[0].endswith(companion)
        for call in sandbox.download_file.await_args_list
    )
    if failure == "transport":
        assert rewards is None and "fixture companion transport" in error
        assert "DownloadVerifierDirError" in error and not warnings
    elif failure == "absent":
        assert rewards == {"reward": 1.0} and error is None and not warnings
    else:
        assert rewards == {"reward": 0.0} and error is None
        assert any(companion in warning for warning in warnings)


class _Files:
    def __init__(self, directory, *, search_lists_directory):
        self.directory = directory
        self.search_lists_directory = search_lists_directory
        self.calls = []
        self.info_calls = []
        self.batch_calls = 0

    async def search_files(self, source, pattern):
        names = ["reward.txt"]
        if self.directory and self.search_lists_directory:
            names.append("reward.json")
        return SimpleNamespace(files=[f"{source}/{name}" for name in names])

    async def list_files(self, source):
        return (
            [SimpleNamespace(name="reward.json", is_dir=True)] if self.directory else []
        )

    async def get_file_info(self, source):
        self.info_calls.append(source)
        name = source.rsplit("/", 1)[-1]
        if name == "reward.txt":
            return SimpleNamespace(is_dir=False)
        if name == "reward.json" and self.directory:
            return SimpleNamespace(is_dir=True)
        raise FileNotFoundError(source)

    async def download_file(self, source, destination):
        self.calls.append(source)
        if source.endswith("/reward.txt"):
            Path(destination).write_text("1")
        else:
            # The old backend lets the SDK decide directory errors. Recovery
            # used to discard this and accept the companion's passing value.
            raise IsADirectoryError(source)

    async def download_files(self, *, files):
        self.batch_calls += 1
        responses = []
        for request in files:
            await self.download_file(request.source, request.destination)
            responses.append(
                SimpleNamespace(
                    source=request.source, result=request.destination, error=None
                )
            )
        return responses


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("search_lists_directory", [False, True])
@pytest.mark.parametrize("directory", [False, True])
async def test_daytona_capture_preserves_unsafe_or_absent_companions(
    tmp_path, monkeypatch, recovery, search_lists_directory, directory
):
    import benchflow.sandbox.daytona as daytona

    fs = _Files(directory, search_lists_directory=search_lists_directory)
    backend = object.__new__(daytona.DaytonaSandbox)
    backend._sandbox = SimpleNamespace(fs=fs)
    backend.logger = logging.getLogger("fixture.daytona.capture")
    monkeypatch.setattr(daytona, "DaytonaNotFoundError", FileNotFoundError)
    monkeypatch.setattr(daytona, "FileDownloadRequest", SimpleNamespace)
    sandbox, verifier = _task(tmp_path)
    sandbox.is_mounted = False

    async def download_dir(*, source_dir, target_dir, **_):
        await backend._sdk_download_dir(source_dir, target_dir)

    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk export unavailable")
        if recovery
        else download_dir
    )
    sandbox.download_file = backend._sdk_download_file
    rewards, error, warnings = await _assess(sandbox, verifier)
    # Persist/parse the observable result exactly as downstream readers see it.
    saved = json.loads(
        json.dumps({"rewards": rewards, "verifier_error": error, "warnings": warnings})
    )
    assert saved["rewards"] == {"reward": 0.0 if directory else 1.0}
    assert saved["verifier_error"] is None
    if directory:
        assert any("reward.json" in warning for warning in warnings)
    else:
        assert not warnings
    if not recovery:
        assert fs.batch_calls == 1
        assert "/logs/verifier/reward.txt" in fs.calls
    elif directory:
        assert "/logs/verifier/reward.json" in fs.info_calls
    else:
        assert "/logs/verifier/reward.txt" in fs.calls


async def test_daytona_not_found_without_a_parent_inventory_is_not_absence(
    tmp_path, monkeypatch
):
    """FIX-49 r5: a lost backend must not masquerade as an absent companion."""
    import benchflow.sandbox.daytona as daytona

    fs = _Files(False, search_lists_directory=False)
    fs.list_files = AsyncMock(
        side_effect=httpx.ConnectError("fixture parent inventory unavailable")
    )
    backend = object.__new__(daytona.DaytonaSandbox)
    backend._sandbox = SimpleNamespace(fs=fs)
    backend.logger = logging.getLogger("fixture.daytona.capture")
    monkeypatch.setattr(daytona, "DaytonaNotFoundError", FileNotFoundError)
    sandbox, verifier = _task(tmp_path)
    sandbox.is_mounted = False
    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk export unavailable")
    )
    sandbox.download_file = backend._sdk_download_file
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert rewards is None and "parent inventory unavailable" in error and not warnings
    fs.list_files.assert_awaited_once_with("/logs/verifier")
