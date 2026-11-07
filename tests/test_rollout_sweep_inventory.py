"""Guards PY8 FF-sw-rollout fixes against the 363d815b artifact regressions."""

import asyncio
import os
import threading
from unittest.mock import patch

import pytest

from benchflow.rollout import _artifacts
from tests.test_artifacts_collection import LocalTransport, _collect, _layout


@pytest.mark.parametrize("kind", ["bytes", "entries"])
async def test_mounted_limits_precede_hashing(tmp_path, kind):
    logs, workspace, trial = _layout(tmp_path)
    root = trial / "artifacts"
    if kind == "bytes":
        with (root / "sparse").open("wb") as stream:
            stream.truncate(1024**4)
    else:
        for index in range(4):
            (root / str(index)).write_text("x")
    with patch.object(
        _artifacts, "_sha256", side_effect=AssertionError("hashed before admission")
    ):
        result = await _collect(
            LocalTransport(),
            logs,
            workspace,
            trial,
            mounted=True,
            max_bytes=10,
            max_files=2,
        )
    assert result["collections"][0]["status"] == "over_limit"


async def test_mounted_inventory_runs_off_event_loop(tmp_path, monkeypatch):
    logs, workspace, trial = _layout(tmp_path)
    original = _artifacts._scan_local
    controller = threading.get_ident()

    def scan(*args, **kwargs):
        assert threading.get_ident() != controller, "inventory blocked the event loop"
        return original(*args, **kwargs)

    monkeypatch.setattr(_artifacts, "_scan_local", scan)
    await _collect(LocalTransport(), logs, workspace, trial, mounted=True)


def test_hash_read_is_bounded_even_when_file_grows(tmp_path):
    """Guards 363d815b by enforcing the cap on actual reads, as well as stat."""
    path = tmp_path / "growing"
    path.write_bytes(b"x" * 100)
    with pytest.raises(_artifacts.EvidenceLimitError):
        _artifacts._sha256(path, 10)


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
async def test_inventory_never_opens_special_files_or_link_targets(tmp_path, kind):
    """Guards 363d815b across file-type replacements and unreadable targets."""
    logs, workspace, trial = _layout(tmp_path)
    path = trial / "artifacts" / "entry"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to(tmp_path / "absent")
    result = await asyncio.wait_for(
        _collect(LocalTransport(), logs, workspace, trial, mounted=True), 5
    )
    assert result["total_bytes"] == 0
