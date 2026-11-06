"""Guards the PY8 evidence fixes against commit 363d815b."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.taskmd import verify


class LocalBox:
    async def download_file(self, source, target):
        shutil.copyfile(source, target)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


@pytest.fixture
def capture_box(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    monkeypatch.setattr(verify, "COPY_DIR", str(stage))
    monkeypatch.setattr(verify, "COPY_TAR", str(stage / "copy.tar"))
    monkeypatch.setitem(verify.KEPT_CAPS, "bytes", 1024)
    monkeypatch.setattr(verify, "COPY_TAR_LIMIT", 32_768)

    async def execute(box, command, **kwargs):
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    monkeypatch.setattr(verify, "root_exec", execute)
    return LocalBox()


async def test_oversized_single_file_is_bounded_before_download(tmp_path, capture_box):
    """A transfer must never receive the unbounded source file."""
    source = tmp_path / "answer.txt"
    source.write_bytes(b"x" * 1_000_000)
    downloads = []
    original = capture_box.download_file

    async def download(path, target):
        downloads.append(Path(path).stat().st_size)
        await original(path, target)

    capture_box.download_file = download
    refused = await verify.copy_kept(capture_box, [str(source)], tmp_path / "fs")
    assert str(source) in refused
    assert all(size < verify.COPY_TAR_LIMIT for size in downloads)
