"""Guards FF-sw-daytona batch-result handling at 363d815b."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import benchflow.sandbox.daytona as daytona_mod
from tests.test_daytona_download import _fake_daytona_sandbox, _FakeFs


@pytest.fixture(autouse=True)
def _daytona_sdk_loaded():
    """A real DaytonaSandbox loads the optional Daytona SDK when it is built
    (``_load_daytona_sdk`` in ``__init__``), before any download. The fake
    skips ``__init__``, so load it here, before each test body: run alone,
    ``FileDownloadRequest`` was still None, and a first load inside the test
    would replace a name the test patched (``DaytonaNotFoundError``)."""
    daytona_mod._load_daytona_sdk()


@pytest.mark.parametrize(
    "response", ["error", "missing", "empty", "none", "missing-file"]
)
@pytest.mark.parametrize("retry_fails", [False, True])
async def test_batch_requires_every_file(tmp_path, response, retry_fails):
    """Guard 363d815b: partial SDK success is never a complete directory."""

    class Fs(_FakeFs):
        async def download_files(self, *, files):
            results = []
            for index, request in enumerate(files):
                Path(request.destination).write_text("partial" if index else "complete")
                if index == 1 and response == "missing-file":
                    Path(request.destination).unlink()
                if index == 1 and response == "missing":
                    continue
                results.append(
                    SimpleNamespace(
                        source=request.source,
                        result=request.destination,
                        error="dummy error" if index and response == "error" else None,
                    )
                )
            return (
                None if response == "none" else [] if response == "empty" else results
            )

        async def download_file(self, source, destination):
            assert not Path(destination).exists(), "partial destination must be removed"
            if retry_fails:
                raise FileNotFoundError("dummy missing file")
            await super().download_file(source, destination)

    fs = Fs()
    sandbox = _fake_daytona_sandbox(fs)
    if retry_fails:
        with pytest.raises(
            RuntimeError,
            match=r"Daytona individual download_dir failed \(0 recovered; recovered no files",
        ):
            await sandbox._sdk_download_dir("/logs/verifier", tmp_path)
    else:
        await sandbox._sdk_download_dir("/logs/verifier", tmp_path)
        assert fs.download_file_calls
        assert (tmp_path / "ctrf.json").read_text() == "ctrf.json"


async def test_complete_batch_does_not_retry(tmp_path):
    """Guard FF-sw-daytona: complete SDK responses remain a single batch."""

    class Fs(_FakeFs):
        async def download_files(self, *, files):
            results = []
            for request in files:
                Path(request.destination).write_text("complete")
                results.append(
                    SimpleNamespace(
                        source=request.source, result=request.destination, error=None
                    )
                )
            return results

    fs = Fs()
    await _fake_daytona_sandbox(fs)._sdk_download_dir("/logs/verifier", tmp_path)
    assert not fs.download_file_calls
    assert (tmp_path / "ctrf.json").read_text() == "complete"


async def test_metadata_disappearance_is_not_silent_success(tmp_path, monkeypatch):
    """Guard FF-sw-daytona: a listed file remains required during metadata lookup."""
    import benchflow.sandbox.daytona as daytona

    monkeypatch.setattr(daytona, "DaytonaNotFoundError", FileNotFoundError)

    class Fs(_FakeFs):
        async def get_file_info(self, path):
            if path.endswith("ctrf.json"):
                raise FileNotFoundError("dummy disappeared")
            return SimpleNamespace(is_dir=False)

    with pytest.raises(
        RuntimeError, match=r"Daytona download_dir lost a listed file before transfer"
    ):
        await _fake_daytona_sandbox(Fs())._sdk_download_dir("/logs/verifier", tmp_path)


def test_each_test_here_passes_in_a_fresh_interpreter():
    """These tests ran green only after another module had loaded the
    Daytona SDK. A fresh interpreter runs the module, and the test that
    patches an SDK name runs first and alone."""
    import subprocess
    import sys

    here = Path(__file__).resolve()
    module = str(here.relative_to(here.parents[1]))
    patched = f"{module}::test_metadata_disappearance_is_not_silent_success"
    for args in (
        [patched],
        [module, "-k", "not test_each_test_here_passes_in_a_fresh_interpreter"],
    ):
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
            cwd=here.parents[1],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, (args, done.stdout[-3000:], done.stderr[-2000:])
