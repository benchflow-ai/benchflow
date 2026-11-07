"""Guards the fix at integration commit 363d815b."""

import subprocess
import sys

import pytest

from benchflow.sandbox._cache_reclaim import RECLAIM_CACHES_PY


@pytest.mark.parametrize(
    "name", ["uv-results.json", ".uv-results.json", "uv-build123", ".uv-build123"]
)
@pytest.mark.parametrize("directory", [False, True])
def test_tmp_names_never_identify_caches(tmp_path, name, directory):
    """Guards 363d815b: task outputs outside WORKDIR keep their bytes."""
    (tmp_path / "app").mkdir()
    (tmp_path / "tmp").mkdir()
    output = tmp_path / "tmp" / name
    if directory:
        output.mkdir()
        output = output / "answer"
    output.write_text("correct answer")
    subprocess.run(
        [sys.executable, "-c", RECLAIM_CACHES_PY, str(tmp_path / "app"), str(tmp_path)],
        check=True,
    )
    assert output.exists() and output.read_text() == "correct answer"
