"""Recovery publication regression from 7c24f4d88."""

import os
from pathlib import Path

import pytest

from benchflow._utils import reachable_write as rw


def test_recovery_jobs_on_another_filesystem(tmp_path):
    """Guards R12-6 introduced by 7c24f4d88: no-xdev starts at the attempt."""
    import tempfile

    if not Path("/dev/shm").is_dir():
        pytest.skip("requires an independent temporary filesystem")
    if os.stat(tmp_path).st_dev == os.stat("/dev/shm").st_dev:
        pytest.skip("requires two filesystems")
    with tempfile.TemporaryDirectory(prefix="taskmd-r13-", dir="/dev/shm") as temp:
        root = Path(temp)
        attempt = root / "attempt"
        (attempt / "verifier").mkdir(parents=True)
        (attempt / "verifier" / "reward.json").write_text('{"reward":1}')
        (root / "verifier").mkdir()
        (root / "verifier" / "reward.json").write_text('{"reward":0}')
        rw.replace_verifier_root(root, attempt)
        assert (root / "verifier" / "reward.json").read_text() == '{"reward":1}'
        assert (
            attempt / "previous-verifier" / "reward.json"
        ).read_text() == '{"reward":0}'
