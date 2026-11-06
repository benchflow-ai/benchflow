"""V3 records use integration's held verifier-folder publication API."""

import json
import os
from types import SimpleNamespace

import pytest

from benchflow._utils.trusted_path import TrustedPathReport
from benchflow.rollout._setup import (
    _TEST_STDOUT_SCAN_BYTES,
    _verifier_missed_a_program,
    _write_trusted_path_record,
)
from benchflow.task import RolloutPaths
from tests.reachable_fixture import on_disk


@pytest.mark.parametrize("existing", ["absent", "file", "symlink", "hardlink"])
def test_trusted_path_record_publishes_through_the_verifier_handle(tmp_path, existing):
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    marker = tmp_path / "dummy-marker"
    marker.write_text("unchanged")
    record = on_disk(paths.verifier_dir) / "trusted-path.json"
    if existing == "file":
        record.write_text("old")
    elif existing == "symlink":
        record.symlink_to(marker)
    elif existing == "hardlink":
        os.link(marker, record)
    report = TrustedPathReport(
        path="/usr/bin:/bin", dropped=(("/app/bin", "workspace"),)
    )
    _write_trusted_path_record(paths, report, True, [("check", "/app/bin")])
    assert marker.read_text() == "unchanged"
    assert not record.is_symlink()
    assert json.loads(record.read_text())["refused_programs_run"] == [
        {"program": "check", "folder": "/app/bin"}
    ]
    assert record.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("capability", [False, True])
@pytest.mark.parametrize(
    "kind", ["missing", "file", "symlink", "directory", "old-prefix", "new-suffix"]
)
def test_missing_program_scan_uses_bounded_regular_file_handles(
    tmp_path, capability, kind
):
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    path = on_disk(paths.test_stdout_path)
    if kind == "file":
        path.write_text("check: command not found\n")
    elif kind == "symlink":
        marker = tmp_path / "dummy-marker"
        marker.write_text("check: command not found\n")
        path.symlink_to(marker)
    elif kind == "directory":
        path.mkdir()
    elif kind == "old-prefix":
        path.write_text("command not found\n" + "x" * _TEST_STDOUT_SCAN_BYTES)
    elif kind == "new-suffix":
        path.write_text("x" * _TEST_STDOUT_SCAN_BYTES + "\ncommand not found\n")
    selected = paths if capability else SimpleNamespace(test_stdout_path=path)
    assert _verifier_missed_a_program(
        SimpleNamespace(test_return_code=0), selected
    ) is (kind in {"file", "new-suffix"})
