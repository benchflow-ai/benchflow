"""A verifier that locks down /logs/agent keeps its score.

Terminal-Bench 4.1's distributed-dedup isolates its benchmark workers by
making every world-writable folder root's and 0700, as root in its own
verifier sandbox. On Docker that includes the bind-mounted /logs/agent, so
the host could no longer read back the trajectory copy it had published.
The custody record read it after the verifier scored, and its PermissionError
discarded the score ("separate verifier sandbox failed").
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from benchflow.rollout._separate_verifier import _trajectory_record
from benchflow.task.paths import RolloutPaths


def _paths(tmp_path):
    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.agent_dir.mkdir(parents=True, exist_ok=True)
    (paths.agent_dir / "acp_trajectory.jsonl").write_text('{"type": "x"}\n')
    return paths


def test_a_readable_copy_is_recorded_with_its_digest(tmp_path):
    record = _trajectory_record(SimpleNamespace(), _paths(tmp_path))
    assert record["bytes"] == 14 and len(record["sha256"]) == 64
    assert record["path"] == "/logs/agent/acp_trajectory.jsonl"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode")
def test_a_copy_the_verifier_locked_is_recorded_unreadable_not_raised(tmp_path):
    paths = _paths(tmp_path)
    folder = tmp_path / "trial" / "agent"
    folder.chmod(0o000)
    try:
        record = _trajectory_record(SimpleNamespace(), paths)
    finally:
        folder.chmod(0o755)
    assert "PermissionError" in record["unreadable"]
    assert "sha256" not in record
