"""Artifact timestamp serialization regressions."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone


def test_result_timestamps_are_timezone_aware_iso8601(tmp_path):
    """Guards PR #928 / issue #528: result.json timestamps are UTC ISO."""
    from benchflow.rollout import _build_rollout_result

    rollout_dir = tmp_path / "trial"
    rollout_dir.mkdir()
    _build_rollout_result(
        rollout_dir,
        task_name="t1",
        rollout_name="trial-1",
        agent="test",
        agent_name="openhands",
        model="m",
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 3, 24, 10, 0),
        timing={},
    )

    data = json.loads((rollout_dir / "result.json").read_text())
    assert data["started_at"] == "2026-03-24T10:00:00Z"
    assert data["finished_at"].endswith("Z")
    assert " " not in data["finished_at"]
    assert datetime.fromisoformat(data["finished_at"].replace("Z", "+00:00"))


def test_result_timestamps_normalize_offsets_to_utc(tmp_path):
    """Guards PR #928 / issue #528: aware result timestamps serialize in UTC."""
    from benchflow.rollout import _build_rollout_result

    rollout_dir = tmp_path / "trial"
    rollout_dir.mkdir()
    _build_rollout_result(
        rollout_dir,
        task_name="t1",
        rollout_name="trial-1",
        agent="test",
        agent_name="openhands",
        model="m",
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 3, 24, 3, 0, tzinfo=timezone(timedelta(hours=-7))),
        timing={},
    )

    data = json.loads((rollout_dir / "result.json").read_text())
    assert data["started_at"] == "2026-03-24T10:00:00Z"


def test_native_rollout_started_at_is_utc_aware_at_source(tmp_path, monkeypatch):
    """Guards PR #928: native rollout timestamps are not local naive values."""
    from benchflow.rollout import _setup as rollout_setup
    from benchflow.rollout._setup import _init_rollout

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "task.toml").write_text('version = "1.0"\n')

    local_wall_time = datetime(2026, 7, 23, 5, 10, 58)
    utc_instant = datetime(2026, 7, 23, 12, 10, 58, tzinfo=UTC)

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return local_wall_time
            assert tz is UTC
            return utc_instant

    monkeypatch.setattr(rollout_setup, "datetime", FakeDateTime)

    _, _, _, started_at, job_name, _ = _init_rollout(
        task_dir,
        job_name=None,
        rollout_name=None,
        jobs_dir=tmp_path / "jobs",
    )

    assert job_name == "2026-07-23__05-10-58"
    assert started_at == utc_instant
