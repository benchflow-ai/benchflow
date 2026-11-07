"""Guards commit 36287d0d: issue #528 timezone-aware serialization."""

import json
import os
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from benchflow.rollout._results import _build_rollout_result, _write_config
from benchflow.skill_policy import resolve_task_skill_policy


@pytest.fixture
def local_zone():
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "Etc/GMT+5"
    time.tzset()
    yield
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


def test_result_times_are_iso_with_the_actual_local_offset(tmp_path, local_zone):
    start = datetime.now() - timedelta(seconds=5)
    _build_rollout_result(
        tmp_path,
        task_name="task",
        rollout_name="trial",
        agent="nop",
        agent_name="nop",
        model=None,
        n_tool_calls=2,
        prompts=["go"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=start,
        timing={},
    )
    data = json.loads((tmp_path / "result.json").read_text())
    for key in ("started_at", "finished_at"):
        assert "T" in data[key]
        parsed = datetime.fromisoformat(data[key])
        assert parsed.utcoffset() == timedelta(hours=-5)
    assert datetime.fromisoformat(data["started_at"]).timestamp() == start.timestamp()
    # Legacy readers retain their count fields; canonical counts must agree.
    for key in ("n_tool_calls", "n_skill_invocations", "n_prompts"):
        assert data[key] == data["agent_result"][key]


@pytest.mark.parametrize("offset", [UTC, timezone(timedelta(hours=9))])
def test_config_preserves_an_aware_start_instant(tmp_path, offset):
    start = datetime(2026, 11, 4, 12, tzinfo=offset)
    policy = resolve_task_skill_policy(
        task_path=tmp_path,
        skill_mode="no-skill",
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
    )
    _write_config(
        tmp_path,
        task_path=tmp_path,
        agent="nop",
        model=None,
        environment="docker",
        skill_policy=policy,
        sandbox_user=None,
        context_root=None,
        timeout=30,
        started_at=start,
        agent_env={},
    )
    value = json.loads((tmp_path / "config.json").read_text())["started_at"]
    assert "T" in value
    assert datetime.fromisoformat(value) == start
