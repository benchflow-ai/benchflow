"""Train-mode trainer artifact emission (issue #385).

Scored rollouts must emit trainer-ready Verifiers / ORS JSONL artifacts:

- per-rollout:  ``rollout_dir/trainer/verifiers.jsonl``
- per-job:      ``job_dir/verifiers.jsonl``

These tests drive the artifact path without standing up a real sandbox —
they invoke the exporters with simulated scored-rollout inputs and
``_build_rollout_result`` to assert the artifacts land where the
architecture says they should.
"""

from __future__ import annotations

import json
from datetime import datetime

from benchflow.rollout import _build_rollout_result
from benchflow.trajectories.export import (
    ROLLOUT_ARTIFACT_RELPATH,
    acp_events_to_messages,
    reward_map_to_verify_result,
    write_job_verifiers_jsonl,
    write_rollout_verifiers_jsonl,
)


def _acp_trajectory() -> list[dict]:
    """A representative ACP trajectory shape (see _capture._capture_session_trajectory)."""
    return [
        {"type": "user_message", "text": "Archive the email from Alice."},
        {"type": "agent_thought", "text": "Reading inbox first."},
        {
            "type": "tool_call",
            "tool_call_id": "tc1",
            "kind": "bash",
            "title": "ls inbox",
            "status": "completed",
            "content": [{"text": "alice@example.com"}],
        },
        {"type": "agent_message", "text": "Archived 1 email."},
    ]


# ── unit-level helpers ────────────────────────────────────────────────


def test_acp_events_to_messages_prepends_prompts_and_keeps_order():
    msgs = acp_events_to_messages(_acp_trajectory(), prompts=["Solve the task."])
    # Leading user message comes from the prompt list, then the ACP-captured
    # user_message, then assistant turns (thought + tool_call + message).
    assert msgs[0] == {"role": "user", "content": "Solve the task."}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Archive the email from Alice."
    roles = [m["role"] for m in msgs]
    # user → user → assistant (thought) → assistant (tool_call) → assistant (message)
    assert roles == ["user", "user", "assistant", "assistant", "assistant"]
    # Tool-call rendering preserves the title so the trainer keeps something
    # to anchor on.
    assert "ls inbox" in msgs[3]["content"]


def test_acp_events_to_messages_handles_empty_trajectory():
    msgs = acp_events_to_messages([], prompts=["only prompt"])
    assert msgs == [{"role": "user", "content": "only prompt"}]


def test_reward_map_to_verify_result_lifts_scalars_and_rubric():
    rewards = {
        "reward": 0.75,
        "exact_match": 1.0,
        "rubric": [
            {"name": "clarity", "score": 0.5},
            {"name": "correctness", "score": 1.0},
        ],
    }
    vr = reward_map_to_verify_result(rewards)
    assert vr.reward == 0.75
    assert vr.items["exact_match"] == 1.0
    assert vr.items["clarity"] == 0.5
    assert vr.items["correctness"] == 1.0
    assert vr.error is None


def test_reward_map_to_verify_result_handles_none():
    vr = reward_map_to_verify_result(None, error="verifier crashed")
    assert vr.reward == 0.0
    assert vr.items == {}
    assert vr.error == "verifier crashed"


# ── rollout-level write ───────────────────────────────────────────────


def test_write_rollout_verifiers_jsonl_emits_canonical_path(tmp_path):
    rollout_dir = tmp_path / "rollout-1"
    rollout_dir.mkdir()
    record = write_rollout_verifiers_jsonl(
        rollout_dir,
        task_id="t1",
        prompts=["Do the thing."],
        trajectory=_acp_trajectory(),
        rewards={"reward": 1.0, "exact_match": 1.0},
        model="claude-haiku-4-5",
        environment="bench",
    )
    artifact = rollout_dir / ROLLOUT_ARTIFACT_RELPATH
    assert artifact.exists(), (
        "trainer/verifiers.jsonl must exist after a scored rollout"
    )
    lines = artifact.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    # The record on disk is what the helper returned.
    assert parsed["reward"] == 1.0
    assert parsed == record
    # Verifiers RolloutOutput required fields are present.
    for field in (
        "prompt",
        "completion",
        "reward",
        "metrics",
        "is_completed",
        "is_truncated",
        "example_id",
        "info",
    ):
        assert field in parsed
    assert parsed["info"]["task_id"] == "t1"
    assert parsed["info"]["model"] == "claude-haiku-4-5"
    # Reward survived the ORS round-trip with valid metadata.
    assert parsed["info"]["reward_valid"] is True


def test_write_rollout_verifiers_jsonl_marks_invalid_when_no_rewards(tmp_path):
    rollout_dir = tmp_path / "rollout-failed"
    rollout_dir.mkdir()
    write_rollout_verifiers_jsonl(
        rollout_dir,
        task_id="t1",
        prompts=["Do the thing."],
        trajectory=_acp_trajectory(),
        rewards=None,
        model="claude-haiku-4-5",
        environment="bench",
        error="verifier timed out",
    )
    parsed = json.loads((rollout_dir / ROLLOUT_ARTIFACT_RELPATH).read_text().strip())
    assert parsed["reward"] == 0.0
    assert parsed["info"]["reward_valid"] is False


# ── job-level aggregation ─────────────────────────────────────────────


def test_write_job_verifiers_jsonl_aggregates_all_rollouts(tmp_path):
    job_dir = tmp_path / "job-x"
    for i in range(3):
        rdir = job_dir / f"rollout-{i}"
        rdir.mkdir(parents=True)
        write_rollout_verifiers_jsonl(
            rdir,
            task_id=f"task-{i}",
            prompts=["Do the thing."],
            trajectory=_acp_trajectory(),
            rewards={"reward": float(i) / 2.0},
            model="m",
            environment="bench",
            example_id=i,
        )
    artifact = write_job_verifiers_jsonl(job_dir)
    assert artifact == job_dir / "verifiers.jsonl"
    lines = artifact.read_text().splitlines()
    assert len(lines) == 3
    example_ids = sorted(json.loads(line)["example_id"] for line in lines)
    assert example_ids == [0, 1, 2]


def test_write_job_verifiers_jsonl_returns_none_when_no_rollouts(tmp_path):
    empty_job = tmp_path / "empty-job"
    empty_job.mkdir()
    assert write_job_verifiers_jsonl(empty_job) is None
    assert not (empty_job / "verifiers.jsonl").exists()


# ── end-to-end: _build_rollout_result wires the seam ─────────────────


def test_build_rollout_result_emits_trainer_artifact(tmp_path):
    """Every scored rollout that reaches result-building must emit the artifact.

    Drives ``_build_rollout_result`` with simulated scored-rollout inputs —
    the integration boundary issue #385 says was missing.
    """
    rollout_dir = tmp_path / "rollout-final"
    rollout_dir.mkdir()
    _build_rollout_result(
        rollout_dir,
        task_name="archive-alice",
        rollout_name="r1",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="claude-haiku-4-5",
        n_tool_calls=1,
        prompts=["Archive the email from Alice."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0, "exact_match": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    artifact = rollout_dir / ROLLOUT_ARTIFACT_RELPATH
    assert artifact.exists(), (
        "scored rollouts must emit trainer/verifiers.jsonl from "
        "_build_rollout_result (issue #385)"
    )
    parsed = json.loads(artifact.read_text().strip())
    assert parsed["reward"] == 1.0
    assert parsed["info"]["task_id"] == "archive-alice"
    assert parsed["info"]["model"] == "claude-haiku-4-5"
