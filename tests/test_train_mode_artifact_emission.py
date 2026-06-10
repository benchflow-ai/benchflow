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
import re
from datetime import datetime

from benchflow.rollout import _build_rollout_result
from benchflow.trajectories.export import (
    ROLLOUT_ARTIFACT_RELPATH,
    acp_events_to_messages,
    reward_map_to_verify_result,
    write_job_verifiers_jsonl,
    write_rollout_verifiers_jsonl,
)

_FAKE_GEMINI_KEY = "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"
_FAKE_HEADER_SECRET = "plainSecretNoPrefix123"


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


def _secret_bearing_acp_trajectory() -> list[dict]:
    """A trajectory with secrets in user, assistant, and tool-call content."""
    return [
        {
            "type": "user_message",
            "text": f"User pasted GEMINI_API_KEY={_FAKE_GEMINI_KEY}",
        },
        {
            "type": "agent_message",
            "text": f"Trying request with x-api-key: {_FAKE_HEADER_SECRET}",
        },
        {
            "type": "tool_call",
            "tool_call_id": "tc-secret",
            "kind": "bash",
            "title": "curl service",
            "status": "completed",
            "content": [
                {
                    "text": (
                        f"x-goog-api-key: {_FAKE_GEMINI_KEY}\n"
                        f"api_key={_FAKE_HEADER_SECRET}"
                    )
                }
            ],
        },
    ]


def _assert_no_trainer_secret_leak(text: str) -> None:
    assert "***REDACTED***" in text
    assert _FAKE_GEMINI_KEY not in text
    assert _FAKE_HEADER_SECRET not in text
    residual = re.findall(r"AIzaSy[A-Za-z0-9_-]+", text)
    assert residual == [], f"live Gemini key shapes survived: {residual}"


# unit-level helpers


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


# rollout-level write


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


def test_write_rollout_verifiers_jsonl_redacts_trajectory_secrets(tmp_path):
    """Guards the fix from PR #585 against trainer/verifiers.jsonl leaks."""
    rollout_dir = tmp_path / "rollout-secret"
    rollout_dir.mkdir()
    record = write_rollout_verifiers_jsonl(
        rollout_dir,
        task_id="t-secret",
        prompts=[f"Prompt includes {_FAKE_GEMINI_KEY}"],
        trajectory=_secret_bearing_acp_trajectory(),
        rewards={"reward": 1.0},
        model="m",
        environment="bench",
    )

    artifact = rollout_dir / ROLLOUT_ARTIFACT_RELPATH
    text = artifact.read_text()
    _assert_no_trainer_secret_leak(text)
    _assert_no_trainer_secret_leak(json.dumps(record))
    json.loads(text)


# job-level aggregation


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


def test_write_job_verifiers_jsonl_redacts_aggregated_records(tmp_path):
    """Guards the fix from PR #585 against job verifiers.jsonl leaks."""
    job_dir = tmp_path / "job-secret"
    rollout_dir = job_dir / "rollout-secret"
    rollout_dir.mkdir(parents=True)
    write_rollout_verifiers_jsonl(
        rollout_dir,
        task_id="t-secret",
        prompts=[f"Prompt includes {_FAKE_GEMINI_KEY}"],
        trajectory=_secret_bearing_acp_trajectory(),
        rewards={"reward": 1.0},
        model="m",
        environment="bench",
    )
    # Simulate a legacy/raw per-rollout artifact so the job aggregation boundary
    # proves it redacts too, instead of merely concatenating already-redacted
    # rollout output.
    raw_record = {
        "example_id": 0,
        "prompt": [{"role": "user", "content": f"Prompt includes {_FAKE_GEMINI_KEY}"}],
        "completion": [
            {
                "role": "assistant",
                "content": f"Trying request with x-api-key: {_FAKE_HEADER_SECRET}",
            }
        ],
        "reward": 1.0,
        "metrics": {},
        "is_completed": True,
        "is_truncated": False,
        "info": {"task_id": "t-secret", "environment": "bench", "model": "m"},
    }
    (rollout_dir / ROLLOUT_ARTIFACT_RELPATH).write_text(json.dumps(raw_record) + "\n")

    artifact = write_job_verifiers_jsonl(job_dir)
    assert artifact == job_dir / "verifiers.jsonl"
    text = artifact.read_text()
    _assert_no_trainer_secret_leak(text)
    assert len(text.splitlines()) == 1
    json.loads(text)


# end-to-end: _build_rollout_result wires the seam


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


def test_build_rollout_result_emits_atif_and_adp(tmp_path):
    """Scored rollouts emit the ecosystem trajectory formats out of the box.

    ATIF (trainer/atif.json) and ADP (trainer/adp.jsonl) must land beside
    verifiers.jsonl from _build_rollout_result — library-only emitters that
    no run ever calls are a doc-claim violation, not a feature.
    """
    rollout_dir = tmp_path / "rollout-formats"
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
        rewards={"reward": 0.45},
        started_at=datetime.now(),
        timing={},
    )
    atif = json.loads((rollout_dir / "trainer/atif.json").read_text())
    assert atif["session_id"] == "archive-alice__r1"
    assert atif["agent"]["name"] == "claude-agent-acp"
    assert atif["steps"], "non-empty trajectory must produce ATIF steps"

    adp_line = (rollout_dir / "trainer/adp.jsonl").read_text().strip()
    adp = json.loads(adp_line)
    assert adp["id"] == "archive-alice__r1"
    assert adp["details"]["task_id"] == "archive-alice"
    action_rewards = [
        item["reward"] for item in adp["content"] if "reward" in item
    ]
    assert action_rewards == [0.45], (
        "terminal reward must attach to the final action per ADP convention"
    )


def test_build_rollout_result_atif_for_empty_trajectory_is_prompt_only(tmp_path):
    """Oracle runs (no agent events) emit a prompts-only ATIF, never agent steps.

    The emitter folds prompts into user steps by design, so the document
    stays schema-valid (steps non-empty); what must never happen is a
    fabricated agent step from an empty trajectory.
    """
    rollout_dir = tmp_path / "rollout-oracle"
    rollout_dir.mkdir()
    _build_rollout_result(
        rollout_dir,
        task_name="archive-alice",
        rollout_name="oracle",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=["Archive the email from Alice."],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        trajectory_source=None,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    atif = json.loads((rollout_dir / "trainer/atif.json").read_text())
    sources = [s.get("source") for s in atif["steps"]]
    assert sources == ["user"], (
        f"oracle ATIF must contain only the prompt-derived user step, got {sources}"
    )
    assert (rollout_dir / "trainer/adp.jsonl").exists(), (
        "ADP records prompts even without actions"
    )
