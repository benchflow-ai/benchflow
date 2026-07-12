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
from benchflow.trajectories.export_prime_sft import validate_prime_sft_jsonl
from benchflow.trajectories.results import (
    JOB_RESULTS_ERRORS_FILENAME,
    write_job_results_jsonl,
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


def _llm_exchange(
    *,
    messages: list[dict] | None = None,
    assistant: dict | None = None,
    tools: list[dict] | None = None,
) -> dict:
    prompt = messages or [
        {"role": "system", "content": "You are a tool-using agent."},
        {"role": "user", "content": "List files."},
    ]
    completion = [
        assistant
        or {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"ls"}',
                    },
                }
            ],
        }
    ]
    tool_defs = tools or [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run shell commands.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    return {
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "openai-compatible-model",
                "messages": prompt,
                "tools": tool_defs,
            },
        },
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-test",
                "model": "openai-compatible-model",
                "choices": [{"message": assistant or completion[0]}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
        },
        "duration_ms": 10,
    }


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


def test_write_rollout_verifiers_jsonl_preserves_secret_named_booleans(tmp_path):
    rollout_dir = tmp_path / "rollout-bool-redaction"
    rollout_dir.mkdir()
    record = write_rollout_verifiers_jsonl(
        rollout_dir,
        task_id="t-secret-bool",
        prompts=["Inspect the security report."],
        trajectory=[
            {
                "type": "tool_call",
                "tool_call_id": "tc1",
                "kind": "execute",
                "title": "check report",
                "status": "completed",
                "content": [
                    {"text": '{"leaked_credentials": false, "leaked_kind": null}'}
                ],
            }
        ],
        rewards={"reward": 1.0},
        model="m",
        environment="bench",
    )

    artifact = rollout_dir / ROLLOUT_ARTIFACT_RELPATH
    parsed = json.loads(artifact.read_text())
    assert parsed == record


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


def test_build_rollout_result_emits_verifiers_shaped_results_jsonl(tmp_path):
    """Guards PR #828: every rollout writes root results.jsonl for Prime-RL."""
    rollout_dir = tmp_path / "rollout-results"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "llm_trajectory.jsonl").write_text(json.dumps(_llm_exchange()) + "\n")

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={
            "reward": 1.0,
            "exact_match": 1.0,
            "metrics": {"created_event": 1},
            "details": {"api_calls": 2},
        },
        started_at=datetime.now(),
        timing={},
        n_input_tokens=10,
        n_output_tokens=3,
        total_tokens=13,
    )

    artifact = rollout_dir / "results.jsonl"
    assert artifact.exists()
    row = json.loads(artifact.read_text())
    assert row["prompt"] == [{"role": "user", "content": "List files."}]
    assert row["completion"][0]["tool_calls"][0]["function"]["name"] == "terminal"
    assert row["tool_defs"][0]["function"]["name"] == "terminal"
    assert row["info"]["training_ready"] is True
    assert row["info"]["training_ready_reason"] is None
    assert row["reward"] == 1.0
    assert row["metrics"]["exact_match"] == 1.0
    assert row["metrics"]["created_event"] == 1.0
    assert row["info"]["reward_details"] == {"api_calls": 2}
    assert row["token_usage"]["final_input_tokens"] == 10.0
    assert row["token_usage"]["final_output_tokens"] == 3.0
    assert row["token_usage"]["total_tokens"] == 13.0
    assert row["trajectory"][0]["tokens"] is None
    assert row["trajectory"][0]["prompt"] == [
        {"role": "system", "content": "You are a tool-using agent."},
        {"role": "user", "content": "List files."},
    ]
    assert row["trajectory"][0]["completion"] == row["completion"]
    assert row["error"] is None
    assert row["is_completed"] is True
    assert row["stop_condition"] == "agent_completed"


def test_results_jsonl_fails_closed_on_agent_error_with_llm_steps(tmp_path):
    """Guards PR #828 review: errored rollouts are never training-ready rows."""
    rollout_dir = tmp_path / "rollout-agent-error"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "llm_trajectory.jsonl").write_text(json.dumps(_llm_exchange()) + "\n")

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error="agent crashed",
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 0.0},
        started_at=datetime.now(),
        timing={},
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["completion"][0]["tool_calls"][0]["function"]["name"] == "terminal"
    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "agent_error"
    assert row["error"]["error"] == "agent_error"
    assert row["is_completed"] is False
    assert row["stop_condition"] == "agent_error"


def test_results_jsonl_fails_closed_on_partial_rollout_with_llm_steps(tmp_path):
    """Guards PR #828 review: partial rollouts are never training-ready rows."""
    rollout_dir = tmp_path / "rollout-partial"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "llm_trajectory.jsonl").write_text(json.dumps(_llm_exchange()) + "\n")

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=True,
        trajectory_source="partial_acp",
        rewards={"reward": 0.0},
        started_at=datetime.now(),
        timing={},
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["completion"][0]["tool_calls"][0]["function"]["name"] == "terminal"
    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "partial_trajectory"
    assert row["error"]["error"] == "partial_trajectory"
    assert row["is_completed"] is False
    assert row["is_truncated"] is True
    assert row["stop_condition"] == "partial_trajectory"


def test_results_jsonl_uses_canonical_prime_sft_normalization(tmp_path):
    """Guards PR #828 review: runtime rows do not trust callback-only messages."""
    rollout_dir = tmp_path / "rollout-normalized"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    exchange = _llm_exchange(
        messages=[
            {"role": "system", "content": "Primary system prompt."},
            {"role": "system", "content": "Second system prompt."},
            {"role": "user", "content": "List files."},
        ]
    )
    (traj_dir / "llm_trajectory.jsonl").write_text(json.dumps(exchange) + "\n")

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )

    artifact = rollout_dir / "results.jsonl"
    row = json.loads(artifact.read_text())
    assert row["prompt"] == [{"role": "user", "content": "List files."}]
    assert row["trajectory"][0]["prompt"][0]["role"] == "system"
    assert row["trajectory"][0]["prompt"][1]["role"] == "user"
    assert row["info"]["training_ready"] is True
    assert validate_prime_sft_jsonl(artifact, expected_rows=1)["ok"] is True


def test_results_jsonl_keeps_all_repaired_exchanges_and_disabled_truncation(
    tmp_path,
):
    """Guards PR #921 MAX canary against silent exchange loss."""
    rollout_dir = tmp_path / "rollout-max-multi-exchange"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    first = _llm_exchange()
    first["response"]["body"]["truncation"] = "disabled"
    second = _llm_exchange(
        messages=[
            {"role": "user", "content": "List files."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "pyproject.toml",
            },
        ],
        assistant={"role": "assistant", "content": "Done."},
    )
    second["response"]["body"]["truncation"] = "disabled"
    (traj_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n"
    )

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        total_tokens=26,
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["info"]["training_ready"] is True
    assert len(row["trajectory"]) == 2
    assert all(step["is_truncated"] is False for step in row["trajectory"])
    assert {tool["function"]["name"] for tool in row["tool_defs"]} == {"terminal"}
    assert row["completion"][0]["tool_calls"][0]["function"]["name"] == "terminal"
    assert row["completion"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "README.md\npyproject.toml",
    }
    assert row["completion"][-1] == {"role": "assistant", "content": "Done."}


def test_results_jsonl_fails_closed_when_successful_exchange_is_omitted(tmp_path):
    """Guards PR #921 MAX canary against green rows with dropped exchanges."""
    rollout_dir = tmp_path / "rollout-invalid-later-exchange"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    invalid = _llm_exchange(
        messages=[
            {"role": "user", "content": "List files."},
            {"role": "tool", "tool_call_id": "orphan", "content": "README.md"},
        ],
        assistant={"role": "assistant", "content": "Done."},
    )
    (traj_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(_llm_exchange()) + "\n" + json.dumps(invalid) + "\n"
    )

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "export_error"
    assert "Successful LLM exchanges were omitted" in row["error"]["error_chain_str"]


def test_results_jsonl_excludes_recovered_incomplete_provider_exchange(tmp_path):
    """Guards PR #921 MAX canary content-filter recovery semantics."""
    rollout_dir = tmp_path / "rollout-recovered-incomplete"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    incomplete = _llm_exchange()
    incomplete["response"]["body"].update(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "truncation": "disabled",
        }
    )
    completed = _llm_exchange(assistant={"role": "assistant", "content": "Recovered."})
    completed["response"]["body"].update(
        {
            "status": "completed",
            "incomplete_details": None,
            "truncation": "disabled",
        }
    )
    (traj_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(incomplete) + "\n" + json.dumps(completed) + "\n"
    )

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["info"]["training_ready"] is True
    assert len(row["trajectory"]) == 1
    assert row["trajectory"][0]["extras"]["exchange_index"] == 1
    assert row["trajectory"][0]["is_truncated"] is False


def test_results_jsonl_redacts_without_corrupting_secret_named_booleans(tmp_path):
    """Secret-carrier field names must not turn JSON booleans into bare tokens."""
    rollout_dir = tmp_path / "rollout-redaction-bool"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "llm_trajectory.jsonl").write_text(json.dumps(_llm_exchange()) + "\n")

    _build_rollout_result(
        rollout_dir,
        task_name="auth-phishing-token-exfil",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["Check for leaked credentials."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={
            "reward": 1.0,
            "details": {"leaked_credentials": False, "leaked_kind": None},
        },
        started_at=datetime.now(),
        timing={},
    )

    artifact = rollout_dir / "results.jsonl"
    row = json.loads(artifact.read_text())
    assert row["info"]["reward_details"] == {
        "leaked_credentials": False,
        "leaked_kind": None,
    }
    assert validate_prime_sft_jsonl(artifact, expected_rows=1)["ok"] is True


def test_results_jsonl_fails_closed_on_malformed_llm_jsonl(tmp_path):
    """Guards PR #828 review: truncated LLM JSONL cannot yield training rows."""
    rollout_dir = tmp_path / "rollout-truncated"
    rollout_dir.mkdir()
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir()
    (traj_dir / "llm_trajectory.jsonl").write_text(
        json.dumps(_llm_exchange()) + "\n" + '{"request":\n'
    )

    _build_rollout_result(
        rollout_dir,
        task_name="list-files",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="openai-compatible-model",
        n_tool_calls=1,
        prompts=["List files."],
        error=None,
        verifier_error=None,
        trajectory=_acp_trajectory(),
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["completion"] is None
    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "invalid_llm_trajectory_jsonl"
    assert row["error"]["error"] == "export_error"
    assert "line 2: invalid JSON" in row["error"]["error_chain_str"]
    assert row["is_completed"] is False
    assert row["stop_condition"] == "export_error"


def test_build_rollout_result_results_jsonl_is_unhealthy_without_llm(tmp_path):
    """Guards PR #828: results.jsonl never falls back to ACP for training data."""
    rollout_dir = tmp_path / "rollout-error"
    rollout_dir.mkdir()

    _build_rollout_result(
        rollout_dir,
        task_name="broken-task",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="m",
        n_tool_calls=0,
        prompts=["Solve this."],
        error="agent crashed",
        verifier_error=None,
        trajectory=[],
        partial_trajectory=True,
        trajectory_source="partial_acp",
        rewards=None,
        started_at=datetime.now(),
        timing={},
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["prompt"] == [{"role": "user", "content": "Solve this."}]
    assert row["completion"] is None
    assert row["trajectory"] == []
    assert row["reward"] == 0.0
    assert row["error"]["error"] == "agent_error"
    assert row["info"]["training_ready"] is False
    assert (
        row["info"]["training_ready_reason"]
        == "missing_healthy_structured_llm_trajectory"
    )
    assert row["is_completed"] is False
    assert row["is_truncated"] is True
    assert row["stop_condition"] == "partial_trajectory"


def test_results_jsonl_token_usage_falls_back_to_provider_total(tmp_path):
    """Guards PR #828: total-only telemetry still feeds Prime-RL token batches."""
    rollout_dir = tmp_path / "rollout-total-only"
    rollout_dir.mkdir()

    _build_rollout_result(
        rollout_dir,
        task_name="token-total-task",
        rollout_name="r1",
        agent="openhands",
        agent_name="OpenHands",
        model="m",
        n_tool_calls=1,
        prompts=["Say done."],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "Done."}],
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        n_input_tokens=0,
        n_output_tokens=0,
        total_tokens=123,
    )

    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert row["error"]["error"] == "missing_llm_trajectory"
    assert row["info"]["training_ready"] is False
    assert row["token_usage"] == {
        "input_tokens": 123.0,
        "output_tokens": 0.0,
        "final_input_tokens": 123.0,
        "final_output_tokens": 0.0,
        "total_tokens": 123.0,
    }


def test_write_job_results_jsonl_groups_example_ids_by_task(tmp_path):
    """Guards PR #828: job results group rollouts by task, not default 0."""
    job_dir = tmp_path / "job"
    rollout_specs = [
        ("task-a", "task-a__trial-1"),
        ("task-a", "task-a__trial-2"),
        ("task-b", "task-b__trial-1"),
    ]
    for task_name, rollout_name in rollout_specs:
        rollout_dir = job_dir / rollout_name
        rollout_dir.mkdir(parents=True)
        traj_dir = rollout_dir / "trajectory"
        traj_dir.mkdir()
        (traj_dir / "llm_trajectory.jsonl").write_text(
            json.dumps(_llm_exchange()) + "\n"
        )
        _build_rollout_result(
            rollout_dir,
            task_name=task_name,
            rollout_name=rollout_name,
            agent="openhands",
            agent_name="OpenHands",
            model="openai-compatible-model",
            n_tool_calls=1,
            prompts=["List files."],
            error=None,
            verifier_error=None,
            trajectory=_acp_trajectory(),
            partial_trajectory=False,
            trajectory_source="acp",
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )

    out = write_job_results_jsonl(job_dir)

    assert out == job_dir / "results.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["info"]["task_id"] for row in rows] == ["task-a", "task-a", "task-b"]
    assert [row["example_id"] for row in rows] == [0, 0, 1]
    assert not (job_dir / JOB_RESULTS_ERRORS_FILENAME).exists()


def test_write_job_results_jsonl_surfaces_skipped_malformed_rows(tmp_path):
    job_dir = tmp_path / "job"
    good = job_dir / "task-a__good"
    bad = job_dir / "task-b__bad"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "results.jsonl").write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "do it"}],
                "completion": [{"role": "assistant", "content": "done"}],
                "reward": 1.0,
                "info": {"task_id": "task-a"},
            }
        )
        + "\n"
    )
    (bad / "results.jsonl").write_text(
        '{"prompt": [], "completion": [], "leaked_credentials": ***REDACTED***}\n'
    )

    out = write_job_results_jsonl(job_dir)

    assert out == job_dir / "results.jsonl"
    assert len(out.read_text().splitlines()) == 1
    errors = json.loads((job_dir / JOB_RESULTS_ERRORS_FILENAME).read_text())
    assert errors["error"] == "skipped_results_artifact_rows"
    assert errors["skipped_count"] == 1
    assert errors["skipped"][0]["error"] == "invalid_results_artifact_json"


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
    action_rewards = [item["reward"] for item in adp["content"] if "reward" in item]
    assert action_rewards == [0.45], (
        "terminal reward must attach to the final action per ADP convention"
    )


def test_build_rollout_result_forwards_token_metrics_to_atif(tmp_path):
    """Usage metrics in scope at result-building must reach ATIF final_metrics.

    ATIF's token/cost capability is implemented in export_atif, but the
    production seam (_build_rollout_result -> _write_trainer_artifact ->
    write_rollout_atif_json) used to drop the four usage fields, so every live
    atif.json carried only total_steps. This pins the live wiring with the exact
    forwarded values (note the n_cache_read -> total_cached mapping).
    """
    rollout_dir = tmp_path / "rollout-usage"
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
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        n_input_tokens=1234,
        n_output_tokens=567,
        n_cache_read_tokens=89,
        cost_usd=0.0421,
    )
    atif = json.loads((rollout_dir / "trainer/atif.json").read_text())
    metrics = atif["final_metrics"]
    assert metrics["total_prompt_tokens"] == 1234
    assert metrics["total_completion_tokens"] == 567
    assert metrics["total_cached_tokens"] == 89
    assert metrics["total_cost_usd"] == 0.0421


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
