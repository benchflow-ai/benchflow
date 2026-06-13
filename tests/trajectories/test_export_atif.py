"""ATIF (Agent Trajectory Interchange Format) trajectory export.

Record shape pinned against the Harbor pydantic models
(harbor-framework/harbor, src/harbor/models/trajectories/*.py) and the
ATIF RFC (rfcs/0001-trajectory-format.md), schema version ATIF-v1.7.
"""

import json

import pytest

from benchflow.trajectories.export_atif import (
    acp_events_to_atif_steps,
    trajectory_to_atif_record,
    write_rollout_atif_json,
)


def _sample_events():
    return [
        {"type": "user_message", "text": "List the files."},
        {"type": "agent_thought", "text": "I should run ls."},
        {
            "type": "tool_call",
            "tool_call_id": "tc1",
            "kind": "execute",
            "title": "ls",
            "status": "completed",
            "content": [
                {"type": "content", "content": {"type": "text", "text": "README.md"}}
            ],
        },
        {"type": "agent_message", "text": "One file: README.md."},
    ]


def test_golden_record():
    rec = trajectory_to_atif_record(
        session_id="sess-1",
        agent_name="claude-code",
        agent_version="2.0.1",
        model="claude-haiku-4-5",
        events=_sample_events(),
        prompts=["Solve the task."],
        total_prompt_tokens=220,
        total_completion_tokens=80,
        total_cost_usd=0.00135,
    )
    assert rec == {
        "schema_version": "ATIF-v1.7",
        "session_id": "sess-1",
        "agent": {
            "name": "claude-code",
            "version": "2.0.1",
            "model_name": "claude-haiku-4-5",
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "Solve the task."},
            {"step_id": 2, "source": "user", "message": "List the files."},
            {
                "step_id": 3,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "tc1",
                        "function_name": "execute",
                        "arguments": {},
                        "extra": {"title": "ls", "status": "completed"},
                    }
                ],
                "reasoning_content": "I should run ls.",
                "observation": {
                    "results": [{"source_call_id": "tc1", "content": "README.md"}]
                },
            },
            {"step_id": 4, "source": "agent", "message": "One file: README.md."},
        ],
        "final_metrics": {
            "total_steps": 4,
            "total_prompt_tokens": 220,
            "total_completion_tokens": 80,
            "total_cost_usd": 0.00135,
        },
    }


def test_empty_trajectory_raises():
    with pytest.raises(ValueError):
        trajectory_to_atif_record(session_id="s", agent_name="a", events=[], prompts=[])


def test_omitted_optionals_stay_absent():
    rec = trajectory_to_atif_record(
        session_id="",
        agent_name="",
        events=[{"type": "agent_message", "text": "hi"}],
    )
    assert "session_id" not in rec
    assert rec["agent"] == {"name": "unknown", "version": "unknown"}
    assert rec["final_metrics"] == {"total_steps": 1}


def test_missing_tool_call_id_synthesized_and_referenced():
    steps = acp_events_to_atif_steps(
        [
            {
                "type": "tool_call",
                "tool_call_id": "",
                "kind": "read",
                "title": "cat notes.txt",
                "status": "completed",
                "content": [{"text": "hello"}],
            }
        ]
    )
    (step,) = steps
    call_id = step["tool_calls"][0]["tool_call_id"]
    assert call_id == "call_1"
    assert step["observation"]["results"][0]["source_call_id"] == call_id


def test_empty_tool_output_omits_observation():
    steps = acp_events_to_atif_steps(
        [
            {
                "type": "tool_call",
                "tool_call_id": "tc1",
                "kind": "execute",
                "title": "touch a",
                "status": "failed",
                "content": [],
            }
        ]
    )
    (step,) = steps
    assert "observation" not in step
    assert step["tool_calls"][0]["extra"]["status"] == "failed"


def test_thought_before_user_message_flushes_in_order():
    steps = acp_events_to_atif_steps(
        [
            {"type": "agent_thought", "text": "hmm"},
            {"type": "user_message", "text": "continue"},
        ]
    )
    assert steps == [
        {"step_id": 1, "source": "agent", "message": "", "reasoning_content": "hmm"},
        {"step_id": 2, "source": "user", "message": "continue"},
    ]


def test_trailing_thoughts_join_into_standalone_step():
    steps = acp_events_to_atif_steps(
        [
            {"type": "agent_thought", "text": "first"},
            {"type": "agent_thought", "text": "second"},
        ]
    )
    assert steps == [
        {
            "step_id": 1,
            "source": "agent",
            "message": "",
            "reasoning_content": "first\n\nsecond",
        }
    ]


def test_oracle_event_renders_command():
    steps = acp_events_to_atif_steps([{"type": "oracle", "command": "bash run.sh"}])
    assert steps == [
        {"step_id": 1, "source": "agent", "message": "[oracle: bash run.sh]"}
    ]


def test_empty_and_malformed_events_skipped():
    steps = acp_events_to_atif_steps(
        [
            {"type": "agent_message", "text": ""},
            "not-a-dict",
            {"type": "unknown_event", "text": "x"},
            {"type": "user_message", "text": "real"},
        ]
    )
    assert steps == [{"step_id": 1, "source": "user", "message": "real"}]


def test_write_rollout_atif_json(tmp_path):
    rec = write_rollout_atif_json(
        tmp_path,
        session_id="sess-1",
        agent_name="claude-code",
        prompts=["Solve the task."],
        trajectory=_sample_events(),
        model="claude-haiku-4-5",
    )
    out = tmp_path / "trainer" / "atif.json"
    assert json.loads(out.read_text()) == rec
    assert rec["schema_version"] == "ATIF-v1.7"
    assert [s["step_id"] for s in rec["steps"]] == [1, 2, 3, 4]


def test_write_skips_empty_trajectory(tmp_path):
    rec = write_rollout_atif_json(
        tmp_path,
        session_id="s",
        agent_name="a",
        prompts=[],
        trajectory=[],
    )
    assert rec is None
    assert not (tmp_path / "trainer" / "atif.json").exists()


def test_write_redacts_secrets(tmp_path):
    events = [
        {
            "type": "tool_call",
            "tool_call_id": "tc1",
            "kind": "execute",
            "title": "env",
            "status": "completed",
            "content": [{"text": "OPENAI_API_KEY=sk-abc123def456ghi789"}],
        }
    ]
    rec = write_rollout_atif_json(
        tmp_path,
        session_id="s",
        agent_name="a",
        prompts=None,
        trajectory=events,
    )
    raw = (tmp_path / "trainer" / "atif.json").read_text()
    assert "sk-abc123def456ghi789" not in raw
    content = rec["steps"][0]["observation"]["results"][0]["content"]
    assert "***REDACTED***" in content


def test_write_scrubs_non_finite_cost(tmp_path):
    write_rollout_atif_json(
        tmp_path,
        session_id="s",
        agent_name="a",
        prompts=None,
        trajectory=[{"type": "agent_message", "text": "hi"}],
        total_cost_usd=float("nan"),
    )
    parsed = json.loads((tmp_path / "trainer" / "atif.json").read_text())
    assert parsed["final_metrics"]["total_cost_usd"] is None
