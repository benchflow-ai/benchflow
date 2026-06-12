"""ADP (Agent Data Protocol) trajectory export.

Record shape pinned against the ADP pydantic schemas
(neulab/agent-data-protocol, schema/trajectory.py, schema/action/*.py,
schema/observation/*.py), schema version 1.3.1.
"""

import json

from benchflow.trajectories.export_adp import (
    acp_events_to_adp_content,
    trajectory_to_adp_record,
    write_job_adp_jsonl,
    write_rollout_adp_jsonl,
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
    rec = trajectory_to_adp_record(
        trajectory_id="rollout-1",
        events=_sample_events(),
        prompts=["Solve the task."],
        reward=1.0,
        details={"task_id": "demo/list-files", "environment": "demo"},
    )
    assert rec == {
        "schema_version": "1.3.1",
        "id": "rollout-1",
        "content": [
            {
                "class_": "text_observation",
                "content": "Solve the task.",
                "source": "user",
            },
            {
                "class_": "text_observation",
                "content": "List the files.",
                "source": "user",
            },
            {
                "class_": "api_action",
                "tool_call_id": "tc1",
                "function": "execute",
                "kwargs": {},
                "description": "ls",
                "reasoning_content": "I should run ls.",
            },
            {
                "class_": "text_observation",
                "tool_call_id": "tc1",
                "content": "README.md",
                "source": "environment",
            },
            {
                "class_": "message_action",
                "content": "One file: README.md.",
                "reward": 1.0,
            },
        ],
        "details": {"task_id": "demo/list-files", "environment": "demo"},
    }


def test_no_reward_leaves_content_untouched():
    rec = trajectory_to_adp_record(trajectory_id="r", events=_sample_events())
    assert all("reward" not in item for item in rec["content"])
    assert rec["details"] == {}


def test_reward_lands_on_last_action_not_observation():
    events = [
        {"type": "agent_message", "text": "starting"},
        {
            "type": "tool_call",
            "tool_call_id": "tc1",
            "kind": "execute",
            "title": "ls",
            "status": "completed",
            "content": [{"text": "out"}],
        },
    ]
    rec = trajectory_to_adp_record(trajectory_id="r", events=events, reward=0.5)
    api_action, observation = rec["content"][-2:]
    assert api_action["class_"] == "api_action"
    assert api_action["reward"] == 0.5
    assert "reward" not in observation


def test_every_tool_action_pairs_with_environment_observation():
    events = [
        {
            "type": "tool_call",
            "tool_call_id": "",
            "kind": "execute",
            "title": "touch a",
            "status": "completed",
            "content": [],
        },
        {
            "type": "tool_call",
            "tool_call_id": "dup",
            "kind": "read",
            "title": "cat a",
            "status": "completed",
            "content": [{"text": "x"}],
        },
        {
            "type": "tool_call",
            "tool_call_id": "dup",
            "kind": "read",
            "title": "cat b",
            "status": "completed",
            "content": [{"text": "y"}],
        },
    ]
    content = acp_events_to_adp_content(events)
    actions = [c for c in content if c["class_"] == "api_action"]
    observations = [c for c in content if c["class_"] == "text_observation"]
    assert [a["tool_call_id"] for a in actions] == ["call_000001", "dup", "call_000002"]
    assert [o["tool_call_id"] for o in observations] == [
        "call_000001",
        "dup",
        "call_000002",
    ]
    # Empty output still yields the matched observation ADP requires.
    assert observations[0]["content"] == ""
    assert all(o["source"] == "environment" for o in observations)


def test_thought_before_user_message_flushes_in_order():
    content = acp_events_to_adp_content(
        [
            {"type": "agent_thought", "text": "hmm"},
            {"type": "user_message", "text": "continue"},
        ]
    )
    assert content == [
        {"class_": "message_action", "content": "", "reasoning_content": "hmm"},
        {"class_": "text_observation", "content": "continue", "source": "user"},
    ]


def test_trailing_thoughts_join_into_standalone_action():
    content = acp_events_to_adp_content(
        [
            {"type": "agent_thought", "text": "first"},
            {"type": "agent_thought", "text": "second"},
        ]
    )
    assert content == [
        {
            "class_": "message_action",
            "content": "",
            "reasoning_content": "first\n\nsecond",
        }
    ]


def test_oracle_event_renders_command():
    content = acp_events_to_adp_content([{"type": "oracle", "command": "bash run.sh"}])
    assert content == [{"class_": "message_action", "content": "[oracle: bash run.sh]"}]


def test_empty_and_malformed_events_skipped():
    content = acp_events_to_adp_content(
        [
            {"type": "agent_message", "text": ""},
            "not-a-dict",
            {"type": "unknown_event", "text": "x"},
            {"type": "user_message", "text": "real"},
        ]
    )
    assert content == [
        {"class_": "text_observation", "content": "real", "source": "user"}
    ]


def test_write_rollout_adp_jsonl(tmp_path):
    rec = write_rollout_adp_jsonl(
        tmp_path,
        trajectory_id="rollout-1",
        task_id="demo/list-files",
        prompts=["Solve the task."],
        trajectory=_sample_events(),
        model="claude-haiku-4-5",
        environment="demo",
        reward=1.0,
    )
    out = tmp_path / "trainer" / "adp.jsonl"
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == rec
    assert rec["details"] == {
        "task_id": "demo/list-files",
        "environment": "demo",
        "model": "claude-haiku-4-5",
    }


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
    rec = write_rollout_adp_jsonl(
        tmp_path,
        trajectory_id="r",
        task_id="t",
        prompts=None,
        trajectory=events,
        model=None,
        environment="demo",
    )
    raw = (tmp_path / "trainer" / "adp.jsonl").read_text()
    assert "sk-abc123def456ghi789" not in raw
    observation = rec["content"][-1]
    assert "***REDACTED***" in observation["content"]


def test_write_job_adp_jsonl_aggregates_rollouts(tmp_path):
    for i in range(2):
        write_rollout_adp_jsonl(
            tmp_path / f"rollout-{i}",
            trajectory_id=f"rollout-{i}",
            task_id=f"t{i}",
            prompts=None,
            trajectory=[{"type": "agent_message", "text": f"hi {i}"}],
            model="m",
            environment="demo",
        )
    out = write_job_adp_jsonl(tmp_path)
    assert out == tmp_path / "adp.jsonl"
    lines = out.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["rollout-0", "rollout-1"]


def test_write_job_adp_jsonl_returns_none_without_rollouts(tmp_path):
    assert write_job_adp_jsonl(tmp_path) is None
    assert write_job_adp_jsonl(tmp_path / "missing") is None


def test_write_skips_empty_trajectory(tmp_path):
    # No events and no prompts -> empty content, no score -> no artifact, mirror
    # ATIF's empty-record contract so the job aggregate never sees content==[].
    for prompts in (None, [""]):
        rec = write_rollout_adp_jsonl(
            tmp_path,
            trajectory_id="empty",
            task_id="t",
            prompts=prompts,
            trajectory=[],
            model=None,
            environment="demo",
        )
        assert rec is None
        assert not (tmp_path / "trainer" / "adp.jsonl").exists()


def test_write_job_adp_jsonl_excludes_empty_rollout(tmp_path):
    write_rollout_adp_jsonl(
        tmp_path / "real",
        trajectory_id="real",
        task_id="t",
        prompts=None,
        trajectory=[{"type": "agent_message", "text": "did the work"}],
        model="m",
        environment="demo",
    )
    write_rollout_adp_jsonl(
        tmp_path / "empty",
        trajectory_id="empty",
        task_id="t",
        prompts=None,
        trajectory=[],
        model="m",
        environment="demo",
    )
    out = write_job_adp_jsonl(tmp_path)
    ids = [json.loads(line)["id"] for line in out.read_text().splitlines()]
    assert ids == ["real"]


def test_reward_with_no_action_is_surfaced(caplog):
    # An agent that crashes after consuming the prompt (no action) but is still
    # scored: the reward must not vanish — it lands in details.terminal_reward.
    import logging

    for reward in (0.0, 1.0):
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            rec = trajectory_to_adp_record(
                trajectory_id="crashed",
                events=[],
                prompts=["do the task"],
                reward=reward,
            )
        # No action carries a reward...
        assert all("reward" not in item for item in rec["content"])
        # ...so the exact score is surfaced in details, with a warning.
        assert rec["details"]["terminal_reward"] == reward
        assert any("terminal reward" in r.message for r in caplog.records)


def test_write_path_no_action_keeps_score(tmp_path):
    # The write seam must still emit a scored crash rollout (content empty but a
    # terminal reward present), carrying the score in details.terminal_reward.
    rec = write_rollout_adp_jsonl(
        tmp_path,
        trajectory_id="crashed",
        task_id="t",
        prompts=None,
        trajectory=[],
        model="m",
        environment="demo",
        reward=1.0,
    )
    assert rec is not None
    assert rec["details"]["terminal_reward"] == 1.0
    line = json.loads((tmp_path / "trainer" / "adp.jsonl").read_text().strip())
    assert line["details"]["terminal_reward"] == 1.0
