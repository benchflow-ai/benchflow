"""Native host CLI accounting and launch contracts."""

import json

from benchflow.robotics.host import command_line, parse_events


def test_codex_usage_comes_from_completed_turns_only(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Observed both images"},
                },
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 9,
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 50,
                        "cached_input_tokens": 10,
                        "output_tokens": 7,
                    },
                },
            ]
        )
    )
    result = parse_events(events, "codex")
    assert result["completed"] and not result["failed"]
    assert result["usage"] == {
        "input_tokens": 150,
        "cached_input_tokens": 50,
        "output_tokens": 16,
    }
    assert result["native_cost_usd"] is None


def test_claude_error_result_preserves_native_cost(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "result",
                "is_error": True,
                "usage": {"input_tokens": 12},
                "total_cost_usd": 0.001,
            }
        )
    )
    result = parse_events(events, "claude")
    assert result["failed"] and result["native_cost_usd"] == 0.001
    assert result["usage"]["input_tokens"] == 12


def test_started_thread_without_completion_is_not_success(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text('{"type":"thread.started"}\npartial')
    assert not parse_events(events, "codex")["completed"]


def test_host_launches_fresh_exact_model_sessions(tmp_path):
    codex = command_line("codex", "gpt-6-astra", "max", tmp_path)
    claude = command_line("claude", "claude-fable-5-1", "max", tmp_path)
    assert "--ephemeral" in codex and "--ignore-user-config" in codex
    assert codex[codex.index("--model") + 1] == "gpt-6-astra"
    assert "--no-session-persistence" in claude and "--safe-mode" in claude
    assert claude[claude.index("--model") + 1] == "claude-fable-5-1"
    assert "resume" not in codex and "--resume" not in claude
    assert "--dangerously-bypass-approvals-and-sandbox" not in codex
    assert "--dangerously-skip-permissions" not in claude
