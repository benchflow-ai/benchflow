"""The native harness's parsers and command builders, on the pinned CLIs' own output.

The samples under ``tests/fixtures/native_harness/<cli>-<version>/`` were
recorded from the pinned CLIs against the deterministic fake model with
``record_samples.py`` (re-record them when a pin moves). Each test replays a
sample through the parser into an ``ACPSession``, the object the ACP path
records into, and checks the trajectory the rollout would write.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.acp.session import ACPSession
from benchflow.acp.types import McpServerSpec, StopReason
from benchflow.agents.codex_config import codex_config_overrides
from benchflow.agents.registry import pinned_npm_package
from benchflow.native_harness.claude_code import (
    ClaudeCodeParser,
    claude_code_launch,
    claude_code_mcp_config,
    tool_info,
    tool_result_update,
)
from benchflow.native_harness.codex import (
    CodexExecParser,
    codex_launch,
    codex_mcp_overrides,
    unwrap_shell,
)
from benchflow.native_harness.spec import NativeTurn
from benchflow.trajectories._capture import _capture_session_trajectory

FIXTURES = Path(__file__).parent / "fixtures" / "native_harness"
CLAUDE = FIXTURES / f"claude-code-{pinned_npm_package('claude-code')[1]}"
CODEX = FIXTURES / f"codex-{pinned_npm_package('codex')[1]}"
_VOLATILE = {"receipt", "ts", "started_at", "finished_at"}


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _replay(parser, path: Path) -> tuple[list[dict], object]:
    session = ACPSession("s")
    for event in _events(path):
        for update in parser.feed(event):
            session.handle_update(update)
    session.mark_prompt_end()
    trajectory = [
        {k: v for k, v in e.items() if k not in _VOLATILE}
        for e in _capture_session_trajectory(session)
    ]
    return trajectory, parser.outcome()


def test_samples_exist_for_the_pinned_versions():
    """A pin bump without re-recorded samples fails here, not in a rollout."""
    for directory in (CLAUDE, CODEX):
        assert sorted(p.name for p in directory.glob("*.jsonl")) == [
            "cancelled.jsonl",
            "resumed.jsonl",
            "tool-error.jsonl",
            "turn.jsonl",
        ], directory


# ---------------------------------------------------------------------------
# Claude Code stream-json
# ---------------------------------------------------------------------------


def test_claude_turn_maps_like_the_acp_adapter():
    trajectory, outcome = _replay(ClaudeCodeParser("/work"), CLAUDE / "turn.jsonl")
    assert trajectory == [
        {"type": "agent_message", "text": "Writing hello.txt."},
        {
            "type": "tool_call",
            "tool_call_id": "toolu_fake_hello_0",
            "kind": "execute",
            "title": "printf 'Hello, world!\\n' > hello.txt && echo wrote",
            "status": "completed",
            "content": [
                {"type": "content", "content": {"type": "text", "text": "Write hello.txt"}},
                {
                    "type": "content",
                    "content": {"type": "text", "text": "```console\nwrote\n```"},
                },
            ],
            "raw_input": {
                "command": "printf 'Hello, world!\\n' > hello.txt && echo wrote",
                "description": "Write hello.txt",
            },
            "raw_output": "wrote",
            "tool_name": "Bash",
        },
        {"type": "agent_message", "text": "Created hello.txt."},
    ]
    assert outcome.stop_reason is StopReason.END_TURN
    assert outcome.completed and outcome.error is None
    assert outcome.usage == {
        "input_tokens": 2000,
        "output_tokens": 100,
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
        "total_tokens": 2100,
    }
    assert outcome.cost_usd == pytest.approx(0.0025)


def test_claude_streamed_text_is_not_recorded_twice():
    """--include-partial-messages streams the text the assistant message repeats."""
    parser = ClaudeCodeParser("/work")
    texts = []
    for event in _events(CLAUDE / "turn.jsonl"):
        for update in parser.feed(event):
            if update["sessionUpdate"] == "agent_message_chunk":
                texts.append(update["content"]["text"])
    assert texts == ["Writing hello.txt.", "Created hello.txt."]


def test_claude_resumed_turn_reports_its_own_usage():
    """result.usage is per invocation; modelUsage and total_cost_usd are cumulative."""
    trajectory, outcome = _replay(ClaudeCodeParser("/work"), CLAUDE / "resumed.jsonl")
    call = trajectory[1]
    assert call["title"] == "echo again >> hello.txt"
    assert call["raw_output"] == "(Bash completed with no output)"
    # No output: the adapter adds no content beyond the description.
    assert call["content"] == [
        {"type": "content", "content": {"type": "text", "text": "Append a line"}}
    ]
    assert outcome.usage["input_tokens"] == 2000
    assert outcome.cost_usd == pytest.approx(0.005)


def test_claude_tool_error_is_a_failed_call_with_fenced_output():
    trajectory, outcome = _replay(ClaudeCodeParser("/work"), CLAUDE / "tool-error.jsonl")
    call = trajectory[1]
    assert call["status"] == "failed"
    assert call["content"][1]["content"]["text"].startswith("```\nExit code 2\n")
    assert outcome.stop_reason is StopReason.END_TURN


def test_claude_sigint_result_is_an_error_the_client_turns_into_cancelled():
    trajectory, outcome = _replay(ClaudeCodeParser("/work"), CLAUDE / "cancelled.jsonl")
    assert outcome.completed and outcome.stop_reason is None
    assert outcome.error and "stop_reason=tool_use" in outcome.error
    assert trajectory[1]["status"] == "failed"


def test_claude_init_event_names_the_pinned_cli():
    parser = ClaudeCodeParser("/work")
    for event in _events(CLAUDE / "turn.jsonl"):
        parser.feed(event)
    assert parser.init is not None
    assert parser.init["claude_code_version"] == pinned_npm_package("claude-code")[1]
    assert parser.session_id == "00000000-0000-4000-8000-000000000000"


def _result(**fields) -> dict:
    return {"type": "result", "session_id": "s", "usage": {}, **fields}


@pytest.mark.parametrize(
    ("result", "stop", "error"),
    [
        (_result(subtype="success", stop_reason="end_turn"), StopReason.END_TURN, None),
        (_result(subtype="success", stop_reason="max_tokens"), StopReason.MAX_TOKENS, None),
        (_result(subtype="success", stop_reason="refusal"), StopReason.REFUSAL, None),
        (_result(subtype="error_max_turns"), StopReason.MAX_TURN_REQUESTS, None),
        (
            _result(subtype="success", is_error=True, result="API Error: 429"),
            None,
            "API Error: 429",
        ),
        (
            _result(
                subtype="error_during_execution",
                is_error=True,
                errors=["boom"],
                api_error_status=529,
            ),
            None,
            "boom (HTTP 529)",
        ),
    ],
)
def test_claude_result_maps_to_the_adapters_stop_reasons(result, stop, error):
    parser = ClaudeCodeParser()
    parser.feed(result)
    outcome = parser.outcome()
    assert (outcome.stop_reason, outcome.error) == (stop, error)


def test_claude_replayed_turn_answers_on_the_result_alone():
    """Adapter #453: a cache-replayed turn streams nothing but its result."""
    parser = ClaudeCodeParser()
    updates = parser.feed(
        _result(subtype="success", result="cached answer", usage={"output_tokens": 0})
    )
    assert updates == [
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "cached answer"},
        }
    ]


def test_claude_subagent_updates_carry_their_parent():
    parser = ClaudeCodeParser()
    updates = parser.feed(
        {
            "type": "assistant",
            "parent_tool_use_id": "toolu_parent",
            "message": {
                "id": "m1",
                "content": [
                    {"type": "text", "text": "child says"},
                    {"type": "tool_use", "id": "toolu_child", "name": "Read", "input": {}},
                ],
            },
        }
    )
    assert [u["_meta"]["claudeCode"]["parentToolUseId"] for u in updates] == [
        "toolu_parent",
        "toolu_parent",
    ]


def test_claude_plan_tools_are_not_tool_calls():
    """The adapter renders TodoWrite/Task* as ACP plans, which BenchFlow drops."""
    parser = ClaudeCodeParser()
    todo = {"type": "tool_use", "id": "t1", "name": "TodoWrite", "input": {"todos": []}}
    assert parser.feed({"type": "assistant", "message": {"id": "m", "content": [todo]}}) == []
    result = {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
    assert parser.feed({"type": "user", "message": {"content": [result]}}) == []


@pytest.mark.parametrize(
    ("name", "tool_input", "title", "kind"),
    [
        ("Bash", {"command": "ls"}, "ls", "execute"),
        ("Bash", {}, "Terminal", "execute"),
        ("Read", {"file_path": "/work/a.py", "offset": 3, "limit": 2}, "Read a.py (3 - 4)", "read"),
        ("Read", {"file_path": "/etc/x"}, "Read /etc/x", "read"),
        ("Write", {"file_path": "/work/b.txt", "content": "x"}, "Write b.txt", "edit"),
        ("Write", {}, "Preparing file…", "edit"),
        ("Edit", {"file_path": "/work/c.py", "old_string": "a", "new_string": "b"}, "Edit c.py", "edit"),
        ("Glob", {"pattern": "*.py", "path": "src"}, "Find `src` `*.py`", "search"),
        ("Grep", {"pattern": "foo", "-i": True, "output_mode": "count"}, 'grep -i -c "foo"', "search"),
        ("WebFetch", {"url": "https://x"}, "Fetch https://x", "fetch"),
        ("WebSearch", {"query": "q"}, 'Search "q"', "fetch"),
        ("Agent", {"description": "Explore", "prompt": "go"}, "Explore", "think"),
        ("Skill", {"skill": "pdf"}, "Load skill: pdf", "other"),
        ("mcp__srv__tool", {}, "mcp__srv__tool", "other"),
    ],
)
def test_tool_titles_and_kinds_follow_the_adapter(name, tool_input, title, kind):
    info = tool_info(name, tool_input, "/work")
    assert (info["title"], info["kind"]) == (title, kind)


def test_edit_result_diff_comes_from_the_structured_patch():
    """The adapter's PostToolUse hook diff, from the message-level tool_use_result."""
    update = tool_result_update(
        "Edit",
        {"file_path": "/work/c.py"},
        {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
        {
            "filePath": "/work/c.py",
            "structuredPatch": [
                {"oldStart": 1, "newStart": 1, "oldLines": 1, "newLines": 1, "lines": ["-a", "+b"]}
            ],
        },
    )
    assert update == {
        "content": [{"type": "diff", "path": "/work/c.py", "oldText": "a", "newText": "b"}]
    }


def test_read_result_is_line_numbered_from_the_structured_file():
    update = tool_result_update(
        "Read",
        {"file_path": "/work/a"},
        {"type": "tool_result", "tool_use_id": "t", "content": "1\tx"},
        {"type": "text", "file": {"content": "x\ny\n", "startLine": 5}},
    )
    assert update["content"][0]["content"]["text"] == "```\n5\tx\n6\ty\n```"


def test_agent_result_strips_the_model_directed_trailer():
    raw = "the report\nagentId: abc-1 (use SendMessage to continue)\n<usage>tokens</usage>"
    update = tool_result_update(
        "Agent", {}, {"type": "tool_result", "tool_use_id": "t", "content": raw}, None
    )
    assert update["content"][0]["content"]["text"] == "the report"


# ---------------------------------------------------------------------------
# Claude Code command builder
# ---------------------------------------------------------------------------


def test_claude_first_turn_names_its_session_and_streams_partial_messages():
    argv = claude_code_launch(NativeTurn(cwd="/app", new_session_id="u-1")).argv
    assert argv[:5] == ("-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages")
    assert "--forward-subagent-text" in argv
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert argv[argv.index("--session-id") + 1] == "u-1"
    assert "--resume" not in argv and "--model" not in argv


def test_claude_later_turns_resume_and_pass_model_and_effort():
    argv = claude_code_launch(
        NativeTurn(cwd="/app", resume_id="u-1", model="claude-haiku-4-5", reasoning_effort="high")
    ).argv
    assert argv[argv.index("--resume") + 1] == "u-1"
    assert "--session-id" not in argv
    assert argv[argv.index("--model") + 1] == "claude-haiku-4-5"
    assert argv[argv.index("--effort") + 1] == "high"


def test_claude_mcp_servers_become_one_mcp_config_document():
    servers = (
        McpServerSpec(name="fs", command="npx", args=["srv"], env={"K": "v"}),
        McpServerSpec(name="web", type="http", url="http://h/mcp", headers={"A": "b"}),
    )
    argv = claude_code_launch(NativeTurn(cwd="/app", mcp_servers=servers)).argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == claude_code_mcp_config(servers)
    assert claude_code_mcp_config(servers) == {
        "mcpServers": {
            "fs": {"type": "stdio", "command": "npx", "args": ["srv"], "env": {"K": "v"}},
            "web": {"type": "http", "url": "http://h/mcp", "headers": {"A": "b"}},
        }
    }


# ---------------------------------------------------------------------------
# Codex exec --json
# ---------------------------------------------------------------------------


def test_codex_turn_maps_items_to_tool_calls_and_messages():
    trajectory, outcome = _replay(CodexExecParser("/work", turn=1), CODEX / "turn.jsonl")
    assert [e["type"] for e in trajectory] == ["agent_message", "tool_call", "agent_message"]
    call = trajectory[1]
    assert call["tool_call_id"] == "turn1-item_1"
    assert call["kind"] == "execute" and call["status"] == "completed"
    # codex-acp 1.13.1's shape: the model's command (exec reports it wrapped
    # in /bin/bash -lc), a terminal reference, {formatted_output, exit_code}.
    assert call["title"] == "printf 'Hello, world!\\n' > hello.txt && echo wrote"
    assert call["raw_input"] == {"command": call["title"], "cwd": "/work"}
    assert call["raw_output"] == {"formatted_output": "wrote\n", "exit_code": 0}
    assert call["content"] == [{"type": "terminal", "terminalId": "turn1-item_1"}]
    assert outcome.stop_reason is StopReason.END_TURN
    assert outcome.usage is None
    assert outcome.usage_total["input_tokens"] == 2000


def test_codex_resumed_turn_reports_the_thread_total():
    """exec prints the thread's running total; the client takes the difference."""
    _, outcome = _replay(CodexExecParser("/work", turn=2), CODEX / "resumed.jsonl")
    assert outcome.usage_total["input_tokens"] == 4000
    assert outcome.session_id == "00000000-0000-4000-8000-000000000000"


def test_codex_item_ids_are_unique_across_turns():
    """exec restarts item ids at item_0 in every process."""
    first = CodexExecParser("/w", turn=1).feed(_events(CODEX / "turn.jsonl")[3])
    second = CodexExecParser("/w", turn=2).feed(_events(CODEX / "resumed.jsonl")[3])
    assert first[0]["toolCallId"] != second[0]["toolCallId"]


def test_codex_failed_command_is_a_failed_call():
    trajectory, _ = _replay(CodexExecParser("/work"), CODEX / "tool-error.jsonl")
    assert trajectory[1]["status"] == "failed"
    assert trajectory[1]["raw_output"]["exit_code"] == 2


def test_codex_interrupted_turn_has_no_outcome_and_a_pending_call():
    trajectory, outcome = _replay(CodexExecParser("/work"), CODEX / "cancelled.jsonl")
    assert not outcome.completed and outcome.error is None
    assert trajectory[-1]["status"] == "in_progress"


def test_codex_turn_failed_and_stream_errors_are_errors():
    parser = CodexExecParser()
    parser.feed({"type": "error", "message": "stream disconnected"})
    assert parser.outcome().error == "stream disconnected"
    parser.feed({"type": "turn.failed", "error": {"message": "unexpected status 401"}})
    outcome = parser.outcome()
    assert outcome.completed and outcome.error == "unexpected status 401"


def test_codex_file_changes_arrive_completed():
    updates = CodexExecParser(turn=3).feed(
        {
            "type": "item.completed",
            "item": {
                "id": "item_4",
                "type": "file_change",
                "changes": [{"path": "/app/a.py", "kind": "update"}],
                "status": "completed",
            },
        }
    )
    assert updates[0]["sessionUpdate"] == "tool_call"
    assert updates[0]["status"] == "completed" and updates[0]["title"] == "Edit /app/a.py"


# ---------------------------------------------------------------------------
# Codex command builder
# ---------------------------------------------------------------------------

_CONFIG = {
    "model_provider": "benchflow-litellm",
    "model": "gpt-5.4",
    "model_providers": {
        "benchflow-litellm": {
            "name": "litellm",
            "base_url": "http://127.0.0.1:4000/v1",
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "supports_websockets": False,
        }
    },
    "web_search": "disabled",
}


def test_codex_exec_takes_every_setting_from_codex_config():
    argv = list(codex_launch(NativeTurn(cwd="/app"), _CONFIG).argv)
    assert argv[:2] == ["exec", "--json"]
    assert "--ignore-user-config" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert argv[-1] == "-"
    overrides = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    assert 'model_provider="benchflow-litellm"' in overrides
    assert 'model_providers.benchflow-litellm.base_url="http://127.0.0.1:4000/v1"' in overrides
    assert 'web_search="disabled"' in overrides
    # As codex-acp asks for reasoning summaries on every turn.
    assert 'model_reasoning_summary="auto"' in overrides
    # The built-in provider points at the gateway too.
    assert 'openai_base_url="http://127.0.0.1:4000/v1"' in overrides
    # Nothing phones home: plugin marketplace sync and analytics are off.
    assert {"features.plugins=false", "analytics.enabled=false"} <= set(overrides)


@pytest.mark.parametrize(
    ("reported", "script"),
    [
        ("/bin/bash -lc 'ls -la'", "ls -la"),
        ("/bin/bash -lc \"printf 'a\"'!'\"'\"", "printf 'a!'"),
        ("bash -c 'x'", "x"),
        ("python3 -c 'print(1)'", "python3 -c 'print(1)'"),
        ("unbalanced 'quote", "unbalanced 'quote"),
    ],
)
def test_exec_commands_are_unwrapped_from_their_shell(reported, script):
    assert unwrap_shell(reported) == script


def test_codex_resume_continues_the_thread():
    argv = codex_launch(NativeTurn(cwd="/app", resume_id="t-1", reasoning_effort="high"), _CONFIG).argv
    assert argv[:3] == ("exec", "resume", "t-1")
    assert "model_reasoning_effort=\"high\"" in argv


def test_codex_mcp_servers_become_config_tables():
    servers = (McpServerSpec(name="fs", command="npx", args=["srv"], tools=["read"]),)
    assert codex_mcp_overrides(servers) == {
        "mcp_servers": {"fs": {"command": "npx", "args": ["srv"], "enabled_tools": ["read"]}}
    }


def test_codex_config_overrides_refuse_what_they_cannot_write():
    with pytest.raises(ValueError, match="bare key"):
        codex_config_overrides({"model_providers": {"a.b": {"base_url": "x"}}})
    with pytest.raises(ValueError, match="TOML"):
        codex_config_overrides({"model": None})
    assert codex_config_overrides({"a": {"b": [1, 2]}, "c": True}) == ["a.b=[1, 2]", "c=true"]
