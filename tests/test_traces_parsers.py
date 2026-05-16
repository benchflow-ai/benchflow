"""Tests for benchflow.traces.parsers — Claude Code + opentraces parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.traces.parsers import (
    parse_claude_code_session,
    parse_opentraces_file,
    parse_opentraces_record,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def claude_session_file(tmp_path: Path) -> Path:
    """Create a minimal Claude Code JSONL session file."""
    entries = [
        {
            "type": "user",
            "sessionId": "sess-001",
            "timestamp": "2026-01-15T10:00:00Z",
            "cwd": "/home/user/my-project",
            "gitBranch": "main",
            "message": {
                "role": "user",
                "content": "Create a hello.txt file with the text 'Hello World'",
            },
        },
        {
            "type": "assistant",
            "sessionId": "sess-001",
            "timestamp": "2026-01-15T10:00:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "I'll create that file for you."},
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": "/home/user/my-project/hello.txt",
                            "content": "Hello World",
                        },
                    },
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        },
        {
            "type": "assistant",
            "sessionId": "sess-001",
            "timestamp": "2026-01-15T10:00:10Z",
            "message": {
                "role": "assistant",
                "content": "Done! I've created hello.txt with the text 'Hello World'.",
                "usage": {"input_tokens": 80, "output_tokens": 30},
            },
        },
    ]
    path = tmp_path / "sess-001.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries))
    return path


@pytest.fixture()
def opentraces_record() -> dict:
    """Minimal opentraces TraceRecord dict."""
    return {
        "schema_version": "0.3.0",
        "trace_id": "trace-abc-123",
        "session_id": "sess-ot-001",
        "timestamp_start": "2026-01-15T09:00:00Z",
        "timestamp_end": "2026-01-15T09:05:00Z",
        "agent": {"name": "claude-code", "version": "1.0.32"},
        "environment": {
            "cwd": "/home/user/project",
            "vcs": {"branch": "feature-x", "remote": "https://github.com/user/project"},
        },
        "task": {"tags": ["refactor", "python"]},
        "steps": [
            {
                "thought": "I need to refactor the utils module.",
                "action": {
                    "tool_call": {
                        "name": "Edit",
                        "input": {"file_path": "src/utils.py"},
                    }
                },
                "observation": {"content": "File edited successfully."},
            },
            {
                "thought": "Now I should run the tests.",
                "action": {
                    "tool_call": {
                        "name": "Bash",
                        "input": {"command": "pytest"},
                    }
                },
                "observation": "All tests passed.",
            },
        ],
        "outcome": {"status": "success"},
        "metrics": {"tokens": {"input": 500, "output": 200}},
    }


@pytest.fixture()
def opentraces_file(tmp_path: Path, opentraces_record: dict) -> Path:
    """Create a JSONL file with opentraces records."""
    path = tmp_path / "traces.jsonl"
    lines = [json.dumps(opentraces_record)]
    # Add a second record with different trace_id
    rec2 = {**opentraces_record, "trace_id": "trace-def-456", "session_id": "sess-ot-002"}
    lines.append(json.dumps(rec2))
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Claude Code parser tests
# ---------------------------------------------------------------------------


class TestClaudeCodeParser:
    def test_parse_basic_session(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.session_id == "sess-001"
        assert trace.agent_name == "claude-code"
        assert trace.model == "claude-sonnet-4-20250514"
        assert trace.cwd == "/home/user/my-project"
        assert trace.git.branch == "main"

    def test_extracts_user_prompt(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.first_user_prompt == "Create a hello.txt file with the text 'Hello World'"

    def test_extracts_tool_calls(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.n_tool_calls == 1
        assert trace.tool_names_used == ["Write"]

    def test_extracts_files_edited(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.files_edited == ["/home/user/my-project/hello.txt"]

    def test_infers_outcome(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.outcome == "success"

    def test_token_usage(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.total_input_tokens == 180
        assert trace.total_output_tokens == 80

    def test_timestamps(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        assert trace.started_at is not None
        assert trace.ended_at is not None
        assert trace.duration_sec == 10.0

    def test_steps_count(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file)

        # 1 user + 2 assistant = 3 steps
        assert len(trace.steps) == 3

    def test_override_session_id(self, claude_session_file: Path) -> None:
        trace = parse_claude_code_session(claude_session_file, session_id="custom-id")

        assert trace.session_id == "custom-id"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_claude_code_session(tmp_path / "nonexistent.jsonl")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No valid entries"):
            parse_claude_code_session(path)

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON lines are skipped, valid ones are parsed."""
        entries = [
            "not valid json",
            json.dumps(
                {
                    "type": "user",
                    "sessionId": "s1",
                    "message": {"role": "user", "content": "hello"},
                }
            ),
        ]
        path = tmp_path / "partial.jsonl"
        path.write_text("\n".join(entries))

        trace = parse_claude_code_session(path)
        assert len(trace.steps) == 1


# ---------------------------------------------------------------------------
# opentraces parser tests
# ---------------------------------------------------------------------------


class TestOpentracesParser:
    def test_parse_record(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert trace.trace_id == "trace-abc-123"
        assert trace.session_id == "sess-ot-001"
        assert trace.agent_name == "claude-code"
        assert trace.outcome == "success"
        assert trace.cwd == "/home/user/project"

    def test_git_context(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert trace.git.branch == "feature-x"
        assert trace.git.repo == "https://github.com/user/project"

    def test_timestamps(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert trace.started_at is not None
        assert trace.ended_at is not None
        assert trace.duration_sec == 300.0

    def test_tags(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert "refactor" in trace.tags
        assert "python" in trace.tags

    def test_token_metrics(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert trace.total_input_tokens == 500
        assert trace.total_output_tokens == 200

    def test_tao_steps_parsed(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        # Each TAO step produces a thought step + an action/observation step
        assert len(trace.steps) >= 2

    def test_tool_calls_extracted(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert "Edit" in trace.tool_names_used
        assert "Bash" in trace.tool_names_used

    def test_metadata_includes_version(self, opentraces_record: dict) -> None:
        trace = parse_opentraces_record(opentraces_record)

        assert trace.metadata.get("agent_version") == "1.0.32"
        assert trace.metadata.get("opentraces_schema_version") == "0.3.0"

    def test_parse_file(self, opentraces_file: Path) -> None:
        traces = parse_opentraces_file(opentraces_file)

        assert len(traces) == 2
        assert traces[0].trace_id == "trace-abc-123"
        assert traces[1].trace_id == "trace-def-456"

    def test_minimal_record(self) -> None:
        """Parse a record with only required fields."""
        record = {
            "trace_id": "min-001",
            "session_id": "s-001",
            "agent": {"name": "test-agent"},
        }
        trace = parse_opentraces_record(record)

        assert trace.trace_id == "min-001"
        assert trace.agent_name == "test-agent"
        assert trace.outcome == "unknown"
        assert len(trace.steps) == 0
