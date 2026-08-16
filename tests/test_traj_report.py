"""Format-aware trajectory report tests for the upload preview."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchflow.publish.redact import REDACTED
from benchflow.publish.traj_capture import stage_trajectory_artifacts
from benchflow.publish.traj_report import (
    TrajectoryFormat,
    build_trajectory_report,
)


@dataclass(frozen=True)
class _Artifact:
    relname: str
    local_path: Path
    size_bytes: int
    created_at: datetime | None


def _artifact(
    path: Path, records: list[dict], *, relname: str = "capture.jsonl"
) -> _Artifact:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return _Artifact(
        relname=f"trajectory/{relname}",
        local_path=path,
        size_bytes=path.stat().st_size,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_report_prefers_acp_events_and_previews_redacted_content(
    tmp_path: Path,
) -> None:
    """Guards the trajectory-report follow-up to PR #992."""
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    (trajectory / "acp_trajectory.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {"type": "user_message", "text": "API_KEY=opaque-prefixless"},
                {"type": "agent_thought", "text": "Inspect the repository"},
                {"type": "tool_call", "kind": "read", "title": "Open README"},
                {"type": "agent_message", "text": "Done"},
            )
        )
    )
    (trajectory / "llm_trajectory.jsonl").write_text(
        json.dumps({"request": {}, "response": {}}) + "\n"
    )

    with stage_trajectory_artifacts(
        trajectory,
        source_id="report-demo",
    ) as staged:
        report = build_trajectory_report(
            staged.files,
            masked_values=staged.redaction_replacements,
            preview_steps=3,
        )

    assert report.primary_file == "trajectory/acp_trajectory.jsonl"
    assert report.format is TrajectoryFormat.BENCHFLOW_ACP
    assert report.file_count == 2
    assert report.total_steps == 4
    assert report.thinking_steps == 1
    assert report.tool_call_steps == 1
    assert report.human_steps == 1
    assert report.masked_values == 1
    assert len(report.preview) == 3
    assert REDACTED in report.preview[0].summary
    assert "opaque-prefixless" not in report.preview[0].summary


def test_report_understands_claude_thinking_tools_and_tool_results(
    tmp_path: Path,
) -> None:
    """Guards the trajectory-report follow-up to PR #992 for Claude JSONL."""
    artifact = _artifact(
        tmp_path / "claude.jsonl",
        [
            {
                "type": "user",
                "timestamp": "2026-02-03T04:05:06Z",
                "message": {"role": "user", "content": "Fix the tests"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-02-03T04:05:07Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Inspect first"},
                        {"type": "text", "text": "I will inspect the tests."},
                        {"type": "tool_use", "name": "Read", "input": {}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "test output"}],
                },
            },
        ],
    )

    report = build_trajectory_report((artifact,), masked_values=0)

    assert report.format is TrajectoryFormat.CLAUDE_CODE
    assert report.total_steps == 3
    assert report.thinking_steps == 1
    assert report.tool_call_steps == 1
    assert report.human_steps == 1
    assert report.created_at == datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
    assert report.preview[1].kind == "Thinking + tool"
    assert report.preview[2].kind == "Tool result"


def test_report_filters_codex_metadata_and_counts_response_items(
    tmp_path: Path,
) -> None:
    """Guards the trajectory-report follow-up to PR #992 for Codex JSONL."""
    artifact = _artifact(
        tmp_path / "codex.jsonl",
        [
            {
                "type": "session_meta",
                "timestamp": "2026-03-04T01:02:03Z",
                "payload": {"type": "session_meta"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "developer", "content": []},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Review this"}],
                },
            },
            {"type": "response_item", "payload": {"type": "reasoning"}},
            {
                "type": "response_item",
                "payload": {"type": "function_call", "name": "exec_command"},
            },
            {
                "type": "response_item",
                "payload": {"type": "function_call_output", "output": "ok"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Finished"}],
                },
            },
        ],
    )

    report = build_trajectory_report((artifact,), masked_values=0, preview_steps=4)

    assert report.format is TrajectoryFormat.CODEX
    assert report.total_steps == 5
    assert report.thinking_steps == 1
    assert report.tool_call_steps == 1
    assert report.human_steps == 1
    assert [step.kind for step in report.preview] == [
        "Human",
        "Thinking",
        "Tool call",
        "Tool result",
    ]


def test_report_counts_only_new_human_messages_in_llm_exchange_history(
    tmp_path: Path,
) -> None:
    """Guards the trajectory-report follow-up to PR #992 for LLM exchanges."""
    first_messages = [{"role": "user", "content": "First prompt"}]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second prompt"},
    ]
    artifact = _artifact(
        tmp_path / "llm.jsonl",
        [
            {
                "request": {"body": {"messages": first_messages}},
                "response": {"body": {"choices": [{"message": {"content": "A"}}]}},
            },
            {
                "request": {"body": {"messages": second_messages}},
                "response": {
                    "body": {
                        "reasoning": "think",
                        "choices": [
                            {"message": {"content": "B", "tool_calls": [{"id": "1"}]}}
                        ],
                    }
                },
            },
        ],
        relname="llm_trajectory.jsonl",
    )

    report = build_trajectory_report((artifact,), masked_values=0)

    assert report.format is TrajectoryFormat.LLM_EXCHANGE
    assert report.total_steps == 2
    assert report.human_steps == 2
    assert report.thinking_steps == 1
    assert report.tool_call_steps == 1


def test_report_skips_claude_metadata_and_previews_first_100_words(
    tmp_path: Path,
) -> None:
    """Guards PR #1008 against metadata labels replacing full step text."""
    prompt_words = [f"word-{index}" for index in range(1, 106)]
    artifact = _artifact(
        tmp_path / "claude-with-preamble.jsonl",
        [
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "content": "queue metadata",
            },
            {
                "type": "attachment",
                "attachment": {
                    "type": "skill_listing",
                    "addedLines": ["attachment metadata"],
                },
            },
            {
                "type": "user",
                "message": {"role": "user", "content": " ".join(prompt_words)},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Inspect the repository"},
                        {"type": "text", "text": "before choosing a tool"},
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "README.md"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": "Repository contents"}
                    ],
                },
            },
        ],
    )

    report = build_trajectory_report((artifact,), masked_values=0, preview_steps=4)

    assert report.format is TrajectoryFormat.CLAUDE_CODE
    assert report.total_steps == 4
    assert report.human_steps == 1
    assert [step.kind for step in report.preview] == [
        "Human",
        "Thinking",
        "Tool call",
        "Tool result",
    ]
    assert report.preview[0].summary == " ".join(prompt_words[:100]) + "…"
    assert report.preview[1].summary == (
        "Inspect the repository before choosing a tool"
    )
    assert report.preview[2].summary == 'Read: {"file_path": "README.md"}'
    assert report.preview[3].summary == "Repository contents"
    assert all("queue-operation" not in step.summary for step in report.preview)
    assert all("attachment" not in step.summary for step in report.preview)


def test_report_rejects_preview_limits_outside_the_public_cli_contract(
    tmp_path: Path,
) -> None:
    """Guards the trajectory-report follow-up to PR #992 preview bound."""
    artifact = _artifact(tmp_path / "generic.jsonl", [{"type": "message"}])

    with pytest.raises(ValueError, match="0-20"):
        build_trajectory_report((artifact,), masked_values=0, preview_steps=21)
