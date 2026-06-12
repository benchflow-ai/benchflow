"""Trajectory secret-redaction regressions."""

from __future__ import annotations

from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_text,
)


def test_redact_trajectory_text_removes_token_shapes() -> None:
    """Guards PR #684: trajectory text must not retain live token prefixes."""
    raw = (
        "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz "
        "authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz "
        "x-api-key: AIzaSyabcdefghijklmnopqrstuvwxyz123456"
    )

    redacted = redact_trajectory_text(raw)

    assert "sk-proj-" not in redacted
    assert "ghp_" not in redacted
    assert "AIzaSy" not in redacted
    assert redacted.count("***REDACTED***") == 3


def test_redact_acp_trajectory_jsonl_scrubs_serialized_events() -> None:
    """Guards PR #684: hosted/acp trajectory JSONL is redacted after encoding."""
    jsonl = redact_acp_trajectory_jsonl(
        [
            {
                "type": "tool_call",
                "content": [
                    {
                        "text": (
                            "curl -H 'Authorization: Bearer "
                            "sk-ant-abcdefghijklmnopqrstuvwxyz'"
                        )
                    }
                ],
            }
        ]
    )

    assert "sk-ant-" not in jsonl
    assert "***REDACTED***" in jsonl
