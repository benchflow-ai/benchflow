"""The ACP capture records a tool call's final title, not its first one.

claude-agent-acp titles a call before its input has streamed in ("Preparing file…", "Write",
"Edit", "Read File") and sends the real title, "<verb> <path>", in a later
``tool_call_update``. ACP says an update's fields replace the call's, but the
capture kept the first title, so every consumer of ``acp_trajectory.jsonl``
saw "Preparing file…" and the viewer had to rebuild the title.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import _events_to_trajectory
from benchflow.trajectories.viewer.payload import _build_acp_payload

_INPUT = {"file_path": "/app/hello.txt", "content": "Hello, world!"}


def _claude_write(session: ACPSession) -> None:
    """The update sequence of a claude-agent-acp Write."""
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "toolu_1",
            "title": "Preparing file…",
            "kind": "edit",
            "status": "pending",
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "toolu_1",
            "title": "Write /app/hello.txt",
            "rawInput": _INPUT,
        }
    )
    # Later updates with an empty title, or none, keep it.
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "toolu_1", "title": ""}
    )
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "toolu_1",
            "status": "completed",
        }
    )


def test_capture_keeps_the_final_title(monkeypatch) -> None:
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "off")
    session = ACPSession("s")
    _claude_write(session)

    assert session.tool_calls[0].title == "Write /app/hello.txt"
    [event] = _events_to_trajectory(session.events)
    assert event["title"] == "Write /app/hello.txt"
    assert event["raw_input"] == _INPUT
    assert event["status"] == "completed"
    # The live dashboard shows the final title too.
    assert session.progress_snapshot() == (1, "Write /app/hello.txt")
    assert session.distinct_tool_titles >= 1


def test_a_title_change_is_a_recorded_change(monkeypatch) -> None:
    """A retitle advances the call's receipt, so the streaming writer (which
    skips byte-identical snapshots) writes the new title."""
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "off")
    session = ACPSession("s")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "title": "Edit",
            "kind": "edit",
            "status": "in_progress",
        },
        _receipt={
            "source": "benchflow_host",
            "clock": "unix",
            "first_observed_ns": "1",
            "last_observed_ns": "1",
        },
    )
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "title": "Edit a.py"},
        _receipt={
            "source": "benchflow_host",
            "clock": "unix",
            "first_observed_ns": "2",
            "last_observed_ns": "2",
        },
    )

    [event] = _events_to_trajectory(session.events)
    assert event["title"] == "Edit a.py"
    assert event["receipt"]["last_observed_ns"] == "2"


def test_viewer_shows_the_captured_final_title_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """The viewer's rebuild of provisional titles (kept for trajectories
    captured before this fix) leaves a captured final title as it is."""
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "off")
    session = ACPSession("s")
    _claude_write(session)
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "toolu_2",
            "title": "Write notes.md",
            "kind": "edit",
            "status": "completed",
            "rawInput": {"file_path": "/app/elsewhere.md"},
        }
    )
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    (trajectory / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _events_to_trajectory(session.events))
    )
    (tmp_path / "result.json").write_text("{}")

    steps = _build_acp_payload(tmp_path, None).to_payload()["steps"]
    titles = [step["tool"]["title"] for step in steps if step["kind"] == "tool"]
    assert titles == ["Write /app/hello.txt", "Write notes.md"]
