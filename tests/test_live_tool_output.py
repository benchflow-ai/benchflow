"""Guards a report that a command printing tens of MB
stalled the runner.

codex-acp 2.x streams a running command's output as ``_meta.
terminal_output_delta`` chunks, so the tool call grows with every update and
each live write of ``acp_trajectory.jsonl`` serialized and redacted it whole:
the work grew with the square of the output (a 100 MB output outlasted 300 s
on 0.9). The live file now holds a note in place of a tool output longer
than ``_LIVE_OUTPUT_LIMIT`` characters; the final trajectory keeps it whole.
"""

from __future__ import annotations

import json

import pytest

from benchflow.acp.session import ACPSession
from benchflow.trajectories import _capture, _snapshot
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    make_trajectory_sink,
)

_CHUNK = ("0123456789abcdef" * 512)[:-1] + "\n"  # 8 KiB


def _command(call_id: str, chunks: list[str]) -> list[dict]:
    """A codex-acp 2.x command: the opening call, one update per output
    chunk, and the closing update with the exit code."""
    return [
        {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": "rg -n forbidden llm_trajectory.jsonl",
            "kind": "execute",
            "status": "in_progress",
            "rawInput": {"command": ["bash", "-lc", "rg ..."], "cwd": "/app"},
        },
        *(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "_meta": {"terminal_output_delta": {"terminal_id": call_id, "data": c}},
            }
            for c in chunks
        ),
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "completed",
            "_meta": {"terminal_exit": {"terminal_id": call_id, "exit_code": 0}},
        },
    ]


def _stream(path, updates) -> ACPSession:
    """Drive *updates* through the rollout's live sink (no event loop, so
    every update writes the live file)."""
    session = ACPSession("review")
    session.on_change = make_trajectory_sink(TrajectoryWriter(path), [])
    session.record_user_prompt("review the run")
    for update in updates:
        session.handle_update(update)
    session.mark_prompt_end()
    return session


@pytest.fixture
def redacted_chars(monkeypatch):
    """Characters handed to the live writer's redactor. Past 16 MiB it
    redacts nothing more (the session logs the failed write and goes on), so
    the quadratic case fails in seconds, not minutes."""
    count = [0]
    redact = _snapshot.redact_trajectory_obj

    def counting(obj):
        count[0] += len(json.dumps(obj))
        if count[0] > 16 << 20:
            raise AssertionError(f"{count[0]} characters redacted")
        return redact(obj)

    monkeypatch.setattr(_snapshot, "redact_trajectory_obj", counting)
    return count


def test_a_growing_command_output_is_not_redacted_whole_on_every_write(
    tmp_path, monkeypatch, redacted_chars
):
    monkeypatch.setattr(_capture, "_LIVE_OUTPUT_LIMIT", 64 * 1024, raising=False)
    path = tmp_path / "acp_trajectory.jsonl"

    session = _stream(path, _command("exec-big", [_CHUNK] * 512))  # 4 MiB

    assert redacted_chars[0] < 4 << 20
    live = path.read_text()
    assert len(live) < 64 * 1024
    assert f"[{512 * 8192} characters of tool output" in live
    final = _capture_session_trajectory(session)
    big = final[-1]
    assert big["tool_call_id"] == "exec-big" and big["status"] == "completed"
    assert big["raw_output"] == {
        "formatted_output": _CHUNK * 512,
        "exit_code": 0,
    }
    TrajectoryWriter(path).write_final(final)
    written = json.loads(path.read_text().splitlines()[-1])
    assert written["raw_output"]["formatted_output"] == _CHUNK * 512


def test_a_long_output_sent_whole_gives_way_to_the_note_live_only(tmp_path):
    path = tmp_path / "acp_trajectory.jsonl"
    output = "y" * ((1 << 20) + 1)
    updates = _command("exec-whole", [])
    updates[-1]["rawOutput"] = {"formatted_output": output, "exit_code": 0}

    session = _stream(path, updates)

    live = [json.loads(line) for line in path.read_text().splitlines()]
    assert live[-1]["raw_output"] == {
        "formatted_output": f"[{len(output)} characters of tool output are not in "
        "the live trajectory; the final trajectory has them]",
        "exit_code": 0,
    }
    final = _capture_session_trajectory(session)
    assert final[-1]["raw_output"]["formatted_output"] == output


def test_a_short_output_is_the_same_live_and_final(tmp_path):
    path = tmp_path / "acp_trajectory.jsonl"
    session = _stream(path, _command("exec-small", ["a\n", "b\n"]))

    live = path.read_text()
    final = _capture_session_trajectory(session)
    TrajectoryWriter(tmp_path / "final.jsonl").write_final(final)
    assert live == (tmp_path / "final.jsonl").read_text()
    assert final[-1]["raw_output"] == {"formatted_output": "a\nb\n", "exit_code": 0}
