"""Host receipt timing regressions for GH #1033; never execution timestamps."""

import json
import os
from datetime import UTC, datetime

import pytest

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _snapshot_session_trajectory,
)
from benchflow.trajectories.viewer.payload import _build_acp_payload


def test_receipts_survive_streaming_merging_and_final_capture(monkeypatch):
    """GH #1033: receipt times survive chunk merging without capture-time invention."""
    clock = iter([100, 200, 300, 400, 500])
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: next(clock))
    session = ACPSession("clock-test")
    session.record_user_prompt("go")
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "a"},
        }
    )
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "b"},
        }
    )
    live = _snapshot_session_trajectory(session)
    assert live[1]["text"] == "ab"
    assert live[1]["receipt"] == {
        "source": "benchflow_host",
        "clock": "unix",
        "first_observed_ns": "200",
        "last_observed_ns": "300",
    }
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "kind": "bash",
            "title": "echo hi",
        }
    )
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "completed"}
    )
    final = _capture_session_trajectory(session)
    assert final[:2] == live
    assert final[0]["receipt"]["first_observed_ns"] == "100"
    assert final[2]["receipt"]["first_observed_ns"] == "400"
    assert final[2]["receipt"]["last_observed_ns"] == "500"
    assert final[2]["status"] == "completed"
    assert (
        final
        == _snapshot_session_trajectory(session)
        == _capture_session_trajectory(session)
    )


def test_legacy_capture_does_not_backfill_receipts():
    """GH #1033: old shims without observed events retain unknown timing."""
    session = ACPSession("legacy")
    session.message_chunks.append("historical")
    assert _capture_session_trajectory(session) == [
        {"type": "agent_message", "text": "historical"}
    ]


def test_receipts_do_not_trust_remote_timestamps_and_snapshot_is_detached(monkeypatch):
    """GH #1033: timestamp provenance stays host-local despite remote clock fields."""
    clock = iter([100, 90])  # Wall clock can move backwards: do not invent durations.
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: next(clock))
    session = ACPSession("clock-test")
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "t",
            "status": "in_progress",
            "timestamp": "remote",
            "receipt": {"source": "robot"},
        }
    )
    before = _snapshot_session_trajectory(session)
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "completed"}
    )
    after = _capture_session_trajectory(session)
    assert before[0]["receipt"]["last_observed_ns"] == "100"
    assert after[0]["receipt"]["last_observed_ns"] == "90"
    assert after[0]["receipt"]["source"] == "benchflow_host"


def test_pending_receipt_snapshot_cannot_mutate_session(monkeypatch):
    """GH #1033: live receipt metadata must not alias the mutable session."""
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: 123)
    session = ACPSession("s")
    session.handle_update({"sessionUpdate": "text_update", "text": "hello"})
    snapshot = _snapshot_session_trajectory(session)
    snapshot[0]["receipt"]["first_observed_ns"] = "999"
    assert (
        _capture_session_trajectory(session)[0]["receipt"]["first_observed_ns"] == "123"
    )


def test_unchanged_tool_polls_do_not_rewrite_trajectory(tmp_path, monkeypatch):
    """Guards the trajectory write dedup against a regression from host receipt times.

    Each unchanged status poll used to advance the call's ``last_observed_ns``,
    so every poll produced a new payload and rewrote the whole trajectory
    (1 tool call + 20 unchanged polls: 1 write expected, 21 with the regression).
    """
    clock = [1_000]
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: clock[0])
    writer = TrajectoryWriter(tmp_path / "acp_trajectory.jsonl")
    writes: list[str] = []
    real_replace = os.replace

    def counting_replace(src, dst):
        writes.append(str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", counting_replace)
    session = ACPSession("dedup")
    session.on_change = writer
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "kind": "execute",
            "title": "sleep 60",
            "status": "in_progress",
        }
    )
    for _ in range(20):
        clock[0] += 1_000
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "t",
                "status": "in_progress",
            }
        )
    assert len(writes) == 1
    on_disk = json.loads(writer.path.read_text())
    assert on_disk["receipt"]["last_observed_ns"] == "1000"

    # A real change is still persisted and advances the receipt.
    clock[0] += 1_000
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "completed"}
    )
    assert len(writes) == 2
    on_disk = json.loads(writer.path.read_text())
    assert on_disk["status"] == "completed"
    assert on_disk["receipt"]["last_observed_ns"] == "22000"


def test_viewer_reads_host_receipt_times_from_captured_trajectory(
    tmp_path, monkeypatch
):
    """Guards the #1033 timing contract against the host receipt-time change's receipt-only encoding.

    The host receipt-time change wrote host times only as ``receipt`` nanosecond strings, but the
    #1034 viewer reads ISO-8601 ``ts`` (text/timeout events) and
    ``started_at`` / ``finished_at`` (tool calls), so every step had t=None.
    """
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    base_ns = int(t0.timestamp()) * 1_000_000_000
    clock = [base_ns]
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: clock[0])

    def at(seconds: float) -> None:
        clock[0] = base_ns + int(seconds * 1_000_000_000)

    rollout = tmp_path / "rollout"
    writer = TrajectoryWriter(rollout / "trajectory" / "acp_trajectory.jsonl")
    session = ACPSession("viewer")
    session.on_change = writer
    session.record_user_prompt("go")
    at(1)
    session.handle_update(
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "plan"},
        }
    )
    at(2)
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "kind": "execute",
            "title": "make",
            "status": "in_progress",
        }
    )
    at(4.5)
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "completed"}
    )
    at(5)  # A repeated terminal poll must not move the finish time.
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "completed"}
    )
    at(6)
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "done"},
        }
    )
    at(7)
    session.record_agent_timeout(
        timeout_sec=7, pending_tool_call_ids=[], terminal_trajectory_complete=True
    )
    session.mark_prompt_end()

    on_disk = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert on_disk == _capture_session_trajectory(session)
    assert on_disk[0]["ts"] == "2026-01-01T10:00:00+00:00"
    assert on_disk[2]["started_at"] == "2026-01-01T10:00:02+00:00"
    assert on_disk[2]["finished_at"] == "2026-01-01T10:00:04.500000+00:00"
    assert on_disk[2]["receipt"]["first_observed_ns"] == str(base_ns + 2 * 10**9)

    steps = _build_acp_payload(rollout, None).steps
    assert [step.t for step in steps] == [
        t0.timestamp() + offset for offset in (0, 1, 2, 6, 7)
    ]
    assert steps[2].dur == pytest.approx(2.5)


def test_pending_tool_call_has_start_but_no_finish_time(monkeypatch):
    """Guards the #1033 timing contract: no invented finish for unfinished calls."""
    monkeypatch.setattr("benchflow.acp.session.time.time_ns", lambda: 10**18)
    session = ACPSession("pending")
    session.handle_update(
        {"sessionUpdate": "tool_call", "toolCallId": "t", "status": "in_progress"}
    )
    (event,) = _capture_session_trajectory(session)
    assert event["started_at"] == "2001-09-09T01:46:40+00:00"
    assert "finished_at" not in event
