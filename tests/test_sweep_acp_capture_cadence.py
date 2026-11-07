"""Guards the fix at 363d815b: chunks never serialize history."""

import json

import pytest

from benchflow.acp.session import ACPSession
from benchflow.trajectories._capture import TrajectoryWriter, make_trajectory_sink


@pytest.mark.asyncio
@pytest.mark.parametrize("sink_kind", ["direct", "cumulative"])
async def test_burst_is_coalesced_and_prompt_end_forces_complete_flush(
    tmp_path, monkeypatch, sink_kind
):
    writer = TrajectoryWriter(tmp_path / "trace.jsonl")
    session = ACPSession("s")
    session.on_change = (
        writer
        if sink_kind == "direct"
        else make_trajectory_sink(
            writer, [{"type": "notice", "severity": "info", "title": "prior"}]
        )
    )
    calls = []
    original = writer.write_events

    def count(events, **kwargs):
        calls.append(len(events))
        return original(events, **kwargs)

    monkeypatch.setattr(writer, "write_events", count)
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "content": [{"type": "text", "text": "x" * 100_000}],
            "status": "completed",
        }
    )
    for _ in range(1000):
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "a"},
            }
        )
    assert len(calls) <= 1
    session.mark_prompt_end()
    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert records[-1]["text"] == "a" * 1000
    assert len(calls) <= 2


@pytest.mark.asyncio
async def test_background_serialization_does_not_run_on_receive_thread(
    tmp_path, monkeypatch
):
    import asyncio
    import threading

    writer = TrajectoryWriter(tmp_path / "trace.jsonl")
    session = ACPSession("s")
    session.on_change = writer
    thread_ids = []
    original = writer.write_events

    def record_thread(events, **kwargs):
        thread_ids.append(threading.get_ident())
        return original(events, **kwargs)

    monkeypatch.setattr(writer, "write_events", record_thread)
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "hello"},
        }
    )
    for _ in range(100):
        if thread_ids:
            break
        await asyncio.sleep(0.01)
    assert thread_ids and threading.get_ident() not in thread_ids
    session.mark_prompt_end()


def test_unchanged_completed_history_is_not_reserialized(tmp_path, monkeypatch):
    from benchflow.trajectories import _snapshot

    writer = TrajectoryWriter(tmp_path / "trace.jsonl")
    session = ACPSession("s")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "content": [{"type": "text", "text": "x" * 100_000}],
            "status": "completed",
        }
    )
    calls = []
    original = _snapshot.json.dumps

    def count(record, *args, **kwargs):
        if isinstance(record, dict) and record.get("tool_call_id") == "t":
            calls.append(1)
        return original(record, *args, **kwargs)

    monkeypatch.setattr(_snapshot.json, "dumps", count)
    writer.flush(session)
    initial = len(calls)
    for _ in range(5):
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "a"},
            }
        )
        writer.flush(session)
    assert len(calls) == initial
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", "rawOutput": "updated"}
    )
    writer.flush(session)
    assert len(calls) > initial
    assert "updated" in writer.path.read_text()


@pytest.mark.asyncio
async def test_legacy_and_steps_only_sinks_keep_their_finalization_contract(tmp_path):
    from types import SimpleNamespace

    legacy = ACPSession("legacy")
    legacy.message_chunks.append("legacy text")
    steps = SimpleNamespace(steps=[{"type": "agent_message", "text": "adapter text"}])
    for index, session in enumerate([legacy, steps]):
        writer = TrajectoryWriter(tmp_path / f"trace-{index}.jsonl")
        session.on_change = make_trajectory_sink(writer, [])
        session.on_change(session)
        assert writer.path.exists()
        assert getattr(writer, "_task", None) is None
