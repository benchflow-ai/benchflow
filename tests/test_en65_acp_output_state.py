"""Guards the streamed-output state regression present in 16ae05af."""

import asyncio
import json
import random

import pytest

from benchflow.acp.session import ACPSession, ToolCallRecord
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    make_trajectory_sink,
)


def _receipt(tick):
    return {
        "source": "benchflow_host",
        "clock": "unix",
        "first_observed_ns": str(tick),
        "last_observed_ns": str(tick),
    }


def _session():
    session = ACPSession("stream")
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "title": "Bash",
            "kind": "execute",
            "status": "in_progress",
        },
        _receipt=_receipt(1),
    )
    return session


def _update(session, tick, **fields):
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", **fields},
        _receipt=_receipt(tick),
    )


def _chunk(session, tick, text):
    _update(session, tick, _meta={"terminal_output_delta": {"data": text}})


@pytest.mark.parametrize("count", [32, 128])
def test_output_updates_do_not_materialize_the_growing_stream(monkeypatch, count):
    """Guards 16ae05af: receipt state checks must not copy N growing outputs."""
    original = ToolCallRecord.raw_output
    reads = []

    def measured(record):
        reads.append(len(record._terminal_output))
        return original.fget(record)

    monkeypatch.setattr(ToolCallRecord, "raw_output", property(measured, original.fset))
    session = _session()
    for tick in range(2, count + 2):
        _chunk(session, tick, "x" * 8192)
    assert reads == []
    (event,) = _capture_session_trajectory(session)
    assert event["raw_output"]["formatted_output"] == "x" * (8192 * count)
    assert event["receipt"]["last_observed_ns"] == str(count + 1)


def test_empty_and_authoritative_output_keep_exact_receipt_equality():
    """Guards 16ae05af: an equivalent rawOutput or hidden delta is not a change."""
    session = _session()
    _chunk(session, 2, "")  # First empty delta exposes an empty output record.
    assert session._tool_receipts["t"]["last_observed_ns"] == "2"
    _chunk(session, 3, "")
    assert session._tool_receipts["t"]["last_observed_ns"] == "2"
    _chunk(session, 4, "abc")
    _update(session, 5, rawOutput={"formatted_output": "abc", "exit_code": None})
    assert session._tool_receipts["t"]["last_observed_ns"] == "4"
    _chunk(session, 6, "hidden")
    _update(session, 7, _meta={"terminal_exit": {"exit_code": 3}})
    assert session._tool_receipts["t"]["last_observed_ns"] == "4"
    _update(session, 8, rawOutput={"exit_code": 3})
    assert session.tool_calls[0].raw_output == {
        "formatted_output": "abchidden",
        "exit_code": 3,
    }
    assert session._tool_receipts["t"]["last_observed_ns"] == "8"
    _update(session, 9, _meta={"terminal_exit": {"exit_code": 3, "ignored": True}})
    assert session._tool_receipts["t"]["last_observed_ns"] == "8"
    _update(session, 10, _meta={"terminal_exit": {"exit_code": 4}})
    assert session._tool_receipts["t"]["last_observed_ns"] == "10"
    _update(session, 11, rawOutput="authoritative")
    _chunk(session, 12, "also hidden")
    assert session.tool_calls[0].raw_output == "authoritative"
    assert session._tool_receipts["t"]["last_observed_ns"] == "11"


@pytest.mark.parametrize("seed", range(12))
def test_state_comparison_matches_materialized_output_equality(seed):
    """Guards 16ae05af: the optimization retains the original value-equality rule."""
    rng = random.Random(seed)
    session = _session()
    record = session.tool_calls[0]
    for tick in range(2, 102):
        before = record.raw_output
        last_receipt = session._tool_receipts["t"]["last_observed_ns"]
        choice = rng.randrange(4)
        if choice == 0:
            _chunk(session, tick, rng.choice(["", "ab", "é中", " "]))
        elif choice == 1:
            _update(
                session,
                tick,
                rawOutput=rng.choice(
                    [
                        {},
                        {"exit_code": 0},
                        {"formatted_output": ""},
                        {"formatted_output": "ab", "exit_code": None},
                        "literal",
                    ]
                ),
            )
        elif choice == 2:
            _update(
                session,
                tick,
                _meta={"terminal_exit": {"exit_code": rng.choice([None, 0, 1])}},
            )
        else:
            output = record.raw_output
            _update(
                session,
                tick,
                rawOutput=dict(output) if isinstance(output, dict) else output,
            )
        expected = str(tick) if before != record.raw_output else last_receipt
        assert session._tool_receipts["t"]["last_observed_ns"] == expected


@pytest.mark.asyncio
async def test_async_capture_keeps_full_output_and_cancels_stale_writes(tmp_path):
    """Guards 16ae05af: cancellation/final capture retain bytes, parent and exit fields."""
    session = _session()
    writer = TrajectoryWriter(tmp_path / "acp.jsonl")
    session.on_change = make_trajectory_sink(writer, [])
    sent = []

    async def produce():
        for tick in range(2, 258):
            text = str(tick) + "x" * 8192
            sent.append(text)
            _chunk(session, tick, text)
            await asyncio.sleep(0)

    producer = asyncio.create_task(produce())
    while len(sent) < 32:
        await asyncio.sleep(0)
    producer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await producer
    _update(
        session,
        1000,
        status="completed",
        _meta={
            "terminal_exit": {"exit_code": 7},
            "claudeCode": {"parentToolUseId": "parent"},
        },
    )
    session.mark_prompt_end()
    trajectory = _capture_session_trajectory(session)
    assert trajectory[0]["raw_output"] == {
        "formatted_output": "".join(sent),
        "exit_code": 7,
    }
    assert trajectory[0]["parent_tool_call_id"] == "parent"
    writer.write_final(trajectory)
    final = writer.path.read_bytes()
    await asyncio.sleep(0.3)
    assert writer.path.read_bytes() == final
    assert [json.loads(line) for line in final.splitlines()] == trajectory
