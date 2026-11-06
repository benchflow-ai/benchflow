"""Guards the fix at 363d815b: omitted status is unchanged."""

import pytest

from benchflow.acp.session import ACPSession


@pytest.mark.parametrize("terminal", ["completed", "failed"])
@pytest.mark.parametrize(
    "partial",
    [
        {"title": "Finished command"},
        {"content": []},
        {"rawOutput": "done"},
        {"kind": "execute"},
        {"status": "unknown-status"},
    ],
)
def test_partial_update_preserves_terminal_state(terminal, partial):
    session = ACPSession("partial")
    session.handle_update(
        {"sessionUpdate": "tool_call", "toolCallId": "t", "status": terminal}
    )
    record = session.tool_calls[0]
    finished = record.finished_observed_ns
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", **partial}
    )
    assert record.status.value == terminal
    assert record.finished_observed_ns == finished
    assert not session.pending_tool_call_ids()


def test_unseen_partial_update_is_pending():
    session = ACPSession("partial")
    session.handle_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "new",
            "title": "Starting command",
        }
    )
    assert session.tool_calls[0].status.value == "in_progress"


@pytest.mark.parametrize(
    "partial",
    [
        {"title": "Kill the CLI"},
        {"content": []},
        {"rawInput": {"command": "sleep 30"}},
        {"kind": "execute"},
        {"status": "unknown-status"},
        {"status": None},
    ],
)
def test_partial_update_on_a_live_call_marks_it_running(partial):
    # The terminal rule must not freeze a pending call: an update means the
    # agent started it (the deterministic agent-crash golden, 0d56ebfa).
    session = ACPSession("partial")
    session.handle_update(
        {"sessionUpdate": "tool_call", "toolCallId": "t", "status": "pending"}
    )
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": "t", **partial}
    )
    assert session.tool_calls[0].status.value == "in_progress"
    assert session.pending_tool_call_ids() == ["t"]
