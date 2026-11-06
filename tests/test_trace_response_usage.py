"""Guards the fix at 363d815b."""

import json

import pytest

from benchflow.traces.parsers import parse_claude_code_session


@pytest.mark.parametrize("same_request", [True, False])
@pytest.mark.parametrize("with_id", [True, False])
def test_usage_is_final_per_response_without_losing_blocks(
    tmp_path, same_request, with_id
):
    """Guards FF-sw-trajectories response identity and independent missing IDs."""
    entries = []
    for index in range(3):
        entries.append(
            {
                "type": "assistant",
                "requestId": "r" if same_request else str(index),
                "message": {
                    "id": "response" if with_id else None,
                    "content": [{"type": "text", "text": f"block {index}"}],
                    "usage": {"input_tokens": 1000, "output_tokens": 50 + index},
                },
            }
        )
    session = tmp_path / "session.jsonl"
    session.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    trace = parse_claude_code_session(session)
    assert len(trace.steps) == 3
    assert trace.total_input_tokens == (1000 if same_request and with_id else 3000)
    assert trace.total_output_tokens == (52 if same_request and with_id else 153)


def test_later_content_without_usage_keeps_final_usage_record(tmp_path):
    """Guards FF-sw-trajectories against erasing usage on a trailing block."""
    entries = [
        {
            "type": "assistant",
            "requestId": "r",
            "message": {
                "id": "response",
                "content": [{"type": "text", "text": "block"}],
                "usage": {"input_tokens": 1000, "output_tokens": 50},
            },
        }
        for _ in range(2)
    ]
    entries.append(
        {
            "type": "assistant",
            "requestId": "r",
            "message": {
                "id": "response",
                "content": [{"type": "text", "text": "done"}],
            },
        }
    )
    session = tmp_path / "session.jsonl"
    session.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    trace = parse_claude_code_session(session)
    assert len(trace.steps) == 3
    assert (trace.total_input_tokens, trace.total_output_tokens) == (1000, 50)
