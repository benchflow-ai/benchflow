"""The answer channel contains complete model text and no shim diagnostics."""

import pytest

from benchflow.agents import deepagents_acp_shim, harvey_lab_acp_shim


@pytest.mark.parametrize("shim", [deepagents_acp_shim, harvey_lab_acp_shim])
@pytest.mark.parametrize(
    "answer", ['{"answer":42}', "λ" * 12000], ids=["json", "long-unicode"]
)
def test_prompt_preserves_answer(shim, answer, tmp_path, monkeypatch):
    incoming = iter(
        [
            {"id": 1, "method": "session/new", "params": {"cwd": str(tmp_path)}},
            {"id": 2, "method": "session/prompt", "params": {"prompt": []}},
        ]
    )

    def recv():
        try:
            return next(incoming)
        except StopIteration:
            raise EOFError from None

    sent = []

    def run(**kwargs):
        shim._emit_text(kwargs["session_id"], answer)
        return dict(
            message_count=1,
            turn_count=1,
            tool_call_count=0,
            wall_clock_seconds=0,
            mirrored_output_count=0,
        )

    monkeypatch.setattr(shim, "recv", recv)
    monkeypatch.setattr(shim, "send", sent.append)
    name = "run_deep_agent" if shim is deepagents_acp_shim else "_run_harvey_lab_agent"
    monkeypatch.setattr(shim, name, run)
    shim.main()
    chunks = [
        msg["params"]["update"]["content"]["text"]
        for msg in sent
        if msg.get("method") == "session/update"
        and msg["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
    ]
    assert "".join(chunks) == answer
    assert sent[-1]["result"]["stopReason"] == "end_turn"
