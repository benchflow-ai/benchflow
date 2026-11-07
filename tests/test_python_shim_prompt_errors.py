"""Failed prompts must use the ACP error channel, even after productive work."""

import pytest

from benchflow.agents import deepagents_acp_shim, harvey_lab_acp_shim


@pytest.mark.parametrize("shim", [deepagents_acp_shim, harvey_lab_acp_shim])
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection lost"),
        RuntimeError("context_length_exceeded"),
        RuntimeError("You've hit your usage limit"),
    ],
)
@pytest.mark.parametrize("partial", [False, True])
def test_prompt_errors_are_not_successful_turns(
    shim, error, partial, tmp_path, monkeypatch
):
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
        if partial:
            shim._emit_text(kwargs["session_id"], "working")
            shim._emit_tool_call(kwargs["session_id"], "tool-1", "read", "{}")
        raise error

    monkeypatch.setattr(shim, "recv", recv)
    monkeypatch.setattr(shim, "send", sent.append)
    name = "run_deep_agent" if shim is deepagents_acp_shim else "_run_harvey_lab_agent"
    monkeypatch.setattr(shim, name, run)
    shim.main()
    (reply,) = [msg for msg in sent if msg.get("id") == 2]
    assert "result" not in reply
    assert reply["error"]["code"] == -32603
    assert str(error) in reply["error"]["message"]
    chunks = [
        msg["params"]["update"]["content"]["text"]
        for msg in sent
        if msg.get("method") == "session/update"
        and msg["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
    ]
    assert chunks == (["working"] if partial else [])
