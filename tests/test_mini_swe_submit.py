"""ACP tool-call lifecycle for the mini-swe shim's submit handling.

Regression for the multi-action submit case: when a turn carries several tool
calls and one of them is the submit (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`),
the shim must emit each executed action's real result, close the submit action
with the submission, and emit nothing for actions that never ran.

Gated on ``minisweagent`` (only installed in the sandbox), like the docker-gated
smoke test — skips cleanly in CI, runs where the runtime is present.
"""

import tempfile

import pytest

pytest.importorskip("minisweagent")

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel

from benchflow.agents import mini_swe_acp_shim as shim


def _collect(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(shim, "send", lambda msg: events.append(msg))
    return events


def _updates_by(events, session_update):
    out = {}
    for ev in events:
        upd = ev.get("params", {}).get("update", {})
        if upd.get("sessionUpdate") == session_update:
            out[upd["toolCallId"]] = upd
    return out


def test_multi_action_turn_with_submit(monkeypatch):
    events = _collect(monkeypatch)
    agent_cls = shim._acp_agent_class()

    with tempfile.TemporaryDirectory() as td:
        agent = agent_cls(
            LitellmModel(model_name="gpt-4o-mini"),
            LocalEnvironment(cwd=td),
            session_id="s",
            system_template="x",
            instance_template="y",
        )
        message = {
            "role": "assistant",
            "content": "",
            "extra": {
                "actions": [
                    {"command": "echo first", "tool_call_id": "call_a"},
                    {
                        "command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone-marker\\n'",
                        "tool_call_id": "call_submit",
                    },
                    {"command": "echo after", "tool_call_id": "call_c"},
                ]
            },
        }
        with pytest.raises(Submitted):
            agent.execute_actions(message)

    starts = _updates_by(events, "tool_call")
    results = _updates_by(events, "tool_call_update")

    # Executed action: real output, completed.
    assert "call_a" in starts and "call_a" in results
    assert results["call_a"]["status"] == "completed"
    assert "first" in results["call_a"]["content"][0]["content"]["text"]

    # Submit action: closed with the submission text, completed.
    assert results["call_submit"]["status"] == "completed"
    assert "done-marker" in results["call_submit"]["content"][0]["content"]["text"]

    # Action after submit never ran: no start, no result (not falsely completed).
    assert "call_c" not in starts
    assert "call_c" not in results
