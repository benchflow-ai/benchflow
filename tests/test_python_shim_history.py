"""Successive ACP prompts retain model/tool state, isolated by session id."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchflow.agents import deepagents_acp_shim, harvey_lab_acp_shim


@pytest.mark.parametrize("shim", [deepagents_acp_shim, harvey_lab_acp_shim])
def test_history_survives_followups_and_other_sessions(shim, tmp_path, monkeypatch):
    sent = []
    session_ids = []
    calls = iter(["new", "remember", "followup", "new", "followup", "old"])

    def recv():
        try:
            action = next(calls)
        except StopIteration:
            raise EOFError from None
        if action == "new":
            return {"id": 1, "method": "session/new", "params": {"cwd": str(tmp_path)}}
        return {
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_ids[0] if action == "old" else session_ids[-1],
                "prompt": [
                    {
                        "type": "text",
                        "text": "RAVEN" if action == "remember" else "recall",
                    }
                ],
            },
        }

    def send(msg):
        sent.append(msg)
        if "sessionId" in msg.get("result", {}):
            session_ids.append(msg["result"]["sessionId"])

    monkeypatch.setattr(shim, "recv", recv)
    monkeypatch.setattr(shim, "send", send)
    if shim is deepagents_acp_shim:
        memory = ModuleType("langgraph.checkpoint.memory")
        memory.InMemorySaver = dict
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.memory", memory)

        class Graph:
            def __init__(self, checkpointer=None):
                self.state = checkpointer if checkpointer is not None else {}

            def stream(self, inputs, config, stream_mode):
                history = self.state.setdefault(
                    config.get("configurable", {}).get("thread_id", "one"), []
                )
                instruction = inputs["messages"][0]["content"]
                if instruction == "RAVEN":
                    history.extend(["RAVEN", "tool-result"])
                answer = "RAVEN+tool-result" if history else "empty"
                yield {
                    "model": {"messages": [{"role": "assistant", "content": answer}]}
                }

        monkeypatch.setattr(
            shim, "build_deep_agent", lambda *a, **k: Graph(k.get("checkpointer"))
        )
    else:
        monkeypatch.setattr(shim, "_patch_sandbox_module", lambda: None)
        monkeypatch.setattr(shim, "_install_parse_doc_helper", lambda: None)
        monkeypatch.setattr(shim, "_load_system_prompt", lambda: "system")
        tools = ModuleType("harness.tools")
        tools.ToolExecutor = lambda **kw: SimpleNamespace(
            execute=lambda *a: "tool-result", get_metrics=lambda: {}
        )
        tools.get_all_tool_definitions = lambda: []
        monkeypatch.setitem(sys.modules, "harness.tools", tools)

        class Adapter:
            make_system_message = staticmethod(lambda s: "system")
            make_user_message = staticmethod(lambda s: s)
            make_tool_result_messages = staticmethod(
                lambda results: [r[1] for r in results]
            )

            def chat(self, messages, tools):
                if messages[-1] == "RAVEN":
                    return SimpleNamespace(
                        message="call",
                        input_tokens=1,
                        output_tokens=1,
                        text="",
                        tool_calls=[
                            SimpleNamespace(id="t", name="read", arguments="{}")
                        ],
                    )
                answer = (
                    "RAVEN+tool-result"
                    if "RAVEN" in messages and "tool-result" in messages
                    else "empty"
                )
                return SimpleNamespace(
                    message=answer,
                    input_tokens=1,
                    output_tokens=1,
                    text=answer,
                    tool_calls=[],
                )

        monkeypatch.setattr(shim, "_create_adapter", lambda *a: Adapter())
    shim.main()
    answers = [
        m["params"]["update"]["content"]["text"]
        for m in sent
        if m.get("method") == "session/update"
        and m["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        and m["params"]["update"]["content"]["text"] in ("RAVEN+tool-result", "empty")
    ]
    assert answers == [
        "RAVEN+tool-result",
        "RAVEN+tool-result",
        "empty",
        "RAVEN+tool-result",
    ]
