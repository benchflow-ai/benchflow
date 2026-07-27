#!/usr/bin/env python3
"""ACP shim for the Multi-Agent Medical Assistant — runs a LangGraph
supervisor→specialists graph as an ACP server.

Adapts the **Multi-Agent-Medical-Assistant** pattern
(https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant): a LangGraph
``StateGraph`` with a supervisor/router that dispatches to a knowledge-base
(RAG-style) or web-search specialist, an answer node that emits a confidence, a
**confidence-gated handoff** to the web specialist when confidence is low, and an
output **guardrail** node. The upstream's heavy stack (Azure embeddings, Qdrant,
torch CV weights, Tavily) is replaced with a small in-process knowledge base so
the whole multi-agent graph runs on the deepseek-only proxy path; the router →
specialists → confidence handoff → guardrail control flow is preserved.

Architecture:
  benchflow ACP client ←stdio→ this shim ←in-process→ LangGraph medical graph
                                          ←ChatOpenAI→ BenchFlow LiteLLM proxy

Each graph node is streamed back as its OWN ACP ``tool_call`` step, so the
multi-agent structure (which specialist ran, in what order) is visible in the
captured trajectory — even though every node's raw LLM call rides the single
per-rollout proxy log. The final answer is written to ``<cwd>/answer.md`` for the
task verifier.

Self-contained: like the other shims (deepagents, openclaw, harvey-lab), this
file is uploaded and run inside the sandbox with NO benchflow installed, so it
must NOT ``import benchflow``. It depends only on the stdlib plus langgraph /
langchain-openai installed into the shim's venv.
"""

import json
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping

_DIAG_TRUNCATE = 2000
_TOOL_INPUT_TRUNCATE = 500
_TOOL_RESULT_TRUNCATE = 1000
_DEFAULT_MODEL = "deepseek-v4-pro"
_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# A tiny stand-in knowledge base — the RAG specialist's corpus (replaces the
# upstream Qdrant collection so the graph runs without embeddings).
_KB = {
    "metformin": "Metformin is first-line for type 2 diabetes; common side effects "
    "are GI upset (nausea, diarrhea) and, rarely, lactic acidosis.",
    "aspirin": "Low-dose aspirin is used for secondary cardiovascular prevention; "
    "bleeding risk rises with dose and with anticoagulants.",
    "ibuprofen": "Ibuprofen is an NSAID for pain/inflammation; avoid in renal "
    "impairment and use caution with anticoagulants.",
}


# ── ACP stdio I/O ─────────────────────────────────────────────────────────────


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def recv():
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if not line:
            continue
        return json.loads(line)


# ── Provider / model resolution (deepseek over the OpenAI-compatible proxy) ────


def resolve_base_url(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    for key in ("BENCHFLOW_PROVIDER_BASE_URL", "DEEPSEEK_BASE_URL"):
        val = env.get(key)
        if val and val.strip():
            return val.strip()
    return _DEFAULT_DEEPSEEK_BASE_URL


def resolve_api_key(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    for key in ("BENCHFLOW_PROVIDER_API_KEY", "DEEPSEEK_API_KEY"):
        val = env.get(key)
        if val and val.strip():
            return val.strip()
    return ""


def resolve_model_id(requested: str, env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    model = (requested or "").strip()
    if not model:
        model = (env.get("BENCHFLOW_PROVIDER_MODEL") or "").strip()
    if not model:
        model = _DEFAULT_MODEL
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


def build_chat_model(model_id: str, env: Mapping[str, str] | None = None):
    """deepseek-v4-pro is OpenAI-compatible → ChatOpenAI at the proxy base_url.

    Imported lazily so the module parses (and its pure helpers stay testable)
    without langchain_openai installed in the host dev venv — it is only present
    inside the shim's sandbox venv.
    """
    from langchain_openai import ChatOpenAI

    env = os.environ if env is None else env
    kwargs = {
        "model": model_id,
        "base_url": resolve_base_url(env),
        "api_key": resolve_api_key(env) or "EMPTY",
        "temperature": _float_env(env, "BENCHFLOW_MODEL_TEMPERATURE", 0.2),
        "max_tokens": _int_env(env, "BENCHFLOW_MODEL_MAX_TOKENS", 2048),
    }
    return ChatOpenAI(**kwargs)


def _float_env(env, key, default):
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(env, key, default):
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


# ── The LangGraph medical assistant (router → specialists → handoff → guardrail)


def build_graph(chat_model):
    """Construct the supervisor→specialists StateGraph. Lazy import so the module
    parses without langgraph in the host venv."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        query: str
        route: str
        evidence: str
        answer: str
        confidence: float
        safe: bool
        web_tried: bool

    threshold = _float_env(os.environ, "MEDICAL_CONFIDENCE_THRESHOLD", 0.6)

    def supervisor(state: S) -> S:
        out = chat_model.invoke(
            [
                {
                    "role": "user",
                    "content": f"You route a medical question to a specialist. Question: {state['query']!r}. "
                    "Reply with ONE word: 'kb' if a drug/condition knowledge base likely covers "
                    "it, else 'web'.",
                }
            ]
        ).content.lower()
        return {"route": "web" if "web" in out else "kb"}

    def retrieve_kb(state: S) -> S:
        q = state["query"].lower()
        hits = [v for k, v in _KB.items() if k in q]
        return {"evidence": " ".join(hits)}

    def web_search(state: S) -> S:
        ev = f"[web] current guidance relevant to: {state['query']}"
        return {
            "evidence": (state.get("evidence", "") + " " + ev).strip(),
            "web_tried": True,
        }

    def answer(state: S) -> S:
        out = chat_model.invoke(
            [
                {
                    "role": "user",
                    "content": f"Question: {state['query']}\nEvidence: {state.get('evidence') or '(none)'}\n"
                    "Answer concisely for a clinician. Then on a NEW line output "
                    "'CONFIDENCE: <0.0-1.0>' for how well the evidence supports your answer.",
                }
            ]
        ).content
        m = re.search(r"CONFIDENCE:\s*([01](?:\.\d+)?)", out)
        conf = float(m.group(1)) if m else 0.5
        ans = re.split(r"CONFIDENCE:", out)[0].strip()
        return {"answer": ans, "confidence": conf}

    def guardrail(state: S) -> S:
        out = chat_model.invoke(
            [
                {
                    "role": "user",
                    "content": f"Output guardrail. The assistant answered: {state['answer']!r}. Is it safe and "
                    "appropriately cautious as medical information (not a diagnosis)? Reply yes/no.",
                }
            ]
        ).content.lower()
        return {"safe": "yes" in out}

    def after_supervisor(state: S) -> str:
        return "web_search" if state["route"] == "web" else "retrieve_kb"

    def after_answer(state: S) -> str:
        if state.get("confidence", 0.0) < threshold and not state.get("web_tried"):
            return "web_search"  # confidence-gated handoff to the fallback specialist
        return "guardrail"

    g = StateGraph(S)
    for name, fn in (
        ("supervisor", supervisor),
        ("retrieve_kb", retrieve_kb),
        ("web_search", web_search),
        ("answer", answer),
        ("guardrail", guardrail),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        after_supervisor,
        {"retrieve_kb": "retrieve_kb", "web_search": "web_search"},
    )
    g.add_edge("retrieve_kb", "answer")
    g.add_edge("web_search", "answer")
    g.add_conditional_edges(
        "answer", after_answer, {"web_search": "web_search", "guardrail": "guardrail"}
    )
    g.add_edge("guardrail", END)
    return g.compile()


def run_graph(
    model_id: str,
    query: str,
    cwd: str,
    session_id: str,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Stream the medical graph, emitting one ACP tool_call per node, and write
    the final answer to ``<cwd>/answer.md`` for the verifier."""
    start = time.time()
    chat_model = build_chat_model(model_id, env)
    graph = build_graph(chat_model)

    final: dict = {"query": query}
    order: list[str] = []
    for chunk in graph.stream({"query": query}, stream_mode="updates"):
        for node_name, delta in (chunk or {}).items():
            order.append(node_name)
            call_id = f"{node_name}_{len(order)}"
            _emit_tool_call(session_id, call_id, node_name, delta or {})
            _emit_tool_result(session_id, call_id, json.dumps(delta or {}))
            if isinstance(delta, dict):
                final.update(delta)

    answer_text = final.get("answer") or "(no answer produced)"
    _emit_text(session_id, answer_text)
    out_path = os.path.join(cwd, "answer.md")
    try:
        with open(out_path, "w") as fh:
            fh.write(answer_text + "\n")
    except OSError as exc:
        _emit_text(
            session_id, f"[medical-assistant: could not write {out_path}: {exc}]"
        )

    return {
        "path": " → ".join(order),
        "route": final.get("route"),
        "confidence": final.get("confidence"),
        "safe": final.get("safe"),
        "wall_clock_seconds": round(time.time() - start, 2),
    }


# ── ACP notification helpers ──────────────────────────────────────────────────


def _emit_text(session_id: str, text: str):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text[:_DIAG_TRUNCATE]},
                },
            },
        }
    )


def _emit_tool_call(session_id: str, call_id: str, node: str, delta):
    try:
        input_text = json.dumps(delta)
    except (TypeError, ValueError):
        input_text = str(delta)
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": node,
                    "kind": _node_kind(node),
                    "status": "in_progress",
                    "input": input_text[:_TOOL_INPUT_TRUNCATE],
                },
            },
        }
    )


def _emit_tool_result(session_id: str, call_id: str, result: str):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "text",
                                "text": result[:_TOOL_RESULT_TRUNCATE],
                            },
                        }
                    ],
                },
            },
        }
    )


def _node_kind(node: str) -> str:
    n = (node or "").lower()
    if "supervisor" in n:
        return "think"
    if "web" in n:
        return "search"
    if "retrieve" in n or "kb" in n:
        return "read"
    if "guardrail" in n:
        return "other"
    return "other"


# ── Main ACP loop ─────────────────────────────────────────────────────────────


def main():
    session_id = "medical-assistant-shim"
    cwd = "/app"
    model = ""

    while True:
        try:
            msg = recv()
        except EOFError:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {"image": False, "audio": False},
                        },
                        "agentInfo": {"name": "medical-assistant", "version": "1.0"},
                    },
                }
            )

        elif method == "session/new":
            cwd = params.get("cwd", "/app")
            session_id = f"medical-{uuid.uuid4().hex[:8]}"
            send({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session_id}})

        elif method == "session/set_model":
            model = params.get("modelId", "")
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/prompt":
            text = ""
            for part in params.get("prompt", []):
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
            model_id = resolve_model_id(model)
            try:
                metrics = run_graph(model_id, text.strip(), cwd, session_id)
                _emit_text(
                    session_id,
                    f"[medical-assistant: path {metrics['path']} · "
                    f"confidence={metrics['confidence']} · safe={metrics['safe']} · "
                    f"{metrics['wall_clock_seconds']}s]",
                )
            except Exception as e:
                _emit_text(
                    session_id, f"[medical-assistant error: {type(e).__name__}: {e}]"
                )
            send({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})

        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/request_permission":
            options = params.get("options", [])
            option_id = options[0].get("optionId", "default") if options else "default"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "outcome": {"outcome": "selected", "optionId": option_id}
                    },
                }
            )

        else:
            if req_id is not None:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
