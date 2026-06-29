"""A LangGraph supervisor→specialists agent — the Multi-Agent-Medical-Assistant
PATTERN, hosted by BenchFlow.

A real ``langgraph.StateGraph``: a supervisor/router that dispatches to a KB
(RAG-style) or web-search specialist, an answer node that emits a confidence, a
**confidence-gated handoff** to the web specialist when confidence is low, and an
output **guardrail** node — mirroring the medical assistant's router + specialists
+ confidence-based handoff + guardrails (minus its Qdrant/CV/web stack). Every
node's LLM call uses ``langchain_openai.ChatOpenAI`` pointed at the BenchFlow
provider proxy (``BENCHFLOW_PROVIDER_*``), so per-node usage + the raw-LLM
trajectory are captured by BenchFlow.
"""

from __future__ import annotations

import os
import re
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# A tiny stand-in knowledge base — the RAG specialist's corpus.
_KB = {
    "metformin": "Metformin is first-line for type 2 diabetes; common side effects "
                 "are GI upset (nausea, diarrhea) and, rarely, lactic acidosis.",
    "aspirin": "Low-dose aspirin is used for secondary cardiovascular prevention; "
               "bleeding risk rises with dose and with anticoagulants.",
    "ibuprofen": "Ibuprofen is an NSAID for pain/inflammation; avoid in renal "
                 "impairment and use caution with anticoagulants.",
}


class S(TypedDict, total=False):
    query: str
    route: str
    evidence: str
    answer: str
    confidence: float
    safe: bool
    web_tried: bool
    trace: list


# agent_name -> (base_url, api_key, model). A runner sets one entry per
# specialist to route that agent through its OWN proxy, so each gets a separate
# llm_trajectory + usage. Empty => every node shares one endpoint (shared/native).
_AGENT_PROVIDERS: dict[str, tuple[str, str, str]] = {}


def set_agent_provider(agent: str, base: str, key: str, model: str) -> None:
    _AGENT_PROVIDERS[agent] = (base, key, model)


def _llm(agent: str = "default") -> ChatOpenAI:
    cfg = _AGENT_PROVIDERS.get(agent)
    if cfg is not None:
        base, key, model = cfg
    else:
        base = os.environ.get("BENCHFLOW_PROVIDER_BASE_URL") or "https://api.deepseek.com/v1"
        key = (os.environ.get("BENCHFLOW_PROVIDER_API_KEY")
               or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY")
        model = os.environ.get("BENCHFLOW_PROVIDER_MODEL") or "deepseek/deepseek-v4-pro"
    return ChatOpenAI(model=model, base_url=base, api_key=key,
                      temperature=0.2, max_tokens=2048)


def _log(state: S, node: str, note: str) -> None:
    state.setdefault("trace", []).append({"node": node, "note": note})


def supervisor(state: S) -> S:
    out = _llm("supervisor").invoke([{"role": "user", "content":
        f"You route a medical question to a specialist. Question: {state['query']!r}. "
        "Reply with ONE word: 'kb' if a drug/condition knowledge base likely covers "
        "it, else 'web'."}]).content.lower()
    route = "web" if "web" in out else "kb"
    _log(state, "supervisor", f"route={route}")
    return {"route": route, "trace": state["trace"]}


def retrieve_kb(state: S) -> S:
    q = state["query"].lower()
    hits = [v for k, v in _KB.items() if k in q]
    _log(state, "retrieve_kb", f"hits={len(hits)}")
    return {"evidence": " ".join(hits), "trace": state["trace"]}


def web_search(state: S) -> S:
    ev = f"[web] current guidance relevant to: {state['query']}"
    _log(state, "web_search", "fallback evidence")
    return {"evidence": (state.get("evidence", "") + " " + ev).strip(),
            "web_tried": True, "trace": state["trace"]}


def answer(state: S) -> S:
    out = _llm("answer").invoke([{"role": "user", "content":
        f"Question: {state['query']}\nEvidence: {state.get('evidence') or '(none)'}\n"
        "Answer concisely for a clinician. Then on a NEW line output "
        "'CONFIDENCE: <0.0-1.0>' for how well the evidence supports your answer."}]
    ).content
    m = re.search(r"CONFIDENCE:\s*([01](?:\.\d+)?)", out)
    conf = float(m.group(1)) if m else 0.5
    ans = re.split(r"CONFIDENCE:", out)[0].strip()
    _log(state, "answer", f"confidence={conf}")
    return {"answer": ans, "confidence": conf, "trace": state["trace"]}


def guardrail(state: S) -> S:
    out = _llm("guardrail").invoke([{"role": "user", "content":
        f"Output guardrail. The assistant answered: {state['answer']!r}. Is it safe and "
        "appropriately cautious as medical information (not a diagnosis)? Reply yes/no."}]
    ).content.lower()
    _log(state, "guardrail", f"safe={'yes' in out}")
    return {"safe": "yes" in out, "trace": state["trace"]}


def _after_supervisor(state: S) -> str:
    return "web_search" if state["route"] == "web" else "retrieve_kb"


def _after_answer(state: S) -> str:
    threshold = float(os.environ.get("MEDICAL_CONFIDENCE_THRESHOLD", "0.6"))
    if state.get("confidence", 0.0) < threshold and not state.get("web_tried"):
        return "web_search"  # confidence-gated handoff to the fallback specialist
    return "guardrail"


def build_graph():
    g = StateGraph(S)
    for name, fn in (("supervisor", supervisor), ("retrieve_kb", retrieve_kb),
                     ("web_search", web_search), ("answer", answer),
                     ("guardrail", guardrail)):
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _after_supervisor,
                            {"retrieve_kb": "retrieve_kb", "web_search": "web_search"})
    g.add_edge("retrieve_kb", "answer")
    g.add_edge("web_search", "answer")
    g.add_conditional_edges("answer", _after_answer,
                            {"web_search": "web_search", "guardrail": "guardrail"})
    g.add_edge("guardrail", END)
    return g.compile()


def run(query: str) -> S:
    return build_graph().invoke({"query": query, "trace": []})
