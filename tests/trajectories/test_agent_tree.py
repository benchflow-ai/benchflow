"""Reconstructing an unmixed agent tree from bf.*-tagged proxy exchanges."""

from __future__ import annotations

from benchflow.trajectories import LLMExchange, LLMRequest, LLMResponse, Trajectory
from benchflow.trajectories.agents import build_agent_tree


def _ex(
    agent_id: str, parent: str | None, run_id: str, *, span_kind: str = "chat"
) -> LLMExchange:
    bf = {
        "agent_id": agent_id,
        "agent_name": agent_id,
        "span_kind": span_kind,
        "run_id": run_id,
        "session_id": "run-1",
    }
    if parent is not None:
        bf["parent_agent_id"] = parent
    return LLMExchange(
        request=LLMRequest(body={"model": "m", "messages": [], "bf": bf}),
        response=LLMResponse(
            body={"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        ),
    )


def test_build_agent_tree_groups_unmixed_and_links_parents():
    # medical-shaped: supervisor(root) -> answer(x2, web handoff) -> guardrail
    exchanges = [
        _ex("supervisor", None, "supervisor#1"),
        _ex("answer", "supervisor", "answer#1"),
        _ex("answer", "supervisor", "answer#2"),
        _ex("guardrail", "answer", "guardrail#1"),
    ]
    tree = build_agent_tree(Trajectory(session_id="run-1", exchanges=exchanges))

    assert {n.agent_id for n in tree.nodes()} == {"supervisor", "answer", "guardrail"}
    assert [r.agent_id for r in tree.roots] == ["supervisor"]

    supervisor = tree.find("supervisor")
    answer = tree.find("answer")
    guardrail = tree.find("guardrail")
    assert supervisor and answer and guardrail

    # tree shape: supervisor -> answer -> guardrail
    assert [c.agent_id for c in supervisor.children] == ["answer"]
    assert [c.agent_id for c in answer.children] == ["guardrail"]
    assert guardrail.children == []

    # NOT MIXED: each agent node holds only its own calls
    assert len(supervisor.exchanges) == 1
    assert len(answer.exchanges) == 2
    assert len(guardrail.exchanges) == 1
    for node in tree.nodes():
        assert all(
            ex.request.body["bf"]["agent_id"] == node.agent_id for ex in node.exchanges
        )


def test_build_agent_tree_handles_concurrent_seats_as_sibling_roots():
    # arena-shaped: independent seats, no parent -> distinct sibling root sub-trees,
    # never merged into one mixed node.
    exchanges = [
        _ex("seat-0", None, "seat-0#1"),
        _ex("seat-1", None, "seat-1#1"),
        _ex("seat-0", None, "seat-0#2"),
    ]
    tree = build_agent_tree(exchanges)
    assert sorted(r.agent_id for r in tree.roots) == ["seat-0", "seat-1"]
    assert len(tree.find("seat-0").exchanges) == 2
    assert len(tree.find("seat-1").exchanges) == 1
