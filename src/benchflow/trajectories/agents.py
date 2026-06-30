"""Reconstruct a multi-agent run as an agent tree from one shared proxy log.

A hosted multi-agent workflow routes every agent's raw LLM call through one
BenchFlow proxy, tagging each call with ``bf.*`` fields in request metadata
(recorded by the callback under ``request.body['bf']`` — see
``providers/litellm_logging.py``). This module turns that flat, ordered exchange
list back into the **agent tree**: one node per agent holding only its own calls
(unmixed), linked to its parent via ``bf.parent_agent_id`` — the same
parent-pointer reconstruction LangSmith does from ``parent_run_id``. See
``docs/reference/multi-agent-trajectory.md``.

Pure data + pure functions: no I/O, no async.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .types import LLMExchange, Trajectory

__all__ = ["AgentNode", "AgentTree", "build_agent_tree"]


def _bf(exchange: LLMExchange) -> dict:
    body = exchange.request.body
    bf = body.get("bf") if isinstance(body, dict) else None
    return bf if isinstance(bf, dict) else {}


@dataclass(eq=False)
class AgentNode:
    """One agent in the run, holding only its own LLM exchanges (unmixed)."""

    agent_id: str
    agent_name: str = ""
    parent_agent_id: str | None = None
    exchanges: list[LLMExchange] = field(default_factory=list)
    children: list[AgentNode] = field(default_factory=list)


@dataclass(eq=False)
class AgentTree:
    """The agent forest for one run.

    ``roots`` are agents with no (in-run) parent — a supervisor, or each
    independent seat in a concurrent arena. ``orphans`` are agents whose
    ``parent_agent_id`` names an agent not present in this run (kept as roots of
    their own sub-tree rather than silently dropped).
    """

    roots: list[AgentNode] = field(default_factory=list)
    orphans: list[AgentNode] = field(default_factory=list)

    def nodes(self) -> Iterator[AgentNode]:
        """Yield every agent node, pre-order from each root (then orphans)."""
        stack: list[AgentNode] = list(reversed(self.roots + self.orphans))
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def find(self, agent_id: str) -> AgentNode | None:
        for node in self.nodes():
            if node.agent_id == agent_id:
                return node
        return None


def build_agent_tree(source: Trajectory | Iterable[LLMExchange]) -> AgentTree:
    """Group exchanges by ``bf.agent_id`` and link them by ``bf.parent_agent_id``.

    Exchanges keep their original order within each agent node. An agent is a
    root when it has no ``parent_agent_id`` (or points at itself); it is an orphan
    when its parent is named but absent from the run.
    """
    exchanges = (
        source.exchanges if isinstance(source, Trajectory) else list(source)
    )
    nodes: dict[str, AgentNode] = {}
    order: list[str] = []  # first-seen agent order, for stable output
    for exchange in exchanges:
        bf = _bf(exchange)
        agent_id = bf.get("agent_id") or "unknown"
        node = nodes.get(agent_id)
        if node is None:
            node = AgentNode(
                agent_id=agent_id,
                agent_name=bf.get("agent_name") or agent_id,
                parent_agent_id=bf.get("parent_agent_id") or None,
            )
            nodes[agent_id] = node
            order.append(agent_id)
        elif not node.agent_name and bf.get("agent_name"):
            node.agent_name = bf["agent_name"]
        node.exchanges.append(exchange)

    roots: list[AgentNode] = []
    orphans: list[AgentNode] = []
    for agent_id in order:
        node = nodes[agent_id]
        parent_id = node.parent_agent_id
        if not parent_id or parent_id == agent_id:
            roots.append(node)
        elif parent_id in nodes:
            nodes[parent_id].children.append(node)
        else:
            orphans.append(node)
    return AgentTree(roots=roots, orphans=orphans)
