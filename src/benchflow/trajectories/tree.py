"""Tree data model for the tree-native Rollout.

A ``Rollout`` is one RL episode, and it is a *tree of states*. This module is
the pure data model for that tree — types and pure functions only, no I/O, no
async. The engine (checkpoint/fork, scheduling, scoring) is wired in elsewhere.

The execution model has three structures over the one tree:

- ``Step`` — one edge: a (reason -> act) -> (tool-in -> tool-out) cycle.
- ``RolloutNode`` — one state sₜ; ``RolloutTree`` wraps the root node.
- ``Trajectory`` — a derived view: the root-to-leaf path's ordered Steps,
  computed by :func:`trajectory`, never stored.

A linear rollout genuinely *is* a degree-1 tree (every node has at most one
child), so the data model costs nothing extra for the common case.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class Step:
    """One edge of the Rollout tree: a (reason -> act) -> (tool-in -> tool-out) cycle.

    Han's atomic unit — one ``Step`` is one "turn". The payload of what
    actually happened on the edge lives in the free-form ``data`` dict; this
    type stays minimal so the engine and adapters can shape ``data`` freely.
    """

    id: str
    data: dict = field(default_factory=dict)


@dataclass(eq=False)
class RolloutNode:
    """One state sₜ in the Rollout tree.

    ``step_in`` is the edge taken from ``parent`` to reach this state; it is
    ``None`` at the root, which has no incoming edge.

    Nodes have *identity*, not structural value: two nodes are equal only when
    they are the same object (``eq=False``). Value equality would recurse
    through the ``parent`` backref and conflate distinct states.
    """

    id: str
    parent: RolloutNode | None = None
    children: list[RolloutNode] = field(default_factory=list)
    step_in: Step | None = None
    state: dict = field(default_factory=dict)


@dataclass(eq=False)
class RolloutTree:
    """Wraps the root :class:`RolloutNode` of a Rollout tree.

    A Rollout is a tree of states; this is the structure that owns it. New
    states are reached with :meth:`advance`, which adds one child per call —
    a second child of the same node makes that node a branch point.
    """

    root: RolloutNode = field(default_factory=lambda: RolloutNode(id="root"))
    _node_count: int = 1

    def advance(self, node: RolloutNode, step: Step) -> RolloutNode:
        """Grow the tree by one edge: add a child of ``node`` reached via ``step``.

        Returns the new child node, wired to its parent and incoming Step. If
        ``node`` already has a child, this makes ``node`` a branch point.
        """
        child = RolloutNode(id=f"n{self._node_count}", parent=node, step_in=step)
        self._node_count += 1
        node.children.append(child)
        return child

    def attach(self, parent: RolloutNode) -> RolloutNode:
        """Add a *pending* child of ``parent`` — a node with no incoming Step yet.

        A pending node is a placeholder for a continuation whose Step does not
        exist yet (a branch child before its first ``execute()``). It is filled
        later with :meth:`populate`. Like :meth:`advance`, a second child of
        the same node makes ``parent`` a branch point.
        """
        child = RolloutNode(id=f"n{self._node_count}", parent=parent, step_in=None)
        self._node_count += 1
        parent.children.append(child)
        return child

    def populate(self, node: RolloutNode, step: Step) -> RolloutNode:
        """Fill a pending ``node``'s incoming Step in place — return the node.

        The counterpart to :meth:`attach`: a branch child is first ``attach``-ed
        as a pending node, then ``populate``-d with its real continuation Step
        once that Step exists. This keeps the child's real work on the child
        node itself — no content-free placeholder Step on the path to it.
        """
        if node.step_in is not None:
            raise ValueError(
                f"node {node.id!r} already has an incoming Step — not pending"
            )
        node.step_in = step
        return node

    def find(self, node_id: str) -> RolloutNode | None:
        """Return the node carrying ``node_id``, or ``None`` if there is none."""
        for node in self.nodes():
            if node.id == node_id:
                return node
        return None

    def nodes(self) -> Iterator[RolloutNode]:
        """Yield every node in the tree, in pre-order from the root."""
        stack: list[RolloutNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))


def trajectory(leaf: RolloutNode) -> list[Step]:
    """Return the ordered Steps on the root-to-leaf path ending at ``leaf``.

    The derived view: a Trajectory is a pure function of the tree, never
    stored. The root contributes no Step (it has no incoming edge).
    """
    steps: list[Step] = []
    node: RolloutNode | None = leaf
    while node is not None and node.step_in is not None:
        steps.append(node.step_in)
        node = node.parent
    steps.reverse()
    return steps


def branch_points(tree: RolloutTree) -> list[RolloutNode]:
    """Return the nodes with more than one child, in pre-order from the root.

    A branch point is where the Rollout forked — the snapshot-and-fork
    operation produced N continuations. A linear (degree-1) rollout has none.
    """
    return [node for node in tree.nodes() if len(node.children) > 1]
