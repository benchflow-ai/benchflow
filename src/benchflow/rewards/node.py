"""Node-scoped scoring — the Reward plane's ``score(node)`` path.

The architecture's ``Reward`` contract is ``score(node) -> VerifyResult``
(``docs/architecture.md`` § "The four contracts"). :func:`score_node` is the
contract-shaped entry point: it runs a set of per-dimension scorers over a
``RolloutNode`` and aggregates their tagged events into a ``VerifyResult``.

A scorer examines the node — the leaf for an **outcome** reward, its
root-to-leaf path (via ``trajectory(node)``) for a **process** reward — and
emits a ``RewardEvent`` tagged ``(space, granularity)``.

The headline ``VerifyResult.reward`` is the outcome signal (the Output space —
"did it finish the job?"); process-space events ride along for credit
assignment up the tree.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.tree import RolloutNode

_OUTPUT_SPACE = "output"


@runtime_checkable
class NodeScorer(Protocol):
    """Internal per-dimension scorer for one ``RolloutNode``.

    *This is not the Reward contract.* The architecture's Reward contract is
    ``score(node) -> VerifyResult`` (:func:`score_node`); a ``NodeScorer`` is
    one scoring dimension inside it — it returns a single tagged
    ``RewardEvent``, and :func:`score_node` aggregates a list of them into the
    contract-shaped ``VerifyResult``.

    An *outcome* scorer reads ``node`` (typically the leaf) directly; a
    *process* scorer walks ``trajectory(node)`` — the root-to-leaf path.
    """

    async def score(self, node: RolloutNode) -> RewardEvent: ...


async def score_node(node: RolloutNode, scorers: list[NodeScorer]) -> VerifyResult:
    """Score a tree node — the Reward plane's ``score(node)`` entry point.

    Runs each scorer over ``node``, collecting its tagged ``RewardEvent``.
    ``VerifyResult.reward`` is the outcome signal — the first Output-space
    event's reward. The result is tagged ``(space="output",
    granularity="terminal")``: a node score is the terminal outcome of that
    node's subtree.

    When **no** scorer reports an Output-space event, ``reward`` is ``0.0``
    *and* ``error`` is populated — a node with no outcome scorer is "nobody
    scored", which must not be confused with an honest "scored 0.0". Callers
    that need a number can still read ``reward``; callers that need to know
    whether scoring happened check ``error``.

    Every scorer's reward is recorded per-source in ``items``; all events
    (including process-space ones) are carried in ``events``. If two scorers
    share a ``source`` name, the **last** one wins in ``items`` (dict
    semantics) — but both events are kept in ``events``, so no signal is lost.
    """
    events = [await scorer.score(node) for scorer in scorers]
    items = {event.source: event.reward for event in events}
    outcome_events = [e for e in events if e.space == _OUTPUT_SPACE]
    error = (
        None
        if outcome_events
        else "no output-space scorer ran — reward 0.0 means 'nobody scored'"
    )
    outcome = outcome_events[0].reward if outcome_events else 0.0
    return VerifyResult(
        reward=outcome,
        items=items,
        events=events,
        error=error,
        space=_OUTPUT_SPACE,
        granularity="terminal",
    )
