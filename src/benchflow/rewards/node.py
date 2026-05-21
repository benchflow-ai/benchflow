"""Node-scoped scoring — the Reward plane's ``score(node)`` path.

The architecture's ``Reward`` contract is ``score(node) -> VerifyResult``
(``docs/architecture.md`` § "The four contracts"). A scorer examines a
``RolloutNode`` — the leaf for an **outcome** reward, its root-to-leaf path
(via ``trajectory(node)``) for a **process** reward — and emits a
``RewardEvent`` tagged ``(space, granularity)``.

``score_node`` is the entry point: it runs the scorers and aggregates their
tagged events into a ``VerifyResult``. The headline ``VerifyResult.reward``
is the outcome signal (the Output space — "did it finish the job?");
process-space events ride along for credit assignment up the tree.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.tree import RolloutNode

_OUTPUT_SPACE = "output"


@runtime_checkable
class NodeScorer(Protocol):
    """Scores one ``RolloutNode``, emitting a tagged ``RewardEvent``.

    An *outcome* scorer reads ``node`` (typically the leaf) directly; a
    *process* scorer walks ``trajectory(node)`` — the root-to-leaf path.
    """

    async def score(self, node: RolloutNode) -> RewardEvent: ...


async def score_node(
    node: RolloutNode, scorers: list[NodeScorer]
) -> VerifyResult:
    """Score a tree node — the Reward plane's ``score(node)`` entry point.

    Runs each scorer over ``node``, collecting its tagged ``RewardEvent``.
    ``VerifyResult.reward`` is the outcome signal — the first Output-space
    event's reward, or ``0.0`` if no scorer reports one. Every scorer's
    reward is recorded per-source in ``items``; all events (including
    process-space ones) are carried in ``events``.
    """
    events = [await scorer.score(node) for scorer in scorers]
    items = {event.source: event.reward for event in events}
    outcome = next(
        (event.reward for event in events if event.space == _OUTPUT_SPACE),
        0.0,
    )
    return VerifyResult(reward=outcome, items=items, events=events)
