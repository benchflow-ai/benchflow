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

This module also provides :class:`PathReward` — the adapter that lifts a
legacy path-based :class:`~benchflow.rewards.protocol.RewardFunc` (or a
:class:`~benchflow.rewards.rubric.Rubric`) into the canonical
:class:`~benchflow.rewards.protocol.Reward` contract. The adapter reads
``rollout_dir`` from ``node.state[PATH_STATE_KEY]`` and returns a tagged
``VerifyResult``, so existing path-based scorers compose under the same
contract as native ``NodeScorer``-based ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Protocol, cast, runtime_checkable

from benchflow.rewards.events import Granularity, RewardEvent, Space
from benchflow.rewards.protocol import RewardFunc, VerifyResult
from benchflow.trajectories.tree import RolloutNode

_OUTPUT_SPACE: Final[Space] = "output"


def _as_score(value: Any) -> float | None:
    """Coerce a reward-map value to a float score, or ``None`` if it isn't one.

    Centralises the "real number, not a bool" guard the reward map needs in
    three places (headline, extra scalars, rubric scores) — ``bool`` is an
    ``int`` subclass and must not slip through as ``1.0``/``0.0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def verify_result_from_reward_map(
    rewards: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> VerifyResult:
    """Lift a validated verifier reward ``dict`` into a canonical ``VerifyResult``.

    This is the **single** dict→``VerifyResult`` conversion point. The legacy
    ``Verifier`` produces ``{"reward": float, "rubric": [...], ...scalars...}``;
    Phase 1 lifts it here, during scoring, so the ``VerifyResult`` is the live
    source of truth — not something re-derived at export time. Unlike a bare
    lift, it produces the full ``events`` list — one ``terminal`` Output event
    for the headline reward plus one ``process`` event per rubric item carrying
    that item's ``(space, granularity)`` — so ``rewards.jsonl`` /
    ``verifiers.jsonl`` tags are *sourced* from the result, not defaulted at the
    writer.

    Note the asymmetry the architecture mandates (``docs/architecture.md``,
    "Evaluation"): the headline reward is the Output/terminal outcome; rubric
    items default to ``(output, step)`` — process signals along the path.

    A ``None`` map (verifier crashed/timed out) yields ``reward=0.0`` with
    ``error`` populated, so a downstream ``reward_valid`` flag reads ``False``.
    """
    if rewards is None:
        return VerifyResult(
            reward=0.0,
            items={},
            events=[],
            error=error or "no rewards",
            space=_OUTPUT_SPACE,
            granularity="terminal",
        )

    reward = rewards.get("reward")
    headline = _as_score(reward) or 0.0

    items: dict[str, float] = {}
    events: list[RewardEvent] = []

    if reward is not None:
        events.append(
            RewardEvent(
                type="terminal",
                reward=headline,
                source="verifier",
                space=_OUTPUT_SPACE,
                granularity="terminal",
            )
        )

    for key, value in rewards.items():
        if key in ("reward", "rubric", "space", "granularity"):
            continue
        score = _as_score(value)
        if score is not None:
            items[str(key)] = score

    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            rubric_item = cast(dict[str, Any], item)
            score = _as_score(rubric_item.get("score"))
            if score is None:
                continue
            name = str(rubric_item.get("name") or f"rubric_{i}")
            items[name] = score
            events.append(
                RewardEvent(
                    type="process",
                    reward=score,
                    source=name,
                    step=i,
                    space=cast(Space, rubric_item.get("space", "output")),
                    granularity=cast(
                        Granularity, rubric_item.get("granularity", "step")
                    ),
                )
            )

    return VerifyResult(
        reward=headline,
        items=items,
        events=events,
        error=error,
        space=_OUTPUT_SPACE,
        granularity="terminal",
    )

#: ``node.state`` key under which a rollout's on-disk directory is recorded.
#: :class:`PathReward` reads this to bridge node-scoped scoring to the legacy
#: ``score(rollout_dir: Path) -> float`` shape.
PATH_STATE_KEY: Final = "rollout_dir"


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


class PathReward:
    """Adapter — lift a legacy path-based ``RewardFunc`` into ``Reward``.

    ``PathReward`` makes any :class:`~benchflow.rewards.protocol.RewardFunc`
    (including a :class:`~benchflow.rewards.rubric.Rubric` and built-ins like
    ``TestRewardFunc`` / ``LLMJudgeRewardFunc``) satisfy the canonical
    :class:`~benchflow.rewards.protocol.Reward` contract: ``score(node) ->
    VerifyResult``. It reads the rollout's on-disk directory from
    ``node.state[PATH_STATE_KEY]`` and lifts the path-based result into a
    tagged ``VerifyResult``.

    A wrapped function returning a bare ``float`` is promoted to a tagged
    ``VerifyResult(reward=…, items={source: reward}, events=[terminal-event],
    space="output", granularity="terminal")``. A wrapped function already
    returning a ``VerifyResult`` (e.g. a ``Rubric``) is passed through, with
    the headline reward recorded as the canonical outcome.

    If ``node.state[PATH_STATE_KEY]`` is missing, the adapter records the
    error on the returned ``VerifyResult`` (``reward=0.0``) rather than
    raising — the canonical contract distinguishes "nobody scored" from
    "scored 0.0" by populating ``error``.
    """

    def __init__(self, func: RewardFunc, source: str | None = None) -> None:
        self._func = func
        self.source = source or type(func).__name__

    async def score(self, node: RolloutNode) -> VerifyResult:
        rollout_dir = node.state.get(PATH_STATE_KEY)
        if rollout_dir is None:
            return VerifyResult(
                reward=0.0,
                error=(
                    f"PathReward[{self.source}]: node.state["
                    f"{PATH_STATE_KEY!r}] is missing — cannot adapt "
                    "path-based RewardFunc"
                ),
                space=_OUTPUT_SPACE,
                granularity="terminal",
            )
        result = await self._func.score(Path(rollout_dir))
        if isinstance(result, VerifyResult):
            # Already canonical — preserve the inner result, just pin the
            # outer (space, granularity) tag for the headline.
            return VerifyResult(
                reward=result.reward,
                items=result.items,
                events=result.events,
                error=result.error,
                space=result.space,
                granularity=result.granularity,
            )
        reward = float(result)
        event = RewardEvent(
            type="terminal",
            reward=reward,
            source=self.source,
            space=_OUTPUT_SPACE,
            granularity="terminal",
        )
        return VerifyResult(
            reward=reward,
            items={self.source: reward},
            events=[event],
            space=_OUTPUT_SPACE,
            granularity="terminal",
        )
