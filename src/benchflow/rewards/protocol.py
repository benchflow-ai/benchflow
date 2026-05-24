"""Core reward protocols and result type.

The Reward plane has **one canonical contract** — :class:`Reward`, matching the
architecture's ``score(node) -> VerifyResult`` shape (``docs/architecture.md``
§ "The four contracts"). A ``RolloutNode`` carries its tree context
(``node.path``, ``node.subtree``, ``node.state``), so one method expresses both
**outcome** reward (read the leaf) and **process** reward (walk the path).

:class:`RewardFunc` is the **legacy path-based shape** kept for backward
compatibility with existing benchmarks: ``score(rollout_dir: Path) -> float``.
A ``RewardFunc`` is not a ``Reward`` by itself — it is adapted to the canonical
contract via :class:`~benchflow.rewards.node.PathReward`, which reads
``rollout_dir`` from ``node.state`` and lifts the float into a tagged
:class:`VerifyResult`. New scorers should target :class:`Reward` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from benchflow.rewards.events import Granularity, RewardEvent, Space

if TYPE_CHECKING:
    from benchflow.trajectories.tree import RolloutNode


@runtime_checkable
class Reward(Protocol):
    """The canonical Reward-plane contract — ``score(node) -> VerifyResult``.

    Matches the architecture's Reward contract (``docs/architecture.md``
    § "The four contracts"). One method expresses both outcome reward (read
    the leaf node) and process reward (walk ``node.path``), since a
    ``RolloutNode`` carries its tree/path/subtree context.

    Implementations may be ad-hoc (one outcome scorer per task) or composed
    (a :class:`~benchflow.rewards.node.PathReward` adapter around a legacy
    :class:`RewardFunc`, a :class:`~benchflow.rewards.rubric.Rubric`, or an
    aggregation of :class:`~benchflow.rewards.node.NodeScorer` dimensions via
    :func:`~benchflow.rewards.node.score_node`).
    """

    async def score(self, node: RolloutNode) -> VerifyResult: ...


@runtime_checkable
class RewardFunc(Protocol):
    """Legacy path-based scoring shape — ``score(rollout_dir) -> float``.

    *Not* the canonical Reward contract — :class:`Reward` is. ``RewardFunc``
    is the historical shape benchmarks ship today (``test.sh``-style
    verifiers, ``LLMJudgeRewardFunc``, etc.); it is preserved verbatim so
    existing benchmarks keep working. Wrap a ``RewardFunc`` in
    :class:`~benchflow.rewards.node.PathReward` to expose it under the
    canonical node-based contract.
    """

    async def score(self, rollout_dir: Path) -> float: ...


@dataclass
class VerifyResult:
    """Aggregated result from a Rubric or node evaluation.

    The architecture's Reward contract result (``docs/architecture.md``,
    "The four contracts"): ``{reward, items, events, space, granularity}`` —
    ``space`` and ``granularity`` are the ``(space, granularity)`` tag the
    architecture mandates on every reward record.

    Attributes:
        reward:      Weighted aggregate score across all reward functions.
        items:       Per-function scores keyed by source name.
        events:      Reward events collected during scoring.
        error:       Error message if scoring failed, else None.
        space:       Evaluation space of the headline ``reward`` — "output"
                     (did it finish the job?), "action", "reasoning",
                     "memory", or "latent". Defaults to "output".
        granularity: "terminal" (whole trajectory) or "step" (one edge).
                     Defaults to "terminal".
    """

    reward: float
    items: dict[str, float] = field(default_factory=dict)
    events: list[RewardEvent] = field(default_factory=list)
    error: str | None = None
    space: Space = "output"
    granularity: Granularity = "terminal"
