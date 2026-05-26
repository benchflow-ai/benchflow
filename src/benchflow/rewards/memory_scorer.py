"""Memory-space scoring — capability #1 (SkillsBench).

The architecture's **Memory space** (``docs/architecture.md`` § "Evaluation —
the five spaces") asks one question: *did the agent update its memory / skills
correctly?* — and answers it by **diffing the store**. Skills *are* memory
(Han, ``architecture.md`` § "skill 是属于 memory").

This module is a single :class:`~benchflow.rewards.node.NodeScorer` —
:class:`MemoryScorer` — plus the pure diff it rests on (:func:`skill_delta`,
:class:`SkillDelta`). The scorer reads a before/after skill snapshot recorded
on a ``RolloutNode`` and emits one ``RewardEvent`` tagged ``space="memory"``,
so :func:`~benchflow.rewards.node.score_node` aggregates it like any other
scoring dimension. This is purely additive — nothing in ``node.py`` or
``events.py`` changes.

The skill/memory-delta contract on the node
-------------------------------------------
The Memory scorer reads ``node.state[MEMORY_STATE_KEY]`` — a plain ``dict``
the Environment/Agent plane is expected to record when it snapshots the skill
store before and after a Rollout (the SkillsBench learner store). The contract
is deliberately minimal:

``before``  : ``dict[str, str]`` — skill name → version/content hash *before*
              the node's work. A skill "changes" iff its value changes; the
              value can be a semantic version, a content hash, or the skill
              body itself — the scorer only checks (in)equality.
``after``   : ``dict[str, str]`` — the same snapshot *after* the node's work.
``expected``: ``list[str]`` *(optional)* — the hidden fixture: the set of
              skill names the task expects to have changed. When present the
              scorer scores *correctness* (precision × recall over the
              expected set). When absent the scorer can only score
              *activity* — 1.0 for any well-formed change, 0.0 for none.

All three keys are optional; a node that never recorded a delta scores 0.0
rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from benchflow.rewards.events import RewardEvent
from benchflow.trajectories.tree import RolloutNode

#: ``node.state`` key under which the before/after skill snapshot is recorded.
MEMORY_STATE_KEY: Final = "memory_delta"

_MEMORY_SPACE: Final = "memory"


@dataclass(frozen=True)
class SkillDelta:
    """The diff of a skill/memory store — added, updated, and removed skills.

    A value type: two deltas with the same three sets are equal. ``updated``
    is a skill present in both snapshots whose value differs; ``added`` /
    ``removed`` are presence changes.
    """

    added: set[str]
    updated: set[str]
    removed: set[str]

    @property
    def changed(self) -> set[str]:
        """The union of all three change sets — every skill that moved."""
        return self.added | self.updated | self.removed


def skill_delta(*, before: dict[str, str], after: dict[str, str]) -> SkillDelta:
    """Diff two skill-store snapshots — *the* Memory-space primitive.

    A skill is *added* when it appears only in ``after``, *removed* when only
    in ``before``, and *updated* when present in both with a different value.
    """
    before_keys = set(before)
    after_keys = set(after)
    return SkillDelta(
        added=after_keys - before_keys,
        removed=before_keys - after_keys,
        updated={
            name for name in before_keys & after_keys if before[name] != after[name]
        },
    )


def _score_delta(delta: SkillDelta, expected: set[str] | None) -> float:
    """Turn a :class:`SkillDelta` into a scalar Memory-space reward.

    With a hidden fixture (``expected`` not ``None``) the reward is the
    F1-style product of **precision** (changes that were expected) and
    **recall** (expected skills that changed) — a spurious change drags
    precision down, a missed skill drags recall down. The empty-expected,
    empty-change case is a correct no-op and scores 1.0.

    Without a fixture the scorer cannot judge correctness, only activity:
    1.0 for any change, 0.0 for none.
    """
    changed = delta.changed
    if expected is None:
        return 1.0 if changed else 0.0

    if not expected:
        # The task expects the store left alone — a no-op is perfect.
        return 0.0 if changed else 1.0

    hits = changed & expected
    recall = len(hits) / len(expected)
    precision = len(hits) / len(changed) if changed else 0.0
    return recall * precision


class MemoryScorer:
    """Memory-space :class:`~benchflow.rewards.node.NodeScorer`.

    Reads the before/after skill snapshot from ``node.state[MEMORY_STATE_KEY]``
    (see the module docstring for the contract) and emits one ``RewardEvent``
    tagged ``space="memory"``, ``granularity="terminal"``.

    A node that never recorded a delta scores 0.0 — *nothing changed* — rather
    than failing; the Memory space simply has no signal for that node.
    """

    def __init__(self, source: str = "memory") -> None:
        self.source = source

    async def score(self, node: RolloutNode) -> RewardEvent:
        """Score ``node`` in the Memory space — diff its skill store."""
        record = node.state.get(MEMORY_STATE_KEY)
        if not isinstance(record, dict):
            reward = 0.0
        else:
            before = dict(record.get("before") or {})
            after = dict(record.get("after") or {})
            raw_expected = record.get("expected")
            expected = None if raw_expected is None else set(raw_expected)
            reward = _score_delta(skill_delta(before=before, after=after), expected)

        return RewardEvent(
            type="terminal",
            reward=reward,
            source=self.source,
            space=_MEMORY_SPACE,
            granularity="terminal",
        )
