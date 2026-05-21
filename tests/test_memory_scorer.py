"""Tests for the Memory-space NodeScorer — capability #1 (SkillsBench).

The architecture's Memory space (``docs/architecture.md`` § "Evaluation — the
five spaces") asks: *did the agent update its memory / skills correctly?* —
answered by diffing the skill/memory store. Skills *are* memory (Han).

``MemoryScorer`` is a :class:`~benchflow.rewards.node.NodeScorer`: it reads a
before/after skill snapshot off ``node.state`` and emits a ``RewardEvent``
tagged ``space="memory"`` so :func:`~benchflow.rewards.node.score_node`
aggregates it like any other scoring dimension.
"""

import pytest

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.memory_scorer import (
    MEMORY_STATE_KEY,
    MemoryScorer,
    SkillDelta,
    skill_delta,
)
from benchflow.rewards.node import NodeScorer, score_node
from benchflow.trajectories.tree import RolloutNode

# ---------------------------------------------------------------------------
# skill_delta — the diff over the store
# ---------------------------------------------------------------------------


def test_skill_delta_detects_added_skill():
    delta = skill_delta(before={}, after={"git-bisect": "v1"})
    assert delta.added == {"git-bisect"}
    assert delta.updated == set()
    assert delta.removed == set()


def test_skill_delta_detects_updated_skill():
    delta = skill_delta(before={"git-bisect": "v1"}, after={"git-bisect": "v2"})
    assert delta.updated == {"git-bisect"}
    assert delta.added == set()
    assert delta.removed == set()


def test_skill_delta_detects_removed_skill():
    delta = skill_delta(before={"git-bisect": "v1"}, after={})
    assert delta.removed == {"git-bisect"}


def test_skill_delta_unchanged_skill_is_not_a_change():
    delta = skill_delta(
        before={"git-bisect": "v1"}, after={"git-bisect": "v1"}
    )
    assert delta.added == set()
    assert delta.updated == set()
    assert delta.removed == set()
    assert not delta.changed


def test_skill_delta_changed_is_the_union_of_all_changes():
    delta = skill_delta(
        before={"keep": "v1", "drop": "v1", "bump": "v1"},
        after={"keep": "v1", "bump": "v2", "new": "v1"},
    )
    assert delta.changed == {"drop", "bump", "new"}


# ---------------------------------------------------------------------------
# MemoryScorer — emits a memory-space RewardEvent
# ---------------------------------------------------------------------------


async def test_memory_scorer_satisfies_the_node_scorer_protocol():
    assert isinstance(MemoryScorer(), NodeScorer)


async def test_memory_scorer_emits_a_memory_space_event():
    node = RolloutNode(
        id="leaf",
        state={MEMORY_STATE_KEY: {"before": {}, "after": {"s": "v1"}}},
    )
    event = await MemoryScorer().score(node)
    assert isinstance(event, RewardEvent)
    assert event.space == "memory"
    assert event.granularity == "terminal"


async def test_memory_scorer_rewards_a_correct_skill_update():
    """Expected skills changed, nothing spurious — full reward."""
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {"git-bisect": "v1"},
                "after": {"git-bisect": "v2"},
                "expected": ["git-bisect"],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 1.0


async def test_memory_scorer_rewards_a_correct_skill_addition():
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {},
                "after": {"new-skill": "v1"},
                "expected": ["new-skill"],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 1.0


async def test_memory_scorer_zero_when_expected_skill_untouched():
    """The agent should have updated a skill and did not — no reward."""
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {"git-bisect": "v1"},
                "after": {"git-bisect": "v1"},
                "expected": ["git-bisect"],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 0.0


async def test_memory_scorer_penalises_a_spurious_change():
    """A change to a skill that should NOT have changed is reward-hacking-ish.

    One expected skill correctly updated, one unrelated skill spuriously
    touched: precision is 1/2, so the reward is 0.5.
    """
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {"git-bisect": "v1", "unrelated": "v1"},
                "after": {"git-bisect": "v2", "unrelated": "v2"},
                "expected": ["git-bisect"],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 0.5


async def test_memory_scorer_partial_credit_for_partial_recall():
    """Two skills expected, only one changed: recall 1/2, precision 1/1 -> 0.5."""
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {"a": "v1", "b": "v1"},
                "after": {"a": "v2", "b": "v1"},
                "expected": ["a", "b"],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 0.5


async def test_memory_scorer_no_expected_means_no_change_is_correct():
    """When the task expects no skill change, leaving the store alone scores 1.0."""
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {"a": "v1"},
                "after": {"a": "v1"},
                "expected": [],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 1.0


async def test_memory_scorer_no_expected_penalises_a_spurious_change():
    """Task expects no change but the agent touched the store -> reward 0.0."""
    node = RolloutNode(
        id="leaf",
        state={
            MEMORY_STATE_KEY: {
                "before": {"a": "v1"},
                "after": {"a": "v2"},
                "expected": [],
            }
        },
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 0.0


async def test_memory_scorer_missing_state_scores_zero_not_crash():
    """A node with no recorded memory delta — scorer must not crash."""
    event = await MemoryScorer().score(RolloutNode(id="leaf"))
    assert event.reward == 0.0
    assert event.space == "memory"


async def test_memory_scorer_missing_expected_falls_back_to_any_change():
    """No 'expected' fixture: reward any well-formed skill change as 1.0.

    Without a hidden fixture the scorer cannot judge *correctness*, only
    *activity* — it rewards that the store was updated at all.
    """
    node = RolloutNode(
        id="leaf",
        state={MEMORY_STATE_KEY: {"before": {}, "after": {"s": "v1"}}},
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 1.0


async def test_memory_scorer_missing_expected_no_change_scores_zero():
    """No fixture and no change — nothing happened, no reward."""
    node = RolloutNode(
        id="leaf",
        state={MEMORY_STATE_KEY: {"before": {"s": "v1"}, "after": {"s": "v1"}}},
    )
    event = await MemoryScorer().score(node)
    assert event.reward == 0.0


async def test_memory_scorer_source_is_overridable():
    scorer = MemoryScorer(source="skills")
    event = await scorer.score(RolloutNode(id="leaf"))
    assert event.source == "skills"


async def test_memory_scorer_default_source():
    event = await MemoryScorer().score(RolloutNode(id="leaf"))
    assert event.source == "memory"


# ---------------------------------------------------------------------------
# Aggregation — score_node treats it like any other scorer
# ---------------------------------------------------------------------------


class _OutcomeScorer:
    source = "outcome"

    async def score(self, node: RolloutNode) -> RewardEvent:
        return RewardEvent(
            type="terminal",
            reward=float(node.state.get("reward", 0.0)),
            source=self.source,
            space="output",
        )


async def test_score_node_aggregates_the_memory_scorer():
    """score_node carries the memory event alongside the output reward."""
    node = RolloutNode(
        id="leaf",
        state={
            "reward": 1.0,
            MEMORY_STATE_KEY: {
                "before": {},
                "after": {"s": "v1"},
                "expected": ["s"],
            },
        },
    )
    result = await score_node(node, [_OutcomeScorer(), MemoryScorer()])
    # output reward is the headline — memory rides along, does not displace it
    assert result.reward == 1.0
    assert result.items["memory"] == 1.0
    assert any(e.space == "memory" for e in result.events)


async def test_memory_scorer_alone_does_not_provide_an_outcome_signal():
    """A node scored only in the memory space has no output reward.

    score_node must flag 'nobody scored' the outcome — the memory space is a
    process-style signal, not the terminal/verifiable reward.
    """
    node = RolloutNode(
        id="leaf",
        state={MEMORY_STATE_KEY: {"before": {}, "after": {"s": "v1"}}},
    )
    result = await score_node(node, [MemoryScorer()])
    assert result.reward == 0.0
    assert result.error is not None
    assert "nobody scored" in result.error


# ---------------------------------------------------------------------------
# SkillDelta value semantics
# ---------------------------------------------------------------------------


def test_skill_delta_is_a_value_type():
    a = SkillDelta(added={"x"}, updated=set(), removed=set())
    b = SkillDelta(added={"x"}, updated=set(), removed=set())
    assert a == b


@pytest.mark.parametrize(
    ("before", "after", "expected_changed"),
    [
        ({}, {}, set()),
        ({"a": "1"}, {"a": "1"}, set()),
        ({}, {"a": "1"}, {"a"}),
        ({"a": "1"}, {}, {"a"}),
        ({"a": "1"}, {"a": "2"}, {"a"}),
    ],
)
def test_skill_delta_changed_property(before, after, expected_changed):
    assert skill_delta(before=before, after=after).changed == expected_changed
