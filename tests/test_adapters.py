"""Tests for external framework adapters (ENG-51).

Covers:
- InspectAdapter with a bare Scene
- InspectAdapter with Scene + Rubric
- ORSAdapter.verify_result_to_ors
- ORSAdapter.reward_event_to_ors
- Convenience functions to_inspect_task / to_ors_reward
- Round-trip: BenchFlow types -> adapter -> expected format
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchflow._types import Role, Scene, Turn
from benchflow.adapters.inspect_ai import InspectAdapter, to_inspect_task
from benchflow.adapters.ors import ORSAdapter, to_ors_reward
from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import VerifyResult
from benchflow.rewards.rubric import Rubric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantReward:
    """Minimal RewardFunc stub that returns a fixed score."""

    def __init__(self, value: float) -> None:
        self._value = value

    async def score(self, rollout_dir: Path) -> float:
        return self._value


def _make_scene(*, name: str = "test-scene", n_turns: int = 2) -> Scene:
    return Scene(
        name=name,
        roles=[Role(name="agent", agent="dummy", model="gpt-4")],
        turns=[Turn(role="agent", prompt=f"prompt-{i}") for i in range(n_turns)],
    )


# ---------------------------------------------------------------------------
# InspectAdapter
# ---------------------------------------------------------------------------


class TestInspectAdapter:
    def test_scene_only(self) -> None:
        scene = _make_scene()
        result = InspectAdapter(scene=scene).to_inspect_task()

        assert result["name"] == "test-scene"
        assert len(result["dataset"]) == 2
        assert result["dataset"][0] == {"input": "prompt-0", "role": "agent"}
        assert result["dataset"][1] == {"input": "prompt-1", "role": "agent"}
        assert "scorer" not in result

    def test_scene_with_rubric(self) -> None:
        scene = _make_scene()
        rubric = Rubric(
            reward_funcs=[_ConstantReward(1.0), _ConstantReward(0.5)],
            weights=[0.7, 0.3],
        )
        result = InspectAdapter(scene=scene, rubric=rubric).to_inspect_task()

        assert result["scorer"]["type"] == "benchflow_rubric"
        assert result["scorer"]["reward_funcs"] == 2
        assert result["scorer"]["weights"] == [0.7, 0.3]

    def test_empty_scene(self) -> None:
        scene = Scene(name="empty")
        result = InspectAdapter(scene=scene).to_inspect_task()
        assert result["dataset"] == []
        assert result["name"] == "empty"

    def test_none_prompt_becomes_empty_string(self) -> None:
        scene = Scene(
            name="null-prompt",
            roles=[Role(name="r", agent="a")],
            turns=[Turn(role="r", prompt=None)],
        )
        result = InspectAdapter(scene=scene).to_inspect_task()
        assert result["dataset"][0]["input"] == ""

    def test_rubric_without_weights(self) -> None:
        scene = _make_scene()
        rubric = Rubric(reward_funcs=[_ConstantReward(1.0)])
        result = InspectAdapter(scene=scene, rubric=rubric).to_inspect_task()
        assert result["scorer"]["weights"] is None


# ---------------------------------------------------------------------------
# ORSAdapter
# ---------------------------------------------------------------------------


class TestORSAdapter:
    def test_verify_result_success(self) -> None:
        event = RewardEvent(
            type="terminal",
            reward=0.8,
            source="TestReward",
            step=None,
            ts="2025-01-01T00:00:00",
        )
        vr = VerifyResult(
            reward=0.8,
            items={"TestReward": 0.8},
            events=[event],
            error=None,
        )
        ors = ORSAdapter.verify_result_to_ors(vr)

        assert ors["reward"] == 0.8
        assert ors["is_valid"] is True
        assert ors["metadata"]["items"] == {"TestReward": 0.8}
        assert len(ors["metadata"]["events"]) == 1
        assert ors["metadata"]["error"] is None

    def test_verify_result_with_error(self) -> None:
        vr = VerifyResult(reward=0.0, items={}, events=[], error="boom")
        ors = ORSAdapter.verify_result_to_ors(vr)

        assert ors["is_valid"] is False
        assert ors["metadata"]["error"] == "boom"

    def test_invalid_reward_values_are_not_valid(self) -> None:
        """Guards ENG-91 P1 dogfood ORS reward-validity regression."""
        for reward in (math.nan, 1.7, -0.2):
            ors = to_ors_reward(
                VerifyResult(
                    reward=reward,
                    items={"score": reward},
                    events=[],
                    error=None,
                )
            )

            assert ors["is_valid"] is False
            assert ors["reward"] == 0.0
            assert "invalid reward" in ors["metadata"]["error"]
            json.dumps(ors, allow_nan=False)

    def test_reward_event_to_ors(self) -> None:
        event = RewardEvent(
            type="dense",
            reward=0.5,
            source="mid-check",
            step=3,
            ts="2025-06-01T12:00:00",
        )
        d = ORSAdapter.reward_event_to_ors(event)

        assert d == {
            "type": "dense",
            "reward": 0.5,
            "source": "mid-check",
            "step": 3,
            "space": "output",
            "granularity": "terminal",
            "timestamp": "2025-06-01T12:00:00",
        }

    def test_empty_events(self) -> None:
        vr = VerifyResult(reward=1.0, items={"A": 1.0}, events=[])
        ors = ORSAdapter.verify_result_to_ors(vr)
        assert ors["metadata"]["events"] == []

    def test_reward_event_preserves_non_output_space_and_step_granularity(
        self,
    ) -> None:
        """ORS event dict carries (space, granularity) — issue #391.

        Without these tags, memory/action/reasoning process rewards become
        indistinguishable from output-space terminal rewards after export.
        """
        event = RewardEvent(
            type="dense",
            reward=0.5,
            source="memory-scorer",
            step=2,
            space="memory",
            granularity="step",
        )
        d = ORSAdapter.reward_event_to_ors(event)
        assert d["space"] == "memory"
        assert d["granularity"] == "step"

    def test_verify_result_preserves_headline_space_and_granularity(self) -> None:
        """ORS metadata carries the aggregate ``(space, granularity)`` tag."""
        vr = VerifyResult(
            reward=0.7,
            items={"action-scorer": 0.7},
            events=[],
            space="action",
            granularity="step",
        )
        ors = ORSAdapter.verify_result_to_ors(vr)
        assert ors["metadata"]["space"] == "action"
        assert ors["metadata"]["granularity"] == "step"

    def test_verify_result_events_keep_per_event_tags(self) -> None:
        """A mixed event list keeps each event's own ``(space, granularity)``."""
        events = [
            RewardEvent(
                type="terminal",
                reward=1.0,
                source="output-judge",
                space="output",
                granularity="terminal",
            ),
            RewardEvent(
                type="dense",
                reward=0.4,
                source="reasoning-scorer",
                step=1,
                space="reasoning",
                granularity="step",
            ),
            RewardEvent(
                type="dense",
                reward=0.2,
                source="memory-scorer",
                step=2,
                space="memory",
                granularity="step",
            ),
        ]
        vr = VerifyResult(reward=0.6, items={}, events=events)
        ors = ORSAdapter.verify_result_to_ors(vr)
        exported = ors["metadata"]["events"]
        assert [(e["space"], e["granularity"]) for e in exported] == [
            ("output", "terminal"),
            ("reasoning", "step"),
            ("memory", "step"),
        ]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_to_inspect_task(self) -> None:
        scene = _make_scene(name="conv-test", n_turns=1)
        result = to_inspect_task(scene)
        assert result["name"] == "conv-test"
        assert len(result["dataset"]) == 1

    def test_to_ors_reward(self) -> None:
        vr = VerifyResult(reward=0.9, items={"X": 0.9}, events=[])
        ors = to_ors_reward(vr)
        assert ors["reward"] == 0.9
        assert ors["is_valid"] is True


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_inspect_preserves_all_turns(self) -> None:
        """Every turn in the original Scene appears in the Inspect dataset."""
        scene = _make_scene(n_turns=5)
        result = to_inspect_task(scene)
        assert len(result["dataset"]) == 5
        for i, sample in enumerate(result["dataset"]):
            assert sample["input"] == f"prompt-{i}"

    def test_ors_preserves_all_events(self) -> None:
        """Every RewardEvent maps to an ORS event dict."""
        events = [
            RewardEvent(
                type="terminal", reward=float(i), source=f"src-{i}", step=i, ts=f"t{i}"
            )
            for i in range(4)
        ]
        vr = VerifyResult(reward=0.5, items={}, events=events)
        ors = to_ors_reward(vr)
        assert len(ors["metadata"]["events"]) == 4
        for i, ev in enumerate(ors["metadata"]["events"]):
            assert ev["reward"] == float(i)
            assert ev["source"] == f"src-{i}"
            assert ev["step"] == i
            assert ev["timestamp"] == f"t{i}"


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------


class TestReexport:
    def test_importable_from_benchflow(self) -> None:
        from benchflow import InspectAdapter, ORSAdapter, to_inspect_task, to_ors_reward

        assert InspectAdapter.__module__ == "benchflow.adapters.inspect_ai"
        assert ORSAdapter.__module__ == "benchflow.adapters.ors"
        assert callable(to_inspect_task)
        assert callable(to_ors_reward)
