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

import pytest

from benchflow._types import Role, Scene, Turn
from benchflow.adapters.inspect_ai import InspectAdapter, to_inspect_task
from benchflow.adapters.ors import (
    ORSAdapter,
    ors_tool_outputs_to_reward_events,
    to_ors_reward,
    write_ors_tool_outputs_jsonl,
)
from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import VerifyResult
from benchflow.rewards.rubric import Rubric


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


# InspectAdapter


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


# ORSAdapter


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

    def test_tool_outputs_to_reward_events(self) -> None:
        """ORS runtime tool rewards become verifier evidence records."""
        records = ORSAdapter.tool_outputs_to_reward_events(
            [
                {
                    "tool": "tool-call-check",
                    "reward": 0.25,
                    "step": 1,
                    "tool_call_id": "call-1",
                    "timestamp": "2026-06-05T00:00:00Z",
                },
                {
                    "tool": "ors-terminal",
                    "reward": {"reward": 0.88},
                    "step": 2,
                    "finished": True,
                    "toolCallId": "call-2",
                },
            ]
        )

        assert records == [
            {
                "type": "dense",
                "reward": 0.25,
                "source": "tool-call-check",
                "step": 1,
                "space": "action",
                "granularity": "step",
                "timestamp": "2026-06-05T00:00:00Z",
                "tool_call_id": "call-1",
            },
            {
                "type": "terminal",
                "reward": 0.88,
                "source": "ors-terminal",
                "step": 2,
                "space": "output",
                "granularity": "terminal",
                "tool_call_id": "call-2",
                "finished": True,
            },
        ]

    def test_write_tool_outputs_jsonl(self, tmp_path: Path) -> None:
        """The runtime helper writes the artifact consumed by ``ors-episode``."""
        output_path = tmp_path / "trajectory" / "ors-rewards.jsonl"

        records = write_ors_tool_outputs_jsonl(
            [
                {"tool": "search", "reward": 0.2},
                {"tool": "submit", "reward": 0.8, "done": True},
            ],
            output_path,
        )

        assert output_path.exists()
        assert [
            json.loads(line) for line in output_path.read_text().splitlines()
        ] == records
        assert records[-1]["type"] == "terminal"
        assert records[-1]["finished"] is True

    def test_tool_outputs_reject_invalid_rewards(self) -> None:
        """Runtime ORS reward evidence fails closed before verifier handoff."""
        for output in (
            {"reward": -0.1},
            {"reward": 1.1},
            {"reward": float("nan")},
            {"reward": "not-a-number"},
        ):
            with pytest.raises(ValueError, match="reward"):
                ORSAdapter.tool_outputs_to_reward_events([output])

    def test_tool_output_convenience_function(self) -> None:
        records = ors_tool_outputs_to_reward_events(
            [{"tool": "submit", "result": {"reward": 0.7}, "finished": True}]
        )
        assert records == [
            {
                "type": "terminal",
                "reward": 0.7,
                "source": "submit",
                "step": 1,
                "space": "output",
                "granularity": "terminal",
                "finished": True,
            }
        ]

    def test_finished_record_is_never_dense_terminal_contradiction(self) -> None:
        """A finished record (no explicit type) is forced terminal/terminal."""
        records = ORSAdapter.tool_outputs_to_reward_events(
            [{"tool": "submit", "reward": 0.9, "finished": True}]
        )
        assert len(records) == 1
        rec = records[0]
        assert rec["type"] == "terminal"
        assert rec["granularity"] == "terminal"
        # The forbidden internally-contradictory shape must never be emitted.
        assert not (rec["type"] == "dense" and rec["granularity"] == "terminal")

    def test_finished_with_explicit_dense_type_is_rejected(self) -> None:
        """An explicit non-terminal type on a finished record is contradictory."""
        with pytest.raises(ValueError, match=r"finished.*non-terminal type"):
            ORSAdapter.tool_outputs_to_reward_events(
                [{"tool": "submit", "reward": 0.9, "type": "dense", "finished": True}]
            )

    def test_finished_with_explicit_step_granularity_is_rejected(self) -> None:
        """An explicit step granularity on a finished record is contradictory."""
        with pytest.raises(ValueError, match=r"finished.*non-terminal granularity"):
            ORSAdapter.tool_outputs_to_reward_events(
                [{"tool": "submit", "reward": 0.9, "granularity": "step", "done": True}]
            )


# Convenience functions


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

    def test_top_level_ors_tool_output_reexport(self) -> None:
        from benchflow import write_ors_tool_outputs_jsonl as top_level_writer

        assert top_level_writer is write_ors_tool_outputs_jsonl


# Round-trip tests


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


# Top-level re-export


class TestReexport:
    def test_importable_from_benchflow(self) -> None:
        from benchflow import InspectAdapter, ORSAdapter, to_inspect_task, to_ors_reward

        assert InspectAdapter.__module__ == "benchflow.adapters.inspect_ai"
        assert ORSAdapter.__module__ == "benchflow.adapters.ors"
        assert callable(to_inspect_task)
        assert callable(to_ors_reward)
