"""Tests for the unified Reward contract — guards the fix from issue #397.

Pins the Reward plane's single canonical contract: ``Reward.score(node) ->
VerifyResult`` (``docs/architecture.md`` § "The four contracts"). Legacy
path-based ``RewardFunc``s adapt into the canonical contract via
``PathReward``, which reads ``rollout_dir`` from ``node.state[PATH_STATE_KEY]``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.rewards import (
    PATH_STATE_KEY,
    PathReward,
    Reward,
    RewardFunc,
    Rubric,
    TestRewardFunc,
    VerifyResult,
)
from benchflow.rewards.events import RewardEvent
from benchflow.trajectories.tree import RolloutNode


class _ConstFunc:
    """Path-based RewardFunc returning a constant — deterministic test fixture."""

    def __init__(self, value: float) -> None:
        self._value = value

    async def score(self, rollout_dir: Path) -> float:
        return self._value


class _NativeNodeReward:
    """A Reward implemented directly against the canonical node contract."""

    async def score(self, node: RolloutNode) -> VerifyResult:
        return VerifyResult(
            reward=float(node.state.get("score", 0.0)),
            space="output",
            granularity="terminal",
        )


# ---------------------------------------------------------------------------
# Canonical contract
# ---------------------------------------------------------------------------


def test_reward_protocol_is_runtime_checkable_and_node_based() -> None:
    """The canonical Reward Protocol accepts node-based scorers."""
    assert isinstance(_NativeNodeReward(), Reward)


def test_path_reward_satisfies_canonical_reward_contract() -> None:
    """A PathReward-wrapped RewardFunc is a Reward — the two paths unify here."""
    adapter = PathReward(_ConstFunc(0.7))
    assert isinstance(adapter, Reward)


async def test_legacy_reward_func_must_be_adapted_before_acting_as_reward(
    tmp_path: Path,
) -> None:
    """A bare path-based ``RewardFunc`` is not usable as a ``Reward``.

    ``Reward`` and ``RewardFunc`` are structurally similar (both expose a
    ``score`` method) so ``isinstance`` cannot tell them apart at runtime —
    that is why #397 surfaced in the first place. The real test is
    behavioural: feeding a ``RolloutNode`` into a legacy ``RewardFunc`` fails
    (no ``rollout_dir`` attr / not a ``Path``), and wrapping it in
    ``PathReward`` is what makes it actually work under the canonical
    contract. Pinning this stops a future refactor from collapsing the two
    protocols and re-opening the issue.
    """
    bare = TestRewardFunc()
    assert isinstance(bare, RewardFunc)  # has the legacy ``score`` method shape
    node = RolloutNode(id="leaf", state={PATH_STATE_KEY: str(tmp_path)})

    # Direct node-call fails — TestRewardFunc expects a Path-like rollout_dir.
    with pytest.raises((AttributeError, TypeError)):
        await bare.score(node)  # type: ignore[arg-type]

    # Wrapped, it works and returns the canonical VerifyResult.
    wrapped = PathReward(bare)
    result = await wrapped.score(node)
    assert isinstance(result, VerifyResult)


# ---------------------------------------------------------------------------
# PathReward adapter behaviour
# ---------------------------------------------------------------------------


async def test_path_reward_reads_rollout_dir_from_node_state(tmp_path: Path) -> None:
    """The adapter bridges node-scoped scoring to path-based RewardFuncs."""
    node = RolloutNode(id="leaf", state={PATH_STATE_KEY: str(tmp_path)})
    adapter = PathReward(_ConstFunc(0.42), source="const")

    result = await adapter.score(node)

    assert isinstance(result, VerifyResult)
    assert result.reward == 0.42
    assert result.items == {"const": 0.42}
    assert result.space == "output"
    assert result.granularity == "terminal"
    assert result.error is None
    assert len(result.events) == 1
    event = result.events[0]
    assert event.source == "const"
    assert event.reward == 0.42


async def test_path_reward_missing_rollout_dir_records_error() -> None:
    """A node with no rollout_dir scores 0.0 with error — never raises.

    Distinguishes 'nobody scored' (error populated) from an honest 0.0 score,
    matching ``score_node``'s contract.
    """
    node = RolloutNode(id="leaf")  # no state
    adapter = PathReward(_ConstFunc(1.0), source="x")

    result = await adapter.score(node)

    assert result.reward == 0.0
    assert result.error is not None
    assert "rollout_dir" in result.error


async def test_path_reward_passes_through_verify_result(tmp_path: Path) -> None:
    """A Rubric returns ``VerifyResult``; the adapter must not double-wrap it."""
    rubric = Rubric(reward_funcs=[_ConstFunc(0.5), _ConstFunc(1.0)])
    node = RolloutNode(id="leaf", state={PATH_STATE_KEY: str(tmp_path)})

    result = await PathReward(rubric, source="rubric").score(node)

    assert isinstance(result, VerifyResult)
    # Rubric's weighted mean with equal weights — 0.75, threaded through.
    assert result.reward == pytest.approx(0.75)
    # Underlying per-function items are preserved (not collapsed to {"rubric"}).
    assert "_ConstFunc" in result.items or "_ConstFunc_0" in result.items


async def test_path_reward_default_source_is_inner_class_name(
    tmp_path: Path,
) -> None:
    """Without an explicit source, the adapter uses the wrapped class name."""
    node = RolloutNode(id="leaf", state={PATH_STATE_KEY: str(tmp_path)})

    result = await PathReward(_ConstFunc(0.9)).score(node)

    assert "_ConstFunc" in result.items


async def test_path_reward_promotes_float_to_tagged_terminal_event(
    tmp_path: Path,
) -> None:
    """A bare float becomes a terminal Output-space ``RewardEvent``.

    The architecture mandates every reward record be tagged ``(space,
    granularity)`` — the adapter is the seam that puts that tag onto legacy
    path-based scorers' raw floats.
    """
    node = RolloutNode(id="leaf", state={PATH_STATE_KEY: str(tmp_path)})

    result = await PathReward(_ConstFunc(0.6), source="t").score(node)

    assert len(result.events) == 1
    event: RewardEvent = result.events[0]
    assert event.type == "terminal"
    assert event.space == "output"
    assert event.granularity == "terminal"
