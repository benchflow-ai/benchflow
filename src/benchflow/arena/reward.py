"""Per-seat reward vector from one shared aggregate — Seam 4.

The canonical Reward contract (``benchflow.rewards.protocol.Reward``) is already
*per-node*, and each seat in an arena run is one ``RolloutNode`` — so per-seat
scoring needs only a new, opt-in scorer, not a change to the scalar
``RewardFunc`` path every single-agent benchmark ships. Kept here as a pure
function of the floor's standings so it is trivially testable without the
rollout tree; wrap it in a ``Reward`` adapter to plug into the reward plane.
"""

from __future__ import annotations

import enum

__all__ = ["FloorMode", "SharedEnvReward"]


class FloorMode(enum.StrEnum):
    PVP = "pvp"  # competitive: each seat scored by its own net result
    COOP = "coop"  # cooperative: every seat shares one joint outcome


class SharedEnvReward:
    """Reduce one shared ``standings`` map to a per-seat reward vector."""

    def __init__(
        self, starting_bankroll: int = 1000, mode: FloorMode = FloorMode.PVP
    ) -> None:
        self.start = starting_bankroll
        self.mode = mode

    def score(self, standings: dict[str, int]) -> dict[str, float]:
        if not standings:
            return {}
        if self.mode is FloorMode.COOP:
            # joint reduction: the floor succeeds together — the worst seat's
            # net is everyone's reward (swap for sum/mean as a task needs).
            joint = float(min(standings.values()) - self.start)
            return {seat: joint for seat in standings}
        # PvP: each seat's net chips (zero-sum floors sum to ~0).
        return {seat: float(chips - self.start) for seat, chips in standings.items()}
