"""Built-in reward functions."""

from __future__ import annotations

from dataclasses import dataclass

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.rubric import RewardContext


@dataclass
class TestRewardFunc:
    """Compatibility reward function for ``test.sh -> reward.txt`` tasks."""

    __test__ = False

    reward_path: str = "verifier/reward.txt"

    async def score(self, ctx: RewardContext) -> RewardEvent:
        path = ctx.rollout_dir / self.reward_path
        value = float(path.read_text().strip())
        return RewardEvent(
            type="terminal",
            source="test_reward_func",
            value=value,
            tag="reward",
        )
