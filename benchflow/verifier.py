import json
from typing import Dict, Optional

class Verifier:
    def __init__(self, task_config: Dict):
        self.task_config = task_config
        self.reward_range = task_config.get('reward_range', [0.0, 1.0])

    def validate_reward(self, reward: float) -> bool:
        return self.reward_range[0] <= reward <= self.reward_range[1]

    def validate_rollout(self, rollout: Dict) -> Optional[str]:
        if not self.validate_reward(rollout['reward']):
            return f"Reward {rollout['reward']} out of range {self.reward_range}"
        return None