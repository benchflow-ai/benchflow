import json
from typing import Dict

class Task:
    def __init__(self, task_config: Dict):
        self.task_config = task_config
        self.verifier = Verifier(task_config)

    def validate_rollout(self, rollout: Dict) -> Optional[str]:
        return self.verifier.validate_rollout(rollout)