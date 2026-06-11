import json
from typing import Dict

def evaluate_task(task: Task, rollout: Dict) -> Dict:
    error = task.validate_rollout(rollout)
    if error:
        return {'error': error}
    return {'reward': rollout['reward'], 'metrics': rollout['metrics']}