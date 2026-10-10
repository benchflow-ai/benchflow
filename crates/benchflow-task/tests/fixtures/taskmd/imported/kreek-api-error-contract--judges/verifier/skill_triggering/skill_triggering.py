# Synced from eval/verifier/shared/skill_triggering/skill_triggering.py. Do not edit in tests/.
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
import consult_lib as cl  # noqa: E402
from rewardkit import criterion  # noqa: E402


@criterion
def skill_trigger_rate(workspace: Path) -> float:
    """Share of the task's intended skills the agent read. Zero-weight readout."""
    intended = list(cl.load_task_meta().get("intended_skills") or [])
    if not intended:
        return 1.0
    read = set(cl.read_skill_names(cl.load_trajectory()))
    return len([s for s in intended if s in read]) / len(intended)
