# Synced from eval/verifier/shared/change_quality/change_quality.py. Do not edit in tests/.
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
import consult_lib as cl  # noqa: E402
from rewardkit import criterion  # noqa: E402


@criterion
def change_quality_level(workspace: Path) -> float:
    """1.0 for source and tests, 0.7 for source only, 0.25 otherwise."""
    kinds = {cl.classify(p) for p in cl.changed_paths(workspace)}
    if "source" in kinds and "test" in kinds:
        return 1.0
    if "source" in kinds:
        return 0.7
    return 0.25
