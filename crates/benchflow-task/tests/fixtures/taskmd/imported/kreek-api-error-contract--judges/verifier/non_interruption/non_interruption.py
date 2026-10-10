# Synced from eval/verifier/shared/non_interruption/non_interruption.py. Do not edit in tests/.
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
import consult_lib as cl  # noqa: E402
from rewardkit import criterion  # noqa: E402


@criterion
def non_interruption_score(workspace: Path) -> float:
    """1 minus 0.25 per agent message that ends by asking a question. Zero-weight readout."""
    questions = cl.question_message_count(cl.load_trajectory())
    return max(0.0, 1.0 - 0.25 * questions)
