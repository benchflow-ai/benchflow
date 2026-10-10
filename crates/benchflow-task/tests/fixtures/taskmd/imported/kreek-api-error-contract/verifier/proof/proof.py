# Synced from eval/verifier/shared/proof/proof.py. Do not edit in tests/.
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consult_lib as cl  # noqa: E402
import submitted  # noqa: E402  (per-task: tests/proof/submitted.py)
from rewardkit import criterion  # noqa: E402


@criterion
def proof_level(workspace: Path) -> float:
    """Proof ladder from the previous scorer: 100/85/60/35/15, as 1.0/0.85/0.6/0.35/0.15."""
    submitted_proof = bool(submitted.submitted_proof(workspace))
    post_write = cl.has_post_write_proof(cl.load_trajectory())
    if submitted_proof and post_write:
        return 1.0
    if submitted_proof:
        return 0.85
    if post_write:
        return 0.6
    meta = cl.load_task_meta()
    if cl.visible_tests_pass(workspace, meta) and cl.hidden_check_passes(workspace):
        return 0.35
    return 0.15
