"""Proof: two-role Scene desugars to live rollout Steps on Daytona.

Usage:
    uv run python tests/conformance/proof_multi_agent.py

Runs a coder -> reviewer -> coder rollout where all roles share one sandbox:
1. Coder writes ``/app/result.txt``.
2. Reviewer inspects that file and writes ``/app/review.txt``.
3. Coder reads the review and updates ``/app/result.txt``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchflow.rollout import Role, Rollout, RolloutConfig, Scene, Turn

TASK_PATH = Path(__file__).parent / "acp_smoke"

CODER_INSTRUCTION = """Create /app/result.txt with exactly:
hello from coder

Verify the file exists before stopping."""

REVIEWER_INSTRUCTION = """Read /app/result.txt and verify it contains
"hello from coder". Then create /app/review.txt with exactly:
review passed

Verify /app/review.txt exists before stopping."""

FINAL_CODER_INSTRUCTION = """Read /app/review.txt. If it says review passed,
append this exact line to /app/result.txt:
review acknowledged

Verify /app/result.txt contains both lines before stopping."""


async def main() -> None:
    rollout = await Rollout.create(
        RolloutConfig(
            task_path=TASK_PATH,
            environment="daytona",
            scenes=[
                Scene(
                    name="coder-reviewer-proof",
                    roles=[
                        Role(
                            "coder",
                            "claude-agent-acp",
                            "claude-haiku-4-5-20251001",
                        ),
                        Role(
                            "reviewer",
                            "claude-agent-acp",
                            "claude-haiku-4-5-20251001",
                        ),
                    ],
                    turns=[
                        Turn("coder", CODER_INSTRUCTION),
                        Turn("reviewer", REVIEWER_INSTRUCTION),
                        Turn("coder", FINAL_CODER_INSTRUCTION),
                    ],
                )
            ],
        )
    )

    result = await rollout.run()
    print("\n" + "=" * 60)
    print("MULTI-AGENT STEP PROOF RESULTS")
    print("=" * 60)
    print(f"rollout: {result.rollout_name}")
    print(f"reward: {result.rewards}")
    print(f"error: {result.error}")
    print(f"verifier_error: {result.verifier_error}")
    print(f"trajectory events: {len(result.trajectory)}")


if __name__ == "__main__":
    asyncio.run(main())
