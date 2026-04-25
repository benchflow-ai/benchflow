"""Dogfood: SWE-bench Pro progressive disclosure with BaseUser.

Demonstrates the BaseUser abstraction on a SWE-bench Pro task — the original
motivation for this feature (Josh's GitHub/Microsoft use case).

The user:
  Round 0: terse problem description (one sentence from the spec).
  Round 1+: if tests failed, append the failing test names and a section of
            the original spec as a hint.
  Stops when reward >= 1.0 or max_user_rounds hit.

Usage:
    GEMINI_API_KEY=... python examples/swebench_pro_user_dogfood.py
    GEMINI_API_KEY=... python examples/swebench_pro_user_dogfood.py --task openlibrary

Tasks available (oracle-validated 5/5 on 2026-04-24):
    ansible, flipt, openlibrary, navidrome, qutebrowser
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import benchflow as bf
from benchflow.trial import Scene, TrialConfig
from benchflow.user import FunctionUser, RoundResult

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")


SWEBENCH_PRO_TASKS = {
    "ansible": "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2",
    "flipt": "instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1",
    "openlibrary": "instance_internetarchive__openlibrary-00bec1e7c8f3272c469a58e1377df03f955ed478-v13642507b4fc1f8d234172bf8129942da2c2ca26",
    "navidrome": "instance_navidrome__navidrome-0130c6dc13438b48cf0fdfab08a89e357b5517c9",
    "qutebrowser": "instance_qutebrowser__qutebrowser-01d1d1494411380d97cac14614a829d3a69cecaf-v2ef375ac784985212b1805e1d0431dc8f1b3c171",
}


def make_progressive_user() -> FunctionUser:
    """User that compresses the instruction on round 0 and discloses hints on failure."""

    def progressive(
        round: int, instruction: str, rr: RoundResult | None
    ) -> str | None:
        # Round 0: terse — first line of the spec.
        if round == 0:
            first_line = instruction.split("\n", 1)[0].strip()
            return (
                f"{first_line}\n\n"
                "Read /app/ to understand the codebase, then implement the fix. "
                "Run tests when you think you're done."
            )

        # Stop on success.
        if rr and rr.rewards:
            score = rr.rewards.get("reward", rr.rewards.get("exact_match", 0))
            if score >= 1.0:
                print(f"  [User] reward={score} at round {round}, stopping.")
                return None

        # Round 1: show failing tests + first half of spec.
        if round == 1:
            half = len(instruction) // 2
            return (
                "The verifier reported these issues:\n\n"
                f"{(rr.verifier_output or '<no output>')[:1500]}\n\n"
                "Here is the first half of the spec for context:\n\n"
                f"{instruction[:half]}\n\n"
                "Continue working in /app/."
            )

        # Round 2: full spec.
        if round == 2:
            return (
                "Still failing. Full specification:\n\n"
                f"{instruction}\n\n"
                "Address every requirement listed above."
            )

        return None

    return FunctionUser(progressive)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="flipt",
        choices=list(SWEBENCH_PRO_TASKS),
        help="Which SWE-bench Pro task to run",
    )
    parser.add_argument("--agent", default="gemini")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--backend", default="daytona")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    task_dir = SWEBENCH_PRO_TASKS[args.task]
    task_path = Path(".ref/swebenchpro") / task_dir
    if not task_path.exists():
        print(f"Task not found: {task_path}")
        return

    config = TrialConfig(
        task_path=task_path,
        scenes=[Scene.single(agent=args.agent, model=args.model)],
        environment=args.backend,
        sandbox_user="agent",
        user=make_progressive_user(),
        max_user_rounds=args.max_rounds,
        jobs_dir="/tmp/swebench-pro-jobs/progressive",
    )

    print(f"Progressive disclosure on {args.task}")
    print(f"  Agent:  {args.agent} / {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Rounds:  up to {args.max_rounds}")
    print()

    result = await bf.run(config)

    print("\n=== RESULT ===")
    print(f"  Task: {result.task_name}")
    print(f"  Rewards: {result.rewards}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Error: {result.error}")
    if result.trial_dir:
        print(f"  Trial dir: {result.trial_dir}")


if __name__ == "__main__":
    asyncio.run(main())
