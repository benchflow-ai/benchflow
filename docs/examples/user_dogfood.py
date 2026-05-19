"""Dogfood: run edit-pdf task with progressive-disclosure User.

Demonstrates the BaseUser abstraction — a FunctionUser that:
  Round 0: gives a terse version of the instruction
  Round 1+: if tests failed, gives a hint based on the full instruction
  Stops when tests pass or max_rounds hit.

Usage:
    GEMINI_API_KEY=... python docs/examples/user_dogfood.py
"""

import asyncio
import logging

from benchflow.user import FunctionUser, RoundResult

import benchflow as bf
from benchflow.rollout import RolloutConfig, Scene

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")


def progressive_user(
    round: int, instruction: str, rr: RoundResult | None
) -> str | None:
    # Round 0: give a terse version — first line of instruction.md.
    if round == 0:
        first_line = instruction.split("\n", 1)[0].strip()
        return (
            f"{first_line}\n\n"
            "Read the task files to understand what's needed, then implement "
            "the solution. Run tests when you think you're done."
        )

    # Stop on success.
    if rr and rr.rewards:
        score = rr.rewards.get("exact_match", rr.rewards.get("reward", 0))
        if score >= 1.0:
            print(f"  [User] Tests passed at round {round}! Stopping.")
            return None

    # Round 1: nudge with more detail from the full instruction.
    if round == 1:
        return (
            "The tests failed. Re-read the full instruction carefully — "
            "you may have missed important details:\n\n" + instruction
        )

    # Round 2+: give up.
    if round >= 2:
        return None

    return None


async def main():
    from benchflow._utils.benchmark_repos import resolve_source

    task_path = resolve_source("benchflow-ai/skillsbench", path="tasks/edit-pdf")

    config = RolloutConfig(
        task_path=task_path,
        scenes=[Scene.single(agent="gemini", model="gemini-2.5-flash")],
        environment="daytona",
        user=FunctionUser(progressive_user),
        max_user_rounds=4,
    )

    print("Running progressive-disclosure rollout on edit-pdf...")
    print("  Agent: gemini/flash")
    print(f"  Max rounds: {config.max_user_rounds}")
    print(f"  Environment: {config.environment}")
    print()

    result = await bf.run(config)

    print("\n=== RESULT ===")
    print(f"  Task: {result.task_name}")
    print(f"  Rewards: {result.rewards}")
    print(f"  Error: {result.error}")
    print(f"  Tool calls: {result.n_tool_calls}")

    if result.rewards:
        score = result.rewards.get("exact_match", result.rewards.get("reward", 0))
        if score >= 1.0:
            print("  PASSED (final verify)")
        else:
            print(f"  FAILED (score={score})")


if __name__ == "__main__":
    asyncio.run(main())
