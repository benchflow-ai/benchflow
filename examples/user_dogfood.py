"""Dogfood: run regex-log task with progressive-disclosure User.

Demonstrates the BaseUser abstraction — a FunctionUser that:
  Round 0: gives a terse version of the instruction
  Round 1+: if tests failed, gives a hint based on the full instruction
  Stops when tests pass or max_rounds hit.

Usage:
    GEMINI_API_KEY=... python examples/user_dogfood.py
"""

import asyncio
import logging
from pathlib import Path

import benchflow as bf
from benchflow.trial import Scene, TrialConfig
from benchflow.user import FunctionUser, RoundResult

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")


def progressive_user(
    round: int, instruction: str, rr: RoundResult | None
) -> str | None:
    if round == 0:
        return (
            "Write a regex that matches dates (YYYY-MM-DD) in log lines "
            "containing an IPv4 address. Save it to /app/regex.txt"
        )

    if rr and rr.rewards:
        score = rr.rewards.get("exact_match", rr.rewards.get("reward", 0))
        if score >= 1.0:
            print(f"  [User] Tests passed at round {round}! Stopping.")
            return None

    if round == 1:
        return (
            "The tests failed. Important details you may have missed:\n"
            "- Match only the LAST date in each line\n"
            "- Feb can have up to 29 days\n"
            "- Dates/IPs must not be preceded/followed by alphanumeric chars\n"
            "- Use re.findall with re.MULTILINE\n"
            "Fix /app/regex.txt"
        )

    if round == 2:
        return (
            "Still failing. Here's the full instruction:\n\n" + instruction
        )

    return None


async def main():
    task_path = Path(".ref/terminal-bench-2/regex-log")
    if not task_path.exists():
        print(f"Task not found at {task_path}")
        return

    config = TrialConfig(
        task_path=task_path,
        scenes=[Scene.single(agent="gemini", model="gemini-2.5-flash")],
        environment="daytona",
        user=FunctionUser(progressive_user),
        max_user_rounds=4,
    )

    print("Running progressive-disclosure trial on regex-log...")
    print(f"  Agent: gemini/flash")
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
