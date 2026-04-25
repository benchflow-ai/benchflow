#!/usr/bin/env python3
"""Coder-Reviewer demo — a simple two-agent Scene run via bf.run().

Demonstrates:
  - Multi-role Scene (coder + reviewer) in a shared sandbox
  - Outbox-based message passing between roles
  - Standard bf.run(TrialConfig) API — same path for single or multi-agent

Requirements:
  - pip install benchflow
  - GEMINI_API_KEY or DAYTONA_API_KEY set
  - A Harbor-format task directory (e.g. .ref/terminal-bench-2/regex-log)

Usage:
  python docs/notebooks/coder-reviewer-demo.py --task .ref/terminal-bench-2/regex-log
  python docs/notebooks/coder-reviewer-demo.py --task .ref/terminal-bench-2/regex-log --env docker

Terminology:
  - Turn:        One prompt → one ACP session (one role acts)
  - Multi-turn:  Same role, multiple turns (e.g. self-review: agent → agent)
  - Round:       One A→B exchange between different roles
  - Multi-round: Different roles exchanging turns (e.g. coder → reviewer → coder)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import benchflow as bf
from benchflow.trial import Role, Scene, TrialConfig, Turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("coder-reviewer-demo")


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

def baseline_config(task_path: Path, env: str, agent: str, model: str) -> TrialConfig:
    """Pattern 1: Single agent, single turn — the baseline."""
    return TrialConfig(
        task_path=task_path,
        scenes=[Scene.single(agent=agent, model=model)],
        environment=env,
    )


def coder_reviewer_config(
    task_path: Path,
    env: str,
    coder_agent: str = "gemini",
    coder_model: str = "gemini-3.1-flash-lite-preview",
    reviewer_agent: str = "gemini",
    reviewer_model: str = "gemini-3.1-flash-lite-preview",
) -> TrialConfig:
    """Pattern 3: Coder + Reviewer — multi-round with outbox messaging.

    Flow:
      1. Coder attempts the task
      2. Reviewer reads coder's work, writes feedback to /app/.outbox/coder.json
      3. Coder receives feedback (injected by scheduler), revises solution

    The outbox convention:
      - Agent writes: /app/.outbox/{recipient}.json
      - Format: {"to": "role_name", "content": "your message"}
      - Scheduler reads, clears, and injects into next role's prompt
    """
    return TrialConfig(
        task_path=task_path,
        scenes=[Scene(
            name="code-review",
            roles=[
                Role("coder", coder_agent, coder_model),
                Role("reviewer", reviewer_agent, reviewer_model),
            ],
            turns=[
                Turn("coder"),
                Turn("reviewer",
                     "Review the code in /app/. Check for correctness, edge cases, "
                     "and adherence to the task requirements in /app/instruction.md. "
                     "Write your feedback to /app/.outbox/coder.json as: "
                     '{"to": "coder", "content": "Your specific feedback here."}'),
                Turn("coder",
                     "Read the reviewer's feedback and fix the issues. "
                     "Focus only on what was flagged — don't start over."),
            ],
        )],
        environment=env,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_comparison(
    task_path: Path,
    env: str,
    agent: str,
    model: str,
) -> None:
    """Run baseline vs coder-reviewer and compare."""
    logger.info("=== Baseline (single agent, single turn) ===")
    baseline = baseline_config(task_path, env, agent, model)
    baseline_result = await bf.run(baseline)
    baseline_reward = (baseline_result.rewards or {}).get("reward")
    logger.info(f"Baseline: reward={baseline_reward}, tools={baseline_result.n_tool_calls}")

    logger.info("=== Coder + Reviewer (multi-round) ===")
    review = coder_reviewer_config(task_path, env, agent, model, agent, model)
    review_result = await bf.run(review)
    review_reward = (review_result.rewards or {}).get("reward")
    logger.info(f"Reviewed: reward={review_reward}, tools={review_result.n_tool_calls}")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Baseline:  reward={baseline_reward}  tool_calls={baseline_result.n_tool_calls}")
    print(f"  Reviewed:  reward={review_reward}  tool_calls={review_result.n_tool_calls}")
    if baseline_reward is not None and review_reward is not None:
        lift = review_reward - baseline_reward
        print(f"  Lift:      {lift:+.2f}")
    print("=" * 60)


async def run_single(task_path: Path, env: str, agent: str, model: str) -> None:
    """Run coder-reviewer only."""
    config = coder_reviewer_config(task_path, env, agent, model, agent, model)
    result = await bf.run(config)
    reward = (result.rewards or {}).get("reward")
    print(f"reward={reward}  tool_calls={result.n_tool_calls}  error={result.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coder-Reviewer demo")
    parser.add_argument("--task", type=Path, required=True, help="Path to task directory")
    parser.add_argument("--env", default="daytona", choices=["daytona", "docker"])
    parser.add_argument("--agent", default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--compare", action="store_true",
                        help="Run baseline and coder-reviewer side by side")
    args = parser.parse_args()

    if not args.task.exists():
        print(f"Task directory not found: {args.task}", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        asyncio.run(run_comparison(args.task, args.env, args.agent, args.model))
    else:
        asyncio.run(run_single(args.task, args.env, args.agent, args.model))


if __name__ == "__main__":
    main()
