#!/usr/bin/env python3
"""Coder-Reviewer demo — a simple two-agent Scene run via bf.run().

Demonstrates:
  - Multi-role Scene (coder + reviewer) in a shared sandbox
  - File-based handoff through the shared workspace: the reviewer writes
    feedback to a declared file and the coder's next prompt names it.
    BenchFlow does not inject messages between turns.
  - Standard bf.run(RolloutConfig) API — same path for single or multi-agent

Requirements:
  - uv tool install --prerelease allow 'benchflow==0.6.0',
    or run from a checkout with uv run
  - GEMINI_API_KEY or DAYTONA_API_KEY set
  - A BenchFlow task directory (e.g. benchflow-ai/skillsbench/tasks/edit-pdf)

Usage:
  uv run python docs/examples/coder-reviewer-demo.py --task benchflow-ai/skillsbench/tasks/edit-pdf
  uv run python docs/examples/coder-reviewer-demo.py --task benchflow-ai/skillsbench/tasks/edit-pdf --sandbox docker

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
from benchflow.rollout import Role, RolloutConfig, Scene, Turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("coder-reviewer-demo")


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------


def baseline_config(task_path: Path, env: str, agent: str, model: str) -> RolloutConfig:
    """Pattern 1: Single agent, single turn — the baseline."""
    return RolloutConfig(
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
) -> RolloutConfig:
    """Pattern 3: Coder + Reviewer — explicit multi-role prompts.

    Flow:
      1. Coder attempts the task
      2. Reviewer reads coder's work, writes feedback to /app/review-feedback.md
      3. Coder reads the declared feedback file and revises the solution
    """
    return RolloutConfig(
        task_path=task_path,
        scenes=[
            Scene(
                name="code-review",
                roles=[
                    Role("coder", coder_agent, coder_model),
                    Role("reviewer", reviewer_agent, reviewer_model),
                ],
                turns=[
                    Turn("coder"),
                    Turn(
                        "reviewer",
                        "Review the code in /app/. Check for correctness, edge cases, "
                        "and adherence to the task requirements in /app/instruction.md. "
                        "Write your feedback to /app/review-feedback.md.",
                    ),
                    Turn(
                        "coder",
                        "Read /app/review-feedback.md and fix the issues. "
                        "Focus only on what was flagged — don't start over.",
                    ),
                ],
            )
        ],
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
    logger.info(
        f"Baseline: reward={baseline_reward}, tools={baseline_result.n_tool_calls}"
    )

    logger.info("=== Coder + Reviewer (multi-round) ===")
    review = coder_reviewer_config(task_path, env, agent, model, agent, model)
    review_result = await bf.run(review)
    review_reward = (review_result.rewards or {}).get("reward")
    logger.info(f"Reviewed: reward={review_reward}, tools={review_result.n_tool_calls}")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(
        f"  Baseline:  reward={baseline_reward}  tool_calls={baseline_result.n_tool_calls}"
    )
    print(
        f"  Reviewed:  reward={review_reward}  tool_calls={review_result.n_tool_calls}"
    )
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
    parser.add_argument(
        "--task", required=True, help="Task ref (org/repo/path or local path)"
    )
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "docker"])
    parser.add_argument("--agent", default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline and coder-reviewer side by side",
    )
    args = parser.parse_args()

    from benchflow._utils.benchmark_repos import resolve_source

    if "/" in args.task and not Path(args.task).exists():
        # Treat as org/repo or org/repo/path ref.
        parts = args.task.split("/", 2)
        repo = f"{parts[0]}/{parts[1]}"
        path = parts[2] if len(parts) > 2 else None
        args.task = resolve_source(repo, path=path)
    else:
        args.task = Path(args.task)
        if not args.task.exists():
            print(f"Task directory not found: {args.task}", file=sys.stderr)
            sys.exit(1)

    if args.compare:
        asyncio.run(run_comparison(args.task, args.sandbox, args.agent, args.model))
    else:
        asyncio.run(run_single(args.task, args.sandbox, args.agent, args.model))


if __name__ == "__main__":
    main()
