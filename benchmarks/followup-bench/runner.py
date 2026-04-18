"""Followup-bench runner — Scene-based two-agent code review benchmark.

Uses the Scene runtime to orchestrate:
  1. Coder agent attempts a task
  2. Reviewer agent reads the coder's work, writes targeted feedback
  3. Coder agent reads feedback and revises
  4. Verifier scores the final result

Measures: does an independent review step improve the score?

Usage:
    python -m benchmarks.followup_bench.runner \
        --tasks-dir .ref/terminal-bench-2/tasks \
        --coder gemini --coder-model gemini-3.1-flash-lite-preview \
        --reviewer gemini --reviewer-model gemini-3.1-flash-lite-preview \
        --env daytona --concurrency 4
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from benchflow._scene import Role, Scene
from benchflow.runtime import Agent, Environment, RuntimeConfig

logger = logging.getLogger(__name__)


@dataclass
class FollowupResult:
    task_name: str
    single_turn_reward: float | None
    followup_reward: float | None
    lift: float
    n_rounds: int
    messages: list[dict]


CODER_INSTRUCTION = """You are a coding agent. Solve the task described in /app/instruction.md.
Work in the /app/ directory. Write your solution there.

If you receive feedback from a reviewer, read it carefully and revise your solution accordingly.
Only fix what the reviewer flagged — don't start over from scratch."""

REVIEWER_INSTRUCTION = """You are a code reviewer. The coder agent just attempted a task.

1. Read the task instruction at /app/instruction.md
2. Read the coder's work in /app/ (look at modified/created files)
3. Write a specific, actionable review

Write your review to the coder by creating /app/.outbox/coder.json:
```json
{"to": "coder", "content": "Your specific review feedback here"}
```

Be specific — reference file names, line numbers, and concrete issues.
If the code is correct, write: {"to": "coder", "content": "LGTM — no changes needed."}"""


async def run_followup_task(
    task_path: Path,
    coder_agent: str,
    coder_model: str,
    reviewer_agent: str,
    reviewer_model: str,
    environment: str = "daytona",
    jobs_dir: str = "jobs/followup-bench",
) -> FollowupResult:
    """Run one task with the coder→reviewer→coder Scene flow."""
    from benchflow.sdk import SDK

    scene = Scene(
        roles={
            "coder": Role(
                name="coder",
                agent=coder_agent,
                model=coder_model,
                instruction=CODER_INSTRUCTION,
            ),
            "reviewer": Role(
                name="reviewer",
                agent=reviewer_agent,
                model=reviewer_model,
                instruction=REVIEWER_INSTRUCTION,
            ),
        },
        max_rounds=4,
    )

    async def role_runner(env, role, prompt):
        sdk = SDK()
        await sdk.run(
            task_path=task_path,
            agent=role.agent,
            model=role.model,
            prompts=[prompt],
            environment=environment,
            jobs_dir=f"{jobs_dir}/{role.name}",
        )

    env = Environment.from_task(task_path, backend=environment)
    async with env:
        trajectory = await scene.run(env, role_runner)

    messages = [
        {"sender": m.sender, "recipient": m.recipient, "content": m.content, "turn": m.turn}
        for m in trajectory
    ]

    return FollowupResult(
        task_name=task_path.name,
        single_turn_reward=None,
        followup_reward=None,
        lift=0.0,
        n_rounds=scene._round,
        messages=messages,
    )
