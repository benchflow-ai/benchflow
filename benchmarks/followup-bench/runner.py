"""Followup-bench runner — Scene-based two-agent code review benchmark.

Uses the Scene runtime to orchestrate coder + reviewer in a SHARED sandbox:
  1. Coder agent attempts the task (in shared /app/)
  2. Reviewer agent reads the coder's work, writes feedback to /app/.outbox/coder.json
  3. Coder agent reads feedback, revises its solution
  4. Verifier scores the final /app/ state

Both agents run inside the SAME sandbox container — they share the
filesystem. The Scene scheduler manages turn-taking and outbox routing.

Usage:
    python -m benchmarks.followup_bench.runner \
        --task-dir .ref/terminal-bench-2/tasks/some-task \
        --coder gemini --coder-model gemini-3.1-flash-lite-preview \
        --reviewer gemini --reviewer-model gemini-3-pro-preview \
        --env daytona
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from benchflow._acp_run import connect_acp, execute_prompts
from benchflow._agent_setup import install_agent
from benchflow._scene import Role, Scene
from benchflow.agents.registry import AGENTS, AGENT_LAUNCH
from benchflow.runtime import Environment

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

If you receive feedback from a reviewer, read it carefully and revise your solution.
Only fix what the reviewer flagged — don't start over."""

REVIEWER_INSTRUCTION = """You are a code reviewer. The coder agent just attempted a task.

1. Read the task at /app/instruction.md
2. Read the coder's work in /app/ (modified/created files)
3. Write a specific, actionable review

Send your review by creating /app/.outbox/coder.json with:
  {"to": "coder", "content": "Your specific feedback here — reference files, lines, issues."}

If correct, write: {"to": "coder", "content": "LGTM — no changes needed."}"""


async def _role_runner(env, role: Role, prompt: str) -> None:
    """Run one agent turn inside the shared sandbox.

    Installs the agent (if not already), connects via ACP, sends prompt,
    waits for completion. All within the existing env — no new sandbox.
    """
    agent_name = role.agent
    agent_cfg = AGENTS.get(agent_name)
    if not agent_cfg:
        raise ValueError(f"Unknown agent: {agent_name}")

    trial_dir = Path(tempfile.mkdtemp(prefix=f"followup-{role.name}-"))

    await install_agent(env, agent_name, trial_dir)

    agent_env = {}
    agent_launch = AGENT_LAUNCH.get(agent_name, agent_name)

    client, session, _ = await connect_acp(
        env=env,
        agent=agent_name,
        agent_launch=agent_launch,
        agent_env=agent_env,
        sandbox_user="agent",
        model=role.model,
        trial_dir=trial_dir,
        environment="daytona",
        agent_cwd="/app",
    )

    try:
        await execute_prompts(
            acp_client=client,
            session=session,
            prompts=[prompt],
            timeout=600,
        )
    finally:
        await client.close()


async def run_followup_task(
    task_path: Path,
    coder_agent: str = "gemini",
    coder_model: str = "gemini-3.1-flash-lite-preview",
    reviewer_agent: str = "gemini",
    reviewer_model: str = "gemini-3-pro-preview",
    environment: str = "daytona",
) -> FollowupResult:
    """Run one task with the coder→reviewer→coder Scene flow in a shared sandbox."""
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

    env = Environment.from_task(task_path, backend=environment)

    async with env:
        trajectory = await scene.run(env._inner, _role_runner)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run followup-bench on a single task")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--coder", default="gemini")
    parser.add_argument("--coder-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--reviewer", default="gemini")
    parser.add_argument("--reviewer-model", default="gemini-3-pro-preview")
    parser.add_argument("--env", default="daytona")
    args = parser.parse_args()

    result = asyncio.run(run_followup_task(
        task_path=args.task_dir,
        coder_agent=args.coder,
        coder_model=args.coder_model,
        reviewer_agent=args.reviewer,
        reviewer_model=args.reviewer_model,
        environment=args.env,
    ))

    print(f"Task: {result.task_name}")
    print(f"Rounds: {result.n_rounds}")
    print(f"Messages: {len(result.messages)}")
    for m in result.messages:
        print(f"  [{m['turn']}] {m['sender']} → {m['recipient']}: {m['content'][:100]}")
