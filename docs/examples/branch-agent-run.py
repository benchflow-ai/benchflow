#!/usr/bin/env python3
"""Branch a real agent run into two children, each scored by the task verifier.

Shape:

  1. setup -> start -> install_agent -> connect -> execute(FIRST_PROMPT).
     The agent writes draft.txt. This state is the checkpoint.
  2. rollout.branch(2, run_child, snapshot_layers={"sandbox"}, child_labels=...).
     The container filesystem is snapshotted. Each child starts from the
     snapshot with a FRESH agent session, gets its own self-contained prompt,
     and is scored by the task's own verifier. V(checkpoint) is their mean.
  3. The parent world is restored. branch() leaves the agent disconnected, so
     the parent reconnects, finishes the task, is verified and finalized
     (result.json with a `branches` block, tree.json, per-child observation.json).

Children do not remember the parent's conversation: only files (and declared
database state) are restored. "Continue" or "now finish" would reach an agent
that never saw the first prompt, so every child prompt must stand on its own.

Requirements:
  - a BenchFlow checkout: uv sync --extra dev (add --extra sandbox-daytona for Daytona)
  - Docker running, or DAYTONA_API_KEY set (Daytona direct mode snapshots)
  - agent credentials as for any eval, e.g. CLAUDE_CODE_OAUTH_TOKEN for claude-agent-acp

Usage:
  uv run python docs/examples/branch-agent-run.py --sandbox docker
  BENCHFLOW_DAYTONA_OWNER=$USER uv run python docs/examples/branch-agent-run.py --sandbox daytona
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from benchflow.rollout import BranchChild, Rollout, RolloutConfig

TASK = Path(__file__).resolve().parents[2] / "tests/examples/hello-world-task"

FIRST_PROMPT = (
    "This is step 1 of a two-step exercise. In the current working directory, "
    "create a file named draft.txt containing exactly one line: Hello world\n"
    "Do not create any other files, and do not create hello.txt."
)


def child_prompts(instruction: str) -> dict[str, str]:
    """One self-contained prompt per child label."""
    return {
        # The task instruction verbatim.
        "baseline": instruction,
        # A deliberately misleading hint, so the two arms can diverge.
        "hint-reuse-draft": (
            "The file draft.txt in the current working directory already holds "
            "the final text. Rename draft.txt to hello.txt without changing its "
            "contents, then stop."
        ),
    }


async def main(args: argparse.Namespace) -> None:
    instruction = (TASK / "instruction.md").read_text().strip()
    prompts = child_prompts(instruction)
    rollout = Rollout(
        RolloutConfig(
            task_path=TASK,
            agent=args.agent,
            model=args.model,
            environment=args.sandbox,
            jobs_dir=Path(args.jobs_dir),
            job_name="branch-agent-run",
        )
    )

    async def run_child(node, *, child: BranchChild) -> float | None:
        """Run one child: a fresh agent session from the checkpoint."""
        rewards = None
        try:
            await rollout.connect()
            await rollout.execute([prompts[child.label]], node=node)
            rewards = await rollout.verify()
        finally:
            await rollout.disconnect()
        # None (no canonical reward) marks the child unscored, never 0.
        return (rewards or {}).get("reward")

    try:
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()
        await rollout.connect()
        await rollout.execute([FIRST_PROMPT])

        value = await rollout.branch(
            2,
            run_child,
            snapshot_layers={"sandbox"},
            child_labels=list(prompts),
        )
        print(f"V(checkpoint) = {value}")

        await rollout.connect()
        await rollout.execute([instruction])
        await rollout.verify()
        result = await rollout.finalize()
    except BaseException:
        await rollout.cleanup()
        raise
    print(f"parent rewards = {result.rewards}; artifacts under {args.jobs_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sandbox", choices=["docker", "daytona"], default="docker")
    parser.add_argument("--agent", default="claude-agent-acp")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--jobs-dir", default="jobs")
    asyncio.run(main(parser.parse_args()))
