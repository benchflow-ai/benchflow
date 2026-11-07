#!/usr/bin/env python3
"""Run one task with a real agent, then walk its trajectory and token usage.

Credentials come from your environment, as for `bench eval run`: for Claude,
CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) or ANTHROPIC_API_KEY; for
Codex, a ChatGPT login or OPENAI_API_KEY. Run `bench doctor` to see which one
each agent would use.

Usage:
  uv run python docs/examples/python-sdk/run-agent.py
  uv run python docs/examples/python-sdk/run-agent.py --sandbox daytona \\
      --agent claude-agent-acp --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter
from pathlib import Path

import benchflow as bf

HELLO_WORLD = Path(__file__).resolve().parents[3] / "tests/examples/hello-world-task"


async def main(args: argparse.Namespace) -> int:
    result = await bf.run(
        bf.RolloutConfig(
            task_path=Path(args.task),
            agent=args.agent,
            model=args.model,
            environment=args.sandbox,
            jobs_dir=args.jobs_dir,
        )
    )
    print(result)
    print(
        f"reward {result.reward}  passed {result.passed}  tool calls {result.n_tool_calls}"
    )
    if result.error:
        # error_category is stable (e.g. provider_auth, timeout, sandbox_setup).
        print(f"agent error [{result.error_category}]: {result.error}")

    # Token usage is None when the provider reported none (usage_source "unavailable").
    print(
        f"tokens in {result.n_input_tokens} out {result.n_output_tokens} "
        f"total {result.total_tokens} (source {result.usage_source})"
    )

    # The trajectory is a list of ACP events (dicts) with a "type" key:
    # user_message, agent_message, agent_thought, tool_call, ...
    print("events:", dict(Counter(e.get("type") for e in result.trajectory)))
    for event in result.trajectory:
        if event.get("type") == "tool_call":
            print(f"  tool: {event.get('title')} [{event.get('status')}]")
        elif event.get("type") == "agent_message":
            print(f"  says: {str(event.get('text', ''))[:100]!r}")

    # The same rollout, read back from disk later (for example in a notebook).
    again = bf.RolloutResult.load(result.rollout_dir)
    assert again.reward == result.reward
    return 0 if result.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sandbox", default="docker", choices=["docker", "daytona"])
    parser.add_argument("--agent", default="claude-agent-acp")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--task", default=str(HELLO_WORLD))
    parser.add_argument("--jobs-dir", default="jobs/python-sdk-examples")
    # BenchFlow logs progress through the standard logging module.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
