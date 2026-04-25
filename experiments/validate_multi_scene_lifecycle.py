"""Dogfood: test Scene-based Trial lifecycle against real TB2 tasks.

Validates that the current Trial class can express:
1. Single-agent (baseline) — one connect/execute cycle
2. Two-stage BYOS — skill-gen scene then solve scene
3. Three-stage followup — coder/reviewer/revision

Uses the existing Trial.connect/execute/disconnect phases.
This script proves the lifecycle works before we add YAML parsing.

Usage:
    source .env && export GOOGLE_API_KEY="$GEMINI_API_KEY"
    uv run python experiments/dogfood_scene.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

from benchflow.trial import Trial, TrialConfig

TASK = Path(__file__).resolve().parents[0].parent / ".ref" / "terminal-bench-2" / "regex-log"
AGENT = os.environ.get("ABLATION_AGENT", "gemini")
MODEL = os.environ.get("ABLATION_MODEL", "gemini-3.1-flash-lite-preview")


async def test_single_agent():
    """Scene 1: single agent, one turn — baseline."""
    logger.info("=== TEST 1: Single-agent baseline ===")
    trial = await Trial.create(TrialConfig(
        task_path=TASK, agent=AGENT, model=MODEL, environment="daytona",
    ))
    result = await trial.run()
    logger.info(f"Result: reward={result.rewards}, tools={result.n_tool_calls}, err={result.error}")
    return result


async def test_two_stage_byos():
    """Scene 2: skill-gen (unscored) → solve (scored). Same agent, same sandbox."""
    logger.info("=== TEST 2: Two-stage BYOS (skill-gen → solve) ===")

    skill_gen_prompt = (
        "Analyze the task in /app/instruction.md. Think about what domain knowledge "
        "and procedures would help solve it. Write a concise skill document to "
        "/app/generated-skill.md that captures the key steps and patterns needed."
    )

    trial = await Trial.create(TrialConfig(
        task_path=TASK, agent=AGENT, model=MODEL, environment="daytona",
    ))

    await trial.setup()
    await trial.start()
    await trial.install_agent()

    # Scene 1: skill-gen (unscored)
    logger.info("--- Scene 1: skill-gen ---")
    await trial.connect()
    await trial.execute(prompts=[skill_gen_prompt])
    await trial.disconnect()

    # Scene 2: solve (scored) — agent sees generated-skill.md in filesystem
    logger.info("--- Scene 2: solve ---")
    await trial.connect()
    await trial.execute()  # uses default instruction.md
    await trial.disconnect()

    rewards = await trial.verify()
    await trial.cleanup()

    result = trial._build_result()
    logger.info(f"Result: reward={rewards}, tools={result.n_tool_calls}, err={result.error}")
    return result


async def test_followup_bench():
    """Scene 3: coder → reviewer → revision. Three turns, same sandbox."""
    logger.info("=== TEST 3: Followup-bench (coder → reviewer → revision) ===")

    instruction = (TASK / "instruction.md").read_text()

    coder_prompt = f"""{instruction}

When you are done, create /app/.outbox/reviewer.json with:
{{"to": "reviewer", "content": "Task complete, please review"}}"""

    reviewer_prompt = """You are an expert code reviewer. Read the coder's work in /app/.
IMPORTANT: Do NOT modify any files except /app/.outbox/coder.json.
Review for correctness, completeness, and bugs.
Write your review to /app/.outbox/coder.json:
  {"to": "coder", "content": "YOUR FEEDBACK"}"""

    trial = await Trial.create(TrialConfig(
        task_path=TASK, agent=AGENT, model=MODEL, environment="daytona",
    ))

    await trial.setup()
    await trial.start()
    await trial.install_agent()

    # Scene: coder → reviewer → revision (fixed turn order)
    logger.info("--- Turn 1: coder ---")
    await trial.connect()
    await trial.execute(prompts=[coder_prompt])
    await trial.disconnect()

    # Read coder's outbox, clear it
    await trial.env.exec("mkdir -p /app/.outbox && chmod 777 /app/.outbox")

    logger.info("--- Turn 2: reviewer ---")
    await trial.connect()
    await trial.execute(prompts=[reviewer_prompt])
    await trial.disconnect()

    # Read reviewer feedback
    feedback_result = await trial.env.exec("cat /app/.outbox/coder.json 2>/dev/null || echo '{}'")
    import json
    try:
        feedback = json.loads(feedback_result.stdout or "{}").get("content", "No feedback")
    except json.JSONDecodeError:
        feedback = "No structured feedback"
    await trial.env.exec("rm -rf /app/.outbox/*")

    revision_prompt = f"""{instruction}

You previously attempted this task and received the following review:

REVIEWER FEEDBACK:
{feedback}

Please address the reviewer's feedback and fix any issues."""

    logger.info("--- Turn 3: revision ---")
    await trial.connect()
    await trial.execute(prompts=[revision_prompt])
    await trial.disconnect()

    rewards = await trial.verify()
    await trial.cleanup()

    result = trial._build_result()
    logger.info(f"Result: reward={rewards}, tools={result.n_tool_calls}, err={result.error}")
    return result


async def main():
    results = {}

    r1 = await test_single_agent()
    results["baseline"] = r1.rewards

    r2 = await test_two_stage_byos()
    results["byos"] = r2.rewards

    r3 = await test_followup_bench()
    results["followup"] = r3.rewards

    logger.info("\n=== DOGFOOD RESULTS ===")
    for name, rewards in results.items():
        reward = (rewards or {}).get("reward", "N/A")
        logger.info(f"  {name}: reward={reward}")


if __name__ == "__main__":
    asyncio.run(main())
