"""Multi-turn Scene demo — interactive agent benchmarking with benchflow.

This script demonstrates benchflow's Scene-based Trial lifecycle, showing
how to run multi-turn, multi-agent evaluations that Harbor #1316 proposed
and PR #1462 built 600+ lines to implement.

With benchflow, the same patterns are YAML configs — no new runtime code.

Run:
    pip install benchflow==0.3.0a8
    export DAYTONA_API_KEY="dtn_..."
    export GEMINI_API_KEY="AIza..."
    python docs/notebooks/multi-turn-scene-demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for dev installs
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import benchflow as bf
from benchflow.trial import Trial, TrialConfig, Scene, Role, Turn


# ── Demo 1: Single-agent baseline ─────────────────────────────────────
# The simplest case — one agent, one task, one turn.
# Harbor equivalent: `harbor run`

async def demo_single_agent():
    print("=" * 60)
    print("Demo 1: Single-agent baseline")
    print("=" * 60)

    result = await bf.run(
        "gemini",
        task_path=".ref/terminal-bench-2/regex-log",
        model="gemini-3.1-flash-lite-preview",
    )

    reward = (result.rewards or {}).get("reward", 0.0)
    print(f"  Task:       regex-log")
    print(f"  Reward:     {reward}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Error:      {result.error or 'none'}")
    return reward


# ── Demo 2: Multi-turn self-review ─────────────────────────────────────
# Same agent gets two prompts: solve, then review your own work.
# Harbor equivalent: prompts list in YAML

async def demo_multi_turn():
    print("\n" + "=" * 60)
    print("Demo 2: Multi-turn self-review")
    print("=" * 60)

    config = TrialConfig(
        task_path=Path(".ref/terminal-bench-2/regex-log"),
        scenes=[Scene(
            name="self-review",
            roles=[Role("agent", "gemini", "gemini-3.1-flash-lite-preview")],
            turns=[
                Turn("agent"),  # solve (uses instruction.md)
                Turn("agent", "Review your solution. Check edge cases, test it mentally with concrete inputs, and fix any issues you find."),
            ],
        )],
        environment="daytona",
    )

    trial = await Trial.create(config)
    result = await trial.run()

    reward = (result.rewards or {}).get("reward", 0.0)
    print(f"  Task:       regex-log")
    print(f"  Reward:     {reward}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Error:      {result.error or 'none'}")
    return reward


# ── Demo 3: Coder + Reviewer (two-role Scene) ────────────────────────
# Two agents in one scene — coder solves, reviewer critiques, coder fixes.
# Harbor equivalent: PR #1462 (600+ lines of User orchestration)
# Benchflow: 15 lines of config.

async def demo_coder_reviewer():
    print("\n" + "=" * 60)
    print("Demo 3: Coder + Reviewer (independent code review)")
    print("=" * 60)

    config = TrialConfig(
        task_path=Path(".ref/terminal-bench-2/regex-log"),
        scenes=[Scene(
            name="code-review",
            roles=[
                Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
                Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
            ],
            turns=[
                Turn("coder"),
                Turn("reviewer",
                     "You are a code reviewer. Read the coder's work in /app/. "
                     "Check for correctness and edge cases. "
                     "Write specific feedback to /app/.outbox/coder.json: "
                     '{"to": "coder", "content": "YOUR FEEDBACK"}'),
                Turn("coder",
                     "Read the reviewer's feedback at /app/.outbox/coder.json "
                     "and fix the issues they found. Focus on the specific "
                     "problems mentioned."),
            ],
        )],
        environment="daytona",
    )

    trial = await Trial.create(config)
    result = await trial.run()

    reward = (result.rewards or {}).get("reward", 0.0)
    print(f"  Task:       regex-log")
    print(f"  Reward:     {reward}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Error:      {result.error or 'none'}")
    return reward


# ── Demo 4: Interactive user simulation (harbor #1316) ────────────────
# A "user" role reveals task info gradually. Agent asks questions.
# This is exactly what harbor #1316 proposed.

async def demo_interactive_user():
    print("\n" + "=" * 60)
    print("Demo 4: Interactive user simulation (harbor #1316)")
    print("=" * 60)

    config = TrialConfig(
        task_path=Path(".ref/terminal-bench-2/regex-log"),
        scenes=[Scene(
            name="interactive",
            roles=[
                Role("user", "gemini", "gemini-3.1-flash-lite-preview"),
                Role("agent", "gemini", "gemini-3.1-flash-lite-preview"),
            ],
            turns=[
                # User gives vague instruction first
                Turn("user",
                     "You are simulating a user who wants help writing a regex. "
                     "Read /app/instruction.md for the full spec, but only tell "
                     "the agent: 'I need a regex to find dates in log files that "
                     "have IP addresses.' Save your message to /app/.outbox/agent.json"),
                Turn("agent",
                     "Read the user's request at /app/.outbox/agent.json. "
                     "Ask clarifying questions or start solving. "
                     "Save your regex to /app/regex.txt"),
            ],
        )],
        environment="daytona",
    )

    trial = await Trial.create(config)
    result = await trial.run()

    reward = (result.rewards or {}).get("reward", 0.0)
    print(f"  Task:       regex-log (via simulated user)")
    print(f"  Reward:     {reward}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Error:      {result.error or 'none'}")
    return reward


# ── Demo 5: Two-scene BYOS (skill generation → solve) ────────────────
# First scene: agent generates a skill. Second scene: agent uses it.

async def demo_byos():
    print("\n" + "=" * 60)
    print("Demo 5: BYOS (skill generation → solve)")
    print("=" * 60)

    config = TrialConfig(
        task_path=Path(".ref/terminal-bench-2/regex-log"),
        scenes=[
            Scene(
                name="skill-gen",
                roles=[Role("gen", "gemini", "gemini-3.1-flash-lite-preview")],
                turns=[Turn("gen",
                    "Read /app/instruction.md. Write a concise skill document "
                    "to /app/generated-skill.md capturing the key domain "
                    "knowledge and step-by-step procedure needed.")],
            ),
            Scene(
                name="solve",
                roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
                turns=[Turn("solver")],
            ),
        ],
        environment="daytona",
    )

    trial = await Trial.create(config)
    result = await trial.run()

    reward = (result.rewards or {}).get("reward", 0.0)
    print(f"  Task:       regex-log (with skill generation)")
    print(f"  Reward:     {reward}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Error:      {result.error or 'none'}")
    return reward


# ── Run all demos ─────────────────────────────────────────────────────

async def main():
    print("BenchFlow Multi-Turn Scene Demo")
    print(f"Version: {bf.__version__}")
    print()

    results = {}
    results["single"] = await demo_single_agent()
    results["multi-turn"] = await demo_multi_turn()
    results["coder-reviewer"] = await demo_coder_reviewer()
    results["interactive-user"] = await demo_interactive_user()
    results["byos"] = await demo_byos()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, reward in results.items():
        status = "PASS" if reward == 1.0 else "FAIL"
        print(f"  {name:<20} reward={reward}  [{status}]")

    print(f"\nAll patterns demonstrated with the same task (regex-log).")
    print(f"Harbor #1316 needs 600+ lines of runtime code.")
    print(f"BenchFlow needs 15 lines of YAML config per pattern.")


if __name__ == "__main__":
    asyncio.run(main())
