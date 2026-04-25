"""Proof: 2-agent scene on a live Daytona sandbox.

Usage: env -u ANTHROPIC_API_KEY python proof_multi_agent.py

Runs a coder→reviewer scene where:
  1. Coder writes a file and sends a message to reviewer
  2. Reviewer inspects the file and responds
  3. Scene exits deterministically
  4. Trajectory JSONL contains ordered cross-role messages
"""

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import contextlib

from harbor.models.task.task import Task

from benchflow._acp_run import connect_acp, execute_prompts
from benchflow._agent_setup import install_agent
from benchflow._credentials import upload_subscription_auth, write_credential_files
from benchflow._env_setup import _create_environment
from benchflow._sandbox import setup_sandbox_user
from benchflow._scene import Role, Scene
from benchflow.agents.registry import AGENT_LAUNCH, AGENTS

TASK_PATH = Path(__file__).parent / "acp_smoke"

CODER_INSTRUCTION = """You are a coder. You MUST complete BOTH steps below before stopping.

Step 1: Create a file called `result.txt` in the current directory with this exact content:
hello from coder

Step 2: Create the file `/app/.outbox/reviewer.json` with this exact JSON content:
{"to": "reviewer", "content": "I wrote result.txt, please review it"}

You MUST create BOTH files. Do NOT stop until both /app/result.txt AND /app/.outbox/reviewer.json exist. Verify both files exist before stopping."""

REVIEWER_INSTRUCTION = """You are a reviewer. You MUST complete BOTH steps below before stopping.

Step 1: Read the file `/app/result.txt` and verify it contains "hello from coder".

Step 2: Create the file `/app/.outbox/coder.json` with this exact JSON content:
{"to": "coder", "content": "Reviewed result.txt - content is correct"}

You MUST create /app/.outbox/coder.json before stopping. Verify the file exists before stopping."""


_INSTALLED_AGENTS: set[str] = set()


async def role_runner(env, role: Role, prompt: str) -> None:
    """Run one role via ACP — thin wrapper around existing benchflow internals."""
    agent_config = AGENTS[role.agent]
    if role.agent not in _INSTALLED_AGENTS:
        trial_dir = Path(f"/tmp/multi-agent-proof/{role.name}")
        trial_dir.mkdir(parents=True, exist_ok=True)
        agent_config = AGENTS.get(role.agent)
        await install_agent(env, role.agent, trial_dir)
        if agent_config:
            await write_credential_files(
                env, role.agent, {}, agent_config, role.model, "/home/agent"
            )
            await upload_subscription_auth(env, role.agent, "/home/agent")
        await setup_sandbox_user(env, sandbox_user="agent", workspace="/app")
        _INSTALLED_AGENTS.add(role.agent)
    await env.exec("mkdir -p /app/.outbox && chmod 777 /app/.outbox")

    trial_dir = Path(f"/tmp/multi-agent-proof/{role.name}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    launch_cmd = AGENT_LAUNCH.get(role.agent, role.agent)
    acp_client, session, _agent_name = await connect_acp(
        env=env,
        agent=role.agent,
        agent_launch=launch_cmd,
        agent_env={},
        sandbox_user="agent",
        model=role.model,
        trial_dir=trial_dir,
        environment="daytona",
        agent_cwd="/app",
    )
    try:
        _trajectory, n_tools = await execute_prompts(
            acp_client,
            session,
            [prompt],
            timeout=120,
        )
        logging.info(f"[{role.name}] finished: {n_tools} tool calls")
    finally:
        with contextlib.suppress(Exception):
            await acp_client.close()


async def main() -> None:
    task = Task(TASK_PATH)
    env = _create_environment(
        environment_type="daytona",
        task=task,
        task_path=TASK_PATH,
        trial_name="multi-agent-proof",
        trial_paths=None,
    )
    try:
        await env.start(force_build=False)

        scene = Scene(
            roles={
                "coder": Role(
                    name="coder",
                    agent="claude-agent-acp",
                    model="claude-haiku-4-5-20251001",
                    instruction=CODER_INSTRUCTION,
                ),
                "reviewer": Role(
                    name="reviewer",
                    agent="claude-agent-acp",
                    model="claude-haiku-4-5-20251001",
                    instruction=REVIEWER_INSTRUCTION,
                ),
            },
            max_rounds=4,
        )

        trajectory = await scene.run(env, role_runner)

        out = Path("/tmp/multi-agent-proof-trajectory.jsonl")
        scene.save_trajectory(out)

        print(f"\n{'=' * 60}")
        print("MULTI-AGENT PROOF RESULTS")
        print(f"{'=' * 60}")
        print(f"Rounds: {scene._round}")
        print(f"Messages: {len(trajectory)}")
        for msg in trajectory:
            print(f"  [{msg.turn}] {msg.sender} → {msg.recipient}: {msg.content[:60]}")
        print(f"Trajectory: {out}")

        # Verify workspace state
        r = await env.exec("cat /app/result.txt 2>/dev/null || echo MISSING")
        print(f"result.txt: {(r.stdout or '').strip()!r}")

        if (
            len(trajectory) >= 2
            and trajectory[0].sender == "coder"
            and trajectory[1].sender == "reviewer"
        ):
            print("\n=== MULTI-AGENT PROOF: PASS ===")
        else:
            print("\n=== MULTI-AGENT PROOF: FAIL ===")
            print("Expected >= 2 messages with coder→reviewer→coder flow")

    finally:
        await env.stop(delete=True)


if __name__ == "__main__":
    asyncio.run(main())
