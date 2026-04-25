"""ACP transport bring-up and the multi-turn prompt loop.

Owns the live agent-side of a run:
    - connect_acp: spawn the agent process inside the container, wrap it in
      the ACP stdio transport, run initialize → session_new → set_model
    - execute_prompts: send each prompt through the session, capture the
      ACP-native trajectory, and report tool-call counts

The one allowed horizontal phase import in this refactor lives here:
``from benchflow._sandbox import build_priv_drop_cmd``. connect_acp wraps
the agent launch command in the sandbox user's privilege-drop prefix
before handing it to the transport. It is a single pure-function call
with no shared state — not a coupling of concerns.

Does not own:
    - Verifier hardening or model-set authentication — see _sandbox /
      _credentials respectively
"""

import asyncio
import logging
from pathlib import Path

from benchflow._sandbox import build_priv_drop_cmd
from benchflow._trajectory import _capture_session_trajectory
from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.agents.providers import strip_provider_prefix
from benchflow.agents.registry import AGENTS
from benchflow.process import DaytonaProcess, DaytonaPtyProcess, DockerProcess

logger = logging.getLogger(__name__)


_ACP_CONNECT_MAX_RETRIES = 3
_ACP_CONNECT_BASE_DELAY = 2.0


async def connect_acp(
    env,
    agent: str,
    agent_launch: str,
    agent_env: dict,
    sandbox_user: str | None,
    model: str | None,
    trial_dir: Path,
    environment: str,
    agent_cwd: str,
) -> tuple[ACPClient, object, str]:
    """Create ACP transport, connect, init session, set model. Return (client, session, agent_name).

    Retries with exponential backoff on ConnectionError (Daytona SSH storms).
    """
    # Resolve agent binary path for non-docker environments
    if environment != "docker":
        which_result = await env.exec(
            f"which {agent_launch.split()[0]}", timeout_sec=10
        )
        if which_result.return_code == 0 and (which_result.stdout or "").strip():
            full_path = which_result.stdout.strip()
            parts = agent_launch.split()
            parts[0] = full_path
            agent_launch = " ".join(parts)
            logger.info(f"Resolved agent path: {agent_launch}")

    if sandbox_user:
        agent_launch = build_priv_drop_cmd(agent_launch, sandbox_user)
        logger.info(f"Agent sandboxed as: {sandbox_user}")

    last_err: Exception | None = None
    acp_client: ACPClient | None = None
    for attempt in range(_ACP_CONNECT_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _ACP_CONNECT_BASE_DELAY * (2 ** (attempt - 1))
            logger.info(f"ACP connect retry {attempt}/{_ACP_CONNECT_MAX_RETRIES} after {delay:.0f}s")
            await asyncio.sleep(delay)

        try:
            if environment == "docker":
                live_proc = DockerProcess.from_harbor_env(env)
            else:
                is_dind = hasattr(env, "_strategy") and hasattr(env._strategy, "_compose_cmd")
                if is_dind:
                    live_proc = await DaytonaPtyProcess.from_harbor_env(env)
                    logger.info("Using PTY transport for DinD compose task")
                else:
                    live_proc = await DaytonaProcess.from_harbor_env(env)

            agent_log = trial_dir / "agent" / f"{agent.replace('-', '_')}.txt"
            transport = ContainerTransport(
                container_process=live_proc,
                command=agent_launch,
                env=agent_env,
                cwd=agent_cwd,
                agent_log_path=agent_log,
            )
            acp_client = ACPClient(transport)
            await acp_client.connect()

            init_result = await asyncio.wait_for(acp_client.initialize(), timeout=60)
            agent_name = init_result.agent_info.name if init_result.agent_info else agent
            logger.info(f"ACP agent: {agent_name}")

            session = await asyncio.wait_for(acp_client.session_new(cwd=agent_cwd), timeout=60)
            logger.info(f"Session: {session.session_id}")
            break
        except ConnectionError as e:
            # Close the failed client before retrying
            if acp_client:
                try:
                    await acp_client.close()
                except Exception:
                    pass
                acp_client = None
            last_err = e
            if attempt == _ACP_CONNECT_MAX_RETRIES:
                raise
            logger.warning(f"ACP connect failed (attempt {attempt + 1}): {e}")
            continue
        except Exception:
            # Non-retryable error — close client to prevent leak
            if acp_client:
                try:
                    await acp_client.close()
                except Exception:
                    pass
            raise

    agent_cfg = AGENTS.get(agent)
    if model and (agent_cfg is None or agent_cfg.supports_acp_set_model):
        acp_model_id = strip_provider_prefix(model)
        try:
            await asyncio.wait_for(acp_client.set_model(acp_model_id), timeout=60)
            logger.info(f"Model set to: {acp_model_id} (from {model})")
        except Exception as e:
            logger.warning(f"Failed to set model via ACP: {e}")
    elif model:
        logger.info(f"Skipping ACP set_model for {agent} — launch/env config owns model selection")

    return acp_client, session, agent_name


async def execute_prompts(
    acp_client: ACPClient,
    session,
    prompts: list[str],
    timeout: int,
) -> tuple[list[dict], int]:
    """Send prompts via ACP and capture trajectory. Return (trajectory, n_tool_calls)."""
    for i, prompt in enumerate(prompts):
        logger.info(f"Prompt {i + 1}/{len(prompts)}: {(prompt or '<instruction.md>')[:80]}...")
        prompt_result = await asyncio.wait_for(
            acp_client.prompt(prompt),
            timeout=timeout,
        )
        logger.info(
            f"  → {prompt_result.stop_reason.value}, "
            f"{len(session.tool_calls)} total tool calls"
        )
    trajectory = _capture_session_trajectory(session)
    return trajectory, len(session.tool_calls)
