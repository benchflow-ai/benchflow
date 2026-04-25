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
import contextlib
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

# models.dev provider inference — used when acp_model_format="provider/model"
# to reconstruct "provider/model" from a bare model name.
_MODELSDEV_PROVIDER_HEURISTICS: list[tuple[str, str]] = [
    # (substring in model name, models.dev provider ID)
    ("gemini", "google"),
    ("gemma", "google"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("haiku", "anthropic"),
    ("sonnet", "anthropic"),
    ("opus", "anthropic"),
    ("mistral", "mistral"),
    ("codestral", "mistral"),
]


def _format_acp_model(model: str, agent: str) -> str:
    """Format a model ID for ACP session/set_model based on agent requirements.

    Most agents expect a bare model name (e.g. "claude-sonnet-4-6").
    Agents with acp_model_format="provider/model" (e.g. opencode) need the
    models.dev provider prefix (e.g. "google/gemini-3.1-pro-preview").

    Strips benchflow's custom provider prefixes first, then re-adds the
    models.dev provider prefix when the agent requires it.
    """
    bare = strip_provider_prefix(model)
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg or agent_cfg.acp_model_format != "provider/model":
        return bare
    # Already has a slash — assume it's provider/model already
    if "/" in bare:
        return bare
    # Infer the models.dev provider from the bare model name
    m = bare.lower()
    for substring, provider in _MODELSDEV_PROVIDER_HEURISTICS:
        if substring in m:
            return f"{provider}/{bare}"
    logger.warning(
        "Cannot infer models.dev provider for %r — defaulting to anthropic/", bare
    )
    return f"anthropic/{bare}"


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
                with contextlib.suppress(Exception):
                    await acp_client.close()
                acp_client = None
            if attempt == _ACP_CONNECT_MAX_RETRIES:
                raise
            logger.warning(f"ACP connect failed (attempt {attempt + 1}): {e}")
            continue
        except Exception:
            # Non-retryable error — close client to prevent leak
            if acp_client:
                with contextlib.suppress(Exception):
                    await acp_client.close()
            raise

    agent_cfg = AGENTS.get(agent)
    if model and (agent_cfg is None or agent_cfg.supports_acp_set_model):
        acp_model_id = _format_acp_model(model, agent)
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
    idle_timeout: int | None = None,
) -> tuple[list[dict], int]:
    """Send prompts via ACP and capture trajectory. Return (trajectory, n_tool_calls).

    timeout      — wall-clock budget for each prompt (full agent budget).
    idle_timeout — abort the prompt if no tool call or message arrives for
                   this many seconds. Catches agents that hung silently while
                   the agent process is still alive (e.g. gemini-cli not
                   responding). None disables idle detection.
    """
    for i, prompt in enumerate(prompts):
        logger.info(f"Prompt {i + 1}/{len(prompts)}: {(prompt or '<instruction.md>')[:80]}...")
        if idle_timeout is None:
            prompt_result = await asyncio.wait_for(
                acp_client.prompt(prompt),
                timeout=timeout,
            )
        else:
            prompt_result = await _prompt_with_idle_watchdog(
                acp_client, session, prompt, timeout, idle_timeout
            )
        logger.info(
            f"  → {prompt_result.stop_reason.value}, "
            f"{len(session.tool_calls)} total tool calls"
        )
    trajectory = _capture_session_trajectory(session)
    return trajectory, len(session.tool_calls)


async def _prompt_with_idle_watchdog(
    acp_client: ACPClient,
    session,
    prompt: str,
    timeout: int,
    idle_timeout: int,
):
    """Run acp_client.prompt() with both a wall-clock and an idle watchdog.

    The watchdog polls session.tool_calls every few seconds and aborts if no
    progress was made in idle_timeout. This catches agents that hung silently
    while the local process is still alive (no output to stdout, no tool calls
    appended).
    """
    def _activity_count() -> int:
        # Match the docstring contract: idle = no tool call AND no message
        # AND no thought. Sum all three so streamed text resets the timer.
        return (
            len(session.tool_calls)
            + len(session.message_chunks)
            + len(session.thought_chunks)
        )

    prompt_task = asyncio.create_task(acp_client.prompt(prompt))
    last_progress = asyncio.get_event_loop().time()
    last_count = _activity_count()
    poll_interval = min(30, max(5, idle_timeout // 4))
    deadline = last_progress + timeout

    try:
        while not prompt_task.done():
            await asyncio.sleep(poll_interval)
            # Re-check done() after the sleep — the prompt may have completed
            # during the poll interval. Without this, we'd cancel an already-
            # completed task and discard a successful result.
            if prompt_task.done():
                break
            now = asyncio.get_event_loop().time()
            cur_count = _activity_count()
            if cur_count > last_count:
                last_progress = now
                last_count = cur_count
            if now - last_progress >= idle_timeout:
                raise TimeoutError(
                    f"Agent idle for {idle_timeout}s with no new tool call, "
                    f"message, or thought "
                    f"(last activity {int(now - last_progress)}s ago, "
                    f"{len(session.tool_calls)} tool calls so far)"
                )
            if now > deadline:
                raise TimeoutError(
                    f"Agent prompt exceeded wall-clock budget {timeout}s"
                )

        return prompt_task.result()
    finally:
        # Always cancel + drain the prompt task on exit, including the
        # external-cancellation path (CancelledError from sleep). Without this
        # an outer cancel leaks the prompt task — it keeps running in the
        # background until Trial.cleanup() eventually kills the agent process.
        if not prompt_task.done():
            prompt_task.cancel()
            with contextlib.suppress(BaseException):
                await prompt_task
