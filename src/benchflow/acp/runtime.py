"""ACP transport bring-up and the multi-turn prompt loop.

Owns the live agent-side of a run:
    - connect_acp: spawn the agent process inside the container, wrap it in
      the ACP stdio transport, run initialize → session_new → model/effort config
    - execute_prompts: send each prompt through the session, capture the
      ACP-native trajectory, and report tool-call counts

The one allowed horizontal phase import in this refactor lives here:
``from benchflow.sandbox.lockdown import build_priv_drop_cmd``. connect_acp wraps
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
from datetime import UTC, datetime
from pathlib import Path

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.types import McpServerSpec
from benchflow.agents.protocol import ACPSessionAdapter
from benchflow.agents.providers import (
    find_provider,
    find_provider_for_bare_model,
    strip_provider_prefix,
)
from benchflow.agents.registry import AGENTS
from benchflow.diagnostics import (
    AgentPromptTimeoutDiagnostic,
    AgentPromptTimeoutError,
    IdleTimeoutDiagnostic,
    IdleTimeoutError,
)
from benchflow.sandbox.lockdown import build_priv_drop_cmd
from benchflow.sandbox.process import DaytonaProcess, DaytonaPtyProcess, DockerProcess
from benchflow.trajectories._capture import _capture_session_trajectory

# Re-exported for backwards compatibility — tests and downstream code
# import ``IdleTimeoutError`` from this module. The canonical definition
# lives in :mod:`benchflow.diagnostics` (issue #503).
__all__ = [
    "AgentPromptTimeoutError",
    "IdleTimeoutError",
    "connect_acp",
    "execute_prompts",
]

logger = logging.getLogger(__name__)


_ACP_CONNECT_MAX_RETRIES = 3
_ACP_CONNECT_BASE_DELAY = 2.0
_PROMPT_CANCEL_DRAIN_TIMEOUT_SEC = 0.25


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


def _codex_model_name(model_id: str) -> str:
    """Return the bare model name from a Codex ACP ``model[effort]`` ID."""
    return model_id.split("[", 1)[0]


def _codex_reasoning_effort(model_id: str) -> str:
    """Return the reasoning effort suffix from a Codex ACP model ID."""
    if "[" not in model_id or not model_id.endswith("]"):
        return ""
    return model_id.rsplit("[", 1)[1][:-1]


def _codex_session_model_id(model: str, session: object | None) -> str:
    """Map a bare Codex model to the exact ACP modelId returned by session/new.

    ``@agentclientprotocol/codex-acp`` validates ``session/set_model`` against
    model IDs shaped as ``model[reasoning-effort]``. BenchFlow's public model
    IDs stay effort-free, so use the session's advertised model state to choose
    the concrete Codex ACP ID.
    """
    state = getattr(session, "model_state", None)
    if not isinstance(state, dict):
        return model

    requested_name = _codex_model_name(model)
    current_model = state.get("currentModelId")
    if (
        isinstance(current_model, str)
        and _codex_model_name(current_model) == requested_name
    ):
        return current_model

    available = state.get("availableModels")
    if not isinstance(available, list):
        return model
    candidates = [
        entry.get("modelId")
        for entry in available
        if isinstance(entry, dict)
        and isinstance(entry.get("modelId"), str)
        and _codex_model_name(entry["modelId"]) == requested_name
    ]
    if not candidates:
        return model

    preferred_efforts = ("medium", "high", "low", "minimal", "none")
    for effort in preferred_efforts:
        for candidate in candidates:
            if _codex_reasoning_effort(candidate) == effort:
                return candidate
    return candidates[0]


def _format_acp_model(model: str, agent: str) -> str:
    """Format a model ID for ACP session/set_model based on agent requirements.

    NOTE on "provider/model": this is NOT a non-standard ACP method. BenchFlow
    only ever sends the standard ``session/set_model`` request (see
    ``ACPClient.set_model``). ``provider/model`` is one of three *modelId string
    formats* (``acp_model_format`` in the agent registry) describing how the
    ``modelId`` argument of that standard call must be shaped for a given agent.
    It deliberately does not appear in ``acp.meta.AGENT_METHODS``.

    Most agents expect a bare model name (e.g. "claude-sonnet-4-6").
    Agents with acp_model_format="provider/model" (e.g. opencode) need the
    models.dev provider prefix (e.g. "google/gemini-3.1-pro-preview").
    Agents with acp_model_format="registered-provider/model" (e.g. pi-acp)
    need BenchFlow's registered provider prefix preserved for custom providers
    because the launcher registers that provider key in the agent config.

    Strips benchflow's custom provider prefixes first, then re-adds the
    models.dev provider prefix when the agent requires it.
    """
    bare = strip_provider_prefix(model)
    agent_cfg = AGENTS.get(agent)
    if agent_cfg and agent_cfg.acp_model_format == "registered-provider/model":
        # Proxy mode: BenchFlow's LiteLLM proxy serves the model under the alias
        # "benchflow-…", which the pi-acp launcher registers under the "litellm"
        # provider (BENCHFLOW_PROVIDER_NAME) in models.json. The alias carries no
        # provider prefix, so sending it bare leaves Pi unable to resolve it — the
        # model calls then bypass the proxy and no llm_trajectory.jsonl is written.
        # Send the registered-provider-qualified id so Pi hits the gateway route.
        if model.startswith("benchflow-"):
            return f"litellm/{model}"
        return model if find_provider(model) else bare
    if not agent_cfg or agent_cfg.acp_model_format != "provider/model":
        return bare
    # Already has a slash — assume it's provider/model already
    if "/" in bare:
        return bare
    # BenchFlow's LiteLLM proxy registers every model under "openai/<alias>"
    # (aliases are always "benchflow-…"). Send that so provider/model agents
    # (e.g. opencode) hit a registered route instead of a guessed provider that
    # the proxy never serves (the heuristic would default to anthropic/).
    if bare.startswith("benchflow-"):
        return f"openai/{bare}"
    # Provider ownership lives in the registry: if a ProviderConfig claims this
    # bare model family via its declared model_prefixes, route through it
    # (e.g. mimo-v2.5 -> xiaomi, deepseek-v4-flash -> deepseek). This keeps
    # provider/model-family knowledge in the provider registry instead of
    # growing provider-specific branches in the runtime.
    registry_match = find_provider_for_bare_model(bare)
    if registry_match is not None:
        return f"{registry_match[0]}/{bare}"
    # Fallback: infer a models.dev provider for families without a registered
    # ProviderConfig (e.g. openai/anthropic/google) from the bare model name.
    m = bare.lower()
    for substring, provider in _MODELSDEV_PROVIDER_HEURISTICS:
        if substring in m:
            return f"{provider}/{bare}"
    logger.warning(
        "Cannot infer models.dev provider for %r — defaulting to anthropic/", bare
    )
    return f"anthropic/{bare}"


def _select_acp_model_id(
    model: str,
    agent: str,
    session: object | None,
) -> str:
    """Return the concrete modelId to send through ACP session/set_model."""
    formatted = _format_acp_model(model, agent)
    if agent == "codex-acp":
        return _codex_session_model_id(formatted, session)
    return formatted


def _model_selection_owned_by_env(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Return True when launch/env config should own model selection.

    Custom provider runtimes such as Bedrock can expose a model ID through
    agent-native env vars (for Claude ACP this is ``ANTHROPIC_MODEL``).
    In that case ACP model configuration can be actively harmful because the
    agent may validate the model ID against its native catalog before it ever
    hits the custom provider endpoint.
    """
    if not model:
        return False
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg:
        return False
    # LiteLLM routing: when the model is delivered through an agent-native env
    # var (e.g. LLM_MODEL/ANTHROPIC_MODEL + BENCHFLOW_LITELLM_MODEL_VIA_ENV)
    # the agent must not also receive ACP model config. Agents without a native
    # model env mapping (codex-acp) still need ACP configuration so they do not
    # fall back to their own defaults.
    if agent_env.get("BENCHFLOW_LITELLM_MODEL_VIA_ENV") in {"1", "true", "True"}:
        mapped_model_env = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_MODEL")
        return bool(mapped_model_env and agent_env.get(mapped_model_env))
    if agent_env.get("BENCHFLOW_LITELLM_MODEL_ALIAS"):
        return False
    provider = find_provider(model)
    if provider is None:
        return False
    _provider_name, provider_cfg = provider
    if provider_cfg.auth_type != "aws":
        return False
    if agent_env.get("CLAUDE_CODE_USE_BEDROCK") in {"1", "true", "True"}:
        return False
    mapped_model_env = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_MODEL")
    if not mapped_model_env:
        return False
    override = agent_env.get(mapped_model_env)
    if not override:
        return True
    return strip_provider_prefix(model) == override


def _resolve_acp_model_input(agent: str, model: str, agent_env: dict[str, str]) -> str:
    """Pick the model string that should be sent through ACP model config."""
    litellm_alias = agent_env.get("BENCHFLOW_LITELLM_MODEL_ALIAS")
    if litellm_alias and agent != "codex-acp":
        return litellm_alias
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg:
        return model
    provider = find_provider(model)
    if provider is None:
        return model
    _provider_name, provider_cfg = provider
    if provider_cfg.auth_type != "aws":
        return model
    mapped_model_env = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_MODEL")
    if not mapped_model_env:
        return model
    return agent_env.get(mapped_model_env, model)


def _session_config_option_ids(session: object | None) -> set[str]:
    options = getattr(session, "config_options", None)
    if not isinstance(options, (list, tuple)):
        return set()
    ids: set[str] = set()
    for option in options:
        if isinstance(option, dict) and isinstance(option.get("id"), str):
            ids.add(option["id"])
    return ids


def _resolve_acp_model_option_id(
    agent_cfg: object | None, session: object | None
) -> str | None:
    """Config option id to drive model selection — capability-first.

    The running agent's advertised ``session/new`` config options are the source
    of truth; the registry's ``acp_model_config_id`` is an override/hint. Returns
    ``None`` when no model config option applies, in which case the caller falls
    back to ``session/set_model``.

    This is what lets the ``@agentclientprotocol`` family migrate from
    ``session/set_model`` to a ``"model"`` config option with no registry
    change: a member that advertises a model option is configured through it
    automatically, while a member that does not (e.g. current ``codex-acp``,
    which advertises only ``fast-mode``) keeps using ``session/set_model``.
    """
    declared = getattr(agent_cfg, "acp_model_config_id", "") or ""
    if declared:
        return declared
    if "model" in _session_config_option_ids(session):
        return "model"
    return None


async def _set_acp_model(
    acp_client: ACPClient,
    *,
    agent: str,
    model_id: str,
) -> None:
    try:
        await asyncio.wait_for(acp_client.set_model(model_id), timeout=60)
        logger.info(f"Model set to: {model_id}")
    except Exception as e:
        logger.error(
            "ACP session/set_model failed for agent=%s model=%s: %s",
            agent,
            model_id,
            e,
        )
        raise RuntimeError(
            f"Failed to set model {model_id!r} via ACP for agent {agent!r}: {e}"
        ) from e


async def _set_acp_config_option(
    acp_client: ACPClient,
    session: object | None,
    *,
    agent: str,
    config_id: str,
    value: str,
    label: str,
) -> None:
    option_ids = _session_config_option_ids(session)
    if option_ids and config_id not in option_ids:
        raise RuntimeError(
            f"ACP agent {agent!r} does not expose {label} config option "
            f"{config_id!r}; available options: {sorted(option_ids)!r}"
        )
    try:
        await asyncio.wait_for(
            acp_client.set_config_option(config_id, value), timeout=60
        )
        logger.info(f"ACP {label} config option {config_id!r} set to: {value}")
    except Exception as e:
        logger.error(
            "ACP session/set_config_option failed for agent=%s config=%s value=%s: %s",
            agent,
            config_id,
            value,
            e,
        )
        raise RuntimeError(
            f"Failed to set ACP {label} config option {config_id!r}="
            f"{value!r} for agent {agent!r}: {e}"
        ) from e


async def _configure_acp_session(
    acp_client: ACPClient,
    session: object | None,
    *,
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    reasoning_effort: str | None,
) -> None:
    agent_cfg = AGENTS.get(agent)

    if model and _model_selection_owned_by_env(agent, model, agent_env):
        logger.info(
            f"Skipping ACP model configuration for {agent} — launch/env config owns model selection"
        )
    elif model:
        acp_model_input = _resolve_acp_model_input(agent, model, agent_env)
        acp_model_id = _select_acp_model_id(acp_model_input, agent, session)
        model_option_id = _resolve_acp_model_option_id(agent_cfg, session)
        if model_option_id:
            # Capability-first: the agent advertises a model config option (or
            # the registry declares one), so configure the model through it —
            # this is how the @agentclientprotocol family is replacing
            # session/set_model, and it needs no per-agent registry change.
            await _set_acp_config_option(
                acp_client,
                session,
                agent=agent,
                config_id=model_option_id,
                value=acp_model_id,
                label="model",
            )
        elif not agent_cfg or agent_cfg.supports_acp_set_model:
            # No model config option advertised/declared — use the legacy
            # session/set_model path. Fails closed if the agent rejects it.
            await _set_acp_model(acp_client, agent=agent, model_id=acp_model_id)
        else:
            logger.info(
                f"Skipping ACP model configuration for {agent} — launch/env config owns model selection"
            )

    if not reasoning_effort:
        return
    if not agent_cfg or not agent_cfg.acp_effort_config_id:
        raise RuntimeError(
            f"reasoning_effort={reasoning_effort!r} was requested for agent "
            f"{agent!r}, but that agent does not declare an ACP effort config option"
        )
    await _set_acp_config_option(
        acp_client,
        session,
        agent=agent,
        config_id=agent_cfg.acp_effort_config_id,
        value=reasoning_effort,
        label="reasoning effort",
    )


async def connect_acp(
    env,
    agent: str,
    agent_launch: str,
    agent_env: dict,
    sandbox_user: str | None,
    model: str | None,
    rollout_dir: Path,
    environment: str,
    agent_cwd: str,
    reasoning_effort: str | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
) -> tuple[ACPClient, object, ACPSessionAdapter, str]:
    """Create ACP transport, connect, init session, and configure model/effort.

    Returns ``(client, session, session_adapter, agent_name)``. ``session`` is
    the raw :class:`~benchflow.acp.session.ACPSession` (still passed to
    ``execute_prompts`` for trajectory capture and the idle watchdog).
    ``session_adapter`` is the :class:`ACPSessionAdapter` bound to ``client``;
    registering an ``on_ask_user`` handler on it is the only way a handler
    reaches the live ``session/request_permission`` path on the wire. Without
    instantiating the adapter here, every handler the kernel registers stayed
    dormant and the auto-approve policy ran unconditionally (#382 follow-up).

    ``mcp_servers`` are the task's configured MCP servers (mapped from
    ``[[environment.mcp_servers]]``); they are attached to the ACP session at
    ``session/new`` so the agent can reach them. ``None`` attaches none.

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
    session: object | None = None
    agent_name = agent
    for attempt in range(_ACP_CONNECT_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _ACP_CONNECT_BASE_DELAY * (2 ** (attempt - 1))
            logger.info(
                f"ACP connect retry {attempt}/{_ACP_CONNECT_MAX_RETRIES} after {delay:.0f}s"
            )
            await asyncio.sleep(delay)

        try:
            if environment == "docker":
                live_proc = DockerProcess.from_sandbox_env(env)
            else:
                is_dind = hasattr(env, "_strategy") and hasattr(
                    env._strategy, "_compose_cmd"
                )
                if is_dind:
                    live_proc = await DaytonaPtyProcess.from_sandbox_env(env)
                    logger.info("Using PTY transport for DinD compose task")
                else:
                    live_proc = await DaytonaProcess.from_sandbox_env(env)

            agent_log = rollout_dir / "agent" / f"{agent.replace('-', '_')}.txt"
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
            agent_name = (
                init_result.agent_info.name if init_result.agent_info else agent
            )
            logger.info(f"ACP agent: {agent_name}")

            session = await asyncio.wait_for(
                acp_client.session_new(cwd=agent_cwd, mcp_servers=mcp_servers),
                timeout=60,
            )
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

    if acp_client is None or session is None:
        raise RuntimeError("ACP connection did not initialize")

    try:
        await _configure_acp_session(
            acp_client,
            session,
            agent=agent,
            model=model,
            agent_env=agent_env,
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        with contextlib.suppress(Exception):
            await acp_client.close()
        raise

    session_adapter = ACPSessionAdapter(acp_client)
    return acp_client, session, session_adapter, agent_name


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
        logger.info(
            f"Prompt {i + 1}/{len(prompts)}: {(prompt or '<instruction.md>')[:80]}..."
        )
        session.record_user_prompt(prompt)
        if idle_timeout is None:
            try:
                prompt_result = await _prompt_with_wall_clock_budget(
                    acp_client, session, prompt, timeout
                )
            except AgentPromptTimeoutError as e:
                e.executed_prompts = prompts[: i + 1]
                raise
        else:
            try:
                prompt_result = await _prompt_with_idle_watchdog(
                    acp_client, session, prompt, timeout, idle_timeout
                )
            except AgentPromptTimeoutError as e:
                e.executed_prompts = prompts[: i + 1]
                raise
        session.mark_prompt_end()
        # SDK ``PromptResponse.stop_reason`` is a plain string (e.g. "end_turn").
        logger.info(
            f"  → {prompt_result.stop_reason}, "
            f"{len(session.tool_calls)} total tool calls"
        )
    trajectory = _capture_session_trajectory(session)
    return trajectory, len(session.tool_calls)


async def _cancel_and_drain_prompt_task(prompt_task: asyncio.Task) -> bool:
    if prompt_task.done():
        return True
    prompt_task.cancel()
    done, _pending = await asyncio.wait(
        {prompt_task}, timeout=_PROMPT_CANCEL_DRAIN_TIMEOUT_SEC
    )
    if done:
        with contextlib.suppress(BaseException):
            prompt_task.result()
        return True

    logger.warning(
        "ACP prompt task did not finish within %.2fs after cancellation",
        _PROMPT_CANCEL_DRAIN_TIMEOUT_SEC,
    )

    def _consume_prompt_result(task: asyncio.Task) -> None:
        with contextlib.suppress(BaseException):
            task.result()

    prompt_task.add_done_callback(_consume_prompt_result)
    return False


def _agent_prompt_timeout_error(session, timeout: int) -> AgentPromptTimeoutError:
    session.mark_prompt_end()
    pending_tool_call_ids = session.pending_tool_call_ids()
    terminal_complete = not pending_tool_call_ids
    session.record_agent_timeout(
        timeout_sec=float(timeout),
        pending_tool_call_ids=pending_tool_call_ids,
        terminal_trajectory_complete=terminal_complete,
    )
    diagnostic = AgentPromptTimeoutDiagnostic(
        timeout_sec=float(timeout),
        n_tool_calls=len(session.tool_calls),
        pending_tool_call_ids=pending_tool_call_ids,
        terminal_event_recorded=True,
        terminal_trajectory_complete=terminal_complete,
    )
    return AgentPromptTimeoutError(
        f"Agent prompt exceeded wall-clock budget {timeout}s",
        trajectory=_capture_session_trajectory(session),
        diagnostic=diagnostic,
    )


async def _prompt_with_wall_clock_budget(
    acp_client: ACPClient,
    session,
    prompt: str,
    timeout: int,
):
    """Run a prompt until either it finishes or BenchFlow's budget expires."""
    prompt_task = asyncio.create_task(acp_client.prompt(prompt))
    try:
        done, _pending = await asyncio.wait({prompt_task}, timeout=timeout)
        if done:
            return prompt_task.result()
        if await _cancel_and_drain_prompt_task(prompt_task):
            raise _agent_prompt_timeout_error(session, timeout)
        raise TimeoutError(f"Agent prompt exceeded wall-clock budget {timeout}s")
    finally:
        if not prompt_task.done():
            await _cancel_and_drain_prompt_task(prompt_task)


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
    last_activity_at = datetime.now(UTC)
    last_count = _activity_count()
    # poll_interval considers BOTH idle_timeout and wall-clock timeout so that
    # short overall budgets don't overshoot (e.g. timeout=30s with default
    # poll_interval=30s could overshoot 100%). Cap at 30s, floor at 1s.
    poll_interval = max(1, min(30, idle_timeout // 4, max(1, timeout // 4)))
    # deadline is loop-INVARIANT (fixed at loop start), NOT re-derived from
    # last_progress inside the loop. This is load-bearing: the idle branch below
    # resets last_progress while a tool call is in-flight, so re-deriving the
    # deadline from it would let an agent that emits a tool_call and then hangs
    # forever push the wall-clock backstop out indefinitely. Keep this fixed.
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
                last_activity_at = datetime.now(UTC)
                last_count = cur_count
            # An in-flight tool call means the agent is actively executing a tool
            # (e.g. a long build/test/solver shell command), not hung. Those tools
            # emit no ACP updates until they return, so a >idle_timeout run would
            # otherwise false-fire the watchdog and discard real work. Treat a
            # pending tool call as progress and defer to the wall-clock `timeout`
            # backstop below for a tool that never returns. A genuine model-side
            # hang has no pending tool call (the prior tool already completed via
            # tool_call_update), so it still trips the idle path.
            elif session.pending_tool_call_ids():
                last_progress = now
                last_activity_at = datetime.now(UTC)
            if now - last_progress >= idle_timeout:
                diag = IdleTimeoutDiagnostic(
                    idle_timeout_sec=idle_timeout,
                    idle_duration_sec=int(now - last_progress),
                    wall_clock_elapsed_sec=int(now - (deadline - timeout)),
                    n_tool_calls=len(session.tool_calls),
                    n_message_chunks=len(session.message_chunks),
                    n_thought_chunks=len(session.thought_chunks),
                    last_activity_at=last_activity_at.isoformat(),
                )
                raise IdleTimeoutError(
                    f"Agent idle for {idle_timeout}s with no new tool call, "
                    f"message, or thought "
                    f"(last activity {int(now - last_progress)}s ago, "
                    f"{len(session.tool_calls)} tool calls so far)",
                    diag,
                )
            if now > deadline:
                if await _cancel_and_drain_prompt_task(prompt_task):
                    raise _agent_prompt_timeout_error(session, timeout)
                raise TimeoutError(
                    f"Agent prompt exceeded wall-clock budget {timeout}s"
                )

        return prompt_task.result()
    finally:
        # Always cancel + drain the prompt task on exit, including the
        # external-cancellation path (CancelledError from sleep). Bound the
        # drain so a non-cooperative Daytona/ACP read cannot hide the watchdog
        # timeout forever; cleanup will tear down the live process.
        if not prompt_task.done():
            await _cancel_and_drain_prompt_task(prompt_task)
