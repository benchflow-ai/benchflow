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

from benchflow.acp.client import ACPClient, ACPError
from benchflow.acp.container_transport import ContainerTransport
from benchflow.agents.protocol import ACPSessionAdapter
from benchflow.agents.providers import find_provider, strip_provider_prefix
from benchflow.agents.registry import AGENTS
from benchflow.diagnostics import IdleTimeoutDiagnostic, IdleTimeoutError
from benchflow.sandbox.lockdown import build_priv_drop_cmd
from benchflow.sandbox.process import DaytonaProcess, DaytonaPtyProcess, DockerProcess
from benchflow.trajectories._capture import _capture_session_trajectory

# Re-exported for backwards compatibility — tests and downstream code
# import ``IdleTimeoutError`` from this module. The canonical definition
# lives in :mod:`benchflow.diagnostics` (issue #503).
__all__ = ["IdleTimeoutError", "connect_acp", "execute_prompts"]

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
        return model if find_provider(model) else bare
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
    options = getattr(session, "config_options", None) or []
    ids: set[str] = set()
    for option in options:
        if isinstance(option, dict) and isinstance(option.get("id"), str):
            ids.add(option["id"])
    return ids


def _advertised_model_option_id(
    session: object | None, agent_cfg: object | None
) -> str | None:
    """Config option id to drive model selection, discovered from the session.

    Prefers the registry-declared ``acp_model_config_id``; otherwise falls back
    to the conventional ``"model"`` id when the running agent advertises it.
    This is what lets the ``@agentclientprotocol`` family keep working when an
    agent version drops ``session/set_model`` in favor of the model config
    option — no registry change required.
    """
    advertised = _session_config_option_ids(session)
    declared = getattr(agent_cfg, "acp_model_config_id", "") or ""
    if declared and declared in advertised:
        return declared
    if "model" in advertised:
        return "model"
    return None


def _is_acp_method_not_found(exc: BaseException | None) -> bool:
    """True when ``exc`` or any of its causes is a JSON-RPC -32601.

    ``-32601`` ("Method not found") is how an ACP agent reports that it does not
    implement a method. ``_set_acp_model`` wraps the raw :class:`ACPError` in a
    ``RuntimeError``, so walk the ``__cause__`` chain rather than only the top
    frame. The ``seen`` guard avoids looping on a self-referential cause.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ACPError) and exc.code == -32601:
            return True
        exc = exc.__cause__
    return False


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
    model_owned_by_env = _model_selection_owned_by_env(agent, model, agent_env)

    if model and model_owned_by_env:
        logger.info(
            f"Skipping ACP model configuration for {agent} — launch/env config owns model selection"
        )
    elif model and agent_cfg and agent_cfg.acp_model_config_id:
        acp_model_input = _resolve_acp_model_input(agent, model, agent_env)
        acp_model_id = _select_acp_model_id(acp_model_input, agent, session)
        await _set_acp_config_option(
            acp_client,
            session,
            agent=agent,
            config_id=agent_cfg.acp_model_config_id,
            value=acp_model_id,
            label="model",
        )
    elif model and (not agent_cfg or agent_cfg.supports_acp_set_model):
        acp_model_input = _resolve_acp_model_input(agent, model, agent_env)
        acp_model_id = _select_acp_model_id(acp_model_input, agent, session)
        try:
            await _set_acp_model(acp_client, agent=agent, model_id=acp_model_id)
        except RuntimeError as exc:
            # Capability fallback: if the running agent version has dropped
            # session/set_model (JSON-RPC -32601) — as the @agentclientprotocol
            # family is doing in favor of config options — recover by routing
            # the model through the advertised model config option instead of
            # failing the rollout. This defuses the same break that hit
            # claude-agent-acp for codex-acp and any future family member with
            # no registry change. Any other failure still fails closed.
            fallback_id = _advertised_model_option_id(session, agent_cfg)
            if not (_is_acp_method_not_found(exc) and fallback_id):
                raise
            logger.warning(
                "ACP agent %r does not implement session/set_model (-32601); "
                "falling back to the %r config option",
                agent,
                fallback_id,
            )
            await _set_acp_config_option(
                acp_client,
                session,
                agent=agent,
                config_id=fallback_id,
                value=acp_model_id,
                label="model",
            )
    elif model:
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
                acp_client.session_new(cwd=agent_cwd), timeout=60
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
            prompt_result = await asyncio.wait_for(
                acp_client.prompt(prompt),
                timeout=timeout,
            )
        else:
            prompt_result = await _prompt_with_idle_watchdog(
                acp_client, session, prompt, timeout, idle_timeout
            )
        session.mark_prompt_end()
        # SDK ``PromptResponse.stop_reason`` is a plain string (e.g. "end_turn").
        logger.info(
            f"  → {prompt_result.stop_reason}, "
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
    # poll_interval considers BOTH idle_timeout and wall-clock timeout so that
    # short overall budgets don't overshoot (e.g. timeout=30s with default
    # poll_interval=30s could overshoot 100%). Cap at 30s, floor at 1s.
    poll_interval = max(1, min(30, idle_timeout // 4, max(1, timeout // 4)))
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
                diag = IdleTimeoutDiagnostic(
                    idle_timeout_sec=idle_timeout,
                    idle_duration_sec=int(now - last_progress),
                    wall_clock_elapsed_sec=int(now - (deadline - timeout)),
                    n_tool_calls=len(session.tool_calls),
                    n_message_chunks=len(session.message_chunks),
                    n_thought_chunks=len(session.thought_chunks),
                    last_activity_at=datetime.now(UTC).isoformat(),
                )
                raise IdleTimeoutError(
                    f"Agent idle for {idle_timeout}s with no new tool call, "
                    f"message, or thought "
                    f"(last activity {int(now - last_progress)}s ago, "
                    f"{len(session.tool_calls)} tool calls so far)",
                    diag,
                )
            if now > deadline:
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
            prompt_task.cancel()
            done, _pending = await asyncio.wait(
                {prompt_task}, timeout=_PROMPT_CANCEL_DRAIN_TIMEOUT_SEC
            )
            if done:
                with contextlib.suppress(BaseException):
                    prompt_task.result()
            else:
                logger.warning(
                    "ACP prompt task did not finish within %.2fs after cancellation",
                    _PROMPT_CANCEL_DRAIN_TIMEOUT_SEC,
                )

                def _consume_prompt_result(task: asyncio.Task) -> None:
                    with contextlib.suppress(BaseException):
                        task.result()

                prompt_task.add_done_callback(_consume_prompt_result)
