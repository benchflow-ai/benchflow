"""Session-factory connect + drive — the non-ACP kernel path.

A ``protocol="session-factory"`` agent declares a ``session_factory`` dotted
entrypoint (``"module:callable"``) that builds an :class:`~benchflow.agents.
protocol.Agent`. Unlike ACP there is no transport: :func:`connect_session_factory`
imports the entrypoint, builds the Agent, and calls ``agent.connect(sandbox,
role)`` → :class:`~benchflow.agents.protocol.Session`. The drive loop
:func:`execute_prompts_session_factory` calls ``session.prompt(text)`` per turn
and captures the session's ``steps`` as the trajectory; ``on_change`` is wired
by the kernel's ``_attach_trajectory_writer`` (same as ACP).

LLM-usage capture is **protocol-agnostic** — the agent's provider traffic is
routed through the litellm proxy (via the ``BENCHFLOW_PROVIDER_*`` env the kernel
mints), and the proxy logs raw request/response + token usage to
``llm_trajectory.jsonl`` regardless of agent protocol. So token counts flow for a
session-factory agent exactly as for ACP; that is what keeps a healthy run valid
(``_maybe_classify_api_error`` nulls reward only when tokens==0 AND tool_calls==0,
and a session-factory agent reports 0 tool calls — its one-shot CLI exposes no
per-call stream — so captured tokens are load-bearing).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.diagnostics import (
    AgentPromptTimeoutDiagnostic,
    AgentPromptTimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionFactorySandbox:
    """Sandbox handle plus session-factory connect metadata.

    Session-factory agents still receive the declared ``Agent.connect(sandbox,
    role)`` shape. The wrapper delegates the sandbox API while exposing the
    kernel-resolved environment that an in-process agent may need during connect.
    """

    sandbox: Any
    agent_env: dict[str, str]
    agent_cwd: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sandbox, name)


def _load_session_factory(dotted: str) -> Any:
    """Resolve a ``"module:callable"`` entrypoint to the callable.

    The dotted form is the registry's ``AgentConfig.session_factory`` (e.g.
    ``"omnigent.agent:build_omnigent_agent"``). Raises ValueError on a malformed
    spec so a typo fails loud at connect rather than silently no-op'ing.
    """
    if ":" not in dotted:
        raise ValueError(f"session_factory must be 'module:callable', got {dotted!r}")
    mod_name, _, attr = dotted.partition(":")
    if not mod_name or not attr:
        raise ValueError(f"session_factory must be 'module:callable', got {dotted!r}")
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


async def connect_session_factory(
    env: Any,
    agent: str,
    session_factory: str,
    agent_env: dict[str, str],
    sandbox_user: str | None,
    model: str | None,
    rollout_dir: Path | None,
    timeout: float,
    agent_cwd: str | None = None,
    **_ignored: Any,
) -> tuple[None, object, None, str]:
    """Build the session-factory Agent and connect → Session.

    Returns ``(None, session, None, agent_name)`` — shape-compatible with
    ``connect_acp``'s ``(client, session, adapter, agent_name)`` so the kernel
    unpacks both paths into the same four slots. A session-factory agent has no
    ACP client or session adapter (the session IS the protocol-conformant
    object), so those two slots are ``None``.

    ``agent_env`` carries the kernel's resolved per-role provider routing
    (``BENCHFLOW_PROVIDER_BASE_URL`` = the litellm proxy, ``_API_KEY`` = the
    master key, ``_MODEL`` = the route alias). It is exposed on the sandbox
    wrapper because a session-factory agent runs in-process on the host (no
    subprocess env injection), and it is what makes the agent's LLM traffic flow
    through the proxy → captured.
    """
    factory = _load_session_factory(session_factory)
    kwargs: dict[str, Any] = {}
    if sandbox_user:
        kwargs["exec_user"] = sandbox_user
    agent_obj = factory(**kwargs)
    # Inject the kernel-resolved workspace (the same cwd ACP agents + the verifier
    # use) under BENCHFLOW_AGENT_CWD so the agent runs where the verifier reads,
    # rather than a hardcoded path.
    connect_env = dict(agent_env)
    if agent_cwd:
        connect_env["BENCHFLOW_AGENT_CWD"] = agent_cwd
    connect_sandbox = SessionFactorySandbox(env, connect_env, agent_cwd)
    connect_coro = agent_obj.connect(connect_sandbox, "agent")
    try:
        if timeout > 0:
            session = await asyncio.wait_for(connect_coro, timeout=timeout)
        else:
            session = await connect_coro
    except TimeoutError as exc:
        raise TimeoutError(
            f"session-factory connect exceeded {timeout}s budget"
        ) from exc
    logger.info("session-factory agent %r connected via %s", agent, session_factory)
    return None, session, None, agent


async def execute_prompts_session_factory(
    session: Any,
    prompts: list[str],
    timeout: int,
    idle_timeout: int | None = None,
) -> tuple[list[dict], int]:
    """Drive a session-factory Session: one ``prompt`` per turn, capture steps.

    Returns ``(trajectory, n_tool_calls)``. ``n_tool_calls`` is always ``0`` — a
    session-factory agent (e.g. omnigent's one-shot ``omnigent run -p``) exposes
    no per-tool-call stream; run validity rests on the proxy-captured token
    usage instead.

    ``timeout`` is the per-prompt wall-clock budget. ``idle_timeout`` is accepted
    for signature parity with ``execute_prompts`` but does not apply (there is no
    incremental tool-call stream to watch go idle); the agent's own subprocess
    timeout is the inner backstop. On budget exhaustion this raises
    ``AgentPromptTimeoutError`` (the kernel drive site already handles it),
    carrying the steps captured so far.
    """
    for i, prompt in enumerate(prompts):
        logger.info("Prompt %d/%d: %s...", i + 1, len(prompts), (prompt or "")[:80])
        try:
            stop_reason = await asyncio.wait_for(
                session.prompt(prompt), timeout=timeout
            )
        except TimeoutError as exc:
            # One-shot agent: on budget exhaustion there is no pending tool-call
            # stream, so the snapshot is terminal-complete with 0 tool calls.
            diagnostic = AgentPromptTimeoutDiagnostic(
                timeout_sec=float(timeout),
                n_tool_calls=0,
                terminal_trajectory_complete=True,
            )
            raise AgentPromptTimeoutError(
                f"session-factory prompt exceeded {timeout}s budget",
                trajectory=list(session.steps),
                diagnostic=diagnostic,
                executed_prompts=prompts[: i + 1],
            ) from exc
        logger.info("  → %s", stop_reason)
    return list(session.steps), 0
