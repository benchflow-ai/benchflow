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
import functools
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.agents.errors import UsageLimitError
from benchflow.agents.launch_env import agent_launch_env
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

    This is not a trust boundary. A session-factory agent is in-process host
    code: it can reach the wrapped sandbox (``.sandbox``) and every method the
    wrapper passes through, root execs included. What holds is the state:
    ``connect_session_factory`` admits ``SESSION_FACTORY`` before the agent
    connects, so the sandbox is TAINTED for every service from then on, and
    the transfer primitives named here are admitted again per call.
    """

    sandbox: Any
    agent_env: dict[str, str]
    agent_cwd: str | None = None
    # The trial's agent user: what an exec that names no user runs as.
    exec_user: str | None = None

    def __getattr__(self, name: str) -> Any:
        from benchflow.trust.state import TRANSFER_PRIMITIVES

        attr = getattr(self.sandbox, name)
        if name in TRANSFER_PRIMITIVES and callable(attr):
            # The in-process agent's transfers are its own ingress: each is
            # admitted here, and the sandbox's primitive gets the admission.
            return functools.partial(_agent_transfer, self.sandbox, name, attr)
        return attr

    # Keep the container barrier or select the native backend's account
    # before applying the exec barrier (sandbox/agent_exec.py).
    async def exec(self, command: str, *args: Any, **kwargs: Any) -> Any:
        # exec(command, cwd, env, timeout_sec, user, service)
        command, args, kwargs = self._agent_call(command, args, kwargs, 3)
        return await self.sandbox.exec(command, *args, **kwargs)

    async def exec_in_service(
        self, service: str, command: str, *args: Any, **kwargs: Any
    ) -> Any:
        # exec_in_service(service, command, cwd, env, timeout_sec, user)
        command, args, kwargs = self._agent_call(command, args, kwargs, 3)
        return await self.sandbox.exec_in_service(service, command, *args, **kwargs)

    def _agent_call(
        self, command: str, args: tuple[Any, ...], kwargs: dict[str, Any], at: int
    ) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        from benchflow.sandbox.agent_exec import agent_exec_call

        positional = len(args) > at
        user = args[at] if positional else kwargs.get("user")
        command, user = agent_exec_call(
            command,
            user,
            sandbox_user=self.exec_user,
            identity_switch=getattr(self.sandbox, "agent_identity_switch", ()) or (),
            sandbox=self.sandbox,
        )
        if positional:
            args = (*args[:at], user, *args[at + 1 :])
        elif user is not None:
            kwargs = {**kwargs, "user": user}
        return command, args, kwargs


async def _agent_transfer(
    sandbox: Any, name: str, method: Any, *args: Any, **kwargs: Any
) -> Any:
    """A session-factory agent's ``upload_file``/``upload_dir``/... on the
    sandbox, admitted as its ingress first."""
    from benchflow.trust.state import IngressKind, admit

    service = "main"
    if name == "upload_dir":
        service = kwargs.get("service", args[2] if len(args) > 2 else "main")
    kwargs.pop("admission", None)
    admission = admit(sandbox, str(service), IngressKind.SESSION_FACTORY)
    return await method(*args, admission=admission, **kwargs)


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
    from benchflow.trust.state import IngressKind, admit

    admit(env, "main", IngressKind.SESSION_FACTORY)
    factory = _load_session_factory(session_factory)
    kwargs: dict[str, Any] = {}
    if sandbox_user:
        kwargs["exec_user"] = sandbox_user
    agent_obj = factory(**kwargs)
    # Inject the kernel-resolved workspace (the same cwd ACP agents + the verifier
    # use) under BENCHFLOW_AGENT_CWD so the agent runs where the verifier reads,
    # rather than a hardcoded path.
    # Only the selected route's credentials, as for an ACP process.
    connect_env = agent_launch_env(agent, model, agent_env)
    if agent_cwd:
        connect_env["BENCHFLOW_AGENT_CWD"] = agent_cwd
    connect_sandbox = SessionFactorySandbox(env, connect_env, agent_cwd, sandbox_user)
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
        except UsageLimitError:
            # A spent login is not the agent's failure: raised as is, it is an
            # unscored usage_limit; wrapped below it would be a scored timeout.
            # The steps so far are kept by the rollout's disconnect capture.
            raise
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
        except Exception as exc:
            # Non-timeout failure (provider crash, one-shot CLI non-zero exit,
            # ValueError, …): the turn did NOT complete, so the snapshot is
            # PARTIAL. Preserve the steps captured so far by wrapping in the
            # kernel's trajectory-carrying timeout error with
            # terminal_trajectory_complete=False, so Rollout.execute()'s existing
            # AgentPromptTimeoutError handler commits the partial trajectory +
            # flag instead of the bare exception bubbling up and discarding it
            # (#825). ``Exception`` (not ``BaseException``) deliberately lets
            # asyncio ``CancelledError`` propagate untouched.
            diagnostic = AgentPromptTimeoutDiagnostic(
                timeout_sec=float(timeout),
                n_tool_calls=0,
                terminal_trajectory_complete=False,
            )
            raise AgentPromptTimeoutError(
                f"session-factory prompt failed before completion: {exc}",
                trajectory=list(session.steps),
                diagnostic=diagnostic,
                executed_prompts=prompts[: i + 1],
            ) from exc
        logger.info("  → %s", stop_reason)
    return list(session.steps), 0
