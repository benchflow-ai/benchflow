"""Protocol-branched seat driver — the one place a seat's wire protocol matters.

Every agent path (raw ACP, ai-sdk, omnigent) resolves to one
:class:`~benchflow.agents.registry.AgentConfig`; the only thing that differs at
run time is ``cfg.protocol``:

* ``acp`` / ``acpx`` — connect via :func:`connect_acp` and run prompts via
  :func:`execute_prompts` (covers raw ACP **and** ai-sdk, whose ``server.mjs`` is
  just another ``launch_cmd``). This is the verified path.
* ``session-factory`` — ``import_callable(cfg.session_factory)`` builds an
  :class:`~benchflow.agents.protocol.Agent`, then ``Agent.connect`` →
  ``Session.prompt`` (omnigent). No ``session-factory`` agent is registered in
  this repo yet, so this branch is structural / best-effort and **unverified**;
  its narrower ``connect(sandbox, role)`` surface cannot pin per-seat cwd / env /
  model the way ACP does.

The drive *mode* (``auto-loop`` vs ``service-rounds``) is orthogonal to protocol
and lives in the runner; this module only connects, prompts, and closes a seat.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.acp.runtime import connect_acp, execute_prompts
from benchflow.agents.registry import AgentConfig

__all__ = ["SeatConn", "import_callable", "connect_seat", "prompt_seat", "close_seat"]

_ACP_PROTOCOLS = {"acp", "acpx"}


def import_callable(ref: str) -> Callable[..., Any]:
    """Resolve a ``"module.path:callable"`` (or dotted ``module.path.callable``)
    reference via :mod:`importlib`. Never ``eval`` — the reference is data."""
    if ":" in ref:
        mod_name, _, attr = ref.partition(":")
    else:
        mod_name, _, attr = ref.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"not a 'module:callable' reference: {ref!r}")
    obj = getattr(importlib.import_module(mod_name), attr)
    if not callable(obj):
        raise TypeError(f"{ref!r} resolved to a non-callable {type(obj).__name__}")
    return obj


@dataclass
class SeatConn:
    """A live, protocol-agnostic seat connection.

    ``session`` carries the assignable ``on_change`` hook the runner wires to a
    trajectory sink. ``client`` is the ACP client (acp only; used to close).
    """

    protocol: str
    session: Any
    client: Any = None
    name: str = ""


async def connect_seat(
    cfg: AgentConfig,
    *,
    env: Any,
    agent_cwd: str,
    agent_env: dict[str, str],
    model: str | None,
    rollout_dir: Path,
    environment: str,
    seat_id: str,
    reasoning_effort: str | None = None,
) -> SeatConn:
    """Connect ``cfg`` inside ``env`` (a shared sandbox) at ``agent_cwd``."""
    if cfg.protocol in _ACP_PROTOCOLS:
        client, session, _adapter, name = await connect_acp(
            env=env,
            agent=cfg.name,
            agent_launch=cfg.launch_cmd,
            agent_env=agent_env,
            sandbox_user=None,
            model=model,
            rollout_dir=rollout_dir,
            environment=environment,
            agent_cwd=agent_cwd,
            reasoning_effort=reasoning_effort,
        )
        return SeatConn(protocol="acp", session=session, client=client, name=name)

    if cfg.protocol == "session-factory":
        if not cfg.session_factory:
            raise ValueError(
                f"{cfg.name}: protocol=session-factory requires `session_factory`"
            )
        agent = import_callable(cfg.session_factory)()
        session = await agent.connect(env, seat_id)
        return SeatConn(protocol="session-factory", session=session, name=cfg.name)

    raise ValueError(f"{cfg.name}: unsupported protocol {cfg.protocol!r}")


async def prompt_seat(
    conn: SeatConn, prompt: str, *, timeout: int, idle_timeout: int | None = None
) -> tuple[list[dict], int]:
    """Run one prompt on ``conn``; return ``(trajectory, n_tool_calls)``.

    Re-entrant: call again to drive another round on the same live session (the
    ``service-rounds`` loop does exactly this)."""
    if conn.protocol in _ACP_PROTOCOLS:
        return await execute_prompts(
            acp_client=conn.client,
            session=conn.session,
            prompts=[prompt],
            timeout=timeout,
            idle_timeout=idle_timeout,
        )
    # session-factory: Session.prompt returns a StopReason; steps live on the
    # session. No ACP idle watchdog at this layer — bound by the wall clock only.
    await asyncio.wait_for(conn.session.prompt(prompt), timeout)
    steps = list(getattr(conn.session, "steps", []) or [])
    n_tools = sum(
        1 for s in steps if isinstance(s, dict) and s.get("type") == "tool_call"
    )
    return steps, n_tools


async def close_seat(conn: SeatConn) -> None:
    """Best-effort close of the underlying transport."""
    if conn.client is not None:
        with contextlib.suppress(Exception):
            await conn.client.close()
