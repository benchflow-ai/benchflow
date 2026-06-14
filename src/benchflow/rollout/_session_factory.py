"""Non-ACP *session-factory* CONNECT helpers for the rollout engine.

These back the additive non-ACP path: a registered agent whose
``protocol == "session-factory"`` is driven over the transport-agnostic
:class:`~benchflow.agents.protocol.Session` contract instead of ACP. The kernel
(see :mod:`benchflow.rollout`) resolves the agent's ``session_factory``
entrypoint and opens a live ``Session`` through it; ACP remains the default.

Kept in their own module (rather than inline in the 2k-line rollout engine) so
the non-ACP concern has one cohesive home.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from benchflow.agents.protocol import Session
    from benchflow.agents.registry import AgentConfig


def is_session_factory_agent(agent_cfg: AgentConfig | None) -> bool:
    """True when ``agent_cfg`` selects the non-ACP session-factory CONNECT path.

    Gated on BOTH the protocol marker and a non-empty ``session_factory`` so a
    misconfigured entry (protocol set, no entrypoint) fails loud in
    :func:`resolve_session_factory` rather than silently down the ACP path.
    Defensive ``getattr`` tolerates the bare stubs older rollout tests inject.
    """
    if agent_cfg is None:
        return False
    return getattr(agent_cfg, "protocol", "acp") == "session-factory" and bool(
        getattr(agent_cfg, "session_factory", "")
    )


def resolve_session_factory(agent_cfg: AgentConfig) -> Any:
    """Import the ``module:callable`` entrypoint named by ``agent_cfg.session_factory``.

    Returns the resolved factory (a callable that builds an object satisfying
    the Agent Protocol); ``Any`` because the entrypoint is dynamic.
    """
    spec = getattr(agent_cfg, "session_factory", "")
    if ":" not in spec:
        raise RuntimeError(
            f"session-factory agent {getattr(agent_cfg, 'name', '?')!r} has an "
            f"invalid session_factory {spec!r}; expected 'module:callable'."
        )
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


async def connect_session_factory(
    agent_cfg: AgentConfig,
    *,
    sandbox: Any,
    role: str,
    agent_env: dict[str, str] | None = None,
) -> Session:
    """Instantiate the registered Agent factory and open a live Session.

    Returns the :class:`~benchflow.agents.protocol.Session` the factory's
    ``connect(sandbox, role)`` produces — the kernel uses it as both
    ``_session`` and ``_session_adapter``.

    ``agent_env`` carries the resolved per-role agent environment — the
    ``BENCHFLOW_PROVIDER_*`` provider routing plus the agent's ``env_mapping``
    translation. A session-factory agent runs IN-PROCESS on the host, so unlike
    the ACP path (where this dict is injected into the agent subprocess's
    environment) the factory must receive it explicitly; otherwise the
    in-process agent reads the host ``os.environ`` and never sees the resolved
    gateway/model. The factory's ``connect`` is passed ``agent_env`` when it
    accepts the keyword, with a fallback for older factory signatures.
    """
    factory = resolve_session_factory(agent_cfg)
    agent = factory()
    env = dict(agent_env or {})
    try:
        return await agent.connect(sandbox, role, agent_env=env)
    except TypeError:
        # Factory predates the agent_env keyword — fall back to the bare
        # contract signature so older Agent implementations still connect.
        return await agent.connect(sandbox, role)
