"""Central resolution of a task's effective network policy.

Backends historically keyed network enforcement off the deprecated
``allow_internet`` boolean, so ``network_mode`` (the documented authority) and
``allowed_hosts`` never reached the sandbox — ``allowlist`` was validated but
unenforced. This module makes ``network_mode`` authoritative while keeping
``allow_internet`` as a derived back-compat input, and decides per backend
whether an ``allowlist`` can be enforced (compose backends, via an egress
proxy) or must fail closed (backends with only a binary block-all control).

See ``_egress.py`` for the Docker/compose allowlist mechanism and Linear
ENG-219 for the roadmap context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from benchflow.task.config import NetworkMode, SandboxConfig

#: Sandboxes that can enforce a per-host allowlist (docker-compose egress proxy).
#: Others expose only a binary block-all control, so ``allowlist`` is rejected
#: at preflight by ``runtime_capabilities`` and fails closed here as defense in
#: depth. (daytona-dind shares the compose mechanism and is a natural follow-up,
#: but the ``daytona`` preflight sandbox string can't distinguish dind from the
#: direct strategy, so it is intentionally excluded from this first cut.)
ALLOWLIST_CAPABLE_SANDBOXES: frozenset[str] = frozenset({"docker"})


def _normalize_sandbox(sandbox: str | None) -> str:
    return (sandbox or "").replace("_", "-").strip().lower()


def sandbox_supports_allowlist(sandbox: str | None) -> bool:
    """Whether *sandbox* can enforce a per-host ``allowlist``."""
    return _normalize_sandbox(sandbox) in ALLOWLIST_CAPABLE_SANDBOXES


class EffectivePolicy(StrEnum):
    """The concrete network posture a backend must apply."""

    OPEN = "open"
    BLOCK_ALL = "block-all"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True)
class NetworkDecision:
    policy: EffectivePolicy
    allowed_hosts: tuple[str, ...] = ()
    downgraded_from: NetworkMode | None = None  # set when allowlist failed closed
    note: str = ""
    model_lane: bool = (
        False  # keep a lane to the model proxy under a restrictive policy
    )


def resolve_network_mode(env_config: SandboxConfig) -> NetworkMode:
    """Return the authoritative ``NetworkMode``.

    ``network_mode`` wins. ``allow_internet`` is deprecated; an explicit
    ``False`` still tightens a ``public`` policy to ``no-network`` so legacy
    callers that mutate ``allow_internet`` after validation (e.g. the
    ``preserve_agent_network`` lift in ``sandbox/setup.py``) keep working.
    """
    mode = env_config.network_mode
    if env_config.allow_internet is False and mode is NetworkMode.PUBLIC:
        return NetworkMode.NO_NETWORK
    return mode


def resolve_network_decision(
    env_config: SandboxConfig, sandbox: str
) -> NetworkDecision:
    """Resolve the effective policy a *sandbox* should enforce for *env_config*."""
    mode = resolve_network_mode(env_config)
    lane = bool(getattr(env_config, "allow_model_endpoint", True))
    if mode is NetworkMode.NO_NETWORK:
        return NetworkDecision(EffectivePolicy.BLOCK_ALL, model_lane=lane)
    if mode is NetworkMode.ALLOWLIST:
        hosts = tuple(env_config.allowed_hosts or ())
        if sandbox_supports_allowlist(sandbox):
            return NetworkDecision(
                EffectivePolicy.ALLOWLIST, allowed_hosts=hosts, model_lane=lane
            )
        # Defense in depth: runtime_capabilities rejects allowlist at preflight on
        # these sandboxes, but if that gate is bypassed we fail CLOSED (never open).
        return NetworkDecision(
            EffectivePolicy.BLOCK_ALL,
            allowed_hosts=hosts,
            downgraded_from=NetworkMode.ALLOWLIST,
            note=(
                f"network_mode='allowlist' is not enforceable on sandbox "
                f"'{sandbox}' — failing closed to no-network"
            ),
            model_lane=lane,
        )
    return NetworkDecision(EffectivePolicy.OPEN)


def network_blocks_all(env_config: SandboxConfig, sandbox: str) -> bool:
    """Back-compat shim for the historic ``not allow_internet`` block-all gate."""
    return (
        resolve_network_decision(env_config, sandbox).policy
        is EffectivePolicy.BLOCK_ALL
    )
