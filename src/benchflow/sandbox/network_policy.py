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

import socket
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from benchflow.task.config import NetworkMode, SandboxConfig

#: Sandboxes that can enforce a per-host allowlist (docker-compose egress proxy).
#: Others expose only a binary block-all control, so ``allowlist`` is rejected
#: at preflight by ``runtime_capabilities`` and fails closed here as defense in
#: depth. (daytona-dind shares the compose mechanism and is a natural follow-up,
#: but the ``daytona`` preflight sandbox string can't distinguish dind from the
#: direct strategy, so it is intentionally excluded from this first cut.)
ALLOWLIST_CAPABLE_SANDBOXES: frozenset[str] = frozenset({"docker", "daytona"})

#: Sandboxes whose allowlist is enforced by IPv4 CIDR (not hostname/SNI). They
#: need hostname->IP resolution at lockdown and cannot express wildcards.
IP_BASED_ALLOWLIST_SANDBOXES: frozenset[str] = frozenset({"daytona"})


def _normalize_sandbox(sandbox: str | None) -> str:
    return (sandbox or "").replace("_", "-").strip().lower()


def allowlist_is_ip_based(sandbox: str | None) -> bool:
    """True if the sandbox enforces allowlists by IPv4 CIDR (daytona)."""
    return _normalize_sandbox(sandbox) in IP_BASED_ALLOWLIST_SANDBOXES


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


def blockall_enforcement_violation(*, block_all: bool, canary_reachable: bool) -> bool:
    """Fail-closed check for a block-all policy.

    A sandbox that resolves to ``BLOCK_ALL`` must have no off-box route. If an
    external canary is still reachable from inside, the platform did not honor the
    block (e.g. daytona's ``network_block_all`` flag was ignored) — the run should
    abort rather than produce a falsely-rewarded "offline" result.
    """
    return block_all and canary_reachable


def network_is_restrictive(env_config: SandboxConfig, sandbox: str) -> bool:
    """True iff the resolved policy is anything other than fully-open egress."""
    return (
        resolve_network_decision(env_config, sandbox).policy is not EffectivePolicy.OPEN
    )


def proxy_unavailable_is_fatal(*, usage_mode: str, network_restrictive: bool) -> bool:
    """Whether an unavailable LiteLLM usage proxy must abort the run instead of
    silently falling back to the direct provider.

    Fatal when usage tracking is ``required``, or when the network policy is
    restrictive (the direct provider would be blocked by the egress allowlist, so
    skipping the proxy leaves the model unreachable). ``off`` is an explicit
    opt-out and is never forced fatal here.
    """
    return usage_mode == "required" or (network_restrictive and usage_mode != "off")


def lockdown_complete(
    attached_networks: set[str], default_net: str, internal_net: str | None
) -> bool:
    """True iff a docker relock actually took effect.

    After install-before-lockdown swaps the container's networks, it must be
    detached from the public bridge (*default_net*) and, when an egress sidecar
    is in use, attached to *internal_net*. If a ``network connect``/``disconnect``
    silently failed the container could sit on BOTH nets (egress proxy bypassed)
    or on NONE (stranded) — callers fail closed when this returns ``False``
    rather than running with an unenforced policy.
    """
    on_public_bridge = default_net in attached_networks
    missing_internal = (
        internal_net is not None and internal_net not in attached_networks
    )
    return not (on_public_bridge or missing_internal)


# --- Daytona allowlist parity (enforce-when-faithful) -----------------------
#
# Daytona has a native allowlist (`network_allow_list`) but it is **IPv4-CIDR**
# based (max 10 entries, no hostnames/wildcards), whereas the docker `bf-egress`
# proxy matches on hostname/SNI. To make daytona enforce an allowlist (instead
# of failing closed at preflight) we resolve the hostname allowlist to /32 CIDRs
# at lockdown. We only do this when the policy is *faithfully* expressible as an
# IPv4 list; otherwise we fail closed with a precise reason (the policy a user
# wrote can't be honored on this sandbox, so don't silently weaken it).

#: Daytona caps `network_allow_list` at 10 IPv4 CIDR entries.
DAYTONA_MAX_ALLOWLIST_CIDRS = 10


@dataclass(frozen=True)
class DaytonaAllowlistPlan:
    """Outcome of mapping a hostname allowlist onto daytona's IPv4-CIDR control.

    Exactly one state holds: ``cidrs`` non-empty (enforce these), or
    ``reject_reason`` set (fail closed — the allowlist can't be faithfully
    represented as an IPv4 list on daytona).
    """

    cidrs: tuple[str, ...] = ()
    reject_reason: str | None = None
    #: (hostname, primary IPv4) pairs to pin in the sandbox's /etc/hosts so the
    #: agent resolves allowlisted hosts WITHOUT DNS egress (the resolvers are not
    #: allowlisted) and without IP-rotation drift (pinned to the allowlisted IP).
    host_ips: tuple[tuple[str, str], ...] = ()

    @property
    def enforceable(self) -> bool:
        return self.reject_reason is None


def resolve_ipv4(host: str) -> tuple[str, ...]:
    """Resolve *host* to its IPv4 addresses (empty tuple if none / on failure)."""
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
    except OSError:
        return ()
    seen: list[str] = []
    for info in infos:
        ip = str(info[4][0])
        if ip not in seen:
            seen.append(ip)
    return tuple(seen)


def plan_daytona_allowlist(
    allowed_hosts: tuple[str, ...],
    *,
    model_host: str | None,
    resolve: Callable[[str], tuple[str, ...]] = resolve_ipv4,
) -> DaytonaAllowlistPlan:
    """Map a hostname allowlist (+ the model host) onto daytona IPv4 CIDRs.

    Enforce-when-faithful: returns CIDRs when the policy is expressible as a
    <=10 IPv4 list with every host resolving; otherwise returns a precise
    ``reject_reason`` so the caller fails closed (never silently downgrades a
    policy the user explicitly requested).
    """
    wild = [h for h in allowed_hosts if h.startswith("*.")]
    if wild:
        return DaytonaAllowlistPlan(
            reject_reason=(
                "daytona's allowlist is IPv4-CIDR based and cannot express "
                f"wildcard host(s) {sorted(wild)}; use the 'docker' sandbox for "
                "wildcard allowlists or list exact hostnames"
            )
        )

    hosts = list(allowed_hosts)
    if model_host:
        hosts.append(model_host)

    cidrs: list[str] = []
    host_ips: list[tuple[str, str]] = []
    seen: set[str] = set()
    unresolved: list[str] = []
    for host in hosts:
        ips = resolve(host)
        if not ips:
            if host not in unresolved:
                unresolved.append(host)
            continue
        # pin host -> first resolved IP (which we also allowlist below)
        host_ips.append((host, ips[0]))
        for ip in ips:
            cidr = f"{ip}/32"
            if cidr not in seen:
                seen.add(cidr)
                cidrs.append(cidr)

    if unresolved:
        return DaytonaAllowlistPlan(
            reject_reason=(
                "could not resolve an IPv4 address for allowlist host(s) "
                f"{sorted(unresolved)} (required to build the daytona network "
                "allow list); failing closed"
            )
        )
    if not cidrs:
        return DaytonaAllowlistPlan(
            reject_reason="allowlist resolved to zero hosts; failing closed"
        )
    if len(cidrs) > DAYTONA_MAX_ALLOWLIST_CIDRS:
        return DaytonaAllowlistPlan(
            reject_reason=(
                f"allowlist resolves to {len(cidrs)} IPv4 addresses, exceeding "
                f"daytona's {DAYTONA_MAX_ALLOWLIST_CIDRS}-CIDR limit; reduce the "
                "host list or use the 'docker' sandbox"
            )
        )
    return DaytonaAllowlistPlan(cidrs=tuple(cidrs), host_ips=tuple(host_ips))
