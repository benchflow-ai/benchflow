"""Daytona allowlist lockdown orchestration."""

from __future__ import annotations

import shlex
from typing import Any

from benchflow.sandbox import network_policy
from benchflow.sandbox.protocol import SandboxStartupError

_EGRESS_CANARY_PORT = 443
_CANARY_CANDIDATES = ("1.1.1.1", "8.8.8.8", "9.9.9.9")


def pick_daytona_canary(cidrs: tuple[str, ...]) -> str:
    """First canary IP whose /32 is not already in the allow list."""
    for host in _CANARY_CANDIDATES:
        if f"{host}/32" not in cidrs:
            return host
    return _CANARY_CANDIDATES[0]


async def relock_daytona_network(
    sandbox_wrapper: Any,
    *,
    extra_allowed_hosts: tuple[str, ...] = (),
) -> dict[str, str]:
    """Apply the task's allowlist as a Daytona IPv4 CIDR list."""
    decision = network_policy.resolve_network_decision(
        sandbox_wrapper.task_env_config, "daytona"
    )
    if decision.policy is not network_policy.EffectivePolicy.ALLOWLIST:
        return {}
    if sandbox_wrapper._compose_mode:
        # DinD: update_network_settings governs the OUTER sandbox, but the
        # agent runs in inner containers whose egress is ungoverned. Fail closed.
        raise SandboxStartupError(
            "daytona compose/DinD does not support network_mode='allowlist' "
            "enforcement (settings apply to the outer sandbox only); use the "
            "'docker' sandbox or 'no-network'"
        )

    model_host = (
        extra_allowed_hosts[0] if decision.model_lane and extra_allowed_hosts else None
    )
    plan = network_policy.plan_daytona_allowlist(
        decision.allowed_hosts, model_host=model_host
    )
    if not plan.enforceable:
        raise SandboxStartupError(
            f"daytona cannot enforce network_mode='allowlist': {plan.reject_reason}"
        )

    # Pin allowlisted hosts in /etc/hosts so the agent resolves them without DNS
    # egress and without IP-rotation drift. TLS SNI/cert still use the hostname.
    if plan.host_ips:
        lines = "".join(f"{ip}\t{host}\n" for host, ip in plan.host_ips)
        await sandbox_wrapper.exec(
            f"printf %s {shlex.quote(lines)} >> /etc/hosts",
            user="root",
            timeout_sec=20,
        )

    sandbox = sandbox_wrapper._require_sandbox()
    await sandbox.update_network_settings(network_allow_list=",".join(plan.cidrs))

    canary = pick_daytona_canary(plan.cidrs)
    if network_policy.blockall_enforcement_violation(
        block_all=True,
        canary_reachable=await sandbox_wrapper._egress_reachable(canary),
    ):
        raise SandboxStartupError(
            f"daytona applied a {len(plan.cidrs)}-CIDR allow list but the "
            f"sandbox could not confirm {canary}:{_EGRESS_CANARY_PORT} is "
            "blocked (a non-allowlisted host) — the platform did not enforce "
            "the allow list, or the probe could not run; failing closed"
        )

    sandbox_wrapper.logger.info(
        "relock_network: ALLOWLIST applied (daytona, %d cidrs)",
        len(plan.cidrs),
    )
    return {}
