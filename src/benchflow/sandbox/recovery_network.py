"""Restore declared network policy without starting an agent or inference runtime."""

from __future__ import annotations

import shlex
from typing import Any

from benchflow.sandbox.egress_denylist import (
    PROXY_URL,
    denylist_agent_env,
    egress_denylist_for,
    start_egress_denylist,
)
from benchflow.sandbox.lockdown import (
    enforce_agent_egress_firewall,
    enforce_sandbox_uid_egress,
)


async def prepare_recovery_network(
    env: Any, sandbox_config: Any, sandbox_user: str | None
) -> None:
    """Recreate the original sandbox UID policy; root is not restricted.

    Call after user setup and before verification. Container network mode is
    reproduced separately by the provider. No solver environment or credentials
    are copied, and no model gateway is created. Any setup/probe failure aborts
    recovery rather than permitting verification with omitted policy (GH1136).
    """
    policy = egress_denylist_for(sandbox_config)
    no_web = getattr(sandbox_config, "allow_internet", True) is False
    if policy is None and not no_web:
        return
    if not sandbox_user:
        if policy is not None:
            raise RuntimeError("Denylist recovery requires the original sandbox user")
        # Original no-web enforcement also has no UID firewall without a user.
        return
    identity = await env.exec(
        f"id -u {shlex.quote(sandbox_user)}", user="root", timeout_sec=30
    )
    uid = (identity.stdout or "").strip()
    if identity.return_code != 0 or not uid.isdecimal() or int(uid) == 0:
        raise RuntimeError("Denylist recovery requires a resolved non-root sandbox UID")
    if policy is None:
        # Original model bootstrap may retain network access for its gateway;
        # reproduce its UID restriction even though recovery has no gateway.
        await enforce_sandbox_uid_egress(env, sandbox_user)
        return
    await start_egress_denylist(env, sandbox_user, policy, model_gateway_url=None)
    await enforce_agent_egress_firewall(
        env, sandbox_user, denylist_agent_env({}, policy)
    )
    # Use the policy principal, not root's readiness probe, to establish that
    # the recreated user can reach the local proxy after firewall installation.
    probe = (
        "import urllib.request; "
        "o=urllib.request.build_opener(urllib.request.ProxyHandler({})); "
        f"assert o.open({PROXY_URL + '/healthz'!r}, timeout=5).status == 200"
    )
    result = await env.exec(
        shlex.join(["python3", "-I", "-c", probe]),
        user=sandbox_user,
        timeout_sec=15,
    )
    if result.return_code != 0:
        raise RuntimeError("Recovery denylist proxy is unavailable to the sandbox UID")
