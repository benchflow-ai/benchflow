"""Docker network-policy lockdown orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchflow.sandbox._egress import (
    _EGRESS_INTERNAL_NET,
    _EGRESS_PORT,
    _EGRESS_SERVICE,
    build_egress_override,
)
from benchflow.sandbox.network_policy import (
    EffectivePolicy,
    lockdown_complete,
    resolve_network_decision,
)
from benchflow.sandbox.protocol import SandboxStartupError


def docker_network_policy_compose_paths(sandbox: Any) -> list[Path]:
    """Return compose overrides for the sandbox's active network policy."""
    if not sandbox._network_locked:
        # Stay open during the install phase; relock_network() applies the
        # restrictive policy once the agent has been installed.
        return []
    decision = resolve_network_decision(sandbox.task_env_config, "docker")
    if decision.policy is EffectivePolicy.OPEN:
        return []

    lane = None
    if decision.model_lane:
        from benchflow.providers.litellm_runtime import _docker_host_address

        lane = _docker_host_address()

    # An allowlist, or a no-network run that keeps only the model lane open, is
    # enforced by the egress sidecar; both need a writable rollout dir to stage
    # the proxy compose override.
    if sandbox.rollout_paths and (decision.policy is EffectivePolicy.ALLOWLIST or lane):
        extra_hosts = sandbox._extra_allowed_hosts if decision.model_lane else ()
        hosts = tuple(
            decision.allowed_hosts
            if decision.policy is EffectivePolicy.ALLOWLIST
            else ()
        ) + tuple(extra_hosts)
        return [
            build_egress_override(
                hosts,
                out_dir=sandbox.rollout_paths.rollout_dir,
                model_lane=lane,
            )
        ]

    # BLOCK_ALL with no lane, or nowhere to stage the proxy: fail closed.
    return [sandbox._DOCKER_COMPOSE_NO_NETWORK_PATH]


async def relock_docker_network(
    sandbox: Any,
    *,
    compose_project_name: str,
    extra_allowed_hosts: tuple[str, ...] = (),
) -> dict[str, str]:
    """Apply a restrictive Docker network policy after agent installation."""
    decision = resolve_network_decision(sandbox.task_env_config, "docker")
    if decision.policy is EffectivePolicy.OPEN:
        return {}

    # Gate docker_network_policy_compose_paths() to emit the real override now.
    sandbox._network_locked = True
    sandbox._extra_allowed_hosts = (
        tuple(extra_allowed_hosts) if decision.model_lane else ()
    )
    cid = await sandbox._main_container_id()
    if not cid:
        sandbox.logger.warning("relock_network: no 'main' container; skipping")
        return {}

    paths = sandbox._network_policy_compose_paths()
    use_sidecar = bool(paths and paths[0] != sandbox._DOCKER_COMPOSE_NO_NETWORK_PATH)

    if use_sidecar:
        # Bring up ONLY the egress sidecar (creates the bf_egress_* networks);
        # --no-deps leaves the already-running 'main' container in place.
        await sandbox._run_docker_compose_command(
            ["up", "--detach", "--no-deps", _EGRESS_SERVICE]
        )
        await sandbox._docker_cli(
            [
                "network",
                "connect",
                f"{compose_project_name}_{_EGRESS_INTERNAL_NET}",
                cid,
            ],
            check=False,
        )

    await sandbox._docker_cli(
        ["network", "disconnect", f"{compose_project_name}_default", cid],
        check=False,
    )

    inspect_res = await sandbox._docker_cli(
        [
            "inspect",
            cid,
            "--format",
            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
        ],
        check=False,
    )
    if inspect_res.return_code != 0:
        raise SandboxStartupError(
            "relock_network: could not inspect container networks "
            f"(docker inspect rc={inspect_res.return_code}); failing closed "
            "rather than running with an unverified network policy"
        )

    attached: set[str] = set((inspect_res.stdout or "").split())
    internal_net = (
        f"{compose_project_name}_{_EGRESS_INTERNAL_NET}" if use_sidecar else None
    )
    if not lockdown_complete(attached, f"{compose_project_name}_default", internal_net):
        raise SandboxStartupError(
            f"relock_network: {decision.policy.name} lockdown did not take "
            f"effect (container networks={sorted(attached)}); refusing to run "
            "with an unenforced network policy"
        )

    sandbox.logger.info(
        "relock_network: %s applied (sidecar=%s)", decision.policy.name, use_sidecar
    )
    if not use_sidecar:
        return {}

    proxy = f"http://{_EGRESS_SERVICE}:{_EGRESS_PORT}"
    return {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }
