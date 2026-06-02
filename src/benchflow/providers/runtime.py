"""Provider runtime helpers for host-side proxy processes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from benchflow.agents.providers import find_provider, strip_provider_prefix
from benchflow.agents.registry import AGENTS
from benchflow.providers.bedrock_proxy import BedrockProxyServer
from benchflow.providers.bedrock_runtime import BEDROCK_THINKING_EFFORT_ENV
from benchflow.usage_tracking import UsageTrackingConfig

logger = logging.getLogger(__name__)

BEDROCK_PROXY_BIND_HOST = "0.0.0.0"
BEDROCK_PROXY_LOCAL_HOST = "127.0.0.1"


@dataclass
class ProviderRuntime:
    """State for a lazily-started provider-side helper process."""

    kind: str
    agent_base_url: str
    backend_model: str | None = None
    frontend_model: str | None = None
    server: Any | None = None

    @property
    def base_url(self) -> str:
        return self.agent_base_url


def needs_provider_runtime(model: str | None) -> bool:
    """True when the model needs Bedrock-specific runtime handling."""
    if not model:
        return False
    result = find_provider(model)
    return result is not None and result[0] == "aws-bedrock"


def _apply_agent_provider_mapping(
    agent_env: dict[str, str],
    *,
    agent: str,
    base_url: str,
    backend_model: str,
) -> dict[str, str]:
    updated = dict(agent_env)
    updated["BENCHFLOW_PROVIDER_BASE_URL"] = base_url
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg:
        return updated
    if agent == "claude-agent-acp":
        updated.pop("ANTHROPIC_BASE_URL", None)
        updated.pop("ANTHROPIC_AUTH_TOKEN", None)
        updated["CLAUDE_CODE_USE_BEDROCK"] = "1"
        updated["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] = "1"
        updated["ANTHROPIC_BEDROCK_BASE_URL"] = base_url
        updated["ANTHROPIC_MODEL"] = backend_model
        return updated

    mapped_base_url = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_BASE_URL")
    if mapped_base_url:
        updated[mapped_base_url] = base_url
    mapped_model = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_MODEL")
    if mapped_model:
        updated[mapped_model] = _bedrock_frontend_model(
            agent=agent,
            backend_model=backend_model,
        )
    return updated


def _bedrock_frontend_model(*, agent: str, backend_model: str) -> str:
    """Return the model name the upstream agent should see."""
    return backend_model


def _direct_bedrock_frontend_model(*, agent: str, backend_model: str) -> str:
    """Return the Bedrock-native model name for agents that can call AWS directly."""
    if agent == "openhands":
        return f"bedrock/{backend_model}"
    raise ValueError(f"Agent {agent!r} does not support direct Bedrock routing")


def _agent_supports_direct_bedrock(agent: str) -> bool:
    return agent == "openhands"


def _apply_direct_bedrock_agent_mapping(
    agent_env: dict[str, str],
    *,
    agent: str,
    backend_model: str,
    environment: str,
) -> dict[str, str]:
    """Wire agent env vars for direct AWS Bedrock access without a host proxy."""
    if not _agent_supports_direct_bedrock(agent):
        raise RuntimeError(
            f"Bedrock-routed models are not supported on the "
            f"'{environment}' sandbox for agent {agent!r}: the host-side "
            "Bedrock proxy is unreachable from that remote sandbox, and this "
            "agent does not support direct Bedrock routing. "
            "Use an agent with direct Bedrock support, run with '--sandbox docker', "
            "or select a non-Bedrock model."
        )

    updated = dict(agent_env)
    updated["BENCHFLOW_PROVIDER_MODEL"] = backend_model
    updated.pop("BENCHFLOW_PROVIDER_BASE_URL", None)
    for env_name in _agent_base_url_envs(agent):
        updated.pop(env_name, None)

    if agent == "openhands":
        updated["LLM_MODEL"] = _direct_bedrock_frontend_model(
            agent=agent,
            backend_model=backend_model,
        )
        if updated.get("AWS_REGION") and not updated.get("AWS_REGION_NAME"):
            updated["AWS_REGION_NAME"] = updated["AWS_REGION"]
        if updated.get("AWS_BEARER_TOKEN_BEDROCK"):
            updated["LLM_API_KEY"] = updated["AWS_BEARER_TOKEN_BEDROCK"]
        # Propagate the explicit Claude 4.8+ thinking-effort override (e.g. MAX
        # mode) into the remote sandbox so the Daytona litellm shim honors it. On
        # Docker the host proxy reads it directly, but the sandbox has its own
        # environment, so forward it here when set on the host and not already
        # provided by the run config.
        effort_override = os.environ.get(BEDROCK_THINKING_EFFORT_ENV)
        if effort_override and BEDROCK_THINKING_EFFORT_ENV not in updated:
            updated[BEDROCK_THINKING_EFFORT_ENV] = effort_override
    return updated


def _docker_host_address() -> str:
    """Return the address containers should use to reach the host.

    On Docker Desktop (macOS/Windows) ``host.docker.internal`` is defined
    automatically.  On Linux it is not, so we query the Docker bridge
    gateway which routes to the host.
    """
    import subprocess
    import sys

    if sys.platform != "linux":
        return "host.docker.internal"
    try:
        out = subprocess.check_output(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ],
            text=True,
            timeout=10,
        ).strip()
        if out:
            return out
    except Exception:
        logger.debug("Could not detect Docker bridge gateway, falling back")
    return "host.docker.internal"


# Remote cloud sandbox environments where the agent runs on a *different*
# machine than the host proxy. The canonical environment set produced by the
# runtime is {docker, daytona, modal}; these are the ones a host-bound proxy
# cannot be reached from. Any other (unknown) value is treated conservatively
# as reachable — see ``host_proxy_reachable_from_agent``.
_REMOTE_UNREACHABLE_ENVIRONMENTS = {"daytona", "modal"}


def host_proxy_reachable_from_agent(environment: str) -> bool:
    """True when a host-side proxy bound to the host can be reached by the agent.

    The host telemetry/Bedrock proxy binds to the *host* machine. An agent
    only reaches it when it shares the host's network namespace:

    - ``docker``: the container reaches the host via the docker bridge /
      ``host.docker.internal``.

    Remote cloud sandboxes (``daytona``, ``modal``) run the agent on a
    different machine. ``127.0.0.1`` there is the *sandbox's* own loopback,
    and the Daytona SSH gateway rejects ``ssh -R`` reverse tunnels, so there
    is no address that routes back to the host proxy.

    An unrecognized environment is treated as reachable (conservative: assume
    same-host so the proxy is still wired up rather than silently skipped).
    """
    return environment not in _REMOTE_UNREACHABLE_ENVIRONMENTS


def _bedrock_proxy_command(
    *,
    environment: str,
) -> str:
    """Return the address the agent uses to reach a host-bound proxy.

    Precondition: ``host_proxy_reachable_from_agent(environment)`` is True —
    this is only ever reached for environments that share the host's network
    namespace. The reachability predicate above is the single gate that
    decides whether a host proxy is usable at all.
    """
    assert host_proxy_reachable_from_agent(environment), (
        f"_bedrock_proxy_command called for unreachable environment "
        f"{environment!r}; host_proxy_reachable_from_agent must gate this"
    )
    if environment == "docker":
        return _docker_host_address()
    return BEDROCK_PROXY_LOCAL_HOST


def _agent_base_url_envs(agent: str) -> list[str]:
    envs: list[str] = []
    agent_cfg = AGENTS.get(agent)
    if agent_cfg:
        mapped = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_BASE_URL")
        if mapped:
            envs.append(mapped)
    envs.extend(
        [
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_BEDROCK_BASE_URL",
            "OPENAI_BASE_URL",
            "GOOGLE_GEMINI_BASE_URL",
            "GEMINI_API_BASE_URL",
            "LLM_BASE_URL",
        ]
    )
    seen: set[str] = set()
    return [e for e in envs if not (e in seen or seen.add(e))]


def validate_usage_proxy_preconditions(
    usage_cfg: UsageTrackingConfig,
    *,
    environment: str,
    model: str | None,
    disable_usage_proxy: bool | None = None,
) -> Any:
    """Return the first reason usage telemetry cannot be wired, if any."""
    from benchflow.providers.usage_proxy_runtime import (
        validate_usage_proxy_preconditions as _validate_usage_proxy_preconditions,
    )

    return _validate_usage_proxy_preconditions(
        usage_cfg,
        environment=environment,
        model=model,
        disable_usage_proxy=disable_usage_proxy,
    )


async def ensure_usage_proxy_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: ProviderRuntime | None,
    environment: str,
    session_id: str = "",
    usage_tracking: UsageTrackingConfig | dict[str, Any] | str | None = None,
    sandbox: Any | None = None,
) -> tuple[dict[str, str], ProviderRuntime | None]:
    """Start a reachable usage proxy and wire agent provider env vars to it."""
    from benchflow.providers.usage_proxy_runtime import (
        ensure_usage_proxy_runtime as _ensure_usage_proxy_runtime,
    )

    return await _ensure_usage_proxy_runtime(
        agent=agent,
        agent_env=agent_env,
        model=model,
        runtime=runtime,
        environment=environment,
        session_id=session_id,
        usage_tracking=usage_tracking,
        sandbox=sandbox,
    )


def extract_usage(runtime: ProviderRuntime | None) -> dict[str, Any]:
    """Extract aggregate token/cost metrics from a usage proxy runtime."""
    from benchflow.providers.usage_proxy_runtime import extract_usage as _extract_usage

    return _extract_usage(runtime)


async def ensure_bedrock_proxy_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: ProviderRuntime | None,
    environment: str,
) -> tuple[dict[str, str], ProviderRuntime | None]:
    """Wire Bedrock-routed models through the best reachable path.

    Same-host agents use BenchFlow's host-side Bedrock proxy, which translates
    OpenAI/Anthropic-shaped requests into Bedrock Converse. Remote sandboxes
    cannot reach that host-bound proxy, so agents with native Bedrock support
    are configured to call AWS directly instead.
    """
    if not needs_provider_runtime(model):
        return agent_env, runtime
    assert model is not None

    if not host_proxy_reachable_from_agent(environment):
        if runtime is not None:
            await stop_provider_runtime(runtime)
        logger.info(
            "Skipping host-side Bedrock proxy: the '%s' sandbox runs the "
            "agent on a remote host unreachable from the host proxy; wiring "
            "%s for direct AWS Bedrock access.",
            environment or "unknown",
            agent,
        )
        return (
            _apply_direct_bedrock_agent_mapping(
                agent_env,
                agent=agent,
                backend_model=strip_provider_prefix(model),
                environment=environment,
            ),
            None,
        )

    if runtime is None:
        backend_model = strip_provider_prefix(model)
        frontend_model = _bedrock_frontend_model(
            agent=agent,
            backend_model=backend_model,
        )
        logger.info("Starting host-side Bedrock proxy")
        server = BedrockProxyServer(
            host=BEDROCK_PROXY_BIND_HOST,
            port=0,
            backend_model=backend_model,
            frontend_model=frontend_model,
            runtime_env=agent_env,
        )
        await server.start()
        runtime = ProviderRuntime(
            kind="aws-bedrock",
            agent_base_url=(
                f"http://{_bedrock_proxy_command(environment=environment)}:{server.port}"
            ),
            backend_model=backend_model,
            frontend_model=frontend_model,
            server=server,
        )

    return _apply_agent_provider_mapping(
        agent_env,
        agent=agent,
        base_url=runtime.base_url,
        backend_model=runtime.backend_model or strip_provider_prefix(model),
    ), runtime


async def stop_provider_runtime(runtime: ProviderRuntime | None) -> None:
    """Stop a previously-started provider runtime."""
    if runtime is None:
        return
    if runtime.kind == "aws-bedrock" and runtime.server is not None:
        await runtime.server.stop()
    if runtime.kind == "usage-proxy" and runtime.server is not None:
        await runtime.server.stop()
