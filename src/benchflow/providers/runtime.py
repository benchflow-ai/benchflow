"""Provider runtime helpers for host-side proxy processes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from benchflow.agents.providers import find_provider, strip_provider_prefix
from benchflow.agents.registry import AGENTS
from benchflow.providers.bedrock_proxy import BedrockProxyServer

logger = logging.getLogger(__name__)

BEDROCK_PROXY_BIND_HOST = "0.0.0.0"
BEDROCK_PROXY_LOCAL_HOST = "127.0.0.1"


@dataclass
class ProviderRuntime:
    """State for a lazily-started provider-side helper process."""

    kind: str
    host: str
    port: int
    backend_model: str | None = None
    frontend_model: str | None = None
    server: Any | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def needs_provider_runtime(model: str | None) -> bool:
    """True when the model must be routed through a local helper runtime."""
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


def _bedrock_proxy_command(
    *,
    environment: str,
) -> str:
    if environment == "docker":
        return _docker_host_address()
    return BEDROCK_PROXY_LOCAL_HOST


async def ensure_bedrock_proxy_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: ProviderRuntime | None,
    environment: str,
) -> tuple[dict[str, str], ProviderRuntime | None]:
    """Start the host-side Bedrock proxy if needed and wire env vars to it."""
    if not needs_provider_runtime(model):
        return agent_env, runtime
    assert model is not None

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
            host=_bedrock_proxy_command(environment=environment),
            port=server.port,
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
