"""Provider runtime boundary.

BenchFlow now owns one provider-side runtime: a LiteLLM proxy. Provider-specific
translation belongs to LiteLLM; this module keeps rollout orchestration decoupled
from the concrete host/sandbox process launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchflow.usage_tracking import UsageTrackingConfig


@dataclass
class ProviderRuntime:
    """State for a lazily-started provider gateway process."""

    kind: str
    agent_base_url: str
    backend_model: str | None = None
    server: Any | None = None
    config_key: str | None = None
    master_key: str | None = None

    @property
    def base_url(self) -> str:
        return self.agent_base_url


async def ensure_litellm_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: ProviderRuntime | None,
    environment: str,
    session_id: str = "",
    usage_tracking: UsageTrackingConfig | dict[str, Any] | str | None = None,
    sandbox: Any | None = None,
    sandbox_setup_timeout: int = 120,
) -> tuple[dict[str, str], ProviderRuntime | None]:
    from benchflow.providers.litellm_runtime import (
        ensure_litellm_runtime as _ensure_litellm_runtime,
    )

    return await _ensure_litellm_runtime(
        agent=agent,
        agent_env=agent_env,
        model=model,
        runtime=runtime,
        environment=environment,
        session_id=session_id,
        usage_tracking=usage_tracking,
        sandbox=sandbox,
        sandbox_setup_timeout=sandbox_setup_timeout,
    )


def extract_usage(runtime: ProviderRuntime | None) -> dict[str, Any]:
    from benchflow.providers.litellm_runtime import extract_usage as _extract_usage

    return _extract_usage(runtime)


async def stop_provider_runtime(runtime: ProviderRuntime | None) -> None:
    from benchflow.providers.litellm_runtime import (
        stop_litellm_runtime as _stop_litellm_runtime,
    )

    await _stop_litellm_runtime(runtime)
