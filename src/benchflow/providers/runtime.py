"""Provider runtime boundary.

BenchFlow now owns one provider-side runtime: a LiteLLM proxy. Provider-specific
translation belongs to LiteLLM; this module keeps rollout orchestration decoupled
from the concrete host/sandbox process launcher.
"""

from __future__ import annotations

import contextlib
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.usage_tracking import UsageTrackingConfig, usage_unavailable

if TYPE_CHECKING:
    from benchflow.providers.litellm_runtime import LiteLLMProcess
    from benchflow.trial_budget import TrialBudget


@dataclass
class ProviderRuntime:
    """State for a lazily-started provider gateway process."""

    kind: str
    agent_base_url: str
    backend_model: str | None = None
    # Typed (not Any) so cross-module reads of the server's accessors —
    # e.g. the dashboard's live_usage_tokens dig in rollout — are bound by
    # the type checker: a rename on LiteLLMProcess breaks ty, not silently
    # blanks a signal behind a getattr default.
    server: LiteLLMProcess | None = None
    config_key: str | None = None
    master_key: str | None = None
    source_agent: str | None = None
    source_model: str | None = None
    # Controller-only upstream config for re-entrant connects and proxy restarts.
    source_env: dict[str, str] | None = field(default=None, repr=False)
    # The current role's traffic goes through this proxy (native usage then
    # has provider coverage). Retired runtimes live in rollout's ledger.
    proxy_active: bool = True
    # stop imports the callback log and removes it. A failed replacement can
    # leave this runtime in rollout's cleanup slot; retain that final import.
    stopped: bool = False
    # How a web-restricted run's model was admitted (restricted_models).
    restricted_web_model: dict[str, str | None] | None = None

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
    required_skill_names: tuple[str, ...] = (),
    live_trajectory_path: Path | None = None,
    prior_runtimes: tuple[ProviderRuntime, ...] = (),
    force_sandbox_local: bool = False,
    install_layer_bypass_reason: str | None = None,
    disallow_web_tools: bool = False,
    trial_budget: TrialBudget | None = None,
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
        required_skill_names=required_skill_names,
        live_trajectory_path=live_trajectory_path,
        prior_runtimes=prior_runtimes,
        force_sandbox_local=force_sandbox_local,
        install_layer_bypass_reason=install_layer_bypass_reason,
        disallow_web_tools=disallow_web_tools,
        trial_budget=trial_budget,
    )


def extract_usage(runtime: ProviderRuntime | None) -> dict[str, Any]:
    if runtime is None:
        # What the LiteLLM runtime answers for no runtime, without importing
        # it (and httpx with it) in a rollout that started no model proxy.
        return usage_unavailable()
    from benchflow.providers.litellm_runtime import extract_usage as _extract_usage

    return _extract_usage(runtime)


async def stop_provider_runtime(runtime: ProviderRuntime | None) -> None:
    if runtime is None:
        return
    from benchflow.providers.litellm_runtime import (
        stop_litellm_runtime as _stop_litellm_runtime,
    )

    await _stop_litellm_runtime(runtime)


def reserve_host_proxy(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    environment: str,
    required_skill_names: tuple[str, ...] = (),
    force_sandbox_local: bool = False,
    trial_budget: TrialBudget | None = None,
) -> Any:
    """Start the host model proxy a trial's agent will connect through while
    its sandbox starts (``litellm_runtime.reserve_host_proxy``); the
    reservation, or None."""
    from benchflow.providers.litellm_runtime import (
        reserve_host_proxy as _reserve_host_proxy,
    )

    return _reserve_host_proxy(
        agent=agent,
        agent_env=agent_env,
        model=model,
        environment=environment,
        required_skill_names=required_skill_names,
        force_sandbox_local=force_sandbox_local,
        trial_budget=trial_budget,
    )


def host_proxy_claim(reservation: Any) -> AbstractContextManager[None]:
    """Around ``ensure_litellm_runtime``: it may take the reserved proxy."""
    if reservation is None:
        return contextlib.nullcontext()
    from benchflow.providers.litellm_runtime import connecting_with

    return connecting_with(reservation)


def release_host_proxy(reservation: Any) -> None:
    """Give back a reservation no connection took."""
    if reservation is None:
        return
    from benchflow.providers.litellm_runtime import (
        release_host_proxy as _release_host_proxy,
    )

    _release_host_proxy(reservation)


async def drain_host_proxies() -> None:
    """Stop the warm host proxies no trial took (a job's end); nothing to do
    in a process that never started one."""
    module = sys.modules.get("benchflow.providers.litellm_runtime")
    if module is not None:
        await module.drain_host_proxies()
