"""Provider runtime helpers for host-side proxy processes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from benchflow.agents.providers import find_provider, strip_provider_prefix
from benchflow.agents.registry import AGENTS
from benchflow.providers.bedrock_proxy import BedrockProxyServer
from benchflow.trajectories.pricing import PRICING_USD_PER_MTOK, PricingEntry
from benchflow.trajectories.proxy import TrajectoryProxy

logger = logging.getLogger(__name__)

BEDROCK_PROXY_BIND_HOST = "0.0.0.0"
BEDROCK_PROXY_LOCAL_HOST = "127.0.0.1"
USAGE_PROXY_BIND_HOST = "0.0.0.0"
PROMPT_CACHE_RETENTION_ENV = "BENCHFLOW_PROVIDER_PROMPT_CACHE_RETENTION"
_PROMPT_CACHE_RETENTION_VALUES = {"in_memory", "24h"}


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


def _usage_unavailable() -> dict[str, Any]:
    return {
        "n_input_tokens": None,
        "n_output_tokens": None,
        "n_cache_read_tokens": None,
        "n_cache_creation_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "usage_source": "unavailable",
        "price_source": None,
    }


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


def _infer_default_provider_url(agent: str, model: str | None) -> str | None:
    bare = strip_provider_prefix(model) if model else ""
    m = bare.lower()
    if "claude" in m or "anthropic" in m or agent == "claude-agent-acp":
        return "https://api.anthropic.com"
    if (
        "gpt" in m
        or "openai" in m
        or m.startswith(("o1", "o3", "o4"))
        or agent in {"codex-acp", "opencode"}
    ):
        return "https://api.openai.com/v1"
    if "gemini" in m or "gemma" in m or agent == "gemini":
        return "https://generativelanguage.googleapis.com"
    return None


def _resolve_usage_proxy_target(
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
) -> str | None:
    if agent_env.get("BENCHFLOW_PROVIDER_BASE_URL"):
        return agent_env["BENCHFLOW_PROVIDER_BASE_URL"]
    for env_name in _agent_base_url_envs(agent):
        if agent_env.get(env_name):
            return agent_env[env_name]
    return _infer_default_provider_url(agent, model)


def _pricing_for_model(model: str | None) -> PricingEntry | None:
    if not model:
        return None
    bare = strip_provider_prefix(model).lower()
    for prefix, pricing in PRICING_USD_PER_MTOK.items():
        if bare.startswith(prefix):
            return pricing
    return None


def _estimate_cost_usd(
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    cache_tokens_included_in_input: bool = False,
) -> float | None:
    pricing = _pricing_for_model(model)
    if pricing is None:
        return None
    priced_input_tokens = input_tokens
    if cache_tokens_included_in_input:
        priced_input_tokens = max(
            input_tokens - cache_read_tokens - cache_creation_tokens, 0
        )
    cost = (
        priced_input_tokens * pricing.input
        + output_tokens * pricing.output
        + cache_read_tokens * pricing.cache_read
        + cache_creation_tokens * pricing.cache_creation
    ) / 1_000_000
    return round(cost, 10)


def _model_from_trajectory(runtime: ProviderRuntime) -> str | None:
    # Prefer the model the provider actually reported in captured exchanges;
    # backend_model is only the model requested at proxy-creation time and can
    # be stale if a role switched models. Falls back to it when no exchange
    # carries a model (e.g. Gemini, which puts the model in the URL path).
    trajectory = getattr(runtime.server, "trajectory", None)
    if trajectory:
        for exchange in trajectory.exchanges:
            response_model = exchange.response.body.get("model")
            if response_model:
                return response_model
            request_model = exchange.request.body.get("model")
            if request_model:
                return request_model
    return runtime.backend_model


def _cache_tokens_are_input_breakdown(trajectory: Any) -> bool:
    for exchange in trajectory.exchanges:
        usage = exchange.response.body.get("usage", {})
        if (usage.get("prompt_tokens_details") or {}).get("cached_tokens") is not None:
            return True
        if (usage.get("input_tokens_details") or {}).get("cached_tokens") is not None:
            return True
    return False


async def ensure_usage_proxy_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: ProviderRuntime | None,
    environment: str,
    session_id: str = "",
) -> tuple[dict[str, str], ProviderRuntime | None]:
    """Start the host-side usage proxy and wire env vars to it.

    For remote cloud sandboxes (e.g. Daytona) the host proxy is unreachable
    from the agent — it runs on a different machine and there is no reverse
    tunnel back to the host. In that case the proxy is skipped: the agent
    talks to the provider directly with its real key and host-side usage
    telemetry reports ``usage_source: "unavailable"``.
    """
    if agent == "oracle":
        return agent_env, runtime
    if not host_proxy_reachable_from_agent(environment):
        if runtime is not None:
            await stop_provider_runtime(runtime)
        logger.info(
            "Skipping host-side usage telemetry proxy: the '%s' sandbox runs "
            "the agent on a remote host unreachable from the host proxy; the "
            "agent will call the provider directly and usage telemetry will "
            "be unavailable for this run.",
            environment or "unknown",
        )
        return agent_env, None
    target = _resolve_usage_proxy_target(agent, agent_env, model)
    if not target:
        return agent_env, runtime
    target = target.rstrip("/")

    # A multi-role scene can switch providers between connect_as() calls. The
    # running proxy forwards to a fixed upstream, so reusing it would route the
    # new role's traffic to the wrong endpoint — retire it and start a fresh
    # one for the new target.
    if runtime is not None and getattr(runtime.server, "target", None) != target:
        await stop_provider_runtime(runtime)
        runtime = None

    if runtime is None:
        prompt_cache_retention = agent_env.get(PROMPT_CACHE_RETENTION_ENV)
        if (
            prompt_cache_retention is not None
            and prompt_cache_retention not in _PROMPT_CACHE_RETENTION_VALUES
        ):
            raise ValueError(
                f"{PROMPT_CACHE_RETENTION_ENV} must be one of: "
                f"{', '.join(sorted(_PROMPT_CACHE_RETENTION_VALUES))}"
            )
        logger.info("Starting host-side usage telemetry proxy")
        server = TrajectoryProxy(
            target=target,
            session_id=session_id,
            agent_name=agent,
            host=USAGE_PROXY_BIND_HOST,
            port=0,
            prompt_cache_retention=prompt_cache_retention,
        )
        await server.start()
        runtime = ProviderRuntime(
            kind="usage-proxy",
            host=_bedrock_proxy_command(environment=environment),
            port=server.port,
            backend_model=strip_provider_prefix(model) if model else None,
            server=server,
        )

    updated = dict(agent_env)
    updated["BENCHFLOW_PROVIDER_BASE_URL"] = runtime.base_url
    agent_cfg = AGENTS.get(agent)
    mapped_base = (
        agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_BASE_URL") if agent_cfg else None
    )
    for env_name in _agent_base_url_envs(agent):
        if env_name in updated or env_name == mapped_base:
            updated[env_name] = runtime.base_url
    return updated, runtime


def extract_usage(runtime: ProviderRuntime | None) -> dict[str, Any]:
    """Extract aggregate token/cost metrics from a usage proxy runtime."""
    if runtime is None or runtime.kind != "usage-proxy" or runtime.server is None:
        return _usage_unavailable()
    trajectory = getattr(runtime.server, "trajectory", None)
    if trajectory is None or not trajectory.exchanges:
        return _usage_unavailable()

    input_tokens = trajectory.total_input_tokens
    output_tokens = trajectory.total_output_tokens
    cache_read_tokens = trajectory.total_cache_read_tokens
    cache_creation_tokens = trajectory.total_cache_creation_tokens
    total_tokens = trajectory.total_provider_tokens
    model = _model_from_trajectory(runtime)
    pricing = _pricing_for_model(model)
    cost_usd = _estimate_cost_usd(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_tokens_included_in_input=_cache_tokens_are_input_breakdown(trajectory),
    )
    return {
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "n_cache_read_tokens": cache_read_tokens,
        "n_cache_creation_tokens": cache_creation_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "usage_source": "provider_response",
        "price_source": pricing.price_source
        if cost_usd is not None and pricing
        else None,
    }


async def ensure_bedrock_proxy_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: ProviderRuntime | None,
    environment: str,
) -> tuple[dict[str, str], ProviderRuntime | None]:
    """Start the host-side Bedrock proxy if needed and wire env vars to it.

    Unlike the usage telemetry proxy (pure telemetry — safe to skip when the
    agent cannot reach the host), the Bedrock proxy is *load-bearing*: it
    translates Anthropic/OpenAI requests into AWS Bedrock Converse calls and
    signs them with host AWS credentials. There is no direct path for the
    agent to reach Bedrock without it. So on a remote sandbox where the host
    proxy is unreachable, the run cannot succeed — we fail fast here with an
    actionable error instead of injecting an unreachable ``127.0.0.1`` base
    URL that would surface as an opaque mid-run ``ECONNREFUSED``.
    """
    if not needs_provider_runtime(model):
        return agent_env, runtime
    assert model is not None

    if not host_proxy_reachable_from_agent(environment):
        if runtime is not None:
            await stop_provider_runtime(runtime)
        raise RuntimeError(
            f"Bedrock-routed models are not supported on the "
            f"'{environment}' sandbox: the host-side Bedrock proxy that "
            f"translates and signs requests to AWS Bedrock binds to the "
            f"host machine and is unreachable from the agent, which runs "
            f"on a separate remote host. Re-run with '--sandbox docker', "
            f"or select a model that is not routed through AWS Bedrock."
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
    if runtime.kind == "usage-proxy" and runtime.server is not None:
        await runtime.server.stop()
