"""Provider-route helpers shared by env resolution and usage proxy routing."""

from __future__ import annotations

from dataclasses import dataclass

from benchflow.agents.providers import find_provider, strip_provider_prefix

GOOGLE_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
EXPLICIT_PROVIDER_BASE_URL_ENVS = "BENCHFLOW_EXPLICIT_PROVIDER_BASE_URL_ENVS"

GENERIC_PROVIDER_BASE_URL_ENVS = (
    "BENCHFLOW_PROVIDER_BASE_URL",
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "LITELLM_BASE_URL",
)
GENERIC_PROVIDER_API_KEY_ENVS = (
    "BENCHFLOW_PROVIDER_API_KEY",
    "LLM_API_KEY",
    "LITELLM_API_KEY",
)
GEMINI_BASE_URL_ENVS = ("GOOGLE_GEMINI_BASE_URL", "GEMINI_API_BASE_URL")


@dataclass(frozen=True)
class ProviderRoute:
    """Routing contract for a model family before any proxy is installed."""

    provider_name: str
    default_base_url: str | None = None
    base_url_envs: tuple[str, ...] = ()
    allow_inherited_generic_base_url: bool = True


def is_native_google_gemini_model(model: str | None) -> bool:
    """True for direct Google AI Studio Gemini/Gemma models.

    Registered providers (for example ``google-vertex/`` or ``litellm/``) own
    their own routing contract and are intentionally excluded here.
    """
    if not model or find_provider(model) is not None:
        return False
    bare = strip_provider_prefix(model).lower()
    return "gemini" in bare or "gemma" in bare


def provider_route_for_model(model: str | None) -> ProviderRoute | None:
    """Return the canonical route contract for model families that need one."""
    if is_native_google_gemini_model(model):
        return ProviderRoute(
            provider_name="google-gemini",
            default_base_url=GOOGLE_GEMINI_DEFAULT_BASE_URL,
            base_url_envs=GEMINI_BASE_URL_ENVS,
            allow_inherited_generic_base_url=False,
        )
    return None


def mark_explicit_provider_base_url_envs(
    agent_env: dict[str, str],
    explicit_agent_env_keys: set[str],
) -> None:
    """Record explicit generic base-url overrides for later proxy routing.

    After ``resolve_agent_env`` has merged .env and host defaults, the usage
    proxy no longer knows which generic base URL came from a user's run-specific
    override. This marker preserves that distinction without changing the
    public agent-env API.
    """
    agent_env.pop(EXPLICIT_PROVIDER_BASE_URL_ENVS, None)
    explicit = [
        key for key in GENERIC_PROVIDER_BASE_URL_ENVS if key in explicit_agent_env_keys
    ]
    if explicit:
        agent_env[EXPLICIT_PROVIDER_BASE_URL_ENVS] = ",".join(explicit)


def drop_inherited_cross_provider_overrides(
    agent_env: dict[str, str],
    *,
    model: str | None,
    explicit_agent_env_keys: set[str],
) -> None:
    """Remove inherited generic provider env vars that do not belong to model.

    Explicit ``agent_env`` values stay authoritative. Only broad defaults copied
    from .env / host env are removed.
    """
    if not model:
        return

    provider = find_provider(model)
    if provider is not None:
        _, provider_cfg = provider
        # Providers with an empty base URL (for example vllm/) are explicitly
        # user-supplied endpoints, so inherited BENCHFLOW_PROVIDER_* is their
        # normal configuration path.
        if provider_cfg.base_url:
            for key in {"BENCHFLOW_PROVIDER_BASE_URL", "BENCHFLOW_PROVIDER_API_KEY"}:
                if key not in explicit_agent_env_keys:
                    agent_env.pop(key, None)
        return

    route = provider_route_for_model(model)
    if route is None or route.allow_inherited_generic_base_url:
        return

    inherited_generic_keys = (
        set(GENERIC_PROVIDER_BASE_URL_ENVS) | set(GENERIC_PROVIDER_API_KEY_ENVS)
    ) - explicit_agent_env_keys
    for key in inherited_generic_keys:
        agent_env.pop(key, None)


def resolve_native_usage_proxy_target(
    agent_env: dict[str, str],
    model: str | None,
) -> str | None:
    """Resolve usage-proxy target for native provider families.

    Returning ``None`` means callers should use their legacy/general routing
    path. For native Gemini, generic inherited base URLs are deliberately
    ignored; only explicit generic overrides, Gemini-specific base vars, or the
    Google default can route the run.
    """
    route = provider_route_for_model(model)
    if route is None:
        return None

    explicit_generic = tuple(
        key
        for key in agent_env.get(EXPLICIT_PROVIDER_BASE_URL_ENVS, "").split(",")
        if key
    )
    for env_name in explicit_generic:
        if env_name in GENERIC_PROVIDER_BASE_URL_ENVS and agent_env.get(env_name):
            return agent_env[env_name]

    for env_name in route.base_url_envs:
        if agent_env.get(env_name):
            return agent_env[env_name]

    return route.default_base_url
