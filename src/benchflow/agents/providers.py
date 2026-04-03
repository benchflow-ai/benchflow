"""LLM provider registry.

Every provider that benchflow routes models through lives here — both
"custom" providers (like zai/) that need explicit endpoint config, and
"native" providers (like google-vertex/) that agents already support
but we still register so is_vertex_model() and infer_env_key_for_model()
have a single source of truth.

Native providers have empty models lists — agents know them natively.
Custom providers include model metadata so agent shims (e.g. openclaw)
can write the config files they need.

Adding a new provider = one entry in PROVIDERS. No new functions needed.
"""

from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for a custom LLM provider."""

    name: str
    base_url: str  # may contain {placeholders} expanded via url_params
    api_protocol: str  # "openai-completions" | "anthropic-messages"
    auth_type: str  # "api_key" | "adc"
    auth_env: str | None = None  # env var holding the API key (None for ADC)
    url_params: dict[str, str] = field(default_factory=dict)  # {placeholder: ENV_VAR}
    models: list[dict] = field(default_factory=list)  # model metadata for agents


# ── Provider registry ──

PROVIDERS: dict[str, ProviderConfig] = {
    # ── Native Vertex AI providers (agents support these natively) ──
    "google-vertex": ProviderConfig(
        name="google-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="openai-completions",
        auth_type="adc",
        url_params={"project_id": "GOOGLE_CLOUD_PROJECT", "location": "GOOGLE_CLOUD_LOCATION"},
    ),
    "anthropic-vertex": ProviderConfig(
        name="anthropic-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="anthropic-messages",
        auth_type="adc",
        url_params={"project_id": "GOOGLE_CLOUD_PROJECT", "location": "GOOGLE_CLOUD_LOCATION"},
    ),
    # ── Custom providers (need explicit endpoint config in agent shims) ──
    "zai": ProviderConfig(
        name="zai",
        base_url="https://api.z.ai/api/paas/v4",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="ZAI_API_KEY",
        models=[
            {
                "id": "glm-5",
                "name": "GLM-5",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 200000,
                "maxTokens": 131072,
            },
        ],
    ),
    "vertex-zai": ProviderConfig(
        name="vertex-zai",
        base_url=(
            "https://aiplatform.googleapis.com/v1/projects/"
            "{project_id}/locations/global/endpoints/openapi"
        ),
        api_protocol="openai-completions",
        auth_type="adc",
        url_params={"project_id": "GOOGLE_CLOUD_PROJECT"},
        models=[
            {
                "id": "zai-org/glm-5-maas",
                "name": "GLM-5 (Vertex AI)",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 200000,
                "maxTokens": 131072,
            },
        ],
    ),
}


def find_provider(model: str) -> tuple[str, ProviderConfig] | None:
    """Find the custom provider for a model ID based on its prefix.

    Returns (provider_name, config) or None if no custom provider matches.
    Matches longest prefix first to handle nested prefixes (e.g. vertex-zai/ vs vertex/).
    """
    m = model.lower()
    # Sort by prefix length descending so longer prefixes match first
    candidates = []
    for name, cfg in PROVIDERS.items():
        prefix = f"{name}/"
        if m.startswith(prefix):
            candidates.append((len(prefix), name, cfg))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, name, cfg = candidates[0]
    return name, cfg


def resolve_base_url(provider: ProviderConfig, env: dict[str, str]) -> str:
    """Expand {placeholders} in a provider's base_url using env vars.

    Raises KeyError if a required env var is missing.
    """
    if not provider.url_params:
        return provider.base_url
    replacements = {}
    for placeholder, env_var in provider.url_params.items():
        value = env.get(env_var)
        if not value:
            raise KeyError(
                f"Provider {provider.name!r} requires {env_var} for "
                f"{{{placeholder}}} in base_url, but it is not set."
            )
        replacements[placeholder] = value
    return provider.base_url.format_map(replacements)


def resolve_auth_env(model: str) -> str | None:
    """Return the env var name needed for a model's provider, or None.

    Returns None for ADC-based providers and unknown models.
    """
    result = find_provider(model)
    if result is None:
        return None
    _, cfg = result
    if cfg.auth_type == "adc":
        return None
    return cfg.auth_env
