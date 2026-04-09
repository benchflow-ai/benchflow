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
    base_url: str  # primary endpoint; may contain {placeholders} expanded via url_params
    api_protocol: str  # protocol for base_url: "openai-completions" | "anthropic-messages"
    auth_type: str  # "api_key" | "adc"
    auth_env: str | None = None  # env var holding the API key (None for ADC)
    url_params: dict[str, str] = field(default_factory=dict)  # {placeholder: ENV_VAR}
    models: list[dict] = field(default_factory=list)  # model metadata for agents
    # Multi-protocol support: {protocol: base_url} for providers with multiple APIs.
    # base_url + api_protocol is the primary; endpoints adds alternatives.
    endpoints: dict[str, str] = field(default_factory=dict)
    credential_files: list[dict] = field(default_factory=list)
    # Files to write into container (e.g. GCP ADC).
    # Each dict: {"path": str, "env_source": str, "post_env": {k: v} (optional)}

    @property
    def all_endpoints(self) -> dict[str, str]:
        """Merged view: endpoints dict with base_url/api_protocol as fallback."""
        merged = {self.api_protocol: self.base_url}
        merged.update(self.endpoints)
        return merged


# ── Provider registry ──

PROVIDERS: dict[str, ProviderConfig] = {
    # ── Native Vertex AI providers (agents support these natively) ──
    "google-vertex": ProviderConfig(
        name="google-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="openai-completions",
        auth_type="adc",
        url_params={"project_id": "GOOGLE_CLOUD_PROJECT", "location": "GOOGLE_CLOUD_LOCATION"},
        credential_files=[{
            "path": "{home}/.config/gcloud/application_default_credentials.json",
            "env_source": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
            "post_env": {
                "GOOGLE_APPLICATION_CREDENTIALS":
                    "{home}/.config/gcloud/application_default_credentials.json",
            },
        }],
    ),
    "anthropic-vertex": ProviderConfig(
        name="anthropic-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="anthropic-messages",
        auth_type="adc",
        url_params={"project_id": "GOOGLE_CLOUD_PROJECT", "location": "GOOGLE_CLOUD_LOCATION"},
        credential_files=[{
            "path": "{home}/.config/gcloud/application_default_credentials.json",
            "env_source": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
            "post_env": {
                "GOOGLE_APPLICATION_CREDENTIALS":
                    "{home}/.config/gcloud/application_default_credentials.json",
            },
        }],
    ),
    # ── Custom providers (need explicit endpoint config in agent shims) ──
    "zai": ProviderConfig(
        name="zai",
        base_url="https://api.z.ai/api/paas/v4",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="ZAI_API_KEY",
        endpoints={
            "openai-completions": "https://api.z.ai/api/paas/v4",
            "anthropic-messages": "https://api.z.ai/api/anthropic",
        },
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
            {
                "id": "glm-5.1",
                "name": "GLM-5.1",
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
    Matches longest prefix first to handle nested prefixes (e.g. google-vertex/ vs google/).
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


def resolve_base_url(
    provider: ProviderConfig,
    env: dict[str, str],
    protocol: str | None = None,
) -> str:
    """Expand {placeholders} in a provider's base_url using env vars.

    If *protocol* is given and the provider has an ``endpoints`` entry for it,
    that URL is used instead of the primary ``base_url``.

    Raises KeyError if a required env var is missing.
    """
    url = provider.base_url
    if protocol and provider.endpoints.get(protocol):
        url = provider.endpoints[protocol]
    if not provider.url_params:
        return url
    replacements = {}
    for placeholder, env_var in provider.url_params.items():
        value = env.get(env_var)
        if not value:
            raise KeyError(
                f"Provider {provider.name!r} requires {env_var} for "
                f"{{{placeholder}}} in base_url, but it is not set."
            )
        replacements[placeholder] = value
    return url.format_map(replacements)


def strip_provider_prefix(model: str) -> str:
    """Strip the provider prefix from a model ID.

    "anthropic-vertex/claude-sonnet-4-6" → "claude-sonnet-4-6"
    "zai/glm-5" → "glm-5"
    "claude-sonnet-4-6" → "claude-sonnet-4-6"  (no prefix = unchanged)
    """
    result = find_provider(model)
    if result:
        prefix = f"{result[0]}/"
        return model[len(prefix):]
    # Not a known provider — still strip unknown prefix if present
    if "/" in model:
        return model.split("/", 1)[1]
    return model


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
