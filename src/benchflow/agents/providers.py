"""LLM provider registry.

Every provider that benchflow routes models through lives here — both
"custom" providers (like zai/) that need explicit endpoint config, and
"native" providers (like google-vertex/) that agents already support
but we still register so ``is_vertex_model()`` and
``infer_env_key_for_model()`` have a single source of truth.

Adding a new provider is a registry-only change: append one entry to
``PROVIDERS`` below. No new functions, no shim edits, no SDK edits.
``tests/test_registry_invariants.py`` runs contract checks against every
entry — read it for the executable schema.

Required fields
---------------
- ``name``           Must equal the dict key.
- ``base_url``       Primary endpoint URL. May contain ``{placeholder}``
                     tokens that get expanded from env vars via
                     ``url_params``. Empty string is allowed for
                     "user-supplied at runtime" providers (e.g. ``vllm``).
- ``api_protocol``   "anthropic-messages", "openai-completions", or
                     "openai-responses" — the
                     wire protocol the primary ``base_url`` speaks.
- ``auth_type``      "api_key" | "adc" | "aws" | "none".
                     - "api_key": ``auth_env`` **must** be set.
                     - "adc": Application Default Credentials (GCP). The
                       SDK writes the credential file from
                       ``credential_files`` and sets the corresponding env.
                     - "aws": Bedrock API-key auth via
                       ``AWS_BEARER_TOKEN_BEDROCK`` plus region.
                     - "none": no auth.

Common optional fields
----------------------
- ``auth_env``         Env var holding the API key. Must be set iff
                       ``auth_type == "api_key"``.
- ``url_params``       ``{placeholder: ENV_VAR}`` — every placeholder in
                       ``base_url`` (or any ``endpoints`` URL) must have an
                       entry, and every entry must be referenced somewhere.
- ``endpoints``        ``{api_protocol: url}`` for providers that expose
                       multiple protocol surfaces (e.g. zai serves
                       openai-responses, openai-completions, and
                       anthropic-messages). Picked at runtime based on the
                       agent's ``api_protocol``.
- ``models``           Optional list of model metadata dicts (id, name,
                       contextWindow, etc.) consumed by agent shims. ``id``
                       is required and must be unique within the provider.
- ``model_prefixes``   Bare-model-name family tokens this provider owns
                       (e.g. ``["deepseek"]``), used by
                       ``find_provider_for_bare_model()`` to route
                       prefix-stripped ids. Tokens must be lowercase and
                       unique across providers (longest token wins).
- ``credential_files`` List of dicts with ``"path"`` and ``"env_source"``
                       (and optional ``"post_env"``) — used by ADC providers
                       to write the credential blob into the container.

Look at the existing entries below for worked examples:
``zai`` (multi-endpoint, models metadata), ``google-vertex`` (ADC,
credential_files, url_params), ``vllm`` (user-supplied base_url).
"""

from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for a custom LLM provider."""

    name: str
    base_url: (
        str  # primary endpoint; may contain {placeholders} expanded via url_params
    )
    api_protocol: str  # protocol for base_url: "openai-responses" | "openai-completions" | "anthropic-messages"
    auth_type: str  # "api_key" | "adc" | "aws" | "none"
    auth_env: str | None = None  # env var holding the API key (None for ADC)
    url_params: dict[str, str] = field(default_factory=dict)  # {placeholder: ENV_VAR}
    models: list[dict] = field(default_factory=list)  # model metadata for agents
    # Bare-model-name family tokens this provider owns (e.g. ["deepseek"],
    # ["qwen"]). Used by find_provider_for_bare_model() to route a *prefix-less*
    # model id (after strip_provider_prefix) — e.g. "deepseek-v4-flash" or
    # "qwen3.6-max" — back to its provider when no "provider/" prefix is present.
    # A token matches when the model id equals it or continues with a
    # non-letter (version digit / "-" / "."), so "glm" matches "glm-4.6" and
    # "glm5" but not "glmnext". Longest token wins across providers.
    model_prefixes: list[str] = field(default_factory=list)
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
        url_params={
            "project_id": "GOOGLE_CLOUD_PROJECT",
            "location": "GOOGLE_CLOUD_LOCATION",
        },
        credential_files=[
            {
                "path": "{home}/.config/gcloud/application_default_credentials.json",
                "env_source": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "post_env": {
                    "GOOGLE_APPLICATION_CREDENTIALS": "{home}/.config/gcloud/application_default_credentials.json",
                },
            }
        ],
    ),
    "anthropic-vertex": ProviderConfig(
        name="anthropic-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="anthropic-messages",
        auth_type="adc",
        url_params={
            "project_id": "GOOGLE_CLOUD_PROJECT",
            "location": "GOOGLE_CLOUD_LOCATION",
        },
        credential_files=[
            {
                "path": "{home}/.config/gcloud/application_default_credentials.json",
                "env_source": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "post_env": {
                    "GOOGLE_APPLICATION_CREDENTIALS": "{home}/.config/gcloud/application_default_credentials.json",
                },
            }
        ],
    ),
    # ── Azure AI Foundry providers ──
    #
    # Foundry exposes different protocol surfaces. Keep those surfaces explicit
    # in provider prefixes, while sharing one Azure resource/key env contract.
    # AZURE_RESOURCE is normally derived from AZURE_API_ENDPOINT in env.py;
    # users can also set it directly via --agent-env.
    "azure-foundry-openai": ProviderConfig(
        name="azure-foundry-openai",
        base_url="https://{resource}.openai.azure.com/openai/v1",
        # Use the broadest OpenAI-compatible default for agents that do not
        # declare their own protocol (e.g. pi-acp). Responses-native agents
        # still select the explicit openai-responses endpoint below.
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="AZURE_API_KEY",
        url_params={"resource": "AZURE_RESOURCE"},
        endpoints={
            "openai-responses": "https://{resource}.openai.azure.com/openai/v1",
        },
    ),
    "azure-foundry-anthropic": ProviderConfig(
        name="azure-foundry-anthropic",
        base_url="https://{resource}.services.ai.azure.com/anthropic",
        api_protocol="anthropic-messages",
        auth_type="api_key",
        auth_env="AZURE_API_KEY",
        url_params={"resource": "AZURE_RESOURCE"},
    ),
    # ── OpenAI first-party API ──
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="OPENAI_API_KEY",
        endpoints={
            "openai-completions": "https://api.openai.com/v1",
            "openai-responses": "https://api.openai.com/v1",
        },
    ),
    # OpenAI US data-residency endpoint. Same key, regional URL.
    "us-openai": ProviderConfig(
        name="us-openai",
        base_url="https://us.api.openai.com/v1",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="OPENAI_API_KEY",
        endpoints={
            "openai-completions": "https://us.api.openai.com/v1",
            "openai-responses": "https://us.api.openai.com/v1",
        },
    ),
    # GitHub Models exposes an OpenAI-compatible endpoint that GitHub Actions
    # can call with the workflow-scoped GITHUB_TOKEN and `models: read`.
    "github-models": ProviderConfig(
        name="github-models",
        base_url="https://models.github.ai/inference",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="GITHUB_TOKEN",
    ),
    # TODO: add eu-openai (https://eu.api.openai.com/v1) when needed.
    # ── OpenAI-compatible inference servers (user-supplied base_url) ──
    "vllm": ProviderConfig(
        name="vllm",
        base_url="",  # user-supplied via --agent-env BENCHFLOW_PROVIDER_BASE_URL=...
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="OPENAI_API_KEY",  # vLLM uses OpenAI-compatible auth
    ),
    "litellm": ProviderConfig(
        name="litellm",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="LITELLM_API_KEY",
        url_params={"base_url": "LITELLM_BASE_URL"},
    ),
    "aws-bedrock": ProviderConfig(
        name="aws-bedrock",
        base_url="",  # LiteLLM supplies the runtime URL later
        api_protocol="openai-responses",
        auth_type="aws",
        endpoints={
            "anthropic-messages": "",
        },
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
            "openai-responses": "https://api.z.ai/api/paas/v4",
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
    "kimi": ProviderConfig(
        name="kimi",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="KIMI_API_KEY",
        url_params={"base_url": "KIMI_BASE_URL"},
        model_prefixes=["kimi", "moonshot"],
    ),
    "minimax": ProviderConfig(
        name="minimax",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="MINIMAX_API_KEY",
        url_params={"base_url": "MINIMAX_BASE_URL"},
        model_prefixes=["minimax"],
    ),
    "qwen-dashscope": ProviderConfig(
        name="qwen-dashscope",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="QWEN_API_KEY",
        url_params={"base_url": "QWEN_BASE_URL"},
        model_prefixes=["qwen"],
    ),
    "glm": ProviderConfig(
        name="glm",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="GLM_API_KEY",
        url_params={"base_url": "GLM_BASE_URL"},
        model_prefixes=["glm"],
        models=[
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
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="DEEPSEEK_API_KEY",
        url_params={"base_url": "DEEPSEEK_BASE_URL"},
        model_prefixes=["deepseek"],
    ),
    "xiaomi": ProviderConfig(
        name="xiaomi",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="XIAOMI_API_KEY",
        url_params={"base_url": "XIAOMI_BASE_URL"},
        model_prefixes=["xiaomi", "mimo"],
    ),
    "doubao-seed-2-lite": ProviderConfig(
        name="doubao-seed-2-lite",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="DOUBAO_SEED_2_LITE_API_KEY",
        url_params={"base_url": "DOUBAO_VOLCES_BASE_URL"},
        model_prefixes=["doubao-seed-2-lite"],
    ),
    "doubao-seed-2-pro": ProviderConfig(
        name="doubao-seed-2-pro",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="DOUBAO_SEED_2_PRO_API_KEY",
        url_params={"base_url": "DOUBAO_VOLCES_BASE_URL"},
        model_prefixes=["doubao-seed-2-pro"],
    ),
    "hunyuan": ProviderConfig(
        name="hunyuan",
        base_url="{base_url}",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="HUNYUAN_API_KEY",
        url_params={"base_url": "HUNYUAN_BASE_URL"},
        model_prefixes=["hunyuan"],
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


def _bare_model_matches_token(model: str, token: str) -> bool:
    """True if a bare model id belongs to the family named by *token*.

    Matches when the (lower-cased) model id equals the token, or starts with
    it and the next character is *not* a letter — i.e. a version digit, "-",
    or "." continues the same family. This routes both hyphen-style ids
    (``deepseek-v4-flash`` → ``deepseek``, ``glm-4.6`` → ``glm``) and
    version-suffixed ids (``qwen3.6-max`` → ``qwen``), while refusing
    different words (``glmnext`` does NOT match ``glm``).
    """
    if model == token:
        return True
    if model.startswith(token):
        return not model[len(token)].isalpha()
    return False


def find_provider_for_bare_model(model: str) -> tuple[str, ProviderConfig] | None:
    """Map a BARE (prefix-stripped) model id to a registered provider.

    ``find_provider`` only matches an explicit ``provider/`` prefix, so once
    ``strip_provider_prefix`` has run (ACP set_model, BENCHFLOW_PROVIDER_MODEL,
    Harbor YAML) a bare id like ``deepseek-v4-flash`` no longer resolves and
    callers wrongly default it to anthropic. This consults each provider's
    declared ``model_prefixes`` (the registry owns that knowledge) so any
    registered provider routes correctly without the prefix.

    Resolution (deterministic):
      1. Longest ``model_prefixes`` token match wins across all providers
         (e.g. ``doubao-seed-2-pro`` beats a hypothetical ``doubao``).
      2. If no token matches, an exact ``models[].id`` declared by exactly
         one provider is used as a fallback (ambiguous ids are skipped).

    Returns ``(provider_name, config)`` or ``None``. Inputs that still carry a
    registered ``provider/`` prefix return ``None`` (use ``find_provider``).
    """
    m = model.lower().strip()
    if not m or find_provider(m) is not None:
        return None

    # 1. Longest model-family-token match.
    best: tuple[int, str, ProviderConfig] | None = None
    for name, cfg in PROVIDERS.items():
        for token in cfg.model_prefixes:
            t = token.lower()
            if _bare_model_matches_token(m, t) and (best is None or len(t) > best[0]):
                best = (len(t), name, cfg)
    if best is not None:
        return best[1], best[2]

    # 2. Exact declared-model-id fallback (only when unambiguous).
    matches = [
        (name, cfg)
        for name, cfg in PROVIDERS.items()
        if any(str(meta.get("id", "")).lower() == m for meta in cfg.models)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


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


def provider_host_for_model(model: str, env: dict[str, str]) -> str | None:
    """Hostname of a model's resolved provider base_url, or ``None``.

    Used to allowlist the model provider's host under a restrictive network
    policy so the agent can reach it directly over HTTPS (a clean CONNECT
    tunnel). Returns ``None`` when the model has no registered provider prefix
    or the provider's base_url env is unset — don't guess; leave the allowlist
    unchanged.
    """
    from urllib.parse import urlparse

    # A bare id (prefix already stripped, e.g. 'deepseek-v4-flash') no longer
    # matches find_provider; consult the bare-model registry so the host still
    # resolves and gets allowlisted (otherwise a restrictive run can't reach it).
    found = find_provider(model) or find_provider_for_bare_model(
        strip_provider_prefix(model)
    )
    if found is None:
        return None
    _, cfg = found
    try:
        url = resolve_base_url(cfg, env)
    except KeyError:
        return None
    if not url:
        return None
    return urlparse(url if "://" in url else f"https://{url}").hostname


def strip_provider_prefix(model: str) -> str:
    """Strip a *registered* provider prefix. Unregistered inputs pass through.

    "anthropic-vertex/claude-sonnet-4-6" → "claude-sonnet-4-6"
    "zai/glm-5" → "glm-5"
    "vllm/Qwen/Qwen3-Coder" → "Qwen/Qwen3-Coder"  (HF org/model kept intact)
    "Qwen/Qwen3-Coder" → "Qwen/Qwen3-Coder"       (no registered prefix → unchanged)
    "claude-sonnet-4-6" → "claude-sonnet-4-6"

    Single normalization point for downstream callers (ACP set_model,
    BENCHFLOW_PROVIDER_MODEL env var, Harbor YAML parse). If a model ID
    reaches an agent launcher still prefixed, fix the routing into this
    function — do NOT strip again at the call site. See PRs #154 and #155
    for the symptomatic-patch anti-pattern that caused the original bug.
    """
    result = find_provider(model)
    if result:
        return model[len(result[0]) + 1 :]
    return model


def resolve_auth_env(model: str) -> str | None:
    """Return the env var name needed for a model's provider, or None.

    Returns None for ADC-based, AWS-auth providers, and unknown models.
    """
    result = find_provider(model)
    if result is None:
        return None
    _, cfg = result
    if cfg.auth_type in {"adc", "aws", "none"}:
        return None
    return cfg.auth_env
