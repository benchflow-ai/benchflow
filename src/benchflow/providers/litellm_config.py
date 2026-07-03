"""Pure LiteLLM routing/config helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from benchflow.agents.providers import (
    ProviderConfig,
    find_provider,
    resolve_base_url,
    strip_provider_prefix,
)

AZURE_API_VERSION_ENV = "AZURE_API_VERSION"
AZURE_DEFAULT_API_VERSION = "preview"
BEDROCK_THINKING_EFFORT_ENV = "BENCHFLOW_BEDROCK_THINKING_EFFORT"
LITELLM_MODEL_ALIAS_ENV = "BENCHFLOW_LITELLM_MODEL_ALIAS"
LITELLM_MODEL_VIA_ENV = "BENCHFLOW_LITELLM_MODEL_VIA_ENV"
LITELLM_MASTER_KEY_ENV = "BENCHFLOW_LITELLM_MASTER_KEY"

# Per-token USD prices for models LiteLLM's built-in ``model_cost`` does not
# already know (custom OpenAI-compatible endpoints such as private vLLM servers
# or niche hosted models). When a route's upstream model matches a key here, the
# price is injected into the LiteLLM deployment as ``input_cost_per_token`` /
# ``output_cost_per_token`` so LiteLLM computes ``response_cost`` itself —
# BenchFlow keeps no cost-calculation logic, only this price *data*. Mainstream
# models (OpenAI, Anthropic, Gemini, Bedrock, Azure, …) are already priced by
# LiteLLM and must NOT be listed here.
#
# Keys are matched as a lowercase substring of the bare model id. Values are USD
# *per token* (i.e. price-per-million-tokens / 1e6). Add entries for the custom
# models you run and VERIFY the numbers against the provider's current pricing.
#
# Example:
#   "minimax-m3": (0.30e-6, 1.20e-6),   # $0.30 / $1.20 per 1M in/out — VERIFY
MODEL_COST_PER_TOKEN: dict[str, tuple[float, float]] = {
    # deepseek-v4 is not in LiteLLM's built-in price table (it tops out at
    # deepseek-v3.2), so a deepseek-v4 run otherwise records $0/null cost.
    # v4 list price is not yet published — these are v3-class PLACEHOLDERS;
    # VERIFY against api-docs.deepseek.com/quick_start/pricing before relying on
    # absolute cost figures. (Single flat input rate uses the cache-miss price,
    # so cache-hit-heavy runs are slightly over-counted.)
    "deepseek-v4-pro": (
        0.28e-6,
        0.42e-6,
    ),  # $0.28 / $0.42 per 1M in/out — PLACEHOLDER, VERIFY
    "deepseek-v4-flash": (
        0.28e-6,
        0.42e-6,
    ),  # $0.28 / $0.42 per 1M in/out — PLACEHOLDER, VERIFY
}


def custom_cost_per_token(model: str) -> tuple[float, float] | None:
    """Return (input, output) USD-per-token for a custom model, or None."""
    lowered = model.lower()
    for key, price in MODEL_COST_PER_TOKEN.items():
        if key in lowered:
            return price
    return None


_BEDROCK_ADAPTIVE_THINKING_RE = re.compile(
    r"claude-(?:(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)|fable-5(?!\d))",
    re.IGNORECASE,
)
_BEDROCK_LITELLM_EFFORT_LIMIT_RE = re.compile(
    r"claude-(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)",
    re.IGNORECASE,
)
# Efforts a user may *request*, low→high. LiteLLM 1.88.0rc1's Bedrock Converse
# transform only accepts up to ``high`` for the Claude 4.x Bedrock IDs covered
# by ``_BEDROCK_LITELLM_EFFORT_LIMIT_RE`` and raises BadRequestError on
# ``xhigh``/``max`` (#737), so those requested values are clamped to the
# accepted ceiling below before they reach the wire. Other adaptive-thinking
# Bedrock models, such as Fable 5, keep their requested effort. Kept in sync
# with the standalone proxy patch ``litellm_bedrock_patch`` (which cannot import
# this module — it is deployed into the sandbox alone).
_BEDROCK_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_BEDROCK_THINKING_EFFORTS = set(_BEDROCK_EFFORT_LADDER)
_BEDROCK_LITELLM_MAX_EFFORT = "high"


def _clamp_bedrock_effort(effort: str) -> str:
    """Clamp a requested effort to the highest LiteLLM-accepted rung (#737)."""
    ladder = _BEDROCK_EFFORT_LADDER
    if effort not in ladder:
        return effort
    return ladder[min(ladder.index(effort), ladder.index(_BEDROCK_LITELLM_MAX_EFFORT))]


@dataclass(frozen=True)
class LiteLLMRoute:
    """Resolved LiteLLM model route for one BenchFlow model ID."""

    requested_model: str
    model_alias: str
    upstream_model: str
    provider_name: str
    litellm_params: dict[str, str | int | float | bool]
    required_env: tuple[str, ...] = ()

    @property
    def config_key(self) -> str:
        payload = {
            "requested_model": self.requested_model,
            "model_alias": self.model_alias,
            "upstream_model": self.upstream_model,
            "provider_name": self.provider_name,
            "litellm_params": self.litellm_params,
            "required_env": self.required_env,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def safe_model_alias(model: str) -> str:
    """Return a deterministic proxy-facing model alias for a BenchFlow model ID."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-")
    cleaned = cleaned.replace("--", "-")
    if not cleaned:
        cleaned = "model"
    if len(cleaned) > 96:
        digest = hashlib.sha1(model.encode()).hexdigest()[:10]
        cleaned = f"{cleaned[:80].rstrip('-')}-{digest}"
    return f"benchflow-{cleaned}"


def _env_ref(name: str) -> str:
    return f"os.environ/{name}"


def _canonical_azure_resource_base(
    env: dict[str, str],
    *,
    default_suffix: str,
) -> str:
    endpoint = (env.get("AZURE_API_ENDPOINT") or "").strip()
    if endpoint:
        if "://" not in endpoint:
            endpoint = f"https://{endpoint}"
        parsed = urlparse(endpoint)
        host = parsed.netloc or parsed.path.split("/", 1)[0]
        if host:
            return f"https://{host.strip('/')}/"
        return endpoint.rstrip("/") + "/"

    resource = (env.get("AZURE_RESOURCE") or "").strip()
    if not resource:
        raise ValueError(
            "Azure Foundry models require AZURE_API_ENDPOINT or AZURE_RESOURCE."
        )
    return f"https://{resource}.{default_suffix}/"


def _registered_api_key_ref(cfg: ProviderConfig) -> str | None:
    if cfg.auth_type == "api_key" and cfg.auth_env:
        return _env_ref(cfg.auth_env)
    return None


def _bedrock_thinking_effort(model: str, env: dict[str, str]) -> str | None:
    if not _BEDROCK_ADAPTIVE_THINKING_RE.search(model):
        return None
    effort = (env.get(BEDROCK_THINKING_EFFORT_ENV) or "high").strip().lower()
    if effort not in _BEDROCK_THINKING_EFFORTS:
        effort = "high"
    if _BEDROCK_LITELLM_EFFORT_LIMIT_RE.search(model):
        return _clamp_bedrock_effort(effort)
    return effort


def _route_registered_provider(
    *,
    model: str,
    provider_name: str,
    provider_cfg: ProviderConfig,
    env: dict[str, str],
) -> LiteLLMRoute:
    bare = strip_provider_prefix(model)
    params: dict[str, str | int | float | bool]
    required_env: list[str] = []

    if provider_name == "aws-bedrock":
        required_env.extend(["AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"])
        params = {"model": f"bedrock/{bare}"}
        effort = _bedrock_thinking_effort(bare, env)
        if effort:
            params["reasoning_effort"] = effort
        return LiteLLMRoute(
            requested_model=model,
            model_alias=safe_model_alias(model),
            upstream_model=str(params["model"]),
            provider_name=provider_name,
            litellm_params=params,
            required_env=tuple(required_env),
        )

    if provider_name == "azure-foundry-openai":
        required_env.append("AZURE_API_KEY")
        api_base = _canonical_azure_resource_base(
            env,
            default_suffix="openai.azure.com",
        )
        params = {
            "model": f"azure/{bare}",
            "api_key": _env_ref("AZURE_API_KEY"),
            "api_base": api_base,
            "api_version": env.get(AZURE_API_VERSION_ENV, AZURE_DEFAULT_API_VERSION),
        }
        return LiteLLMRoute(
            requested_model=model,
            model_alias=safe_model_alias(model),
            upstream_model=str(params["model"]),
            provider_name=provider_name,
            litellm_params=params,
            required_env=tuple(required_env),
        )

    if provider_name == "azure-foundry-anthropic":
        required_env.append("AZURE_API_KEY")
        api_base = _canonical_azure_resource_base(
            env,
            default_suffix="services.ai.azure.com",
        ).rstrip("/")
        if not api_base.endswith("/anthropic"):
            api_base = f"{api_base}/anthropic"
        params = {
            "model": f"azure_ai/{bare}",
            "api_key": _env_ref("AZURE_API_KEY"),
            "api_base": api_base,
        }
        return LiteLLMRoute(
            requested_model=model,
            model_alias=safe_model_alias(model),
            upstream_model=str(params["model"]),
            provider_name=provider_name,
            litellm_params=params,
            required_env=tuple(required_env),
        )

    if provider_cfg.auth_type == "adc":
        params = {"model": f"vertex_ai/{bare}"}
        return LiteLLMRoute(
            requested_model=model,
            model_alias=safe_model_alias(model),
            upstream_model=str(params["model"]),
            provider_name=provider_name,
            litellm_params=params,
        )

    protocol = (
        "openai-completions"
        if "openai-completions" in provider_cfg.all_endpoints
        else provider_cfg.api_protocol
    )
    explicit_api_base = (env.get("BENCHFLOW_PROVIDER_BASE_URL") or "").strip()
    explicit_api_key = (env.get("BENCHFLOW_PROVIDER_API_KEY") or "").strip()
    if explicit_api_base and explicit_api_key:
        api_base = explicit_api_base
    else:
        try:
            api_base = resolve_base_url(
                provider_cfg,
                env,
                protocol=protocol,
            )
        except KeyError as exc:
            missing = ", ".join(sorted(provider_cfg.url_params.values()))
            raise ValueError(
                f"Provider {provider_name!r} for model {model!r} requires {missing}."
            ) from exc

    # User-supplied-base_url providers (e.g. vllm) carry an empty config base_url
    # and resolve to "". Honor the runtime-supplied BENCHFLOW_PROVIDER_BASE_URL
    # so traffic reaches the user's endpoint instead of defaulting to api.openai.com.
    if not api_base:
        api_base = (env.get("BENCHFLOW_PROVIDER_BASE_URL") or "").strip()

    if protocol == "anthropic-messages":
        upstream = f"anthropic/{bare}"
    else:
        upstream = f"openai/{bare}"
    params = {"model": upstream}
    if api_base:
        params["api_base"] = api_base
    api_key_ref = (
        _env_ref("BENCHFLOW_PROVIDER_API_KEY")
        if explicit_api_base and explicit_api_key
        else _registered_api_key_ref(provider_cfg)
    )
    if api_key_ref:
        params["api_key"] = api_key_ref
        if api_key_ref == _env_ref("BENCHFLOW_PROVIDER_API_KEY"):
            required_env.append("BENCHFLOW_PROVIDER_API_KEY")
        elif provider_cfg.auth_env:
            required_env.append(provider_cfg.auth_env)

    return LiteLLMRoute(
        requested_model=model,
        model_alias=safe_model_alias(model),
        upstream_model=upstream,
        provider_name=provider_name,
        litellm_params=params,
        required_env=tuple(required_env),
    )


def resolve_litellm_route(model: str, env: dict[str, str]) -> LiteLLMRoute:
    """Resolve a BenchFlow model ID to one LiteLLM proxy route."""
    provider = find_provider(model)
    if provider is not None:
        provider_name, provider_cfg = provider
        return _route_registered_provider(
            model=model,
            provider_name=provider_name,
            provider_cfg=provider_cfg,
            env=env,
        )

    lower = model.lower()
    bare = strip_provider_prefix(model)
    if lower.startswith("anthropic/"):
        upstream = model
        required = ("ANTHROPIC_API_KEY",)
    elif lower.startswith("gemini/"):
        upstream = model
        required = ("GEMINI_API_KEY",)
    elif "gemini" in lower:
        upstream = f"gemini/{bare}"
        required = ("GEMINI_API_KEY",)
    elif lower.startswith("openai/"):
        upstream = model
        required = ("OPENAI_API_KEY",)
    elif "claude" in lower or "haiku" in lower or "sonnet" in lower or "opus" in lower:
        upstream = f"anthropic/{bare}"
        required = ("ANTHROPIC_API_KEY",)
    else:
        upstream = f"openai/{bare}"
        required = ("OPENAI_API_KEY",)

    params: dict[str, str | int | float | bool] = {"model": upstream}
    if upstream.lower().startswith("gemini/"):
        explicit_api_base = (env.get("BENCHFLOW_PROVIDER_BASE_URL") or "").strip()
        explicit_api_key = (env.get("BENCHFLOW_PROVIDER_API_KEY") or "").strip()
        if explicit_api_base:
            params["api_base"] = explicit_api_base
            if explicit_api_key:
                params["api_key"] = _env_ref("BENCHFLOW_PROVIDER_API_KEY")
                required = ("BENCHFLOW_PROVIDER_API_KEY",)

    key = required[0] if required else None
    if key and "api_key" not in params:
        params["api_key"] = _env_ref(key)
    return LiteLLMRoute(
        requested_model=model,
        model_alias=safe_model_alias(model),
        upstream_model=upstream,
        provider_name="native",
        litellm_params=params,
        required_env=required,
    )


def litellm_proxy_config(
    route: LiteLLMRoute,
    *,
    master_key: str,
    callback_module: str = "benchflow_litellm_callback",
) -> dict[str, object]:
    """Build the LiteLLM ``config.yaml`` payload for one route."""
    params = dict(route.litellm_params)
    cost = custom_cost_per_token(route.upstream_model)
    if cost is not None:
        params.setdefault("input_cost_per_token", cost[0])
        params.setdefault("output_cost_per_token", cost[1])
    openai_alias = f"openai/{route.model_alias}"
    bare_requested = strip_provider_prefix(route.requested_model)
    model_list: list[dict[str, object]] = [
        {"model_name": route.model_alias, "litellm_params": dict(params)},
        {"model_name": openai_alias, "litellm_params": dict(params)},
    ]
    for model_name in (bare_requested, f"openai/{bare_requested}"):
        if model_name and model_name not in {
            entry["model_name"] for entry in model_list
        }:
            model_list.append(
                {"model_name": model_name, "litellm_params": dict(params)}
            )

    # Responses-API bridge entries. A responses-only client (e.g. the codex CLI,
    # which dropped the chat wire) hits /v1/responses, but a chat-only OpenAI-
    # compatible backend (deepseek, vllm) has no /responses endpoint — LiteLLM's
    # native responses path then 404s the model. Register the same model under an
    # ``openai/chat_completions/<name>`` model_name whose UPSTREAM carries that
    # prefix; LiteLLM strips it and bridges responses→chat_completions
    # (use_chat_completions_api), so /v1/responses reaches the chat backend. Only
    # for openai/-prefixed upstreams (chat-only); native-responses providers are
    # untouched, and the bridge names are never sent on the /chat route.
    upstream = str(params.get("model", ""))
    if upstream.startswith("openai/") and bare_requested:
        bridge_params = {**params, "model": f"openai/chat_completions/{bare_requested}"}
        existing = {entry["model_name"] for entry in model_list}
        # Non-slashed bridge model_names (the codex CLI mis-parses a slashed model
        # id and then sends no request); the prefix that triggers the bridge lives
        # on the UPSTREAM (bridge_params["model"]), not the client-facing name.
        for name in (route.model_alias, bare_requested):
            bridge_name = f"{name}-responses-bridge"
            if bridge_name not in existing:
                existing.add(bridge_name)
                model_list.append(
                    {"model_name": bridge_name, "litellm_params": dict(bridge_params)}
                )
    return {
        "model_list": model_list,
        "general_settings": {"master_key": master_key},
        "router_settings": {
            # BenchFlow owns task-level retry classification. Keep LiteLLM from
            # multiplying deterministic provider rejects into proxy-local retry
            # storms or deployment cooldown fast-fails (#830).
            "num_retries": 0,
            "disable_cooldowns": True,
        },
        "litellm_settings": {
            "callbacks": [f"{callback_module}.proxy_handler_instance"],
            "drop_params": True,
            "set_verbose": False,
            # Force the anthropic /v1/messages bridge onto /chat/completions.
            # LiteLLM routes openai/-prefixed upstreams (e.g. the vllm
            # provider) through its Responses-API adapter, whose *streaming*
            # path never fires the success callback -- so streaming
            # claude-agent-acp rollouts produced no llm_trajectory.jsonl
            # (#833). /chat/completions logs success correctly.
            "use_chat_completions_url_for_anthropic_messages": True,
        },
    }
