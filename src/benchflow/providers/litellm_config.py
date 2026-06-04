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
LITELLM_LOG_PATH_ENV = "BENCHFLOW_LITELLM_LOG_PATH"
LITELLM_PROVIDER_NAME = "litellm"
_BEDROCK_ADAPTIVE_THINKING_RE = re.compile(
    r"claude-(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)", re.IGNORECASE
)
_BEDROCK_THINKING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class LiteLLMRoute:
    """Resolved LiteLLM model route for one BenchFlow model ID."""

    requested_model: str
    model_alias: str
    upstream_model: str
    provider_name: str
    litellm_params: dict[str, str | int | float | bool]
    required_env: tuple[str, ...] = ()
    extra_env: dict[str, str] | None = None

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

    if protocol == "anthropic-messages":
        upstream = f"anthropic/{bare}"
    else:
        upstream = f"openai/{bare}"
    params = {"model": upstream}
    if api_base:
        params["api_base"] = api_base
    api_key_ref = _registered_api_key_ref(provider_cfg)
    if api_key_ref:
        params["api_key"] = api_key_ref
        if provider_cfg.auth_env:
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
    key = required[0] if required else None
    if key:
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
    model_list: list[dict[str, object]] = [
        {
            "model_name": route.model_alias,
            "litellm_params": dict(route.litellm_params),
        }
    ]
    openai_alias = f"openai/{route.model_alias}"
    model_list.append(
        {
            "model_name": openai_alias,
            "litellm_params": dict(route.litellm_params),
        }
    )
    return {
        "model_list": model_list,
        "general_settings": {"master_key": master_key},
        "litellm_settings": {
            "callbacks": [f"{callback_module}.proxy_handler_instance"],
            "drop_params": True,
            "set_verbose": False,
        },
    }
