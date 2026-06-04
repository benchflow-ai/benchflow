"""Agent environment variable resolution.

Pure helpers that build the env-var dict handed to the agent process. No I/O
beyond filesystem reads (Vertex ADC discovery, subscription-auth file probe).
No state. Every function here is callable from a test in isolation.

Owns:
    - Fallback inheritance of well-known agent env vars from .env
    - Vertex AI ADC injection for google-vertex/* models
    - Provider detection (BENCHFLOW_PROVIDER_*) and env_mapping translation
    - Subscription auth detection (host login files substituting for API keys)
    - The full resolve_agent_env pipeline that runs all of the above

Does not own:
    - Writing credential files into the container — see _credentials.py
    - Setting model via ACP session/set_model — see _acp_run.py
"""

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from benchflow._dotenv import load_dotenv_env
from benchflow.agents.codex_config import apply_codex_provider_config
from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)

_AUTH_CONTEXT_GROUPS = (
    frozenset(
        {
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_OAUTH_TOKEN",
        }
    ),
    frozenset({"GEMINI_API_KEY", "GOOGLE_API_KEY"}),
    frozenset({"OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"}),
)
_EXPLICIT_AGENT_NATIVE_BRIDGE_KEYS = frozenset({"LLM_API_KEY"})
_BEDROCK_PROVIDER_PLACEHOLDER_API_KEY = "benchflow-litellm"
_CODEX_API_KEY_ENV = "CODEX_API_KEY"
_CODEX_ACCESS_TOKEN_ENV = "CODEX_ACCESS_TOKEN"
_CODEX_AUTH_JSON_ENV = "CODEX_AUTH_JSON"
_CLAUDE_CODE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_CLAUDE_OAUTH_TOKEN_ENV = "CLAUDE_OAUTH_TOKEN"
_CUSTOM_OPENAI_ENDPOINT_KEYS = frozenset(
    {"BENCHFLOW_PROVIDER_BASE_URL", "OPENAI_BASE_URL"}
)
_CANONICAL_OPENAI_URL = "https://api.openai.com/v1"
_GENERIC_PROVIDER_OVERRIDE_KEYS = frozenset(
    {
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    }
)
_AZURE_RESOURCE_ENV = "AZURE_RESOURCE"
_AZURE_ENDPOINT_ENV = "AZURE_API_ENDPOINT"
_AZURE_HOST_SUFFIXES = (".openai.azure.com", ".services.ai.azure.com")


def _derive_azure_resource(agent_env: dict[str, str]) -> None:
    """Populate AZURE_RESOURCE from AZURE_API_ENDPOINT when not already set."""
    if agent_env.get(_AZURE_RESOURCE_ENV):
        return
    endpoint = agent_env.get(_AZURE_ENDPOINT_ENV, "").strip()
    if not endpoint:
        return
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    parsed = urlparse(endpoint)
    host = (parsed.netloc or parsed.path.split("/", 1)[0]).split("@")[-1]
    host = host.split(":", 1)[0].lower()
    for suffix in _AZURE_HOST_SUFFIXES:
        if host.endswith(suffix):
            resource = host[: -len(suffix)]
            if resource:
                agent_env[_AZURE_RESOURCE_ENV] = resource
            return


def _missing_provider_base_url_message(
    *,
    provider_name: str,
    model: str,
    required_envs: list[str],
) -> str:
    """Return a user-facing error when a provider URL template cannot resolve."""
    required = ", ".join(required_envs)
    if provider_name.startswith("azure-foundry-"):
        return (
            f"Azure AI Foundry model {model!r} requires {_AZURE_RESOURCE_ENV} "
            f"or {_AZURE_ENDPOINT_ENV} to build the provider base URL. "
            f"Export {_AZURE_ENDPOINT_ENV}=https://<resource>.openai.azure.com/ "
            f"or pass --agent-env {_AZURE_RESOURCE_ENV}=<resource>."
        )
    return (
        f"Provider {provider_name!r} for model {model!r} requires {required} "
        "to build the provider base URL."
    )


def _provider_supports_agent_protocol(provider, agent_protocol: str) -> bool:
    """Return True when a registered provider can serve the agent protocol."""
    if not agent_protocol:
        return True
    if agent_protocol in provider.all_endpoints:
        return True
    # Providers like vllm have no canonical URL; the caller supplies both URL
    # and semantics at runtime, so preserve their existing flexible behavior.
    return not provider.base_url


def _unsupported_provider_protocol_message(
    *,
    agent: str,
    agent_protocol: str,
    provider_name: str,
    model: str,
    supported_protocols: list[str],
) -> str:
    supported = ", ".join(supported_protocols)
    return (
        f"Agent {agent!r} requires provider protocol {agent_protocol!r}, but "
        f"provider {provider_name!r} for model {model!r} only supports "
        f"{supported}. Use a provider prefix that matches the agent protocol."
    )


def _normalize_openhands_model(model: str) -> str:
    """Translate benchflow model IDs to OpenHands/LiteLLM model IDs.

    OpenHands expects provider-qualified model names for some providers even
    when benchflow uses bare model IDs or its own provider prefixes.
    """
    from benchflow.agents.providers import find_provider, strip_provider_prefix
    from benchflow.agents.registry import is_vertex_model

    if model.startswith(("gemini/", "vertex_ai/", "openhands/")):
        return model
    if model.startswith("google/gemini"):
        return f"gemini/{model.split('/', 1)[1]}"
    stripped = strip_provider_prefix(model)
    if model.startswith("aws-bedrock/") and stripped.startswith(
        ("anthropic.", "us.anthropic.", "global.anthropic.")
    ):
        return f"anthropic/{stripped}"
    lower = model.lower()
    if is_vertex_model(model) and "gemini" in lower:
        return f"vertex_ai/{stripped}"
    if "gemini" in lower:
        return f"gemini/{stripped}"
    provider = find_provider(model)
    if provider is not None:
        _, cfg = provider
        if cfg.api_protocol == "openai-completions":
            return f"openai/{stripped}"
    return stripped


def auto_inherit_env(
    agent_env: dict[str, str],
    *,
    source_env: dict[str, str] | None = None,
) -> None:
    """Copy well-known agent env vars from a source mapping into agent_env."""
    from benchflow.agents.providers import PROVIDERS

    source = source_env if source_env is not None else os.environ
    explicit_keys = set(agent_env)
    keys = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "AWS_REGION_NAME",
        _CLAUDE_CODE_OAUTH_TOKEN_ENV,
        _CLAUDE_OAUTH_TOKEN_ENV,
        "CODEX_ACCESS_TOKEN",
        "CODEX_API_KEY",
        "CODEX_AUTH_JSON",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_BEDROCK_THINKING_EFFORT",
        # AZURE_API_KEY / AZURE_RESOURCE are picked up automatically below via
        # cfg.auth_env / cfg.url_params; only AZURE_API_ENDPOINT is listed
        # explicitly because it is a user-facing convenience input, not a
        # provider-config field.
        _AZURE_ENDPOINT_ENV,
        "AZURE_API_VERSION",
    }
    for cfg in PROVIDERS.values():
        if cfg.auth_env:
            keys.add(cfg.auth_env)
        for env_var in cfg.url_params.values():
            keys.add(env_var)
    for key in keys:
        value = source.get(key)
        # An exported-but-blank var ("export X=" or "export X=' '") is
        # effectively unset; copying it can shadow a real value resolved
        # downstream (e.g. a blank BENCHFLOW_PROVIDER_BASE_URL blocks
        # provider URL resolution).
        if value and value.strip():
            agent_env.setdefault(key, value)
    # Mirror GEMINI_API_KEY / GOOGLE_API_KEY in both directions (#342).
    if "GEMINI_API_KEY" in agent_env and "GOOGLE_API_KEY" not in agent_env:
        agent_env["GOOGLE_API_KEY"] = agent_env["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in agent_env and "GEMINI_API_KEY" not in agent_env:
        agent_env["GEMINI_API_KEY"] = agent_env["GOOGLE_API_KEY"]
    # Mirror GEMINI_API_KEY as GOOGLE_GENERATIVE_AI_API_KEY (opencode/models.dev convention)
    if (
        "GEMINI_API_KEY" in agent_env
        and "GOOGLE_GENERATIVE_AI_API_KEY" not in agent_env
    ):
        agent_env["GOOGLE_GENERATIVE_AI_API_KEY"] = agent_env["GEMINI_API_KEY"]
    if (
        "AWS_DEFAULT_REGION" in explicit_keys and "AWS_REGION" not in explicit_keys
    ) or ("AWS_DEFAULT_REGION" in agent_env and "AWS_REGION" not in agent_env):
        agent_env["AWS_REGION"] = agent_env["AWS_DEFAULT_REGION"]
    if (
        "AWS_REGION" in explicit_keys and "AWS_DEFAULT_REGION" not in explicit_keys
    ) or ("AWS_REGION" in agent_env and "AWS_DEFAULT_REGION" not in agent_env):
        agent_env["AWS_DEFAULT_REGION"] = agent_env["AWS_REGION"]
    if "AWS_REGION" in agent_env and "AWS_REGION_NAME" not in agent_env:
        agent_env["AWS_REGION_NAME"] = agent_env["AWS_REGION"]
    if (
        _CLAUDE_OAUTH_TOKEN_ENV in agent_env
        and _CLAUDE_CODE_OAUTH_TOKEN_ENV not in agent_env
    ):
        agent_env[_CLAUDE_CODE_OAUTH_TOKEN_ENV] = agent_env[_CLAUDE_OAUTH_TOKEN_ENV]
    _derive_azure_resource(agent_env)
    # CLAUDE_CODE_OAUTH_TOKEN is a separate auth path — Claude CLI reads it
    # directly. Don't map to ANTHROPIC_API_KEY (different auth mechanism).


def _is_codex_native_openai_context(
    agent: str,
    model: str | None,
    required_key: str | None,
) -> bool:
    """True when Codex can use its own OpenAI auth mechanisms directly.

    Native auth covers bare model IDs (``gpt-*``) and the first-party
    ``openai/`` provider prefix pointing at ``api.openai.com``. Custom or
    proxy providers (``vllm/``, ``us-openai/``, etc.) must supply
    ``OPENAI_API_KEY`` explicitly — subscription/access-token auth does not
    apply to them.
    """
    if agent != "codex-acp" or required_key != "OPENAI_API_KEY":
        return False
    if model is None:
        return True

    from benchflow.agents.providers import find_provider

    result = find_provider(model)
    if result is None:
        return True
    name, cfg = result
    return name == "openai" and cfg.base_url == _CANONICAL_OPENAI_URL


def _has_custom_openai_endpoint(agent_env: dict[str, str]) -> bool:
    """True when Codex is being pointed at an OpenAI-compatible non-OpenAI URL.

    The first-party ``openai/`` provider populates these keys with the
    canonical ``api.openai.com`` URL — that is the native endpoint, not a
    custom proxy, so it must not disqualify subscription/access-token auth.
    """
    for key in _CUSTOM_OPENAI_ENDPOINT_KEYS:
        value = agent_env.get(key)
        if value and value.rstrip("/") != _CANONICAL_OPENAI_URL:
            return True
    return False


def _can_use_codex_subscription_auth(
    agent: str,
    model: str | None,
    required_key: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Codex subscription auth is only valid for the native OpenAI endpoint."""
    return _is_codex_native_openai_context(
        agent,
        model,
        required_key,
    ) and not _has_custom_openai_endpoint(agent_env)


def _can_use_subscription_auth(
    agent: str,
    model: str | None,
    required_key: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Return True when host subscription files can satisfy provider auth."""
    if agent == "codex-acp" and required_key == "OPENAI_API_KEY":
        return _can_use_codex_subscription_auth(
            agent,
            model,
            required_key,
            agent_env,
        )
    return True


def _normalize_codex_auth_env(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
) -> None:
    """Bridge Codex's API-key alias to the auth file writer.

    codex-acp advertises both CODEX_API_KEY and OPENAI_API_KEY, while
    BenchFlow writes ~/.codex/auth.json from OPENAI_API_KEY before launching
    ACP. Keep CODEX_ACCESS_TOKEN separate: it is a subscription/access-token
    path that Codex reads directly from the process environment.
    """
    if not _is_codex_native_openai_context(agent, model, "OPENAI_API_KEY"):
        return
    if "OPENAI_API_KEY" not in agent_env and _CODEX_API_KEY_ENV in agent_env:
        agent_env["OPENAI_API_KEY"] = agent_env[_CODEX_API_KEY_ENV]


def _has_codex_access_token_auth(
    agent: str,
    model: str | None,
    required_key: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Return True when Codex's subscription access token satisfies OpenAI auth."""
    return _can_use_codex_subscription_auth(
        agent,
        model,
        required_key,
        agent_env,
    ) and bool(agent_env.get(_CODEX_ACCESS_TOKEN_ENV))


def _has_codex_auth_json_auth(
    agent: str,
    model: str | None,
    required_key: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Return True when inline Codex auth.json can satisfy native OpenAI auth."""
    return _can_use_codex_subscription_auth(
        agent,
        model,
        required_key,
        agent_env,
    ) and bool(agent_env.get(_CODEX_AUTH_JSON_ENV))


def inject_vertex_credentials(agent_env: dict[str, str], model: str) -> None:
    """Inject ADC credentials and defaults for Vertex AI models."""
    from benchflow.agents.registry import is_vertex_model

    if not is_vertex_model(model):
        return
    adc_path = Path.home() / ".config/gcloud/application_default_credentials.json"
    if not adc_path.exists():
        raise ValueError(
            f"Vertex AI model {model!r} requires ADC credentials. "
            f"Run: gcloud auth application-default login"
        )
    agent_env.setdefault("GOOGLE_APPLICATION_CREDENTIALS_JSON", adc_path.read_text())
    agent_env.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    if "GOOGLE_CLOUD_PROJECT" not in agent_env:
        raise ValueError(
            f"GOOGLE_CLOUD_PROJECT required for Vertex AI model {model!r}. "
            f"Export it or pass via --agent-env GOOGLE_CLOUD_PROJECT=<project>"
        )


def resolve_provider_env(
    agent_env: dict[str, str],
    model: str,
    agent: str,
) -> None:
    """Detect provider for model, inject BENCHFLOW_PROVIDER_* and env_mapping."""
    from benchflow.agents.providers import (
        find_provider,
        resolve_base_url,
        strip_provider_prefix,
    )

    agent_env.setdefault("BENCHFLOW_PROVIDER_MODEL", strip_provider_prefix(model))
    agent_cfg = AGENTS.get(agent)
    # Agent-declared protocol takes precedence over provider's primary so
    # multi-endpoint providers (e.g. zai) route to the right URL.
    agent_protocol = agent_cfg.api_protocol if agent_cfg else ""
    _prov = find_provider(model)
    if _prov:
        _prov_name, _prov_cfg = _prov
        agent_env.setdefault("BENCHFLOW_PROVIDER_NAME", _prov_name)
        if not _provider_supports_agent_protocol(_prov_cfg, agent_protocol):
            raise ValueError(
                _unsupported_provider_protocol_message(
                    agent=agent,
                    agent_protocol=agent_protocol,
                    provider_name=_prov_name,
                    model=model,
                    supported_protocols=sorted(_prov_cfg.all_endpoints),
                )
            )
        if "BENCHFLOW_PROVIDER_BASE_URL" not in agent_env:
            try:
                base_url = resolve_base_url(
                    _prov_cfg, agent_env, protocol=agent_protocol or None
                )
            except KeyError as exc:
                raise ValueError(
                    _missing_provider_base_url_message(
                        provider_name=_prov_name,
                        model=model,
                        required_envs=list(_prov_cfg.url_params.values()),
                    )
                ) from exc
            agent_env.setdefault(
                "BENCHFLOW_PROVIDER_BASE_URL",
                base_url,
            )
        agent_env.setdefault(
            "BENCHFLOW_PROVIDER_PROTOCOL",
            agent_protocol or _prov_cfg.api_protocol,
        )
        if _prov_cfg.models:
            agent_env.setdefault(
                "BENCHFLOW_PROVIDER_MODELS", json.dumps(_prov_cfg.models)
            )
        if _prov_cfg.auth_type == "api_key" and _prov_cfg.auth_env:
            _key = agent_env.get(_prov_cfg.auth_env, "")
            if _key:
                agent_env.setdefault("BENCHFLOW_PROVIDER_API_KEY", _key)
        elif _prov_cfg.auth_type == "aws":
            agent_env.setdefault(
                "BENCHFLOW_PROVIDER_API_KEY",
                _BEDROCK_PROVIDER_PLACEHOLDER_API_KEY,
            )
    else:
        # No registered provider prefix — bridge the model's well-known API key
        # to BENCHFLOW_PROVIDER_API_KEY so env_mapping can translate it to
        # agent-native vars (e.g. GEMINI_API_KEY → LLM_API_KEY for openhands).
        from benchflow.agents.registry import infer_env_key_for_model

        _inferred = infer_env_key_for_model(model)
        if _inferred and _inferred in agent_env:
            agent_env.setdefault("BENCHFLOW_PROVIDER_API_KEY", agent_env[_inferred])
    # Apply agent env_mapping: translate BENCHFLOW_PROVIDER_* → agent-native vars
    if agent_cfg and agent_cfg.env_mapping:
        for src, dst in agent_cfg.env_mapping.items():
            if src in agent_env:
                agent_env.setdefault(dst, agent_env[src])
    if agent == "openhands":
        agent_env.setdefault("LLM_MODEL", _normalize_openhands_model(model))


def check_subscription_auth(agent: str, required_key: str) -> bool:
    """Return True if host subscription auth can substitute for required_key."""
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg or not agent_cfg.subscription_auth:
        return False
    sa = agent_cfg.subscription_auth
    if sa.replaces_env != required_key:
        return False
    return Path(sa.detect_file).expanduser().is_file()


def validate_aws_bedrock_env(agent_env: dict[str, str], model: str) -> None:
    """Validate Bedrock API-key auth and normalize region aliases."""
    token = agent_env.get("AWS_BEARER_TOKEN_BEDROCK")
    region = agent_env.get("AWS_REGION") or agent_env.get("AWS_DEFAULT_REGION")
    if not token:
        raise ValueError(
            f"AWS_BEARER_TOKEN_BEDROCK required for Bedrock model {model!r} but not set. "
            "Export it or pass via agent_env."
        )
    if not region:
        raise ValueError(
            f"AWS_REGION or AWS_DEFAULT_REGION required for Bedrock model {model!r} "
            "but not set. Export one of them or pass via agent_env."
        )
    agent_env.setdefault("AWS_REGION", region)
    agent_env.setdefault("AWS_DEFAULT_REGION", region)


def _shares_auth_context(required_key: str | None, candidate_key: str | None) -> bool:
    """True when both keys represent the same provider auth context."""
    if not required_key or not candidate_key:
        return False
    if required_key == candidate_key:
        return True
    return any(
        required_key in group and candidate_key in group
        for group in _AUTH_CONTEXT_GROUPS
    )


def _configure_codex_custom_provider(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
) -> None:
    """Expose BenchFlow provider routing through Codex's native config model."""
    if agent != "codex-acp" or not model:
        return

    base_url = agent_env.get("BENCHFLOW_PROVIDER_BASE_URL") or agent_env.get(
        "OPENAI_BASE_URL"
    )
    provider_model = agent_env.get("BENCHFLOW_PROVIDER_MODEL")
    if not base_url or not provider_model:
        return

    apply_codex_provider_config(
        agent_env,
        base_url=base_url,
        model=provider_model,
        provider_name=agent_env.get("BENCHFLOW_PROVIDER_NAME", "openai-compatible"),
        strict=True,
    )


def _drop_inherited_generic_provider_overrides(
    agent_env: dict[str, str],
    *,
    model: str | None,
    explicit_agent_env_keys: set[str],
) -> None:
    """Let registered providers use their own endpoint/key over host defaults."""
    if not model:
        return

    from benchflow.agents.providers import find_provider
    from benchflow.agents.registry import infer_env_key_for_model

    provider = find_provider(model)
    if provider is None:
        if infer_env_key_for_model(model) is None:
            return
    else:
        _, provider_cfg = provider
        # Providers with an empty base URL (for example vllm/) are explicitly
        # user-supplied endpoints, so inherited BENCHFLOW_PROVIDER_* is the normal
        # configuration path. Bedrock is the exception: LiteLLM supplies its
        # runtime URL later, so stale generic provider vars must not shadow it.
        if not provider_cfg.base_url and provider_cfg.auth_type != "aws":
            return
    for key in _GENERIC_PROVIDER_OVERRIDE_KEYS - explicit_agent_env_keys:
        agent_env.pop(key, None)


def resolve_agent_env(
    agent: str,
    model: str | None,
    agent_env: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve agent environment from explicit overrides, then .env defaults."""
    agent_env = dict(agent_env or {})
    explicit_agent_env_keys = set(agent_env)
    # Inherit from .env file first, then from os.environ as fallback.
    # Both sources use setdefault so explicit agent_env keys take priority.
    auto_inherit_env(agent_env, source_env=load_dotenv_env())
    auto_inherit_env(agent_env)
    _normalize_codex_auth_env(agent, model, agent_env)
    pre_provider_env = dict(agent_env)
    agent_cfg = AGENTS.get(agent)
    # Oracle runs solve.sh and never calls an LLM — model env vars and
    # API-key validation are skipped even if a caller forwards a model.
    if model and agent != "oracle":
        inject_vertex_credentials(agent_env, model)
        _drop_inherited_generic_provider_overrides(
            agent_env,
            model=model,
            explicit_agent_env_keys=explicit_agent_env_keys,
        )
        resolve_provider_env(agent_env, model, agent)
        from benchflow.agents.providers import find_provider

        provider = find_provider(model)
        if provider is not None:
            _, provider_cfg = provider
            if provider_cfg.auth_type == "aws":
                validate_aws_bedrock_env(agent_env, model)
        if agent_cfg and agent_cfg.env_mapping:
            for src, dst in agent_cfg.env_mapping.items():
                if src in agent_env and dst not in explicit_agent_env_keys:
                    # Provider resolution must override unrelated fallback
                    # vars auto-inherited from the source env, but preserve
                    # explicit agent_env overrides supplied by the caller.
                    agent_env[dst] = agent_env[src]
        # Validate required API key for the chosen model
        from benchflow.agents.registry import infer_env_key_for_model

        required_key = infer_env_key_for_model(model)
        mapped_provider_key = (
            agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_API_KEY")
            if agent_cfg
            else None
        )
        has_agent_native_bridge_key = bool(
            mapped_provider_key
            and pre_provider_env.get(mapped_provider_key)
            and (
                _shares_auth_context(required_key, mapped_provider_key)
                or (
                    mapped_provider_key in _EXPLICIT_AGENT_NATIVE_BRIDGE_KEYS
                    and mapped_provider_key in explicit_agent_env_keys
                )
            )
        )
        if has_agent_native_bridge_key and mapped_provider_key is not None:
            # Only pre-existing same-provider aliases or explicit generic bridge
            # keys can satisfy provider auth. Values synthesized by env_mapping
            # or inherited from another provider context must not bypass the
            # model's required credential.
            agent_env["BENCHFLOW_PROVIDER_API_KEY"] = pre_provider_env[
                mapped_provider_key
            ]
        has_oauth = any(
            key in agent_env and _shares_auth_context(required_key, key)
            for key in (
                _CLAUDE_CODE_OAUTH_TOKEN_ENV,
                _CLAUDE_OAUTH_TOKEN_ENV,
                "ANTHROPIC_AUTH_TOKEN",
            )
        )
        has_codex_access_token = _has_codex_access_token_auth(
            agent,
            model,
            required_key,
            agent_env,
        )
        has_codex_auth_json = _has_codex_auth_json_auth(
            agent,
            model,
            required_key,
            agent_env,
        )
        if (
            required_key
            and required_key not in agent_env
            and not has_oauth
            and not has_agent_native_bridge_key
            and not has_codex_access_token
            and not has_codex_auth_json
        ):
            if _can_use_subscription_auth(
                agent,
                model,
                required_key,
                agent_env,
            ) and check_subscription_auth(agent, required_key):
                agent_env["_BENCHFLOW_SUBSCRIPTION_AUTH"] = "1"
                logger.info(
                    "Using host subscription auth (no %s set)",
                    required_key,
                )
            else:
                raise ValueError(
                    f"{required_key} required for model {model!r} but not set. "
                    "Pass it explicitly (for example via --agent-env/agent_env) "
                    "or define it in .env."
                )
        _configure_codex_custom_provider(agent, model, agent_env)
    else:
        # No model specified — still check subscription auth for required env vars
        if agent_cfg:
            for req_key in agent_cfg.requires_env:
                if (
                    req_key not in agent_env
                    and not _has_codex_access_token_auth(
                        agent,
                        model,
                        req_key,
                        agent_env,
                    )
                    and not _has_codex_auth_json_auth(
                        agent,
                        model,
                        req_key,
                        agent_env,
                    )
                    and _can_use_subscription_auth(agent, model, req_key, agent_env)
                    and check_subscription_auth(agent, req_key)
                ):
                    agent_env["_BENCHFLOW_SUBSCRIPTION_AUTH"] = "1"
                    logger.info(
                        "Using host subscription auth (no %s set)",
                        req_key,
                    )
    # Increase output token limit to avoid truncation errors
    agent_env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "128000")
    # Disable telemetry/non-essential traffic in container
    agent_env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    return agent_env
