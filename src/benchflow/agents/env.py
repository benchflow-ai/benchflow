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

import contextlib
import json
import logging
import os
from pathlib import Path

from benchflow._dotenv import load_dotenv_env
from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)

_AUTH_CONTEXT_GROUPS = (
    frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}),
    frozenset({"GEMINI_API_KEY", "GOOGLE_API_KEY"}),
    frozenset({"OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"}),
)
_EXPLICIT_AGENT_NATIVE_BRIDGE_KEYS = frozenset({"LLM_API_KEY"})
_BEDROCK_PROXY_PLACEHOLDER_API_KEY = "bedrock-proxy"
_CODEX_API_KEY_ENV = "CODEX_API_KEY"
_CODEX_ACCESS_TOKEN_ENV = "CODEX_ACCESS_TOKEN"
_CUSTOM_OPENAI_ENDPOINT_KEYS = frozenset(
    {"BENCHFLOW_PROVIDER_BASE_URL", "OPENAI_BASE_URL"}
)


def _normalize_openhands_model(model: str) -> str:
    """Translate benchflow model IDs to OpenHands/LiteLLM model IDs.

    OpenHands expects provider-qualified model names for some providers even
    when benchflow uses bare model IDs or its own provider prefixes.
    """
    from benchflow.agents.providers import strip_provider_prefix
    from benchflow.agents.registry import is_vertex_model

    if model.startswith(("gemini/", "vertex_ai/", "openhands/")):
        return model
    stripped = strip_provider_prefix(model)
    lower = model.lower()
    if is_vertex_model(model) and "gemini" in lower:
        return f"vertex_ai/{stripped}"
    if "gemini" in lower:
        return f"gemini/{stripped}"
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
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_ACCESS_TOKEN",
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "BENCHFLOW_PROVIDER_PROMPT_CACHE_RETENTION",
    }
    for cfg in PROVIDERS.values():
        if cfg.auth_env:
            keys.add(cfg.auth_env)
        for env_var in cfg.url_params.values():
            keys.add(env_var)
    for key in keys:
        if key in source:
            agent_env.setdefault(key, source[key])
    # Mirror GEMINI_API_KEY as GOOGLE_API_KEY (some agents expect one or the other)
    if "GEMINI_API_KEY" in agent_env and "GOOGLE_API_KEY" not in agent_env:
        agent_env["GOOGLE_API_KEY"] = agent_env["GEMINI_API_KEY"]
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
    # CLAUDE_CODE_OAUTH_TOKEN is a separate auth path — Claude CLI reads it
    # directly. Don't map to ANTHROPIC_API_KEY (different auth mechanism).


def _is_codex_native_openai_context(
    agent: str,
    model: str | None,
    required_key: str | None,
) -> bool:
    """True when Codex can use its own OpenAI auth mechanisms directly."""
    if agent != "codex-acp" or required_key != "OPENAI_API_KEY":
        return False
    if model is None:
        return True

    from benchflow.agents.providers import find_provider

    return find_provider(model) is None


def _has_custom_openai_endpoint(agent_env: dict[str, str]) -> bool:
    """True when Codex is being pointed at an OpenAI-compatible non-OpenAI URL."""
    return any(agent_env.get(key) for key in _CUSTOM_OPENAI_ENDPOINT_KEYS)


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
        # URL params missing — will fail later with clear error
        with contextlib.suppress(KeyError):
            agent_env.setdefault(
                "BENCHFLOW_PROVIDER_BASE_URL",
                resolve_base_url(_prov_cfg, agent_env, protocol=agent_protocol or None),
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
                _BEDROCK_PROXY_PLACEHOLDER_API_KEY,
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
            for key in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN")
        )
        has_codex_access_token = _has_codex_access_token_auth(
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
