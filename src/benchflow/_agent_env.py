"""Agent environment variable resolution.

Pure helpers that build the env-var dict handed to the agent process. No I/O
beyond filesystem reads (Vertex ADC discovery, subscription-auth file probe).
No state. Every function here is callable from a test in isolation.

Owns:
    - Auto-inheritance of well-known API keys from host os.environ
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

from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)

_AUTH_CONTEXT_GROUPS = (
    frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}),
    frozenset({"GEMINI_API_KEY", "GOOGLE_API_KEY"}),
)
_EXPLICIT_AGENT_NATIVE_BRIDGE_KEYS = frozenset({"LLM_API_KEY"})


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


def auto_inherit_env(agent_env: dict[str, str]) -> None:
    """Copy well-known API keys from host os.environ into agent_env."""
    from benchflow.agents.providers import PROVIDERS

    keys = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    }
    for cfg in PROVIDERS.values():
        if cfg.auth_env:
            keys.add(cfg.auth_env)
        for env_var in cfg.url_params.values():
            keys.add(env_var)
    for key in keys:
        if key in os.environ:
            agent_env.setdefault(key, os.environ[key])
    # Mirror GEMINI_API_KEY as GOOGLE_API_KEY (some agents expect one or the other)
    if "GEMINI_API_KEY" in agent_env and "GOOGLE_API_KEY" not in agent_env:
        agent_env["GOOGLE_API_KEY"] = agent_env["GEMINI_API_KEY"]
    # Mirror GEMINI_API_KEY as GOOGLE_GENERATIVE_AI_API_KEY (opencode/models.dev convention)
    if (
        "GEMINI_API_KEY" in agent_env
        and "GOOGLE_GENERATIVE_AI_API_KEY" not in agent_env
    ):
        agent_env["GOOGLE_GENERATIVE_AI_API_KEY"] = agent_env["GEMINI_API_KEY"]
    # CLAUDE_CODE_OAUTH_TOKEN is a separate auth path — Claude CLI reads it
    # directly. Don't map to ANTHROPIC_API_KEY (different auth mechanism).


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
            f"Export it or pass via --ae GOOGLE_CLOUD_PROJECT=<project>"
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
    """Resolve agent environment: auto-inherit keys, provider vars, env_mapping."""
    agent_env = dict(agent_env or {})
    explicit_agent_env_keys = set(agent_env)
    auto_inherit_env(agent_env)
    pre_provider_env = dict(agent_env)
    agent_cfg = AGENTS.get(agent)
    # Oracle runs solve.sh and never calls an LLM — model env vars and
    # API-key validation are skipped even if a caller forwards a model.
    if model and agent != "oracle":
        inject_vertex_credentials(agent_env, model)
        resolve_provider_env(agent_env, model, agent)
        if agent_cfg and agent_cfg.env_mapping:
            for src, dst in agent_cfg.env_mapping.items():
                if src in agent_env and dst not in explicit_agent_env_keys:
                    # Provider resolution must override unrelated host-native
                    # vars auto-inherited from the environment, but preserve
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
        if has_agent_native_bridge_key:
            # Only pre-existing same-provider aliases or explicit generic bridge
            # keys can satisfy provider auth. Values synthesized by env_mapping
            # or inherited from another provider context must not bypass the
            # model's required credential.
            agent_env.setdefault(
                "BENCHFLOW_PROVIDER_API_KEY",
                pre_provider_env[mapped_provider_key],
            )
        has_oauth = any(
            key in agent_env and _shares_auth_context(required_key, key)
            for key in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN")
        )
        if (
            required_key
            and required_key not in agent_env
            and not has_oauth
            and not has_agent_native_bridge_key
        ):
            if check_subscription_auth(agent, required_key):
                agent_env["_BENCHFLOW_SUBSCRIPTION_AUTH"] = "1"
                logger.info(
                    "Using host subscription auth (no %s set)",
                    required_key,
                )
            else:
                raise ValueError(
                    f"{required_key} required for model {model!r} but not set. "
                    f"Export it, pass via agent_env, or log in with the "
                    f"agent CLI (e.g. claude login, codex --login)."
                )
        elif (
            required_key
            and required_key in agent_env
            and check_subscription_auth(agent, required_key)
        ):
            logger.warning(
                "%s is set (possibly inherited from host env) AND "
                "subscription auth credentials exist — the env var takes "
                "precedence. If the key is stale, unset it: "
                "env -u %s <command>",
                required_key,
                required_key,
            )
    else:
        # No model specified — still check subscription auth for required env vars
        if agent_cfg:
            for req_key in agent_cfg.requires_env:
                if req_key not in agent_env and check_subscription_auth(agent, req_key):
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
