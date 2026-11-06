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
from benchflow.agents.login_dirs import login_display, login_path
from benchflow.agents.muse_config import validate_muse_model
from benchflow.agents.registry import AGENTS, infer_env_key_for_model, is_vertex_model
from benchflow.errors import MissingCredentialError

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
_MUSE_AGENT = "muse-code"
_MUSE_AUTH_JSON_ENV = "MUSE_AUTH_JSON"
# The shim's own settings (muse_acp_shim.QUOTA_WAIT_ENV, QUOTA_SLACK_ENV): no credentials.
_MUSE_SETTINGS_ENV = ("BENCHFLOW_MUSE_QUOTA_WAIT_SEC", "BENCHFLOW_MUSE_QUOTA_SLACK_SEC")
# muse_acp_shim.PROMPT_BUDGET_ENV: set by the rollout, never inherited from the host.
_MUSE_PROMPT_BUDGET_ENV = "BENCHFLOW_MUSE_PROMPT_BUDGET_SEC"
_CLAUDE_CODE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_CLAUDE_OAUTH_TOKEN_ENV = "CLAUDE_OAUTH_TOKEN"
_SUBSCRIPTION_AUTH_MARKER = "_BENCHFLOW_SUBSCRIPTION_AUTH"
# The run's credential policy (launch_env.CredentialPolicy), by variable names
# only, recorded at the end of resolution. Every placement reads it.
CREDENTIAL_POLICY_MARKER = "_BENCHFLOW_CREDENTIAL_POLICY"
# A BenchFlow alias and the name its CLI reads.
_LOGIN_ALIASES = {
    "CLAUDE_OAUTH_TOKEN": "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY": "OPENAI_API_KEY",
}
_CUSTOM_OPENAI_ENDPOINT_KEYS = frozenset(
    {"BENCHFLOW_PROVIDER_BASE_URL", "OPENAI_BASE_URL"}
)
_LITELLM_RUNTIME_MARKER_KEYS = frozenset(
    {
        "BENCHFLOW_LITELLM_MASTER_KEY",
        "BENCHFLOW_LITELLM_MODEL_ALIAS",
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV",
    }
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


def unroutable_agent_model(agent: str, model: str | None) -> str | None:
    """Why no provider route serves ``agent`` with ``model``, or None.

    The protocol check ``resolve_provider_env`` makes when a trial starts,
    made before a job exists so the pair is refused up front, as an unknown
    agent name is (REPORT B11: the refusal came inside the job, as an errored
    trial). Only a model of a registered provider is checked, and only
    against a registered agent (or its alias) that declares its protocol.
    """
    if not model:
        return None
    from benchflow.agents.providers import find_provider, find_provider_for_bare_model
    from benchflow.agents.registry import AGENT_ALIASES

    name = AGENT_ALIASES.get(agent, agent)
    agent_cfg = AGENTS.get(name)
    agent_protocol = agent_cfg.api_protocol if agent_cfg else ""
    provider = find_provider(model) or find_provider_for_bare_model(model)
    if provider is None or _provider_supports_agent_protocol(
        provider[1], agent_protocol
    ):
        return None
    provider_name, provider_cfg = provider
    return _unsupported_provider_protocol_message(
        agent=name,
        agent_protocol=agent_protocol,
        provider_name=provider_name,
        model=model,
        supported_protocols=sorted(provider_cfg.all_endpoints),
    )


def _normalize_openhands_model(model: str) -> str:
    """Translate benchflow model IDs to OpenHands/LiteLLM model IDs.

    OpenHands expects provider-qualified model names for some providers even
    when benchflow uses bare model IDs or its own provider prefixes.
    """
    from benchflow.agents.providers import (
        find_provider,
        find_provider_for_bare_model,
        strip_provider_prefix,
    )

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
    provider = find_provider(model) or find_provider_for_bare_model(model)
    if provider is not None:
        _, cfg = provider
        if cfg.api_protocol == "openai-completions":
            return f"openai/{stripped}"
    return stripped


def inheritable_env_names() -> frozenset[str]:
    """Every name auto_inherit_env copies from the host or .env.

    launch_env treats each of them as a credential unless it names it as a
    runtime setting, so the two lists cannot drift apart.
    """
    from benchflow.agents.providers import PROVIDERS

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
    return frozenset(keys)


def auto_inherit_env(
    agent_env: dict[str, str],
    *,
    source_env: dict[str, str] | None = None,
) -> None:
    """Copy well-known agent env vars from a source mapping into agent_env."""
    source = source_env if source_env is not None else os.environ
    explicit_keys = set(agent_env)
    for key in inheritable_env_names():
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
        # An unregistered provider prefix (or another bare model family)
        # does not become OpenAI merely because a Codex host login exists.
        return "/" not in model and infer_env_key_for_model(model) == "OPENAI_API_KEY"
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


def _has_provider_api_key(required_key: str | None, agent_env: dict[str, str]) -> bool:
    """Choose API-key auth before either native routing or host-file fallback.

    The generic key has not necessarily been mapped to the agent's variable
    yet. Both decisions must see it, including calls made by plan validation
    before provider resolution. Codex's alias can likewise arrive unmapped.
    """
    keys = ("BENCHFLOW_PROVIDER_API_KEY", required_key)
    if required_key == "OPENAI_API_KEY":
        keys += (_CODEX_API_KEY_ENV,)
    return any(agent_env.get(key) for key in keys if key is not None)


def _can_use_subscription_auth(
    agent: str,
    model: str | None,
    required_key: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Return True when host subscription files can satisfy provider auth."""
    # A supplied credential selects its account; discovering another host
    # login would also authorize uploading that account's complete login.
    # Use the same family/precedence table as login_label, on every fallback
    # path (native, provider and no-model), including inline Codex auth.
    keys = _LOGIN_KEYS_BY_FAMILY.get(required_key or "", ())
    if _has_provider_api_key(required_key, agent_env) or any(
        agent_env.get(key) for key in keys
    ):
        return False
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


# Agents that call Vertex themselves with the host's ADC login:
# the Gemini CLI's own Vertex backend. Every other agent on a Vertex model goes
# through the proxy, which keeps the ADC in its own storage.
_VERTEX_NATIVE_AGENTS = frozenset({"gemini"})
VERTEX_ADC_ENV = "GOOGLE_APPLICATION_CREDENTIALS_JSON"


def uses_native_vertex_login(agent: str, model: str | None) -> bool:
    """True for the one native Vertex route: Gemini CLI on a google-vertex/ model."""
    return agent in _VERTEX_NATIVE_AGENTS and bool(
        model and model.lower().startswith("google-vertex/")
    )


def uses_native_subscription_auth(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
) -> bool:
    """Return True when an agent should use CLI/subscription auth directly.

    This is the split point: API-key runs can be routed through
    LiteLLM, while subscription-auth runs stay on the native Codex/Claude ACP
    path and report usage from the agent protocol response.
    """
    if agent_env.get("BENCHFLOW_PROVIDER_NAME") == "litellm" or any(
        agent_env.get(key) for key in _LITELLM_RUNTIME_MARKER_KEYS
    ):
        return False

    if uses_native_vertex_login(agent, model):
        return True

    if agent == _MUSE_AGENT:
        # Muse Code's subscription works only through the CLI's own login
        # (Meta's terms), so a login run talks to Meta directly and reports
        # usage over ACP; there is no API key to put behind the proxy.
        return bool(agent_env.get(_MUSE_AUTH_JSON_ENV))

    if agent == "codex-acp":
        if _has_provider_api_key("OPENAI_API_KEY", agent_env):
            return False
        required_key = "OPENAI_API_KEY"
        if not _can_use_codex_subscription_auth(
            agent,
            model,
            required_key,
            agent_env,
        ):
            return False
        return (
            bool(agent_env.get(_CODEX_ACCESS_TOKEN_ENV))
            or bool(agent_env.get(_CODEX_AUTH_JSON_ENV))
            or agent_env.get(_SUBSCRIPTION_AUTH_MARKER) == "1"
            or check_subscription_auth(agent, required_key)
        )

    # Registry-driven Claude-CLI gate: any agent whose subscription_auth
    # substitutes ANTHROPIC_API_KEY runs the Claude Code CLI and can take
    # OAuth/subscription auth natively (claude-agent-acp, omnigent claude-*).
    claude_cfg = AGENTS.get(agent)
    if (
        claude_cfg is not None
        and claude_cfg.subscription_auth is not None
        and claude_cfg.subscription_auth.replaces_env == "ANTHROPIC_API_KEY"
    ):
        if _has_provider_api_key("ANTHROPIC_API_KEY", agent_env):
            return False
        if model is not None and infer_env_key_for_model(model) != "ANTHROPIC_API_KEY":
            return False
        return (
            bool(agent_env.get(_CLAUDE_CODE_OAUTH_TOKEN_ENV))
            or bool(agent_env.get(_CLAUDE_OAUTH_TOKEN_ENV))
            or bool(agent_env.get("ANTHROPIC_AUTH_TOKEN"))
            or agent_env.get(_SUBSCRIPTION_AUTH_MARKER) == "1"
            or check_subscription_auth(agent, "ANTHROPIC_API_KEY")
        )

    return False


# The next step for a missing credential, by the variable the model needs.
_LOGIN_HINTS = {
    "ANTHROPIC_API_KEY": (
        "`claude login` works too (it writes ~/.claude/.credentials.json); "
        "`bench doctor` lists the logins it finds"
    ),
    "OPENAI_API_KEY": (
        "for a ChatGPT plan run `codex login` (it writes ~/.codex/auth.json); "
        "`bench doctor` lists the logins it finds"
    ),
}
_LOGIN_LABEL_ENV = "BENCHFLOW_LOGIN_LABEL"
# The credentials each family reads, in the order its CLI prefers them (an
# API key wins over a subscription token in Claude Code and in codex-acp).
_LOGIN_KEYS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        _CLAUDE_CODE_OAUTH_TOKEN_ENV,
        _CLAUDE_OAUTH_TOKEN_ENV,
    ),
    "OPENAI_API_KEY": (
        "OPENAI_API_KEY",
        _CODEX_API_KEY_ENV,
        _CODEX_ACCESS_TOKEN_ENV,
        _CODEX_AUTH_JSON_ENV,
    ),
}


def _login_family(agent: str, model: str | None) -> str | None:
    subscription = getattr(AGENTS.get(agent), "subscription_auth", None)
    if subscription is not None:
        return subscription.replaces_env
    return infer_env_key_for_model(model) if model else None


def _select_login(
    candidates: list[str], env: dict[str, str], explicit_keys: set[str]
) -> str | None:
    """One login: an explicitly supplied one first, then the family's order."""
    present = [key for key in dict.fromkeys(candidates) if env.get(key)]
    explicit = [key for key in present if key in explicit_keys]
    return (explicit or present or [None])[0]


def _bridge_outranked(
    required_key: str | None,
    bridge_key: str,
    env: dict[str, str],
    explicit_keys: set[str],
) -> bool:
    """True when another login of the same family is the selected one."""
    family = list(_LOGIN_KEYS_BY_FAMILY.get(required_key or "", ()))
    if bridge_key not in family or bridge_key == required_key:
        return False
    selected = _select_login(family, env, explicit_keys)
    return selected is not None and selected != bridge_key


def _record_credential_policy(
    agent: str,
    model: str | None,
    env: dict[str, str],
    explicit_keys: set[str],
) -> None:
    """Compute the run's credential table once and record it by names."""
    from benchflow.agents.launch_env import DIRECT, NATIVE, compute_credential_policy

    policy = compute_credential_policy(agent, model, env, explicit_keys)
    selected = policy.login[0] if policy.login else None
    canonical = _LOGIN_ALIASES.get(selected or "")
    # Proxy routes already chose their key in provider resolution; only a
    # login the agent itself presents needs its CLI's variable pointed at it.
    if selected and canonical and policy.route in (NATIVE, DIRECT):
        env[canonical] = env[selected]
    if policy.login and policy.route in (NATIVE, DIRECT):
        # A resolved environment may be resolved again when a role connects.
        # Retaining an inherited competing login would make it look explicit
        # on that pass and could silently select a different account. Empty
        # guards suppress re-inheritance without trusting a prior policy marker.
        for key in _LOGIN_KEYS_BY_FAMILY.get(_login_family(agent, model) or "", ()):
            if key not in policy.login and env.get(key):
                env[key] = ""
    env[CREDENTIAL_POLICY_MARKER] = policy.to_json()


def _recorded_login(env: dict[str, str]) -> list[str]:
    try:
        data = json.loads(env.get(CREDENTIAL_POLICY_MARKER) or "{}")
    except json.JSONDecodeError:
        return []
    login = data.get("login") if isinstance(data, dict) else None
    return [n for n in login if isinstance(n, str)] if isinstance(login, list) else []


def login_label(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    explicit_env: dict[str, str] | None = None,
) -> str | None:
    """Which login the agent runs on, as a label: never a credential's value.

    ``BENCHFLOW_LOGIN_LABEL`` in the agent's environment (for example the
    name of a pooled account) wins. Otherwise the variable the credential is
    in and where it came from (``CLAUDE_CODE_OAUTH_TOKEN (agent_env)`` for one
    passed with ``--agent-env``/``agent_env``, ``(environment)`` for one
    inherited from the shell or ``.env``), or the host login file BenchFlow
    copied in (``host login (~/.claude/.credentials.json)``).
    """
    custom = " ".join((agent_env.get(_LOGIN_LABEL_ENV) or "").split())
    if custom:
        return custom[:80]
    explicit_env = explicit_env or {}
    cfg = AGENTS.get(agent)
    family = (
        cfg.subscription_auth.replaces_env
        if cfg is not None and cfg.subscription_auth is not None
        else (infer_env_key_for_model(model) if model else None)
    )
    keys = _LOGIN_KEYS_BY_FAMILY.get(family or "", (family,) if family else ())
    recorded = _recorded_login(agent_env)
    if recorded and recorded[0] in keys:
        keys = (recorded[0],)
    for key in keys:
        if agent_env.get(key):
            where = "agent_env" if explicit_env.get(key) else "environment"
            return f"{key} ({where})"
    if agent_env.get(_SUBSCRIPTION_AUTH_MARKER) == "1":
        if cfg is not None and cfg.subscription_auth is not None:
            return f"host login ({login_display(cfg.subscription_auth.detect_file)})"
        return "host login"
    return None


def inject_vertex_credentials(agent_env: dict[str, str], model: str) -> None:
    """Inject ADC credentials and defaults for Vertex AI models."""
    if not is_vertex_model(model):
        return
    adc_path = Path.home() / ".config/gcloud/application_default_credentials.json"
    if not adc_path.exists():
        raise MissingCredentialError(
            f"Vertex AI model {model!r} requires ADC credentials.",
            hint="gcloud auth application-default login",
        )
    agent_env.setdefault("GOOGLE_APPLICATION_CREDENTIALS_JSON", adc_path.read_text())
    agent_env.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    if "GOOGLE_CLOUD_PROJECT" not in agent_env:
        raise MissingCredentialError(
            f"GOOGLE_CLOUD_PROJECT required for Vertex AI model {model!r}.",
            hint="export GOOGLE_CLOUD_PROJECT=<project>, or pass "
            "--agent-env GOOGLE_CLOUD_PROJECT=<project>",
        )


def resolve_provider_env(
    agent_env: dict[str, str],
    model: str,
    agent: str,
) -> None:
    """Detect provider for model, inject BENCHFLOW_PROVIDER_* and env_mapping."""
    from benchflow.agents.providers import (
        REGISTRY_BASE_PROVIDERS,
        ZAI_CODING_REGISTRY_BASE_ENV,
        find_provider,
        find_provider_for_bare_model,
        resolve_base_url,
        strip_provider_prefix,
    )

    agent_env.setdefault("BENCHFLOW_PROVIDER_MODEL", strip_provider_prefix(model))
    agent_cfg = AGENTS.get(agent)
    # Agent-declared protocol takes precedence over provider's primary so
    # multi-endpoint providers (e.g. zai) route to the right URL.
    agent_protocol = agent_cfg.api_protocol if agent_cfg else ""
    # Resolve bare family ids (e.g. "deepseek-v4-pro") too, not just explicit
    # "provider/" prefixes — otherwise the provider env (NAME/BASE_URL/API_KEY)
    # is never emitted and harnesses that rely on it misroute (openhands' litellm
    # saw no provider and defaulted incorrectly). Mirrors acp/runtime.py.
    _prov = find_provider(model) or find_provider_for_bare_model(model)
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
            if _prov_name in REGISTRY_BASE_PROVIDERS and base_url:
                agent_env[ZAI_CODING_REGISTRY_BASE_ENV] = "1"
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
    return login_path(sa.detect_file).is_file()


def validate_aws_bedrock_env(agent_env: dict[str, str], model: str) -> None:
    """Validate Bedrock API-key auth and normalize region aliases."""
    token = agent_env.get("AWS_BEARER_TOKEN_BEDROCK")
    region = agent_env.get("AWS_REGION") or agent_env.get("AWS_DEFAULT_REGION")
    if not token:
        raise MissingCredentialError(
            f"AWS_BEARER_TOKEN_BEDROCK required for Bedrock model {model!r} but not set. "
            "Export it in your shell or put it in a .env file.",
            hint="export AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION=us-west-2",
        )
    if not region:
        raise MissingCredentialError(
            f"AWS_REGION or AWS_DEFAULT_REGION required for Bedrock model {model!r} "
            "but not set. Export one of them in your shell or put it in a .env file.",
            hint="export AWS_REGION=us-west-2",
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

    # Native ChatGPT subscription auth must keep Codex on its built-in `openai`
    # provider (chatgpt auth mode). A custom model_provider injected here carries
    # env_key=OPENAI_API_KEY + wire_api=responses against api.openai.com, which
    # forces API-key mode and makes Codex demand OPENAI_API_KEY -- defeating the
    # subscription path. Verified by ACP repro.
    if uses_native_subscription_auth(agent, model, agent_env):
        return

    base_url = agent_env.get("BENCHFLOW_PROVIDER_BASE_URL") or agent_env.get(
        "OPENAI_BASE_URL"
    )
    provider_model = agent_env.get("BENCHFLOW_PROVIDER_MODEL")
    if not base_url or not provider_model:
        return

    from benchflow.providers.litellm_config import strip_provider_prefix

    # Codex resolves model metadata (tool mode, apply_patch tool type,
    # multi-agent version) by model slug. The proxy alias in
    # BENCHFLOW_PROVIDER_MODEL (``benchflow-azure-foundry-openai-gpt-5.6-luna``)
    # is unknown to it: Codex warns "Model metadata for `...` not found.
    # Defaulting to fallback metadata" and offers a reduced tool surface (no
    # native apply_patch, no code mode, no subagents; #1145). The proxy serves
    # the bare slug next to the alias, so that is the id Codex gets.
    apply_codex_provider_config(
        agent_env,
        base_url=base_url,
        model=strip_provider_prefix(model),
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


def _bridge_explicit_openhands_provider_env(
    agent_env: dict[str, str],
    explicit_agent_env_keys: set[str],
) -> None:
    """Let explicit OpenHands endpoint/key flags configure the provider route."""
    if (
        "LLM_BASE_URL" in explicit_agent_env_keys
        and agent_env.get("LLM_BASE_URL")
        and "BENCHFLOW_PROVIDER_BASE_URL" not in agent_env
    ):
        agent_env["BENCHFLOW_PROVIDER_BASE_URL"] = agent_env["LLM_BASE_URL"]
    if (
        "LLM_API_KEY" in explicit_agent_env_keys
        and agent_env.get("LLM_API_KEY")
        and "BENCHFLOW_PROVIDER_API_KEY" not in agent_env
    ):
        agent_env["BENCHFLOW_PROVIDER_API_KEY"] = agent_env["LLM_API_KEY"]


def _inherit_the_muse_login(agent_env: dict[str, str]) -> None:
    """Copy the Muse login and the shim's settings from .env, then the host.

    Only a muse-code run reads them: auto_inherit_env hands every agent the
    names it lists, so a host that exports MUSE_AUTH_JSON for Muse would
    otherwise put Meta's subscription login into every other agent's sandbox
    (an OpenCode or Claude run on the same machine). Explicit agent env wins,
    as in auto_inherit_env, and a blank value counts as unset.
    """
    for source in (load_dotenv_env(), os.environ):
        for key in (_MUSE_AUTH_JSON_ENV, *_MUSE_SETTINGS_ENV):
            value = source.get(key)
            if value and value.strip():
                agent_env.setdefault(key, value)


def _keep_only_the_muse_login(
    agent_env: dict[str, str],
    explicit_agent_env_keys: set[str],
    *,
    require_login: bool = True,
) -> None:
    """Drop every inherited credential but the Muse login; refuse a run without one.

    Muse Code talks to Meta with its own login and needs no other key. Anything
    else auto-inherited from the host (another provider's key, a reviewer's
    Azure credentials in the same process) would only reach a third party's
    agent sandbox. Explicit ``agent_env`` entries are the caller's choice and
    stay, and so do the shim's own settings (the quota wait), which hold no
    credential.
    """
    keep = {_MUSE_AUTH_JSON_ENV, *_MUSE_SETTINGS_ENV}
    for key in list(agent_env):
        if key not in explicit_agent_env_keys and key not in keep:
            del agent_env[key]
    if require_login and not agent_env.get(_MUSE_AUTH_JSON_ENV):
        # A MissingCredentialError, not a bare ValueError: check_credentials
        # (the bench eval run preflight) re-raises only this type, so any other
        # error let the job start and every trial fail at setup instead.
        raise MissingCredentialError(
            "muse-code runs on a Muse Code subscription login: set MUSE_AUTH_JSON to "
            "the contents of the auth.json that `TBH_CREDENTIAL_BACKEND=file muse "
            "login` writes. A META_API_KEY (pay-as-you-go) run has no BenchFlow proxy "
            "route, and BenchFlow never sends API-key traffic around the proxy.",
            hint="run `TBH_CREDENTIAL_BACKEND=file muse login`, then "
            '`export MUSE_AUTH_JSON="$(cat ~/.config/muse/auth.json)"` '
            "(or put MUSE_AUTH_JSON in .env)",
        )


def with_prompt_budget(
    agent: str, agent_env: dict[str, str], timeout_sec: float | None
) -> dict[str, str]:
    """``agent_env`` plus the prompt's wall-clock budget, for the agents that read it.

    Only the Muse shim does: it holds a turn open while a spent subscription
    window resets, and must not wait past BenchFlow's own prompt timeout,
    where the trial would be scored as an agent timeout on a partial
    workspace and the turn's usage lost. Past the budget it fails with the
    usage-limit error instead. Every other agent gets ``agent_env`` unchanged.
    """
    if agent != _MUSE_AGENT or not timeout_sec or timeout_sec <= 0:
        return agent_env
    return {**agent_env, _MUSE_PROMPT_BUDGET_ENV: str(int(timeout_sec))}


def inherit_agent_auth_env(
    agent: str,
    model: str | None,
    agent_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Inherit and normalize auth inputs without requiring a credential.

    Plan inspection and execution must choose the same native/proxy route,
    including agent-specific aliases. Provider validation belongs to execution.
    """
    inherited = dict(agent_env or {})
    auto_inherit_env(inherited, source_env=load_dotenv_env())
    auto_inherit_env(inherited)
    _normalize_codex_auth_env(agent, model, inherited)
    if agent == _MUSE_AGENT:
        _inherit_the_muse_login(inherited)
        _keep_only_the_muse_login(inherited, set(agent_env or {}), require_login=False)
    elif model and agent not in ("oracle", "nop"):
        explicit_keys = set(agent_env or {})
        _drop_inherited_generic_provider_overrides(
            inherited, model=model, explicit_agent_env_keys=explicit_keys
        )
        if agent == "openhands":
            _bridge_explicit_openhands_provider_env(inherited, explicit_keys)
    return inherited


def resolve_agent_env(
    agent: str,
    model: str | None,
    agent_env: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve agent environment from explicit overrides, then .env defaults."""
    if agent == _MUSE_AGENT:
        validate_muse_model(model)
    explicit_agent_env_keys = set(agent_env or {})
    agent_env = inherit_agent_auth_env(agent, model, agent_env)
    # This is a decision for this resolution, never an input carried from a
    # previous role/account. Only the host-file fallback below may set it.
    agent_env.pop(_SUBSCRIPTION_AUTH_MARKER, None)
    agent_env.pop(CREDENTIAL_POLICY_MARKER, None)
    if agent == _MUSE_AGENT:
        _keep_only_the_muse_login(agent_env, explicit_agent_env_keys)
        # A supported Muse subscription never enters the provider pipeline:
        # in particular, no host Vertex ADC is discovered or injected.
        _record_credential_policy(agent, model, agent_env, explicit_agent_env_keys)
        return agent_env
    pre_provider_env = dict(agent_env)
    agent_cfg = AGENTS.get(agent)
    if (
        not model
        and agent_cfg
        and agent_cfg.subscription_auth
        and agent_env.get("BENCHFLOW_PROVIDER_API_KEY")
    ):
        # Without a model there is no provider mapping. Do not silently leave
        # the supplied key unused or select a different account's host login.
        raise MissingCredentialError(
            "A model is required with BENCHFLOW_PROVIDER_API_KEY.",
            hint="pass --model/model so the supplied provider key can be routed",
        )
    # Oracle runs solve.sh and never calls an LLM — model env vars and
    # API-key validation are skipped even if a caller forwards a model.
    from benchflow.agents.registry import is_scripted_agent

    # Native authentication consumes the shared normalized inputs directly.
    # Provider resolution would synthesize a conflicting endpoint or turn a
    # native token alias into a generic API key after check accepted the plan.
    if (
        model
        and not is_scripted_agent(agent)
        and not uses_native_subscription_auth(agent, model, agent_env)
    ):
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
        required_key = infer_env_key_for_model(model)
        mapped_provider_key = (
            agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_API_KEY")
            if agent_cfg
            else None
        )
        has_agent_native_bridge_key = bool(
            mapped_provider_key
            and not (
                "BENCHFLOW_PROVIDER_API_KEY" in explicit_agent_env_keys
                and pre_provider_env.get("BENCHFLOW_PROVIDER_API_KEY")
            )
            and pre_provider_env.get(mapped_provider_key)
            # An agent-native bridge never replaces a higher-ranked login of
            # the same family (an explicit ANTHROPIC_API_KEY over an inherited
            # ANTHROPIC_AUTH_TOKEN): the proxy serves the selected one.
            and not _bridge_outranked(
                required_key,
                mapped_provider_key,
                pre_provider_env,
                explicit_agent_env_keys,
            )
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
        has_explicit_generic_provider_key = bool(
            required_key
            and "BENCHFLOW_PROVIDER_API_KEY" in explicit_agent_env_keys
            and agent_env.get("BENCHFLOW_PROVIDER_API_KEY")
            and (
                "BENCHFLOW_PROVIDER_BASE_URL" in explicit_agent_env_keys
                or "LLM_BASE_URL" in explicit_agent_env_keys
            )
        )
        has_oauth = any(
            agent_env.get(key) and _shares_auth_context(required_key, key)
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
            and not agent_env.get(required_key)
            and not has_oauth
            and not has_agent_native_bridge_key
            and not has_explicit_generic_provider_key
            and not has_codex_access_token
            and not has_codex_auth_json
        ):
            if _can_use_subscription_auth(
                agent,
                model,
                required_key,
                agent_env,
            ) and check_subscription_auth(agent, required_key):
                agent_env[_SUBSCRIPTION_AUTH_MARKER] = "1"
                logger.info(
                    "Using host subscription auth (no %s set)",
                    required_key,
                )
            else:
                # Claude leads with its login, which needs no key (the
                # README's step 3; dogfood p1-13).
                lead = (
                    f"No Claude login found for model {model!r}: run `claude "
                    "setup-token` and export CLAUDE_CODE_OAUTH_TOKEN, or set "
                    "ANTHROPIC_API_KEY. "
                    if required_key == "ANTHROPIC_API_KEY"
                    else f"{required_key} required for model {model!r} but not set. "
                )
                raise MissingCredentialError(
                    lead + "Export it in your shell or put it in a .env file (a "
                    "key typed into a command stays in your shell history).",
                    hint=_LOGIN_HINTS.get(required_key),
                )
        _configure_codex_custom_provider(agent, model, agent_env)
    else:
        # Native login or no model: retain host-login discovery without entering
        # the provider pipeline or translating native credentials into API keys.
        if model and uses_native_vertex_login(agent, model):
            # The Gemini CLI's Vertex login is the host ADC (and its project
            # and location); no other host login of the agent is discovered.
            inject_vertex_credentials(agent_env, model)
        elif agent_cfg:
            # Native authentication still uses the caller's endpoint. Map only
            # the URL: provider resolution would also synthesize an API key and
            # change the selected native account. An explicit agent URL wins.
            native_url_key = agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_BASE_URL")
            if (
                native_url_key
                and agent_env.get("BENCHFLOW_PROVIDER_BASE_URL")
                and native_url_key not in explicit_agent_env_keys
            ):
                agent_env[native_url_key] = agent_env["BENCHFLOW_PROVIDER_BASE_URL"]
            for req_key in agent_cfg.requires_env:
                if (
                    not agent_env.get(req_key)
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
                    agent_env[_SUBSCRIPTION_AUTH_MARKER] = "1"
                    logger.info(
                        "Using host subscription auth (no %s set)",
                        req_key,
                    )
    # Increase output token limit to avoid truncation errors
    agent_env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "128000")
    # Disable telemetry/non-essential traffic in container
    agent_env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    _record_credential_policy(agent, model, agent_env, explicit_agent_env_keys)
    return agent_env
