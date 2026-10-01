"""Helpers for writing Codex ACP provider configuration."""

from __future__ import annotations

import json
import re
from typing import Any

from benchflow.providers.litellm_config import safe_model_alias, strip_provider_prefix

CODEX_CONFIG_ENV = "CODEX_CONFIG"
CODEX_DEFAULT_AUTH_REQUEST_ENV = "DEFAULT_AUTH_REQUEST"
CODEX_MODEL_PROVIDER_ENV = "MODEL_PROVIDER"
# The launcher writes this TOML to Codex's user config.toml (see codex_home_config).
CODEX_HOME_CONFIG_ENV = "BENCHFLOW_CODEX_HOME_CONFIG"
# First line of that file; the launcher removes a file that starts with it when
# a later launch routes Codex to no provider.
CODEX_HOME_CONFIG_MARKER = "# benchflow-codex-home-config"

_CODEX_PROVIDER_ID_PREFIX = "benchflow-"
_LITELLM_MODEL_VIA_ENV = "BENCHFLOW_LITELLM_MODEL_VIA_ENV"
_PROVIDER_MODEL_ENV = "BENCHFLOW_PROVIDER_MODEL"


def _parse_codex_config(
    raw_config: str | None, *, strict: bool = False
) -> dict[str, Any] | None:
    if not raw_config:
        return {}
    try:
        config = json.loads(raw_config)
    except (json.JSONDecodeError, TypeError) as exc:
        if strict:
            raise ValueError(f"{CODEX_CONFIG_ENV} must be valid JSON") from exc
        return None
    if not isinstance(config, dict):
        if strict:
            raise ValueError(f"{CODEX_CONFIG_ENV} must decode to a JSON object")
        return None
    return config


def codex_provider_id(provider_name: str | None) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in (provider_name or "provider").lower()
    ).strip("-")
    return f"{_CODEX_PROVIDER_ID_PREFIX}{safe_name or 'provider'}"


def apply_codex_provider_config(
    agent_env: dict[str, str],
    *,
    base_url: str,
    model: str | None,
    provider_name: str,
    strict: bool = False,
) -> None:
    """Create or update Codex's model provider entry in ``agent_env``."""
    config = _parse_codex_config(agent_env.get(CODEX_CONFIG_ENV), strict=strict)
    if config is None:
        return

    provider_id = (
        agent_env.get(CODEX_MODEL_PROVIDER_ENV)
        or config.get("model_provider")
        or codex_provider_id(provider_name)
    )
    providers = config.get("model_providers")
    providers = {} if not isinstance(providers, dict) else dict(providers)
    provider = providers.get(provider_id)
    provider = dict(provider) if isinstance(provider, dict) else {}
    provider.setdefault("name", provider_name)
    provider["base_url"] = base_url
    provider.setdefault("env_key", "OPENAI_API_KEY")
    provider.setdefault("wire_api", "responses")
    provider.setdefault("supports_websockets", False)

    providers[provider_id] = provider
    config["model_providers"] = providers
    config["model_provider"] = provider_id
    if model:
        config["model"] = model

    agent_env[CODEX_MODEL_PROVIDER_ENV] = str(provider_id)
    agent_env[CODEX_CONFIG_ENV] = json.dumps(config, separators=(",", ":"))
    _apply_codex_default_auth_request(
        agent_env,
        base_url=base_url,
        provider_name=provider_name,
    )


def _toml_scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        # JSON's string escapes are TOML basic-string escapes; TOML also
        # requires DEL escaped, and non-ASCII stays literal (TOML rejects the
        # surrogate-pair escapes ensure_ascii would write).
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    return None


def _toml_value(value: Any) -> str | None:
    """A TOML value for scalars, lists of scalars and one-level tables, else None."""
    scalar = _toml_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, list):
        items = [r for r in map(_toml_scalar, value) if r is not None]
        return "[" + ", ".join(items) + "]" if len(items) == len(value) else None
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            rendered = _toml_scalar(item)
            if not isinstance(key, str) or rendered is None:
                return None
            pairs.append(f"{_toml_scalar(key)} = {rendered}")
        return "{ " + ", ".join(pairs) + " }" if pairs else "{}"
    return None


def codex_home_config(config: dict[str, Any] | None) -> str | None:
    """Codex's user config.toml pinning every thread to the session's provider.

    codex-acp hands CODEX_CONFIG only to the threads it opens for the ACP
    session. Its title generator (codex-acp 1.13.1 through 2.0.0,
    TitleGenerator.ts) starts an ephemeral thread without it and runs a turn
    on the hard-wired ``gpt-5.6-luna`` with the user's first message, which is
    the task prompt. Such a thread takes its provider from the user
    config.toml, else Codex's built-in ``openai`` provider at api.openai.com:
    the prompt left the run, and the request failed with 401 on every
    rollout. This file makes the session's provider the default for every
    thread and points the built-in ``openai`` provider at the same endpoint
    (``openai_base_url``) for a thread that names it. None when CODEX_CONFIG
    routes Codex to no provider of its own.
    """
    if not isinstance(config, dict):
        return None
    provider_id = config.get("model_provider")
    providers = config.get("model_providers")
    provider = providers.get(provider_id) if isinstance(providers, dict) else None
    if not isinstance(provider_id, str) or not isinstance(provider, dict):
        return None
    base_url = provider.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        return None
    lines = [
        CODEX_HOME_CONFIG_MARKER,
        "# Written by BenchFlow for this launch: threads that codex-acp starts",
        "# without the session config use the run's provider too.",
        f"model_provider = {_toml_scalar(provider_id)}",
    ]
    model = config.get("model")
    if isinstance(model, str) and model:
        lines.append(f"model = {_toml_scalar(model)}")
    lines += [
        f"openai_base_url = {_toml_scalar(base_url)}",
        "",
        f"[model_providers.{_toml_scalar(provider_id)}]",
    ]
    for key, value in provider.items():
        rendered = _toml_value(value)
        if isinstance(key, str) and rendered is not None:
            lines.append(f"{_toml_scalar(key)} = {rendered}")
    return "\n".join(lines) + "\n"


def codex_config_overrides(config: dict[str, Any] | None) -> list[str]:
    """CODEX_CONFIG as ``codex -c key=value`` overrides, dotted paths for tables.

    The native harness runs ``codex exec --ignore-user-config`` with these, so
    every setting it runs with comes from the run's CODEX_CONFIG and none from
    a config.toml already in the sandbox (the image's, or one another role's
    ACP launch left). Values are TOML, as ``-c`` parses them. A key that is
    not a valid bare TOML key, or a value this renderer cannot write, is
    refused rather than dropped: a missing ``model_provider`` would send the
    model call to Codex's default provider.
    """
    if not isinstance(config, dict):
        return []
    out: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str) or not _BARE_TOML_KEY.fullmatch(key):
                    raise ValueError(
                        f"{CODEX_CONFIG_ENV} key {key!r} is not a bare key"
                    )
                walk(f"{prefix}.{key}" if prefix else key, item)
            return
        rendered = _toml_value(value)
        if rendered is None:
            raise ValueError(f"{CODEX_CONFIG_ENV}.{prefix} has no TOML rendering")
        out.append(f"{prefix}={rendered}")

    walk("", config)
    return out


_BARE_TOML_KEY = re.compile(r"[A-Za-z0-9_-]+")


def apply_codex_launch_config(
    agent: str,
    agent_env: dict[str, str],
    *,
    model: str | None,
    reasoning_effort: str | None,
    sandboxed: bool = False,
) -> tuple[dict[str, str], bool]:
    """Configure the adapter's sandbox, web policy, launch-owned model effort
    and the provider every Codex thread defaults to."""
    if agent != "codex-acp":
        return agent_env, False
    updated_env = dict(agent_env)
    if sandboxed:
        # BenchFlow's non-root sandbox and UID firewall own isolation. The
        # adapter defaults to workspace-write, whose nested bwrap fails on
        # Docker/Daytona before tools run. CODEX_CONFIG.sandbox_mode does not
        # fix this: codex-acp overrides it with its session's AgentMode.
        updated_env.setdefault("INITIAL_AGENT_MODE", "agent-full-access")
    disable_search = any(
        agent_env.get(key) == "1"
        for key in ("BENCHFLOW_DISALLOW_WEB_TOOLS", "BENCHFLOW_EGRESS_DENYLIST")
    )
    config = _parse_codex_config(agent_env.get(CODEX_CONFIG_ENV), strict=disable_search)
    provider_model = agent_env.get(_PROVIDER_MODEL_ENV)
    # The launch owns the model when CODEX_CONFIG names either the proxy
    # alias or the bare slug the provider config hands Codex (#1145).
    owns_model = bool(
        model
        and agent_env.get(_LITELLM_MODEL_VIA_ENV) in {"1", "true", "True"}
        and provider_model
        and provider_model == safe_model_alias(model)
        and config is not None
        and config.get("model") in {provider_model, strip_provider_prefix(model)}
    )
    if config is not None and (disable_search or (owns_model and reasoning_effort)):
        if disable_search:
            # codex-acp (1.6.0 through 1.13.1) ignores CLI -c flags; its CODEX_CONFIG
            # is forwarded to the Codex app-server's thread configuration.
            config["web_search"] = "disabled"
        if owns_model and reasoning_effort:
            config["model_reasoning_effort"] = reasoning_effort
        updated_env[CODEX_CONFIG_ENV] = json.dumps(config, separators=(",", ":"))
    home_config = codex_home_config(config)
    if home_config is None:
        # A value left from another role's launch would route this one.
        updated_env.pop(CODEX_HOME_CONFIG_ENV, None)
    else:
        updated_env[CODEX_HOME_CONFIG_ENV] = home_config
    return (updated_env if updated_env != agent_env else agent_env), owns_model


def _apply_codex_default_auth_request(
    agent_env: dict[str, str],
    *,
    base_url: str,
    provider_name: str,
) -> None:
    """Provide non-interactive auth for codex-acp's authorization gate.

    ``codex-acp@0.0.45`` checks account authorization before it sends the first
    prompt. Supplying ``OPENAI_API_KEY`` and ``CODEX_CONFIG`` is not enough; the
    wrapper needs a default ACP auth request to complete that gate without an
    IDE round-trip.
    """
    api_key = agent_env.get("OPENAI_API_KEY")
    if not api_key:
        return

    normalized = provider_name.strip().lower()
    if normalized == "litellm":
        # BenchFlow owns this local gateway. Authenticate as a gateway so the
        # proxy master key is used only against the proxy, not as an OpenAI
        # account login key.
        request = {
            "methodId": "gateway",
            "_meta": {
                "gateway": {
                    "baseUrl": base_url,
                    "providerName": "BenchFlow LiteLLM",
                    "headers": {"Authorization": f"Bearer {api_key}"},
                }
            },
        }
        agent_env[CODEX_DEFAULT_AUTH_REQUEST_ENV] = json.dumps(
            request,
            separators=(",", ":"),
        )
        return

    if normalized == "openai" and CODEX_DEFAULT_AUTH_REQUEST_ENV not in agent_env:
        request = {
            "methodId": "api-key",
            "_meta": {"api-key": {"apiKey": api_key}},
        }
        agent_env[CODEX_DEFAULT_AUTH_REQUEST_ENV] = json.dumps(
            request,
            separators=(",", ":"),
        )


def disable_codex_apps(agent_env: dict[str, str]) -> dict[str, str]:
    """Set only Apps in supported ACP config; managed requirements enforce it."""
    config = _parse_codex_config(agent_env.get(CODEX_CONFIG_ENV), strict=True)
    assert config is not None
    features = config.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("CODEX_CONFIG.features must be an object")
    config["features"] = {**features, "apps": False}
    return {**agent_env, CODEX_CONFIG_ENV: json.dumps(config, separators=(",", ":"))}
