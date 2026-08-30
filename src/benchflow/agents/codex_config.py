"""Helpers for writing Codex ACP provider configuration."""

from __future__ import annotations

import json
from typing import Any

CODEX_CONFIG_ENV = "CODEX_CONFIG"
CODEX_DEFAULT_AUTH_REQUEST_ENV = "DEFAULT_AUTH_REQUEST"
CODEX_MODEL_PROVIDER_ENV = "MODEL_PROVIDER"

_CODEX_PROVIDER_ID_PREFIX = "benchflow-"


def codex_provider_id(provider_name: str | None) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in (provider_name or "provider").lower()
    ).strip("-")
    return f"{_CODEX_PROVIDER_ID_PREFIX}{safe_name or 'provider'}"


def apply_codex_custom_provider_config(
    agent_env: dict[str, str],
    *,
    base_url: str,
    model: str | None,
    provider_name: str,
) -> None:
    """Update caller-owned Codex config for a direct custom provider."""

    raw_config = agent_env.get(CODEX_CONFIG_ENV)
    if not raw_config:
        config: dict[str, Any] = {}
    else:
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{CODEX_CONFIG_ENV} must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{CODEX_CONFIG_ENV} must decode to a JSON object")

    configured_provider_id = agent_env.get(CODEX_MODEL_PROVIDER_ENV) or config.get(
        "model_provider"
    )
    provider_id = (
        configured_provider_id
        if isinstance(configured_provider_id, str) and configured_provider_id
        else codex_provider_id(provider_name)
    )
    providers_value = config.get("model_providers")
    providers = {} if not isinstance(providers_value, dict) else dict(providers_value)
    provider_value = providers.get(provider_id)
    provider = dict(provider_value) if isinstance(provider_value, dict) else {}
    provider.setdefault("name", provider_name)
    provider["base_url"] = base_url
    provider["env_key"] = "OPENAI_API_KEY"
    provider.setdefault("wire_api", "responses")
    provider.setdefault("supports_websockets", False)

    _write_codex_provider_config(
        agent_env,
        config=config,
        providers=providers,
        provider_id=provider_id,
        provider=provider,
        model=model,
        base_url=base_url,
        provider_name=provider_name,
    )


def apply_codex_proxy_config(
    agent_env: dict[str, str],
    *,
    base_url: str,
    model: str | None,
    provider_name: str,
) -> None:
    """Replace Codex config with one BenchFlow-owned proxy provider."""

    provider_id = codex_provider_id(provider_name)
    _write_codex_provider_config(
        agent_env,
        config={},
        providers={},
        provider_id=provider_id,
        provider={
            "name": provider_name,
            "base_url": base_url,
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "supports_websockets": False,
        },
        model=model,
        base_url=base_url,
        provider_name=provider_name,
    )


def _write_codex_provider_config(
    agent_env: dict[str, str],
    *,
    config: dict[str, Any],
    providers: dict[str, Any],
    provider_id: str,
    provider: dict[str, Any],
    model: str | None,
    base_url: str,
    provider_name: str,
) -> None:
    """Serialize one already-resolved Codex provider configuration."""

    providers[provider_id] = provider
    config["model_providers"] = providers
    config["model_provider"] = provider_id
    if model:
        config["model"] = model

    agent_env[CODEX_MODEL_PROVIDER_ENV] = provider_id
    agent_env[CODEX_CONFIG_ENV] = json.dumps(config, separators=(",", ":"))
    _apply_codex_default_auth_request(
        agent_env,
        base_url=base_url,
        provider_name=provider_name,
    )


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
