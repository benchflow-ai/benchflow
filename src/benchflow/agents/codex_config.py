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


def apply_codex_provider_config(
    agent_env: dict[str, str],
    *,
    base_url: str,
    model: str | None,
    provider_name: str,
    strict: bool = False,
    isolate: bool = False,
) -> None:
    """Create or update Codex's model provider entry in ``agent_env``.

    ``isolate`` discards caller config and is reserved for a BenchFlow-owned
    proxy boundary. Direct provider configuration keeps unrelated caller
    settings while ``strict`` still validates JSON and pins the credential key.
    """
    raw_config = agent_env.get(CODEX_CONFIG_ENV)
    if not raw_config:
        config: dict[str, Any] = {}
    else:
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f"{CODEX_CONFIG_ENV} must be valid JSON") from exc
            return
    if not isinstance(config, dict):
        if strict:
            raise ValueError(f"{CODEX_CONFIG_ENV} must decode to a JSON object")
        return

    if isolate:
        provider_id = codex_provider_id(provider_name)
        config = {}
        providers: dict[str, Any] = {}
        provider: dict[str, Any] = {
            "name": provider_name,
            "base_url": base_url,
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "supports_websockets": False,
        }
    else:
        provider_id = (
            agent_env.get(CODEX_MODEL_PROVIDER_ENV)
            or config.get("model_provider")
            or codex_provider_id(provider_name)
        )
        providers_value = config.get("model_providers")
        providers = (
            {} if not isinstance(providers_value, dict) else dict(providers_value)
        )
        provider_value = providers.get(provider_id)
        provider = dict(provider_value) if isinstance(provider_value, dict) else {}
        provider.setdefault("name", provider_name)
        provider["base_url"] = base_url
        if strict:
            provider["env_key"] = "OPENAI_API_KEY"
        else:
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
