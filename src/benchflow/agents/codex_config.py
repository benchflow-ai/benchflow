"""Helpers for writing Codex ACP provider configuration."""

from __future__ import annotations

import json
from typing import Any

CODEX_CONFIG_ENV = "CODEX_CONFIG"
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
) -> None:
    """Create or update Codex's model provider entry in ``agent_env``."""
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
