#!/usr/bin/env python3
"""Launch wrapper for pi-acp — bridges BENCHFLOW_PROVIDER_* to Pi config.

Pi natively reads ANTHROPIC_* env vars for Anthropic providers. For
OpenAI-compatible providers (vLLM, etc.), Pi requires a ``models.json``
config file at ``~/.pi/agent/models.json`` that declares the provider's
wire protocol.  This wrapper generates it from BENCHFLOW_PROVIDER_* env
vars injected by the SDK, then execs ``pi-acp``.

See https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md
"""

import json
import os
import sys
from pathlib import Path


def setup_provider():
    """Configure Pi for the detected provider protocol."""
    protocol = os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", "")
    base_url = os.environ.get("BENCHFLOW_PROVIDER_BASE_URL", "")
    api_key = os.environ.get("BENCHFLOW_PROVIDER_API_KEY", "")
    model = os.environ.get("BENCHFLOW_PROVIDER_MODEL", "")
    provider_name = os.environ.get("BENCHFLOW_PROVIDER_NAME", "custom")

    if protocol == "openai-completions" and base_url:
        # Pi uses ~/.pi/agent/models.json to discover non-Anthropic providers.
        # Register the provider so Pi routes API calls through the OpenAI
        # Chat Completions wire format instead of Anthropic Messages.
        config = {
            "providers": {
                provider_name: {
                    "baseUrl": base_url,
                    "api": "openai-completions",
                    "apiKey": api_key or "unused",
                    "models": [
                        {
                            "id": model,
                            "name": model,
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": 128000,
                            "maxTokens": 16384,
                        }
                    ],
                }
            }
        }
        config_dir = Path.home() / ".pi" / "agent"
        config_dir.mkdir(parents=True, exist_ok=True)
        models_path = config_dir / "models.json"
        # Merge with existing config so manually-added providers survive
        if models_path.exists():
            try:
                existing = json.loads(models_path.read_text())
                existing.setdefault("providers", {}).update(config["providers"])
                config = existing
            except (json.JSONDecodeError, OSError):
                pass
        models_path.write_text(json.dumps(config, indent=2))
    else:
        # Anthropic mode — set native env vars that Pi reads directly
        if base_url:
            os.environ.setdefault("ANTHROPIC_BASE_URL", base_url)
        if api_key:
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", api_key)
        if model:
            os.environ.setdefault("ANTHROPIC_MODEL", model)


if __name__ == "__main__":
    setup_provider()
    os.execvp("pi-acp", ["pi-acp", *sys.argv[1:]])
