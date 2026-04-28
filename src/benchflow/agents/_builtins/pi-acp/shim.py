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

# Pi model-metadata defaults if neither the provider registry nor per-run
# overrides supply values. Conservative — most modern models exceed these.
_DEFAULT_CONTEXT_WINDOW = 128000
_DEFAULT_MAX_TOKENS = 16384


def _lookup_model_metadata(model: str) -> dict:
    """Return the registry entry for `model` from BENCHFLOW_PROVIDER_MODELS, or {}."""
    raw = os.environ.get("BENCHFLOW_PROVIDER_MODELS", "")
    if not raw:
        return {}
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == model:
            return entry
    return {}


def _derive_provider_name(model: str) -> str:
    """Derive a unique provider key so concurrent runs don't collide on "custom"."""
    if "/" in model:
        return f"benchflow-{model.split('/')[0]}"
    return f"benchflow-{model}" if model else "benchflow-custom"


def setup_provider() -> None:
    """Configure Pi for the detected provider protocol."""
    protocol = os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", "")
    base_url = os.environ.get("BENCHFLOW_PROVIDER_BASE_URL", "")
    api_key = os.environ.get("BENCHFLOW_PROVIDER_API_KEY", "")
    model = os.environ.get("BENCHFLOW_PROVIDER_MODEL", "")
    provider_name = os.environ.get("BENCHFLOW_PROVIDER_NAME") or _derive_provider_name(
        model
    )

    if protocol == "openai-completions":
        if not base_url:
            raise SystemExit(
                "pi-acp: BENCHFLOW_PROVIDER_PROTOCOL=openai-completions requires "
                "BENCHFLOW_PROVIDER_BASE_URL, but it is empty. Check provider "
                "registry url_params (e.g. missing GOOGLE_CLOUD_PROJECT)."
            )
        # Pi uses ~/.pi/agent/models.json to discover non-Anthropic providers.
        # Register the provider so Pi routes API calls through the OpenAI
        # Chat Completions wire format instead of Anthropic Messages.
        meta = _lookup_model_metadata(model)
        config = {
            "providers": {
                provider_name: {
                    "baseUrl": base_url,
                    "api": "openai-completions",
                    "apiKey": api_key or "unused",
                    "models": [
                        {
                            "id": model,
                            "name": meta.get("name", model),
                            "reasoning": meta.get("reasoning", False),
                            "input": meta.get("input", ["text"]),
                            "contextWindow": meta.get(
                                "contextWindow", _DEFAULT_CONTEXT_WINDOW
                            ),
                            "maxTokens": meta.get("maxTokens", _DEFAULT_MAX_TOKENS),
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
                print(
                    f"Warning: could not parse {models_path}, overwriting.",
                    file=sys.stderr,
                )
        models_path.write_text(json.dumps(config, indent=2))
    else:
        # Anthropic mode — set native env vars that Pi reads directly.
        # DO NOT change setdefault to assignment. Users route through proxies
        # with --ae ANTHROPIC_BASE_URL=<proxy>; overwriting breaks that path.
        # Why: BENCHFLOW_PROVIDER_BASE_URL is the registry default; the user's
        # --ae override must win. Pinned by tests/test_pi_acp_launcher.py::
        # test_setdefault_does_not_overwrite.
        if base_url:
            os.environ.setdefault("ANTHROPIC_BASE_URL", base_url)
        if api_key:
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", api_key)
        if model:
            os.environ.setdefault("ANTHROPIC_MODEL", model)


def main() -> None:
    setup_provider()
    try:
        os.execvp("pi-acp", ["pi-acp", *sys.argv[1:]])
    except FileNotFoundError as e:
        raise SystemExit(
            "pi-acp: 'pi-acp' binary not found on PATH. It should have been "
            "installed by the registry's install_cmd via "
            "'npm install -g pi-acp'. Check the container's install log."
        ) from e


if __name__ == "__main__":
    main()
