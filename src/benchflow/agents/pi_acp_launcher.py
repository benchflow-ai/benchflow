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
_DEFAULT_MAX_TOKENS_CAP = 4096
_BENCHFLOW_BIN_DIR = "/opt/benchflow/bin"
_PI_ACP_BIN = "/opt/benchflow/bin/pi-acp"


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


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str | bytes | bytearray):
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed > 0 else None


def _default_max_tokens(context_window: object) -> int:
    """Return a fallback completion cap that leaves prompt budget available."""
    window = _positive_int(context_window) or _DEFAULT_CONTEXT_WINDOW
    return max(1, min(_DEFAULT_MAX_TOKENS_CAP, window // 4))


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
        context_window = meta.get("contextWindow", _DEFAULT_CONTEXT_WINDOW)
        max_tokens = meta.get("maxTokens")
        if max_tokens is None:
            max_tokens = _default_max_tokens(context_window)
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
                            "contextWindow": context_window,
                            "maxTokens": max_tokens,
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
        # with --agent-env ANTHROPIC_BASE_URL=<proxy>; overwriting breaks that path.
        # Why: BENCHFLOW_PROVIDER_BASE_URL is the registry default; the user's
        # --agent-env override must win. Pinned by tests/test_pi_acp_launcher.py::
        # test_setdefault_does_not_overwrite.
        if base_url:
            os.environ.setdefault("ANTHROPIC_BASE_URL", base_url)
        if api_key:
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", api_key)
        if model:
            os.environ.setdefault("ANTHROPIC_MODEL", model)


def _prepend_benchflow_bin_path() -> None:
    """Let pi-acp find the paired pi wrapper without exposing Node/npm."""
    current = os.environ.get("PATH", "")
    parts = current.split(":") if current else []
    if _BENCHFLOW_BIN_DIR not in parts:
        os.environ["PATH"] = (
            f"{_BENCHFLOW_BIN_DIR}:{current}" if current else _BENCHFLOW_BIN_DIR
        )


def main() -> None:
    setup_provider()
    _prepend_benchflow_bin_path()
    try:
        os.execv(_PI_ACP_BIN, [_PI_ACP_BIN, *sys.argv[1:]])
    except FileNotFoundError as e:
        raise SystemExit(
            f"pi-acp: {_PI_ACP_BIN!r} not found. It should have been "
            "installed by the registry's install_cmd via "
            "'npm install -g pi-acp'. Check the container's install log."
        ) from e


if __name__ == "__main__":
    main()
