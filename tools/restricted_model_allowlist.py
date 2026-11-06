"""Generate restricted-run base models from the pinned LiteLLM wheel.

Run `uv run python -m tools.restricted_model_allowlist [--check]`.
Provider ownership and inference modes are curated here; a gateway's chat
metadata is never evidence for a fixed upstream. Review this policy when
updating the pin, then regenerate the checked-in snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from benchflow.providers.restricted_models import identify, pinned_model_metadata

DEST = (
    Path(__file__).resolve().parents[1]
    / "src/benchflow/providers/restricted_model_allowlist.json"
)
# Official model APIs, including fixed foundation models served by the cloud
# providers. No router, marketplace gateway, deployment or operator-run host.
OWNERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "vertex_ai-language-models": "vertex_ai",
    "vertex_ai-anthropic_models": "vertex_ai",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "deepseek": "deepseek",
    "zai": "zai",
}
MODES = frozenset({"chat", "responses"})
# Positive model families, not all price-table keys owned by a provider (the
# OpenAI table also prices containers and realtime sessions as mode=chat).
FAMILIES = {
    "openai": r"(?:gpt-\d[\w.-]*|chatgpt-\d[\w.-]*|o[1-9]\d*[\w.-]*|codex-mini-latest)",
    "anthropic": r"claude-[\w.-]+",
    "gemini": r"(?:gemini-\d[\w.-]*|gemma-[\w.-]+)",
    "vertex_ai": r"(?:gemini-\d[\w.-]*|claude-[\w.@-]+)",
    "bedrock": r"[a-z][\w]*\.[A-Za-z0-9][\w.:-]*",
    "deepseek": r"deepseek-[\w.-]+",
    "zai": r"glm-[\w.-]+",
}


def snapshot() -> dict:
    metadata = pinned_model_metadata()
    from benchflow.providers.litellm_request_guard import hosted_retrieval_model

    models: dict[tuple[str, str], str] = {}
    for key, info in sorted(metadata.items()):
        if not isinstance(info, dict) or info.get("mode") not in MODES:
            continue
        owner = OWNERS.get(info.get("litellm_provider"))
        if owner is None:
            continue
        # Only canonical ids, not pricing keys with regions, subscriptions,
        # wildcards, ARN deployments or multi-component gateway namespaces.
        candidate = key if key.startswith(f"{owner}/") else f"{owner}/{key}"
        try:
            _, base, _ = identify(candidate)
        except Exception:
            continue
        if not re.fullmatch(FAMILIES[owner], base):
            continue
        if hosted_retrieval_model(candidate) or hosted_retrieval_model(base):
            continue
        models.setdefault((owner, base), key)
    return {
        "litellm_version": version("litellm"),
        "metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "entries": [
            [provider, base, key] for (provider, base), key in sorted(models.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(snapshot(), indent=2) + "\n"
    if args.check:
        if DEST.read_text() != rendered:
            print("Restricted model allowlist is stale; regenerate it from the pin.")
            return 1
    else:
        DEST.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
