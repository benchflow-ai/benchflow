#!/usr/bin/env python3
"""Select the first credentialed provider for the integration-eval CI gate.

Probes a fixed candidate order and exports BENCHFLOW_INTEGRATION_AGENT,
BENCHFLOW_INTEGRATION_MODEL, and BENCHFLOW_JUDGE_MODEL to $GITHUB_ENV. Hard-fails
(exit 1) with a clear message when NO provider is credentialed — never exits 0
with no provider selected (ENG-265). A judge-availability preflight ensures the
chosen judge model has a usable key.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_AGENT = "openhands"


@dataclass(frozen=True)
class ProviderChoice:
    provider: str
    agent: str
    model: str
    judge_model: str


# (provider, api-key env, [extra required env], model id). First credentialed wins.
_CANDIDATES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("deepseek", "DEEPSEEK_API_KEY", (), "deepseek/deepseek-v4-flash"),
    ("glm", "GLM_API_KEY", (), "glm/glm-4-flash"),
    ("qwen", "QWEN_API_KEY", (), "qwen/qwen-flash"),
    ("litellm", "LITELLM_API_KEY", ("LITELLM_BASE_URL",), "openai/gpt-5.4-nano"),
    ("openai", "OPENAI_API_KEY", (), "openai/gpt-5.4-nano"),
    ("github_models", "GITHUB_MODELS_TOKEN", (), "openai/gpt-4o-mini"),
)


class NoProviderAvailable(RuntimeError):
    """Raised when no candidate provider is credentialed."""


def select_provider(env: Mapping[str, str]) -> ProviderChoice:
    """First credentialed provider, or raise NoProviderAvailable (fail closed)."""
    for provider, key_env, extra_envs, model in _CANDIDATES:
        if env.get(key_env) and all(env.get(e) for e in extra_envs):
            # The LLM judge reuses the selected provider's model so it needs no
            # separate credential — guaranteeing judge availability.
            return ProviderChoice(provider, DEFAULT_AGENT, model, model)
    probed = ", ".join(c[1] for c in _CANDIDATES)
    raise NoProviderAvailable(
        "no integration provider is credentialed; set one of: " + probed
    )


def main() -> int:
    try:
        choice = select_provider(os.environ)
    except NoProviderAvailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    exports = {
        "BENCHFLOW_INTEGRATION_AGENT": choice.agent,
        "BENCHFLOW_INTEGRATION_MODEL": choice.model,
        "BENCHFLOW_JUDGE_MODEL": choice.judge_model,
    }
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as fh:
            for k, v in exports.items():
                fh.write(f"{k}={v}\n")
    for k, v in exports.items():
        print(f"{k}={v}")
    print(f"selected provider: {choice.provider}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
