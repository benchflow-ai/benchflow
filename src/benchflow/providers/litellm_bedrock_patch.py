"""LiteLLM startup patch for Bedrock Claude 4.8+ adaptive thinking.

LiteLLM 1.88.0rc1 knows the direct Anthropic Claude 4.8+ IDs, but Bedrock
inference-profile IDs such as ``us.anthropic.claude-opus-4-8`` still need to be
classified as adaptive-thinking models before the Bedrock Converse transform is
called. This module is loaded inside the LiteLLM proxy process via
``sitecustomize`` and is intentionally inert for every non-4.8+ model.
"""

from __future__ import annotations

import os
import re
from typing import Any

BEDROCK_THINKING_EFFORT_ENV = "BENCHFLOW_BEDROCK_THINKING_EFFORT"
BEDROCK_ADAPTIVE_THINKING_RE = re.compile(
    r"claude-(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)"
)

_VALID_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


def _is_new_bedrock_claude(model: str | None) -> bool:
    try:
        return bool(model and BEDROCK_ADAPTIVE_THINKING_RE.search(model.lower()))
    except Exception:
        return False


def _patch_anthropic_gate() -> None:
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    except Exception:
        return

    original = AnthropicConfig._is_adaptive_thinking_model

    def gate(model: str) -> bool:
        if _is_new_bedrock_claude(model):
            return True
        return original(model or "")

    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AnthropicConfig,
        "_is_adaptive_thinking_model",
        staticmethod(gate),
    )


def _patch_bedrock_effort() -> None:
    try:
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )
    except Exception:
        return

    original = AmazonConverseConfig._handle_reasoning_effort_parameter

    def handle(
        self: Any,
        model: str,
        reasoning_effort: str,
        optional_params: dict[Any, Any],
    ) -> None:
        if _is_new_bedrock_claude(model):
            override = os.environ.get(BEDROCK_THINKING_EFFORT_ENV, "").strip().lower()
            if override in _VALID_EFFORTS:
                reasoning_effort = override
        return original(self, model, reasoning_effort, optional_params)

    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AmazonConverseConfig,
        "_handle_reasoning_effort_parameter",
        handle,
    )


def _patch_cost_map() -> None:
    try:
        import litellm
    except Exception:
        return

    for key in list(getattr(litellm, "model_cost", {})):
        if _is_new_bedrock_claude(key):
            litellm.model_cost[key]["supports_adaptive_thinking"] = True


_patch_anthropic_gate()
_patch_bedrock_effort()
_patch_cost_map()
