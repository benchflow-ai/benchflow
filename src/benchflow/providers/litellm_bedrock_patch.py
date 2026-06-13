"""LiteLLM startup patch for Bedrock Claude 4.8+ adaptive thinking.

LiteLLM 1.89.0 knows the direct Anthropic Claude 4.8+ IDs, but Bedrock
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
    r"claude-(?:(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)|fable-5(?!\d))"
)

# Requested efforts low→high. LiteLLM 1.88.0rc1 only accepts up to ``high`` for
# adaptive Claude 4.8+ and raises on ``xhigh``/``max`` (#737), so an override is
# clamped to the accepted ceiling before being handed to the transform. This is
# the standalone (sandbox-deployed) copy of the ladder in ``litellm_config``;
# keep the two in sync.
_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_VALID_EFFORTS = set(_EFFORT_LADDER)
_LITELLM_MAX_EFFORT = "high"


def _clamp_effort(effort: str) -> str:
    if effort not in _EFFORT_LADDER:
        return effort
    return _EFFORT_LADDER[
        min(_EFFORT_LADDER.index(effort), _EFFORT_LADDER.index(_LITELLM_MAX_EFFORT))
    ]


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
            # Clamp xhigh/max (which litellm rejects) to the accepted ceiling so
            # the request runs at the real maximum instead of raising (#737).
            reasoning_effort = _clamp_effort(reasoning_effort)
        return original(self, model, reasoning_effort, optional_params)

    # Marker for the fail-closed startup preflight (#602): lets the runtime
    # verify this override is installed without importing this module (which
    # would itself apply the patches and mask a sitecustomize load failure).
    setattr(handle, "__benchflow_bedrock_patch__", True)  # noqa: B010

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
