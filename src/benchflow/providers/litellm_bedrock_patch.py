"""LiteLLM startup patches for Claude 4.8+ models (Bedrock and direct).

LiteLLM 1.89.0 knows the direct Anthropic Claude 4.8+ IDs, but Bedrock
inference-profile IDs such as ``us.anthropic.claude-opus-4-8`` still need to be
classified as adaptive-thinking models before the Bedrock Converse transform is
called. The module also keeps mid-conversation ``role: "system"`` messages
cacheable on Bedrock InvokeModel (#1135), and registers effort capabilities for
Claude ids missing from LiteLLM's model map so ``output_config.effort`` is not
dropped (``register_claude_effort_capabilities``). It is loaded inside the
LiteLLM proxy process via ``sitecustomize`` and is intentionally inert for
other models.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

BEDROCK_THINKING_EFFORT_ENV = "BENCHFLOW_BEDROCK_THINKING_EFFORT"
# Claude models that use adaptive thinking with ``output_config.effort``: the
# 4.8+ point releases and every 5-family model (Opus, Sonnet, Haiku, Fable),
# including point releases such as ``claude-fable-5-1``.
BEDROCK_ADAPTIVE_THINKING_RE = re.compile(
    r"claude-(?:(?:opus|sonnet|haiku)-4-(?:8|9|1\d)"
    r"|(?:opus|sonnet|haiku|fable)-(?:[5-9]|[1-9]\d))(?!\d)"
)
BEDROCK_LITELLM_EFFORT_LIMIT_RE = re.compile(
    r"claude-(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)"
)
# Claude models whose Bedrock InvokeModel accepts ``role: "system"`` entries in
# ``messages`` (Anthropic's mid-conversation system messages).
BEDROCK_MID_CONVERSATION_SYSTEM_RE = re.compile(
    r"claude-(?:opus-4-8|opus-5|fable-5)(?!\d)"
)

# Requested efforts low→high. LiteLLM 1.88.0rc1 only accepts up to ``high`` for
# the Claude 4.x Bedrock IDs matched by ``BEDROCK_LITELLM_EFFORT_LIMIT_RE`` and
# raises on ``xhigh``/``max`` (#737), so those overrides are clamped before being
# handed to the transform. Other adaptive-thinking Bedrock models, such as Fable
# 5, keep their requested effort. This is the standalone (sandbox-deployed) copy
# of the ladder in ``litellm_config``; keep the two in sync.
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


def _should_clamp_effort(model: str | None) -> bool:
    try:
        return bool(model and BEDROCK_LITELLM_EFFORT_LIMIT_RE.search(model.lower()))
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
            if _should_clamp_effort(model):
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


# Block types that cannot carry a cache breakpoint.
_UNCACHEABLE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _supports_mid_conversation_system(model: object) -> bool:
    try:
        return bool(
            isinstance(model, str)
            and BEDROCK_MID_CONVERSATION_SYSTEM_RE.search(model.lower())
        )
    except Exception:
        return False


def _is_system_message(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"


def _cacheable_block(block: Any) -> dict[str, Any] | None:
    if isinstance(block, dict) and block.get("type") not in _UNCACHEABLE_BLOCK_TYPES:
        return block
    return None


def _relocate_trailing_system_breakpoint(messages: list[Any]) -> list[Any]:
    """Move cache breakpoints off trailing system messages onto the user turn.

    Claude Code puts its moving breakpoint on the newest system message, which
    ends the request. On Bedrock that cache entry is not read back once the next
    turn appends messages after it, so every turn rewrote the whole transcript.
    A breakpoint on the preceding user message is read back.
    """
    tail = len(messages)
    while tail > 0 and _is_system_message(messages[tail - 1]):
        tail -= 1
    if tail in (0, len(messages)):
        return messages
    target = messages[tail - 1]
    if not isinstance(target, dict) or target.get("role") != "user":
        return messages
    content = target.get("content")
    if isinstance(content, str) and content:
        blocks: list[Any] = [{"type": "text", "text": content}]
    elif isinstance(content, list) and content:
        blocks = list(content)
    else:
        return messages
    last = _cacheable_block(blocks[-1])
    if last is None:
        return messages

    moved: object = None
    trailing: list[Any] = []
    for message in messages[tail:]:
        system_content = message.get("content")
        if isinstance(system_content, list):
            new_content = []
            for block in system_content:
                if isinstance(block, dict) and "cache_control" in block:
                    block = dict(block)
                    control = block.pop("cache_control")
                    if moved is None:
                        moved = control
                new_content.append(block)
            message = {**message, "content": new_content}
        trailing.append(message)
    if moved is None:
        return messages
    if "cache_control" not in last:
        blocks[-1] = {**last, "cache_control": moved}
    return [*messages[: tail - 1], {**target, "content": blocks}, *trailing]


def _patch_bedrock_invoke_system_messages() -> None:
    """Keep mid-conversation system messages in place on Bedrock Invoke (#1135).

    LiteLLM 1.91.0 hoists every ``role: "system"`` message into the top-level
    ``system`` field. The cache breakpoint Claude Code places on the newest
    system message moves with it, so no breakpoint is left in ``messages`` and
    each turn re-bills the whole transcript as uncached input; only the system
    prefix is read from cache. Models that accept mid-conversation system
    messages on Bedrock keep them in place instead; everything else keeps
    LiteLLM's behavior.
    """
    try:
        from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
            AmazonAnthropicClaudeMessagesConfig,
        )
    except Exception:
        return

    original = (
        AmazonAnthropicClaudeMessagesConfig._normalize_system_role_messages_for_bedrock
    )

    def normalize(self: Any, anthropic_messages_request: dict) -> None:
        messages = anthropic_messages_request.get("messages")
        if not (
            isinstance(messages, list)
            and any(_is_system_message(m) for m in messages)
            and _supports_mid_conversation_system(
                anthropic_messages_request.get("model")
            )
        ):
            return original(self, anthropic_messages_request)
        # Bedrock rejects per-message fields such as ``output_config`` with
        # "Extra inputs are not permitted"; the top-level effort still applies.
        messages = [
            {"role": "system", "content": m.get("content")}
            if _is_system_message(m)
            else m
            for m in messages
        ]
        anthropic_messages_request["messages"] = _relocate_trailing_system_breakpoint(
            messages
        )
        # Same billing-header filtering LiteLLM applies when it hoists.
        system = anthropic_messages_request.get("system")
        if system is not None:
            filtered = self._filter_billing_headers_from_system(system)
            if filtered:
                anthropic_messages_request["system"] = filtered
            else:
                anthropic_messages_request.pop("system", None)
        return None

    # Marker for the fail-closed startup preflight.
    setattr(normalize, "__benchflow_bedrock_patch__", True)  # noqa: B010

    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AmazonAnthropicClaudeMessagesConfig,
        "_normalize_system_role_messages_for_bedrock",
        normalize,
    )


# Capability flags LiteLLM 1.91.0 reads before it forwards ``output_config``
# (``AnthropicConfig._model_supports_effort_param``, ``_supports_effort_level``)
# or maps ``reasoning_effort`` to adaptive thinking (``_is_adaptive_thinking_model``).
# Every Anthropic and Bedrock entry for a Claude 4.8+ or 5-family id in LiteLLM's
# model map carries exactly this set, both in the 1.91.0 backup and in the
# upstream map on BerriAI/litellm main. No price fields:
# cost stays whatever the loaded map says (unknown offline) rather than
# borrowed from another model.
CLAUDE_EFFORT_CAPABILITIES: dict[str, bool] = {
    "supports_adaptive_thinking": True,
    "supports_reasoning": True,
    "supports_output_config": True,
    "supports_xhigh_reasoning_effort": True,
    "supports_max_reasoning_effort": True,
}
# Upstream also sets ``bedrock_output_config_effort_ceiling`` to ``xhigh`` for
# Fable 5.1 and Sonnet 5. ``xhigh`` is the top of LiteLLM's
# Bedrock effort order (low < medium < high < max < xhigh), so it lowers nothing
# and is not mirrored here.
_ROUTING_PREFIXES = (
    "bedrock/converse/",
    "bedrock/invoke/",
    "bedrock/",
    "anthropic/",
    "vertex_ai/",
    "azure_ai/",
)
_BEDROCK_REGION_PREFIX_RE = re.compile(r"^[a-z-]{2,8}\.(?=anthropic\.)")


def _effort_capability_keys(model: str) -> tuple[str, list[str]]:
    """Return the provider and the model-map keys LiteLLM consults for a model.

    Bedrock ids resolve to the inference-profile id and its base model
    (``us.anthropic.claude-x`` and ``anthropic.claude-x``); other routes to the
    bare Claude id. Returns no keys for models outside the Claude 4.8+/5 family.
    """
    bare = model.strip()
    for prefix in _ROUTING_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if not bare or not BEDROCK_ADAPTIVE_THINKING_RE.search(bare.lower()):
        return "", []
    if "anthropic." in bare:
        base = _BEDROCK_REGION_PREFIX_RE.sub("", bare)
        return "bedrock", list(dict.fromkeys((bare, base)))
    return "anthropic", [bare]


def register_claude_effort_capabilities(models: Iterable[str]) -> list[str]:
    """Register effort capabilities for Claude ids LiteLLM's model map lacks.

    LiteLLM 1.91.0 removes ``output_config`` (the effort setting) whenever the
    model map does not advertise effort support for the model, and without
    ``supports_adaptive_thinking`` it maps ``reasoning_effort`` to nothing or to
    the legacy ``thinking.type=enabled`` shape. The proxy loads LiteLLM's
    upstream map from GitHub at startup and falls back to the pinned backup when
    it cannot, and the backup has no Claude id newer than Fable 5 / Opus 4.8.
    So on a proxy without GitHub access, ``us.anthropic.claude-fable-5-1`` and
    other new ids ran at the provider's default effort.

    Only values LiteLLM cannot resolve for the model are added; entries LiteLLM
    already has keep their values. Returns the keys updated.
    """
    try:
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    except Exception:
        return []

    updated: list[str] = []
    for model in models:
        provider, keys = _effort_capability_keys(str(model))
        for key in keys:
            missing = {
                flag: value
                for flag, value in CLAUDE_EFFORT_CAPABILITIES.items()
                if AnthropicModelInfo._get_model_capability(key, flag) is None
            }
            if not missing:
                continue
            entry = litellm.model_cost.setdefault(
                key, {"litellm_provider": provider, "mode": "chat"}
            )
            entry.update(missing)
            updated.append(key)
    if updated:
        # ``litellm.register_model`` would also work, but it logs a misleading
        # "cache cost fields will default to 0" warning for price-less entries.
        try:
            from litellm.utils import _invalidate_model_cost_lowercase_map

            _invalidate_model_cost_lowercase_map()
        except Exception:
            pass
    return updated


_patch_anthropic_gate()
_patch_bedrock_effort()
_patch_cost_map()
_patch_bedrock_invoke_system_messages()
