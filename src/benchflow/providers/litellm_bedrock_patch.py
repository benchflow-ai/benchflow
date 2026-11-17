"""LiteLLM startup patches for Claude 4.8+ models (Bedrock and direct).

LiteLLM 1.89.0 knows the direct Anthropic Claude 4.8+ IDs, but Bedrock
inference-profile IDs such as ``us.anthropic.claude-opus-4-8`` still need to be
classified as adaptive-thinking models before the Bedrock Converse transform is
called. The module also keeps mid-conversation ``role: "system"`` messages
cacheable on Bedrock InvokeModel (#1135), and registers effort capabilities for
Claude ids missing from LiteLLM's model map so ``output_config.effort`` is not
dropped (``register_claude_effort_capabilities``). The effort a request carries
is the effort sent upstream: LiteLLM's Messages adapters no longer lower it
(``_patch_effort_passthrough``), its Bedrock ceiling no longer caps it
(``_patch_bedrock_effort_ceiling``), a level a model does not take fails
instead of becoming a thinking budget (``_patch_effort_level_gate``), and Opus
4.5's Converse calls carry the effort beside their thinking budget. It is loaded inside the LiteLLM proxy process via
``sitecustomize`` and is intentionally inert for other models.
"""

from __future__ import annotations

import importlib
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
# Claude models whose Bedrock InvokeModel accepts ``role: "system"`` entries in
# ``messages`` (Anthropic's mid-conversation system messages).
BEDROCK_MID_CONVERSATION_SYSTEM_RE = re.compile(
    r"claude-(?:opus-4-8|opus-5|fable-5)(?!\d)"
)

# Requested efforts low→high. This is the standalone (sandbox-deployed) copy
# of the ladder in ``litellm_config``; keep the two in sync. Every level is sent
# as asked: LiteLLM 1.88.0rc1 raised on ``xhigh``/``max`` for the Claude 4.x
# Bedrock ids (#737) and they were lowered to ``high`` here, but the pinned
# LiteLLM 1.103.3 takes and forwards every level.
_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_VALID_EFFORTS = set(_EFFORT_LADDER)


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

    # LiteLLM passes the provider as a second argument (e.g. "bedrock"), so
    # forward whatever the caller sent instead of fixing the vendor signature.
    def gate(model: str, *args: Any, **kwargs: Any) -> bool:
        if _is_new_bedrock_claude(model):
            return True
        return original(model or "", *args, **kwargs)

    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AnthropicConfig,
        "_is_adaptive_thinking_model",
        staticmethod(gate),
    )


_CLAUDE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _effort_beside_thinking_budget(
    model: str, reasoning_effort: str, optional_params: dict[Any, Any]
) -> None:
    """Send ``output_config.effort`` to a Bedrock Claude that takes it without
    adaptive thinking (Opus 4.5). LiteLLM's Converse mapping gave it a thinking
    budget and no effort, so every ``-e`` level ran at the model's default."""
    if reasoning_effort not in _CLAUDE_EFFORTS or "claude" not in model.lower():
        return
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        if AnthropicConfig._is_adaptive_thinking_model(
            model, "bedrock"
        ) or not AnthropicConfig._model_supports_effort_param(model, "bedrock"):
            return
    except Exception:
        return
    output_config = optional_params.get("output_config")
    output_config = dict(output_config) if isinstance(output_config, dict) else {}
    output_config.setdefault("effort", reasoning_effort)
    optional_params["output_config"] = output_config


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
        original(self, model, reasoning_effort, optional_params)
        _effort_beside_thinking_budget(model, reasoning_effort, optional_params)

    # Marker for the fail-closed startup preflight (#602): lets the runtime
    # verify this override is installed without importing this module (which
    # would itself apply the patches and mask a sitecustomize load failure).
    setattr(handle, "__benchflow_bedrock_patch__", True)  # noqa: B010

    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AmazonConverseConfig,
        "_handle_reasoning_effort_parameter",
        handle,
    )


def _patch_effort_passthrough() -> None:
    """Send the effort a request carries, never a lower one.

    LiteLLM 1.103.3's Anthropic Messages adapters, to the Responses API and to
    chat completions (``anthropic/experimental_pass_through``), pass each
    effort through ``normalize_reasoning_effort_value``. That lowers a level
    its model map cannot confirm: ``max`` to ``xhigh``, then ``high``, and
    straight to ``high`` for a model the map does not know. So
    ``bench run -m openrouter/anthropic/claude-opus-5.5 -e max`` sent
    ``reasoning.effort: high`` on every call. BenchFlow decides before launch
    which levels a run may ask for; here the level asked for is the level
    sent, and an upstream that cannot take it fails the call, loudly.
    """
    try:
        from litellm.llms.anthropic.experimental_pass_through import utils
    except Exception:
        return

    def keep(effort: str, model: str, custom_llm_provider: str | None = None) -> str:
        return effort

    # Marker for the fail-closed startup preflight.
    setattr(keep, "__benchflow_bedrock_patch__", True)  # noqa: B010
    # Both adapters import the function from this module when they call it.
    setattr(utils, "normalize_reasoning_effort_value", keep)  # noqa: B010


def _patch_effort_level_gate() -> None:
    """Refuse a level a Claude that takes effort does not take, instead of
    turning it into a thinking budget.

    Claude Code sends adaptive thinking and ``output_config.effort`` to every
    model. For a model without adaptive thinking, LiteLLM's Messages
    pass-through keeps the effort only when the model takes that level, and
    otherwise sends a thinking budget and no effort: Opus 4.5 at ``xhigh`` or
    ``max`` ran on a budget, on Bedrock, Vertex and the Anthropic API alike.
    """
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.llms.anthropic.common_utils import AnthropicError
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )
    except Exception:
        return

    original = AnthropicMessagesConfig._translate_adaptive_effort_for_non_adaptive_model

    def translate(
        model: str,
        optional_params: dict[Any, Any],
        max_tokens: int | None,
        custom_llm_provider: str,
    ) -> None:
        output_config = optional_params.get("output_config")
        effort = (
            output_config.get("effort") if isinstance(output_config, dict) else None
        )
        if (
            isinstance(effort, str)
            and not AnthropicConfig._is_adaptive_thinking_model(
                model, custom_llm_provider
            )
            and AnthropicConfig._model_supports_effort_param(model, custom_llm_provider)
        ):
            error = AnthropicConfig._validate_effort_for_model(
                model, effort, custom_llm_provider
            )
            if error is not None:
                raise AnthropicError(message=error, status_code=400)
        return original(model, optional_params, max_tokens, custom_llm_provider)

    # Marker for the fail-closed startup preflight.
    setattr(translate, "__benchflow_bedrock_patch__", True)  # noqa: B010
    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AnthropicMessagesConfig,
        "_translate_adaptive_effort_for_non_adaptive_model",
        staticmethod(translate),
    )


# The modules that call LiteLLM's Bedrock effort ceiling, each through the
# name it imported: the helper's own, Converse, and both InvokeModel paths.
_BEDROCK_EFFORT_CEILING_MODULES = (
    "litellm.llms.bedrock.common_utils",
    "litellm.llms.bedrock.chat.converse_transformation",
    "litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation",
    "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation",
)


def _patch_bedrock_effort_ceiling() -> None:
    """Send a Bedrock Claude effort as asked, not capped at the model map's
    ceiling.

    LiteLLM 1.103.3 lowers ``output_config.effort`` above a model's
    ``bedrock_output_config_effort_ceiling`` (in the order low < medium < high
    < max < xhigh), on Converse and on both InvokeModel paths, so Opus 4.6's
    ``xhigh`` went out as ``max``. A level the model does not take now fails
    the call instead of running lower.
    """

    def keep(model: str, output_config: object) -> None:
        return None

    # Marker for the fail-closed startup preflight.
    setattr(keep, "__benchflow_bedrock_patch__", True)  # noqa: B010
    for name in _BEDROCK_EFFORT_CEILING_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(module, "normalize_bedrock_opus_output_config_effort"):
            setattr(module, "normalize_bedrock_opus_output_config_effort", keep)  # noqa: B010


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
    """Keep mid-conversation system messages cacheable on Bedrock Invoke (#1135).

    Claude Code sends mid-conversation ``role: "system"`` messages and puts its
    moving cache breakpoint on the newest one. LiteLLM 1.103 hoists only the
    leading system messages; a mid-conversation one stays in place on the
    models its map flags ``supports_mid_conversation_system`` and becomes a
    user turn in place on the others. On Bedrock, three things are added:

    - a system message stays in place only on the models BenchFlow has had
      Bedrock accept one from (``BEDROCK_MID_CONVERSATION_SYSTEM_RE``); on any
      other model LiteLLM's map flags, it becomes a user turn as on the rest;
    - one kept in place carries only its content: Bedrock rejects per-message
      fields such as ``output_config`` with "Extra inputs are not permitted";
    - the breakpoint on trailing system messages moves to the user turn before
      them, where Bedrock reads it back, so a turn does not re-bill the whole
      transcript as uncached input.
    """
    try:
        from litellm.llms.anthropic.experimental_pass_through.messages.mid_conversation_system import (
            convert_mid_conversation_system_turns,
        )
        from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
            AmazonAnthropicClaudeMessagesConfig,
        )
    except Exception:
        return

    original = AmazonAnthropicClaudeMessagesConfig._normalize_system_role_messages

    def normalize(
        self: Any, anthropic_messages_request: dict, *args: Any, **kwargs: Any
    ) -> None:
        original(self, anthropic_messages_request, *args, **kwargs)
        messages = anthropic_messages_request.get("messages")
        if not (
            isinstance(messages, list) and any(_is_system_message(m) for m in messages)
        ):
            return None
        model = kwargs.get("model", args[0] if args else None)
        if not _supports_mid_conversation_system(
            model or anthropic_messages_request.get("model")
        ):
            anthropic_messages_request["messages"] = list(
                convert_mid_conversation_system_turns(messages)
            )
            return None
        anthropic_messages_request["messages"] = _relocate_trailing_system_breakpoint(
            [
                {"role": "system", "content": m.get("content")}
                if _is_system_message(m)
                else m
                for m in messages
            ]
        )
        return None

    # Marker for the fail-closed startup preflight.
    setattr(normalize, "__benchflow_bedrock_patch__", True)  # noqa: B010

    setattr(  # noqa: B010 - avoids static type narrowing on monkey-patched vendor API
        AmazonAnthropicClaudeMessagesConfig,
        "_normalize_system_role_messages",
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
# The current Claude models and the efforts each takes, from Anthropic's
# model reference (2026-11-03): every one takes low to max with adaptive
# thinking. Each is registered on every route shape BenchFlow ships (the
# Anthropic API, OpenRouter's ``anthropic/`` ids, Bedrock and Vertex), so
# LiteLLM keeps the effort a request carries for a model its map lacks.
CLAUDE_EFFORT_LEVELS: dict[str, tuple[str, ...]] = {
    "claude-opus-5-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-sonnet-5-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-haiku-5-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-fable-5-2": ("low", "medium", "high", "xhigh", "max"),
}


# The Claude 4.x models that take ``output_config.effort``, as Anthropic's
# model reference documents them (2026-11-13): Opus 4.7 low to max; Opus 4.6
# and Sonnet 4.6 low, medium, high and max, with adaptive thinking; Opus 4.5
# low, medium and high, without it. Registered where LiteLLM's map lacks an
# id, so a missing entry neither drops the effort nor lets through a level the
# model does not take. Sonnet 4.5 and Haiku 4.5 take no effort.
_CLAUDE_4X_EFFORT_CAPABILITIES: tuple[tuple[re.Pattern[str], dict[str, bool]], ...] = (
    (re.compile(r"claude-opus-4-7(?!\d)"), CLAUDE_EFFORT_CAPABILITIES),
    (
        re.compile(r"claude-(?:opus|sonnet)-4-6(?!\d)"),
        {**CLAUDE_EFFORT_CAPABILITIES, "supports_xhigh_reasoning_effort": False},
    ),
    (
        re.compile(r"claude-opus-4-5(?!\d)"),
        {
            "supports_adaptive_thinking": False,
            "supports_reasoning": True,
            "supports_output_config": True,
            "supports_xhigh_reasoning_effort": False,
            "supports_max_reasoning_effort": False,
        },
    ),
)


def _effort_capabilities(model: str) -> dict[str, bool]:
    """The effort capabilities a Claude id takes, or none for another model."""
    name = model.lower()
    if BEDROCK_ADAPTIVE_THINKING_RE.search(name):
        return CLAUDE_EFFORT_CAPABILITIES
    for pattern, capabilities in _CLAUDE_4X_EFFORT_CAPABILITIES:
        if pattern.search(name):
            return capabilities
    return {}


def claude_route_models() -> list[str]:
    """Each model of :data:`CLAUDE_EFFORT_LEVELS` as each route names it."""
    models: list[str] = []
    for model in CLAUDE_EFFORT_LEVELS:
        dotted = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)
        models += [
            f"anthropic/{model}",
            f"bedrock/us.anthropic.{model}",
            f"bedrock/global.anthropic.{model}",
            f"bedrock/anthropic.{model}",
            f"vertex_ai/{model}",
            f"openrouter/anthropic/{dotted}",
        ]
    return models


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
    bare Claude id. Returns no keys for a model without effort capabilities
    (:func:`_effort_capabilities`).
    """
    bare = model.strip()
    if bare.startswith("openrouter/"):
        # LiteLLM's own OpenRouter provider reads its map under the whole id;
        # BenchFlow's OpenRouter route (``openai/`` and OpenRouter's address)
        # is priced from OpenRouter's bill, so no bare entry is made for it.
        tail = bare.removeprefix("openrouter/")
        if _effort_capabilities(tail):
            return "openrouter", [bare]
        return "", []
    for prefix in _ROUTING_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if not bare or not _effort_capabilities(bare):
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
    So a proxy that could not reach GitHub ran ``us.anthropic.claude-fable-5-1``
    and other new ids at the provider's default effort.

    Only values LiteLLM cannot resolve for the model are added; entries LiteLLM
    already has keep their values. The current Claude models on every route
    shape (:func:`claude_route_models`) are registered too, after *models*.
    Returns the keys updated for *models*.
    """
    try:
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    except Exception:
        return []

    def register(names: Iterable[str]) -> list[str]:
        done: list[str] = []
        for model in names:
            provider, keys = _effort_capability_keys(str(model))
            for key in keys:
                missing = {
                    flag: value
                    for flag, value in _effort_capabilities(key).items()
                    if AnthropicModelInfo._get_model_capability(key, flag) is None
                }
                if not missing:
                    continue
                entry = litellm.model_cost.setdefault(
                    key, {"litellm_provider": provider, "mode": "chat"}
                )
                entry.update(missing)
                done.append(key)
        return done

    updated = register(models)
    if register(claude_route_models()) or updated:
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
_patch_effort_passthrough()
_patch_bedrock_effort_ceiling()
_patch_effort_level_gate()
_patch_cost_map()
_patch_bedrock_invoke_system_messages()
