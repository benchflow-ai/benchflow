"""Metrics derived from structured trajectory events."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SKILL_TOOL_NAMES = frozenset({"invoke_skill", "activate_skill", "skill"})

# OpenHands legacy ACP artifacts render an invoke-skill *result* as a text block
# that begins with the tool header (``Tool: invoke_skill``) and carries a
# ``[skill: <name>]`` result marker. Anchoring the header to the start of the
# block (``\A``) is the structural signal that the tool itself was invoke_skill
# — not that some other tool's output merely quoted such text mid-stream.
_SKILL_RESULT_HEADER_RE = re.compile(
    r"\A\s*Tool:\s*(?:invoke_skill|activate_skill)\b", re.IGNORECASE
)
_SKILL_RESULT_MARKER = "[skill:"

# Only these unclassified tool kinds are eligible for content sniffing. Any tool
# carrying a real ACP kind (read, edit, execute, search, fetch, ...) is trusted
# as-is and never reinterpreted from its output text.
_CONTENT_SNIFFABLE_KINDS = frozenset({"", "other", "tool"})


def _normalized_tool_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_")


def _tool_result_texts(content: Any):
    """Yield the text of structured ACP tool-call *result* blocks.

    ACP serializes tool-call content as a list of blocks. The OpenHands legacy
    invoke-skill envelope is a ``content`` block wrapping a text ``ContentBlock``
    (``{"type": "content", "content": {"type": "text", "text": ...}}``); some
    shims inline the text block directly. Only those structured tool-result
    texts are inspected — nested metadata (locations, diffs, raw inputs) is
    deliberately ignored so an ordinary tool's payload cannot impersonate a
    skill result by burying marker text in an unrelated field.
    """
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, Mapping):
            continue
        body = block.get("content")
        if isinstance(body, Mapping) and body.get("type") == "text":
            text = body.get("text")
            if isinstance(text, str):
                yield text
        elif block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                yield text


def _event_tool_name(event: Mapping[str, Any]) -> str:
    for key in ("tool_name", "toolName", "name", "function_name", "functionName"):
        name = _normalized_tool_name(event.get(key))
        if name:
            return name

    tool = event.get("tool")
    if isinstance(tool, Mapping):
        name = _normalized_tool_name(tool.get("name"))
        if name:
            return name

    function = event.get("function")
    if isinstance(function, Mapping):
        name = _normalized_tool_name(function.get("name"))
        if name:
            return name

    return ""


def content_contains_skill_invocation_tool(content: Any) -> bool:
    """Return whether tool-call content is an OpenHands invoke-skill result.

    Requires the structured legacy envelope: a tool-result text block that
    *begins* with the ``Tool: invoke_skill`` / ``Tool: activate_skill`` header
    and carries a ``[skill: ...]`` marker. The anchored header is what
    distinguishes a genuine invoke-skill tool result from ordinary output that
    merely quotes such text. This is intentionally narrow; it is the only
    text-derived path and is paired with the no-skill experiment-health
    invariant in the result checker as a backstop.
    """
    for text in _tool_result_texts(content):
        if _SKILL_RESULT_HEADER_RE.match(text) and _SKILL_RESULT_MARKER in text.lower():
            return True
    return False


def is_skill_invocation_event(event: Mapping[str, Any]) -> bool:
    """Return whether an ACP trajectory event represents a skill invocation.

    This is the single source of truth for "is this tool call a skill
    invocation", shared by historical trajectory rescans and live ACP capture
    (:mod:`benchflow.acp.session`).

    ``kind == "skill"`` is the canonical representation. Identity signals (tool
    kind, tool name, or title naming ``invoke_skill`` / ``activate_skill``) are
    trusted outright. Older OpenHands ACP artifacts emitted ``invoke_skill``
    calls as ``kind == "other"`` with the structured tool result in ``content``;
    that shape is recognized only when the tool kind is unclassified, so an
    ordinary ``read`` / ``execute`` / ``search`` tool whose output happens to
    quote the marker is never reclassified.
    """
    if event.get("type") != "tool_call":
        return False

    kind = _normalized_tool_name(event.get("kind"))
    if kind in _SKILL_TOOL_NAMES:
        return True

    if _event_tool_name(event) in _SKILL_TOOL_NAMES:
        return True

    title = _normalized_tool_name(event.get("title"))
    if title in {"invoke_skill", "activate_skill"}:
        return True

    if kind not in _CONTENT_SNIFFABLE_KINDS:
        return False

    return content_contains_skill_invocation_tool(event.get("content"))


def count_skill_invocations(trajectory: list[dict[str, Any]]) -> int:
    """Count ACP skill invocations from structured tool-call events.

    BenchFlow records ACP tool calls as dict events. A canonical skill
    invocation has ``type == "tool_call"`` and ``kind == "skill"``. Some legacy
    harness artifacts have structured invoke-skill evidence in tool-call
    content instead, so the counter accepts those shapes while still ignoring
    ordinary agent messages and display text outside tool-call events.
    """
    return sum(
        1
        for event in trajectory
        if isinstance(event, Mapping) and is_skill_invocation_event(event)
    )


def result_skill_invocations(result: Mapping[str, Any]) -> int:
    """Return a result artifact's skill invocation count.

    New artifacts expose ``n_skill_invocations`` at the top level and inside
    ``agent_result``. Older artifacts may have neither; they are treated as
    zero so aggregate readers remain backward compatible.
    """
    value = result.get("n_skill_invocations")
    if value is None:
        agent_result = result.get("agent_result")
        if isinstance(agent_result, Mapping):
            value = agent_result.get("n_skill_invocations")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
