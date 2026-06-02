"""Metrics derived from structured trajectory events."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SKILL_TOOL_NAMES = frozenset({"invoke_skill", "activate_skill", "skill"})
_SKILL_TOOL_LINE_RE = re.compile(r"(?im)^\s*Tool:\s*(invoke_skill|activate_skill)\s*$")


def _normalized_tool_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_")


def _iter_text_fragments(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_text_fragments(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _iter_text_fragments(child)


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
    """Return whether structured tool-call content is an invoke-skill result."""
    text = "\n".join(_iter_text_fragments(content))
    if not text:
        return False
    return bool(_SKILL_TOOL_LINE_RE.search(text)) and "[skill:" in text.lower()


def is_skill_invocation_event(event: Mapping[str, Any]) -> bool:
    """Return whether an ACP trajectory event represents a skill invocation.

    ``kind == "skill"`` is the canonical representation. Older OpenHands ACP
    artifacts emitted ``invoke_skill`` calls as ``kind == "other"`` with the
    structured tool result in ``content``; keep recognizing that shape so
    existing uploaded trajectories can be rescanned accurately.
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

    if kind and kind not in {"other", "tool"}:
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
