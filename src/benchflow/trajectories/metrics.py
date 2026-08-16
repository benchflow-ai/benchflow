"""Metrics derived from structured trajectory events."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SKILL_TOOL_NAMES = frozenset({"invoke_skill", "activate_skill", "skill"})

# ``skill`` is unambiguous as a tool *kind* or tool *name*, but a title is
# display text — a file or command merely named "skill" can produce it. A title
# therefore proves a skill invocation outright only under the explicit
# spellings; the bare word is honored for unclassified kinds alone, behind the
# same gate as content sniffing (see ``is_skill_invocation_event``). Deriving
# the subset here keeps the two sets from drifting apart — the inline-literal
# drift is how PR #597 lost the ``skill`` spelling in the first place.
_EXPLICIT_SKILL_TOOL_NAMES = _SKILL_TOOL_NAMES - {"skill"}

# OpenHands legacy ACP artifacts render an invoke-skill *result* as a text block
# that begins with the tool header (``Tool: invoke_skill``) and carries a
# ``[skill: <name>]`` result marker. Anchoring the header to the start of the
# block (``\A``) is the structural signal that the tool itself was invoke_skill
# — not that some other tool's output merely quoted such text mid-stream.
_SKILL_RESULT_HEADER_RE = re.compile(
    r"\A\s*Tool:\s*(?:invoke_skill|activate_skill)\b", re.IGNORECASE
)
_SKILL_RESULT_MARKER = "[skill:"

# Harnesses that do not set the canonical kind open a skill *result* with a
# recognizable envelope: the pinned opencode (1.17.x) renders a markdown
# ``## Skill: <name>`` header, opencode 1.18.x a ``<skill_content name="...">``
# element, and claude-agent-acp a ``Launching skill: <name>`` line. Each is
# anchored with ``\A`` for the same reason as the OpenHands header above: a
# tool whose output merely quotes such a marker mid-stream is not a skill
# invocation. This content path is what keeps the counter honest when the
# title mutates across exports — issue #998 records a trajectory export whose
# title carries the serialized arguments (``skill {"name": ...}``), which the
# bare-title branch cannot see.
_SKILL_CONTENT_ENVELOPE_RE = re.compile(
    r"\A\s*(?:<skill_content\s+name\s*=\s*\"[^\"]+\""
    r"|#{1,6}\s*Skill:\s*\S"
    r"|Launching\s+skill:\s*\S)",
    re.IGNORECASE,
)

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
    """Return whether tool-call content is a skill-invocation tool result.

    Recognized envelopes, each anchored to the start of a tool-result text
    block:

    * OpenHands legacy — a ``Tool: invoke_skill`` / ``Tool: activate_skill``
      header carrying a ``[skill: ...]`` marker.
    * opencode — the skill body behind a ``## Skill: <name>`` markdown header
      (1.17.x, the pinned agent) or wrapped in a ``<skill_content name="...">``
      element (1.18.x).
    * claude-agent-acp — a ``Launching skill: <name>`` line.

    The anchoring is what distinguishes a genuine skill result from ordinary
    output that merely quotes such text. This is intentionally narrow; it is the
    only text-derived path and is paired with the no-skill experiment-health
    invariant in the result checker as a backstop.
    """
    for text in _tool_result_texts(content):
        if _SKILL_RESULT_HEADER_RE.match(text) and _SKILL_RESULT_MARKER in text.lower():
            return True
        if _SKILL_CONTENT_ENVELOPE_RE.match(text):
            return True
    return False


def is_skill_invocation_event(event: Mapping[str, Any]) -> bool:
    """Return whether an ACP trajectory event represents a skill invocation.

    This is the single source of truth for "is this tool call a skill
    invocation", shared by historical trajectory rescans and live ACP capture
    (:mod:`benchflow.acp.session`).

    ``kind == "skill"`` is the canonical representation. Identity signals (tool
    kind, tool name, or title naming ``invoke_skill`` / ``activate_skill``) are
    trusted outright. Harnesses that do not set the canonical kind are matched
    on their own structured shape -- OpenHands legacy emits ``invoke_skill`` as
    ``kind == "other"`` with the tool result in ``content``, opencode emits
    ``kind == "other"`` / ``title == "skill"`` with a versioned result envelope,
    and claude-agent-acp emits ``kind == "other"`` / ``title == "Skill"`` with a
    ``Launching skill:`` line. All are recognized only when the tool kind is
    unclassified, so an ordinary ``read`` / ``execute`` / ``search`` tool whose
    title or output happens to mention a skill is never reclassified.
    """
    if event.get("type") != "tool_call":
        return False

    kind = _normalized_tool_name(event.get("kind"))
    if kind in _SKILL_TOOL_NAMES:
        return True

    if _event_tool_name(event) in _SKILL_TOOL_NAMES:
        return True

    title = _normalized_tool_name(event.get("title"))
    if title in _EXPLICIT_SKILL_TOOL_NAMES:
        return True

    if kind not in _CONTENT_SNIFFABLE_KINDS:
        return False

    # opencode and claude-agent-acp label the call simply ``skill``. That bare
    # word is too generic to trust on a tool that already declares a real ACP
    # kind, so unlike the explicit ``invoke_skill`` spellings above it is only
    # honored for the unclassified kinds -- the same gate the content sniffing
    # sits behind.
    if title in _SKILL_TOOL_NAMES:
        return True

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
