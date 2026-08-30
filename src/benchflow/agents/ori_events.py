#!/usr/bin/env python3
"""Translate decoded Ori runtime records into ACP session updates.

Installed beside :mod:`ori_acp_shim` and :mod:`ori_jsonl`; stdlib-only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:  # Installed scripts are sibling top-level modules.
    from ori_jsonl import OriUsage, json_text, runtime_event
except ModuleNotFoundError:  # Package import used by BenchFlow's unit tests.
    from .ori_jsonl import (  # type: ignore[no-redef]
        OriUsage,
        json_text,
        runtime_event,
    )

Send = Callable[[dict[str, Any]], None]

_TEXT_EVENTS = frozenset({"assistant.text.delta", "content.delta"})
_REASONING_EVENTS = frozenset({"reasoning.delta"})
_TERMINAL_EVENTS = frozenset(
    {"turn.succeeded", "turn.failed", "session.succeeded", "session.failed"}
)
_EVIDENCE_LIMIT = 8000
_TOOL_CONTENT_LIMIT = 12000


def _content(value: object) -> list[dict[str, object]]:
    return [
        {
            "type": "content",
            "content": {
                "type": "text",
                "text": json_text(value)[:_TOOL_CONTENT_LIMIT],
            },
        }
    ]


def _tool_kind(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bash", "shell", "terminal"}:
        return "execute"
    if lowered in {"read", "read_file"}:
        return "read"
    if lowered in {"write", "write_file", "edit", "apply_patch"}:
        return "edit"
    if lowered in {"glob", "grep", "search"}:
        return "search"
    if lowered in {"browser", "web", "web_search", "web_fetch"}:
        return "fetch"
    if "plan" in lowered or "think" in lowered:
        return "think"
    return "other"


def _tool_title(name: str, tool_input: object) -> str:
    if isinstance(tool_input, dict):
        values: dict[str, Any] = {str(key): value for key, value in tool_input.items()}
        for key in (
            "command",
            "path",
            "file_path",
            "pattern",
            "query",
            "prompt",
            "name",
        ):
            value = values.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
    return name or "tool"


class TurnTranslator:
    """Stateful mapping from one Ori stream onto one ACP prompt."""

    def __init__(self, session_id: str, send: Send) -> None:
        self.session_id = session_id
        self.native_session_id: str | None = None
        self.result: dict[str, Any] | None = None
        self.turn_usage: OriUsage | None = None
        self.last_diagnostic = ""
        self._known_tools: set[str] = set()
        self._send = send

    def consume_diagnostic(self, line_number: int, raw: str) -> None:
        self.last_diagnostic = raw
        self._emit_text(
            f"[ori diagnostic line {line_number}] {raw[:_EVIDENCE_LIMIT]}\n",
            thought=True,
        )

    def consume_document(self, document: dict[str, Any]) -> None:
        if document.get("kind") == "result":
            self.result = document
            self._remember_session_id(document.get("sessionId"))
            self._emit_evidence("result", document)
            return

        event = runtime_event(document)
        if event is None:
            self._emit_evidence("record", document)
            return

        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        self._remember_session_id(event.get("sessionId") or payload.get("sessionId"))
        event_type = str(event.get("type") or "")

        if event_type in _TEXT_EVENTS:
            delta = payload.get("delta")
            if isinstance(delta, str):
                self._emit_text(delta)
            return
        if event_type in _REASONING_EVENTS:
            delta = payload.get("delta")
            if isinstance(delta, str):
                self._emit_text(delta, thought=True)
            return
        if event_type == "tool.started":
            self._start_tool(payload)
            return
        if event_type in {"tool.progress", "tool.succeeded", "tool.failed"}:
            self._update_tool(event_type, payload)
            return
        if event_type in _TERMINAL_EVENTS:
            usage = OriUsage.from_ori(payload.get("usage"))
            if usage is not None:
                # Some streams carry the same snapshot in both turn.* and
                # session.*. Replacement avoids double-counting one turn.
                self.turn_usage = usage
        self._emit_evidence(event_type or "runtime.event", event)

    def _notification(self, update: dict[str, Any]) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": self.session_id, "update": update},
            }
        )

    def _emit_text(self, text: str, *, thought: bool = False) -> None:
        if not text:
            return
        self._notification(
            {
                "sessionUpdate": (
                    "agent_thought_chunk" if thought else "agent_message_chunk"
                ),
                "content": {"type": "text", "text": text},
            }
        )

    def _remember_session_id(self, value: object) -> None:
        if isinstance(value, str) and value:
            self.native_session_id = value

    def _emit_evidence(self, label: str, value: object) -> None:
        rendered = json_text(value)[:_EVIDENCE_LIMIT]
        self._emit_text(f"[ori {label}] {rendered}\n", thought=True)

    def _tool_id(self, payload: dict[str, Any]) -> str:
        value = payload.get("toolCallId")
        if isinstance(value, str) and value:
            return value
        return f"ori-tool-{len(self._known_tools) + 1}"

    def _start_tool(self, payload: dict[str, Any]) -> str:
        tool_id = self._tool_id(payload)
        if tool_id in self._known_tools:
            return tool_id
        self._known_tools.add(tool_id)
        name = str(payload.get("name") or "tool")
        tool_input = payload.get("input")
        self._notification(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_id,
                "title": _tool_title(name, tool_input),
                "kind": _tool_kind(name),
                "status": "in_progress",
            }
        )
        if tool_input is not None:
            self._notification(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_id,
                    "status": "in_progress",
                    "content": _content(tool_input),
                }
            )
        return tool_id

    def _update_tool(self, event_type: str, payload: dict[str, Any]) -> None:
        tool_id = self._tool_id(payload)
        if tool_id not in self._known_tools:
            tool_id = self._start_tool(payload)
        status = {
            "tool.progress": "in_progress",
            "tool.succeeded": "completed",
            "tool.failed": "failed",
        }[event_type]
        output = payload.get("result")
        if output is None:
            output = payload.get("partialResult")
        if output is None and event_type == "tool.failed":
            output = payload.get("failure") or payload.get("error")
        update: dict[str, Any] = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": tool_id,
            "status": status,
        }
        if output is not None:
            update["content"] = _content(output)
        self._notification(update)


__all__ = ["TurnTranslator"]
