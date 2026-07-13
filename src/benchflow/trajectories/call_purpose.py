"""Classify provider calls that belong to an agent versus harness helpers."""

from __future__ import annotations

from typing import Any


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "")
            for item in content
            if isinstance(item, dict)
        )
    return ""


def infer_call_purpose(*, agent_name: str, request_body: dict[str, Any]) -> str:
    """Return a stable purpose label for one captured LLM request."""
    if agent_name != "opencode":
        return "agent"
    messages = request_body.get("messages")
    system = "\n".join(
        _message_text(message.get("content"))
        for message in messages or []
        if isinstance(message, dict) and message.get("role") == "system"
    ).lstrip()
    if system.startswith("You are a title generator."):
        return "title"
    if system.startswith("Summarize what was done in this conversation."):
        return "summary"
    if system.startswith("You are an anchored context summarization assistant"):
        return "compaction"
    tools = request_body.get("tools")
    if isinstance(tools, list) and tools:
        return "agent"
    return "helper"
