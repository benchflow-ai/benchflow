#!/usr/bin/env python3
"""Typed, tolerant decoding helpers for Ori's headless JSONL stream.

This module is installed beside the Ori ACP shim inside the sandbox.  It is
therefore deliberately stdlib-only and must not import :mod:`benchflow`.

Ori normally writes one JSON object per line, but startup/provider diagnostics
can be written as plain text before the terminal JSON result.  Those lines are
evidence, not framing errors: :func:`decode_line` preserves them as diagnostic
records so callers can keep decoding later structured events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DecodedLine:
    """One physical Ori output line without lossy error coercion."""

    line_number: int
    kind: Literal["document", "diagnostic"]
    raw: str
    document: dict[str, Any] | None = None


@dataclass(frozen=True)
class OriUsage:
    """Token usage for one Ori turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    thought_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """ACP total: the sum of every reported token component."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cached_read_tokens
            + self.cached_write_tokens
            + self.thought_tokens
        )

    def __add__(self, other: OriUsage) -> OriUsage:
        return OriUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_read_tokens=(self.cached_read_tokens + other.cached_read_tokens),
            cached_write_tokens=(self.cached_write_tokens + other.cached_write_tokens),
            thought_tokens=self.thought_tokens + other.thought_tokens,
        )

    def as_acp(self) -> dict[str, int]:
        """Return the cumulative ACP ``PromptResponse.usage`` wire shape."""
        return {
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "cachedReadTokens": self.cached_read_tokens,
            "cachedWriteTokens": self.cached_write_tokens,
            "thoughtTokens": self.thought_tokens,
            "totalTokens": self.total_tokens,
        }

    @classmethod
    def from_ori(cls, value: object) -> OriUsage | None:
        """Decode Ori's camelCase usage object, rejecting non-mappings."""
        if not isinstance(value, dict):
            return None
        usage: dict[str, Any] = {str(key): item for key, item in value.items()}
        return cls(
            input_tokens=_nonnegative_int(usage.get("inputTokens")),
            output_tokens=_nonnegative_int(usage.get("outputTokens")),
            cached_read_tokens=_nonnegative_int(usage.get("cacheReadTokens")),
            cached_write_tokens=_nonnegative_int(usage.get("cacheCreationTokens")),
            # Ori currently exposes reasoning inside outputTokens and has no
            # separate thought-token counter. Keep the ACP field explicit.
            thought_tokens=0,
        )


def _nonnegative_int(value: object) -> int:
    try:
        return max(int(str(value)), 0)
    except (TypeError, ValueError):
        return 0


def decode_line(line: str, line_number: int) -> DecodedLine | None:
    """Decode one line, preserving plain/invalid JSON as a diagnostic.

    Returning ``None`` for whitespace lets the streaming caller ignore blank
    framing without losing the original physical line number.
    """
    raw = line.rstrip("\r\n")
    if not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return DecodedLine(line_number, "diagnostic", raw)
    if not isinstance(value, dict):
        return DecodedLine(line_number, "diagnostic", raw)
    return DecodedLine(line_number, "document", raw, value)


def runtime_event(document: dict[str, Any]) -> dict[str, Any] | None:
    """Unwrap an Ori ``runtime.event`` document, if this is one."""
    wrapper = document.get("event")
    if not isinstance(wrapper, dict) or wrapper.get("type") != "runtime.event":
        return None
    event = wrapper.get("event")
    return event if isinstance(event, dict) else None


def terminal_result_error(document: dict[str, Any] | None) -> str:
    """Extract the stable human-readable error from an Ori result object."""
    if not document:
        return ""
    error = document.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return ""


def json_text(value: object) -> str:
    """Stable compact rendering for raw evidence and tool content."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "DecodedLine",
    "OriUsage",
    "decode_line",
    "json_text",
    "runtime_event",
    "terminal_result_error",
]
