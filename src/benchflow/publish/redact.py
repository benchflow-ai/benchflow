"""Structural secret redaction for trajectory contributions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

REDACTED = "[REDACTED]"

DENYLISTED_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "api_key",
        "apikey",
        "cookie",
        "credentials",
        "private_key",
        "set-cookie",
        "x-goog-api-key",
        "aws_bearer_token_bedrock",
        "aws_secret_access_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "secret",
    }
)


class RedactionPattern(NamedTuple):
    pattern: re.Pattern[str]
    replacement: str


VALUE_PATTERNS = (
    RedactionPattern(re.compile(r"sk-[A-Za-z0-9_-]{16,}"), REDACTED),
    RedactionPattern(re.compile(r"AIza[0-9A-Za-z_-]{35}"), REDACTED),
    RedactionPattern(re.compile(r"AQ\.[0-9A-Za-z_-]{20,}"), REDACTED),
    RedactionPattern(re.compile(r"AKIA[0-9A-Z]{16}"), REDACTED),
    RedactionPattern(re.compile(r"ghp_[A-Za-z0-9]{36,}"), REDACTED),
    RedactionPattern(re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), REDACTED),
    RedactionPattern(re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), REDACTED),
    RedactionPattern(
        re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
        f"Bearer {REDACTED}",
    ),
)


def redact_value(value: Any, *, field_name: str | None = None) -> tuple[Any, int]:
    """Return a structurally redacted JSON value and replacement count."""
    if field_name is not None and _is_sensitive_key(field_name):
        return (value, 0) if value == REDACTED else (REDACTED, 1)

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        replacements = 0
        for key, item in value.items():
            clean, count = redact_value(
                item,
                field_name=key if isinstance(key, str) else None,
            )
            redacted[key] = clean
            replacements += count
        return redacted, replacements

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        redacted_items = []
        replacements = 0
        for item in value:
            clean, count = redact_value(item)
            redacted_items.append(clean)
            replacements += count
        return redacted_items, replacements

    if not isinstance(value, str):
        return value, 0

    redacted_text = value
    replacements = 0
    for pattern, replacement in VALUE_PATTERNS:
        redacted_text, count = pattern.subn(replacement, redacted_text)
        replacements += count
    return redacted_text, replacements


def _is_sensitive_key(field_name: str) -> bool:
    normalized = field_name.casefold()
    return normalized in DENYLISTED_KEYS or normalized.endswith(
        ("_api_key", "-api-key", "_secret", "-secret", "_password", "-password")
    )
