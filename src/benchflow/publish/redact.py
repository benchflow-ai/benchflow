"""Structural secret redaction for trajectory contributions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from benchflow.trajectories.types import redact_trajectory_text_with_count

REDACTED = "[REDACTED]"

DENYLISTED_KEYS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "x_api_key",
        "api_key",
        "apikey",
        "cookie",
        "credentials",
        "private_key",
        "set_cookie",
        "x_goog_api_key",
        "aws_bearer_token_bedrock",
        "aws_secret_access_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)


class RedactionPattern(NamedTuple):
    pattern: re.Pattern[str]
    replacement: str


VALUE_PATTERNS = (
    # Google AI Studio's newer token format is not covered by the canonical
    # trajectory redactor yet.
    RedactionPattern(re.compile(r"AQ\.[0-9A-Za-z_-]{20,}"), "***REDACTED***"),
    RedactionPattern(
        re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
        "Bearer ***REDACTED***",
    ),
)


def redact_value(value: Any, *, field_name: str | None = None) -> tuple[Any, int]:
    """Return a structurally redacted JSON value and replacement count."""
    if field_name is not None and _is_sensitive_key(field_name):
        return (value, 0) if value == REDACTED else (REDACTED, 1)

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        replacements = 0
        original_keys = set(value)
        secret_key_index = 0
        for key, item in value.items():
            clean_key = key
            if isinstance(key, str):
                _, key_replacements = _redact_text(key)
                if key_replacements:
                    secret_key_index += 1
                    clean_key = f"***REDACTED_KEY_{secret_key_index}***"
                    while clean_key in original_keys or clean_key in redacted:
                        secret_key_index += 1
                        clean_key = f"***REDACTED_KEY_{secret_key_index}***"
                    replacements += key_replacements
            clean, count = redact_value(
                item,
                field_name=key if isinstance(key, str) else None,
            )
            redacted[clean_key] = clean
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

    return _redact_text(value)


def _redact_text(value: str) -> tuple[str, int]:
    redacted_text, replacements = redact_trajectory_text_with_count(value)
    for pattern, replacement in VALUE_PATTERNS:
        redacted_text, count = pattern.subn(replacement, redacted_text)
        replacements += count
    return redacted_text, replacements


def _is_sensitive_key(field_name: str) -> bool:
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", field_name)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    return normalized in DENYLISTED_KEYS or normalized.endswith(
        (
            "_api_key",
            "_token",
            "_secret",
            "_password",
            "_passwd",
            "_access_key",
            "_secret_key",
            "_account_key",
            "_private_key",
            "_encryption_key",
            "_credential",
            "_credentials",
        )
    )
