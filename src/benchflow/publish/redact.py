"""Structural secret redaction for trajectory contributions."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from benchflow.trajectories.types import redact_trajectory_text_with_count

REDACTED = "<XXX-benchflow-key-values-XXX>"
_CANONICAL_REDACTED = "***REDACTED***"
_LEGACY_REDACTED_VALUES = frozenset({"[REDACTED]", _CANONICAL_REDACTED})
_MAX_REDACTION_PASSES = 16

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
        "access_key",
        "account_key",
        "credential",
        "refresh_token",
        "client_secret",
        "encryption_key",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "token",
    }
)
SENSITIVE_KEY_SUFFIXES = (
    "api_key",
    "token",
    "secret",
    "password",
    "passwd",
    "access_key",
    "secret_key",
    "account_key",
    "private_key",
    "encryption_key",
    "credential",
    "credentials",
)
SEPARATED_SENSITIVE_KEY_SUFFIXES = tuple(
    f"_{suffix}" for suffix in SENSITIVE_KEY_SUFFIXES
)
COMPACT_SENSITIVE_KEY_SUFFIXES = tuple(
    suffix.replace("_", "") for suffix in SENSITIVE_KEY_SUFFIXES
)


class RedactionPattern(NamedTuple):
    pattern: re.Pattern[str]
    replacement: str


VALUE_PATTERNS = (
    # Google AI Studio's newer token format is not covered by the canonical
    # trajectory redactor yet.
    RedactionPattern(re.compile(r"AQ\.[0-9A-Za-z_-]{20,}"), REDACTED),
    RedactionPattern(
        re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
        f"Bearer {REDACTED}",
    ),
)


def redact_value(value: Any, *, field_name: str | None = None) -> tuple[Any, int]:
    """Return a structurally redacted JSON value and replacement count."""
    if field_name is not None and _is_sensitive_key(field_name):
        return (value, 0) if _is_redacted_value(value) else (REDACTED, 1)

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        replacements = 0
        original_keys = set(value)
        secret_key_index = 0
        carrier_field = next(
            (
                item
                for key, item in value.items()
                if isinstance(key, str)
                and _normalize_key(key) in {"key", "name"}
                and isinstance(item, str)
                and _is_sensitive_key(item)
            ),
            None,
        )
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
                field_name=(
                    carrier_field
                    if isinstance(key, str)
                    and _normalize_key(key) in {"value", "values"}
                    else key
                    if isinstance(key, str)
                    else None
                ),
            )
            redacted[clean_key] = clean
            replacements += count
        return redacted, replacements

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _redact_argv(value, redact_other_values=True)

    if not isinstance(value, str):
        return value, 0

    return _redact_text(value)


def redact_value_to_stability(value: Any) -> tuple[Any, int]:
    """Redact until another server-side pass would make no replacements."""
    redacted = value
    replacements = 0
    for _ in range(_MAX_REDACTION_PASSES):
        redacted, count = redact_value(redacted)
        replacements += count
        if count == 0:
            return redacted, replacements
    raise ValueError("trajectory secret redaction did not converge")


def _redact_text(value: str) -> tuple[str, int]:
    # The canonical carrier patterns intentionally ignore their asterisk marker.
    # Protect our public-upload marker with that value during the scan so a second
    # client/server pass is idempotent even inside text such as ``API_KEY=...``.
    protected = value.replace(REDACTED, _CANONICAL_REDACTED)
    redacted_text, replacements = redact_trajectory_text_with_count(protected)
    redacted_text = redacted_text.replace(_CANONICAL_REDACTED, REDACTED)
    for pattern, replacement in VALUE_PATTERNS:
        redacted_text, count = pattern.subn(replacement, redacted_text)
        replacements += count
    cli_redacted, cli_replacements = _redact_cli_text(redacted_text)
    redacted_text = cli_redacted
    replacements += cli_replacements
    return redacted_text, replacements


def _redact_cli_text(value: str) -> tuple[str, int]:
    if "--" not in value:
        return value, 0
    try:
        argv = shlex.split(value)
    except ValueError:
        return value, 0
    redacted, replacements = _redact_argv(argv, redact_other_values=False)
    return (shlex.join(redacted), replacements) if replacements else (value, 0)


def _redact_argv(
    values: Sequence[Any], *, redact_other_values: bool
) -> tuple[list[Any], int]:
    redacted: list[Any] = []
    replacements = 0
    redact_next = False
    for item in values:
        if redact_next:
            if _is_redacted_value(item):
                redacted.append(item)
            else:
                redacted.append(REDACTED)
                replacements += 1
            redact_next = False
            continue

        if isinstance(item, str) and item.startswith("-"):
            option, separator, option_value = item.partition("=")
            if _is_sensitive_key(option):
                if separator and option_value:
                    if _is_redacted_value(option_value):
                        redacted.append(item)
                    else:
                        redacted.append(f"{option}={REDACTED}")
                        replacements += 1
                else:
                    redacted.append(item)
                    redact_next = not separator
                continue

        if redact_other_values:
            clean, count = redact_value(item)
            redacted.append(clean)
            replacements += count
        else:
            redacted.append(item)
    return redacted, replacements


def _is_redacted_value(value: Any) -> bool:
    return isinstance(value, str) and (
        value == REDACTED or value in _LEGACY_REDACTED_VALUES
    )


def _is_sensitive_key(field_name: str) -> bool:
    normalized = _normalize_key(field_name)
    return (
        normalized in DENYLISTED_KEYS
        or normalized.endswith(SEPARATED_SENSITIVE_KEY_SUFFIXES)
        or normalized.replace("_", "").endswith(COMPACT_SENSITIVE_KEY_SUFFIXES)
    )


def _normalize_key(field_name: str) -> str:
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", field_name)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
