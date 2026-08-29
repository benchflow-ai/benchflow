"""Structural secret redaction for trajectory contributions."""

from __future__ import annotations

import re
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from benchflow.trajectories.redaction import (
    REDACTION_CATEGORY_API_KEY,
    REDACTION_CATEGORY_BEARER_TOKEN,
    REDACTION_CATEGORY_CREDENTIAL_FIELD,
    REDACTION_CATEGORY_PASSWORD,
    REDACTION_CATEGORY_PRIVATE_KEY,
    REDACTION_CATEGORY_URL_CREDENTIAL,
    redact_trajectory_text_with_categories,
)

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
    category: str


VALUE_PATTERNS = (
    # Google AI Studio's newer token format is not covered by the canonical
    # trajectory redactor yet.
    RedactionPattern(
        re.compile(r"AQ\.[0-9A-Za-z_-]{20,}"), REDACTED, REDACTION_CATEGORY_API_KEY
    ),
    RedactionPattern(
        re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
        f"Bearer {REDACTED}",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
)

# Display order and pluralized labels for the per-category breakdown shown to
# contributors ("Masked for you: 2 API keys, 1 bearer token"). Keys are the
# canonical category names from ``benchflow.trajectories.types``; the taxonomy
# names what the redaction rules actually detect — nothing more.
REDACTION_CATEGORY_LABELS: tuple[tuple[str, str], ...] = (
    (REDACTION_CATEGORY_API_KEY, "API keys"),
    (REDACTION_CATEGORY_BEARER_TOKEN, "bearer tokens"),
    (REDACTION_CATEGORY_PRIVATE_KEY, "private key blocks"),
    (REDACTION_CATEGORY_PASSWORD, "passwords"),
    (REDACTION_CATEGORY_URL_CREDENTIAL, "URL credentials"),
    (REDACTION_CATEGORY_CREDENTIAL_FIELD, "credential-bearing field values"),
)


def redaction_breakdown(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    """Return non-zero category counts in canonical display order."""
    ordered = [
        (category, counts[category])
        for category, _ in REDACTION_CATEGORY_LABELS
        if counts.get(category)
    ]
    # Unknown categories cannot arise from this module's own accumulation, but
    # persisted/forwarded counts must never be dropped silently.
    ordered.extend(
        (category, count)
        for category, count in counts.items()
        if count and category not in dict(REDACTION_CATEGORY_LABELS)
    )
    return tuple(ordered)


def format_redaction_breakdown(counts: Mapping[str, int]) -> str:
    """Render category counts as prose, e.g. ``2 API keys, 1 bearer token``."""
    plurals = dict(REDACTION_CATEGORY_LABELS)
    parts = []
    for category, count in redaction_breakdown(counts):
        label = category if count == 1 else plurals.get(category, f"{category}s")
        parts.append(f"{count} {label}")
    return ", ".join(parts)


def redact_value(
    value: Any,
    *,
    field_name: str | None = None,
    categories: Counter[str] | None = None,
) -> tuple[Any, int]:
    """Return a structurally redacted JSON value and replacement count.

    When *categories* is given, every replacement also increments the matching
    per-category counter (see ``REDACTION_CATEGORY_LABELS``); the counter's
    total always equals the returned replacement count.
    """
    if field_name is not None and _is_sensitive_key(field_name):
        if _is_redacted_value(value):
            return value, 0
        if categories is not None:
            categories[_field_category(field_name)] += 1
        return REDACTED, 1

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
                _, key_replacements = _redact_text(key, categories=categories)
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
                categories=categories,
            )
            redacted[clean_key] = clean
            replacements += count
        return redacted, replacements

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _redact_argv(value, redact_other_values=True, categories=categories)

    if not isinstance(value, str):
        return value, 0

    return _redact_text(value, categories=categories)


def redact_value_to_stability(
    value: Any, *, categories: Counter[str] | None = None
) -> tuple[Any, int]:
    """Redact until another server-side pass would make no replacements."""
    redacted = value
    replacements = 0
    for _ in range(_MAX_REDACTION_PASSES):
        redacted, count = redact_value(redacted, categories=categories)
        replacements += count
        if count == 0:
            return redacted, replacements
    raise ValueError("trajectory secret redaction did not converge")


def _redact_text(
    value: str, *, categories: Counter[str] | None = None
) -> tuple[str, int]:
    # The canonical carrier patterns intentionally ignore their asterisk marker.
    # Protect our public-upload marker with that value during the scan so a second
    # client/server pass is idempotent even inside text such as ``API_KEY=...``.
    protected = value.replace(REDACTED, _CANONICAL_REDACTED)
    redacted_text, text_categories = redact_trajectory_text_with_categories(protected)
    replacements = sum(text_categories.values())
    if categories is not None:
        categories.update(text_categories)
    redacted_text = redacted_text.replace(_CANONICAL_REDACTED, REDACTED)
    for pattern, replacement, category in VALUE_PATTERNS:
        redacted_text, count = pattern.subn(replacement, redacted_text)
        replacements += count
        if count and categories is not None:
            categories[category] += count
    cli_redacted, cli_replacements = _redact_cli_text(
        redacted_text, categories=categories
    )
    redacted_text = cli_redacted
    replacements += cli_replacements
    return redacted_text, replacements


def _redact_cli_text(
    value: str, *, categories: Counter[str] | None = None
) -> tuple[str, int]:
    if "--" not in value:
        return value, 0
    try:
        argv = shlex.split(value)
    except ValueError:
        return value, 0
    # Count categories only for the accepted result: a zero-replacement parse
    # returns the original text, so nothing may be attributed for it.
    argv_categories: Counter[str] | None = None if categories is None else Counter()
    redacted, replacements = _redact_argv(
        argv, redact_other_values=False, categories=argv_categories
    )
    if not replacements:
        return value, 0
    if categories is not None and argv_categories is not None:
        categories.update(argv_categories)
    return shlex.join(redacted), replacements


def _redact_argv(
    values: Sequence[Any],
    *,
    redact_other_values: bool,
    categories: Counter[str] | None = None,
) -> tuple[list[Any], int]:
    redacted: list[Any] = []
    replacements = 0
    sensitive_option: str | None = None
    for item in values:
        if sensitive_option is not None:
            option = sensitive_option
            sensitive_option = None
            if _is_redacted_value(item):
                redacted.append(item)
            else:
                redacted.append(REDACTED)
                replacements += 1
                if categories is not None:
                    categories[_field_category(option)] += 1
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
                        if categories is not None:
                            categories[_field_category(option)] += 1
                else:
                    redacted.append(item)
                    if not separator:
                        sensitive_option = option
                continue

        if redact_other_values:
            clean, count = redact_value(item, categories=categories)
            redacted.append(clean)
            replacements += count
        else:
            redacted.append(item)
    return redacted, replacements


def _is_redacted_value(value: Any) -> bool:
    return isinstance(value, str) and (
        value == REDACTED or value in _LEGACY_REDACTED_VALUES
    )


def _field_category(field_name: str) -> str:
    """Categorize a masked value by the credential-bearing name that matched.

    Structural redaction fires on the field/option NAME, so the name is the
    only honest signal for what kind of secret the value was.
    """
    compact = _normalize_key(field_name).replace("_", "")
    if compact.endswith(("password", "passwd")):
        return REDACTION_CATEGORY_PASSWORD
    if compact.endswith("apikey"):
        return REDACTION_CATEGORY_API_KEY
    if compact.endswith("authorization") or "bearertoken" in compact:
        return REDACTION_CATEGORY_BEARER_TOKEN
    return REDACTION_CATEGORY_CREDENTIAL_FIELD


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
