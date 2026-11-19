"""Structural secret redaction for trajectory contributions."""

from __future__ import annotations

import re
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from benchflow._utils.credential_names import placeholder_key
from benchflow.trajectories.types import (
    _BACKSLASH_UNIT,
    _WS,
    REDACTION_CATEGORY_API_KEY,
    REDACTION_CATEGORY_BEARER_TOKEN,
    REDACTION_CATEGORY_CREDENTIAL_FIELD,
    REDACTION_CATEGORY_PASSWORD,
    REDACTION_CATEGORY_PRIVATE_KEY,
    REDACTION_CATEGORY_URL_CREDENTIAL,
    REDACTION_CATEGORY_VIEW_BACKSTOP,
    REDACTION_CATEGORY_VIEW_BUDGET,
    _at_key_start,
    _one_view_budget,
    _snake_name,
    is_secret_field,
    is_secret_number,
    redact_trajectory_text_with_categories,
    redact_trajectory_text_with_view_flags,
    view_flags,
)

REDACTED = "<XXX-benchflow-key-values-XXX>"
_CANONICAL_REDACTED = "***REDACTED***"
_LEGACY_REDACTED_VALUES = frozenset({"[REDACTED]", _CANONICAL_REDACTED})
_MAX_REDACTION_PASSES = 16


class RedactionPattern(NamedTuple):
    pattern: re.Pattern[str]
    replacement: str
    category: str


VALUE_PATTERNS = (
    # Google AI Studio's newer token format, as the canonical trajectory
    # redactor masks it too, where a key may start (the same _at_key_start), so `FAQ.`
    # followed by a long word stays.
    RedactionPattern(
        re.compile(_at_key_start(r"AQ\.") + r"[0-9A-Za-z_-]{20,}"),
        REDACTED,
        REDACTION_CATEGORY_API_KEY,
    ),
    RedactionPattern(
        # A backslash in the token is part of it, as _SECVAL reads it.
        re.compile(
            rf"Bearer{_WS}+(?:[A-Za-z0-9._~+/=-]|{_BACKSLASH_UNIT}){{16,}}",
            re.IGNORECASE,
        ),
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
    (REDACTION_CATEGORY_VIEW_BUDGET, "lines masked whole (view work budget)"),
    (REDACTION_CATEGORY_VIEW_BACKSTOP, "lines masked whole (wall-clock backstop)"),
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


@_one_view_budget
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
    if field_name is not None and _scalars(value):
        # A number, a boolean or null (or a list of them) decides as in
        # trajectory files: only a PIN-like integer under a credential's name
        # is masked (is_secret_number), each one in a list.
        items = value if isinstance(value, list) else [value]
        masked = [is_secret_number(field_name, item) for item in items]
        if categories is not None and any(masked):
            categories[_field_category(field_name)] += sum(masked)
        clean = [
            REDACTED if hit else item for hit, item in zip(masked, items, strict=True)
        ]
        return (clean if isinstance(value, list) else clean[0]), sum(masked)

    if field_name is not None and _is_sensitive_key(field_name, value):
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
                    clean_key = placeholder_key(secret_key_index)
                    while clean_key in original_keys or clean_key in redacted:
                        secret_key_index += 1
                        clean_key = placeholder_key(secret_key_index)
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


@_one_view_budget
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
    own = view_flags(protected, _VIEW_ROWS)
    if own is None:
        redacted_text, text_categories = redact_trajectory_text_with_categories(
            protected
        )
    else:
        redacted_text, text_categories = redact_trajectory_text_with_view_flags(
            protected, own
        )
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


# publish is the layer that leaves our
# hands, so its own rules (VALUE_PATTERNS and the command-line options, one
# option and its value at a time) read the decoded views the trajectory
# patterns read (view_flags): what they mask there is masked too, after the
# main pass and by the same position map (redact_trajectory_text_with_view_flags;
# the main pass reads the text as written), so `%62earer <token>` and
# `--api%2Dkey <v>` are masked as `bearer <token>` and `--api-key <v>` are. An
# option starts a word, as the shell splits it.
_CLI_OPTION = re.compile(r"(?<!\S)--[^\s=]++(?:=\S*+|[ \t]++\S++)?+")


def _cli_option_value(match: re.Match[str]) -> str:
    return _redact_unsplit_cli_text(match[0])[0]


# The rules as the views read them: VALUE_PATTERNS' Bearer row keeps its
# scheme and spacing as written (`bearer%20`, `Bearer+`), so the view masks only
# the value, as the row does in plain text.
_VIEW_ROWS = (
    VALUE_PATTERNS[0],
    (
        re.compile(rf"(Bearer{_WS}+)[A-Za-z0-9._~+/=-]{{16,}}", re.IGNORECASE),
        rf"\g<1>{REDACTED}",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    (_CLI_OPTION, _cli_option_value, REDACTION_CATEGORY_CREDENTIAL_FIELD),
)


def _redact_cli_text(
    value: str, *, categories: Counter[str] | None = None
) -> tuple[str, int]:
    if "--" not in value:
        return value, 0
    try:
        argv = shlex.split(value)
    except ValueError:
        # An unbalanced quote or a trailing backslash: the options in the text
        # still carry their values, so they are masked where they stand.
        return _redact_unsplit_cli_text(value, categories=categories)
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


def _redact_unsplit_cli_text(
    value: str, *, categories: Counter[str] | None = None
) -> tuple[str, int]:
    """``_redact_cli_text`` for text ``shlex`` cannot split: the same option
    rules over :func:`_cli_words`, each masked value replaced in place."""
    words = _cli_words(value)
    argv_categories: Counter[str] | None = None if categories is None else Counter()
    redacted, replacements = _redact_argv(
        [word for _, _, word in words],
        redact_other_values=False,
        categories=argv_categories,
    )
    if not replacements:
        return value, 0
    if categories is not None and argv_categories is not None:
        categories.update(argv_categories)
    pieces: list[str] = []
    last = 0
    for (start, end, word), clean in zip(words, redacted, strict=True):
        if clean == word:
            continue
        if clean == REDACTED:
            pieces.append(value[last:start])
        else:  # --option=value: the raw text keeps everything to the "="
            pieces.append(value[last : value.index("=", start, end) + 1])
        pieces.append(REDACTED)
        last = end
    pieces.append(value[last:])
    return "".join(pieces), replacements


def _cli_words(text: str) -> list[tuple[int, int, str]]:
    """The words of *text* as a POSIX shell splits them, with their spans.

    For text ``shlex.split`` refuses: an unterminated quote runs to the end
    of the text and a trailing backslash stays, so every word is returned.
    """
    words: list[tuple[int, int, str]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        start, chars = i, []
        while i < n and not text[i].isspace():
            char = text[i]
            if char in "'\"":
                close = text.find(char, i + 1)
                close = n if close < 0 else close
                chars.append(text[i + 1 : close])
                i = close + 1
            elif char == "\\" and i + 1 < n:
                chars.append(text[i + 1])
                i += 2
            else:
                chars.append(char)
                i += 1
        words.append((start, min(i, n), "".join(chars)))
    return words


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


def _scalars(value: Any) -> bool:
    """A number, a boolean or null, or a non-empty list of only those."""
    if isinstance(value, list):
        return bool(value) and all(_scalar(item) for item in value)
    return _scalar(value)


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float))


def _is_sensitive_key(field_name: str, value: Any = None) -> bool:
    """The trajectory redactor's secret names (``is_secret_field``)."""
    return is_secret_field(field_name, value)


def _normalize_key(field_name: str) -> str:
    return _snake_name(field_name)
