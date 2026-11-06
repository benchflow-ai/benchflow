"""The output choke point for credentials in user-facing diagnostics.

Keep complete source fields until this boundary. Formatting receives opaque,
credential-free stand-ins, so splitting, stripping, joining and truncating cannot
expose part of a matched value. Only the final rendered string gets masks.
"""

from __future__ import annotations

import heapq
import json
import re
import shlex
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any
from urllib.parse import quote, quote_plus

# Consume the entire maximal scheme-character run, including punctuation before
# a valid scheme suffix. Never retry each suffix of a malformed megabyte run.
_URL_USERINFO = re.compile(r"(?i)(?<![a-z0-9+.-])[a-z0-9+.-]+://(?P<userinfo>[^/\s]+)@")


def _literal_spellings(secrets: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for value in secrets:
        if len(value) < 6:
            continue
        values.update(
            (
                value,
                shlex.quote(value),
                json.dumps(value)[1:-1],
                json.dumps(value, ensure_ascii=False)[1:-1],
                repr(value)[1:-1],
                quote(value, errors="surrogatepass"),
                quote(value, safe="", errors="surrogatepass"),
                quote_plus(value, safe="", errors="surrogatepass"),
            )
        )
        # repr and ascii pick their quotes from the whole operand, so a value
        # embedded in a larger string can be rendered in the other quote
        # context than repr(value) uses (FF-sw-cli-other-C1). Render each
        # character in a fixed context instead of guessing the operand's.
        for render in (repr, ascii):
            for quote_char in ("'", '"'):
                values.add(
                    "".join(
                        "\\" + char if char == quote_char else render(char)[1:-1]
                        for char in value
                    )
                )
    return values


def _literal_pattern(secrets: Iterable[str]) -> re.Pattern[str] | None:
    values = _literal_spellings(secrets)
    if not values:
        return None
    return re.compile(
        "|".join(
            re.escape(value)
            for value in sorted(values, key=lambda value: len(value), reverse=True)
        )
    )


def _spans(text: str, pattern: re.Pattern[str] | None) -> Iterator[tuple[int, int]]:
    literals = (match.span() for match in pattern.finditer(text)) if pattern else ()
    urls = (match.span("userinfo") for match in _URL_USERINFO.finditer(text))
    # Both streams refer to original text. Merge overlapping ranges, including
    # credentials that cross a URL's scheme/userinfo/path boundary; never scan
    # a replacement or let the URL alternative hide a longer literal match.
    start = end = -1
    for left, right in heapq.merge(literals, urls):
        if left < end:
            end = max(end, right)
        else:
            if start >= 0:
                yield start, end
            start, end = left, right
    if start >= 0:
        yield start, end


def render_diagnostic(
    fields: Sequence[str],
    secrets: Iterable[str],
    render: Callable[[Sequence[str]], Any],
    *,
    serialize: Callable[[Any], str] = str,
) -> str:
    """Format intact fields, then emit masks once in the final output string.

    The stand-in character is one printable code point, absent from all
    inputs. A line operation cannot split it or strip its padding. It never
    contains a credential byte. Structured output is serialized only after
    masks have been emitted in its final string values.
    No caller needs to mask fragments or infer fragments of a known value.
    """
    used = set("".join(fields))
    codepoint = 0xA1
    while codepoint <= sys.maxunicode:
        marker = chr(codepoint)
        if marker not in used and marker.isprintable() and not marker.isspace():
            break
        codepoint += 1
    else:
        # Even an input containing every Unicode character remains total.
        # Asterisks are already the final safe spelling and need no decoding.
        marker = "*"
    pattern = _literal_pattern(secrets)

    def protect(text: str) -> str:
        parts: list[str] = []
        end = 0
        for left, right in _spans(text, pattern):
            parts.extend((text[end:left], marker * 3))
            end = right
        parts.append(text[end:])
        return "".join(parts)

    def finish(value: Any) -> Any:
        if isinstance(value, str):
            # Joining intact fields can create a match too. Check only original
            # rendered characters: stand-ins are barriers, never new input to
            # the matcher. Structured outputs are finished before serialization
            # so a credential cannot consume JSON keys, quotes or separators.
            return marker.join(protect(part) for part in value.split(marker)).replace(
                marker, "*"
            )
        if isinstance(value, dict):
            return {key: finish(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finish(item) for item in value]
        return value

    return serialize(finish(render([protect(text) for text in fields])))


def redact(text: str, secrets: Iterable[str]) -> str:
    """Mask complete credential spellings and URL userinfo in one boundary."""
    return render_diagnostic((text,), secrets, lambda fields: fields[0])


def render_text(text: str, limit: int | None = None) -> str:
    """Render a complete diagnostic before any preview can shorten it.

    All plain-text previews use this boundary; the truncator is deliberately
    internal to it. Credential discovery stays lazy to avoid a CLI import at
    evaluation-module import time.
    """
    from benchflow._utils.text import truncate_end
    from benchflow.cli._errors import _secrets

    redacted = redact(text, _secrets())
    return redacted if limit is None else truncate_end(redacted, limit)


_OUTPUT_UNIT_LIMIT = 2 * 1024 * 1024
_OMITTED_UNIT = "*** (oversized diagnostic omitted)\n"


def redact_stream(chunks: Iterable[str], secrets: Iterable[str]) -> Iterator[str]:
    """Buffer complete units, redact each once, then publish it.

    A unit is a whole physical line, including its scheme and URL authority.
    There is no match-span carry or raw prefix publication. If any known value
    spans physical lines, the entire capture is one unit instead: splitting
    that value at a newline would lose the same context as splitting a URL.
    Units above the fixed cap are omitted in full, including their remaining
    input through the delimiter. EOF flushes the final unit through the same
    redactor. Buffering and matching are independent of child write/read sizes.
    """
    secrets = tuple(secrets)
    multiline = any("\n" in value for value in secrets if len(value) >= 6)
    pending: list[str] = []
    size = 0
    omitted = False

    for chunk in chunks:
        start = 0
        while start < len(chunk):
            newline = -1 if multiline else chunk.find("\n", start)
            end = len(chunk) if newline < 0 else newline + 1
            if not omitted:
                if size + end - start > _OUTPUT_UNIT_LIMIT:
                    pending.clear()
                    size = 0
                    omitted = True
                    yield redact(_OMITTED_UNIT, secrets)
                else:
                    pending.append(chunk[start:end])
                    size += end - start
            if newline >= 0:
                if not omitted:
                    yield redact("".join(pending), secrets)
                pending.clear()
                size = 0
                omitted = False
            start = end
    if pending:
        yield redact("".join(pending), secrets)


def diagnostic_json(
    value: Any, secrets: Iterable[str], *, indent: int | None = None
) -> str:
    """Render string values through the boundary while retaining the JSON schema."""
    fields: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            fields.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    def render(protected: Sequence[str]) -> Any:
        values = iter(protected)

        def rebuild(item: Any) -> Any:
            if isinstance(item, str):
                return next(values)
            if isinstance(item, dict):
                return {key: rebuild(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [rebuild(child) for child in item]
            return item

        return rebuild(value)

    collect(value)
    return render_diagnostic(
        fields, secrets, render, serialize=lambda item: json.dumps(item, indent=indent)
    )
