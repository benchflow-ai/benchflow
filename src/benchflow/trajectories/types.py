"""Trajectory types — raw LLM API request/response pairs captured from providers."""

import bisect
import functools
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from benchflow._utils import staged_secrets
from benchflow._utils.credential_names import (
    _AUTHORIZATION_NAMES,
    _COOKIE_NAMES,
    _COOKIE_PAIR,
    _COUNT_RESULT,
    _CREDENTIAL_WORDS,
    _FLAG_WORDS,
    _HEADER_QUALIFIERS,
    _KEY_NAMES,
    _KEY_VALUE,
    _NOT_SECRET_NAMES,
    _ONE_WORDS,
    _PASS_COUNT_WORDS,
    _PASS_OUTCOME_PAIRS,
    _PASS_OUTCOME_WORDS,
    _PASSWORD_QUALIFIERS,
    _PATH_OR_CONTENT_NAMES,
    _PRICE_NUMBER,
    _PRICE_WORDS,
    _PROSE,
    _PROVIDER_WORDS,
    _SWITCH,
    _TOKEN_KINDS,
    _VALUE_END,
    entry_kind,
    is_code_authorization_name,
    name_words,
)
from benchflow._utils.credential_names import _IDENTIFIER as _LETTERS_IDENTIFIER
from benchflow._utils.credential_names import _TEST_RESULT as _COUNT_OR_SWITCH

_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadInputTokenCount",
    "cacheReadInputTokens",
    "cacheWriteInputTokenCount",
    "cacheWriteInputTokens",
}
_USAGE_DETAIL_KEYS = {
    "cached_tokens",
}
_USAGE_METADATA_KEYS = {
    "promptTokenCount",
    "candidatesTokenCount",
    "totalTokenCount",
    "cachedContentTokenCount",
    "toolUsePromptTokenCount",
}


def _has_non_null_key(payload: dict[str, Any], keys: set[str]) -> bool:
    return any(key in payload and payload[key] is not None for key in keys)


def _has_provider_usage(payload: dict[str, Any]) -> bool:
    if _has_non_null_key(payload, _USAGE_KEYS):
        return True
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = payload.get(key)
        if isinstance(details, dict) and _has_non_null_key(details, _USAGE_DETAIL_KEYS):
            return True
    return False


def _first_int(*values: Any) -> int:
    """Return the first non-null usage value as an integer."""
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _first_optional_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class TokenUsage:
    """One call's tokens, in BenchFlow's split.

    ``input_tokens`` is the uncached input only; ``cache_read_tokens`` and
    ``cache_creation_tokens`` (cache writes, with their 5-minute and 1-hour
    parts where the provider reports them) stand beside it, never inside it;
    ``output_tokens`` counts reasoning. A count the response does not report
    is None, never 0: OpenAI reports no cache writes.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    provider_total_tokens: int | None = None
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        if self.provider_total_tokens is not None:
            return self.provider_total_tokens
        # The four parts are disjoint (see ``_exchange_token_usage``).
        return sum(
            value or 0
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cache_read_tokens,
                self.cache_creation_tokens,
            )
        )


def _exchange_token_usage(exchange: "LLMExchange") -> TokenUsage:
    usage = exchange.response.body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    usage_metadata = exchange.response.body.get("usageMetadata")
    usage_metadata = usage_metadata if isinstance(usage_metadata, dict) else {}
    # OpenAI may return these keys with an explicit null value, so
    # `or {}` is required — `.get(key, {})` would still yield None.
    prompt_details = usage.get("prompt_tokens_details") or {}
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    input_details = usage.get("input_tokens_details") or {}
    input_details = input_details if isinstance(input_details, dict) else {}

    # Cache reported as a SEPARATE additive component — Anthropic Messages
    # (`cache_read_input_tokens`) and Bedrock Converse (`cacheReadInputToken*`) —
    # is NOT included in that provider's `input_tokens`/`inputTokens` count.
    additive_cache_read = _first_optional_int(
        usage.get("cache_read_input_tokens"),
        usage.get("cacheReadInputTokens"),
        usage.get("cacheReadInputTokenCount"),
    )
    # OpenRouter reports its cache writes in the prompt's details
    # (``prompt_tokens_details.cache_write_tokens``; LiteLLM's copy of the
    # answer adds ``cache_creation_tokens`` beside it), inside prompt_tokens;
    # LiteLLM's Responses API bridge moves them to the input's details
    # (``input_tokens_details.cache_write_tokens``), inside input_tokens.
    cache_creation = _first_optional_int(
        usage.get("cache_creation_input_tokens"),
        usage.get("cacheWriteInputTokens"),
        usage.get("cacheWriteInputTokenCount"),
        prompt_details.get("cache_write_tokens"),
        prompt_details.get("cache_creation_tokens"),
        input_details.get("cache_write_tokens"),
        input_details.get("cache_creation_tokens"),
    )
    # Cache reported as a SUBSET already inside the input count — OpenAI
    # (`*_tokens_details.cached_tokens`) and Gemini (`cachedContentTokenCount`).
    inclusive_cache_read = _first_optional_int(
        prompt_details.get("cached_tokens"),
        input_details.get("cached_tokens"),
        usage_metadata.get("cachedContentTokenCount"),
    )

    # ``input_tokens`` is the UNCACHED input: Anthropic/Bedrock report
    # it so, with the cache beside it; OpenAI/Gemini report the cache-inclusive
    # prompt, whose cache reads (and, for an OpenAI-shaped answer from
    # LiteLLM's Anthropic bridge, cache writes) are taken out. A prompt count
    # is cache-inclusive when it is OpenAI's or Gemini's (``prompt_tokens``,
    # ``promptTokenCount``) or a details block marks its cached part
    # (OpenAI's Responses API ``input_tokens``).
    raw_input: int | None = None
    inclusive = False
    for value, is_inclusive in (
        (usage.get("input_tokens"), bool(input_details)),
        (usage.get("prompt_tokens"), True),
        (usage.get("inputTokens"), False),
        (usage_metadata.get("promptTokenCount"), True),
    ):
        raw_input = _first_optional_int(value)
        if raw_input is not None:
            inclusive = is_inclusive or inclusive_cache_read is not None
            break
    cache_read = (
        inclusive_cache_read
        if inclusive_cache_read is not None
        else additive_cache_read
    )
    # Gemini reports tool-use prompt tokens (`toolUsePromptTokenCount`) separately
    # from `promptTokenCount` — it is additive input, NOT a subset — so fold it in
    # too, or tool-heavy Gemini runs underreport input/cost (and totalTokenCount
    # would exceed input + output). Absent for every other provider.
    tool_use_prompt = _first_int(usage_metadata.get("toolUsePromptTokenCount"))
    input_tokens: int | None = None
    if raw_input is not None:
        inside = (cache_read or 0) + (cache_creation or 0) if inclusive else 0
        input_tokens = max(raw_input - inside, 0) + tool_use_prompt

    # Reasoning/thinking tokens are billed as output. Anthropic/OpenAI already
    # fold them into output_tokens/completion_tokens; Gemini reports them
    # separately as `thoughtsTokenCount`, so add them in for output parity.
    raw_output = _first_optional_int(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
        usage.get("outputTokens"),
        usage_metadata.get("candidatesTokenCount"),
    )
    output_tokens = (
        None
        if raw_output is None
        else raw_output + _first_int(usage_metadata.get("thoughtsTokenCount"))
    )

    # Anthropic splits its cache writes by lifetime.
    written = usage.get("cache_creation")
    written = written if isinstance(written, dict) else {}
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        provider_total_tokens=_first_optional_int(
            usage.get("total_tokens"),
            usage_metadata.get("totalTokenCount"),
            usage.get("totalTokens"),
        ),
        cache_write_5m_tokens=_first_optional_int(
            written.get("ephemeral_5m_input_tokens")
        ),
        cache_write_1h_tokens=_first_optional_int(
            written.get("ephemeral_1h_input_tokens")
        ),
    )


def _known_sum(values: Any) -> int | None:
    """The sum of the counts that are known; None when none is."""
    known = [value for value in values if value is not None]
    return sum(known) if known else None


# Human-facing redaction categories. Every canonical pattern below is tagged
# with the kind of secret its rule actually detects, so the upload preview can
# tell contributors WHAT was masked ("2 API keys, 1 bearer token") instead of
# only how many values. The taxonomy names the real rules — there is no
# PII/email detection, so no such category exists.
REDACTION_CATEGORY_API_KEY = "API key"
REDACTION_CATEGORY_BEARER_TOKEN = "bearer token"
REDACTION_CATEGORY_PRIVATE_KEY = "private key block"
REDACTION_CATEGORY_PASSWORD = "password"
REDACTION_CATEGORY_URL_CREDENTIAL = "URL credential"
REDACTION_CATEGORY_CREDENTIAL_FIELD = "credential-bearing field value"
# a gated line the decoded views had no budget left to read, masked
# whole, by the bound that ran out: the work budget (the same on every machine)
# or the wall-clock backstop (a slower machine may hit it where another doesn't).
REDACTION_CATEGORY_VIEW_BUDGET = "line masked whole (view work budget)"
REDACTION_CATEGORY_VIEW_BACKSTOP = "line masked whole (wall-clock backstop)"

# Quote atom for the header/key-value carriers below: a run of 0-8 optional
# backslashes followed by an optional quote. This makes every carrier fire on raw
# text (``"k": "v"``), Python dict-repr (``'k': 'v'``), AND json.dumps-escaped
# JSON at any nesting depth (``\"k\": \"v\"``, ``\\\"k\\\": …``) — the escaped
# form still appears when a serialized artifact is redacted as text (the
# ADP/ATIF exporters, ``results.py``), or when a provider error body embedded in
# ``error.message`` is itself JSON (#830). (``Trajectory.to_jsonl`` and the ACP
# writer now redact record *values* before ``json.dumps`` via
# ``redact_trajectory_obj`` so they can never emit a corrupt escape.) The
# ``{0,8}`` cap keeps it ReDoS-bounded. Up to three quotes, in any spelling a
# copy or a log writes them (`""password"": v`, `'''api_key'''=v`,
# `&quot;token&quot;: v`, `%22secret%22=v`, `"key": v`),
# each bounded.
_ANY_QUOTE = (
    r"(?:\\{0,8}[\"']|\\{1,8}(?:u00(?:22|27)|[xX](?:22|27))"
    r"|&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);|%(?:25)?2[27])"
)
# Up to three quotes, greedy, written as nested branches: a group repeat would
# cost a repeat context at every position a carrier is tried. A quote is tried
# only where one can start (a backslash, a quote, `&`, `%`).
_ESCQ = (
    rf"(?:(?=[\\\"'&%]){_ANY_QUOTE}"
    rf"(?:(?=[\\\"'&%]){_ANY_QUOTE}(?:(?=[\\\"'&%]){_ANY_QUOTE}|)|)|)"
)
# Secret value class for the carriers: stops at a quote (plain, or the backslash
# of an escaped closing quote), whitespace, and the ``,`` ``}`` ``&`` delimiters
# so a redacted value can never swallow a sibling JSON field or URL query param;
# excludes ``*`` so an already-redacted value isn't re-matched.
# An escape that writes a quote (`\u0022`, `\x27`, `\u{22}`, `\042`, or a
# backslash before an HTML quote, `\&#34;`), its backslash escaped up to
# eight times (`\\u0022` in a json.dumps copy): a quote where a value reads
# its quotes (_VALUE_QUOTE), not a character.
_QUOTE_ESCAPE = (
    r"\\{1,8}(?:u00(?:22|27)|[xX](?:22|27)|u\{0*(?:22|27)\}|0?4[27](?![0-7])"
    r"|&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);)"
)
# A name of its own and its `:` or `=` (perhaps URL-encoded), perhaps after an
# ANSI colour code and up to three quotes in any spelling, and before a tab or
# a space, written or escaped (`HOME=`, `password:`, `"key":`,
# `\u0022key\u0022:`, `""key"":`, `&quot;key&quot;:`, `[32mkey:`, `key\t:`):
# an entry; its value may follow on a later line, as the carriers read it
# (`AWS_SECRET_ACCESS_KEY=` and a line break). An `=` before the end or
# another `=` is no separator (`abc\/def=`, base64).
_NAME_QUOTES = (
    r"(?:\\{0,8}[\"'`]|\\{1,8}(?:u00(?:22|27)|[xX](?:22|27))"
    r"|&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);|%(?:25)?2[27]){0,3}+"
)
_ENTRY_SEPARATOR = (
    rf"{_NAME_QUOTES}(?:[ \t]|\\{{1,8}}t)*+"
    r"(?::|=(?!=|\Z)|%(?:25)?3[AaDd])"
)
_CSI_CODE = (
    r"(?:\x1b|\\{1,8}(?:u001[bB]|x1[bB]|033|e)|\^\[)?\[[0-9;:?]{0,48}[ -/]?[A-Za-z]"
)
_ANSI_CODE = rf"(?:{_CSI_CODE})?"
_NEXT_NAME = rf"{_ANSI_CODE}[A-Za-z0-9_.-]{{1,64}}{_ENTRY_SEPARATOR}"
# A carrier of its own anywhere in the rest of a run of plain characters (up
# to 4,096 of them), before an entry's separator: a credential's word that
# starts a name's segment (at the run's start, after anything but a letter:
# `_` `-` `.` a digit, a quote, `&`; after an ANSI code, whose last letter
# ends it, `\u001b[32mpassword2:`; or where a lower-case letter turns
# upper-case), the rest of the name, the separator
# (`GITHUB_TOKEN: `, `MY_API_KEY=`, `x-api-key:`, `apiKey:`,
# `\u001b[32mGITHUB_TOKEN:`, `\u0022password\u0022:`, a name past 64
# letters that ends in one); or a name that ends in a carrier's own ending
# right before the separator, perhaps numbered, as the carriers and the
# classifier read a glued name (`GITHUBTOKEN:`, `APIKEY=`, `dbpassword:`,
# `encryptionkey:`, `awsaccesskeyid:`, `PGPASSWORD2:`, `admintoken_2:`; the
# classifier's _GLUED_KEYS). A value that goes on through
# an escape, a quote or a line break never reads into one: it stops there,
# as a value that stopped at the backslash left it to its own carrier
# (`\Xab\x1b[32mGITHUB_TOKEN: ...`), or, where the name is glued onto the
# rest of the run, reads on into its value (_GLUED_NEXT). A word inside a
# word is none (`monkey=`, `turkey=`, `tokenizer` mid-name).
_GLUE_WORD = r"(?i:tok|secr|pass|pwd|key|cred|auth|cook|sess|bear|sig)"
_GLUE_END = (
    r"(?i:token|secret|passw(?:or)?d|pass_?phrase|credentials?|authorization|cookie"
    r"|(?:api|auth|access|secret|account|private|encryption|master|primary|secondary"
    r"|signing|session|hmac|license|storage|shared|client|root|public)[_-]?key"
    r"|access[_-]?key[_-]?id)"
)
# (case-sensitive inside the carriers' IGNORECASE patterns)
_SEGMENT_START = r"(?:(?<![A-Za-z])|(?-i:(?<=[a-z])(?=[A-Z])))"
_GLUE_RUN = r"[^\s\"',}&*\\]{0,4096}?"
# The rest of a name: its characters, or an escape that writes a letter or a
# digit (`authorizatio\u006e`, `\x6e`).
_GLUE_NAME = r"(?:[A-Za-z0-9_.-]|\\{1,8}(?:u00[0-9A-Fa-f]{2}|[xX][0-9A-Fa-f]{2})){0,64}"
# So is a scheme's `Bearer` and its token in that run (publish masks a
# bearer token by that shape: `\u{41}uthorization=Bearer <token>`).
_GLUE_SCHEME = r"(?i:bearer)[ \t]++[A-Za-z0-9._~+/=-]{16}"
# An escape that writes `&` or a space (`\u0026`, `\x26`, `\046`, `\u0020`),
# or a colour code (`\u001b[32m`, `\x1b[1;31m`, `\033[1m`, `\e[0m`, up to
# four in a row), right before a name of any kind and its separator: the value stops at it, as at
# the `&` or the space as written and at an escaped line break before an
# entry (_NEXT_ENTRY): `...\u001b[32mGETTOKEN2_URL: `, `...\u0026next=`.
_SEPARATOR_ESCAPE = (
    r"(?:u00(?:26|20)|x(?:26|20)|0?4[06](?![0-7])"
    r"|(?:u001[bB]|x1[bB]|033|e)\[[0-9;:?]{0,48}[ -/]?[A-Za-z])"
)
# A name the rc's classifier decides by its value (its _decided_name): a
# credential's word anywhere in a name and a number after it, perhaps after
# `_`, `-` or up to three letters (`FAKEsession0123456:`, `fakekeyid0123:`,
# `db_token_v2:`), or the word's hint (`key_hint=`).
_GLUE_NUMBERED = (
    r"(?i:token|secret|passw(?:or)?d|key|credential|passphrase|session)s?"
    r"(?:(?:[_-]?[0-9]|[_-]?[A-Za-z]{1,3}[0-9])[A-Za-z0-9_-]{0,48}|[_-]?hint)"
)
_GLUE_AHEAD = (
    rf"(?:(?:{_GLUE_WORD}|{_GLUE_RUN}(?:{_SEGMENT_START}|(?:{_CSI_CODE}){{1,4}}){_GLUE_WORD})"
    rf"{_GLUE_NAME}|{_GLUE_RUN}{_GLUE_END}(?:[_-]?[0-9]{{1,8}})?|{_GLUE_RUN}{_GLUE_NUMBERED}"
    rf"|{_SEPARATOR_ESCAPE}(?:{_CSI_CODE}){{0,4}}[A-Za-z0-9_.-]{{1,64}}){_ENTRY_SEPARATOR}"
    rf"|{_GLUE_RUN}{_GLUE_SCHEME}"
)
# A line break written as an escape (`\n` in a json.dumps copy, the backslash
# itself escaped up to eight times), perhaps more of them and an indent, then
# an entry (_NEXT_NAME): the next line's.
_NEXT_ENTRY = rf"[nrt](?:\\{{1,8}}[nrt]|[ \t]){{0,8}}+(?:\\{{0,8}}[\"'])?{_NEXT_NAME}"
# A backslash is part of a carrier's value wherever it stands. A run
# of backslashes is read whole, from its start, with the escape it starts
# (`\u0041`, `\x41`, `\101`, `\u{41}`) or the character it escapes (`\X`,
# `\\`, `\,`, `\ `), taken whole (atomic), unless a carrier of its own
# follows in its run, right after the backslashes or after the escape
# (_GLUE_AHEAD: `\\credentials: ...`,
# `retry\u0026Authorization: ...`, `\u001b[32mproxyAuthorization: ...`);
# - an escaped line break or tab (`\n`, `\r`, `\t`, as a json.dumps copy
#   writes a line break) where a value character or another escape follows
#   it, after at most eight more of them, and it starts no next line's entry
#   (_NEXT_ENTRY): _VALUE_BREAK, read at the value's start too
#   (`DB_PASSWORD=\nX...`). Before a space, a quote, the next entry or the
#   end it is a line break, where a value stops as at one written;
# - nothing, before a line break as written or the text's end.
_VALUE_BREAK = (
    rf"\\++(?!{_NEXT_ENTRY})[nrt](?!{_GLUE_AHEAD})"
    r"(?=(?:\\{1,8}[nrt]){0,8}+(?:[^\"'\s,}\\*&]|\\++[^nrt\s\"'\\]))"
)
_BACKSLASH_UNIT = (
    rf"(?!{_QUOTE_ESCAPE})"
    rf"(?:\\++(?!{_GLUE_AHEAD})"
    r"(?>[uUxX]\{[0-9A-Fa-f]{1,8}\}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}"
    r"|x[0-9A-Fa-f]{2}|[0-7]{1,3}|(?![nrt])[^\"'\r\n])"
    rf"(?!{_GLUE_AHEAD})"
    rf"|{_VALUE_BREAK}"
    r"|\\++(?=[\r\n]|\Z))"
)
# A quote inside a value: a single quote, a quote after an odd run (a whole
# escape: `ab\"cd`, `\"ab\\\"cd\"` in a json.dumps copy) or an escape that
# writes one (_QUOTE_ESCAPE), where the
# value goes on right after it (after at most eight more of them): a value
# character, or an escape that is no line break, and no carrier of its own in
# its run (_GLUE_AHEAD: `abc\"password\": ...` is a value, then another name's), nor a quoted
# entry of any name (`\&#34;''FAKEyaml0123': '<v>'`: the name, its closing quote and
# separator; the base stopped at every quote, so the entry keeps its own carrier). A double quote
# as written is never read (it may close the copy's own string); nor is a
# quote before a space, `,` `}` `]` `)` `:` or the end, which closes the value.
_QUOTE_UNIT = rf"(?:'|(?:\\\\)*+\\[\"']|{_QUOTE_ESCAPE})"
_VALUE_QUOTE = (
    rf"{_QUOTE_UNIT}"
    rf"(?={_QUOTE_UNIT}{{0,8}}+"
    rf"(?!{_GLUE_AHEAD})"
    rf"(?![A-Za-z0-9_.-]{{1,64}}{_ENTRY_SEPARATOR})"
    r"(?:[^\"'\s,}\]):\\*&]|\\++[^nrt\"'\s\\]))"
)
# What a value holds besides its plain characters: a backslash's escape, a
# quote it goes on after, and before its closing quote the escaped
# backslashes of a run: its pairs before a quote as written (`"ab\\"`), its
# quads before an escaped one (`\"ab\\\\\"`; a pair there is half of the
# close's own escape, one level down). Every escape is taken whole, so a
# json.dumps copy, and JSON inside one, keeps every escape it had. The alternatives start apart or
# each reads a run once (possessive), and every lookahead is bounded, so the
# read is linear.
_VALUE_ESCAPE = (
    rf"{_BACKSLASH_UNIT}|{_VALUE_QUOTE}"
    r"|(?:\\\\)++(?=[\"'])|(?:\\\\\\\\)++(?=\\[\"'])"
)
# Plain characters are read as runs and the whole value once (possessive):
# a value ends its pattern, so nothing reads back into it, and a long value
# keeps no backtracking state per character.
# A `;`, a `|`, a raw ESC, a `^` or a URL-encoded quote (`%22`, `%2527`) is
# the value's own, unless an entry whose value the value would leave behind
# starts right after it: a plain name holding a credential's word (as
# _GLUE_AHEAD reads one), or any name after a colour code (`\x1b[32m`,
# `^[[0m`), perhaps in quotes of any spelling (`%22GITHUB_TOKEN%22:`), then
# `:` or `=` (perhaps URL-encoded) and a space, a tab or a line break, or a
# scheme's word and a space (after escaped line breaks too), or spaces and
# tabs before it: where the value would stop and leave the next value
# behind (`password=v1;GITHUB_TOKEN: v2`, `v1|proxy_authorization=Bearer v2`,
# `v1\x1b[32mtoken: v2`, `v1%22GITHUB_TOKEN: v2`, `v1;token :v2`,
# `v1%22secret =\n\nv2`). Before any other
# name the value reads on as before (`pa;ss`, `v1;GITHUB_TOKEN=v2`, an
# escaped name), so it still masks what the base masked with it.
_SEMI_ENTRY = (
    rf"{_NAME_QUOTES}(?:{_CSI_CODE}[A-Za-z0-9_.-]{{1,64}}|{_ANSI_CODE}[A-Za-z0-9_.-]{{0,64}}?"
    rf"(?:{_SEGMENT_START}{_GLUE_WORD}[A-Za-z0-9_.-]{{0,64}}"
    rf"|{_GLUE_END}(?:[_-]?[0-9]{{1,8}})?))"
    rf"{_NAME_QUOTES}(?:[ \t]*+(?::|=|%(?:25)?3[AaDd])"
    r"(?:\s|(?:\\{1,8}[nrt]|[ \t])*+(?i:bearer|basic|token|apikey)[ \t])"
    r"|[ \t]++(?::|=|%(?:25)?3[AaDd]))"
)
_SEMI_UNIT = (
    rf"[;|\x1b](?!{_SEMI_ENTRY})|\^(?!\[{_SEMI_ENTRY})|%(?!(?:25)*+2[27]{_SEMI_ENTRY})"
)
# A URL-encoded backslash before a braced escape (`%5Cu{79}`, `%255Cx{41}`)
# is read whole, as the escape itself is (_BACKSLASH_UNIT): its `}` closes no
# value. Never into a carrier of its own (`%5Cu{3d}Bearer <token>`).
_BRACED_PCT = rf"%(?:25)*+5[Cc][uUxX]\{{[0-9A-Fa-f]{{1,8}}\}}(?!{_GLUE_AHEAD})"
_SECVAL_UNITS = (
    rf"(?:[^\"'\s,}}\\*&;|\x1b^%]++|{_BRACED_PCT}|{_SEMI_UNIT}|{_VALUE_ESCAPE})++"
)
# A value starts at a quote only right after its opening quote (`"\"ab"`,
# `\"\"\"ab\"\"\"`): a quote anywhere else there is an opening quote _ESCQ
# did not take.
_QUOTE_START = rf"(?:(?:\\\\)*+\\?[\"']|{_QUOTE_ESCAPE})"
_VALUE_START = rf"(?:(?<=[\"'])|(?!{_QUOTE_START}))"
# A value that ends in a carrier's own ending glued onto it, with no stop
# between (the words _GLUE_END reads, and the *_PASS carrier's pass_value:
# `password=<v1>token: <v2>`, `<v1>Accesskey :<v2>`,
# `<v1>secret =\n\n<v2>`), where that name's separator has a space before
# or after it: the value stops at the space and would leave the next value
# behind, its name read into this one. It reads on through the
# separator and the gap, as a carrier's (spaces and tabs, then line breaks,
# written or escaped, before no next line's entry, _NEXT_NAME), and an auth
# scheme's word (`<v1>Authorization: Bearer <v2>`), into the next value: both
# are masked as one. So does a value that stopped after an escaped line break
# past the separator (before `\nBearer <token>`, _GLUE_SCHEME), or read on
# past one into a scheme's word and stopped at the space after it
# (`<v1>Authorization=\n\nBearer <v2>` in a json.dumps copy), or read its
# separator and a scheme's word with no space between
# (`<v1>PROXY-AUTHORIZATION=Bearer <v2>`). So does a
# value that stopped at an escape (_GLUE_AHEAD) before a name glued onto the
# rest of the run, with no quote, that ends in such an ending, through the
# escape, the name and its separator (`password=<v1>\X<v1>authorization=<v2>`,
# `<v1>\x41<v1>client_secret: <v2>`). The ending is checked only where a
# value stops, with one fixed-width lookbehind per width, so the read stays
# linear.
_GLUED_END_WORDS = (
    "token",
    "secret",
    "passwd",
    "password",
    "passphrase",
    "pass_phrase",
    "credential",
    "credentials",
    "authorization",
    "cookie",
    # The rc's *_PASS carrier's own (_LOWER_PASS_NAME_WORDS, any case).
    "pass_value",
    "passvalue",
    *(
        f"{word}{dash}key"
        for word in (
            "api",
            "auth",
            "access",
            "secret",
            "account",
            "private",
            "encryption",
            "master",
            "primary",
            "secondary",
            "signing",
            "session",
            "hmac",
            "license",
            "storage",
            "shared",
            "client",
            "root",
            "public",
        )
        for dash in ("", "_", "-")
    ),
    *(f"access{a}key{b}id" for a in ("", "_", "-") for b in ("", "_", "-")),
)


def _ends_in(words: tuple[str, ...], tail: str = "") -> str:
    by_width: dict[int, list[str]] = {}
    for word in words:
        by_width.setdefault(len(word), []).append(re.escape(word) + tail)
    return (
        "(?i:"
        + "|".join(f"(?<={'|'.join(w)})" for _, w in sorted(by_width.items()))
        + ")"
    )


_GLUED_NAME_END = _ends_in(_GLUED_END_WORDS)
# The gap after the separator, as a carrier's, an auth scheme's word, and the
# next value's first character.
_GLUED_GAP = (
    r"(?:[ \t]|\\{1,8}t)*+"
    rf"(?:(?:\r?\n|\\{{1,8}}[nr])(?:\s|\\{{1,8}}[nrt]){{0,16}}+(?!{_NEXT_NAME}))?+"
    r"(?:(?i:bearer|basic|token|apikey)(?:[ \t]|\\{1,8}t)++)?+"
    r"(?=[^\"'\s,}\\*&;|\x1b^%])"
)
_GLUED_SEP_END = _ends_in(_GLUED_END_WORDS, "[:=]")
_GLUED_BREAK_END = _ends_in(_GLUED_END_WORDS, r"[:=]\\[nr]")
_GLUED_SCHEME_END = "|".join(
    _ends_in(_GLUED_END_WORDS, f"[:=]{scheme}")
    for scheme in ("bearer", "basic", "token", "apikey")
)
_GLUED_NEXT = (
    rf"(?:{_GLUED_NAME_END}[ \t]++(?::|=)"
    rf"|{_GLUED_SEP_END}(?=\s|\\{{1,8}}[nrt])"
    rf"|{_GLUED_BREAK_END}(?=\\{{1,8}}[nr])"
    rf"|(?:{_GLUED_SCHEME_END})(?=[ \t])"
    r"|(?i:(?<=\\[nr]bearer)|(?<=\\[nr]basic)|(?<=\\[nr]token)|(?<=\\[nr]apikey))(?=[ \t])"
    rf"|(?!{_QUOTE_ESCAPE})\\++(?>[uUxX]\{{[0-9A-Fa-f]{{1,8}}\}}|u[0-9A-Fa-f]{{4}}"
    r"|U[0-9A-Fa-f]{8}|x[0-9A-Fa-f]{2}|[0-7]{1,3}|[^nrt\"'\s\\])"
    rf"[A-Za-z0-9_-]{{0,256}}+{_GLUED_NAME_END}[ \t]*+(?::|=|%(?:25)?3[AaDd]))"
    rf"{_GLUED_GAP}"
)
# A name that holds BEARER_TOKEN between two runs of its own (the
# BEARER_TOKEN carrier's: `AWS_BEARER_TOKEN_BEDROCK`) ends in no such ending:
# glued onto a value's first run (up to 64 characters before it) before a
# separator with a space before or after it, the value reads on the same way
# (`api_key=<v1>aws_bearer_token_bedrock: <v2>`).
_GLUED_LEAD = (
    r"[A-Za-z0-9_-]{0,64}?(?i:bearer_token)[A-Za-z0-9_]{0,64}+"
    r"(?:[ \t]++(?::|=)|[:=](?=\s|\\{1,8}[nrt]))"
    rf"{_GLUED_GAP}"
)
_SECVAL = (
    rf"{_VALUE_START}(?:{_GLUED_LEAD})?+{_SECVAL_UNITS}"
    rf"(?:{_GLUED_NEXT}{_SECVAL_UNITS})*+"
)
# Name-prefix class for the env/JSON carriers, LENGTH-CAPPED. An uncapped
# ``[A-Za-z0-9_]*`` before a required marker backtracks O(n²) on a long
# alphanumeric run (e.g. a base64 image field), so a benign trajectory could
# stall the redactor for tens of seconds. Real env-var/header names are short;
# ``{0,64}`` makes each match attempt O(1) → overall O(n) (#830 ReDoS fix).
_NAME = r"[A-Za-z0-9_]{0,64}"
# Where a secret, or the name of a field that carries one, may start: not
# right after a letter or a digit (`FAQ.`, `task-implementation` and
# `mask_live_` hold no key), unless that letter or digit ends an escape.
# json.dumps writes a newline, carriage return, tab, backspace or form feed
# as `\n` `\r` `\t` `\b` `\f` and any other control or non-ASCII character
# as `\uXXXX`: that is the copy _redact_field redacts, and text can arrive
# already escaped. Python writes a byte as `\xXX` and a URL a character as
# `%XX`. A secret after any of them starts a word as one after a space does.
# Each lookbehind is fixed-width, as `re` requires; the escapes are checked
# inside the one for the letter or digit (it is tried at every position,
# and a single negative lookbehind is the cheapest form). Every rule with a
# left boundary uses _KEY_START.
_ESCAPES = (
    r"\\[nrtbf]",
    r"\\u[0-9A-Fa-f]{4}",
    r"\\x[0-9A-Fa-f]{2}",
    r"%[0-9A-Fa-f]{2}",
)
_NOT_AFTER_ESCAPE = "".join(f"(?<!{escape})" for escape in _ESCAPES)
# An ANSI colour or cursor code (CSI: ESC, `[`, parameter characters, at
# most one intermediate character, a final letter: `ESC[32m`,
# `ESC[38;2;255;255;255;48;2;0;0;0m`, `ESC[?25l`, `ESC[38:5:208m`,
# `ESC[2 q`) ends in a letter too, before a key in a coloured terminal log.
# Text spells the ESC and the `[` in many ways (_CSI_OPENERS, a
# fixed-width lookbehind each, grouped by width). There is one lookbehind
# per length of up to 48 characters between the opener and the letter; it
# reads the opener's last character (`[`, or the `B` of `%5B`) first, and
# the group of openers only where that is one. Past 48, an `m`, `h` or `l`
# (colour, set and reset mode: the codes that take long lists) after a
# list (49 or more digits, `;` and `:`, one of the last six a `;` or `:`)
# ends a code whatever opened it, as a lookbehind per length does not
# scale; a run of digits alone (a long number, a hash) is no list.
# All this is tried only where a letter follows a parameter or
# intermediate character, `[` or `B`, and not where that can end no code:
# after an intermediate that follows neither a parameter nor an opener
# (`the q`, `task-i`), after a `[` or `B` that ends no opener (`[link]`,
# `about`, `100%5Bx`), or after a run of one to three parameter
# characters, then perhaps an intermediate, that something else starts
# (`a1b`, `0x3f`, base64, `at 12 noon`).
_CSI_OPENERS = (
    # ESC [ as written.
    "\x1b\\[",
    # \e[ and \E[ (printf, terminfo), ^[[ (caret notation: cat -v, less).
    r"(?:\\[eE]|\^\[)\[",
    # %1B[ (URL-encoded ESC), \33[.
    r"(?:%1[Bb]|\\33)\[",
    # \x1b[ and \x1B[ (Python, C, JavaScript), \033[ (shell), #x1B[ (the
    # escape pytest writes in a JUnit XML report).
    r"(?:\\x1[Bb]|\\033|#x1[Bb])\[",
    # %1B%5B (the ESC and the `[` URL-encoded).
    r"%1[Bb]%5[Bb]",
    # \u001b[ and \u001B[ (JSON), \u{1b}[ (Rust, JavaScript), \x{1b}[ (Perl).
    r"\\(?:u001[Bb]|[ux]\{1[Bb]\})\[",
)
_CSI_OPENED = "(?:" + "|".join(f"(?<={opener})" for opener in _CSI_OPENERS) + ")"
# Parameter characters are 0x30 to 0x3F (the digits and `:;<=>?`),
# intermediate characters 0x20 to 0x2F (space and `!"#$%&'()*+,-./`).
_CSI_PARAMETERS = "0-?"
_CSI_INTERMEDIATES = " -/"
_CSI_EITHER = " -?"
_NOT_AFTER_CSI = (
    rf"(?:(?<![{_CSI_EITHER}\[Bb][A-Za-z])|"
    rf"(?<=[^{_CSI_PARAMETERS}\[Bb][{_CSI_INTERMEDIATES}][A-Za-z])|"
    rf"(?<![{_CSI_EITHER}\[][A-Za-z])(?<!%1[Bb]%5[Bb][A-Za-z])|"
    rf"(?<=[\[Bb][A-Za-z])(?<![\[Bb]{_CSI_OPENED}[A-Za-z])|"
    + "".join(
        rf"(?<=[^{_CSI_PARAMETERS}\[Bb][{_CSI_PARAMETERS}]{{{n}}}{last}[A-Za-z])|"
        for last in ("", f"[{_CSI_INTERMEDIATES}]")
        for n in range(1, 4)
    )
    + rf"(?<=[{_CSI_EITHER}][A-Za-z])"
    + "".join(
        rf"(?<![\[Bb]{_CSI_OPENED}"
        + (rf"[{_CSI_PARAMETERS}]{{{n}}}" if n else "")
        + rf"[{_CSI_EITHER}][A-Za-z])"
        for n in range(48)
    )
    + "".join(rf"(?<![0-9;:]{{{48 - n}}}[;:][0-9;:]{{{n}}}[hlm])" for n in range(6))
    + ")"
)
_KEY_START = rf"(?<![A-Za-z0-9]{_NOT_AFTER_ESCAPE}{_NOT_AFTER_CSI})"
# Where the task judge's evidence (redact_key_families) lets a key start: not
# inside an identifier either (`build_dtn_network_topology_config`).
_IDENT_START = rf"(?<![A-Za-z0-9_-]{_NOT_AFTER_ESCAPE}{_NOT_AFTER_CSI})"


def _at_key_start(literal: str) -> str:
    """*literal* (a fixed-width pattern) where a key may start. The literal
    comes first so the engine finds it by its first characters; _KEY_START
    is then checked behind it, not at every position of the text."""
    return rf"{literal}(?<={_KEY_START}{literal})"


# (the lead, 2026-11-14 19:42Z, the same rule as 0.9.3's #208): the
# generic sk- rule masks a key wherever it stands, glued after a letter or a
# digit included (`python3agent.pysk-...`, `mysql -psk-...`, `read` without
# -r: `nsk-...`), as 0.9.2 does. Its one exception is an exact literal, the
# word `task-notification` or `task-notifications` (Claude Code's
# `<task-notification>` tags), case-sensitive and whole: a character that
# continues a word on either side makes another name. No shape is loosened.
# Where the word starts: not right after a character that continues a word
# (_IN_WORD: an ASCII letter or digit, `_`, `-`, or any non-ASCII character),
# unless that letter or digit ends an escape that writes a character that
# ends a word, or a colour code (_NOT_AFTER_CSI). json.dumps writes a
# newline, carriage return, tab, backspace or form feed as `\n` `\r` `\t`
# `\b` `\f` and any other control or non-ASCII character as `\uXXXX`; Python
# writes a byte as `\xXX` and a URL a character as `%XX`. An escape of an
# ASCII character other than a letter, a digit, `_` or `-` (`\n`, `\u003c`,
# `\x3c`, `%3C`) ends a word as that character does; an escape of anything
# else continues it, a non-ASCII character included. Each lookbehind is
# fixed-width, as `re` requires.
_WORD_END_HEX = (
    r"(?:[01][0-9A-Fa-f]|2[0-9A-Ca-cEeFf]|3[A-Fa-f]|40|5[B-Eb-e]|60|7[B-Fb-f])"
)
_NOT_AFTER_WORD_END_ESCAPE = (
    r"(?<!\\[nrtbf])"
    rf"(?<!\\u00{_WORD_END_HEX})"
    rf"(?<!\\x{_WORD_END_HEX})"
    rf"(?<!%{_WORD_END_HEX})"
)
# The escapes that continue a word, as read ahead of it. Their backslash may
# itself be URL-encoded (`%5C`, `%255C`): text with an escape written in it,
# URL-encoded again.
_IN_WORD_ESCAPE = (
    rf"(?:\\|%(?:25)?5[Cc]){{1,8}}(?:u(?!00{_WORD_END_HEX})[0-9A-Fa-f]{{4}}"
    rf"|x(?!{_WORD_END_HEX})[0-9A-Fa-f]{{2}}|U[0-9A-Fa-f]{{8}})"
    rf"|%(?:25)?(?!{_WORD_END_HEX})[0-9A-Fa-f]{{2}}"
)
# A character that continues a word next to `task-notification`: anything but
# ASCII punctuation, space and control characters (every non-ASCII character
# counts, a letter or digit of any script and every dash alike).
_IN_WORD = r"[^\x00-\x2c\x2e\x2f\x3a-\x40\x5b-\x5e\x60\x7b-\x7f]"
_WORD_START = rf"(?<!{_IN_WORD}{_NOT_AFTER_WORD_END_ESCAPE}{_NOT_AFTER_CSI})"
# The one word the generic sk- rule leaves, seen from just after its `sk-`: an
# exact literal, not a shape (no key is the word `notification`).
_TAG_WORD = rf"(?<={_WORD_START}task-)notifications?(?!{_IN_WORD}|{_IN_WORD_ESCAPE})"


# Spaces and tabs as written, escaped (`\t` in a json.dumps copy or in text
# already escaped, the backslash itself escaped up to eight times as in
# _ESCQ) or URL-encoded (%20, %09).
_WS = r"(?:[ \t]|\\{1,8}t|%(?:25)?(?:20|09)|\+)"
# The indent of a next line: as _WS, but no `+` (a diff's added line).
_INDENT = r"(?:[ \t]|\\{1,8}t|%(?:25)?(?:20|09))"
# The `:` or `=` after a carrier's name, as written or URL-encoded once or
# twice (`Authorization%3A%20Basic%20...`, `DB_PASS%3D...`, `%253A`); a
# form-encoded space is `+` (_WS).
_SEP = r"(?:\s*[:=]|%(?:25)?3[AaDd])"
# One line break as written (`\n`, `\r\n`), escaped or URL-encoded.
_NEWLINE = r"(?:\r?\n|\\{1,8}r\\{1,8}n|\\{1,8}n|%0[Dd]%0[Aa]|%0[Aa])"
# A name of its own at the start of a line: `HOME=`, `name:`, `"key":`.
_LINE_WORD = r"(?:\\{0,8}[\"'])?[A-Za-z0-9_.-]{1,64}(?:\\{0,8}[\"'])?[ \t]*"
# A padded base64 run on the next line (`aGVsbG8td29ybGQ=`, `...==`,
# `"...="`) is a value, not a name of its own with `=`, which 0.9.3 masks
#: twelve base64 characters or more,
# a digit, `+` or `/` among them, then its padding. A word with `=` and its
# value on the next line (`proxyAuthorization=\n  <v>`, `HOME=`) is a name.
# A run that a quoted value follows after its `=` (`githubToken2024="<v>"`,
# `2authorization='<v>'`) is a name with its value, as any other word with
# `=` is: the name above never takes that line, and the entry is read under
# its own name (a setting's name above kept the entry's value shown, and a
# credential's masked the name and left the value).
_B64_PADDED = (
    r"(?:\\{0,8}[\"'])?(?=[A-Za-z]{0,63}[0-9+/])[A-Za-z0-9+/]{12,64}={1,2}+"
    r"(?![A-Za-z0-9_.-])(?!\\{0,8}[\"'`][^\s\"'`\\,;)\]}])"
)
_LINE_NAME = rf"(?!{_B64_PADDED}){_LINE_WORD}[:=]"


# A YAML block scalar's indicator after a name's `:` (`apiKey: |`, `>-`,
# `|2+`): the value starts on the next line.
_BLOCK = r"[|>](?:[+-][1-9]?|[1-9][+-]?)?"


# A class's or a def's header right before a carrier's separator: a class
# named like a credential, `class` at a line's start or after whitespace
# (`class Token:`, `    class AccessToken:`), or a def's return annotation
# after its `)` (`def get(self) -> Token:`); prose is no header
# (`maps user -> password:`, `x.class password:`).
# In a json.dumps copy a line starts after the string's quote or an escaped
# line break too (`"class Token:\\n`, `\\nclass Token:`). One fixed-width lookbehind
# per name width and nothing more in it (no boundary: a word that ends in
# `class` before the name holds no credential's entry either), read only
# where three quotes open the next line.
_CODE_HEADER = "|".join(
    rf"(?<=(?:(?<![^\n \t\"'])|(?<=\\[nrt]))class [A-Za-z_][A-Za-z0-9_]{{{n}}}:)"
    rf"|(?<=\) -> [A-Za-z_][A-Za-z0-9_.]{{{n}}}:)"
    for n in range(64)
)
# A later line that opens with three quotes, after a class's or a def's
# header: a docstring's, no value.
_NO_DOCSTRING = (
    rf"(?!(?={_WS}*+(?:{_NEWLINE}(?:{_INDENT}++|(?![-+])))++(?:\\{{0,8}}[\"']){{3}})"
    rf"(?:{_CODE_HEADER}))"
)


# For the scheme carrier (_SCHEME_GAP): after line breaks, written or escaped
# (a blank line in a json.dumps copy: `proxy_authorization=\n\nBearer <token>`),
# an auth scheme and a space start the value, so the scheme is read as one
# and its token is masked. Other carriers read a value there as before.
# Across a break the value never opens with a closing bracket, brace, `;` or
# `,`, quoted or not: code that prints a name and a break
# (`print("password:\n")`) holds no value there.
def _gap(line_name: str, scheme: bool = False) -> str:
    before_scheme = (
        rf"(?:{_NEWLINE}{_INDENT}*+){{1,8}}+(?=(?:Bearer|Basic|Token)[ \t])|"
        if scheme
        else ""
    )
    return (
        rf"{_NO_DOCSTRING}{_WS}*+(?:{_BLOCK}{_WS}*+(?={_NEWLINE}))?+"
        rf"(?:(?:{_NEWLINE}(?:{_INDENT}++|(?![-+])))++(?!{line_name})"
        rf"(?!\\{{0,8}}[\"']?[)\]}};,]))?+"
        rf"(?:{before_scheme}(?={_VALUE_BREAK})(?!\\++[nrt][-+])|(?!{_NEWLINE}))"
    )


# The gap between a name's `:` or `=` and its value: spaces and tabs, and at
# most one line break, when the next line is indented and holds no name of
# its own (a YAML block, `apiKey:\n  <v>`; never `GITHUB_TOKEN=\nHOME=/root`
# or `password:\n  name: x`). The value never starts with a line break,
# but for an escaped one it reads (_VALUE_BREAK: `DB_PASSWORD=\nX...`, the
# lead's ruling on, 2026-11-15 13:33Z; never before the next line's
# entry, `GITHUB_TOKEN=\nHOME=/root`, nor onto a diff's next line,
# `-api_token=\n+the next line`). A next line that opens with three quotes
# is a docstring's, no value, only right after a class's or a def's header
# (_NO_DOCSTRING: `class Token:\n    """...`); after any other name it is
# the value (`password:\n    '''<v>'''`).
# Possessive, so a second pass finds the same gap and nothing more to mask.
# Every carrier uses _GAP, and so does the space between an Authorization
# scheme and its token (`Authorization: Bearer\n  <token>`).
_GAP = _gap(_LINE_NAME)
_SCHEME_GAP = _gap(_LINE_NAME, scheme=True)
# A cookie's value is `name=value` pairs, so an indented next line of them
# is its value (`Cookie:\n  sid=...`); a `name:` of its own still is not.
_COOKIE_GAP = _gap(_LINE_WORD + ":")
# A cookie pair's value in double quotes (`sid="<v>"`, RFC 6265's
# cookie-value), read whole: the pair's run stopped at the quote and left
# the value shown, on the same line and on a next one.
_COOKIE_QUOTED = r'(?<==)\\{0,8}"[^\s;"\'\\*]{0,4096}+\\{0,8}"'
# The names of fields that carry a secret. The carriers below match them in
# text (`NAME=value`, `"name": "value"`), and is_secret_field matches them
# as a whole name: a JSON key at export (publish/redact.py) and an env name
# for config.json (_utils/config_redaction.py). Case is ignored, except in
# _PASS_NAME. A name that ends in _API_KEY_NAME, _KEY_FIELD_NAME or
# _SECRET_NAME_END, or holds BEARER_TOKEN, carries one; the KEY forms are
# spelled out (no bare `_KEY`), so `primary_key` and `sort_key` do not, and
# a price per token is no token: TOKEN right after `per` as a word of its
# own, or after `per` and the kind of token priced (_TOKEN_KINDS:
# `input_cost_per_token`, `COST_PER_INPUT_TOKEN`,
# `output_cost_per_reasoning_token`), or, after a price word just before
# `per` (_PRICE_WORDS, a plural too), after `per` and any one word, the
# price's unit (_PRICED_TOKEN: `cost_per_user_token`,
# `usd_per_request_token`), unless that word is a provider's or a
# credential's (`cost_per_api_access_token`, `price_per_refresh_token`).
# Any other word there makes a credential's name (`MCP_PER_USER_TOKEN`,
# `per_request_token`), as do `developer_token` and `super_secret_token`.
# The words are the classifier's (credential_names), and the lookbehinds
# (one per width) follow TOKEN, so they run only where it matched. Token
# counts end in TOKENS (`n_input_tokens`, `max_tokens`) and are none.
_API_KEY_NAME = r"api[_-]?key"
_KEY_FIELD_NAME = r"(?:master|private)[_-]?key"
_PRICE_TOKEN = "".join(
    rf"(?<!{_KEY_START}per[_-]{kind}TOKEN)"
    for kind in (
        "",
        *(
            "(?:" + "|".join(k for k in _TOKEN_KINDS if len(k) == width) + ")[_-]"
            for width in sorted({len(k) for k in _TOKEN_KINDS})
        ),
    )
)
_PRICE_WORD_BEFORE = (
    "(?:"
    + "|".join(
        f"(?<={word}[_-])" for priced in _PRICE_WORDS for word in (priced, priced + "s")
    )
    + ")"
)
_UNIT_YIELDS = sorted(
    {
        form
        for word in {*_PROVIDER_WORDS, *_CREDENTIAL_WORDS}
        for form in (word, word + "s")
    }
)
_PRICED_TOKEN = (
    "(?:"
    + "".join(
        rf"(?<!{_PRICE_WORD_BEFORE}per[_-][A-Za-z0-9]{{{n}}}[_-]TOKEN)"
        for n in range(1, 25)
    )
    + "|"
    + "|".join(
        rf"(?<=[_-](?:{'|'.join(w for w in _UNIT_YIELDS if len(w) == width)})[_-]TOKEN)"
        for width in sorted({len(w) for w in _UNIT_YIELDS})
    )
    + ")"
)
_SECRET_NAME_ENDINGS = (
    r"SECRET|PASSWORD|PASSWD|"
    r"(?:ACCESS|SECRET|ACCOUNT|PRIVATE|ENCRYPTION)_?KEY|CREDENTIALS?"
)
# A passphrase's name, as a password's is read: PASSPHRASE after anything
# (DB_PASSPHRASE, keyPassphrase), PASS_PHRASE as the whole name or after `_`
# (GPG_PASS_PHRASE; not COMPASS_PHRASE or BYPASS_PHRASE), any case.
_PASSPHRASE_END = r"PASSPHRASE|_PASS_PHRASE"
# Its one value that is no secret (the lead's ruling, review of PR #173,
# 2026-11-13): a switch (the classifier's _SWITCH, the whole value) under a
# passphrase's flag, a flag word (the classifier's _FLAG_WORDS) right before
# it with `_` or `-` (has_passphrase=False, requires-passphrase: true,
# SSH_NO_PASSPHRASE=0, NO_PASS_PHRASE=off). A switch under any other
# passphrase name (SSH_KEY_PASSPHRASE=false, PASSPHRASE=no) is masked. So
# the carrier starts at no flag word, nor right after one, that the
# passphrase and a switch follow (_NOT_AT_A_FLAG), and ends at no
# passphrase that a flag word comes before after `_` or `-`, an escape
# (_ESCAPES: `\nhas_passphrase`, a carrier may start at the `n`) or a
# colour code's `m` (_NOT_AFTER_A_FLAG). No boundary of its own: where the
# name starts is the carrier's _KEY_START.
_FLAG_ENDING = r"(?:PASSPHRASE|PASS_PHRASE)"
_FLAG_VALUE = rf"{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SWITCH}{_VALUE_END}"
_NOT_AT_A_FLAG = (
    rf"(?!{_ESCQ}(?:{'|'.join(_FLAG_WORDS)})[_-]{_FLAG_ENDING}{_FLAG_VALUE})"
    rf"(?!(?=PASS)(?:{'|'.join(f'(?<={word}[_-])' for word in _FLAG_WORDS)})"
    rf"{_FLAG_ENDING}{_FLAG_VALUE})"
)
_NOT_AFTER_A_FLAG = (
    "(?!(?:"
    + "|".join(
        rf"(?<={left}(?:{'|'.join(words)})[_-]{ending})"
        for left in ("[_-]", *_ESCAPES, r"[0-9;:\[]m")
        for ending in ("passphrase", "pass_phrase")
        for words in (
            [word for word in _FLAG_WORDS if len(word) == width]
            for width in sorted({len(word) for word in _FLAG_WORDS})
        )
    )
    + rf"){_FLAG_VALUE})"
)
_SECRET_NAME_END = (
    rf"(?:TOKEN{_PRICE_TOKEN}{_PRICED_TOKEN}|{_SECRET_NAME_ENDINGS}|{_PASSPHRASE_END})"
)
# A secret's name may be numbered after its ending, `_` and one to four
# digits or the digits alone (DB_PASSWORD_2, GITHUB_TOKEN_1, SECRET_2,
# OPENAI_API_KEY_2, PRIVATE_KEY_1, dbPassword2), as a password's is
# (_PASS_NUMBER; the lead's ruling, review of PR #173, 2026-11-13). Only
# digits up to the separator: MAX_TOKENS_2, TOKEN_2FA_ENABLED, TOKEN_V2,
# SECRET_2_OF_3 and PASSWORD_2_HINT go on past them.
_NAME_NUMBER = r"(?:_?[0-9]{1,4})?"
# The words a text must hold for a carrier to match it (_LazyPattern.words):
# each carrier's pattern is a sequence, and its name is one element of that
# sequence that every match must pass through. A text with none of a
# carrier's words cannot match it, so the carrier is not run on it.
# _SECRET_NAME_END is an alternation of TOKEN (then lookbehinds, which are
# zero-width), SECRET, PASSWORD, PASSWD, a word and `_?KEY`, CREDENTIAL
# with an optional S, PASSPHRASE and _PASS_PHRASE, and the carrier's other
# branch is PASS_PHRASE: every branch spells one of these words in full, as
# letters of the pattern with nothing between them. Its branch of names the
# classifier decides (_DECIDED_NAME) spells one of the whole words or
# session.
_SECRET_NAME_WORDS = (
    "token",
    "secret",
    "password",
    "passwd",
    "key",
    "credential",
    "passphrase",
    "pass_phrase",
    "session",
)
# A word that only a decided name brings (_decided_name: SESSION_2,
# SESSION_V2, session_hint): a text counts as holding it only where it is
# followed as such a name follows it, so prose with the word alone does not
# run the carrier (review R192, L3).
_WORD_TAILS = {
    "session": re.compile(
        r"sessions?[_-]?(?:[0-9]|[a-z]{1,3}[0-9]|"
        + "|".join(_PASSWORD_QUALIFIERS)
        + ")"
    ),
}
# The BEARER_TOKEN carrier spells BEARER_TOKEN between two _NAME runs.
_BEARER_TOKEN_WORDS = ("bearer_token",)

# Names the classifier decides (the lead's ruling, part B,
# 2026-11-14), read by the secret-name carrier's second branch in text
# (_DECIDED_NAME) and by is_secret_field as a whole name. Each is a shape the
# redactor reads; the answer is entry_kind's, with no rule of its own:
# - R1, a numbered name (_numbered): one of _NUMBERED_WORDS, perhaps
#   plural, then a number, or one to three letters and a number, as the
#   classifier's _qualified reads a number beside a credential word
#   (KEY_2, KEY_10, TOKEN_12345, SECRET_2_OF_3, TOKENS_2, SESSION_2,
#   TOKEN_V2, TOKEN_2FA, KEY_X509C_FILE; TOKEN_2FA_ENABLED and
#   MAX_TOKENS_2 are its settings). After a plural, only a number that ends
#   the name: a unit after it makes a count's bucket, and a usage count is a
#   setting (LiteLLM's tokens_1h and tokens_5m, its cache writes by TTL).
# - R2, a name that ends in a password's qualifier after another word (the
#   classifier's _PASSWORD_QUALIFIERS: id_token_hint holds the ID token,
#   key_hint a key's first letters; PASSWORD_HINT is a setting).
# - A primary or a secondary key's name (the classifier's _KEY_NAMES:
#   primary_key, primaryKey, COSMOS_PRIMARY_KEY), whose value decides. In
#   text only a value of the classifier's _KEY_VALUE shape is offered (an
#   identifier or a number is a schema's to the classifier, and a list such
#   as `["id", "ts"]` is no such run).
# - A name that holds a credentials file's path or its content (the
#   classifier's _PATH_OR_CONTENT_NAMES: GOOGLE_APPLICATION_CREDENTIALS, in
#   any case, no prefix), whose value decides: a path is kept, inline JSON
#   masked whole (_JSON_VALUE, below).
# The bare words key, session and tokens carry no number and are read by
# none of these (record fields, decided after the rc).
_NUMBERED_WORDS = tuple(word for word in _SECRET_NAME_WORDS if "_" not in word)


def _numbered(words: list[str]) -> bool:
    """Whether one of _NUMBERED_WORDS in *words* is numbered (R1)."""
    for index, word in enumerate(words[:-1]):
        plural = word.endswith("s") and word[:-1] in _NUMBERED_WORDS
        if not plural and word not in _NUMBERED_WORDS:
            continue
        after = words[index + 1 :]
        if after[0].isdigit() and (not plural or len(after) == 1):
            return True
        if (
            not plural
            and after[0].isalpha()
            and len(after[0]) <= 3
            and len(after) > 1
            and after[1].isdigit()
        ):
            return True
    return False


def _decided_by_name(words: list[str]) -> bool:
    """Whether a name of *words* is one the classifier decides: numbered
    (R1), a hint (R2) or a primary or a secondary key's."""
    hinted = len(words) > 1 and words[-1] in _PASSWORD_QUALIFIERS
    return (
        _numbered(words)
        or hinted
        or tuple(words[-2:]) in _KEY_NAMES
        or tuple(words) in _PATH_OR_CONTENT_NAMES
    )


_NUMBERED_WORD = "(?:" + "|".join(_NUMBERED_WORDS) + ")s?"
# Such a name starts where an identifier does: with a letter or `_`, and
# never at an escape's own letter or digits (the n of `\n`, the 1 of
# `\u001b`, the 2 of `%22`) or a colour code's (the 32 of `ESC[32m`): the
# classifier decides the name as written, and `nkey_hint` or `32mkey_hint`
# is no name of its. A first look reads the identifier to its end at once
# (at most 128 characters, the longest such name) and goes on only where a
# `:` or `=` follows it: inside a long identifier the names' backtracking is
# not tried at every word start.
_DECIDED_START = (
    r"(?=[A-Za-z_])(?<!\\)(?<!%)(?<!%[0-9A-Fa-f])(?<!\\[ux][0-9A-Fa-f])"
    r"(?<!\\u[0-9A-Fa-f]{2})(?<!\\u[0-9A-Fa-f]{3})"
    rf"(?=[A-Za-z0-9_-]{{1,128}}+{_ESCQ}{_SEP})"
)


def _decided_name(schema: str) -> str:
    """The numbered, hint and key names as one pattern; a key's name sets
    the empty group *schema* (its value is offered only in _KEY_VALUE's
    shape)."""
    return (
        rf"(?:{_NAME}{_NUMBERED_WORD}(?:[_-]?[0-9]|[_-]?[A-Za-z]{{1,3}}[0-9])"
        r"[A-Za-z0-9_-]{0,48}+"
        rf"|{_NAME}{_NUMBERED_WORD}[_-]?(?:{'|'.join(_PASSWORD_QUALIFIERS)})"
        rf"|{_NAME}(?:{'|'.join('[_-]?'.join(words) for words in _KEY_NAMES)})"
        rf"(?P<{schema}>))"
    )


# How the carrier reads such a name, by its case: one in upper case, as an
# env name is written (KEY_2, TOKEN_V2, SESSION_2), takes any value; one
# with a lower-case letter (key1, encryption_key_sha256, session2,
# key_hint), mostly a variable in code, takes the code checks a lower-case
# pass name's value gets (_LOWER_ENTRY, _LOWER_ENV, _LOWER_TEXT:
# `key_hint="sk-..."` and `"key1": "v"` are masked, `key1 = row[0]` and
# `encryption_key_sha256=cpk.key_hash` stay). A value those checks call
# code goes to the carrier's own names as before (`db_password_2 = x` is
# masked whatever it holds). The path-or-content name takes any value in
# any case.
_UPPER_NAME = r"(?-i:(?![A-Z0-9_-]{0,128}+[a-z]))"
_LOWER_NAME = r"(?-i:(?=[A-Z0-9_-]{0,128}+[a-z]))"
_CONTENT_NAME = (
    "(?:"
    + "|".join("[_-]?".join(words) for words in _PATH_OR_CONTENT_NAMES)
    + r")(?![A-Za-z0-9_-])"
)
_DECIDED_NAME = (
    rf"(?:{_CONTENT_NAME}(?P<content>)|{_UPPER_NAME}{_decided_name('schema')})"
)
# What may stand between where the carrier starts and such a name: the rest
# of an escape (the n of `\\n`, `u001b`, the 22 of `%22`) or of a colour
# code (`32m` after `ESC[`). The carrier's own names take it as part of the
# name; these branches skip it, keep it with the name, and the classifier
# decides the name itself (`\\nGOOGLE_APPLICATION_CREDENTIALS={...}`).
_ESCAPE_TAIL = (
    r"(?:(?<=\\)(?:[nrtbf]|u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2})"
    r"|(?<=%)[0-9A-Fa-f]{2}|(?<=\[)[0-9;:]++[A-Za-z])?+"
)
_DECIDED_LOWER_NAME = _decided_name("schema_lower")
# The whole of an inline JSON value under the path-or-content name: from
# its `{` to the `}` that closes it, read as JSON reads it, so a brace or an
# escaped quote inside a string does not count, three levels deep (the
# lead's ruling,, 2026-11-14). In running text the document may go on
# over line breaks, as a pretty-printed service account does, into each next
# line that starts as a document's line does (an indent, then a quote, a
# brace, a bracket, a comma, a colon, a digit, `-` or a literal's letter)
# and holds no such name of its own; never into a line of other text. One
# not closed there is masked to its first line's end, as before. Inside a
# JSON string (a serialized record: the document's first quote escaped) the
# same, spelled as the copy spells it, and an unescaped quote ends the
# string, so one not closed stops there and the record's own `"}` stays.
# Possessive: a reading that fails reads only the document's own lines, and
# no line is read by two such readings. Past _JSON_CAP characters a document
# counts as not closed (_secret_name_value). A document not closed, or
# nested deeper than three levels (a service account's is flat), is masked
# to its first line's end and then line by line while each next line keeps
# a JSON key's shape, after its key, to that line's end (_open_document; the
# lead's ruling, 2026-11-14). Inside a JSON string the same, its lines
# ending at `\n`; its reading crosses a line break only into a line that
# starts as a document's does and holds no such name, as in running text, so
# no line is read by two readings there either (review R192, M1). A list
# under the name is read the same way from its `[` to the `]` that closes
# it, an object inside it as at the top, a list inside it one level deep
# (_JSON_LIST); one not closed is masked as a document is, and a line of
# brackets, braces and commas alone (its items' own `{`, `},` and `[`) is
# kept and read past (review R192, L2: a mapping or a list is masked whole
# in every layer).
_JSON_CAP = 64 * 1024
_RAW_STRING = r'"(?:[^"\\\r\n]|\\.)*+"'
_RAW_BREAK = r"\r?\n(?=[ \t]*+[\"{}\[\],:0-9tfn-])" + rf"(?![^\r\n]*?{_CONTENT_NAME})"
_RAW_UNIT = rf'(?:[^"{{}}\r\n]++|{_RAW_STRING}|{_RAW_BREAK})'
# A JSON string inside a JSON string, as the copy spells it: `\"`, then
# characters, each escape of the document doubled (`\\n`, `\\\"`,
# `\\\\`), then `\"`; and the document's other characters, its own line
# breaks and tabs escaped once (`\n`).
_COPY_STRING = r'\\"(?:[^"\\\r\n]|\\\\(?:\\[\\"]|[^\\"\r\n]))*+\\"'
# A line of a JSON string's copy (no unescaped quote, no `\n` break; an
# escape is one unit, so the document's own `\\n` is no break).
_COPY_LINE = r'(?:[^"\\\r\n]|\\[^n\r\n])*+'
_COPY_BREAK = (
    r"\\n(?=(?:[ \t]|\\t)*+(?:\\\"|[{}\[\],:0-9tfn-]))"
    + rf"(?!{_COPY_LINE[:-2]}*?{_CONTENT_NAME})"
)
_COPY_UNIT = rf'(?:[^"\\{{}}\r\n]++|{_COPY_BREAK}|\\[^"n\r\n]|{_COPY_STRING})'
# A line of the copy that keeps a JSON key's shape.
_COPY_KEY_LINE = rf'\\n(?:[ \t]|\\t)*+\\"[^"\\]*+\\"[ \t]*+:{_COPY_LINE}'
# A copy of a copy (its first quote escaped more than once) is read with an
# escape as one unit, braces ending or opening, and an unescaped quote
# ending the string, as one level was before.
_DEEP_UNIT = r'(?:[^"\\{}\r\n]++|\\.)'
# Where the document's first quote is, past spaces and escaped line breaks.
_FIRST_QUOTE = r"(?:[ \t]|\\{1,8}[nrt])*+"
# A list's units: the same, a bracket ending or opening a list.
_RAW_LIST_UNIT = rf'(?:[^"{{}}\[\]\r\n]++|{_RAW_STRING}|{_RAW_BREAK})'
_COPY_LIST_UNIT = rf'(?:[^"\\{{}}\[\]\r\n]++|{_COPY_BREAK}|\\[^"n\r\n]|{_COPY_STRING})'
_DEEP_LIST_UNIT = r'(?:[^"\\{}\[\]\r\n]++|\\.)'
# Where a list's first quote is: past its first items' brackets, braces,
# numbers, spaces and escaped line breaks, up to 64 characters (bounded, so
# no reading of the name repeated along a line reads its line again).
_FIRST_ITEM = r'(?:[^"\\\r\n]|\\{1,8}[nrt]){0,64}'

# A line of a list's brackets, braces and commas alone, and the same in a
# JSON string's copy.
_BRACKET_LINE = r"\r?\n[ \t]*+[\[\]{},][\[\]{}, \t]*+\r?+(?=\n|\Z)"
_COPY_BRACKET_LINE = (
    r'\\n(?:[ \t]|\\t)*+[\[\]{},](?:[\[\]{}, \t]|\\t)*+(?:\\r)?+(?=\\n|"|\Z)'
)
_BRACKET_LINE_START = re.compile(r"[ \t]*[\[\]{},][\[\]{}, \t]*\r?\Z")
_COPY_BRACKET_LINE_START = re.compile(
    r"(?:[ \t]|\\t)*[\[\]{},](?:[\[\]{}, \t]|\\t)*(?:\\r)?\Z"
)
# A line of a list that opens an item, perhaps after the last one's close
# (`  {"private_key_id": "<v>"},`, `  }, {"a": ...`): masked after its `{`,
# as a key line is after its key (review R192B, L-3: compact items, one a
# line, were read past by neither rule).
_ITEM_LINE = r"\r?\n[ \t]*+(?:[\]},][ \t]*+)*+\{[^\r\n]*+"
_COPY_ITEM_LINE = rf"\\n(?:[ \t]|\\t)*+(?:[\]}},](?:[ \t]|\\t)*+)*+\{{{_COPY_LINE}"
_ITEM_LINE_START = re.compile(r"[ \t]*(?:[\]},][ \t]*)*\{")
_COPY_ITEM_LINE_START = re.compile(r"(?:[ \t]|\\t)*(?:[\]},](?:[ \t]|\\t)*)*\{")


def _json_object(unit: str, depth: int) -> str:
    """A `{...}` of *unit*s with objects *depth* - 1 levels inside."""
    body = rf"\{{{unit}*+\}}"
    for _ in range(depth - 1):
        body = rf"\{{(?:{unit}|{body})*+\}}"
    return body


# The rest of a JSON string, to its unescaped closing quote: where a
# document inside one is nested deeper or does not close.
_STRING_REST = r'(?:[^"\\\r\n]|\\.)*+'
# A line of a document that keeps a JSON key's shape: an indent, a quoted
# key, `:`, then anything to the line's end.
_KEY_LINE = r'\r?\n[ \t]*+"(?:[^"\\\r\n]|\\.)*+"[ \t]*+:[^\r\n]*+'
_KEY_LINE_START = re.compile(r'[ \t]*"(?:[^"\\\r\n]|\\.)*"[ \t]*:')
# The same line in a JSON string's copy, its line breaks written `\n`: an
# `n` after an odd run of backslashes (`\\n` inside one of its strings is an
# escape of the document's own, no line break).
_COPY_KEY_LINE_START = re.compile(r'(?:[ \t]|\\t)*\\"[^"\\]*\\"[ \t]*:')
_COPY_LINE_BREAK = re.compile(r"(?<!\\)((?:\\\\)*)\\n")


def _next_line(text: str, copy: bool) -> tuple[str, str, str]:
    """*text*'s first line, the line break after it and the rest, as
    str.partition gives them; in a JSON string's copy the break is `\n`."""
    if not copy:
        return text.partition("\n")
    found = _COPY_LINE_BREAK.search(text)
    if found is None:
        return text, "", ""
    return text[: found.end(1)], "\\n", text[found.end() :]


_JSON_VALUE = (
    r"\{(?:"
    rf"(?={_FIRST_QUOTE}\\\")(?P<copy>)(?:"
    rf"(?:{_COPY_UNIT}|{_json_object(_COPY_UNIT, 2)})*+\}}"
    rf"|(?P<open_copy>{_COPY_LINE}(?:{_COPY_KEY_LINE})*+))"
    rf"|(?={_FIRST_QUOTE}\\{{2,8}}\")"
    rf"(?:{_DEEP_UNIT}|{_json_object(_DEEP_UNIT, 2)})*+(?:\}}|{_STRING_REST})"
    rf"|(?:{_RAW_UNIT}|{_json_object(_RAW_UNIT, 2)})*+\}}"
    rf"|(?P<open>[^\r\n]*+(?:{_KEY_LINE})*+))"
)


def _json_list(unit: str, list_unit: str) -> str:
    """A list's body after its `[`, to its `]`: *list_unit*s, objects of
    *unit*s two levels deep and lists of them one level deep."""
    inner = rf"\[(?:{list_unit}|{_json_object(unit, 1)})*+\]"
    return rf"(?:{list_unit}|{_json_object(unit, 2)}|{inner})*+"


_JSON_LIST = (
    r"\[(?:"
    rf"(?={_FIRST_ITEM}\\\")(?P<list_copy>)(?:"
    rf"{_json_list(_COPY_UNIT, _COPY_LIST_UNIT)}\]"
    rf"|(?P<open_list_copy>{_COPY_LINE}(?:{_COPY_KEY_LINE}|{_COPY_BRACKET_LINE}|{_COPY_ITEM_LINE})*+))"
    rf"|(?={_FIRST_ITEM}\\{{2,8}}\")"
    rf"{_json_list(_DEEP_UNIT, _DEEP_LIST_UNIT)}(?:\]|{_STRING_REST})"
    rf"|{_json_list(_RAW_UNIT, _RAW_LIST_UNIT)}\]"
    rf"|(?P<open_list>[^\r\n]*+(?:{_KEY_LINE}|{_BRACKET_LINE}|{_ITEM_LINE})*+))"
)


# The Cookie and Authorization header names (the classifier's _COOKIE_NAMES
# and _AUTHORIZATION_NAMES: Cookie, Set-Cookie, Authorization,
# Proxy-Authorization), their words joined by `_` or `-`, and the same names
# after a prefix (HTTP_COOKIE and REDIRECT_HTTP_AUTHORIZATION in a CGI or
# WSGI env dump, has_authorization). The header's own name holds a
# credential whatever it holds; a prefixed one only when its value is not
# prose (the classifier's _PROSE, as entry_kind reads it; the lead's ruling,
# 2026-11-14), so `has_authorization: 1`, `has_cookie: true` and
# `cookie_policy: strict` stay. In running text the Cookie header's own
# name still masks only cookie pairs (`Set-Cookie: header too large`).
def _header_name(names: tuple[tuple[str, ...], ...]) -> str:
    """A header's names as one pattern, longest first."""
    longest = sorted(names, key=lambda words: len(words), reverse=True)
    return "(?:" + "|".join("[_-]".join(words) for words in longest) + ")"


_COOKIE_NAME = _header_name(_COOKIE_NAMES)
_AUTHORIZATION_NAME = _header_name(_AUTHORIZATION_NAMES)
# Words (see _SECRET_NAME_WORDS): a header's last word, which every one of
# its names spells in full; and the letters its names start with, where a
# carrier may start.
_COOKIE_WORDS = tuple(sorted({words[-1] for words in _COOKIE_NAMES}))
_AUTHORIZATION_WORDS = tuple(sorted({words[-1] for words in _AUTHORIZATION_NAMES}))
_HEADER_WORDS = frozenset((*_COOKIE_WORDS, *_AUTHORIZATION_WORDS))
_COOKIE_STARTS = "".join(sorted({words[0][0] for words in _COOKIE_NAMES}))
_AUTHORIZATION_STARTS = "".join(sorted({words[0][0] for words in _AUTHORIZATION_NAMES}))


# A header's name after a prefix, as the classifier reads one (HTTP_COOKIE,
# has_cookie, x-set-cookie, REDIRECT_HTTP_AUTHORIZATION): the carrier
# starts at the header's name, which follows `_`, `.` or `-` after a letter
# or a digit, and is no header's own name read from its second word on
# (set_cookie is Set-Cookie, proxy_authorization is Proxy-Authorization;
# x_set_cookie is prefixed). A camel-case prefix (httpCookie, http2cookie)
# is read in a quoted entry only (_COOKIE_CAMEL, below): elsewhere in text
# the header's word follows a letter or a digit, and in the corpus every
# such name before `:` or `=` is code (SimpleCookie:, NetworkCookie=,
# BADCOOKIE= 23).
def _prefixed(names: tuple[tuple[str, ...], ...]) -> str:
    """Where one of a header's *names* starts after a prefix."""
    heads = sorted({"[_-]".join(words[:-1]) for words in names if len(words) > 1})
    return "(?<=[A-Za-z0-9][_.-])" + "".join(
        rf"(?:(?<!{head}[_-])|(?<=[A-Za-z0-9_.-]{head}[_-]))" for head in heads
    )


_COOKIE_PREFIXED = _prefixed(_COOKIE_NAMES)
_AUTHORIZATION_PREFIXED = _prefixed(_AUTHORIZATION_NAMES)
_AUTH_SCHEMES = r"(?:Bearer|Token|Basic|ApiKey|Negotiate|NTLM|Bot|Key|SSWS)"
# A scheme word's token under a prose-read Authorization name: it runs through
# `&` and `&amp;` as written or decoded (R192C's scheme_edge, F2b 2026-11-16
# 05:54Z) and stops at whitespace, a quote as written or as an entity at any
# depth, a delimiter, a backslash or an earlier mask.
_SCHEME_TOKEN = (
    r"(?:[^\s\"',;)\]}\\*&]|&(?:amp;)*+(?!(?:quot|apos|#0*+3[49]|#[xX]0*+2[27]);))++"
)
# Digest's credential is its parameter list (`username="...", response="..."`,
# the quotes escaped in a JSON copy): masked whole after the scheme.
_DIGEST_PARAM = (
    r"[A-Za-z0-9_-]{1,32}=(?:\\{0,8}\"[^\"\\\r\n]{0,512}\\{0,8}\"|[^\s,\"\\]{1,512})"
)
# In upper case, with the underscore: `PWD=/app` and `BYPASS=1` are no
# secrets. A numbered one is one too (_PASS_NUMBER: DB_PASS_2,
# SMTP_PASS_01, MYSQL_PWD_2), and so is PASS_VALUE or PASSVALUE, alone or
# after `_` (_PASS_VALUE_NAME: DB_PASS_VALUE), and PASS and a number as the
# whole name (_PASS_ALONE: PASS_2, PASS_10); not PASS_2_OF_3, PASS_1_RATE,
# PASS2, BYPASS_2, COMPASS_VALUE or PASS_RATE. The same names in lower and
# camel case (db_pass, dbPass) are read below (_LOWER_PASS), where a value
# that is code is no secret.
_PASS_NAME = r"[A-Z0-9_]{1,64}_(?:PASS|PWD)"
_PASS_NUMBER = r"_[0-9]{1,4}"
_PASS_VALUE_NAME = r"(?:[A-Z0-9_]{1,64}_)?(?:PASS_VALUE|PASSVALUE)"
# Not after `_`: the carrier also starts inside a name (TESTS_PASS_2).
_PASS_ALONE = rf"(?<!_)PASS{_PASS_NUMBER}"
# Words (see _SECRET_NAME_WORDS): the name ends in `_` and then PASS or PWD
# (and perhaps a number), or it holds PASS_ (PASS_VALUE, PASS_2) or
# PASSVALUE, in upper case as this carrier is matched; in lower and camel
# case it ends in `_pass`, `Pass`, `_pwd` or `Pwd` (and perhaps a number),
# or it holds pass_ (pass_value, pass_2), passvalue or passValue.
_PASS_NAME_WORDS = ("_PASS", "_PWD", "PASS_", "PASSVALUE")
_LOWER_PASS_NAME_WORDS = (
    "_pass",
    "Pass",
    "_pwd",
    "Pwd",
    "pass_",
    "passvalue",
    "passValue",
)
# A *_PASS name that is a test's result, no password. A name that is
# FAIL_TO_PASS or PASS_TO_PASS (SWE-bench's lists of the tests a fix must
# make pass), or another of the classifier's pass names by name
# (_NOT_SECRET_NAMES: TOKENS_TO_PASS, ON_PASS, NORM_PASS), whatever it
# holds, as a whole name: COMMON_PASS, SESSION_PASS and X_ON_PASS only end
# in one (its first word starts the name or follows an escape, and no
# name's character comes before it); and one whose word before `_PASS` counts
# tests or names an outcome (TESTS_PASS, ALL_TESTS_PASS, TOTAL_PASS, N_PASS,
# FIRST_PASS, IS_PASS, ALL_BLOCKERS_PASS; not ADMIN_PASS or LATEST_PASS)
# when it holds a test result (_TEST_RESULT; after a pure count's word,
# N_PASS or TOTAL_PASS, a count of any length: the classifier's
# _COUNT_RESULT, the lead's ruling, review of PR #173, 2026-11-13). The word
# starts the name, follows `_` or follows an escape (`%3DTOTAL_PASS`). In
# text both are checked by lookbehind at the end of a matched name, one per
# width of word, so they cost nothing elsewhere.
_TEST_LIST_NAMES = sorted(
    "_".join(words).upper()
    for words in _NOT_SECRET_NAMES
    if words[-1] == "pass" and len(words) > 1
)
_TEST_LIST_NAME_END = "".join(
    rf"(?<!(?<![A-Z0-9_]{_NOT_AFTER_ESCAPE}){name})" for name in _TEST_LIST_NAMES
)


def _pass_name_end(words: tuple[str, ...]) -> str:
    """One lookbehind per width of word: one of *words* just before `_PASS`."""
    return "|".join(
        rf"(?<=(?<![A-Z0-9]{_NOT_AFTER_ESCAPE})(?:{'|'.join(same)})_PASS)"
        for same in (
            [word for word in words if len(word) == width]
            for width in sorted({len(word) for word in words})
        )
    )


_COUNT_NAME_END = _pass_name_end(tuple(word.upper() for word in _PASS_COUNT_WORDS))
_OUTCOME_NAME_END = _pass_name_end(
    tuple(
        word.upper()
        for word in (
            *_PASS_OUTCOME_WORDS,
            *("_".join(pair) for pair in _PASS_OUTCOME_PAIRS),
        )
        if word not in _PASS_COUNT_WORDS
    )
)
# The value such a name holds as a test result is the classifier's
# (credential_names.is_test_result): in text, its _TEST_RESULT pattern (a
# count of up to five digits, perhaps one of two, or a switch word: imported
# as _COUNT_OR_SWITCH, the one definition), or a list that closes on its
# line. A whole name asks entry_kind. Under any other *_PASS or *_PWD name
# (DB_PASS, SMTP_PASS, MYSQL_PWD) every value is masked: `12345!` and
# `12/34` are passwords there.
# Text sees a list as written or JSON-escaped in a serialized line
# (`[\"tests/a.py::t\"]`), which the classifier's list pattern cannot read.
# So text keeps one only when a `]` closes it on its line, within
# _LIST_SCAN characters (the lead's ruling, review of PR #173, 2026-11-13):
# before a newline, written plain or escaped (`\n`, `\\n`), and, when its
# items' quotes are escaped (once, twice or three times) or it holds
# numbers, before a quote escaped fewer times, where the string that holds
# the list ends (`"TESTS_PASS": "[\"a\"", "x": "]"`). So a serialized line
# reads as the line it holds, one level down. Its items are JSON's: strings
# in double quotes, perhaps escaped, or numbers (not `[hunter2`,
# `[1hunter2]` or Python's `['a']`, no list to the classifier either). A
# list cut off (`TESTS_PASS=["a`, `[1, 2`) is masked, and so is one that
# goes on to the next line. Each scan is one bounded run.
_LIST_SCAN = 500


def _list_items(stop: str = "", quote: str = "") -> str:
    """A bounded run of a list's text: no `]`, no line's end, no *stop*.
    *quote*, an item's escaped quote, goes whole, and any other backslash
    goes with the character after it, so a line's end written with any
    number of them ends the run, and so does a quote escaped fewer times
    than *quote*."""
    item_quote = f"|{re.escape(quote)}" if quote else ""
    return (
        rf"(?:[^\]\n\r\\{stop}]{item_quote}|\\[^nr\n\r\\{stop}]|\\(?=\\))"
        rf"{{0,{_LIST_SCAN}}}+"
    )


_LIST_CLOSED = (
    r"\[[ \t]*+(?:"
    + "".join(
        f"{re.escape(quote)}{_list_items(chr(34), quote)}|"
        for quote in ("\\" * count + '"' for count in (1, 3, 7))
    )
    + rf"\"{_list_items()}"
    + rf"|-?[0-9][0-9.,eE+ \t-]{{0,{_LIST_SCAN}}}+"
    + r"|)\]"
)
_TEST_RESULT = rf"(?:{_COUNT_OR_SWITCH}|{_LIST_CLOSED})"
_COUNT_OF_ANY_LENGTH = rf"(?:{_COUNT_RESULT}|{_LIST_CLOSED})"
# A masked value that opens a list masks its first item (`["hunter2`), not
# the `[` alone.
_LIST_START = r"(?:\[[ \t]*+\\{0,8}[\"'])?"
# Prose after an Authorization name, no credential: the classifier's
# _PROSE, a plain word (letters only, fifteen or fewer: `required`, `none`,
# `Bearer` alone) or a count or switch word (`1`, `true`), perhaps closing
# a sentence or a bracket.
_PROSE_VALUE = _PROSE
# A secret name's value may open with an Authorization scheme, a space and
# then the token (`x-api-key: Bearer <token>`, `OPENAI_API_KEY=Bearer
# <token>`, `x-api-key%3DBearer <token>`): the scheme and the space are kept,
# as the Authorization carrier keeps them, and the token is masked (RPUB's
# MEDIUM, review of PR #173, 2026-11-13: the carrier took the scheme for the
# value and left the token, which publish's own Bearer rule no longer saw).
# A scheme word with no value after it (`TOKEN=Bearer`) or before prose
# (_PROSE_VALUE: `password: Basic authentication is required`) is the value,
# masked as before. Possessive, so a second pass finds nothing more.
_SCHEME = rf"(?:{_AUTH_SCHEMES}{_WS}++(?!{_PROSE_VALUE}))?+"
# A scheme word glued to the end of the value's first word, then a space
# and a token (review of PR #173, 2026-11-13, RPUB's L1: `GH_TOKEN=abcBearer
# <token>`, `x-api-key%3A%20a1bBearer <token>` masked `abcBearer` and left
# the token): the word, the scheme, the space and the token are the value,
# masked as one. Before prose (`password: abcBasic authentication is
# required`) the first word alone is the value, as before, and so it is
# before anything shorter than a token's eight characters (`token:
# EventLoopToken | None`, `api_key: os.environ/X_API_KEY  # note`: code and a
# comment, no token). The same scheme words and prose as _SCHEME; the run
# before the scheme is bounded and the rest possessive, so a second pass
# finds nothing more.
# The value under the path-or-content name that is no document, read
# through backslashes so a Windows path (`C:\\Users\\me\\sa.json`,
# `%USERPROFILE%\\adc.json`) is judged whole by the classifier, which keeps a
# path and masks anything else (review R192, L1). It stops where the other
# values do (a space, a quote, `,`, `}`, `&`, `*`) and before a run of
# backslashes that a quote follows (an escaped closing quote in a JSON copy).
# A run of backslashes is read whole, so the read is linear.
_PATH_RUN = r"(?:[^\"'\s,}*&\\]++|\\++(?![\"']))++"
_BACKSLASH_RUN = re.compile(r"\\{2,}")
_GLUED_SCHEME = (
    rf"(?:[^\"'\s,}}\\*&]{{1,256}}?{_AUTH_SCHEMES}{_WS}++(?!{_PROSE_VALUE})"
    r"(?=[A-Za-z0-9._~+/=-]{8}))?+"
)

# The *_PASS names in lower and camel case, as the classifier reads
# them (credential_names.entry_kind reads every case alike): a name that
# ends in `_pass`, `Pass`, `_pwd` or `Pwd` after a letter or a digit
# (db_pass, dbPass, mysql_pwd, mysqlPwd, by_pass), perhaps numbered as upper
# case is (db_pass_2); pass_value, passvalue or passValue, alone or after
# `_` or a hump (db_pass_value, dbPassValue); pass and a number (pass_2).
# Snake case is lower case (`DB_pass` counts, `Tests_Pass` does not) and a
# hyphen joins no such name. Not compass_value, bypass, getpass, pass_rate,
# pass2 or the bare word `pass`, and not bypass in camel case (ByPass,
# isByPass, byPass_2): the classifier's _ONE_WORDS, no pass word to it.
#
# In text such a name is mostly a variable in code (`first_pass = True`,
# `deterministic_pass=deterministic_pass,`, `tests_pass: bool`), so its
# value is masked only where it is a value, not code. The name must start
# the identifier, with a letter: not after `_`, nor at an escape's own
# letter or digit (the `n` of `\n`, the `2` of `%22`), where it would read
# as another name (`ntotal_pass`). A name that is code's by its place is
# no entry: right after `(`, `,`, `{`, `[` or `.`, perhaps with one space
# or tab between (an argument, a parameter, an object's property, an
# attribute: `f(db_pass="x")`, `self.db_pass = "x"`). Then, by its shape:
#
# - _LOWER_ENTRY, a mapping's entry: a quoted name, `:` or `=`, a quoted
#   value (`"db_pass": "v"`, `'dbPass': 'v'`, and the json.dumps copy every
#   dict value is redacted through, _redact_field): read as upper case is,
#   every value masked but a test's result under a name that counts tests
#   (below).
# - _LOWER_ENV, an env line: `name=value`, nothing between the name, `=`
#   and the value (`db_pass=v`, `export dbPass=v`, `?db_pass=v&`). A quoted
#   value is masked; so is an unquoted one that is no code (_ENV_VALUE).
# - _LOWER_TEXT, any other `:` or `=` (`db_pass: v`, `db_pass = "v"`): a
#   quoted value is masked, and an unquoted one only when it is no
#   identifier, no number and no expression (_TEXT_VALUE).
#
# Code, never masked after a lower-case name: a keyword or switch word, or
# a count of up to five digits, a decimal or a ratio (the classifier's
# _TEST_RESULT, imported as _COUNT_OR_SWITCH, here ended by a backtick too:
# _LOWER_LITERAL; `first_pass=True`, `pass_value=3`, `two_pass=None`); a
# value that starts with an operator, a bracket, a backtick, `$`, `#`, `.`
# or `?` (_CODE_START: `x_pass = [`, `$DB_PASS`, `=> ...`, `== y`); a
# quoted one that is such a literal or starts with `$`, a bracket or `{`
# (`"$DB_PASS"`, `"{{ vault_pass }}"`); an identifier, perhaps followed by
# `.`, `(`, `[`, `)`, `]`, `;`, `:`, `,`, `}`, `*`, a backtick or a space
# (`db_pass`, `os.environ[...]`, `input(...)`, `str`). In an env line only,
# an identifier is a value when it ends the entry (a quote, `&`, a
# backtick, a comment, a line's end or the text's end after it, perhaps
# after spaces: `db_pass=v`) and is no string's prefix (`f"..."`); and
# anywhere an identifier is the start of a value when a character no
# identifier and no call, subscript or statement holds follows it
# (`hunter2!`, `s3cr3t-pw`, `p@ss`).
#
# A name that counts tests keeps a test's result as in upper case: one of
# the classifier's pass names by name (_NOT_SECRET_NAMES: fail_to_pass,
# passToPass, on_pass, norm_pass) as a whole name in snake or camel case
# (_LOWER_TEST_LIST_NAME; not common_pass) whatever it holds, a test's or an
# outcome's (tests_pass, allTestsPass) a test's result (_TEST_RESULT, a
# closed list too), and a pure count's (n_pass, totalPass) a count of any
# length (_LOWER_RESULT). The words are the classifier's, spelled in snake
# and camel case here.
_LOWER_PREFIX = r"[A-Za-z0-9_]{0,63}[A-Za-z0-9]"
# A word of _ONE_WORDS that ends in pass (bypass) is one word in camel case
# too: no `Pass` hump right after `By`, nor after `by` that starts a word
# (x_byPass, 2byPass) or the name (_NOT_ONE_WORD_NAME: byPass, byPass_2,
# byPassValue). by_pass, standbyPass and byPass_db_pass are two words.
_ONE_WORD_HEADS = tuple(
    word[: -len("pass")]
    for word in sorted(_ONE_WORDS)
    if word.endswith("pass") and word != "pass"
)
_NOT_ONE_WORD = "".join(
    rf"(?<!{head.capitalize()}Pass)(?<![_0-9]{head}Pass)" for head in _ONE_WORD_HEADS
)
_NOT_ONE_WORD_NAME = "".join(
    rf"(?!{head}Pass(?:{_PASS_NUMBER}|_value|value|Value)?(?![A-Za-z0-9_]))"
    for head in _ONE_WORD_HEADS
)
_LOWER_PASS_NAME = rf"{_LOWER_PREFIX}(?:_pass|Pass{_NOT_ONE_WORD}|_pwd|Pwd)"
_LOWER_PASS_VALUE_NAME = (
    rf"(?:{_LOWER_PREFIX}(?:_p|P)|p)ass{_NOT_ONE_WORD}(?:_value|value|Value)"
)
_LOWER_PASS_ALONE = rf"pass{_PASS_NUMBER}"


def _camel_spellings(words: tuple[str, ...]) -> tuple[str, str]:
    """*words* in camel case, the first word as written and capitalised
    (failToPass, FailToPass)."""
    rest = "".join(word.capitalize() for word in words[1:])
    return words[0] + rest, words[0].capitalize() + rest


# Read where the name starts, to its end.
_LOWER_TEST_LIST_NAME = (
    "(?!(?:"
    + "|".join(
        spelled
        for words in sorted(_NOT_SECRET_NAMES)
        if words[-1] == "pass" and len(words) > 1
        for spelled in ("_".join(words), *_camel_spellings(words))
    )
    + ")(?![A-Za-z0-9_]))"
)


def _lower_pass_names(words: tuple[str, ...]) -> str:
    """A whole lower- or camel-case name whose last word before `pass` is
    one of *words* (each the classifier's words joined by `_`): after `_`
    or starting the name in snake case (tests_pass, all_tests_pass), and in
    camel case starting the name, after `_` or capitalised after a hump
    (testsPass, all_testsPass, allTestsPass). Read where the name starts
    (the carrier starts it where its identifier does), so it needs no left
    boundary of its own."""
    longest_first = sorted(words, key=lambda word: len(word), reverse=True)
    camel = [_camel_spellings(tuple(word.split("_"))) for word in longest_first]
    snake = "|".join(longest_first)
    starts = "|".join(first for first, _capital in camel)
    humps = "|".join(capital for _first, capital in camel)
    return (
        rf"(?:(?:{_LOWER_PREFIX}_)?(?:(?:{snake})_pass|(?:{starts})Pass)"
        rf"|(?:{_LOWER_PREFIX})?(?:{humps})Pass)"
    )


_LOWER_COUNT_NAME = _lower_pass_names(_PASS_COUNT_WORDS)
_LOWER_OUTCOME_NAME = _lower_pass_names(
    tuple(
        word
        for word in (
            *_PASS_OUTCOME_WORDS,
            *("_".join(pair) for pair in _PASS_OUTCOME_PAIRS),
        )
        if word not in _PASS_COUNT_WORDS
    )
)
# The classifier's test result and count (_COUNT_OR_SWITCH, _COUNT_RESULT)
# end where a carrier's value does (_VALUE_END, the classifier's, as
# _SECVAL reads a value: a backslash's escape goes on); after a lower-case name a
# backtick ends one too, as a Markdown code span does (`first_pass=True`,
# `n_pass=123456`). Derived here, not copied: the shapes stay the
# classifier's.
_LOWER_VALUE_END = _VALUE_END.replace("&]", "&`]")
# Where a key's run ends, for the names it decides by its value (_KEY_VALUE):
# at a backslash too, as before (the value itself is still read whole).
_RUN_END = r"(?![^\"'\s,}\\*&])"
# The rc's own value classes below read _VALUE_ESCAPE too, and start as
# _SECVAL does (_VALUE_START).
_LOWER_LITERAL = _COUNT_OR_SWITCH.replace(_VALUE_END, _LOWER_VALUE_END)
# The classifier's prose, ended by a backtick too: what a prefixed header
# name's unquoted value is no secret as (`has_cookie: true`).
_TEXT_PROSE = _PROSE.replace(_VALUE_END, _LOWER_VALUE_END)
_LOWER_COUNT_OF_ANY_LENGTH = (
    rf"(?:{_COUNT_RESULT.replace(_VALUE_END, _LOWER_VALUE_END)}|{_LIST_CLOSED})"
)
# As upper case: no test's result under an outcome's name, and no count of
# any length under a pure count's.
_LOWER_RESULT = (
    rf"(?!{_LOWER_OUTCOME_NAME}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_TEST_RESULT}"
    rf"|{_LOWER_COUNT_NAME}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_LOWER_COUNT_OF_ANY_LENGTH})"
)
_QUOTE = r"\\{0,8}[\"']"
# A quoted value whole, up to its closing quote on its line (no quote,
# backslash, line end or `*` in it): what a prefixed header's entry holds
# (`"has_cookie": "please sign in first"`), judged and masked as one, as
# entry_kind reads a whole value. It starts with no space, `:`, `,` or
# closing bracket: then the quote before it closed something else
# (`"HTTP_COOKIE=": (f"...`, a key of a dict in code).
_ENTRY_TEXT = rf"(?![\s:,)\]}}]){_VALUE_START}(?:{_BRACED_PCT}|[^\"'\\\r\n*]|{_VALUE_ESCAPE}){{1,4096}}"


def _camel_own_spellings(names: tuple[tuple[str, ...], ...]) -> list[str]:
    """A header's own names of two words or more in camel case, the first
    word as written and capitalised (setCookie, SetCookie,
    proxyAuthorization, ProxyAuthorization)."""
    return sorted(
        {
            spelled
            for words in names
            if len(words) > 1
            for spelled in _camel_spellings(words)
        }
    )


def _camel_prefixed(names: tuple[tuple[str, ...], ...]) -> str:
    """A quoted name that ends in one of a header's *names* after a camel
    hump or a digit (`"sessionCookie"`, `"HTTPCookie"`, `"http2cookie"`,
    `"basicAuthorization"`): in an entry only, where a quoted key is no
    code. Not the header's own name in camel case (`"setCookie"`,
    `"ProxyAuthorization"`)."""
    humps = sorted({"".join(word.capitalize() for word in words) for words in names})
    own = _camel_own_spellings(names)
    last = sorted({words[-1] for words in names})
    return (
        r"(?<=[\"'])"
        + (f"(?!(?:{'|'.join(own)}){_QUOTE})" if own else "")
        + r"[A-Za-z0-9_.-]{0,63}(?:[A-Za-z0-9](?-i:"
        + "|".join(sorted(humps, key=lambda hump: len(hump), reverse=True))
        + ")|[0-9](?:"
        + "|".join(last)
        + f"))(?={_QUOTE})"
    )


_COOKIE_CAMEL = _camel_prefixed(_COOKIE_NAMES)
_AUTHORIZATION_CAMEL = _camel_prefixed(_AUTHORIZATION_NAMES)


def _camel_own(names: tuple[tuple[str, ...], ...]) -> str:
    """A header's own name of two words or more in camel case as a quoted
    key (`"setCookie"`, `"ProxyAuthorization"`), read as the header's own
    name is (`"Set-Cookie"`): in an entry only, where a quoted key is no
    code. Case is kept: `"setcookie"` is one word."""
    own = _camel_own_spellings(names)
    return rf"(?<=[\"'])(?-i:{'|'.join(own)})(?={_QUOTE})"


_COOKIE_OWN_CAMEL = _camel_own(_COOKIE_NAMES)
_AUTHORIZATION_OWN_CAMEL = _camel_own(_AUTHORIZATION_NAMES)
# The Authorization header's name at the end of an upper-case env name
# (HTTP_AUTHORIZATION, REDIRECT_HTTP_AUTHORIZATION, X_PROXY_AUTHORIZATION):
# the header's words and the character before them in upper case, where
# _AUTHORIZATION_PREFIXED reads a prefix.
_AUTHORIZATION_UPPER = rf"(?-i:(?<=[A-Z0-9][_.-]){_AUTHORIZATION_NAME.upper()})"
# How an unquoted value after an Authorization name is read, by the name:
# - _AUTHORIZATION_SPOKEN, the header's own name with no prefix
#   (Authorization, Proxy-Authorization, proxy_authorization, a quoted
#   `"proxyAuthorization"`) and an upper-case env name that ends in it:
#   anything but one prose word is a secret's, and several words are masked
#   from the second word on (_SPOKEN_FIRST, below).
# - _AUTHORIZATION_CODE_NAME, a prefixed name in lower case joined by `_` or
#   `.` (has_authorization, self.basic_authorization,
#   copy_source_authorization), an identifier, mostly code's in text: the
#   code checks a lower-case pass name's and a prefixed cookie name's value
#   get (_ENV_VALUE, _TEXT_VALUE: `has_authorization=<opaque>` is masked,
#   `source_authorization = kwargs.pop(...)` and
#   `copy_source_authorization: Optional[str] = None` stay).
# - any other prefixed name (x-proxy-authorization, X-Authorization,
#   note_Authorization), which no identifier spells: anything but one prose
#   word is a secret's, as before.
_AUTHORIZATION_OWN = rf"(?!{_AUTHORIZATION_PREFIXED}){_AUTHORIZATION_NAME}"
_AUTHORIZATION_LOWER = rf"(?<!-)(?-i:{_AUTHORIZATION_NAME})"
_AUTHORIZATION_SPOKEN = (
    rf"(?:{_AUTHORIZATION_OWN}|{_AUTHORIZATION_OWN_CAMEL}"
    rf"|{_AUTHORIZATION_PREFIXED}{_AUTHORIZATION_UPPER})"
)
_AUTHORIZATION_CODE_NAME = rf"{_AUTHORIZATION_PREFIXED}{_AUTHORIZATION_LOWER}"
# Under such a name the classifier decides the value (entry_kind with the
# whole name, _authorization_value; the lead's ruling after the delta review of
# PR #192): a setting's as code (_CODE_VALUE: self.flag, f(x), Optional[str],
# True, null, a number of at most five digits, a builtin type's name), as the
# name's own words (copy_source_authorization=copy_source_authorization), or as
# prose (`flag`: the short all-letters word stays); anything else, masked: an
# identifier-shaped opaque run, a key, `b'...'`, a quoted string, six digits.
# Such a value is masked whole, as code spells it: a call, a subscript or a
# brace with what it holds (`f("<key>")`, `[1, 2]`, `{}`), a string after its
# prefix or after `=` or `:` (`b'<v>'`, `f"<v>"`, `lambda: '<v>'`), an escape
# but a quote's (a JSON copy's `\"` ends the value, as `"` does, unless it
# opens a string after code), and a lambda to its line's end. It starts at no
# quote (a quoted value is the entry branch's), space or earlier mask (`*`),
# and ends, as _SECVAL does, at a space, `,`, `&`, a backtick (code quoted in
# prose) or a quote that ends the
# text around it, and at a line break (`\n` in a JSON copy). Possessive and
# bounded: each character is read a bounded number of times.
_CODE_QUOTED = (
    r"(?:\"[^\"\\\r\n]{0,4096}+\"|'[^'\\\r\n]{0,4096}+'"
    r"|\\{1,7}\"(?:[^\"\\\r\n]|\\{1,7}[^\"\\\r\n]){0,4096}+\\{1,7}\""
    r"|\\{1,7}'(?:[^'\\\r\n]|\\{1,7}[^'\\\r\n]){0,4096}+\\{1,7}')"
)
# A quoted string first, so a JSON copy's `\"` opens one (`f(\")\", 1)`).
_CODE_INSIDE = rf"(?:{_CODE_QUOTED}|[^()\[\]{{}}\"'\\\r\n]|\\[^nrt\r\n])"
_CODE_GROUP = (
    rf"(?:\({_CODE_INSIDE}{{0,1024}}+\)|\[{_CODE_INSIDE}{{0,1024}}+\]"
    rf"|\{{{_CODE_INSIDE}{{0,1024}}+\}})"
)
_CODE_RUN = (
    rf"(?:lambda\b(?:[^\r\n\"'\\]|\\[^nrt\r\n]|{_CODE_QUOTED})*+"
    rf"|(?=[^\s\"'\\*,)\]}}&`])(?:[^\s\"'\\*,()\[\]{{}}&`]++"
    rf"|(?<=[rRbBuUfF=:\t ]){_CODE_QUOTED}(?=[\s,;)\]}}\"'`]|\\[nrt\"]|\Z)"
    rf"|\\[^nrt\s\"']|{_CODE_GROUP}|[(\[{{]"
    r"|(?<=:)[ \t]++(?=\S))++)"
)


# After a code Authorization name (A.1: has_authorization,
# copy_source_authorization) the value is always read, to its end, and the
# classifier decides with the whole name (_authorization_value): it stays only
# where entry_kind calls it a setting's (code, the name's own words, prose) and
# is masked everywhere else. Fail closed: a shape nobody listed is masked, as
# the rc masks it (review R192C: code-escq, H-1, H-2; the lead's ruling,
# 19:5xZ). The value is the first of:
# - a code run (_CODE_RUN) that ends where the value does (a space, `,`, `;`, a
#   quote, an escaped line break, the text's end, or a closing bracket that one
#   of these follows: `f(has_authorization=True)`, not `0]0`);
# - a quoted string (_code_quoted_value): opened by any quote, escaped at any
#   depth (no bound), read to its close at the depth it opened with (the opening
#   backslash run, captured), an escaped line break, a bare quote (a JSON
#   string's end) or the line's end, however long, and anything glued after
#   its close (`""0`, `"abc"def`). Left to the entry rule: a string that is one
#   prose word and its close at the value's end (`"required"`), one that opens
#   with a scheme's word and a space (the scheme carriers mask its token), an
#   empty one (`""`, `"""`) or a lone quote at the line's end, and an
#   already masked one;
# - a value opened by another kind of quote (_CODE_OTHER_QUOTE), to the line's
#   end;
# - anything else, to where a value ends (_CODE_END; _CODE_ANY): `!@#$`,
#   `abc*def`, `).?]`, a code run with more glued after it; never a quote,
#   escaped or not, which is the string's.
# Backslashes are read in pairs (an odd one escapes the next character; before
# n or r it ends the line; an escaped tab is the value's); each run is compared with the opening run only
# to the first character that differs. Possessive throughout; linear.
def _code_quoted_value(n: str) -> str:
    """The value that opens with a quote, its groups named with *n* (the
    pattern holds it twice)."""
    run, quote = f"code_bs{n}", f"code_q{n}"
    return (
        rf"(?P<code_open{n}>(?P<{run}>\\*+)(?P<{quote}>[\"']))"
        # An already masked string, the mask alone (`"***REDACTED***"`); a key
        # after a mask is read (`"***REDACTED*** <key>&quot;`).
        rf"(?!\*\*\*REDACTED\*\*\*(?:\\*[\"']|&(?:quot|#0*34|#[xX]0*22);|\Z))"
        rf"(?:[^\"'\\\r\n]|(?!(?P={quote})|\"{_STRING_END_AHEAD})[\"']"
        rf"|(?!(?P={run})(?P={quote}))(?:(?:\\\\)++(?!\\)|(?:\\\\)*+\\(?![nr])[\"']?))*+"
        rf"(?P<code_close{n}>(?P={run})(?P={quote}))?+"
        rf"(?P<code_tail{n}>(?:(?!{_CODE_END}){_CODE_ANY_CHAR})*+)"
    )


def _any_quoted_value() -> str:
    """_CODE_RUN's companion for a value that opens with a quote under a name
    _AUTHORIZATION_ANY_NAME reads: read to its closing quote at the
    opening depth; _authorization_any_value decides it whole."""
    return (
        rf"(?!\\{{0,31}}[\"'](?:{_PROSE}|(?:{_AUTH_SCHEMES}|Digest)\b|\*\*\*REDACTED\*\*\*))"
        rf"(?=\\{{0,31}}[\"'][^\"'\r\n]{{0,4096}}?[A-Za-z0-9])"
        rf"(?P<any_bs>\\{{0,31}})(?P<any_q>[\"'])"
        rf"(?:[^\"'\\\r\n]|(?!(?P=any_q))[\"']"
        rf"|(?!(?P=any_bs)(?P=any_q))(?:(?:\\\\)++(?!\\)|(?:\\\\)*+\\(?![nrt])[\"']?))++"
        rf"(?:(?P=any_bs)(?P=any_q))?+"
    )


# Where a value ends after a code run or inside _CODE_ANY: a space; `,` or
# `;` before a space, a bare quote, an escaped quote or line break, or the end;
# a run of closing brackets before one of these or `:` and a space (`f(x=1))`,
# `if f(x=y):`, not `bG].P3` or `mZ})\\c`); a bare
# quote, or one escaped once, before a space, `,`, `:`, `}` or the end (a JSON
# string's or key's end: `"has_authorization":true,"x":1`, `true\\"` in a
# copy); an escaped line break; the end.
# A quote or a comma glued into the value (`x"y`, `2,Txu]`) and an escaped
# quote are the value's.
# A JSON key ahead (`,"x":`, `,\"x\":` in a copy): after a comma, the next
# entry of an object. Bounded; linear.
_JSON_KEY_AHEAD = (
    r"(?:\"[^\"\r\n]{0,256}\"|\\\"(?:[^\"\\\r\n]|\\[^\"\r\n]){0,256}\\\")[ \t]*:"
)
# A bare quote (or one escaped once, in a copy) that ends a JSON string: before
# a space, the end, an escaped line break, closing brackets before a space, a
# separator or the end (`"x")`, `"x"]}`), or a comma before a space, the end or
# the next key (`"has_authorization: true",` in a list).
_STRING_END_AHEAD = (
    r"(?=\s|\Z|\\[nr]|[})\]]++(?=[\s,;.)\]}\"]|\Z)|,(?=\s|\Z|" + _JSON_KEY_AHEAD + r"))"
)
# An escaped line break (`\n`, `\r`) ends a value only inside a JSON string,
# where a bare quote that ends the string follows on the line within 256
# characters. In raw text a backslash and an n are the value's, so
# `qn\nH;3...` is read and judged whole (review R192C, 21:2xZ; fail closed).
# Bounded and possessive; linear.
_ESC_LINE_END = r"\\[nr](?=(?:[^\"\\\r\n]|\\[^\r\n]){0,256}+\"(?=[\s,:}\]]|\Z))"
# What may follow a separator, a closing bracket or a quote where the value
# ends: a space, the end, an escaped line break in a JSON string, the next
# JSON key, or a bare quote that ends a JSON string (`true,"}` in a record's
# text).
_END_FOLLOWS = (
    r"\s|\Z|"
    + _ESC_LINE_END
    + "|"
    + _JSON_KEY_AHEAD
    + r"|\""
    + _STRING_END_AHEAD
    + r"|\\\"(?=\")"
)
_CODE_END_AFTER_CLOSE = (
    rf"\s|,(?={_END_FOLLOWS})|[;:](?={_END_FOLLOWS})|'(?={_STRING_END_AHEAD[3:]}"
    r"|\"" + _STRING_END_AHEAD + r"|(?<!\\)\\\"(?=\")|" + _ESC_LINE_END + r"|\Z"
)
# A value ends too at a closing single quote (`'has_authorization = f(x)'`) or
# at backticks (code quoted in prose: `` `has_authorization: 1`, ``), each
# before what may follow a string's end; whether a string was open there is
# _glued_after's to judge.
_CODE_END = (
    rf"\s|,(?={_END_FOLLOWS})|;(?={_END_FOLLOWS})"
    rf"|'(?={_STRING_END_AHEAD[3:]}|`++(?=[\s,.;:)\]}}\"'`]|\Z)"
    rf"|[)\]}}]++(?:{_CODE_END_AFTER_CLOSE})"
    r"|\""
    + _STRING_END_AHEAD
    + r"|(?<!\\)\\\"(?=\"|"
    + _STRING_END_AHEAD[3:]
    + "|"
    + _ESC_LINE_END
    + r"|\Z"
)
# Inside a value no code reads (_CODE_ANY, a string's glued tail) a bracket,
# a single quote and a quote at the value's end are the value's too: only a
# space, a separator, a JSON string's end or the end close it.
_CODE_ANY_END = (
    r"\s|,(?=\s|\Z|" + _JSON_KEY_AHEAD + r")|;(?=\s|\Z)"
    r"|\"(?=\Z|[})\]](?=[\s,\])}]|\Z)|,(?=\s|\Z|"
    + _JSON_KEY_AHEAD
    + r"))|(?<!\\)\\\"(?=\")|\Z"
)
_CODE_ANY_CHAR = r"(?:[^\\\r\n]|\\[^\r\n]|\\(?=\s|\Z))"
_CODE_ANY = (
    rf"(?!\\*+[\"'])(?=[^\s]){_CODE_ANY_CHAR}(?:(?!{_CODE_ANY_END}){_CODE_ANY_CHAR})*+"
)
# A value opened by a quote the string reader does not read as one (a JSON
# copy's ", an HTML entity, a backtick, a curly or full-width quote): to
# the line's end or a JSON string's end, fail closed.
_CODE_OTHER_QUOTE = (
    r"(?:\\+u0022|&(?:quot|#0*34|#[xX]0*22|apos|#0*39|#[xX]0*27);|[`\u201c\u201d\u2018\u2019\uff02\uff07])"
    r"(?:(?!\"(?=[\s,:}]|\\[nr]|\Z)|\\[nr])[^\r\n])*+"
)


def _code_more(n: str) -> str:
    """Up to eight words after a code name's value (after a space, or a `,` or
    `;` and a space), on its line, before a comment (`#`, `//`), a bare quote
    or a word that holds the header's word (the next name, which the carrier
    reads itself). A word may hold escapes (`\\"<key>\\"` in a JSON copy)."""
    return (
        rf"(?P<code_more{n}>(?:[,;]?+[ \t]++(?!#|//)(?![^\s]*?authorization)"
        r"(?:[^\s\"'\\]|\\[^\s]){1,4096}+(?=[\s\"']|\Z)){1,8}+)?+"
    )


def _code_glue(n: str) -> str:
    """What is glued to a code name's value where it ends at a backtick or a
    quote: that mark, a quote right after a backtick, and the run after them
    to a space or the next quote (review R192C, B2 and B5:
    `` has_authorization: x`)<key> ``). _authorization_value masks it with the
    value where the mark is the value's (_glued_after) and gives it back where
    it closes a string or a code span opened before the name. It never reads
    past a quote, so a quoted name after it is the carrier's again."""
    return (
        rf"(?P<code_glue{n}>(?:`|\\{{0,8}}[\"'])(?:\\{{0,8}}[\"'])?"
        r"(?:[^\s\"'\\]|\\[^\s\"'\\]){0,4096}+)?+"
    )


def _code_value(n: str) -> str:
    """After a code Authorization name: a code run to the value's end, a quoted
    string, or anything else to the next space (see above)."""
    # Case-sensitive: `\\N` and `\\R` are no line breaks, and prose is the
    # classifier's (ASCII letters); a scheme's word reads in any case.
    return (
        rf"(?-i:{_CODE_OTHER_QUOTE}|{_CODE_RUN}(?={_CODE_END})"
        rf"|{_code_quoted_value(n)}|{_CODE_ANY})"
    )


_AUTHORIZATION_ONE_WORD = (
    rf"(?:{_AUTHORIZATION_SPOKEN}|{_AUTHORIZATION_PREFIXED}"
    rf"(?!{_AUTHORIZATION_UPPER}|{_AUTHORIZATION_LOWER}){_AUTHORIZATION_NAME})"
)
# Every word after such a name and a plain `:` or `=` is masked, the first
# word too, prose or not (`Authorization: required`, `Authorization: please
# sign in first`, `Authorization: YmkkyClHmFD`): a bare value under an
# Authorization name is a credential, as 0.9.3 masks it (the
# lead, 2026-11-16 08:46Z, replacing its one kept prose word). A scheme
# word alone (`Authorization: Bearer`) stays, as 0.9.3 keeps it. A
# deliberate over-mask of prose after an Authorization name, the one
# expected class. A word runs to a space, a quote, a
# backslash (an escaped quote or line end), a backtick or `*` (an earlier
# mask), and the words to the line's end or the value's closing quote, on
# the name's own line (a value on the next indented line is read by its
# first word, as before: a line of code there is no value's second word).
# After a URL-encoded `%3A` or `%3D` only `%20` or `+` joins words, and a
# literal space ends the value (`set the Authorization%3A header to ...` is
# prose): there the one-word rule reads `%20`-joined words as one value.
# A value that opens with a scheme (`Bearer <token>`, `Digest ...`) is the
# scheme carriers'.
# No length cap, so a long line cannot end the mask between two words:
# every run is possessive and every character is read once.
# No word is a second Authorization-family name with its `:` or `=`
# (`Authorization: none, Proxy-Authorization: "<v>"`, `... has_authorization=
# "<v>"`, `... HTTP_AUTHORIZATION="<v>"`): the words end before it and the
# same pass reads it as a name of its own, with its value (review R192, M2:
# the run masked the second name and left its quoted value). Nor does a word
# read past one inside it: after a punctuation mark (`retry;HTTP_AUTHORIZATION=`,
# `(Proxy-Authorization:`) the same check is made again, and after a `_`, `.`
# or `-` a word stops where such a name's header word starts, as the carrier
# reads a prefixed name of any length (`X_<70 letters>_AUTHORIZATION=`;
# review R192B, M-A). Each check is made once per character, so the words
# stay linear.
# A shell, template or batch reference's close after a name
# (`${HTTP_AUTHORIZATION}`, `{{HTTP_AUTHORIZATION}}`,
# `${HTTP_AUTHORIZATION:-x}`, `$(Authorization)`, `%AUTHORIZATION%`), and the
# spaces before a separator, a JSON copy's escaped tab among them (
# round 2).
_SHELL_DEFAULT = r"(?:(?::[-=+?]|-)[^{}()\s\"'\\]{0,64}+)?"
_SHELL_CLOSE = rf"{_SHELL_DEFAULT}(?:[}})]{{1,2}}|%(?![0-9A-Fa-f]{{2}}))"
_TAB_GAP = r"(?:[ \t]|\\t)*+"
# The spaces after a separator, a JSON copy's escaped tab among them at any
# escape depth (`${HTTP_AUTHORIZATION}:\t<v>`), as _GAP and
# _ENTRY_VALUE read them after every other name.
_SEP_GAP = r"(?:[ \t]|(?-i:\\{1,8}t))*+"
# What may stand between a name and its separator when the quotes around the
# name are doubled, left open, mixed, encoded or followed by a few more name
# characters (`""proxyAuthorization"":`, `"proxyAuthorization"x=`,
# `"proxyAuthorization":`, `&quot;proxyAuthorization&quot;:`,
# `` `proxyAuthorization`: ``; review R192C, its p_ma5 sweep). Bounded and
# possessive.
_QUOTE_ENTITY = r"&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);"
_QUOTE_JUNK = rf"(?:\\{{0,8}}[\"'`]|{_QUOTE_ENTITY}|\\{{1,8}}u0022|%22){{0,3}}+"
_NAME_JUNK = rf"{_QUOTE_JUNK}(?:[A-Za-z0-9_.-]{{1,16}}+{_QUOTE_JUNK})?+"
_AUTHORIZATION_HERE = (
    rf"(?:proxy[_-]?)?authorization[A-Za-z0-9_.-]{{0,64}}?(?:{_SHELL_CLOSE})?"
    rf"(?:{_ESCQ}{_TAB_GAP}(?:[:=]|%(?:25)?3[AaDd])"
    rf"|{_QUOTE_JUNK}{_TAB_GAP}(?:[:=]|%(?:25)?3[AaDd])"
    rf"(?=[ \t]*+(?:[\"'`\\]|\Z)))"
)
_AUTHORIZATION_AHEAD = rf"(?!{_ESCQ}[A-Za-z0-9_.-]{{0,64}}?{_AUTHORIZATION_HERE})"
# After the one-word branch's separator the same check stops the value only
# before a second name a carrier reads as its own (review R192C, B-1): the
# header's own name (Proxy-Authorization, the camel proxyAuthorization only
# quoted) or its word after a letter or digit and a joiner
# (_AUTHORIZATION_PREFIXED: HTTP_AUTHORIZATION, has_authorization), or after
# any run that ends in a joiner, which other carriers read on their own
# (`_authorization`, `-authorization`, a long run of joiners); a quoted camel
# name only where its quote closes (`"noneAuthorization":`, read by the entry
# rule; review R192C, B-2 and B-3), a joiner in its prefix too
# (`"none_proxyAuthorization":`; B-4). A word glued to the header's word with no
# joiner (`noneAuthorization=<v>`, `proxyAuthorization=<v>` unquoted or with
# its quote left open) is no name any carrier reads; stopping there left its
# value, so the word is the value, masked whole, as the rc does.
_AUTHORIZATION_READ_HERE = (
    rf"(?:(?<=[\"'])(?:proxy[_-]?)?authorization\\{{0,8}}[\"']"
    rf"|(?:proxy[_-])?authorization{_ESCQ})"
    r"[ \t]*+(?:[:=]|%(?:25)?3[AaDd])"
)
_AUTHORIZATION_AHEAD_ONE = (
    rf"(?!{_ESCQ}(?:[A-Za-z0-9_.-]{{0,63}}?[_.-])?{_AUTHORIZATION_READ_HERE}"
    rf"|{_ESCQ}[A-Za-z0-9_.-]{{1,64}}?(?:proxy[_-]?)?authorization\\{{0,8}}[\"']"
    r"[ \t]*+(?:[:=]|%(?:25)?3[AaDd]))"
)
# The one-word branch's value (review R192C, L-2; round 2): its word,
# as _SECVAL reads it, and, where that word holds the header's word, the value
# after it too. The word takes a second Authorization name when the lookahead
# above cannot tell a carrier reads it: after a punctuation mark
# (`none;(Authorization: "<v>"`), an encoded space (`none%20Authorization:`),
# `=`, `->` or `=>`, past its 64-character prefix, or with more after the
# header's word (`noneAuthorizationHeader`). Nothing would read the value then,
# so the branch fails closed: a word that holds the header's word and ends in a
# separator (`:`, `=`, `%3A`, `%3D`, then perhaps `%20` or `+`), or has one
# after spaces or an escaped quote, is masked with the value that follows (a
# quoted string, a JSON copy's escaped one, or a run), on its own line or the
# next one, as a carrier reads a value there (_GAP:
# `authorization%3A%20&quot;proxyAuthorization=\n  <v>`; review R192C's
# cross-line sweep, 2026-11-16); so is a word and a value
# after a `}` (a shell default: `${HTTP_AUTHORIZATION:-x}=<v>`). A word with no
# separator after it is masked alone, as before. Each check is one scan of the
# word, a fixed-width lookbehind or a bounded run; possessive, linear.
_AUTHORIZATION_TOOK_SEPARATOR = (
    r"(?:(?<=[:=])|(?<=%3[AaDd])|(?<=%253[AaDd])|(?<=%3[AaDd]%20)|(?<=%3[AaDd]\+))"
)
# A quote of any spelling, its backslash perhaps escaped (`"`, `\&#34;`,
# `&quot;`, `\u0022`, `%22`): a second value may open with one and close with
# another (`noneAuthorization="<v>&quot;`), as a name's quotes are read (_ESCQ).
_MIXED_QUOTE = (
    r"(?:\\{0,16}(?:[\"']|&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);)"
    r"|\\{1,8}(?:u00(?:22|27)|[xX](?:22|27))|%(?:25)?2[27])"
)
_SECOND_VALUE = (
    r"(?:\"[^\"\r\n]{0,4096}+\"|'[^'\r\n]{0,4096}+'"
    r"|\\{1,3}\"(?:[^\"\\\r\n]|\\[^\"\r\n]){0,4096}+\\{1,3}\""
    rf"|{_QUOTE_ENTITY}(?:(?!{_QUOTE_ENTITY})[^\r\n]){{0,4096}}+(?:{_QUOTE_ENTITY})?+"
    r"|\\{1,8}u0022(?:(?!\\{1,8}u0022)[^\r\n]){0,4096}+(?:\\{1,8}u0022)?+"
    r"|`[^`\r\n]{0,4096}+`?+"
    rf"|{_MIXED_QUOTE}(?:(?!{_MIXED_QUOTE})[^\r\n]){{1,4096}}+(?:{_MIXED_QUOTE})?+"
    rf"|{_VALUE_START}(?:{_BRACED_PCT}|[^\s\"'\\,}}*&]|{_VALUE_ESCAPE}){{1,4096}}+)"
)
_BRACE_TAIL = rf"(?:\}}{_ESCQ}{_TAB_GAP}(?:[:=]|%(?:25)?3[AaDd])(?:%20|\+)?{_SEP_GAP}{_SECOND_VALUE})?+"
# A JSON copy's escaped tab, then a second Authorization name with its
# separator (a shell or template reference's braces before it): the one-word
# value ends before it, and the name is read on its own.
_TAB_BEFORE_NAME = (
    rf"(?!(?-i:\\{{1,8}}t)[{{($%]{{0,3}}[A-Za-z0-9_.-]{{0,64}}?{_AUTHORIZATION_HERE})"
)
_ONE_WORD_VALUE = (
    rf"(?:(?=[^\"'\s,}}\\*&]*?authorization){_VALUE_START}"
    rf"(?:{_BRACED_PCT}|[^\"'\s,}}\\*&]|{_VALUE_ESCAPE})++"
    rf"(?:{_AUTHORIZATION_TOOK_SEPARATOR}{_GAP}{_SECOND_VALUE}"
    rf"|{_NAME_JUNK}{_TAB_GAP}(?:[:=]|%(?:25)?3[AaDd])(?:%20|\+)?{_GAP}{_SECOND_VALUE})?+"
    rf"|{_VALUE_START}(?:{_BRACED_PCT}|[^\"'\s,}}\\*&]|{_TAB_BEFORE_NAME}{_VALUE_ESCAPE})++{_BRACE_TAIL})"
)
# A prefixed name's quoted entry (_ENTRY_TEXT) that holds the header's word
# itself (`HTTP_AUTHORIZATION="noneAuthorization": "<v>"`, a 70-letter prefix,
# the quote left open or mixed): the second name's value after the quote and
# a separator, or after the separator the entry ended in, is masked with it
# (review R192C's p_ma5 sweep). One scan of the entry; linear.
_ENTRY_VALUE = (
    rf"(?:(?=[^\"'\\\r\n*]{{0,4096}}?authorization){_ENTRY_TEXT}"
    rf"(?:{_NAME_JUNK}{_TAB_GAP}(?:[:=]|%(?:25)?3[AaDd])(?:%20|\+)?(?:[ \t]|\\{{1,8}}t)*+{_SECOND_VALUE}"
    rf"|(?:(?<=[:=])|(?<=[:=][ \t])|(?<=[:=][ \t]{{2}})){_SECOND_VALUE})?+"
    rf"|{_ENTRY_TEXT})"
)
_SPOKEN_WORD = (
    _AUTHORIZATION_AHEAD
    + r"(?:[A-Za-z0-9]++"
    + rf"|[_.-](?!{_AUTHORIZATION_HERE})"
    + rf"|{_VALUE_START}(?:{_VALUE_ESCAPE})"
    + rf"|[^\s\"'\\*`A-Za-z0-9_.-]{_AUTHORIZATION_AHEAD})++"
)
_SPOKEN_GAP = r"(?:[ \t]|(?-i:\\{1,8}t))++"
_SPOKEN_WORDS = rf"{_SPOKEN_WORD}(?:{_SPOKEN_GAP}{_SPOKEN_WORD})*+"
# Two words or more (a single word is the one-word branch's, which reads a
# quote or an escape inside it); the first is masked with the rest.
_SPOKEN_FIRST = rf"(?={_SPOKEN_WORD}{_SPOKEN_GAP}{_SPOKEN_WORD})"
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*+"
_CODE_START = r"[-+=:<>!~^|.#$?`(\[{]"
# A character of a value that no identifier, attribute access, call,
# subscript or statement holds (`!` in hunter2!, `-` in s3cr3t-pw).
_RUN_CHAR = (
    r"(?:[^A-Za-z0-9_.(\[)\];:\"'\s,}\\*&`]|\\++(?:[^nrt\"'\r\n\\]|(?=[\r\n]|\Z)))"
)
_QUOTED_CODE = rf"(?:{_LOWER_LITERAL}|[$\[{{(])"
_ENV_VALUE = (
    rf"(?!{_LOWER_LITERAL}|[rRuUbBfF]{{1,3}}{_QUOTE})"
    rf"(?:(?={_IDENT}(?:[\"'&`]|[ \t]*+(?:#|\\|[\r\n]|\Z)|{_RUN_CHAR}))"
    rf"|(?![A-Za-z_]|{_CODE_START}))"
)
_TEXT_VALUE = (
    rf"(?!{_LOWER_LITERAL})(?:(?={_IDENT}{_RUN_CHAR})|(?![A-Za-z_]|{_CODE_START}))"
)
_LOWER_ENTRY = rf"{_QUOTE}{_SEP}{_GAP}{_QUOTE}{_SCHEME}"
_LOWER_ENV = (
    rf"{_ESCQ}(?:=|%(?:25)?3[Dd])"
    rf"(?:{_QUOTE}{_SCHEME}(?!{_QUOTED_CODE})|{_SCHEME}{_ENV_VALUE})"
)
_LOWER_TEXT = (
    rf"{_ESCQ}{_SEP}{_WS}*+"
    rf"(?:{_QUOTE}{_SCHEME}(?!{_QUOTED_CODE})|{_SCHEME}{_TEXT_VALUE})"
)
# A first look, so the grammar above runs only on an identifier that ends
# as a pass name does (`ass`, `wd`, `alue`, or `_` and one to four digits),
# read to its end at once: every name above is an identifier of at most 75
# characters with no identifier character after it, so it passes; most
# identifiers do not, and
# the grammar's backtracking at every word start was half the carrier's time.
_LOWER_PASS_ENDING = (
    r"(?=[A-Za-z0-9_]{1,80}+(?:(?<=ass)|(?<=wd)|(?<=alue)"
    + "".join(f"|(?<=_[0-9]{{{n}}})" for n in range(1, 5))
    + "))"
)
_LOWER_PASS = (
    r"(?<!_)(?<!\\)(?=[A-Za-z])(?<![(,{\[.])(?<![(,{\[.][ \t])"
    rf"{_NOT_ONE_WORD_NAME}{_LOWER_PASS_ENDING}{_LOWER_TEST_LIST_NAME}{_LOWER_RESULT}"
    rf"(?:{_LOWER_PASS_NAME}(?:{_PASS_NUMBER})?"
    rf"|{_LOWER_PASS_VALUE_NAME}|{_LOWER_PASS_ALONE})"
    rf"(?:{_LOWER_ENTRY}|{_LOWER_ENV}|{_LOWER_TEXT})"
)

# The four letters outside ASCII that `re` matches to an ASCII letter when
# case is ignored (U+0130 and U+0131 to `i`, U+017F to `s`, U+212A, the
# Kelvin sign, to `k`): a text that holds one is not put in lower case to
# look for a carrier's words, and every carrier runs on it.
_FOLDED_ONTO_ASCII = ("\u0130", "\u0131", "\u017f", "\u212a")


def _folded(text: str) -> str | None:
    """*text* in lower case, to look for a carrier's words where the carrier
    ignores case, or None when lower case would not show what `re` matches."""
    if not text.isascii() and any(letter in text for letter in _FOLDED_ONTO_ASCII):
        return None
    return text.lower()


class _LazyPattern:
    """A redaction pattern, compiled when it is first used.

    The left boundary (_KEY_START) is long, and the patterns that hold it
    took two seconds to compile where this module was imported. ``pattern``
    and ``flags`` are the source, as on a compiled pattern; every other
    attribute is the compiled pattern's.

    ``words`` is for a carrier that tries _KEY_START at every position of a
    text: the words one of which a text must hold to match it, in lower
    case when the carrier ignores case. redact_trajectory_text_with_categories
    skips the carrier for a text that holds none, and a word of _WORD_TAILS
    counts only as its tail follows it.
    """

    __slots__ = ("_compiled", "flags", "pattern", "words")

    def __init__(
        self, pattern: str, flags: int = 0, *, words: tuple[str, ...] = ()
    ) -> None:
        self.pattern = pattern
        self.flags = flags
        self.words = words
        self._compiled: re.Pattern[str] | None = None

    def compiled(self) -> re.Pattern[str]:
        compiled = self._compiled
        if compiled is None:
            compiled = self._compiled = re.compile(self.pattern, self.flags)
        return compiled

    def subn(
        self, repl: str | Callable[[re.Match[str]], str], string: str
    ) -> tuple[str, int]:
        return self.compiled().subn(repl, string)

    def sub(self, repl: str | Callable[[re.Match[str]], str], string: str) -> str:
        return self.compiled().sub(repl, string)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.compiled(), name)

    def may_match(self, text: str, folded: str | None) -> bool:
        """Whether *text* holds one of ``words`` (*folded* is _folded(text)),
        or the carrier has none, or lower case cannot say."""
        if not self.words:
            return True
        if not self.flags & re.IGNORECASE:
            return any(word in text for word in self.words)
        return folded is None or any(
            word in folded
            and (
                word not in _WORD_TAILS or _WORD_TAILS[word].search(folded) is not None
            )
            for word in self.words
        )


class _GatedPattern(_LazyPattern):
    """A _LazyPattern every match of which holds one of the literal characters
    ``chars``: _redaction_pass skips it for a text that holds none, as it skips
    a carrier for a text with none of its words. Not a carrier (``words`` stays
    empty). The pattern tries _KEY_START at every position, and on a text with
    no such character it can match nothing, so the skip changes no output."""

    __slots__ = ("chars",)

    def __init__(self, pattern: str, flags: int = 0, *, chars: str) -> None:
        super().__init__(pattern, flags)
        self.chars = chars

    def may_match(self, text: str, folded: str | None) -> bool:
        return any(char in text for char in self.chars)


def _prefix_keeper(
    template: str | Callable[[re.Match[str]], str],
) -> str | Callable[[re.Match[str]], str]:
    r"""*template* as a function when it is `\1` and then plain text, as
    every carrier's is. `re` looks a template with a group in it up by the
    pattern on every call, and hashes the pattern's whole program to do so:
    0.6 ms a call for the longest carrier, whatever the text's length. The
    function writes what the template does (a group that did not take part
    is the empty string). A function is the carrier's own and is kept."""
    if callable(template) or not template.startswith("\\1") or "\\" in template[2:]:
        return template
    kept = template[2:]
    return lambda match: (match[1] or "") + kept


# The classifier's answers within one run of a carrier over a text
# (_subn_changed), by name and by name and value: dense rows asked it the
# same questions once per match (review R192, L3). Kept for that run alone,
# so no value outlives the text it was read from; a value longer than
# _ANSWER_VALUE_MAX is asked each time.
_classifier_answers: ContextVar[dict[Any, Any] | None] = ContextVar(
    "_classifier_answers", default=None
)
_ANSWER_VALUE_MAX = 256
_UNASKED = object()


# A name with a token after ``per`` (the price carrier's): words of a name
# before it, ``per``, at most two words, then TOKEN at the name's end.
# Bounded, so a long run is read a bounded number of times.
_PRICE_NAME = (
    r"_{0,2}(?:[A-Za-z0-9]{1,32}[_.-]){0,8}per[_-](?:[A-Za-z0-9]{1,24}[_-]){0,2}token"
    r"(?![A-Za-z0-9_])"
)
# A currency's sign or word after a number (`0.5 $`, `15 usd`): a value
# with its unit, so no number. A word after it (`3e-06 end`) is no unit.
# A flat table of prices after the name (`{"input": 3e-06, "output": ...}`,
# `[1e-06, 2e-06]`), as written or JSON-escaped, read whole so its leaves are
# decided here: a number stays, anything else is masked inside its quotes.
_PRICE_TABLE = r"\{[^{}]{0,4096}+\}|\[[^\[\]]{0,4096}+\]"
_PRICE_LEAF_BODY = r"(?P<leaf>\\{0,7}[\"'][^\"'\\]*+\\{0,7}[\"']|[^\s,:{}\[\]\"'\\]++)"
# A mapping's values follow `:`; a list's items follow `[` or `,`.
_PRICE_DICT_LEAF = re.compile(r"(?P<lead>:[ \t]*+)" + _PRICE_LEAF_BODY)
_PRICE_LIST_LEAF = re.compile(r"(?P<lead>[\[,][ \t]*+)" + _PRICE_LEAF_BODY)
_PRICE_LITERALS = frozenset({"null", "true", "false", "None", "True", "False"})


def _price_table(table: str) -> str:
    """*table* with each leaf that is no number masked (a mixed table: the
    lead's ruling, 2026-11-16 09:58Z, and F2b's 10:35Z note)."""

    def leaf(match: re.Match[str]) -> str:
        text = match["leaf"]
        value = text.strip("\\\"'")
        if _PRICE_NUMBER.fullmatch(value) or value in _PRICE_LITERALS:
            return match[0]
        quoted = re.match(r"(\\{0,7}[\"'])(.*?)(\\{0,7}[\"'])$", text, re.S)
        if quoted is None:
            return match["lead"] + "***REDACTED***"
        return match["lead"] + quoted[1] + "***REDACTED***" + quoted[3]

    pattern = _PRICE_DICT_LEAF if table[:1] == "{" else _PRICE_LIST_LEAF
    return pattern.sub(leaf, table)


_PRICE_VALUE_END = re.compile(r"\\{1,8}[nrt]")
_PRICE_UNIT_AFTER = re.compile(
    r"[ \t]*+(?:[$€£¥%]|(?i:usd|eur|gbp|cents?|dollars?)(?![A-Za-z]))"
)


def _price_value(match: re.Match[str]) -> str:
    """The price carrier's replacement: a price's name keeps a number and
    nothing else, the value alone deciding: a number (the classifier's _PRICE_NUMBER) with no more after it
    than a value's end. A number with a unit (`0.5 $`) and anything else is
    masked."""
    # A JSON copy's escaped line break or tab ends the value, as a raw one does.
    head = _PRICE_VALUE_END.split(match["value"], maxsplit=1)[0]
    # A sentence's or a bracket's close after the number is no part of it.
    value = head.strip("\\\"'").rstrip(".,;)]}")
    if value[:1] in ("{", "["):
        raw = match["value"]
        if raw[:1] in ("{", "[") and raw[-1:] in ("}", "]"):
            return match[1] + _price_table(raw)
        return match[0]  # a table that opens here and closes further on
    if _PRICE_NUMBER.fullmatch(value) and not _PRICE_UNIT_AFTER.match(
        match.string, match.start("value") + len(head)
    ):
        return match[0]
    return match[1] + "***REDACTED***"


def _secret_name_value(match: re.Match[str]) -> str:
    """The secret-name carrier's replacement: the name and its separator
    kept and the value masked. A name of its decided branch (_DECIDED_NAME)
    is kept only where the classifier calls the entry a setting's
    (MAX_TOKENS_2=8192, tokens_1h=5); where it calls it a credential's,
    gives no answer, or fails, the value is masked (fail closed). The
    branch keeps no match without asking: its pattern reads names the
    shape check (_decided_by_name) does not, KEYV2_TOKEN and
    TOKENS_2_SECRET among them, and the classifier decides those too
    (review R192B, M-B). An inline document or list
    not closed, nested too deep or longer than _JSON_CAP is masked by
    _open_document."""
    decided = match["decided"] or match["decided_lower"]
    if decided is not None:
        answers = _classifier_answers.get()
        if answers is None:
            answers = {}
        value = match["value"]
        if match["content"] is not None:
            # As the classifier reads a value: a JSON copy doubles each
            # backslash of a Windows path (creds\\\\sa.json).
            value = _BACKSLASH_RUN.sub(r"\\", value)
        kind = answers.get((decided, value), _UNASKED)
        if kind is _UNASKED:
            try:
                kind = entry_kind(decided, value)
            except Exception:
                kind = None
            if len(value) <= _ANSWER_VALUE_MAX:
                answers[decided, value] = kind
        if kind == "setting":
            return match[0]
        value = match["value"]
        if match["content"] is not None and (
            match["open"] is not None
            or match["open_copy"] is not None
            or match["open_list"] is not None
            or match["open_list_copy"] is not None
            or len(value) > _JSON_CAP
        ):
            copy = match["copy"] is not None or match["list_copy"] is not None
            return match[1] + _open_document(value, copy=copy, listed=value[:1] == "[")
    return match[1] + "***REDACTED***"


_NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
_ESCAPE_LETTERS = re.compile(r"[nrtbf]|u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2}")
_COLOUR_TAIL = re.compile(r"[0-9;:]*[A-Za-z]")
_COPY_ESCAPE = re.compile(r'\\(["\\\\])')
# A quoted literal in a code value under an Authorization name (round
# 2; the lead's ruling, 17:5xZ). The classifier reads a call or a subscript
# around a quoted name of at most 32 characters as code (f("abc123"),
# x["abc"]); one whose literal is key-shaped is masked: 24 characters or more,
# or opaque (digits inside a word: a1b2c3d4, FAKE0123abc). A short name-like
# literal stays (words of letters, each perhaps ending in up to four digits,
# joined by `_`, `.` or `-`; a number of at most five digits): the residual.
# So does an identifier of letters of any length (_IDENTIFIER, the lead's
# ruling of 2026-11-14: kwargs.pop("copy_source_authorization", None)), here;
# the code-shape cap (_CODE_TOKEN_CAP) masks a quoted one of more than 23
# characters (its setting 1, which wins). Each literal is one possessive run;
# linear.
_QUOTED_LITERAL = re.compile(r"\\{0,7}([\"'])([^\"'\\\r\n]{1,4096}+)\\{0,7}\1")
_NAME_LIKE_LITERAL = re.compile(
    r"_{0,2}[A-Za-z]++[0-9]{0,4}+(?:[_.-]{1,2}[A-Za-z]++[0-9]{0,4}+)*+_{0,2}|[0-9]{1,5}"
)
_LETTERS_LITERAL = re.compile(_LETTERS_IDENTIFIER)
_KEY_LITERAL_MIN = 24


def _holds_key_literal(value: str) -> bool:
    """Whether *value*, a code value the classifier kept, quotes a key-shaped
    literal (see _QUOTED_LITERAL)."""
    for found in _QUOTED_LITERAL.finditer(value):
        text = found[2]
        if _LETTERS_LITERAL.fullmatch(text):
            continue
        if len(text) < _KEY_LITERAL_MIN and _NAME_LIKE_LITERAL.fullmatch(text):
            continue
        return True
    return False


# The code-shape cap, word for word: "Inside a
# code shape (call arguments, subscripts, dotted parts, defaults), every token
# of 24 or more characters, quoted or not, is masked unless it is name-like. A
# token is name-like when it splits on `_`, `.`, `-` and lower-to-upper camel
# boundaries into at most 8 parts, each part letters only, 2-15 long, optionally
# ending in at most 2 digits, and the token is not all hex and not
# base64-shaped (mixed case plus digits with no separators). Digit runs inside
# brackets follow the at-most-5-digit number rule."
# One switch (_CODE_TOKEN_CAP), the readings review R192C's oracle states
# (F2b-R192C.md, "Points the lead's text leaves open"): a token is a run of
# anything but spaces, quotes and ( ) [ ] { } , : = * < > | & ^ ! ; (a dotted
# chain is one token, split on `.` for the test); empty parts from doubled or
# edge separators don't count; a part's 2-15 counts its digits; base64-shaped
# then adds nothing to the part rule (a part's two trailing digits are a
# name's, so getUserAccountSettingsV2 is one). It applies to every token of a
# code value, its callee and attributes too; a quoted string of more than 23
# characters inside code is masked whatever it holds (the lead's setting 1); a
# run of six digits or more inside brackets, quoted or not, is masked unless it
# is part of a decimal. A value that is the name's own words is exempt. Linear.
_CODE_TOKEN_CAP = True
_CODE_TOKEN = re.compile(r"[^\s\"'()\[\]{},:=*<>|&^!;]{24,}")
_TOKEN_SPLIT = re.compile(r"[_.\-]+|(?<=[a-z])(?=[A-Z])")
_TOKEN_PART = re.compile(r"(?=.{2,15}$)[A-Za-z]+[0-9]{0,2}")
_QUOTED_IN_CODE = re.compile(r"(\\*)([\"'])(.*?)\1\2")


# The lead's tightening, word for word: "Each part must also be all
# lowercase, Titlecase, or an all-caps acronym of at most 5 letters, and every
# part of 4 or more letters must contain a vowel (a, e, i, o, u, y)." A part's
# trailing digits are not letters.
_PART_CASE = re.compile(r"[a-z]+|[A-Z][a-z]+|[A-Z]{1,5}")
_VOWEL = re.compile(r"[aeiouyAEIOUY]")
# The lead's final ruling, word for word: "Apply A, not B. A part of
# 6-15 capitals is allowed when every part is all caps (a constant name). Turn
# your A switch ON." One switch (_CAPS_CONSTANT). Every part is every part of
# the token, so a constant read through a lowercase name
# (obj.CLIENT_LOGGING_SUPPORTED) is not all caps; A leaves the vowel test alone.
_CAPS_CONSTANT = True
# The same ruling, word for word: "Add an exact-word list, matched
# case-insensitively per part, of consonant-only words common in code. They
# pass the vowel test as if they had a vowel. Start with: `http, https, httpx,
# grpc, mtls, html, xhtml, smtp, sftp, ftps, rtmp, rtsp, dhcp, ntlm, pkcs,
# lstm, tmpfs, pgsql, nntp`" and "Extend it from the corpus's most frequent
# consonant-only parts." The words added (review R192C's consonant_parts.tsv:
# site-packages plus the stdlib, in its rank order) follow the lead's. A part
# matches when its letters, lowercased and without its trailing digits, are a
# listed word (http2, HTTP, pkcs12); it still takes the case test.
_VOWEL_FREE_TEXT = (
    "http, https, httpx, grpc, mtls, html, xhtml, smtp, sftp, ftps, rtmp, "
    "rtsp, dhcp, ntlm, pkcs, lstm, tmpfs, pgsql, nntp, "
    # Added from the corpus.
    "llms, tqdm, tmpl, stmt, kwds, sqrt, spdx, ctrl, hdrs, vllm, gdch, crlf, "
    "jwks, zstd, rpcs, fcntl, nbsp, rcvd, cdll, cmds, xmlns, llvm, sdks, "
    "msvcrt, dcgm, xmlrpc, strs, fqdn, msgs, jwts, stmts, lstsq, pkgs, ssrf, "
    "mpfr, ttft, rlhf, tcsh, httpd, wsdl, wgsl, mbcs, hkdf, ttls, dtls, "
    "snmp, dsls, hdfs, tsql, pbkdf, msft, ntdll, jdbc, gbps, srtp, vpcs, "
    "dtmf, mssql, plpgsql, dbrx"
)
_VOWEL_FREE_WORDS = frozenset(_VOWEL_FREE_TEXT.split(", "))


def _part_like(part: str, constant: bool = False) -> bool:
    letters = part.rstrip("0123456789")
    return (
        _TOKEN_PART.fullmatch(part) is not None
        and (
            _PART_CASE.fullmatch(letters) is not None
            or (constant and re.fullmatch(r"[A-Z]{6,15}", letters) is not None)
        )
        and (
            len(letters) < 4
            or _VOWEL.search(letters) is not None
            or letters.lower() in _VOWEL_FREE_WORDS
        )
    )


def _name_like(token: str) -> bool:
    """The lead's name-like token (see _CODE_TOKEN_CAP)."""
    parts = [part for part in _TOKEN_SPLIT.split(token) if part]
    if not 0 < len(parts) <= 8:
        return False
    constant = _CAPS_CONSTANT and all(
        re.fullmatch(r"[A-Z]+", part.rstrip("0123456789")) for part in parts
    )
    if not all(_part_like(part, constant) for part in parts):
        return False
    return re.fullmatch(r"[0-9A-Fa-f]+", re.sub(r"[_.\-]", "", token)) is None


# The 20:05Z clause's "Digit runs inside brackets follow the at-most-5-digit
# number rule", as review R192C's oracle reads it: inside brackets (depth one
# or more, quoted or not; a bracket inside a quote is no bracket) a run of six
# digits or more is masked unless it lies in a decimal standing alone
# (`digits.digits`, no `.` or word character on either side): `x[1.5]`,
# `f(3.14159265358979)` stay; `x[1.2.3456789]`, `f(")", 1234567)` are masked.
_DECIMAL_ALONE = re.compile(r"(?<![A-Za-z0-9_.])[0-9]+\.[0-9]+(?![A-Za-z0-9_.])")
_SIX_DIGITS = re.compile(r"[0-9]{6,}")
_PAIRS = {")": "(", "]": "[", "}": "{"}


def _code_scan(value: str) -> tuple[str, list[int], bool, int]:
    """One pass over a code value, quote-aware as review R192C's oracle reads
    quotes (a quote closes after the backslash run it opened with): the text
    inside brackets (anything at depth zero, and every quote, a space), the
    positions of closers that meet no opener or the wrong one, whether every
    opener and quote is closed with nothing glued after a group that closes
    at depth zero but another group, `.`, a space or `=` (`A[]0[]` is no
    code), and the end of *value* without its trailing run of unmatched
    closers."""
    inside: list[str] = []
    stack: list[str] = []
    unmatched: list[int] = []
    quote, opened_run, run = "", 0, 0
    after_group = False
    glued = False
    for index, char in enumerate(value):
        if after_group and not quote and char not in "([.= \t)]}":
            glued = True
        # An attribute after a group starts with a name (`f(x).y`, not `A().[]`).
        if (
            after_group
            and char == "."
            and not (
                value[index + 1 : index + 2].isascii()
                and (
                    value[index + 1 : index + 2].isalpha()
                    or value[index + 1 : index + 2] == "_"
                )
            )
        ):
            glued = True
        # A backslash outside a string is no code's (`\T`; a JSON copy's
        # escapes are undone before the scan).
        if (
            char == "\\"
            and not quote
            and value[index + 1 : index + 2] not in ('"', "'")
        ):
            glued = True
        after_group = False
        if quote:
            if char == quote and run == opened_run:
                quote = ""
                inside.append(" ")
            else:
                inside.append(char if stack else " ")
        elif char in "\"'":
            quote, opened_run = char, run
            inside.append(" ")
        elif char in "([{":
            stack.append(char)
            inside.append(" ")
        elif char in ")]}":
            if stack and stack[-1] == _PAIRS[char]:
                stack.pop()
                after_group = not stack
            else:
                unmatched.append(index)
            inside.append(" ")
        else:
            inside.append(char if stack else " ")
        run = run + 1 if char == "\\" else 0
    end = len(value)
    trailing = set(unmatched)
    while end and end - 1 in trailing:
        end -= 1
    closed = (
        not stack
        and not quote
        and not glued
        and all(index >= end for index in unmatched)
    )
    return "".join(inside), unmatched, closed, end


def _long_digits_in_brackets(value: str) -> bool:
    """A run of six digits or more inside brackets that is no decimal standing
    alone (see _DECIMAL_ALONE)."""
    inside = _code_scan(value)[0]
    spans = [m.span() for m in _DECIMAL_ALONE.finditer(inside)]
    return any(
        not any(a <= m.start() and m.end() <= b for a, b in spans)
        for m in _SIX_DIGITS.finditer(inside)
    )


def _holds_key_token(value: str) -> bool:
    """Whether *value*, a code value the classifier keeps, holds what the
    code-shape cap masks."""
    if not _CODE_TOKEN_CAP:
        return False
    if any(len(m.group(3)) > 23 for m in _QUOTED_IN_CODE.finditer(value)):
        return True
    if any(not _name_like(m.group(0)) for m in _CODE_TOKEN.finditer(value)):
        return True
    return _long_digits_in_brackets(value)


def _shorten_name_like(value: str) -> str:
    """*value* with each name-like token's parts of more than 24 characters
    (a dotted chain's segments) cut to a name that is no prose (x0): the
    classifier caps a code name at 24; the cap's name-like tokens pass."""

    def cut(m: re.Match[str]) -> str:
        token = m.group(0)
        if not _name_like(token):
            return token
        return ".".join("x0" if len(seg) > 24 else seg for seg in token.split("."))

    return _CODE_TOKEN.sub(cut, value)


_PROSE_VALUE_WHOLE = re.compile(_PROSE)
# A character outside ASCII that the classifier's rules read as an ASCII
# digit or letter (`\d` takes every script's digits, IGNORECASE folds the
# Kelvin sign and the dotted and dotless i): A.1's shapes are ASCII, so a
# value holding one is masked after a code name (fail closed).
_ASCII_LOOKALIKE = re.compile(r"(?![\x00-\x7f])(?:\d|(?i:[a-z]))")
_OWN_AUTHORIZATION = frozenset("_".join(words) for words in _AUTHORIZATION_NAMES)


def _name_before(text: str, start: int, word: str) -> str:
    """The rest of a name that ends at *start* in the header's *word*: the run
    of a name's characters just before it (at most 128), less an escape's own
    letters (the n of `\\nhas_authorization` in a JSON copy) or a colour
    code's tail (`32m`). They stay where without them the name is the
    header's own (`\\nproxy_authorization`, `ESC[32mproxy_authorization`,
    which the carrier read as prefixed by those letters)."""
    run = text[max(0, start - 128) : start]
    begin = start - (len(run) - len(run.rstrip(_NAME_CHARS)))
    tail = None
    if begin > 0 and text[begin - 1] == "\\":
        tail = _ESCAPE_LETTERS.match(text, begin, start)
    elif begin > 0 and text[begin - 1] == "[":
        tail = _COLOUR_TAIL.match(text, begin, start)
    if tail and _snake_name(text[tail.end() : start] + word) not in _OWN_AUTHORIZATION:
        begin = tail.end()
    return text[begin:start]


# A.1's number (the lead, 14:2xZ: "numbers <=5 digits or decimals code"): a
# kept value that starts like a number must be one. The classifier's test
# results (`12/15`, `0.`, `12345.`) are masked under a code name (the lead,
# 21:21Z: "Test-result counts like `12/15` under a credential name are
# masked").
_A1_NUMBER = re.compile(r"[-+]?(?:[0-9]{1,5}|[0-9]+\.[0-9]+)")
# Quoted prose (14:2xZ and 14:3xZ: "quoted strings masked unless prose"): one
# word of letters, closed at its opening depth, perhaps closing a sentence or
# a bracket.
_QUOTED_PROSE_TAIL = re.compile(r"[)\].;]{0,3}")
# A default (14:2xZ, the code shape's "literal default"): `= <literal or
# number>` after code (`Optional[str] = None`); any other `=` is no code.
_A1_DEFAULT = re.compile(
    r"True|False|None|true|false|null|[-+]?(?:[0-9]{1,5}|[0-9]+\.[0-9]+)"
)


def _default_ok(core: str) -> bool:
    """Whether every `=` outside brackets and strings in *core* opens a
    literal default that runs to the end."""
    depth, quote, run = 0, "", 0
    for index, char in enumerate(core):
        if quote:
            if char == quote and run == 0:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "=" and not depth:
            return _A1_DEFAULT.fullmatch(core[index + 1 :].strip()) is not None
        run = run + 1 if char == "\\" else 0
    return True


_LETTERS_WORD = re.compile(r"[A-Za-z]{1,15}")
_PROSE_WITH_CLOSERS = re.compile(r"[A-Za-z]{1,15}[)\].;]{0,2}")
_STRING_WINDOW = 4096
_STRING_END = re.compile(_STRING_END_AHEAD)


def _kept_code(value: str) -> bool:
    """Whether a value the classifier calls a setting's is shown after a code
    name: no digit or letter of another script, nothing the code-shape cap
    masks, its brackets balanced once the trailing unmatched closers are set
    aside (the lead, 21:21Z: "An unmatched trailing `)`, `]` or `}` closes the
    surrounding code, so the value is what precedes it"; quote-aware, as
    review R192C's oracle reads it), and a number where it starts like one."""
    if _ASCII_LOOKALIKE.search(value) is not None or _holds_key_token(value):
        return False
    _, _, closed, end = _code_scan(value)
    if not closed:
        return False
    core = value[:end]
    if not _default_ok(core):
        return False
    return not (core[:1] in ("+", "-", ".") or core[:1].isdigit()) or (
        _A1_NUMBER.fullmatch(core) is not None
    )


def _string_open(text: str, begin: int, quote: str, run: int) -> bool:
    """Whether a string opened by *quote* after a backslash run of *run* (a
    JSON copy's string, or a quoted string in code) is open at *begin*, on
    its line: an odd count of such quotes before it, within _STRING_WINDOW
    characters. Unknown (a longer line) counts as closed."""
    line = text.rfind("\n", 0, begin) + 1
    if begin - line > _STRING_WINDOW:
        return False
    found = re.findall(r"(?<!\\)" + "\\\\" * run + quote, text[line:begin])
    return len(found) % 2 == 1


def _glued_after(text: str, begin: int, end: int, value: str) -> bool:
    """Whether what follows a value at *end* is the value's own (fail closed):
    after any closers, a quote or backtick that closes no string or code span
    open before the name at *begin* (`2"`, `"A""`, `0,"`, `0``), or a `;` or
    `:` that ends the text or a string (`0;`, `0]:`; after prose it stays:
    `required;`). A `,` separates, as code and JSON write it (`x=True,` at a
    line's end)."""
    at = end
    while at < len(text) and at - end < 64 and text[at] in ")]}":
        at += 1
    # A comma with a quote glued after it (`0,"`), no JSON key's: the quote.
    if (
        text[at : at + 1] == ","
        and text[at + 1 : at + 2] in ('"', "'", "\\")
        and not re.match(_JSON_KEY_AHEAD, text[at + 1 : at + 300])
    ):
        at += 1
    nxt = text[at : at + 1]
    if nxt in ('"', "'"):
        return not _string_open(text, begin, nxt, 0)
    if nxt == "`":
        # Code quoted in prose closes a run of backticks opened before the
        # name on its line (`` `has_authorization: 1` ``); any other is the
        # value's (`0``).
        line = text.rfind("\n", 0, begin) + 1
        return begin - line > _STRING_WINDOW or (
            len(re.findall(r"`+", text[line:begin])) % 2 == 0
        )
    if nxt == "\\":
        escaped = re.match(r"(\\+)([\"'])", text[at : at + 66])
        return escaped is not None and not _string_open(
            text, begin, escaped[2], len(escaped[1])
        )
    if nxt in (";", ":"):
        follow = text[at + 1 : at + 3]
        if follow[:1] not in ("", '"', "'") and follow != '\\"':
            return False
        return _PROSE_WITH_CLOSERS.fullmatch(value) is None
    return False


def _key_words_masked(more: str) -> str:
    """The words after a code name's value with each key-shaped token in them
    masked: 24 characters or more, and not a name, or a dotted chain of names
    (`ModelManagementAuthChecks.can_user_make_model_call`: past the value the
    cap's eight parts count per segment)."""

    def name_or_chain(token: str) -> bool:
        return _name_like(token) or all(_name_like(seg) for seg in token.split("."))

    if len(more) < 24:  # no room for a token of _CODE_TOKEN's 24 characters
        return more
    return _CODE_TOKEN.sub(
        lambda m: m[0] if name_or_chain(m[0]) else "***REDACTED***", more
    )


# A quote written as an HTML entity at a name's end (`&quot;Authorization&quot;:`).
_QUOTE_ENTITY_END = re.compile(r"&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);\Z")


def _next_line_entry_value(match: re.Match[str]) -> str:
    """The next line's entry's quoted value, masked when the name before the
    separator masks that value on its own line (`NAME: '<v>'`, `NAME='<v>'`):
    the line is the name's value, read as the same line would be."""
    text = match.string
    end = match.start()
    while end > 0:
        entity = _QUOTE_ENTITY_END.search(text, max(0, end - 12), end)
        if entity:
            end = entity.start()
        elif text[end - 1] in "\"'\\":
            end -= 1
        else:
            break
    name = _name_before(text, end, "")
    if not name:
        return match[0]
    gap = " " if match["sep"] == ":" else ""
    same = f"{name}{match['sep']}{gap}{match['q']}{match['v']}{match['q']}"
    if redact_trajectory_text(same) == same:
        return match[0]
    return text[match.start() : match.start("v")] + "***REDACTED***"


_BREAK_THEN_SCHEME = re.compile(
    rf"(?:\\{{1,8}}[nrt]|\s)+(?:{_AUTH_SCHEMES}|Digest)", re.IGNORECASE
)
_NEXT_LINE_SCHEME = re.compile(
    rf"((?:[ \t]|\\{{1,8}}[nrt])*+(?:{_AUTH_SCHEMES}|Digest)(?:[ \t]|\\{{1,8}}t)++)"
    r"([^\s\"'\\,;)\]}]+)",
    re.IGNORECASE,
)


def _authorization_value(match: re.Match[str]) -> str:
    """The Authorization carrier's replacement: the name, the separator and a
    scheme kept, the value masked. After a code Authorization name
    (has_authorization, copy_source_authorization) the classifier decides with
    the whole name and value, as every other path does, and it calls every
    value a credential's (the lead's ruling (b), 2026-11-15 01:3xZ): the value
    is masked, and so is a key-shaped token in the words after it. Any other
    name that holds the header's word is _authorization_any_value's."""
    if "any" in match.re.groupindex and match["any"] is not None:
        return _authorization_any_value(match)
    groups = match.groupdict()
    if groups.get("camel") is not None and _key_start_at(match.string, match.start(1)):
        # The letter before the header's word is an escape's or a colour
        # code's: the header's own name, which the carrier above reads.
        return match[0]
    group = "code" if groups.get("code") is not None else "code_text"
    if match[group] is None:
        return match[1] + "***REDACTED***"
    n = "1" if group == "code" else "2"
    text = match.string
    more_group = f"code_more{n}"
    after = match.start(more_group) if match[more_group] is not None else match.end()
    glue = groups.get(f"code_glue{n}") or ""
    stop = after - len(glue)
    more = text[after : match.end()]
    # A value on the next line (the code value is an escaped line break and
    # nothing more): its first word is masked too, prose included (every name
    # that ends in `authorization` holds a credential: the lead, 2026-11-16
    # 10:53Z), a flag's switch, number or null kept.
    if more and re.fullmatch(r"(?:\\{1,8}[nrt]|\s)+", text[match.end(1) : stop]):
        # A scheme there keeps its word, and the token after it is masked
        # (`proxyAuthorization:\n  Bearer <token>` in a JSON copy).
        scheme = _NEXT_LINE_SCHEME.match(more)
        first = re.match(
            r"((?:[ \t]|\\{1,8}t)*+(?:\\{0,8}[\"'])?)([^\s\"'\\,;)\]}]+)", more
        )
        if scheme:
            more = more[: scheme.end(1)] + "***REDACTED***" + more[scheme.end(2) :]
        elif first:
            more = more[: first.end(1)] + "***REDACTED***" + more[first.end(2) :]
    # A value that is line breaks and a scheme (`proxyAuthorization:\n\nBearer
    # <token>` in a JSON copy): the token after it is masked too.
    if more and _BREAK_THEN_SCHEME.fullmatch(text[match.end(1) : stop]):
        token = re.match(r"((?:[ \t]|\\{1,8}t)++)([^\s\"'\\,;)\]}]+)", more)
        if token:
            more = more[: token.end(1)] + "***REDACTED***" + more[token.end(2) :]
    # A value that ends in a separator (`noneAuthorization:`) is a second name:
    # the word after it is its value, masked whole.
    if more and text[match.end(1) : stop].rstrip()[-1:] in (":", "="):
        lead = len(more) - len(more.lstrip(",; \t"))
        found = re.match(r"\S*", more[lead:])
        word = found[0] if found is not None else ""
        more = more[:lead] + "***REDACTED***" + more[lead + len(word) :]
    owned = _owned_glue(match, group, stop, glue) if glue else 0
    if owned:
        return match[1] + "***REDACTED***" + glue[owned:] + _key_words_masked(more)
    return _code_name_value(match, group, n, stop) + glue + _key_words_masked(more)


def _owned_glue(match: re.Match[str], group: str, stop: int, glue: str) -> int:
    """How much of *glue* (_code_glue) after a code name's value at *stop* is
    the value's: none where its first mark closes a string or a code span
    opened before the name (_glued_after); else up to a quote after a
    backtick that closes a string opened before the name, or all of it."""
    text = match.string
    name = _name_before(text, match.start(group), match[group]) + match[group]
    begin = match.start(group) - len(name) + len(match[group])
    value = text[match.end(1) : stop]
    escaped = len(value) - len(value.rstrip("\\"))
    if glue[0] in "\"'" and escaped % 2:
        # The value's reader took the backslashes of an escaped quote
        # (`&quot;x\\"` in a JSON copy): the quote is escaped at that depth.
        if _string_open(text, begin, glue[0], escaped):
            return 0
    elif not _glued_after(text, begin, stop, value):
        return 0
    first = re.match(r"`|(\\*)[\"']", glue)
    at = first.end() if first else 1
    second = re.match(r"(\\*)([\"'])", glue[at:])
    if (
        second
        and glue[0] == "`"
        and _string_open(text, begin, second[2], len(second[1]))
    ):
        return at
    return len(glue)


def _code_name_prose(match: re.Match[str]) -> str:
    """Prose after an Authorization header's name, masked where the name is a
    code Authorization name (is_code_authorization_name) and given back
    elsewhere: the header's own name after an escape's letters or a colour
    code (a key may start at its word), an upper-case or hyphenated name."""
    text, start = match.string, match.end("joint")
    if not match["joint"] and _key_start_at(text, start):
        return match[0]
    name = _reader_name(text, start, text[start : start + 13])
    if not _code_authorization_name(name):
        return match[0]
    return match[1] + "***REDACTED***"


@functools.lru_cache(maxsize=4096)
def _code_authorization_name(name: str) -> bool:
    """is_code_authorization_name, kept per name: a log repeats a few names."""
    return is_code_authorization_name(name)


@functools.cache
def _key_start() -> re.Pattern[str]:
    """The shared start boundary alone, compiled on first use (it holds the
    colour-code reading, which is slow to compile)."""
    return re.compile(_KEY_START)


def _key_start_at(text: str, at: int) -> bool:
    """Whether a key may start at *at* by the shared boundary: no letter or
    digit before it but an escape's or a colour code's."""
    return _key_start().match(text, at) is not None


def _reader_name(text: str, start: int, word: str) -> str:
    """The name that ends at *start* in *word*, as a reader reads it: the run
    of a name's characters before it, less an escape's own letters or a
    colour code's tail always (_name_before keeps them before the header's
    own name, as the carrier reads it)."""
    run = text[max(0, start - 128) : start]
    begin = start - (len(run) - len(run.rstrip(_NAME_CHARS)))
    tail = None
    if begin > 0 and text[begin - 1] == "\\":
        tail = _ESCAPE_LETTERS.match(text, begin, start)
    elif begin > 0 and text[begin - 1] == "[":
        tail = _COLOUR_TAIL.match(text, begin, start)
    return text[tail.end() if tail else begin : start] + word


def _code_name_at(match: re.Match[str], group: str) -> tuple[str, int]:
    """The whole code name that ends in *group* (_name_before) and where it
    begins."""
    name = _name_before(match.string, match.start(group), match[group]) + match[group]
    return name, match.start(group) - len(name) + len(match[group])


def _code_name_value(match: re.Match[str], group: str, n: str, stop: int) -> str:
    """_authorization_value for the value alone, which ends at *stop*: kept
    only where entry_kind calls it a setting's, which it never does under a
    code Authorization name (the lead's ruling (b), 2026-11-15 01:3xZ). The
    whole name is read only where a branch uses it (B8: the camel row)."""
    text = match.string
    kept = text[match.start() : stop]
    opener = match[f"code_open{n}"]
    if opener is not None:
        name, begin = _code_name_at(match, group)
        close, tail = match[f"code_close{n}"], match[f"code_tail{n}"]
        inside = text[
            match.end(f"code_open{n}") : match.start(
                f"code_close{n}" if close is not None else f"code_tail{n}"
            )
        ]
        # A bare or escaped quote that ends the JSON string around the line,
        # open before the name, is no opener: the value is empty. After a
        # quoted name (an entry, `"has_authorization": ")"`; the same quote
        # closes the name) it opens the value (review R192C, B7).
        quote = text[begin - 1 : begin]
        after = text[match.end(group) : match.end(group) + 16]
        entry = quote in ('"', "'") and after.lstrip("\\")[:1] == quote
        if (
            not entry
            and _STRING_END.match(text, match.end(f"code_open{n}"))
            and _string_open(
                text, begin, match[f"code_q{n}"], len(match[f"code_bs{n}"])
            )
        ):
            return kept
        if (
            _code_kind(text, match.start(group), match[group], name, inside)
            == "setting"
        ):
            return kept
        # A masked string keeps its quotes, so a JSON copy stays JSON (the
        # object layer reads its probe back with json.loads): a closed one its
        # opening quote and its close (where nothing is glued after it); one
        # left open, or a lone quote, its opening quote after the mask
        # (`has_authorization: ***REDACTED***"`), where it may close the JSON
        # string around the line.
        if close is None:
            return match[1] + "***REDACTED***" + opener
        return (
            match[1] + opener + "***REDACTED***" + (close if close and not tail else "")
        )
    tab = "\\" in match[1] and _literal_tab_before(
        match, text, _code_name_at(match, group)[1]
    )
    if _code_authorization_name(_reader_name(text, match.start(group), match[group])):
        # Every value is a credential's here (entry_kind; the lead's ruling (b)),
        # closers alone included: masked, unless nothing follows the separator.
        if tab:
            return (
                re.sub(r"(?:\\+t|[ \t])+$", " ", match[1]).rstrip() + " ***REDACTED***"
            )
        return (
            match[1] + "***REDACTED***" if text[match.end(1) : stop].strip() else kept
        )
    name, begin = _code_name_at(match, group)
    # As the classifier reads it: a JSON copy's escaped quote and backslash
    # are the text's own. The value is what precedes its trailing unmatched
    # closers; closers alone are no value.
    value = _COPY_ESCAPE.sub(r"\1", text[match.end(1) : stop])
    value = value[: _code_scan(value)[3]]
    if not value:
        # Closers alone (`cfg.proxy_authorization: ) done`): masked, as the rc
        # masks them; every value under such a name is (the lead's ruling (b)).
        raw = text[match.end(1) : stop]
        return (
            match[1] + "***REDACTED***"
            if raw.strip() or _glued_after(text, begin, stop, value)
            else kept
        )
    if tab:
        # The backslash and t are the value's: masked from the separator.
        return re.sub(r"(?:\\+t|[ \t])+$", " ", match[1]).rstrip() + " ***REDACTED***"
    if _code_kind(text, match.start(group), match[group], name, value) == "setting":
        return kept
    return match[1] + "***REDACTED***"


def _code_kind(text: str, start: int, word: str, name: str, value: str) -> str | None:
    """The kind of *value* under the code name *name* that ends at *start* in
    *word*: the classifier's, a credential's under a code Authorization name.
    Where the reader reads another name, the header's own (an escape's
    letters or a colour code's tail before it, `\\nproxy_authorization`,
    which the carrier reads as prefixed; proxyAuthorization), one prose word
    stays (A.2), as on the rc."""
    if not _code_authorization_name(_reader_name(text, start, word)):
        return "setting" if _PROSE_VALUE_WHOLE.fullmatch(value) else "credential"
    return _classified(name, value)


def _classified(name: str, value: str) -> str | None:
    """entry_kind's answer for *value* under *name*, kept for this run of the
    carrier (_classifier_answers); None where it raises (masked)."""
    answers = _classifier_answers.get()
    kind = answers.get((name, value), _UNASKED) if answers is not None else _UNASKED
    if kind is _UNASKED:
        try:
            kind = entry_kind(name, value)
        except Exception:
            kind = None
        if answers is not None and len(value) <= _ANSWER_VALUE_MAX:
            answers[name, value] = kind
    return kind if isinstance(kind, str) else None


def _literal_tab_before(match: re.Match[str], text: str, begin: int) -> bool:
    """Whether the gap after the separator holds a backslash and a t that are
    no tab: a JSON copy at depth d (its innermost string open before the
    name, opened by a quote after 0, 1, 3 or 7 backslashes) writes a tab
    with 2 ** (d - 1) backslashes; any other run, and any in raw text
    (`has_authorization: \\tx`), is the value's."""
    runs = [len(m[1]) for m in re.finditer(r"(\\+)t", match[1])]
    if not runs:
        return False
    depth = max(
        (
            d
            for d, run in enumerate((0, 1, 3, 7), 1)
            if _string_open(text, begin, '"', run)
        ),
        default=0,
    )
    return depth == 0 or any(run != 2 ** (depth - 1) for run in runs)


# Names that hold the Authorization header's word but that no carrier reads
# (round 2; the lead's 17:0xZ rule): the word glued to a letter or a
# digit (`retryAuthorization`, `noneauthorization`, the camel
# `proxyAuthorization` unquoted; not an escape's letter or a colour code's
# tail, where the carrier reads the name), the word with more after it
# (`AuthorizationHeader`, `authorization_header`, `authorization2`), and a name
# closed by `}`, `)` or `%`, perhaps after a shell default
# (`${HTTP_AUTHORIZATION}`, `$(Authorization)`, `%AUTHORIZATION%`,
# `${HTTP_AUTHORIZATION:-x}`). A name after a joiner (`Proxy-Authorization`,
# `HTTP_AUTHORIZATION`) is the carrier's own. The classifier decides each with
# the whole name (_authorization_any_value): kept as code, the name's own words
# or prose where it says so; anything else masked (fail closed). A value with a
# scheme is the scheme carriers'.
_AUTHORIZATION_ANY_NAME = (
    rf"(?:(?<=[A-Za-z0-9]{_NOT_AFTER_ESCAPE}{_NOT_AFTER_CSI})"
    r"(?:proxy[_-]?)?authorization[A-Za-z0-9_.-]{0,64}+"
    rf"|{_KEY_START}(?:proxy[_-]?)?authorization[A-Za-z0-9_.-]{{1,64}}+"
    rf"|(?<=[A-Za-z0-9_.{{(%-])(?:proxy[_-]?)?authorization(?={_SHELL_CLOSE}))"
)


_OWN_QUOTED_AFTER = re.compile(r"\\{0,8}[\"'][ \t]*+(?:[:=]|%(?:25)?3[AaDd])")
# An Authorization name and its separator earlier on the line: the context in
# which a name glued to the header's word is a second name (A.2's words,
# `Authorization: please retryAuthorization: "<v>"`). Alone, such a name is
# not read (in the corpus every one before `:` or `=` is code: httpAuthorization).
_HEADER_BEFORE = re.compile(
    r"authorization\\{0,8}[\"']?[ \t]*+(?:[:=]|%(?:25)?3[AaDd])", re.IGNORECASE
)
_HEADER_LOOKBACK = 256
_PROSE_WHOLE = re.compile(_PROSE)


def _authorization_any_value(match: re.Match[str]) -> str:
    """_authorization_value for _AUTHORIZATION_ANY_NAME: the value kept where the
    classifier calls the entry, with the whole name, a setting's; masked
    otherwise (fail closed), a quoted value inside its quotes."""
    word = match["any"]
    prefix = _name_before(match.string, match.start("any"), word)
    name = prefix + word
    begin = match.start("any") - len(prefix)
    # The header's own name quoted (`"ProxyAuthorization": "required"`) is the
    # carrier's: it read the entry and kept or masked it.
    # Each check below that keeps the match runs in the order of its cost; any
    # order gives the same answer (B8: a line of `xAuthorization: f(x)`).
    if (
        match.string[begin - 1 : begin] in ('"', "'")
        and _OWN_QUOTED_AFTER.match(match.string, match.end("any"))
        and _snake_name(name) in _OWN_AUTHORIZATION
    ):
        return match[0]
    if prefix[-1:].isalnum():
        text = match.string
        low = max(0, begin - _HEADER_LOOKBACK)
        low = max(low, text.rfind("\n", low, begin), text.rfind("\\n", low, begin) + 1)
        if not _HEADER_BEFORE.search(text, low, begin):
            return match[0]
    # A header's url, name or policy is a setting (AUTHORIZATION_HEADER_NAME).
    words = name_words(name)
    if words and words[-1] in _HEADER_QUALIFIERS:
        return match[0]
    part = "inner" if match["inner"] is not None else "run"
    start, end = match.span(part)
    value = _COPY_ESCAPE.sub(r"\1", match.string[start:end])
    # One prose word stays, as after the header's own name (the residual).
    if _PROSE_WHOLE.fullmatch(value) and _kept_code(value):
        return match[0]
    answers = _classifier_answers.get()
    kind = answers.get((name, value), _UNASKED) if answers is not None else _UNASKED
    if kind is _UNASKED:
        try:
            kind = entry_kind(name, value)
        except Exception:
            kind = None
        if answers is not None and len(value) <= _ANSWER_VALUE_MAX:
            answers[name, value] = kind
    if kind == "setting" and _kept_code(value) and not _holds_key_literal(value):
        return match[0]
    offset = match.start()
    return match[0][: start - offset] + "***REDACTED***" + match[0][end - offset :]


def _open_document(value: str, *, copy: bool = False, listed: bool = False) -> str:
    """An inline document not closed, nested too deep or longer than the
    cap: its first line masked, then each next line that keeps a JSON key's
    shape masked after its key, to the line's end. The carrier matched only
    those lines of a document not closed; of one past the cap, the rest is
    text again (its size bounds how deep that goes). In a JSON string's copy
    (*copy*) a line ends at `\\n`. Of a list (*listed*), a line of brackets,
    braces and commas alone is kept and read past, and a line that opens an
    item is masked after its `{`. The lines are read by index, once."""
    keyed = _COPY_KEY_LINE_START if copy else _KEY_LINE_START
    bracketed = _COPY_BRACKET_LINE_START if copy else _BRACKET_LINE_START
    opened = _COPY_ITEM_LINE_START if copy else _ITEM_LINE_START
    carriage = "\\r" if copy else "\r"

    def line_end(start: int) -> tuple[int, int]:
        """Where the line from *start* ends and the next one starts."""
        if copy:
            found = _COPY_LINE_BREAK.search(value, start)
            return (found.end(1), found.end()) if found else (len(value), len(value))
        stop = value.find("\n", start)
        return (stop, stop + 1) if stop >= 0 else (len(value), len(value))

    stop, start = line_end(0)
    out = ["***REDACTED***" + (carriage if value[:stop].endswith(carriage) else "")]
    while start < len(value):
        next_stop, next_start = line_end(start)
        line = value[start:next_stop]
        if listed and bracketed.match(line):
            out.append(value[stop:next_stop])
            stop, start = next_stop, next_start
            continue
        key = keyed.match(line) or (opened.match(line) if listed else None)
        if key is None:
            tail = value[stop:]
            return "".join(out) + redact_trajectory_text(tail)
        out.append(
            value[stop:start]
            + key.group(0)
            + ("***REDACTED***" if key.group(0).endswith("{") else " ***REDACTED***")
            + (carriage if line.endswith(carriage) else "")
        )
        stop, start = next_stop, next_start
    return "".join(out) + value[stop:]


# The schemes of a URL that may carry credentials in its userinfo, as the
# URL rules below read them (the first rule takes all; the token-as-userinfo
# rule the web and shell ones).
_URL_TOKEN_SCHEMES = r"(?:https?|wss?|ssh|sftp|ftp|ftps)"
_URL_SCHEMES = (
    r"(?:https?|wss?|ssh|sftp|ftp|ftps|postgres|postgresql|mysql|"
    r"mongodb(?:\+srv)?|redis|rediss|amqp|amqps)"
)


def _key_run(chars: str, count: str) -> str:
    """A key's body: *count* of *chars*, stopping before a URL scheme glued
    right after it (review of PR #173, 2026-11-13, RPUB's L2:
    `sk-<key>https://u:<pw>@host` masked the key and `https` with it, and
    the URL rule, which reads the scheme, then left the password). The stop
    is the URL rules' own scheme list, so the URL rule reads what follows."""
    return rf"(?:(?!{_URL_SCHEMES}://){chars}){count}"


# Keys with a shape of their own (a provider prefix, a JWT, a PEM block),
# masked whole wherever they stand. redact_key_families applies only these.
_KEY_FAMILY_PATTERNS: list[tuple[_LazyPattern, str, str]] = [
    # PEM private-key blocks (GCP/Vertex service-account JSON, RSA keys). Redact
    # the whole armored block incl. the base64 body; the carriers below stop at
    # the first space and would leave the key material. Lazy + length-capped body
    # keeps it bounded (#830).
    (
        _LazyPattern(
            r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----"
            r"[\s\S]{0,8192}?"
            r"-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----"
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_PRIVATE_KEY,
    ),
    # --- Token families: redacted WHOLE (prefix included) so the v0.5 leak audit
    # greps (``AIzaSy``/``dtn_``…) see no live-key shape (#537/#585). ---
    # Anthropic: sk-ant-api03-...
    (
        _LazyPattern("sk-ant-" + _key_run("[a-zA-Z0-9_-]", "{12,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Google OAuth access tokens (ya29.), Google AI Studio's newer keys (AQ.,
    # also masked at export, publish/redact.py), GitLab personal access
    # tokens (glpat-) and Stripe live keys (sk_live_, rk_live_). AQ. and the
    # Stripe keys start a word (_KEY_START), so `FAQ.`, `IRAQ.`,
    # `mask_live_...` and `work_live_...` stay.
    (
        _LazyPattern(
            r"ya29\."
            + _key_run("[A-Za-z0-9_-]", "{20,}")
            + "|glpat-"
            + _key_run("[A-Za-z0-9_-]", "{20,}")
            + "|"
            + _at_key_start(r"AQ\.")
            + _key_run("[0-9A-Za-z_-]", "{20,}")
            + "|"
            + _at_key_start("[sr]k_live_")
            + _key_run("[A-Za-z0-9]", "{16,}")
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # BenchFlow proxy master key as older releases made it
    # (sk-benchflow-<token_urlsafe(24)>), OpenAI service account (sk-svcacct-),
    # OpenRouter (sk-or-v1-): rare labels that won't appear in a normal kebab
    # slug, so the hyphen-permitting entropy class is safe here.
    (
        _LazyPattern(
            "sk-(?:benchflow|svcacct|or-v1)-" + _key_run("[A-Za-z0-9_-]", "{12,}")
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # OpenAI org-scoped sk-proj-/sk-admin- and the BenchFlow proxy master key
    # (sk-runtime-<token_urlsafe(24)>): `proj`/`admin`/`runtime` ARE common
    # identifier words, so a bare hyphen class would redact kebab slugs like
    # `sk-proj-refactor-auth` or `task-runtime-configuration`. Real keys are
    # high-entropy base64url and always carry an uppercase char; gate on a
    # lookahead for one so lowercase slugs survive (#830). Before generic sk- so
    # it wins.
    (
        _LazyPattern(
            r"sk-(?:proj|admin|runtime)-(?=[A-Za-z0-9_-]{0,200}[A-Z])"
            + _key_run("[A-Za-z0-9_-]", "{12,}")
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Other provider key families with distinctive prefixes — caught here for the
    # bare-token-in-prose form that no NAME=value carrier sees (a provider
    # exception that echoes the key): gsk_ (Groq) and csk- (Cerebras, masked
    # whole wherever it stands, before the generic sk- rule below would leave
    # its `c`: review R207), here; xai-, r8_ (Replicate), hf_
    # (HuggingFace), fw_ (Fireworks) below. ≥20-char anchors avoid short ids
    # (#830).
    (
        _LazyPattern("(?:gsk_|csk-)" + _key_run("[A-Za-z0-9]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # OpenAI / generic sk- (alphanumeric only — widening to include `-` would
    # match common slugs like `task-sk-us-east-1-...`), wherever it stands, as
    # in 0.9.2, a key glued to what comes before it included. The
    # one exception is the word of Claude Code's `<task-notification>` tags
    # (_TAG_WORD); every other word ending in `sk` is masked as in 0.9.2.
    (
        _LazyPattern(rf"sk-(?!{_TAG_WORD})" + _key_run("[a-zA-Z0-9]", "{12,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Google AI / Gemini: AIzaSy... (≥20 char suffix avoids matching `AIzaSy`
    # alone). Prefix is redacted too so the audit grep for `AIzaSy` is clean.
    (
        _LazyPattern("AIzaSy" + _key_run("[A-Za-z0-9_-]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # AWS access keys: AKIA/ASIA + exactly 16 chars; length anchor avoids
    # matching English words like "ASIAPACIFIC".
    (
        _LazyPattern(r"(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Daytona SDK tokens: dtn_... — ≥16 char suffix avoids short ids (`dtn_v2`).
    (
        _LazyPattern("dtn_" + _key_run("[A-Za-z0-9_]", "{16,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # GitHub tokens: PATs / OAuth / app / refresh (ghp_/gho_/ghu_/ghs_/ghr_) and
    # fine-grained PATs (github_pat_...). ≥20 char suffix avoids short slugs.
    (
        _LazyPattern("(?:ghp|gho|ghu|ghs|ghr)_" + _key_run("[A-Za-z0-9]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    (
        _LazyPattern("github_pat_" + _key_run("[A-Za-z0-9_]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Slack tokens: bot/user/app/refresh/legacy (xoxb-/xoxp-/xoxa-/xoxr-/xoxs-).
    (
        _LazyPattern("xox[baprs]-" + _key_run("[A-Za-z0-9-]", "{10,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    (
        _LazyPattern("xai-" + _key_run("[A-Za-z0-9]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    (
        _LazyPattern("r8_" + _key_run("[A-Za-z0-9]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    (
        _LazyPattern("hf_" + _key_run("[A-Za-z0-9]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    (
        _LazyPattern("fw_" + _key_run("[A-Za-z0-9]", "{20,}")),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # JSON Web Tokens (session/bearer creds): three base64url segments split by
    # dots. The ``eyJ`` prefix is base64url of ``{"`` — a JWT header. The 3rd
    # (signature) segment requires ≥20 chars (real HS256/RS256 sigs are 43+) so
    # short dotted base64/method-chains that merely start ``eyJ`` aren't redacted;
    # a left boundary (_KEY_START) keeps it from starting mid-word. Segment
    # upper bounds keep it ReDoS-bounded (#830).
    (
        _LazyPattern(
            _at_key_start("eyJ")
            + r"[A-Za-z0-9_-]{10,512}\.[A-Za-z0-9_-]{6,512}\."
            + _key_run("[A-Za-z0-9_-]", "{20,512}")
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
]

# The rest of a line of a YAML block's body: up to a line break, as written
# or escaped, or a JSON string's closing quote (the body of a json.dumps copy
# ends there).
_LINE_REST = r"(?:[^\r\n\\\"]|\\(?!\\*[nr])[\s\S])"
_BLOCK_SPACE = r"(?:[ \t]|\\{1,8}t)"
# A YAML block scalar under a name (`apiKey: |\n  <v>\n  <w>`): the
# indicator line, then every line indented at least as far as the first
# (blank lines in between included), as written or escaped. _yaml_block
# masks each body line when is_secret_field holds for the name; _gap hands
# a carrier only the first line's value.
# Its indicator (`|` or `>`) is a literal: a text with neither has no block.
_YAML_BLOCK = _GatedPattern(
    rf"(?P<head>{_KEY_START}(?:\\{{0,8}}[\"'])?(?P<name>[A-Za-z_][A-Za-z0-9_.-]{{0,63}})"
    r"(?:\\{0,8}[\"'])?[ \t]*:"
    + _BLOCK_SPACE
    + r"*+[|>](?:[+-][1-9]?|[1-9][+-]?)?"
    + _BLOCK_SPACE
    + r"*+)"
    rf"(?P<body>(?:{_NEWLINE}{_BLOCK_SPACE}*+(?={_NEWLINE}))*+{_NEWLINE}"
    rf"(?P<indent>{_BLOCK_SPACE}++){_LINE_REST}*+"
    rf"(?:(?:{_NEWLINE}{_BLOCK_SPACE}*+(?={_NEWLINE}))*+{_NEWLINE}(?P=indent){_LINE_REST}*+)*+)",
    chars="|>",
)
_BLOCK_LINE = re.compile(rf"({_NEWLINE}{_BLOCK_SPACE}*+){_LINE_REST}++")


def _yaml_block(match: re.Match[str]) -> str:
    """A YAML block's body under a name is_secret_field holds for, each
    line masked; under any other name, as written."""
    if not is_secret_field(match["name"]):
        return match[0]
    return match["head"] + _BLOCK_LINE.sub(r"\1***REDACTED***", match["body"])


_REDACTION_PATTERNS: list[
    tuple[_LazyPattern, str | Callable[[re.Match[str]], str], str]
] = [
    *_KEY_FAMILY_PATTERNS,
    # --- URL credentials ---
    # Credentials in a URL's userinfo (scheme://user:pass@host) for KNOWN
    # credential-bearing schemes (http/db/cache/mq) — a generic scheme class would
    # scrub bespoke app deep-links (vscode://a:b@…). Drop the whole userinfo; keep
    # scheme + host. The username class excludes `:` so it stops deterministically
    # at the password separator (no catastrophic backtracking), both classes
    # exclude `?`/`#` so the match can't span into a query string and grab an `@`
    # from an email param, and the password may contain `@` so an un-encoded `@` in
    # the password is fully scrubbed (greedy to the last `@`). Length caps keep it
    # ReDoS-bounded. A plain host:port (no `@`) is untouched (#830).
    (
        _LazyPattern(
            rf"({_URL_SCHEMES}://)"
            r"[^\s/?#:@]{0,256}:[^\s/?#]{0,256}@"
        ),
        r"\1***REDACTED***@",
        REDACTION_CATEGORY_URL_CREDENTIAL,
    ),
    # A token alone as the userinfo (https://<token>@host, as GitHub and
    # GitLab accept), sixteen characters or more. It stays only when it reads
    # as a user name (deploy-automation@, build-service-account@): no digit,
    # no capital letter, and a `-`, `.` or `_` at least once in every sixteen
    # characters. A digit, a capital letter or sixteen characters in a row
    # without one of `-._` is a token's. Shorter names (git@, admin@) stay.
    (
        _LazyPattern(
            rf"({_URL_TOKEN_SCHEMES}://)"
            r"(?=[a-z_.~%+-]{0,255}[A-Z0-9]"
            r"|[A-Za-z0-9_.~%+-]{0,240}[A-Za-z0-9~%+]{16})"
            r"[A-Za-z0-9_.~%+-]{16,256}@(?=[A-Za-z0-9\[])"
        ),
        r"\1***REDACTED***@",
        REDACTION_CATEGORY_URL_CREDENTIAL,
    ),
    # Secrets carried as URL query params. Curated EXACT param names (anchored to
    # `?`/`&` with `=` right after the name) so non-secret params (`?page=`,
    # `?key=name`, `?author=`, `?auth=basic`) are untouched; the value stops at
    # `&` so sibling params survive. Bare `key`/`token`/`auth` are intentionally
    # excluded (too often a non-secret field/mode); so are bare `sig`/`pwd` (a
    # signature-scheme version / a working-directory path). A real key value under
    # `?key=` is still caught by its token-family pattern above. Includes AWS
    # SigV4 / Azure SAS / access-key names. Placed BEFORE the carriers below so the
    # `*token*` carrier can't over-consume the rest of the query string past `&`.
    # Like _SECVAL, the value class excludes `*` so an already-redacted
    # `?access_token=***REDACTED***` isn't re-counted — a re-match here made the
    # publish-layer stability loop diverge and the server rescan reject (#1008).
    (
        _LazyPattern(
            r"([?&](?:"
            r"api_?key|access_key|access_token|refresh_token|session_token|"
            r"session_id|sessionid|client_secret|aws_secret_access_key|account_key|"
            r"secret|password_hash|password|passwd|"
            r"signature|x-amz-signature|x-amz-credential|x-amz-security-token|sas"
            r")=)[^&\s\"'*]+",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_URL_CREDENTIAL,
    ),
    # Azure SAS `?sig=<token>` — bare `sig` is excluded from the curated list above
    # (a `?sig=v2` scheme-version flag is a common non-secret), so gate on a
    # high-entropy value length: a real SAS signature is a 40+ char base64 HMAC,
    # while a version flag is short (#830). Excludes `*` like the curated params
    # above so the 14-char `***REDACTED***` marker can never re-match (#1008).
    (
        _LazyPattern(r"([?&]sig=)[^&\s\"'*]{16,}", re.IGNORECASE),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_URL_CREDENTIAL,
    ),
    # A YAML block scalar under a credential's name (_YAML_BLOCK): every
    # body line, where a carrier reads only its first line's value.
    (_YAML_BLOCK, _yaml_block, REDACTION_CATEGORY_CREDENTIAL_FIELD),
    # --- Header / key-value secret carriers ---
    # Match `name: value` / `name=value` in raw, JSON, Python dict-repr
    # (single-quoted), AND json.dumps-escaped (`\"name\": \"value\"`) forms — see
    # _ESCQ. The name is kept; only the value is dropped (#585/#830). No leading
    # boundary on the hyphenated header names — the hyphen keeps them from matching
    # inside underscore variable names.
    (
        _LazyPattern(
            rf"((?:{_ESCQ}(?:x-api-key|x-goog-api-key|api-key){_ESCQ}){_SEP}{_GAP}{_ESCQ}"
            rf"{_SCHEME}){_GLUED_SCHEME}{_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Underscore/camel-case forms `api_key` and `apiKey`. No leading boundary:
    # inline harness JSON and namespaced env dumps like
    # `GEMINI_API_KEY=secret` / JSON keys such as `"openai_api_key"` must redact
    # too (#585). The `\1` capture preserves the matched name+separator.
    (
        _LazyPattern(
            rf"({_ESCQ}{_API_KEY_NAME}{_NAME_NUMBER}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SCHEME})"
            rf"{_GLUED_SCHEME}{_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # `master_key`/`master-key` and `private_key`/`private-key` carriers — the
    # BenchFlow proxy master key and GCP/Vertex service-account private-key field
    # as labelled values (the PEM body itself is scrubbed by the block rule above).
    (
        _LazyPattern(
            rf"({_ESCQ}{_KEY_FIELD_NAME}{_NAME_NUMBER}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SCHEME})"
            rf"{_GLUED_SCHEME}{_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # Bearer-token env vars whose NAME has BEARER_TOKEN in the MIDDLE, e.g.
    # `AWS_BEARER_TOKEN_BEDROCK=` — the secret-suffix rule below anchors its marker
    # at the name end, so a TOKEN in the middle slips through. `BEARER_TOKEN` is an
    # unambiguous secret signal, safe as a substring (#830). Names length-capped.
    (
        _LazyPattern(
            rf"{_KEY_START}"
            rf"({_ESCQ}{_NAME}BEARER_TOKEN{_NAME}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SCHEME})"
            rf"{_GLUED_SCHEME}{_SECVAL}",
            re.IGNORECASE,
            words=_BEARER_TOKEN_WORDS,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # Generic secret-NAME-suffix carriers: names ending in TOKEN/SECRET/PASSWORD or
    # a specific *secret* key suffix (ACCESS_KEY/SECRET_KEY/ACCOUNT_KEY/…). The KEY
    # forms are spelled out (NOT a bare `_KEY`) so common non-secret identifiers
    # like `primary_key`/`foreign_key`/`sort_key` are NOT redacted, while
    # `AWS_SECRET_ACCESS_KEY`/`AZURE_STORAGE_ACCOUNT_KEY` are. TOKEN/SECRET at the
    # name end stay safe from `TOKENIZER`/`SECRETARY` (marker not at the end).
    (
        _LazyPattern(
            rf"{_KEY_START}{_NOT_AT_A_FLAG}"
            rf"({_ESCQ}{_ESCAPE_TAIL}(?P<decided>{_DECIDED_START}{_DECIDED_NAME})"
            rf"{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SCHEME}"
            rf"|{_ESCQ}{_ESCAPE_TAIL}{_LOWER_NAME}"
            rf"(?P<decided_lower>{_DECIDED_START}{_DECIDED_LOWER_NAME})"
            rf"(?:{_LOWER_ENTRY}|{_LOWER_ENV}|{_LOWER_TEXT})"
            rf"|{_ESCQ}(?:{_NAME}{_SECRET_NAME_END}|PASS_PHRASE){_NOT_AFTER_A_FLAG}"
            rf"{_NAME_NUMBER}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SCHEME})"
            rf"(?(schema)(?={_KEY_VALUE}+{_RUN_END}))"
            rf"(?(schema_lower)(?={_KEY_VALUE}+{_RUN_END}))"
            rf"(?P<value>(?(content)(?:{_JSON_VALUE}|{_JSON_LIST}|{_PATH_RUN})|(?!))"
            rf"|{_GLUED_SCHEME}{_SECVAL})",
            re.IGNORECASE,
            words=_SECRET_NAME_WORDS,
        ),
        _secret_name_value,
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # A price's name (the classifier's _price_unit: input_cost_per_token,
    # usd_per_token, cost_per_1k_token, PER-TOKEN, time_per_output_token),
    # which the carrier above leaves out (_PRICE_TOKEN, _PRICED_TOKEN), holds
    # a number only; any other value is masked, as 0.9.3 masks it.
    # The classifier decides (_price_value).
    (
        _LazyPattern(
            rf"{_KEY_START}({_ESCQ}(?P<price>{_PRICE_NAME}){_ESCQ}{_SEP}{_GAP}{_ESCQ})"
            rf"(?P<value>{_PRICE_TABLE}|{_SECVAL})",
            re.IGNORECASE,
            words=("token",),
        ),
        _price_value,
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # The same carrier reads a numbered name, a hint and a primary or a
    # secondary key's name (_DECIDED_NAME: KEY_2=..., TOKEN_V2=...,
    # id_token_hint=..., primary_key=<a key's run>), and masks its value
    # only where the classifier calls the entry a credential's
    # (_secret_name_value: TOKEN_2FA_ENABLED=true, MAX_TOKENS_2=8192 and
    # primary_key=external_subscription_id stay).
    # Names that end in _PASS or _PWD (DB_PASS, MYSQL_PWD, SMTP_PASS), perhaps
    # numbered (DB_PASS_2), or in PASS_VALUE (_PASS_VALUE_NAME), in upper
    # case and with the underscore: `PWD=/app`, `OLDPWD=...` and `BYPASS=1`
    # are no secrets; nor is a test's result (_TEST_LIST_NAME_END,
    # _OUTCOME_NAME_END, _COUNT_NAME_END). The same names in lower and camel
    # case where their value is no code (_LOWER_PASS: `db_pass=v`,
    # `"dbPass": "v"`; not `first_pass=True` or `db_pass = db_pass`).
    (
        _LazyPattern(
            rf"(?=[A-Za-z0-9_\\\"']){_KEY_START}({_ESCQ}"
            rf"(?:(?:{_PASS_NAME}{_TEST_LIST_NAME_END}"
            rf"(?!(?:{_OUTCOME_NAME_END}){_ESCQ}{_SEP}{_GAP}{_ESCQ}{_TEST_RESULT}"
            rf"|(?:{_COUNT_NAME_END}){_ESCQ}{_SEP}{_GAP}{_ESCQ}{_COUNT_OF_ANY_LENGTH})"
            rf"(?:{_PASS_NUMBER})?|{_PASS_VALUE_NAME}|{_PASS_ALONE})"
            rf"{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_SCHEME}|{_LOWER_PASS})){_LIST_START}"
            rf"{_GLUED_SCHEME}{_SECVAL}",
            words=_PASS_NAME_WORDS + _LOWER_PASS_NAME_WORDS,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # Cookie and Set-Cookie headers whose value is cookie pairs (`name=value`,
    # `; `-separated, a session id in any of them, the classifier's
    # _COOKIE_PAIR): the pairs and no further. A header named in prose
    # (`Set-Cookie: header too large`, `cookie_policy: strict`) is no cookie.
    # A name that ends in one after a prefix (_COOKIE_PREFIXED: HTTP_COOKIE,
    # has_cookie) holds a secret unless its value is prose, as entry_kind
    # reads it: a quoted value whole (`"has_cookie": "please sign in"` is
    # masked whole, `"has_cookie": "1"` stays), an unquoted one by its first
    # word on its line, where it is a value and no code, as after a
    # lower-case pass name (_ENV_VALUE, _TEXT_VALUE: `HTTP_COOKIE=<opaque>`
    # is masked as the pairs are; `has_cookie: 1`, `end_cookie = gen_cookie()`
    # and `self.session_cookie = session_cookie` stay).
    (
        _LazyPattern(
            rf"(?=[{_COOKIE_STARTS}\\\"']){_KEY_START}({_ESCQ}"
            rf"(?:(?:{_COOKIE_PREFIXED}{_COOKIE_NAME}|{_COOKIE_CAMEL}){_ESCQ}{_SEP}{_GAP}{_QUOTE}"
            rf"(?!{_PROSE}{_QUOTE})(?={_ENTRY_TEXT}{_QUOTE})(?P<entry>)"
            rf"|{_COOKIE_PREFIXED}{_COOKIE_NAME}{_ESCQ}(?:=|%(?:25)?3[Dd]){_SCHEME}"
            rf"(?!{_TEXT_PROSE}){_ENV_VALUE}"
            rf"|{_COOKIE_PREFIXED}{_COOKIE_NAME}{_ESCQ}{_SEP}{_WS}*+{_SCHEME}"
            rf"(?!{_TEXT_PROSE}){_TEXT_VALUE}"
            rf"|(?:{_COOKIE_NAME}|{_COOKIE_OWN_CAMEL}){_ESCQ}{_SEP}{_COOKIE_GAP}{_ESCQ}"
            rf"(?={_COOKIE_PAIR})))"
            rf"(?(entry){_ENTRY_TEXT}"
            rf"|{_VALUE_START}(?:{_BRACED_PCT}|{_COOKIE_QUOTED}|[^\s;\"'\\*]|{_VALUE_ESCAPE}){{1,4096}}"
            rf"(?:;[ \t]*(?:{_BRACED_PCT}|{_COOKIE_QUOTED}|[^\s;\"'\\*]|{_VALUE_ESCAPE}){{1,4096}}){{0,64}})",
            re.IGNORECASE,
            words=_COOKIE_WORDS,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # authorization header WITH a scheme (Bearer/Token/Basic/…): keep scheme.
    # Proxy-Authorization is the same header for a proxy, HTTP_AUTHORIZATION
    # and REDIRECT_HTTP_AUTHORIZATION the same in a CGI or WSGI env dump.
    # A quoted camel-case prefixed name too (_AUTHORIZATION_CAMEL:
    # `"httpAuthorization": "Bearer <token>"`), here and for Digest.
    (
        _LazyPattern(
            rf"(?=[{_AUTHORIZATION_STARTS}\\\"']){_KEY_START}({_ESCQ}"
            rf"(?:{_AUTHORIZATION_NAME}|{_AUTHORIZATION_CAMEL}|{_AUTHORIZATION_OWN_CAMEL})"
            rf"{_ESCQ}{_SEP}{_SCHEME_GAP}{_ESCQ}"
            rf"{_AUTH_SCHEMES}(?={_WS}|{_NEWLINE}){_GAP}){_SECVAL}",
            re.IGNORECASE,
            words=_AUTHORIZATION_WORDS,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    (
        _LazyPattern(
            rf"(?=[{_AUTHORIZATION_STARTS}\\\"']){_KEY_START}({_ESCQ}"
            rf"(?:{_AUTHORIZATION_NAME}|{_AUTHORIZATION_CAMEL}|{_AUTHORIZATION_OWN_CAMEL})"
            rf"{_ESCQ}{_SEP}{_GAP}{_ESCQ}"
            rf"Digest(?={_WS}|{_NEWLINE}){_GAP}){_DIGEST_PARAM}(?:,{_WS}*+{_DIGEST_PARAM}){{0,32}}",
            re.IGNORECASE,
            words=_AUTHORIZATION_WORDS,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # authorization header with a bare value (no recognized scheme). The negative
    # lookahead skips scheme-prefixed values already handled above, so
    # `Bearer <tok>` isn't double-redacted into `***REDACTED*** ***...`, and
    # prose (_PROSE_VALUE: `Authorization: required`, `has_authorization: 1`).
    # A prefixed name's quoted value (_AUTHORIZATION_PREFIXED) is read whole,
    # as entry_kind reads it: `"has_authorization": "please sign in first"`
    # is masked whole; an unquoted one by its first word.
    (
        _LazyPattern(
            rf"(?=[{_AUTHORIZATION_STARTS}\\\"']){_KEY_START}({_ESCQ}"
            # A code name's entry is the code branches' (its string read at its
            # own depth, `"f(\")\", 1234567)"` whole).
            rf"(?:(?:{_AUTHORIZATION_PREFIXED}(?!{_AUTHORIZATION_LOWER}){_AUTHORIZATION_NAME}"
            rf"|{_AUTHORIZATION_CAMEL}){_ESCQ}{_SEP}{_GAP}{_QUOTE}"
            rf"(?!{_AUTH_SCHEMES}\b|Digest\b|{_AUTH_SCHEMES}{_QUOTE})"
            rf"(?={_ENTRY_TEXT}{_QUOTE}"
            r"(?=[\s,;:.)\]}\\\"'`]|\Z))(?P<entry>)"
            rf"|{_AUTHORIZATION_SPOKEN}{_ESCQ}[ \t]*+[:=]{_WS}*+{_ESCQ}"
            rf"(?!{_AUTH_SCHEMES}\b|Digest\b|[rRuUbBfF]{{1,3}}[\"']){_SPOKEN_FIRST}(?P<words>)"
            rf"|(?P<code>{_AUTHORIZATION_CODE_NAME}){_ESCQ}(?:=|%(?:25)?3[Dd])"
            rf"(?:[ \t]|(?-i:\\{{1,8}}t(?=[^\s\"'\\]))|%(?:25)?(?:20|09))*+{_SCHEME}"
            rf"|(?P<code_text>{_AUTHORIZATION_CODE_NAME}){_ESCQ}{_SEP}(?:[ \t]|(?-i:\\{{1,8}}t(?=[^\s\"'\\]))|%(?:25)?(?:20|09))*+{_SCHEME}"
            rf"|{_AUTHORIZATION_ONE_WORD}{_ESCQ}{_SEP}{_GAP}{_ESCQ}{_AUTHORIZATION_AHEAD_ONE}"
            rf"(?!{_AUTH_SCHEMES}\b|Digest\b|[rRuUbBfF]{{1,3}}[\"'])))"
            rf"(?(entry){_ENTRY_VALUE}|(?(words){_SPOKEN_WORDS}"
            rf"|(?(code){_code_value('1')}{_BRACE_TAIL}{_code_glue('1')}{_code_more('1')}"
            rf"|(?(code_text){_code_value('2')}{_BRACE_TAIL}{_code_glue('2')}{_code_more('2')}"
            rf"|{_ONE_WORD_VALUE}))))"
            # Else any other name that holds the header's word (_AUTHORIZATION_ANY_NAME).
            rf"|(?=[{_AUTHORIZATION_STARTS}])(?P<any>{_AUTHORIZATION_ANY_NAME})"
            rf"(?:{_SHELL_CLOSE})?{_NAME_JUNK}{_TAB_GAP}(?:[:=]|%(?:25)?3[AaDd])(?:%20|\+)?+{_SEP_GAP}"
            rf"(?![rRuUbBfF]{{0,3}}\\{{0,7}}[\"']?(?:{_AUTH_SCHEMES}|Digest)\b)"
            rf"(?:(?P<q>\\{{0,7}}[\"'`]|{_QUOTE_ENTITY}|\\{{1,8}}u0022)(?P<inner>[^\r\n]{{0,4096}}?)(?P=q)"
            rf"|(?P<run>(?![\[{{(](?:\s|\\[nrt]|\Z))(?:{_CODE_RUN}|{_any_quoted_value()}"
            rf"|{_QUOTE_ENTITY}(?:(?!{_QUOTE_ENTITY})[^\r\n]){{1,4096}}+)))",
            re.IGNORECASE,
            words=_AUTHORIZATION_WORDS,
        ),
        _authorization_value,
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # A camel-case code Authorization name in running text (basicAuthorization,
    # httpAuthorization, self.sourceAuthorization), which the carrier above
    # reads only quoted: every value under it is masked, read as a code name's
    # (the lead's ruling (b), 2026-11-15 01:3xZ). It starts at the header's
    # word after a letter or a digit, and the name is read back from there
    # (_name_before); a quoted name is the entry rule's. The header's own name
    # (proxyAuthorization, `\\u2019Authorization` in a JSON copy) keeps one
    # prose word (_code_kind). A value after an escaped line break is the
    # next carrier's (every name ending in `authorization`, below), read
    # across the break.
    (
        _LazyPattern(
            rf"((?<=[A-Za-z0-9])(?P<code_text>(?-i:Authorization))(?P<camel>)(?![A-Za-z0-9_]){_SEP}"
            rf"(?:[ \t]|(?-i:\\{{1,8}}t(?=[^\s\"'\\]))|%(?:25)?(?:20|09))*+{_SCHEME})"
            rf"{_code_value('2')}{_code_glue('2')}{_code_more('2')}",
            re.IGNORECASE,
        ),
        _authorization_value,
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # Prose under a code Authorization name with a capital or a camel hump
    # (`note_Authorization: none`, `"basicAuthorization": "required"`), which
    # the carrier above keeps as prose after any prefixed name: masked (the
    # lead's ruling (b), 2026-11-15 01:3xZ). A pass of its own, so a name it
    # gives back (HTTP_AUTHORIZATION=none, proxy_authorization: required) costs
    # the carrier above nothing; what it reads is one prose word. A scheme word
    # with a token after it is no prose: the two are masked together
    # (`"\x70roxyAuthorization"="Bearer <tok>"`, F2b 2026-11-16 04:08Z).
    (
        _LazyPattern(
            r"((?<=[A-Za-z0-9])(?P<joint>[_.]?)(?-i:[Aa]uthorization|AUTHORIZATION)"
            r"(?![A-Za-z0-9_])"
            rf"(?:\\{{0,8}}[\"']{_SEP}{_GAP}{_QUOTE}|{_SEP}{_GAP}))"
            # A scheme with its token after it is no prose: masked with its token
            # here; a scheme word is never read as prose either, so the scheme rules
            # read it (`xAuthorization=\nBearer <token>`, the scheme on the next line).
            rf"(?:(?:{_AUTH_SCHEMES}|Digest)[ \t]++{_SCHEME_TOKEN}"
            rf"|(?!(?:{_AUTH_SCHEMES}|Digest)\b){_PROSE}|(?:{_AUTH_SCHEMES}|Digest)\b"
            r"(?=[ \t]*+(?:[\r\n\"',;)\]}\\]|\Z)))",
            re.IGNORECASE,
        ),
        _code_name_prose,
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # Every name that ends in `authorization` is a carrier (the lead,
    # 2026-11-16 10:53Z via The rc twin of rel's F4), whatever its value:
    # a prefix of any kind (`http_authorization`, `my_authorization`), glued
    # (`proxyAuthorization`, `PROXYAUTHORIZATION`), camel
    # (`basicAuthorization`), `BEARER_TOKEN_AUTHORIZATION`, a flag's name
    # (`has_authorization`, `requires_authorization`), its value on its own
    # line or the next. This withdraws R192C's B-4, the lead's ruling (b) and
    # the flag and prose rulings for these names only; the header's own names
    # keep the carriers above. A flag's boolean, number or null
    # (`has_authorization: true`, `requires_authorization=0`, `None`; `none` in
    # lower case is prose) is no credential and stays shown; so is an escape's
    # tail: a name never starts inside an escape (`\\nAuthorization: required`
    # and `%3AAuthorization` are the header's own name; `\\nhas_authorization=1`
    # is has_authorization's), and a
    # code string after the separator (`f"Bearer {token}"`) or an object, an
    # array or a call opened at a line's end (`"DescribeEndpointAuthorization": {`); so is a colour code's tail glued to the
    # header's word (`\x1b[32mAuthorization: required`, `\x1b[2 q...`,
    # `%1B%5B32m...`: the header's own name after a code, as _NOT_AFTER_CSI
    # reads one). A pass of its own, after the carriers
    # above, so it reads only what they left; the name is a bounded run
    # before the word.
    (
        _LazyPattern(
            rf"{_KEY_START}(?<!%[0-9A-Fa-f]{{2}})(?<!%25[0-9A-Fa-f]{{2}})"
            rf"({_ESCQ}(?<!%[0-9A-Fa-f]{{2}})(?<!%25[0-9A-Fa-f]{{2}})"
            r"(?!(?:proxy[_-])?authorization(?![A-Za-z0-9_-]))"
            r"(?!(?:(?<=\[)[0-9;:?]{0,48}|(?<=[;:?])[0-9;:?]{1,48})[ -/]?[A-Za-z][A-Za-z0-9_-])"
            rf"(?![A-Za-z](?!{_NOT_AFTER_CSI})[A-Za-z0-9_-])"
            r"(?!(?<=%)(?:25)*+5[Bb][0-9;:?]{0,48}[ -/]?[A-Za-z](?:proxy[_-]?)?authorization(?![A-Za-z0-9_-]))"
            r"(?!(?:(?<=%)(?:25)*+[0-9A-Fa-f]{2}|(?<=\\)(?:u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2}"
            r"|U[0-9A-Fa-f]{8}|[0-7]{1,3}|[nrtbfv]))[A-Za-z0-9_-])"
            r"(?!(?:(?<=%5[Bb])|(?<=%255[Bb]))[0-9;:?]{0,48}[ -/]?[A-Za-z][A-Za-z0-9_-])"
            r"[A-Za-z0-9_-]{0,63}?authorization(?![A-Za-z0-9_-])"
            rf"{_ESCQ}{_SEP}{_GAP}(?![rRuUbBfF]{{1,3}}[\"']|[\[{{(](?:\s|\\{{1,8}}[nrt]|\Z)){_ESCQ}"
            # A scheme word alone (`HTTP_AUTHORIZATION: Bearer`) stays, as
            # under the header's own names.
            rf"(?!(?:{_AUTH_SCHEMES}|Digest)(?![A-Za-z0-9_-])"
            rf"(?!{_WS}*+(?:{_NEWLINE}{_INDENT}*+)?[A-Za-z0-9._~+/=-]))"
            rf"(?:{_AUTH_SCHEMES}{_WS}++)?+){_SECVAL}",
            re.IGNORECASE,
            words=_AUTHORIZATION_WORDS,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # A credential's name with nothing after its `:` or `=`, then on the next
    # line (a blank line between too) an entry whose name is padded-base64
    # shaped and whose value is quoted (`API_KEY:\n  abcdefghij2k='<v>'`,
    # `Authorization:\n  noneauthorization2="<v>"`): that line is the
    # credential's value, and its quoted value is masked, as 0.9.3 masks it.
    # The entry stays a name of its own (_B64_PADDED), so under any other
    # name it is read as one. The name above decides as on its own line (a
    # quoted value there masked or kept: _next_line_entry_value); a pass of
    # its own, after every carrier.
    (
        _LazyPattern(
            r"(?<=[A-Za-z0-9_.\"';-])[ \t]*+(?P<sep>[:=])[ \t]*+"
            rf"(?:{_NEWLINE}[ \t]*+)?{_NEWLINE}{_INDENT}++"
            r"(?P<inner>(?=[A-Za-z]{0,63}[0-9+/])[A-Za-z0-9+/]{12,64}={1,2}+)"
            r"(?P<q>\\{0,8}[\"'`])(?!\*\*\*REDACTED\*\*\*)"
            r"(?P<v>(?:[^\"'`\\\r\n]|\\(?![\"'`\\]))++)(?=(?P=q))"
        ),
        _next_line_entry_value,
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
]

# A whole name (no length cap: these only ever see one name). It needs no
# price lookbehinds: is_secret_field asks the classifier first, and a name
# whose TOKEN they would refuse also ends in TOKEN with no `_` or `-` once
# the underscores are dropped, which is matched anyway.
_SECRET_FIELD = _LazyPattern(
    rf"[A-Za-z0-9_.-]*(?:{_API_KEY_NAME}|{_KEY_FIELD_NAME}|TOKEN|{_SECRET_NAME_ENDINGS})"
    rf"{_NAME_NUMBER}"
    r"|[A-Za-z0-9_.-]*BEARER_TOKEN[A-Za-z0-9_.-]*",
    re.IGNORECASE,
)
_PASS_FIELD = re.compile(rf"({_PASS_NAME})(?:{_PASS_NUMBER})?")
_PASSPHRASE_FIELD = re.compile(rf"(?:passphrase|(?:^|_)pass_phrase){_NAME_NUMBER}$")


def _snake_name(name: str) -> str:
    """*name* in lower snake case, split as the classifier splits it (its
    name_words, the one splitter: a run of digits is a word of its own):
    `privateKey`, `Proxy-Authorization`, `HTTP_COOKIE` and `http2cookie` as
    `private_key`, `proxy_authorization`, `http_cookie` and `http_2_cookie`;
    `ByPass` as `bypass`, one word."""
    return "_".join(name_words(name))


def is_secret_field(name: str, value: Any = None) -> bool:
    """Whether a field called *name* holding *value* carries a secret.

    The carriers' rules applied to a whole name, as written and in snake
    case (``privateKey``): the export redactor asks this of a JSON key or an
    option (``publish/redact.py``), ``redact_trajectory_obj`` of the key of a
    number, and config.json's env filter of an env name. Case is ignored. A
    name with the word ``per`` (a price or not), a name with the word
    ``pass`` or ``passvalue`` in any case (a test's result, a metric or a
    password: ``TOTAL_PASS``, ``total_pass``, ``dbPass``, ``passvalue``), and
    a passphrase's (a flag's, ``has_passphrase``, holds a switch as no
    secret), are the classifier's (``entry_kind``, which reads the value
    too: ``TOTAL_PASS`` holds a count or is a password's). A ``*_PWD`` name
    is one in any case (``MYSQL_PWD``, ``mysql_pwd``, ``mysqlPwd``; not
    ``PWD`` or ``OLDPWD``). A name that ends in a Cookie or Authorization
    header's word is the classifier's too: the header's own name
    (``Cookie``, ``Set-Cookie``, ``Authorization``) carries one whatever it
    holds, one after a prefix (``HTTP_COOKIE``, ``httpCookie``,
    ``has_authorization``, ``http2cookie``) unless it holds prose (its
    _PROSE: ``required``, ``1``, ``true``); ``None`` is the value ``null``.
    A header's url, name or policy (``authorization_url``,
    ``session_cookie_name``) ends in no header word and carries none.

    Every other name the redactor reads is the classifier's too (the
    lead's ruling, part B, 2026-11-14): the carriers' names
    (_SECRET_FIELD, a ``*_PWD`` name, a passphrase's) and the names it
    decides by shape (_decided_by_name: a numbered name, KEY_2, TOKEN_V2;
    a hint, id_token_hint; a primary or a secondary key's, by its value).
    ``entry_kind`` answers for each; where it has no answer (a run-together
    name it does not split, ACCOUNTKEY, defaultazurecredential) a carrier's
    name is a secret's, as before, until it has one. GOOGLE_APPLICATION_
    CREDENTIALS holds a path as a setting and inline JSON, a mapping or a
    list as a credential, by the classifier's value rule. The bare words
    ``key``, ``session`` and ``tokens`` are read by no rule here.
    """
    snake = _snake_name(name)
    words = name_words(name)
    if (
        "per" in words
        or "pass" in words
        or "passvalue" in words
        or words[-1:] == ["passphrase"]
        or words[-2:] == ["pass", "phrase"]
        or (bool(words) and words[-1] in _HEADER_WORDS)
    ):
        return entry_kind(name, value) == "credential"
    # As written, in snake case, and with the underscores dropped
    # (`db_pass_word` is a password's name as `dbpassword` is). A passphrase's
    # name is read in snake case only, so that `compass_phrase` stays.
    named = (
        any(_SECRET_FIELD.fullmatch(c) for c in (name, snake, snake.replace("_", "")))
        or bool(_PASSPHRASE_FIELD.search(snake))
        or bool(_PASS_FIELD.fullmatch(snake.upper()))
    )
    if not named and not _decided_by_name(words):
        return False
    kind = entry_kind(name, value)
    if kind is None:
        return named
    return kind == "credential"


def is_secret_number(name: str, value: Any) -> bool:
    """Whether a number, a boolean or null *value* under the field *name* is
    masked, in trajectory files and at export alike: only an integer of four
    digits or more (a PIN's length; the sign does not count) under a
    credential's name, by the classifier the override record uses
    (``name_kind``). A boolean, a float and null hold no secret, and a flag,
    a price, a count or a tokenizer's token id keeps its type
    (``eos_token: 2``, ``max_token: 4096``). Whatever ``name_kind`` says,
    never under a name with the word ``tokens`` (a count of them:
    ``cache_read_tokens``, ``tokens_prompt``), one that ends in ``id`` or
    ``ids``, or ``primary_key`` or ``foreign_key``: cost and token
    accounting read them back. A price (``usd_per_token``,
    ``code_interpreter_cost_per_session``) and a test's result
    (``FAIL_TO_PASS``) are settings to ``name_kind``."""
    if isinstance(value, bool) or not isinstance(value, int) or abs(value) < 1000:
        return False
    snake = _snake_name(name)
    words = snake.split("_")
    if (
        "tokens" in words
        or words[-1] in ("id", "ids")
        or snake in ("primary_key", "foreign_key")
    ):
        return False
    return entry_kind(name, value) == "credential"


_JUDGE_FAMILY_PATTERNS = [
    (
        _LazyPattern(_IDENT_START + "(?:" + pattern.pattern + ")", pattern.flags),
        replacement,
    )
    for pattern, replacement, _category in _KEY_FAMILY_PATTERNS
]


def redact_key_families(text: str) -> str:
    """Mask only the keys with a shape of their own, as
    :func:`redact_trajectory_text` masks them: provider-prefixed keys, JWTs
    and private-key blocks, and no ``name: value`` carrier or URL
    credential, and only where an identifier starts (``_IDENT_START``:
    ``build_dtn_network_topology_config`` holds no key). For text whose own
    carriers must stay readable: what a task judge is shown of the
    submission (``taskmd.judging.Scrub.evidence``).
    """
    for pattern, replacement in _JUDGE_FAMILY_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_trajectory_text(text: str) -> str:
    """Apply all secret-redaction patterns to *text*.

    Token families (Anthropic/OpenAI/Google/AWS/Daytona/GitHub/Slack) are
    redacted whole, prefix included, so the secret-leak audit greps see no
    live-key shape. Header/key-value carriers keep the field name but drop the
    value, in both JSON (``"x-api-key": "v"``) and raw-text (``x-api-key: v``)
    forms; this includes generic ``*TOKEN*``/``*SECRET*`` carriers so a
    ``GITHUB_TOKEN=...`` env dump is scrubbed even without a known prefix.
    """
    return redact_trajectory_text_with_count(text)[0]


# Each template of the table, as _prefix_keeper writes it.
_PREFIX_KEEPERS = {
    replacement: _prefix_keeper(replacement)
    for _pattern, replacement, _category in _REDACTION_PATTERNS
}


def _subn_changed(
    pattern: Any, decide: Callable[[re.Match[str]], str], text: str
) -> tuple[str, int]:
    """*pattern*'s matches replaced by *decide*, counting only those it
    changes: a match it keeps is no replacement (the export redactor runs
    until a pass counts none)."""
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        out = decide(match)
        changed += out != match[0]
        return out

    token = _classifier_answers.set({}) if _classifier_answers.get() is None else None
    try:
        return pattern.subn(replace, text)[0], changed
    finally:
        if token is not None:
            _classifier_answers.reset(token)


def redact_trajectory_text_with_count(text: str) -> tuple[str, int]:
    """Apply the canonical patterns and report how many matches were replaced."""
    text, categories = redact_trajectory_text_with_categories(text)
    return text, sum(categories.values())


# one mechanism for escaped text. A
# value written with JSON string escapes (`=`, `\"`, `\\`, `\t`, nested
# as a JSON copy of a JSON copy writes them) or with HTML entities
# (`&quot;`, `&#34;`, `&#x22;`) is the same text to a reader, but the
# patterns read only what is written. So after its own pass the redactor
# builds a decoded view of the text, every escape and entity decoded for
# matching only (to _VIEW_DEPTH levels), with a map from each view character
# to the span of the text it came from; runs every pattern on the view as it
# runs them on the text; and masks, in the text, the spans the view's masks
# map back to. The output keeps its own escaping, and a bare quote (a JSON
# string's own) is never masked. Only masks are added. Linear: each level is
# one pass, and the view is redacted once.
# the views also decode the escapes
# other languages and URLs write: a percent escape (`%73`; a nest of `%25`
# wrappers, `%252573`, is decoded whole, as decoding it again and again
# would), `\x73`, an octal `\163`, JavaScript's `\u{73}`, Perl's `\x{73}`,
# Python's `\U00000073`, and a numeric entity without its `;` (`&#115`). Each
# level decodes every family at once, so the levels reach a fixed point
# across families (`%5Cx73` is `\x73` one level down, then `s`). Every
# percent escape is decoded, an encoded separator too (the lead's ruling (A),
# 13:18Z): `Authorization%3A header to ...` reads as its literal form does.
# A `%25` nest is read once from its `%`, and a braced code point's digits
# once: linear.
_VIEW_ESCAPE = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"
    r"|\\u([0-9a-fA-F]{4})"
    r"|\\([\"\\/bfnrt])"
    r"|&(quot|apos|lt|gt|amp);"
    r"|&#([0-9]{1,7});?"
    r"|&#[xX]([0-9a-fA-F]{1,6});?"
    r"|%(?:25)*([0-9a-fA-F]{2})"
    r"|\\x([0-9a-fA-F]{2})"
    r"|\\([0-3][0-7]{2})"
    r"|\\u\{([0-9a-fA-F]++)\}"
    r"|\\x\{([0-9a-fA-F]++)\}"
    r"|\\U([0-9a-fA-F]{8})"
)
# its decode set, as reads it (JSON escapes and HTML entities).
# Every text is read with it first, unchanged; its set (above) reads the
# text again only where it holds one of its escapes (_MORE_ESCAPE), and
# the two views' masks are joined (_escape_flags_whole). Decoding more at a
# level can hide a form a level of its reads (`name\u001b[0m%22%3D%22v`:
# with only `\u001b` decoded, the main pass's encoded-separator rules read the
# value), so only adds masks to 's.
_VIEW_ESCAPE_616 = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"
    r"|\\u([0-9a-fA-F]{4})"
    r"|\\([\"\\/bfnrt])"
    r"|&(quot|apos|lt|gt|amp);"
    r"|&#([0-9]{1,7});"
    r"|&#[xX]([0-9a-fA-F]{1,6});"
)
# An escape only its set reads, or a backslash it decodes in a string's
# frame (_opens_escape: `\\u0061`, `\u005cu0061`).
_MORE_ESCAPE = re.compile(
    r"%[0-9a-fA-F]{2}|\\(?:x|U|[0-3][0-7]{2}|u\{|\\u[0-9a-fA-F]{4}|u005[cC])"
    r"|&#[0-9]{1,7}(?![0-9;])|&#[xX][0-9a-fA-F]{1,6}(?![0-9a-fA-F;])"
)
_en616_set: ContextVar[bool] = ContextVar("_en616_set", default=False)
_VIEW_SIMPLE = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_VIEW_ENTITIES = {"quot": '"', "apos": "'", "lt": "<", "gt": ">", "amp": "&"}
_VIEW_DEPTH = 4
_MASK = "***REDACTED***"


def _view_char(m: re.Match[str]) -> str | None:
    """The character an escape or entity writes, or None (not one)."""
    if m[1] is not None:
        high, low = int(m[1], 16), int(m[2], 16)
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    if m[3] is not None:
        return chr(int(m[3], 16))
    if m[4] is not None:
        return _VIEW_SIMPLE[m[4]]
    if m[5] is not None:
        return _VIEW_ENTITIES[m[5]]
    group = m.lastindex or 0
    written = m[group]
    if group == 6:
        code = int(written, 10)
    elif group in (7, 8, 9, 13):
        code = int(written, 16)
    elif group == 10:
        code = int(written, 8)
    else:
        digits = written.lstrip("0")
        if len(digits) > 6:
            return None
        code = int(digits or "0", 16)
    return chr(code) if code <= 0x10FFFF else None


class _ViewMap:
    """Where each character of a derived text came from in its source, as
    segments: a run copied one for one, an escape decoded to one character,
    or a separator that came from nothing. One segment per run and per
    escape, never one per character."""

    __slots__ = ("at", "cuts", "end", "src")

    def __init__(self) -> None:
        self.at: list[int] = []  # each segment's first index in the derived text
        self.src: list[int] = []  # its first index in the source; -1 for a cut
        self.end: list[int] = []  # an escape's end in the source; -1 for a run
        self.cuts: list[int] = []  # the indices of the cut segments

    def run(self, at: int, src: int) -> None:
        self.at.append(at)
        self.src.append(src)
        self.end.append(-1)

    def escape(self, at: int, src: int, end: int) -> None:
        self.at.append(at)
        self.src.append(src)
        self.end.append(end)

    def cut(self, at: int) -> None:
        self.cuts.append(len(self.at))
        self.at.append(at)
        self.src.append(-1)
        self.end.append(-1)

    def spans(self, start: int, stop: int) -> list[tuple[int, int]]:
        """The source spans the derived characters [start, stop) came from:
        one, or one per stretch between cuts."""
        first = bisect.bisect_right(self.at, start) - 1
        last = bisect.bisect_right(self.at, stop - 1) - 1
        out: list[tuple[int, int]] = []
        low = bisect.bisect_left(self.cuts, first)
        segment, position = first, start
        while True:
            cut = self.cuts[low] if low < len(self.cuts) else -1
            if cut < 0 or cut > last:
                self._add(out, segment, position, last, stop - 1)
                return out
            if cut > segment:
                self._add(out, segment, position, cut - 1, self.at[cut] - 1)
            low += 1
            segment = cut + 1
            if segment > last:
                return out
            position = self.at[segment]

    def _add(
        self, out: list[tuple[int, int]], first: int, start: int, last: int, final: int
    ) -> None:
        if self.src[first] < 0:
            return
        begin = self.src[first] + (start - self.at[first] if self.end[first] < 0 else 0)
        if self.end[last] < 0:
            finish = self.src[last] + (final - self.at[last]) + 1
        else:
            finish = self.end[last]
        if finish > begin:
            out.append((begin, finish))


def _opens_escape(text: str, end: int, keep: str) -> bool:
    """A backslash decoded just before *end* starts an escape that writes
    none of *keep* (`%5Cu{61}` or `\\\\u0061` in a string's frame is
    `\\u{61}` one level down, then `a`). Such a backslash is decoded even in
    the frame, where a quote, backslash or line break would stay as written:
    with what follows it, it writes neither."""
    if _en616_set.get():
        return False
    after = _VIEW_ESCAPE.match("\\" + text[end : end + 24])
    if after is None:
        return False
    char = _VIEW_SIMPLE[after[4]] if after.lastindex == 4 else _view_char(after)
    return char is not None and char not in keep


def _decode_once(
    text: str, kept: bytearray | None = None, keep: str = '"\\\n\r'
) -> tuple[str, _ViewMap, bytearray | None]:
    """*text* with one level of escapes and entities decoded, a map of where
    each character came from, and *kept* carried over. Where *kept* marks a
    character, an escape there that writes one of *keep* (a quote, a
    backslash or a line break) stays as written (a string's own escapes, in
    the frame: decoded, they would end the string or its line)."""
    _charge_view(len(text) * _VIEW_DECODE_NS * 1e-9, "decode")
    out: list[str] = []
    view_map = _ViewMap()
    # The map's segments, appended in place (one run or escape each).
    seg_at, seg_src, seg_end = view_map.at, view_map.src, view_map.end
    new_kept = bytearray() if kept is not None else None
    last = 0
    at = 0
    escapes = _VIEW_ESCAPE_616 if _en616_set.get() else _VIEW_ESCAPE
    for m in escapes.finditer(text):
        # A simple escape (`\"`, `\n`) is the common case: its group is 4.
        char = _VIEW_SIMPLE[m[4]] if m.lastindex == 4 else _view_char(m)
        start = m.start()
        if char is None or (
            kept is not None
            and kept[start]
            and char in keep
            and not (char == "\\" and _opens_escape(text, m.end(), keep))
        ):
            continue
        if start > last:
            out.append(text[last:start])
            seg_at.append(at)
            seg_src.append(last)
            seg_end.append(-1)
            at += start - last
            if new_kept is not None and kept is not None:
                new_kept += kept[last:start]
        out.append(char)
        last = m.end()
        seg_at.append(at)
        seg_src.append(start)
        seg_end.append(last)
        at += 1
        if new_kept is not None and kept is not None:
            new_kept.append(kept[start])
    if last == 0:
        return text, view_map, kept
    if last < len(text):
        out.append(text[last:])
        view_map.run(at, last)
        if new_kept is not None and kept is not None:
            new_kept += kept[last:]
    return "".join(out), view_map, new_kept


@functools.cache
def _view_table() -> tuple[tuple[Any, Any, bool, bool, str, bool], ...]:
    """The redaction table as _redaction_pass reads it: each pattern, its
    replacement (a prefix keeper for a lazy pattern), whether it has carrier
    words, whether it ignores case, its category, and whether every match
    counts (a template's) or only a changed one (a function's). Built once."""
    rows = []
    for pattern, replacement, category in _REDACTION_PATTERNS:
        every = isinstance(replacement, str)
        if isinstance(pattern, _LazyPattern):
            rows.append(
                (
                    pattern,
                    _PREFIX_KEEPERS.get(replacement, replacement),
                    bool(pattern.words) or isinstance(pattern, _GatedPattern),
                    bool(pattern.flags & re.IGNORECASE),
                    category,
                    every,
                )
            )
        else:
            rows.append((pattern, replacement, False, False, category, every))
    return tuple(rows)


# One redaction's view passes, by (view, rows): two paths that reach the same
# view text read it once (a copy of a copy decodes to the same text from the
# top and from a level down). Set for the length of one _escape_flags call.
# a caller with rules of its own reads
# the same views (publish's, publish/redact.py: the layer that leaves our hands
# reads no less than the others). While set, _redaction_pass runs these rows
# in place of the table (view_flags).
_view_rows: ContextVar[tuple[tuple[Any, Any, bool, bool, str, bool], ...] | None] = (
    ContextVar("_view_rows", default=None)
)


def view_flags(
    text: str,
    rows: Iterable[tuple[Any, str | Callable[[re.Match[str]], str], str]],
) -> bytearray | None:
    """The characters of *text*, as written, that *rows* (a pattern, its
    replacement, its category) mask in the decoded views the table reads
    (_escape_flags: the gate, the decoder, the frames and literals and the
    quote view), less every quote and earlier mask; None where there are
    none."""
    token = _view_rows.set(tuple((p, r, False, False, c, False) for p, r, c in rows))
    try:
        return _escape_flags(text)
    finally:
        _view_rows.reset(token)


_view_cache: ContextVar[dict[tuple[str, frozenset[int] | None], bytes] | None] = (
    ContextVar("_view_cache", default=None)
)


def _view_masks(view: str, rows: frozenset[int] | None = None) -> bytes:
    """The characters of *view* the patterns mask (_redaction_pass), every
    pattern or those of the table at *rows*."""
    cache = _view_cache.get()
    if cache is None:
        return bytes(_redaction_pass(view, rows)[2])
    masks = cache.get((view, rows))
    if masks is None:
        masks = cache[view, rows] = bytes(_redaction_pass(view, rows)[2])
    return masks


def _redaction_pass(
    text: str, rows: frozenset[int] | None = None
) -> tuple[str, list[int], bytearray, dict[str, int]]:
    """Every pattern run on *text* as redact_trajectory_text_with_categories
    runs it, tracked: the result; each result character's index in *text*
    (-1 for a mask's); the characters of *text* masked (a replacement masks
    the part of its match between what it keeps at either end); the count
    per category."""
    current = text
    origin = list(range(len(text)))
    masked = bytearray(len(text))
    categories: dict[str, int] = {}
    folded: str | None = None
    fold = True
    token = _classifier_answers.set({}) if _classifier_answers.get() is None else None
    own = _view_rows.get()
    try:
        for index, (pattern, repl, words, icase, category, every) in enumerate(
            _view_table() if own is None else own
        ):
            if rows is not None and own is None and index not in rows:
                continue
            if words:
                if fold and icase:
                    folded, fold = _folded(current), False
                if not pattern.may_match(current, folded):
                    continue
            _charge_view(
                len(current) * _VIEW_ROW_NS[index] * _VIEW_OVERHEAD * 1e-9, "row"
            )
            compiled = (
                pattern.compiled() if isinstance(pattern, _LazyPattern) else pattern
            )
            pieces: list[str] = []
            new_origin: list[int] = []
            last = 0
            count = 0
            for m in compiled.finditer(current):
                out = m.expand(repl) if isinstance(repl, str) else repl(m)
                matched = m[0]
                if out == matched:
                    count += every
                    continue
                count += 1
                width = min(len(matched), len(out))
                head = 0
                while head < width and matched[head] == out[head]:
                    head += 1
                tail = 0
                while tail < width - head and matched[-1 - tail] == out[-1 - tail]:
                    tail += 1
                # What the replacement keeps ends where its mask starts: a value of
                # `*` matches the mask's own stars and is masked all the same.
                at = out.find(_MASK)
                if at >= 0:
                    head = min(head, at)
                    tail = min(tail, len(out) - out.rfind(_MASK) - len(_MASK))
                # The characters the match masks, as one slice of *text*: the
                # first and last that came from it bound the rest, since what
                # is missing between them was masked by an earlier pattern.
                low, high = m.start() + head, m.end() - tail - 1
                while low <= high and origin[low] < 0:
                    low += 1
                while high >= low and origin[high] < 0:
                    high -= 1
                if high >= low:
                    masked[origin[low] : origin[high] + 1] = b"\x01" * (
                        origin[high] + 1 - origin[low]
                    )
                pieces.append(current[last : m.start()])
                new_origin.extend(origin[last : m.start()])
                pieces.append(out)
                new_origin.extend(origin[m.start() : m.start() + head])
                new_origin.extend([-1] * (len(out) - head - tail))
                new_origin.extend(origin[m.end() - tail : m.end()])
                last = m.end()
            if count:
                categories[category] = categories.get(category, 0) + count
                _charge_view(count * _VIEW_MATCH_NS * 1e-9, "match")
            if pieces:
                pieces.append(current[last:])
                new_origin.extend(origin[last:])
                current = "".join(pieces)
                origin = new_origin
                fold = True
    finally:
        if token is not None:
            _classifier_answers.reset(token)
    return current, origin, masked, categories


# Where a view is built. At every level, JSON string literals are found
# (bare quotes paired in order on one line; a JSON string holds no raw line
# break). Each literal's content, decoded, is a view of its own: it reads as
# the text it was written from (a JSON copy's `"\"has_authorization\": \"1\""`
# is an entry inside one string). The whole text is a view too, the frame:
# the text between literals decoded in context, with the literals in place
# and their quote, backslash and line-break escapes as written, so their
# quotes stay delimiters and their lines stay whole (`out\nNAME=["a", "b"]`
# reads as the line it is). Each view is read again the same way for the
# next level; a literal the frame keeps is not viewed twice.
# A literal: a bare quote (one an even run of backslashes leaves unescaped),
# the content on its line, and the next bare quote. A string's own escapes are
# taken two characters at a time, so an escaped quote never closes it.
_LITERAL = re.compile(r'(?<!\\)(?:\\\\)*"((?:[^"\\\n]|\\[^\n])*+)"')
_HAS_ESCAPE_616 = re.compile(
    r"\\[\"\\/bfnrtu]|&(?:quot|apos|lt|gt|amp|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});"
)
_HAS_ESCAPE = re.compile(
    r"\\(?:[\"\\/bfnrtuxU]|[0-3][0-7]{2})|%[0-9a-fA-F]{2}"
    r"|&(?:quot|apos|lt|gt|amp);|&#[0-9]{1,7}|&#[xX][0-9a-fA-F]{1,6}"
)


# No pattern reads a view without two letters or digits in a row: every
# carrier's name has letters, and every key family's body is a run of them.
_VIEW_WORTH = re.compile(r"[A-Za-z0-9]{2}")


def _mark(
    flags: bytearray,
    view_masks: bytes | bytearray,
    maps: tuple[_ViewMap, ...],
    base: int = 0,
) -> None:
    """Mark in *flags* (from *base*) the source spans that *view_masks*' runs
    came from, through *maps* in turn (the innermost first)."""
    size = len(view_masks)
    index = view_masks.find(1)
    while index >= 0:
        end = view_masks.find(0, index)
        end = size if end < 0 else end
        spans = [(index, end)]
        for view_map in maps:
            spans = [pair for a, b in spans for pair in view_map.spans(a, b)]
        for begin, finish in spans:
            flags[base + begin : base + finish] = b"\x01" * (finish - begin)
        index = view_masks.find(1, end)


# in a view: the `sk-` of the word `task-notification(s)`, where the
# word starts as _TAG_WORD's does. `subtask-notification` is another name: the
# sk- rule masks it as a key, as 0.9.2 does, and so does every view. Lazy, as
# every pattern that holds the boundary is; the literal comes first.
_TAG_KEY = _LazyPattern(rf"sk-(?<={_WORD_START}task-)notifications?")
# Where what follows the word starts: after `sk-notification`, so that an `s`
# may be the word's (`task-notifications`) or a glued key's (`sk-proj-`).
_TAG_KEY_WORD = len("sk-notification")


def _tag_word_trim(view: str, masks: bytes, reread: Callable[[str], bytes]) -> bytes:
    """*masks* with a key mask that starts at the `sk-` of `task-notification`
    cut at the word's end. An escape that continues the word in the text
    makes it another name, and the text's own pass masks it as 0.9.2 does:
    `sk-notification`, up to the escape. The view reads the escape
    as the character it writes, and its mask would run on over the escape's
    characters; it stops where 0.9.2's does. What follows the word keeps
    the masks it gets read on its own (*reread*, the view's reading of a
    text): a key glued right after the word (`task-notificationsk-proj-...`)
    is masked as it is anywhere else."""
    trimmed: bytearray | None = None
    for m in _TAG_KEY.compiled().finditer(view):
        start, end = m.span()
        if (
            end >= len(masks)
            or not masks[start]
            or not masks[end]
            or (start and masks[start - 1])
        ):
            continue
        stop = masks.find(0, end)
        stop = len(masks) if stop < 0 else stop
        if trimmed is None:
            trimmed = bytearray(masks)
        tail = start + _TAG_KEY_WORD
        trimmed[end:stop] = reread(view[tail:stop])[end - tail :]
    return masks if trimmed is None else bytes(trimmed)


def _view_read(view: str, depth: int, done: bytearray | None) -> bytes:
    """What *view* masks, by every pattern (_view_masks) and by its own views
    one level down (_escape_masks)."""
    own = _view_masks(view) if _VIEW_WORTH.search(view) else bytes(len(view))
    deeper = _escape_masks(view, depth + 1, done)
    return _tag_word_trim(
        view,
        (int.from_bytes(own, "big") | int.from_bytes(deeper, "big")).to_bytes(
            len(view), "big"
        ),
        lambda tail: _view_read(tail, depth, None),
    )


def _view_into(
    flags: bytearray,
    text: str,
    base: int,
    depth: int,
    kept: bytearray | None,
    keep: str = '"\\\n\r',
) -> None:
    """Mark in *flags* (from *base*) what *text*'s view one level down masks,
    and what that view's own views mask, to _VIEW_DEPTH."""
    view, view_map, view_kept = _decode_once(text, kept, keep)
    if view is text:
        return
    _mark(flags, _view_read(view, depth, view_kept), (view_map,), base)


# Literal views are read as one text, joined by a line no pattern reads
# across (a NUL alone on its line): one decode and one pass for all of a
# level's literals. No escape spans a separator.
_VIEW_SEP = "\n\x00\n"


def _view_batch(
    flags: bytearray,
    text: str,
    spans: list[tuple[int, int]],
    depth: int,
) -> None:
    """Mark in *flags* what the views of *text*'s literal contents at *spans*
    mask, and what their own views mask, read as one text (_VIEW_SEP)."""
    pieces: list[str] = []
    joined = _ViewMap()
    at = 0
    for begin, stop in spans:
        if pieces:
            pieces.append(_VIEW_SEP)
            joined.cut(at)
            at += len(_VIEW_SEP)
        pieces.append(text[begin:stop])
        joined.run(at, begin)
        at += stop - begin
    raw = "".join(pieces)
    view, view_map, _ = _decode_once(raw)
    if view is raw:
        return
    _mark(flags, _view_read(view, depth, None), (view_map, joined))


def _escape_masks(
    text: str,
    depth: int = 0,
    done: bytearray | None = None,
) -> bytearray:
    """The characters of *text* that its decoded views mask (see
    _VIEW_ESCAPE and above); *done* marks the literals already viewed.
    Linear: one pass finds the literals, and each view is one pass."""
    flags = bytearray(len(text))
    has_escape = _HAS_ESCAPE_616 if _en616_set.get() else _HAS_ESCAPE
    if depth >= _VIEW_DEPTH or has_escape.search(text) is None:
        return flags
    kept = bytearray(len(text))
    ends = bytearray(len(text))
    spans: list[tuple[int, int]] = []
    for found in _LITERAL.finditer(text):
        begin, quote = found.span(1)
        if begin == quote:
            continue
        kept[begin:quote] = b"\x01" * (quote - begin)
        if _STRING_CLOSE.match(text, quote + 1):
            ends[quote] = 1
        if (done is None or not done[begin]) and has_escape.search(text, begin, quote):
            spans.append((begin, quote))
    if spans:
        _view_batch(flags, text, spans, depth)
    _view_into(flags, text, 0, depth, kept)
    if '"' in text and _QUOTE_VIEW_ESCAPE.search(text):
        _quote_view(flags, text, ends, depth)
    return flags


# Where a pair of quotes is written with one quote raw and one as an escape
# (`"proxyAuthorization': \u0022<v>"`), raw quotes pair with the wrong
# partner and no view above reads the value (review R192C's mixed forms; the
# lead's ruling A, 2026-11-15 06:0xZ: a known leak class doesn't ship). The
# quote view decodes every escape but a line break's, quotes included, a
# level at a time to _VIEW_DEPTH, and each level is read by every pattern but
# the secret names' carrier, whose documents a decoded quote would carry past
# their string. A level's literals are not paired again: its quotes no longer
# pair. Each level is read, not only the last: a copy of a copy
# (`\\\"x\\ny\\\"`) is one JSON string a level up, and its line break ends
# the value a level down. A
# mask of its own stops at a raw quote that ends a JSON string (one followed
# by `,`, `}`, `]`, `:` or the end): its decoded quotes no longer tell one
# string from the next.
# A backslash run is read from its start and once (possessive): on a long run
# with no quote after it, a backtracking run is quadratic.
_QUOTE_VIEW_ESCAPE = re.compile(r'(?<!\\)\\++(?:"|u0022)|&(?:quot|#0*34|#[xX]0*22);')
_STRING_CLOSE = re.compile(r"[ \t]*+(?:[,}\]:]|\Z)")
_STRING_TAIL = re.compile(r"[ \t]*+[,}\]:][,}\]: \t]*+")


@functools.cache
def _quote_rows() -> frozenset[int]:
    """The table rows the quote view reads: all but the secret names'."""
    return frozenset(
        index
        for index, row in enumerate(_view_table())
        if row[1] is not _secret_name_value
    )


def _quote_view(flags: bytearray, text: str, ends: bytearray, depth: int) -> None:
    """Mark in *flags* what *text*'s quote view masks (see above); *ends*
    marks the raw quotes that end a JSON string."""
    view, kept = text, bytearray(b"\x01") * len(text)
    maps: list[_ViewMap] = []
    quoted = bytearray(len(text))
    for _ in range(depth, _VIEW_DEPTH):
        decoded, view_map, decoded_kept = _decode_once(view, kept, "\n\r")
        if decoded is view or decoded_kept is None:
            break
        view, kept = decoded, decoded_kept
        maps.insert(0, view_map)
        if _VIEW_WORTH.search(view):
            _mark(
                quoted,
                _tag_word_trim(
                    view,
                    _view_masks(view, _quote_rows()),
                    lambda tail: _view_masks(tail, _quote_rows()),
                ),
                tuple(maps),
            )
    size = len(text)
    index = quoted.find(1)
    while index >= 0:
        end = quoted.find(0, index)
        end = size if end < 0 else end
        edge = ends.find(1, index, end)
        stop = end if edge < 0 else edge
        flags[index:stop] = b"\x01" * (stop - index)
        index = quoted.find(1, end)


_VIEW_KEPT = re.compile(
    r'(?<!\\)\\*+"|(?<!\\)\\++u0022|&(?:quot|#0*34|#[xX]0*22);|\*\*\*REDACTED\*\*\*'
)


# Option (e): the decoded views run only on the
# records (lines) that can hold a credential the views would read. A record is
# gated in when it holds an escape the views decode (_VIEW_ESC) and either a cue
# from the redactor's own lists as written (_CUE: the table's carrier words and
# the word-less rows' literal heads) or any escape that writes a character a cue
# is made of (_CUE_ESC: letters, digits and `&-./:=?_`, as `\\u`, `\\x`, `%`,
# a numeric entity or `\\/`, at any depth): a cue written with escapes opens it.
# A gated line brings the line before and after it into the views. So does a
# pair: a line holding a cue and the next line that isn't blank as the views
# read them (whitespace only, as written or decoded: a literal `\t`), when an
# escape lies on either or on a blank line between; the blank lines between
# come in too. A credential split across lines (`"password":` then the value,
# `APIKEY`, a blank line, then `= <v>`) is read whole. Consecutive viewed lines
# are read together, as one text. A line that opens a private-key block (`-----BEGIN … PRIVATE
# KEY-----` as written, or a gated line at any level of its decoding) carries
# the views on through the block's reach: the PEM rule reads up to 8,192
# characters of body, as the views read them (decoded), then the END line. Its
# body holds no escape, and its END may be escaped under a plain BEGIN. A
# backslash run is read from its start and once (possessive): linear on a long
# run.
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----")
_PEM_BODY = 8192
_NONSPACE = re.compile(r"\S")
_BLANKISH = re.compile(r"[\s\\&#;%{}0-9a-fA-FnrtuUxX]*")
_VIEW_ESC = re.compile(
    r"\\|&(?:quot|apos|lt|gt|amp|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});"
)
_CUE = re.compile(
    r"(?i:token|secret|passw(?:or)?d|key|credential|pass_?phrase|session|cookie|authorization|bearer)"
    r"|_PASS|_PWD|PASS_|PASSVALUE|_pass|Pass|_pwd|Pwd|pass_|passvalue|passValue"
    r"|-----BEGIN|sk-|ya29\.|gsk_|csk-|AIza|AKIA|ASIA|dtn_|gh[pousr]_|github_pat_"
    r"|xox[baprs]-|xai-|r8_|hf_|fw_|eyJ|://|(?:[?&]|&(?:amp;)+)sig="
)
_CUE_HEX = r"(?:2[6dDeEfF]|3[0-9aAdDfF]|[4-7][0-9a-fA-F])"
_CUE_ESC = re.compile(
    rf"(?<!\\)\\++u00{_CUE_HEX}|(?<!\\)\\++x{_CUE_HEX}|%(?:25)*+{_CUE_HEX}|(?<!\\)\\++/"
    r"|&#0*(?:38|4[5-9]|5[0-8]|61|6[3-9]|[7-9][0-9]|1[01][0-9]|12[0-6]);"
    rf"|&#[xX]0*{_CUE_HEX};"
    # An entity written as an entity (`&amp;#115;`): the views decode it twice.
    r"|&(?:amp;)++#"
)
# the escapes its views decode open the gate too, beside 's
# (kept verbatim above): a percent escape is an escape the views decode
# (_VIEW_ESC_MORE, as is an entity without its `;`); and a cue character
# written as an octal, `\u{}`, `\x{}`, `\U` or an entity without its `;`
# opens it, as does an escape that writes an escape's own syntax (`#`, `%`,
# `;`: `\x2573` is `%73`, `&#37;73` too) (_CUE_ESC_MORE; its `_CUE_ESC`
# opens on `&amp;#`). Each run is read from its start and once (possessive):
# linear.
_VIEW_ESC_MORE = re.compile(r"%[0-9a-fA-F]{2}|&#[0-9]{1,7}|&#[xX][0-9a-fA-F]{1,6}")
_SYNTAX_HEX = r"(?:2[35]|3[bB])"
_ANY_CUE_HEX = rf"(?:{_CUE_HEX}|{_SYNTAX_HEX})"
_CUE_OCT = r"(?:04[356]|05[5-7]|06[0-7]|07[0-357]|1[0-7]{2})"
_CUE_ESC_MORE = re.compile(
    rf"(?<!\\)\\++(?:u00|x){_SYNTAX_HEX}|%(?:25)*+{_SYNTAX_HEX}"
    rf"|(?<!\\)\\++{_CUE_OCT}"
    rf"|(?<!\\)\\++[ux]\{{0*+{_ANY_CUE_HEX}\}}"
    rf"|(?<!\\)\\++U0{{6}}{_ANY_CUE_HEX}"
    r"|&#0*+(?:3[578]|4[5-9]|5[0-9]|61|6[3-9]|[7-9][0-9]|1[01][0-9]|12[0-6])(?![0-9])"
    rf"|&#[xX]0*+{_ANY_CUE_HEX}(?![0-9a-fA-F])"
)


# the views of one redaction call (a text, an
# object's leaves, an ACP trajectory, a publish value) run under one work budget, counted in units of estimated view seconds on the reference VM
# (n2d-standard-32) and computed from the text alone, so the same text is redacted
# the same way on every machine. Each table row a view runs is charged the
# characters it scans times that row's cost per character (_VIEW_ROW_NS, measured
# on our real trajectory and log files, times _VIEW_OVERHEAD for the pass around
# the regex); a decode its characters; a replacement a fixed cost. All are charged
# before the work runs. The budget is _VIEW_BUDGET_UNITS: at the lead's
# calibration (p99 of our real files' sizes) 0.05% of JSON-record files and 0.08%
# of logs reach it, and junit reports do. A wall-clock backstop catches a machine
# far slower than the reference. Past either bound, the run of viewed lines it
# falls in and every viewed line after it are masked whole (over-mask, never a
# leak: a view masks only inside the viewed lines, and the gate's cue test is
# escape-aware) and counted under the bound that ran out; the base pass still
# runs on every line, and a line the views leave out gets nothing from a view.
# A run is read whole or masked whole, never cut: the gate joins the lines a
# credential may span (a pair, a key block). Recalibrate when the table changes
# (a test checks the lengths agree). Rows 22 and 28, its YAML block and price
# name, were measured the same way on R528's real files when #289 merged in (FX307C,
# 2026-11-16: 195.4 and 147.4 ns a character, times the other rows' median ratio to
# this table, 1.22). Rows 36 and 37, #358's every-name-that-ends-in-authorization
# carrier and its next line's entry under a credential name, the same way when
# #358 merged in (346.2 and 34.4 ns a character on 21.5 MB of
# the 13 real files, times the other rows' median ratio, 1.196); row 36 runs only
# on a text that holds `authorization` (its words).
_VIEW_ROW_NS = (
    0.5, 0.9, 19.0, 0.9, 0.9, 8.1, 0.9, 0.5, 0.6, 0.7, 0.7, 0.6,
    0.5, 0.5, 0.9, 0.7, 0.6, 1.2, 18.4, 9.4, 7.3, 7.3, 237.9, 70.5,
    46.6, 61.6, 40.0, 480.3, 179.4, 134.5, 40.0, 34.5, 34.4, 100.4, 28.6, 54.7,
    414.2, 41.2,
)  # fmt: skip
_VIEW_OVERHEAD = 1.1
_VIEW_DECODE_NS = 11.5
_VIEW_MATCH_NS = 50000.0
_VIEW_BUDGET_UNITS = 5.0
_VIEW_DEADLINE_SECONDS = 120.0


class _ViewBudgetSpent(Exception):
    """The views of one redaction ran out of work budget or of time."""

    def __init__(self, backstop: bool) -> None:
        super().__init__("wall-clock backstop" if backstop else "view work budget")
        self.backstop = backstop


class _ViewBudget:
    """One redaction's view work budget and wall-clock backstop."""

    __slots__ = ("deadline", "left", "spent")

    def __init__(self, units: float, seconds: float) -> None:
        self.left = units
        self.deadline = time.monotonic() + seconds
        self.spent = {"row": 0.0, "decode": 0.0, "match": 0.0}

    def spend(self, units: float, kind: str) -> None:
        self.spent[kind] += units
        self.left -= units
        if self.left < 0:
            raise _ViewBudgetSpent(backstop=False)
        if time.monotonic() > self.deadline:
            raise _ViewBudgetSpent(backstop=True)


_view_budget: ContextVar[_ViewBudget | None] = ContextVar("_view_budget", default=None)


def _charge_view(units: float, kind: str) -> None:
    """Charge *units* of view work of *kind* to this redaction's budget."""
    budget = _view_budget.get()
    if budget is not None:
        budget.spend(units, kind)


# One call's budget, shared by every text it redacts; charges go to
# _view_budget, which is set only while views run (the gate is never charged).
_call_budget: ContextVar[_ViewBudget | None] = ContextVar("_call_budget", default=None)


def _one_view_budget[**P, R](call: Callable[P, R]) -> Callable[P, R]:
    """*call* under one view work budget for all it redacts: an object's
    leaves, an ACP trajectory's events, a publish value's fields. The
    outermost such call sets the budget, and a nested one uses it."""

    @functools.wraps(call)
    def shared(*args: P.args, **kwargs: P.kwargs) -> R:
        if _call_budget.get() is not None:
            return call(*args, **kwargs)
        token = _call_budget.set(
            _ViewBudget(_VIEW_BUDGET_UNITS, _VIEW_DEADLINE_SECONDS)
        )
        try:
            return call(*args, **kwargs)
        finally:
            _call_budget.reset(token)

    return shared


def _gated_lines(text: str) -> tuple[list[int], list[int], bytearray]:
    """Each line's start and end in *text*, and whether the views read it: a
    gated line (see above) and the line before and after it."""
    size = len(text)
    starts: list[int] = []
    ends: list[int] = []
    escaped = bytearray()
    gated = bytearray()
    start = 0
    while start <= size:
        end = text.find("\n", start)
        end = size if end < 0 else end
        starts.append(start)
        ends.append(end)
        found = (
            _VIEW_ESC.search(text, start, end) is not None
            or _VIEW_ESC_MORE.search(text, start, end) is not None
        )
        escaped.append(found)
        gated.append(found and _has_cue(text, start, end))
        start = end + 1
    count = len(gated)
    viewed = bytearray(gated)
    # read[k]: the characters before line k as the views read them, filled
    # forward once from a block's first body line (one pass however blocks
    # overlap); the lines no block reaches count 0.
    read = [0]
    for index in range(count):
        start, end = starts[index], ends[index]
        if gated[index]:
            if index:
                viewed[index - 1] = 1
            if index + 1 < count:
                viewed[index + 1] = 1
        if _PEM_BEGIN.search(text, start, end) is None and not (
            gated[index] and _decodes_to_pem(text[start:end])
        ):
            continue
        viewed[index] = 1
        first = index + 1
        while len(read) <= first:
            read.append(read[-1])
        line = len(read) - 1
        while line < count and read[line] - read[first] <= _PEM_BODY:
            read.append(read[line] + _read_length(text[starts[line] : ends[line] + 1]))
            viewed[line] = 1
            line += 1
    # The pairs: *cued* caches each line's cue test (2 = holds one, 1 = not).
    cued = bytearray(2 if flag else 0 for flag in gated)
    prev, between = -1, False
    for index in range(count):
        start, end = starts[index], ends[index]
        if _blank(text, start, end, escaped[index]):
            between = between or bool(escaped[index])
            continue
        if prev >= 0 and (between or escaped[prev] or escaped[index]):
            for line in (prev, index):
                if not cued[line]:
                    cued[line] = 2 if _has_cue(text, starts[line], ends[line]) else 1
            if cued[prev] == 2 or cued[index] == 2:
                viewed[prev : index + 1] = b"\x01" * (index + 1 - prev)
        prev, between = index, False
    return starts, ends, viewed


def _has_cue(text: str, start: int, end: int) -> bool:
    """Whether text[start:end] holds a cue, as written or written with escapes."""
    return (
        _CUE.search(text, start, end) is not None
        or _CUE_ESC.search(text, start, end) is not None
        or _CUE_ESC_MORE.search(text, start, end) is not None
    )


def _blank(text: str, start: int, end: int, escaped: int) -> bool:
    """Whether text[start:end] is blank as the views read it: whitespace only,
    as written or once its escapes are decoded (a literal `\\t`)."""
    if _NONSPACE.search(text, start, end) is None:
        return True
    return (
        bool(escaped)
        and _BLANKISH.fullmatch(text, start, end) is not None
        and _NONSPACE.search(_read(text[start:end])) is None
    )


def _decodes_to_pem(text: str) -> bool:
    """Whether a level of *text*'s decoding (to _VIEW_DEPTH) opens a
    private-key block."""
    for _ in range(_VIEW_DEPTH):
        decoded = _decode_once(text)[0]
        if decoded is text:
            return False
        if _PEM_BEGIN.search(decoded):
            return True
        text = decoded
    return False


def _read_length(text: str) -> int:
    """The length of *text* as the views read it (a decode never lengthens)."""
    return len(_read(text))


def _read(text: str) -> str:
    """*text* as the views read it: decoded to _VIEW_DEPTH."""
    if "\\" not in text and "&" not in text and "%" not in text:
        return text
    for _ in range(_VIEW_DEPTH):
        decoded = _decode_once(text)[0]
        if decoded is text:
            break
        text = decoded
    return text


def _escape_flags(text: str) -> bytearray | None:
    """_escape_flags_counted's flags."""
    return _escape_flags_counted(text)[0]


def _escape_flags_counted(text: str) -> tuple[bytearray | None, int, str | None]:
    """_escape_flags_whole on the viewed records of *text* (see above), each
    run of consecutive viewed lines read as one text, under the work budget;
    the count of viewed lines masked whole past it; and the category of the
    bound that ran out, if one did."""
    if "\\" not in text and "&" not in text and "%" not in text:
        return None, 0, None
    flags: bytearray | None = None
    fallback = 0
    bound: str | None = None
    pause = _view_budget.set(None)
    try:
        starts, ends, viewed = _gated_lines(text)
    finally:
        _view_budget.reset(pause)
    count = len(viewed)
    token = _view_budget.set(
        _call_budget.get() or _ViewBudget(_VIEW_BUDGET_UNITS, _VIEW_DEADLINE_SECONDS)
    )
    try:
        index = 0
        while index < count:
            if not viewed[index]:
                index += 1
                continue
            if bound is not None:
                start, end = starts[index], ends[index]
                if flags is None:
                    flags = bytearray(len(text))
                flags[start:end] = b"\x01" * (end - start)
                fallback += 1
                index += 1
                continue
            first = index
            while index + 1 < count and viewed[index + 1]:
                index += 1
            run, stop = starts[first], ends[index]
            try:
                part = _escape_flags_whole(text[run:stop])
            except _ViewBudgetSpent as spent:
                bound = (
                    REDACTION_CATEGORY_VIEW_BACKSTOP
                    if spent.backstop
                    else REDACTION_CATEGORY_VIEW_BUDGET
                )
                # The run's own lines go to the fallback: read them again.
                index = first
                continue
            if part is not None:
                if flags is None:
                    flags = bytearray(len(text))
                flags[run:stop] = part
            index += 1
    finally:
        _view_budget.reset(token)
    return flags, fallback, bound


def _reads_more(text: str) -> bool:
    """*text* holds one of its escapes, as written or once its set
    has decoded it (`\\u0025` + `73` is `%73`, `&#92;x73` is `\\x73`)."""
    if _MORE_ESCAPE.search(text) is not None:
        return True
    token = _en616_set.set(True)
    try:
        decoded = _read(text)
    finally:
        _en616_set.reset(token)
    return decoded is not text and _MORE_ESCAPE.search(decoded) is not None


def _escape_flags_whole(text: str) -> bytearray | None:
    """The characters of *text* (as written, before any pattern) that its
    decoded views mask (see _VIEW_ESCAPE), less every quote and earlier mask
    (_VIEW_KEPT): a quote holds no secret; None where there are none."""
    if "\\" not in text and "&" not in text and "%" not in text:
        return None
    token = _view_cache.set({})
    try:
        first = _en616_set.set(True)
        try:
            cover = _escape_masks(text)
        finally:
            _en616_set.reset(first)
        if _reads_more(text):
            more = _escape_masks(text)
            cover = bytearray(
                (int.from_bytes(cover, "big") | int.from_bytes(more, "big")).to_bytes(
                    len(text), "big"
                )
            )
    finally:
        _view_cache.reset(token)
    if cover.find(1) < 0:
        return None
    for found in _VIEW_KEPT.finditer(text):
        cover[found.start() : found.end()] = bytes(found.end() - found.start())
    # Nor the `,`, `}`, `]` or `:` after a raw quote that ends a JSON string:
    # a redacted JSON record stays JSON (the JSON parse check).
    for found in _LITERAL.finditer(text):
        tail = _STRING_TAIL.match(text, found.end(1) + 1)
        if tail is not None and tail.end() > tail.start():
            cover[tail.start() : tail.end()] = bytes(tail.end() - tail.start())
    return cover if cover.find(1) >= 0 else None


def _apply_view(text: str, origin: list[int], flags: bytearray) -> tuple[str, int]:
    """*text*, the main pass's result, with each character the view masks in
    the original (*flags*, by *origin*) masked too: what the main pass left
    visible only. A quote and a mask stay. The count of masks added."""
    table = bytes(flags) + b"\x00"  # origin -1 (a mask) reads the last byte
    cover = bytearray([table[o] for o in origin])
    for found in _VIEW_KEPT.finditer(text):
        cover[found.start() : found.end()] = bytes(found.end() - found.start())
    out: list[str] = []
    count = 0
    index = 0
    size = len(text)
    while index < size:
        start = cover.find(1, index)
        if start < 0:
            out.append(text[index:])
            break
        out.append(text[index:start])
        end = cover.find(0, start)
        end = size if end < 0 else end
        out.append(_MASK)
        count += 1
        index = end
    return "".join(out), count


def redact_trajectory_text_with_view_flags(
    text: str, extra: bytearray
) -> tuple[str, dict[str, int]]:
    """redact_trajectory_text_with_categories, with the characters *extra*
    marks in *text* (as written; view_flags) masked too, through the position
    map the views' own masks use: the main pass reads the text as written,
    and masks are only added (publish's own rules)."""
    flags = _escape_flags(text)
    if flags is not None:
        extra = bytearray(
            (int.from_bytes(flags, "big") | int.from_bytes(extra, "big")).to_bytes(
                len(text), "big"
            )
        )
    current, origin, _, categories = _redaction_pass(text)
    current, count = _apply_view(current, origin, extra)
    if count:
        categories[REDACTION_CATEGORY_CREDENTIAL_FIELD] = (
            categories.get(REDACTION_CATEGORY_CREDENTIAL_FIELD, 0) + count
        )
    return current, categories


def _view_after(original: str, text: str) -> tuple[str, int]:
    """For a caller with a main pass of its own: *text*, its result for
    *original*, with the view applied (the result is read again, tracked)."""
    flags = _escape_flags(original)
    if flags is None:
        return text, 0
    current, origin, _, _ = _redaction_pass(original)
    return _apply_view(current, origin, flags)


def redact_trajectory_text_with_categories(text: str) -> tuple[str, dict[str, int]]:
    """Apply the canonical patterns and report replacements per category.

    Categories name the kind of secret each rule actually detects (API key,
    bearer token, private key block, URL credential, credential-bearing field
    value) so upload previews can itemize what was masked without inventing
    detection that does not exist. The total replacement count is the sum of
    the returned values.
    """
    # What BenchFlow staged in this trial goes first, whole and under its own
    # name, before a pattern can take a token-shaped part of it; outside a
    # trial nothing is registered (``_utils.staged_secrets``).
    text = staged_secrets.mask(text)
    memo = _redaction_memo.get()
    cached = memo.get(text) if memo is not None else None
    if cached is not None:
        return cached[0], dict(cached[1])
    original = text
    # The decoded view reads the text as written, before any pattern masks a
    # name that a value's context needs (in a JSON copy of a copy the
    # one-word rule masks the name first), and masks only what the main pass
    # leaves visible, so the main pass's own output stays as it is.
    flags, fallback, bound = _escape_flags_counted(text)
    if flags is not None:
        text, origin, _, categories = _redaction_pass(text)
        text, count = _apply_view(text, origin, flags)
        if count:
            categories[REDACTION_CATEGORY_CREDENTIAL_FIELD] = (
                categories.get(REDACTION_CATEGORY_CREDENTIAL_FIELD, 0) + count
            )
        if bound is not None and fallback:
            categories[bound] = fallback
        if memo is not None:
            memo.remember(original, text, categories)
        return text, categories
    categories: dict[str, int] = {}
    folded: str | None = None
    fold = True
    for pattern, replacement, category in _REDACTION_PATTERNS:
        repl: str | Callable[[re.Match[str]], str] = replacement
        # Anything else that stands in for a pattern is run as it is.
        if isinstance(pattern, _LazyPattern):
            if pattern.words or isinstance(pattern, _GatedPattern):
                if fold and pattern.flags & re.IGNORECASE:
                    folded, fold = _folded(text), False
                if not pattern.may_match(text, folded):
                    continue
            repl = _PREFIX_KEEPERS.get(replacement, replacement)
        if not isinstance(replacement, str):
            text, count = _subn_changed(pattern, replacement, text)
        else:
            text, count = pattern.subn(repl, text)
        if count:
            categories[category] = categories.get(category, 0) + count
            # The words are looked for in the text as it now stands.
            fold = True
    if memo is not None:
        memo.remember(original, text, categories)
    return text, categories


# Repeated conversation strings dominate redaction time. Cache only within a
# trial, including its child tasks / asyncio.to_thread calls. Outside a trial
# there is no memo. Charge Python object bytes (including wide Unicode), with
# a conservative allowance for each dict slot, rather than counting characters.
_REDACTED_MAX_BYTES = 4 << 20


class _RedactionMemo:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[str, dict[str, int]]] = {}
        self.size = 0
        self.closed = False
        self.lock = threading.Lock()

    def get(self, text: str) -> tuple[str, dict[str, int]] | None:
        with self.lock:
            return self.entries.get(text)

    def remember(self, text: str, redacted: str, categories: dict[str, int]) -> None:
        value = (redacted, dict(categories))
        size = (
            sys.getsizeof(text)
            + sys.getsizeof(redacted)
            + sys.getsizeof(value)
            + sys.getsizeof(value[1])
            + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in categories.items())
            + 256  # dict allocation / slot overhead, including the empty dict
        )
        with self.lock:
            if self.closed or text in self.entries or size > _REDACTED_MAX_BYTES:
                return
            if self.size + size > _REDACTED_MAX_BYTES:
                self.entries.clear()
                self.size = 0
            self.entries[text] = value
            self.size += size

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.entries.clear()
            self.size = 0


_redaction_memo: ContextVar[_RedactionMemo | None] = ContextVar(
    "redaction_memo", default=None
)


@contextmanager
def trial_redaction_memo() -> Iterator[_RedactionMemo]:
    """Discard cached texts on every trial exit, even in inherited contexts."""
    memo = _RedactionMemo()
    token = _redaction_memo.set(memo)
    try:
        yield memo
    finally:
        memo.close()
        _redaction_memo.reset(token)


@_one_view_budget
def redact_trajectory_obj(obj: Any) -> Any:
    """Recursively redact secrets in the string leaves of a JSON-serializable value.

    Prefer this over running :func:`redact_trajectory_text` on an
    already-``json.dumps``-ed record. Redacting the serialized *text* can split a
    ``\\`` escape — a secret sitting next to one leaves a lone ``\\X`` — which
    corrupts the JSON so the trajectory file no longer parses and
    ``bench train convert`` fails. Redacting the *values* before serializing keeps
    escaping valid while preserving coverage: free-text leaves are scrubbed by the
    token-family and inline ``name: value`` carriers, and structured fields (e.g.
    a ``{"x-api-key": "<prefixless>"}`` header) are scrubbed in their key context
    by :func:`_redact_field` so the carriers that key off a field name still fire.
    """
    if isinstance(obj, dict):
        return {key: _redact_field(key, value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [redact_trajectory_obj(item) for item in obj]
    if isinstance(obj, str):
        return redact_trajectory_text(obj)
    return obj


def _redact_field(key: Any, value: Any) -> Any:
    """Redact a dict field's *value*, keeping the field name as carrier context.

    The ``name: value`` carriers (``x-api-key``, ``api_key``, ``authorization``,
    ``*_TOKEN`` …) strip *prefixless* secret values only when they can see the
    field name; in a parsed object the name and value are separate. So redact the
    value inside a ``{"<key>": "<value>"}`` probe and read it back. The probe
    round-trips through json so escaping stays valid; if redacting the serialized
    probe would corrupt it (the escape hazard this module guards), fall back to
    redacting the bare value, which still catches the prefixed token families.
    """
    named = isinstance(key, str)
    if (
        named
        and isinstance(value, dict | list | str)
        and tuple(name_words(key)) in _PATH_OR_CONTENT_NAMES
        and entry_kind(key, value) == "credential"
    ):
        # GOOGLE_APPLICATION_CREDENTIALS holding the service account's JSON
        # as a mapping, a list or a JSON string: the whole value (a path is
        # the classifier's setting and kept).
        return "***REDACTED***"
    if named and isinstance(value, dict | list) and is_secret_field(key, value):
        # An object or list under a credential's name (`{"password": {"a":
        # "<v>"}}`, `{"secret": ["<v>"]}`): every string in it is the
        # secret's, and a number decides as one under the name would.
        return _mask_leaves(key, value)
    if isinstance(value, dict):
        return redact_trajectory_obj(value)
    if isinstance(value, list):
        # Numbers in a list under the field's name decide as one number would.
        return [
            "***REDACTED***"
            if named and is_secret_number(key, item)
            else redact_trajectory_obj(item)
            for item in value
        ]
    if not isinstance(value, str):
        # A PIN-like integer under a credential's name is masked, as the
        # export redactor masks it (is_secret_number); nothing else that is
        # not a string is.
        return "***REDACTED***" if named and is_secret_number(key, value) else value
    if isinstance(key, str):
        import json

        probe = json.dumps({key: value})
        redacted = redact_trajectory_text(probe)
        if redacted == probe:
            return value
        try:
            restored = json.loads(redacted)
        except ValueError:
            pass
        else:
            if isinstance(restored, dict) and isinstance(restored.get(key), str):
                return restored[key]
    return redact_trajectory_text(value)


def _mask_leaves(key: str, value: Any) -> Any:
    """``value``, held under the credential name ``key``, with every string
    leaf masked and its shape kept (keys, lists, booleans, null)."""
    if isinstance(value, dict):
        return {name: _mask_leaves(key, item) for name, item in value.items()}
    if isinstance(value, list):
        return [_mask_leaves(key, item) for item in value]
    if isinstance(value, str):
        return "***REDACTED***" if value else value
    return "***REDACTED***" if is_secret_number(key, value) else value


OBSERVATION_ID_KEY = "observation_id"


def stamp_observation_ids(trajectory: list[dict[str, Any]] | None) -> None:
    """Give every tool call an opaque, unique identity, in place, before any
    redaction.

    The identity is random (``obs-`` and 32 hex digits), so it carries nothing
    from the call and redaction leaves it as it is. Every sighting of one call
    (the events sharing its ``tool_call_id`` in this list) shares it; an event
    that already has one keeps it, so a trajectory stamped once keeps the same
    identities through every later write and every copy. Integrity keys what
    it stores about each observation on this identity (never on a value
    redaction can change, nor on an ID that can collide).
    """
    import uuid

    if not trajectory:
        return
    by_call: dict[str, str] = {}
    for event in trajectory:
        if not isinstance(event, dict) or event.get("type") != "tool_call":
            continue
        call = event.get("tool_call_id")
        existing = event.get(OBSERVATION_ID_KEY)
        if isinstance(existing, str) and existing:
            if isinstance(call, str) and call:
                by_call.setdefault(call, existing)
            continue
        identity = by_call.get(call) if isinstance(call, str) and call else None
        if identity is None:
            identity = f"obs-{uuid.uuid4().hex}"
            if isinstance(call, str) and call:
                by_call[call] = identity
        event[OBSERVATION_ID_KEY] = identity


@_one_view_budget
def redact_acp_trajectory_jsonl(trajectory: list[dict[str, Any]]) -> str:
    """Serialize an ACP trajectory list to redacted JSONL.

    Each event's string values are redacted (env dumps, curl -v output, header
    logs the agent echoed into ``acp_trajectory.jsonl``) *before* serialization,
    so secrets are stripped (#537/#585) without a post-serialization regex
    corrupting backslash escapes. Events are normalized to JSON primitives first
    (applying ``default=str``) so non-JSON leaves are covered too. Returns the
    joined lines without a trailing newline; callers that need one append it.
    """
    import json

    lines = []
    for event in trajectory:
        normalized = json.loads(json.dumps(event, default=str))
        lines.append(json.dumps(redact_trajectory_obj(normalized)))
    return "\n".join(lines)


class LLMRequest(BaseModel):
    """A single request to an LLM API, captured by the proxy."""

    timestamp: datetime = Field(default_factory=datetime.now)
    method: str = "POST"
    path: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A single response from an LLM API, captured by the proxy."""

    timestamp: datetime = Field(default_factory=datetime.now)
    status_code: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)


class LLMExchange(BaseModel):
    """A request-response pair."""

    request: LLMRequest
    response: LLMResponse
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Raw trajectory: ordered list of captured LLM API exchanges."""

    session_id: str
    agent_name: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    exchanges: list[LLMExchange] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # The provider tool evidence a stopped proxy's import read from each
    # exchange before it shed its request body (the full records are in
    # llm_trajectory.jsonl); None while the exchanges are whole.
    _tool_evidence: list[dict[str, Any]] | None = PrivateAttr(default=None)

    @property
    def has_provider_usage(self) -> bool:
        """Whether any exchange contains provider-supplied usage fields."""
        for ex in self.exchanges:
            usage = ex.response.body.get("usage")
            if isinstance(usage, dict) and _has_provider_usage(usage):
                return True
            usage_metadata = ex.response.body.get("usageMetadata")
            if isinstance(usage_metadata, dict) and _has_non_null_key(
                usage_metadata, _USAGE_METADATA_KEYS
            ):
                return True
        return False

    @property
    def total_input_tokens(self) -> int | None:
        """Uncached input tokens over the captured exchanges (None: unreported)."""
        return _known_sum(
            _exchange_token_usage(ex).input_tokens for ex in self.exchanges
        )

    @property
    def total_output_tokens(self) -> int | None:
        """Output tokens, reasoning included, over the captured exchanges."""
        return _known_sum(
            _exchange_token_usage(ex).output_tokens for ex in self.exchanges
        )

    @property
    def total_cache_read_tokens(self) -> int | None:
        """Prompt-cache read tokens over the captured exchanges."""
        return _known_sum(
            _exchange_token_usage(ex).cache_read_tokens for ex in self.exchanges
        )

    @property
    def total_cache_creation_tokens(self) -> int | None:
        """Prompt-cache write tokens over the captured exchanges (None when no
        response reported any: OpenAI reports none)."""
        return _known_sum(
            _exchange_token_usage(ex).cache_creation_tokens for ex in self.exchanges
        )

    @property
    def total_cache_write_5m_tokens(self) -> int | None:
        """The 5-minute part of the cache writes, where reported."""
        return _known_sum(
            _exchange_token_usage(ex).cache_write_5m_tokens for ex in self.exchanges
        )

    @property
    def total_cache_write_1h_tokens(self) -> int | None:
        """The 1-hour part of the cache writes, where reported."""
        return _known_sum(
            _exchange_token_usage(ex).cache_write_1h_tokens for ex in self.exchanges
        )

    @property
    def total_provider_tokens(self) -> int:
        """All tokens over the captured exchanges."""
        return sum(_exchange_token_usage(ex).total_tokens for ex in self.exchanges)

    @property
    def total_cost_usd(self) -> float | None:
        """Extract cost if the API returns it (Anthropic does not, OpenAI does not)."""
        return self.metadata.get("cost_usd")

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Extract all messages from all exchanges (the conversation history)."""
        msgs: list[dict[str, Any]] = []
        for ex in self.exchanges:
            # Request messages
            req_msgs = ex.request.body.get("messages", [])
            if req_msgs and (not msgs or req_msgs != msgs):
                msgs = list(req_msgs)  # latest request has full history
            # Response message
            resp_content = ex.response.body.get("content", [])
            if resp_content:
                msgs.append({"role": "assistant", "content": resp_content})
            # OpenAI format
            choices = ex.response.body.get("choices", [])
            if choices and "message" in choices[0]:
                msgs.append(choices[0]["message"])
        return msgs

    def to_jsonl(self, *, redact_keys: bool = True) -> str:
        """Export as JSONL (one exchange per line).

        Redaction runs on the record's string *values* before ``json.dumps`` so
        the emitted JSON is always valid — regexing the serialized text could
        split a ``\\`` escape next to a secret and produce an unparseable line.
        """
        import json

        lines = []
        for ex in self.exchanges:
            data = ex.model_dump(mode="json")
            if redact_keys:
                data = redact_trajectory_obj(data)
            lines.append(json.dumps(data, default=str))
        return "\n".join(lines)
