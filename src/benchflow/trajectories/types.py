"""Trajectory types — raw LLM API request/response pairs captured from providers."""

import functools
import re
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

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
    """One call's tokens in BenchFlow's split: ``input_tokens`` is
    the uncached input only, beside the cache reads and writes; the 5-minute
    and 1-hour shares of the writes are None where the provider does not
    report them; ``output_tokens`` includes reasoning."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    provider_total_tokens: int | None = None
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None
    # False when the response named a total but no input or output count
    # (a total-only response): the split is unknown, not 0.
    split_reported: bool = True
    # False when the response named one of input and output but not the
    # other (an output-only response): that count is unknown, not 0.
    input_reported: bool = True
    output_reported: bool = True
    # False when the response named no cache-write count at all (OpenAI's
    # chat answers, MiMo's): its writes are unknown, not 0.
    cache_write_reported: bool = True
    # True when the response's input count holds its cache (OpenAI's and
    # Gemini's prompt counts): unnamed writes are then inside input_tokens.
    input_holds_cache: bool = False

    @property
    def total_tokens(self) -> int:
        if self.provider_total_tokens is not None:
            return self.provider_total_tokens
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
            + self.output_tokens
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
    reported_additive_cache_read = _first_int(
        usage.get("cache_read_input_tokens"),
        usage.get("cacheReadInputTokens"),
        usage.get("cacheReadInputTokenCount"),
    )
    # OpenRouter reports its cache writes in the prompt's details
    # (``prompt_tokens_details.cache_write_tokens``; LiteLLM's copy of the
    # answer adds ``cache_creation_tokens`` beside it, and its Responses-API
    # bridge moves them to ``input_tokens_details``), inside the prompt count.
    # DashScope names its explicit-cache writes
    # ``prompt_tokens_details.cache_creation_input_tokens``. An answer with
    # none of these keys reported no writes: unknown, not 0.
    written_counts = (
        usage.get("cache_creation_input_tokens"),
        usage.get("cacheWriteInputTokens"),
        usage.get("cacheWriteInputTokenCount"),
        prompt_details.get("cache_write_tokens"),
        prompt_details.get("cache_creation_tokens"),
        input_details.get("cache_write_tokens"),
        prompt_details.get("cache_creation_input_tokens"),
    )
    cache_creation_tokens = _first_int(*written_counts)
    cache_write_reported = _first_optional_int(*written_counts) is not None
    # Cache reported as a SUBSET already inside the input count — OpenAI
    # (`*_tokens_details.cached_tokens`) and Gemini (`cachedContentTokenCount`).
    inclusive_cache_read = _first_int(
        prompt_details.get("cached_tokens"),
        input_details.get("cached_tokens"),
        usage_metadata.get("cachedContentTokenCount"),
    )
    cache_read_tokens = reported_additive_cache_read or inclusive_cache_read

    # Normalize `input_tokens` to mean the same thing across providers: the
    # UNCACHED input only, with cache reads and writes beside it.
    # Anthropic/Bedrock report it so already (their cache is a separate
    # additive component); OpenAI/Gemini report a cache-inclusive prompt count,
    # so their cache read (a subset of it) comes out. Some normalized responses
    # expose the same cache read in both families (Gemini pass-through:
    # ``prompt_tokens_details`` and ``cache_read_input_tokens``); the inclusive
    # marker decides, so it is subtracted once.
    # A prompt count is cache-inclusive when it is OpenAI's or Gemini's
    # (``prompt_tokens``, ``promptTokenCount``; LiteLLM's chat shape for
    # Anthropic and Bedrock too, whose ``prompt_tokens`` holds the input and
    # the cache reads and writes, as OpenRouter's does) or a details block
    # marks its cached part
    # (the Responses API's ``input_tokens``). Its cache reads and writes come
    # out; an Anthropic Messages or Bedrock ``input_tokens`` is uncached
    # already.
    raw_input_reported: int | None = None
    inclusive = False
    for value, is_inclusive in (
        (usage.get("input_tokens"), bool(input_details)),
        (usage.get("prompt_tokens"), True),
        (usage.get("inputTokens"), False),
        (usage_metadata.get("promptTokenCount"), True),
    ):
        raw_input_reported = _first_optional_int(value)
        if raw_input_reported is not None:
            inclusive = is_inclusive or bool(inclusive_cache_read)
            break
    raw_input = raw_input_reported or 0
    # Gemini reports tool-use prompt tokens (`toolUsePromptTokenCount`) separately
    # from `promptTokenCount` — it is additive input, NOT a subset — so fold it in
    # too, or tool-heavy Gemini runs underreport input/cost (and totalTokenCount
    # would exceed input + output). Absent for every other provider.
    additive_tool_use_prompt = _first_int(usage_metadata.get("toolUsePromptTokenCount"))
    inside = cache_read_tokens + cache_creation_tokens if inclusive else 0
    input_tokens = max(raw_input - inside, 0) + additive_tool_use_prompt
    # Anthropic splits its cache writes by lifetime (5-minute and 1-hour
    # rates), in its own ``cache_creation`` or in LiteLLM's chat shape
    # (``prompt_tokens_details.cache_creation_token_details``); other
    # providers do not, and their split stays unknown.
    written = usage.get("cache_creation")
    if not isinstance(written, dict):
        written = prompt_details.get("cache_creation_token_details")
    written = written if isinstance(written, dict) else {}
    cache_write_5m = _first_optional_int(written.get("ephemeral_5m_input_tokens"))
    cache_write_1h = _first_optional_int(written.get("ephemeral_1h_input_tokens"))

    # Reasoning/thinking tokens are billed as output. Anthropic/OpenAI already
    # fold them into output_tokens/completion_tokens; Gemini reports them
    # separately as `thoughtsTokenCount`, so add them in for output parity.
    output_reported = _first_optional_int(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
        usage.get("outputTokens"),
        usage_metadata.get("candidatesTokenCount"),
        usage_metadata.get("thoughtsTokenCount"),
    )
    output_tokens = _first_int(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
        usage.get("outputTokens"),
        usage_metadata.get("candidatesTokenCount"),
    ) + _first_int(usage_metadata.get("thoughtsTokenCount"))

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        provider_total_tokens=_first_optional_int(
            usage.get("total_tokens"),
            usage_metadata.get("totalTokenCount"),
            usage.get("totalTokens"),
        ),
        cache_write_5m_tokens=cache_write_5m,
        cache_write_1h_tokens=cache_write_1h,
        split_reported=raw_input_reported is not None or output_reported is not None,
        input_reported=raw_input_reported is not None,
        output_reported=output_reported is not None,
        cache_write_reported=cache_write_reported,
        input_holds_cache=inclusive,
    )


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
_GLUE_AHEAD = (
    rf"(?:(?:{_GLUE_WORD}|{_GLUE_RUN}(?:{_SEGMENT_START}|(?:{_CSI_CODE}){{1,4}}){_GLUE_WORD})"
    rf"{_GLUE_NAME}|{_GLUE_RUN}{_GLUE_END}(?:[_-]?[0-9]{{1,8}})?"
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
# its run (_GLUE_AHEAD: `abc\"password\": ...` is a value, then another name's). A double quote
# as written is never read (it may close the copy's own string); nor is a
# quote before a space, `,` `}` `]` `)` `:` or the end, which closes the value.
_QUOTE_UNIT = rf"(?:'|(?:\\\\)*+\\[\"']|{_QUOTE_ESCAPE})"
_VALUE_QUOTE = (
    rf"{_QUOTE_UNIT}"
    rf"(?={_QUOTE_UNIT}{{0,8}}+"
    rf"(?!{_GLUE_AHEAD})"
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
# between (the words _GLUE_END reads: `password=<v1>token: <v2>`,
# `<v1>Accesskey :<v2>`, `<v1>secret =\n\n<v2>`), where that name's
# separator has a space before or after it: the value stops at the space and
# would leave the next value behind, its name read into this one. It reads on through the
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
# One line break as written, escaped (json.dumps writes `\n`, and text can
# arrive escaped again: up to eight backslashes, as in _ESCQ) or URL-encoded.
_NL = r"(?:\r?\n|\\{1,8}r\\{1,8}n|\\{1,8}n|%0[Dd]%0[Aa]|%0[Aa])"
# A space or tab as written or escaped (`\t`).
_SP = r"(?:[ \t]|\\{1,8}t)"
# A YAML block scalar's indicator after a name's `:` (`apiKey: |`, `>-`,
# `|2+`): the value starts on the next line.
_BLOCK = r"[|>](?:[+-][1-9]?|[1-9][+-]?)?"
# A name of its own, the next line's entry rather than a value
# (`GITHUB_TOKEN=\nHOME=/root`).
_OWN_NAME = r"\\{0,8}[\"']?[A-Za-z0-9_.-]{1,64}\\{0,8}[\"']?[ \t]*[:=]"
# The gap between a carrier's `:` or `=` and its value. Raw whitespace has
# always spanned lines (the `\s*` fallback). An escaped line break does too
# (a json.dumps copy of `password:\n\n  v`), blank lines included, and a
# YAML block indicator hands the value to the next line (`apiKey: |\n  v`,
# whose block _YAML_BLOCK masks whole; the indicator is never the value).
# Across a break the value must start
# as one (a letter, digit or a key character, quoted or not) and must not be
# a name of its own, so `print("password:\n")` and `TOKEN=\nHOME=/root`
# gain nothing. Possessive and bounded per character: linear.
_VGAP = (
    rf"(?:{_SP}*+(?:{_BLOCK}{_SP}*+(?={_NL}))?+(?:{_NL}{_SP}*+)++"
    rf"(?!{_OWN_NAME})(?=\\{{0,8}}[\"']?[A-Za-z0-9_+/.~-])|\s*(?!{_BLOCK}{_SP}*+{_NL}))"
)
# The gap between an Authorization scheme and its token. Raw whitespace has
# always spanned lines (the `\s+` fallback). Escaped line breaks and indents
# do too (`Authorization: Bearer\n\n  <tok>` in a json.dumps copy), blank
# lines included, never into a name of its own (`Bearer\nHost: h`), and only
# before what starts as a token (#289's _SCHEME_GAP on the rc).
_SCHEME_GAP = (
    rf"(?:(?:{_SP}*+{_NL})++{_SP}*+(?!{_OWN_NAME})"
    rf"(?=\\{{0,8}}[\"']?[A-Za-z0-9_+/.~-])|\s+)"
)
# The rest of a line of a YAML block's body: up to a line break, as written
# or escaped, or a JSON string's closing quote (the body of a json.dumps copy
# ends there).
_LINE_REST = r"(?:[^\r\n\\\"]|\\(?!\\*[nr])[\s\S])"
# A YAML block scalar under a name: the indicator line, then every line
# indented at least as far as the first (blank lines in between included),
# as written or escaped. _yaml_block masks each body line when the name is
# one a carrier below masks on one line (`apiKey: |\n  <v>\n  <w>`).
_YAML_BLOCK = re.compile(
    rf"(?P<head>(?<![A-Za-z0-9_.-]){_ESCQ}(?P<name>[A-Za-z_][A-Za-z0-9_.-]{{0,63}})"
    rf"{_ESCQ}[ \t]*:{_SP}*+{_BLOCK}{_SP}*+)"
    rf"(?P<body>(?:{_NL}{_SP}*+(?={_NL}))*+{_NL}(?P<indent>{_SP}++){_LINE_REST}*+"
    rf"(?:(?:{_NL}{_SP}*+(?={_NL}))*+{_NL}(?P=indent){_LINE_REST}*+)*+)"
)
_BLOCK_LINE = re.compile(rf"({_NL}{_SP}*+){_LINE_REST}++")
# Where the word `task-notification` starts, for the generic sk- rule's one
# exception (_TAG_WORD, below): not right after a character that continues
# a word (_IN_WORD: an ASCII letter or digit, `_`, `-`, or any non-ASCII
# character), unless that letter or digit ends an escape that writes a
# character that ends a word, or a colour code. The colour-code part
# is 0.10's key boundary, ported for this alone.
# json.dumps writes a newline, carriage return, tab, backspace or form feed
# as `\n` `\r` `\t` `\b` `\f` and any other control or non-ASCII character
# as `\uXXXX`: that is the copy _redact_field redacts, and text can arrive
# already escaped. Python writes a byte as `\xXX` and a URL a character as
# `%XX`. An escape of an ASCII character other than a letter, a digit, `_`
# or `-` (`\n`, `\u003c`, `\x3c`, `%3C`) ends a word as that character does;
# an escape of anything else continues it, a non-ASCII character included,
# since a pattern cannot tell an escaped letter from escaped punctuation.
# Each lookbehind is fixed-width, as `re` requires; the escapes are checked
# inside the one for the letter or digit (it is tried at every position,
# and a single negative lookbehind is the cheapest form).
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
# A character that continues a word next to `task-notification`: anything
# but ASCII punctuation, space and control characters. Every non-ASCII
# character counts, a letter or digit of any script, every dash (Unicode Pd)
# and U+2212 alike, so 0.9.2's masking holds next to all of them, raw or
# escaped (_IN_WORD_ESCAPE reads an escaped non-ASCII character the same way).
_IN_WORD = r"[^\x00-\x2c\x2e\x2f\x3a-\x40\x5b-\x5e\x60\x7b-\x7f]"
_WORD_START = rf"(?<!{_IN_WORD}{_NOT_AFTER_WORD_END_ESCAPE}{_NOT_AFTER_CSI})"
# The one word the generic sk- rule leaves: Claude Code's `<task-notification>`
# and `</task-notification>` tags (`task-notification` or
# `task-notifications`, exactly, as a whole word: a character that continues
# a word on either side makes another name, `task-notification-id`,
# `task-notification\u2010id`), seen from just after its
# `sk-`. An exact literal, not a shape: no key is the word `notification`.
_TAG_WORD = rf"(?<={_WORD_START}task-)notifications?(?!{_IN_WORD}|{_IN_WORD_ESCAPE})"


# A class's or a def's header right before a carrier's separator: a class
# named like a credential, `class` at a line's start or after whitespace
# (`class Token:`, `    class AccessToken:`), or a def's return annotation
# after its `)` (`def get(self) -> Token:`); prose is no header
# (`maps user -> password:`, `x.class password:`). One fixed-width lookbehind
# per name width, read only where three quotes open the next line.
_CODE_HEADER = "|".join(
    rf"(?<=(?<![^\n \t])class [A-Za-z_][A-Za-z0-9_]{{{n}}}:)"
    rf"|(?<=\) -> [A-Za-z_][A-Za-z0-9_.]{{{n}}}:)"
    for n in range(64)
)
# The gap after a carrier's separator: _VGAP's (whitespace, escaped line
# breaks and blank lines, a YAML block's indicator; #288). A
# later line that opens with three quotes is a docstring's, no value, only
# right after a class's or a def's header (`class Token:\n    """Immutable
# token representation."""`, as the rc's _gap reads it); after any other
# name it is the value (`password:\n    '''<v>'''`).
# On its own line a quoted value is read as before
# (`token: """v"""`).
_NO_DOCSTRING = r"(?!(?=[ \t]*+[\r\n]\s*+(?:\\{0,8}[\"']){3})" + f"(?:{_CODE_HEADER}))"
_SEP_GAP = rf"{_NO_DOCSTRING}{_VGAP}"
# The value under an Authorization-family name stops before the next such
# name and its separator wherever that name's own carrier can start (the
# same boundary: no letter, digit, `_` or `-` before it, an escape that ends
# a word, a colour code): `Proxy-Authorization: none.Authorization: <v>`,
# `none.'Authorization: <v>`, `&#34;Authorization&#34;: none.authorization:
# <v>`. That entry keeps its own carrier, as it did before Proxy-Authorization
# was a carrier and read entity quotes.
# Where it cannot start (`none-authorization=<v>`) the value
# reads on, as before; so it does where that carrier could read no value
# after its separator (`noneAuthorization=\\&#34;<v>\\&#34;`: the value is
# masked with the first one, as on 94e7736d5). Each character checks the
# name first (bounded).
# An Authorization-family name (any
# prefix, as the rc reads it): Authorization or Proxy-Authorization where
# its carrier starts (no letter, digit, `_` or `-` before it, an escape that
# ends a word, a colour code), or any name that ends in it, read from a word's
# start or right after an escaped line break (`HTTP_AUTHORIZATION`,
# `X-Proxy-Authorization`, `x_authorization`, `basicAuthorization`): the word
# is read once (possessive) and its end checked (fixed width), so the check
# stays linear. `X-Authorization-Mode` ends in no such name.
_AUTH_NAMED = (
    rf"(?:(?={_ESCQ}(?:proxy[_-]?)?authorization)"
    rf"(?<![A-Za-z0-9_-]{_NOT_AFTER_WORD_END_ESCAPE}{_NOT_AFTER_CSI})"
    rf"{_ESCQ}(?:proxy[_-]?)?authorization"
    rf"|(?:(?<![A-Za-z0-9_-])|(?<=\\[nrt])){_ESCQ}[A-Za-z0-9_-]{{1,64}}+(?<=authorization))"
)
_AUTH_NEXT = (
    rf"(?={_AUTH_NAMED}{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ}"
    rf"(?:(?:Bearer|Token|Basic|ApiKey)\s+)?{_VALUE_START}"
    rf"(?:[^\"'\s,}}\\*&;|\x1b^%]|{_BRACED_PCT}|{_VALUE_ESCAPE}))"
)
_AUTH_SECVAL = (
    rf"{_VALUE_START}(?:(?!{_AUTH_NEXT})"
    rf"(?:[^\"'\s,}}\\*&;|\x1b^%]|{_BRACED_PCT}|{_SEMI_UNIT}|{_VALUE_ESCAPE}))++"
)


_REDACTION_PATTERNS: list[
    tuple[re.Pattern[str], str | Callable[[re.Match[str]], str], str]
] = [
    # PEM private-key blocks (GCP/Vertex service-account JSON, RSA keys). Redact
    # the whole armored block incl. the base64 body; the carriers below stop at
    # the first space and would leave the key material. Lazy + length-capped body
    # keeps it bounded (#830).
    (
        re.compile(
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
        re.compile(r"sk-ant-[a-zA-Z0-9_-]{12,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # BenchFlow proxy master key (sk-benchflow-<token_urlsafe(24)>), OpenAI service
    # account (sk-svcacct-), OpenRouter (sk-or-v1-): rare labels that won't appear
    # in a normal kebab slug, so the hyphen-permitting entropy class is safe here.
    (
        re.compile(r"sk-(?:benchflow|svcacct|or-v1)-[A-Za-z0-9_-]{12,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # OpenAI org-scoped sk-proj-/sk-admin-: `proj`/`admin` ARE common identifier
    # words, so a bare hyphen class would redact kebab slugs like
    # `sk-proj-refactor-auth`. Real keys are high-entropy base64url and always
    # carry an uppercase char; gate on a lookahead for one so lowercase slugs
    # survive (#830). Before generic sk- so it wins.
    (
        re.compile(
            r"sk-(?:proj|admin)-(?=[A-Za-z0-9_-]{0,200}[A-Z])[A-Za-z0-9_-]{12,}"
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Cerebras: csk-.... The `c` before `sk-` is the key's own, so the key is
    # masked whole here, before the generic rule below would leave the `c`.
    (re.compile(r"csk-[A-Za-z0-9]{20,}"), "***REDACTED***", REDACTION_CATEGORY_API_KEY),
    # OpenAI / generic sk- (alphanumeric only — widening to include `-` would
    # match common slugs like `task-sk-us-east-1-...`), wherever it stands, as
    # in 0.9.2, glued keys included (`mysql -psk-...`, `agent.pysk-...`). The
    # one exception is the word of Claude Code's `<task-notification>` tags
    # (_TAG_WORD); every other word ending in `sk` is masked as before.
    (
        re.compile(rf"sk-(?!{_TAG_WORD})[a-zA-Z0-9]{{12,}}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Google AI / Gemini: AIzaSy... (≥20 char suffix avoids matching `AIzaSy`
    # alone). Prefix is redacted too so the audit grep for `AIzaSy` is clean.
    (
        re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # AWS access keys: AKIA/ASIA + exactly 16 chars; length anchor avoids
    # matching English words like "ASIAPACIFIC".
    (
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Daytona SDK tokens: dtn_... — ≥16 char suffix avoids short ids (`dtn_v2`).
    (
        re.compile(r"dtn_[A-Za-z0-9_]{16,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # GitHub tokens: PATs / OAuth / app / refresh (ghp_/gho_/ghu_/ghs_/ghr_) and
    # fine-grained PATs (github_pat_...). ≥20 char suffix avoids short slugs.
    (
        re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    (
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Slack tokens: bot/user/app/refresh/legacy (xoxb-/xoxp-/xoxa-/xoxr-/xoxs-).
    (
        re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
        "***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # Other provider key families with distinctive prefixes — caught here for the
    # bare-token-in-prose form that no NAME=value carrier sees (a provider
    # exception that echoes the key): gsk_ (Groq), xai-, r8_ (Replicate),
    # hf_ (HuggingFace), fw_ (Fireworks). ≥20-char anchors avoid short ids (#830).
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "***REDACTED***", REDACTION_CATEGORY_API_KEY),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "***REDACTED***", REDACTION_CATEGORY_API_KEY),
    (re.compile(r"r8_[A-Za-z0-9]{20,}"), "***REDACTED***", REDACTION_CATEGORY_API_KEY),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "***REDACTED***", REDACTION_CATEGORY_API_KEY),
    (re.compile(r"fw_[A-Za-z0-9]{20,}"), "***REDACTED***", REDACTION_CATEGORY_API_KEY),
    # JSON Web Tokens (session/bearer creds): three base64url segments split by
    # dots. The ``eyJ`` prefix is base64url of ``{"`` — a JWT header. The 3rd
    # (signature) segment requires ≥20 chars (real HS256/RS256 sigs are 43+) so
    # short dotted base64/method-chains that merely start ``eyJ`` aren't redacted;
    # a left boundary keeps it from starting mid-identifier. Segment upper bounds
    # keep it ReDoS-bounded (#830).
    (
        re.compile(
            # or right after an escaped line break or tab (a JWT that starts a
            # line of a json.dumps copy, `...\neyJ...`).
            r"(?:(?<![A-Za-z0-9_-])|(?<=\\[nrt]))"
            r"eyJ[A-Za-z0-9_-]{10,512}\.[A-Za-z0-9_-]{6,512}\.[A-Za-z0-9_-]{20,512}"
        ),
        "***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
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
        re.compile(
            r"((?:https?|ftp|ftps|postgres|postgresql|mysql|mongodb(?:\+srv)?|"
            r"redis|rediss|amqp|amqps)://)"
            r"[^\s/?#:@]{0,256}:[^\s/?#]{0,256}@"
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
        re.compile(
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
        re.compile(r"([?&]sig=)[^&\s\"'*]{16,}", re.IGNORECASE),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_URL_CREDENTIAL,
    ),
    # A YAML block scalar under a carrier's name (_YAML_BLOCK): every body
    # line, before the carriers below would mask only its first word.
    (
        _YAML_BLOCK,
        lambda match: _yaml_block(match),
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # --- Header / key-value secret carriers ---
    # Match `name: value` / `name=value` in raw, JSON, Python dict-repr
    # (single-quoted), AND json.dumps-escaped (`\"name\": \"value\"`) forms — see
    # _ESCQ. The name is kept; only the value is dropped (#585/#830). No leading
    # boundary on the hyphenated header names — the hyphen keeps them from matching
    # inside underscore variable names.
    (
        re.compile(
            rf"((?:{_ESCQ}(?:x-api-key|x-goog-api-key|api-key){_ESCQ})\s*[:=]{_SEP_GAP}{_ESCQ})"
            rf"{_SECVAL}",
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
        re.compile(
            rf"({_ESCQ}api[_-]?key{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_API_KEY,
    ),
    # `master_key`/`master-key` and `private_key`/`private-key` carriers — the
    # BenchFlow proxy master key and GCP/Vertex service-account private-key field
    # as labelled values (the PEM body itself is scrubbed by the block rule above).
    (
        re.compile(
            rf"({_ESCQ}(?:master|private)[_-]key{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # Azure's and Cosmos's primary and secondary keys (`primary_key`,
    # `primaryKey`, `PRIMARY_KEY`, `primary-key`, `primaryMasterKey`, the same
    # for secondary, after any prefix: `AZURE_STORAGE_PRIMARY_KEY`),
    # as the rc reads them: under a key-shaped
    # value (24 or more of a key's characters), so a database's
    # `primary_key=True` or `"primaryKey": "id"` is no credential.
    (
        re.compile(
            rf"({_ESCQ}(?:primary|secondary)(?:[_-]?master)?[_-]?key{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ})"
            rf"(?=[A-Za-z0-9+/=_-]{{24}}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # Bearer-token env vars whose NAME has BEARER_TOKEN in the MIDDLE, e.g.
    # (A scheme word before its token is kept with the name, as the rc's
    # {_SCHEME} reads it: `BEARER_TOKEN_AUTHORIZATION: Bearer <token>`.)
    # `AWS_BEARER_TOKEN_BEDROCK=` — the secret-suffix rule below anchors its marker
    # at the name end, so a TOKEN in the middle slips through. `BEARER_TOKEN` is an
    # unambiguous secret signal, safe as a substring (#830). Names length-capped.
    (
        re.compile(
            rf"(?<![A-Za-z0-9])"
            rf"({_ESCQ}{_NAME}BEARER_TOKEN{_NAME}{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ}"
            rf"(?:(?:Bearer|Token|Basic|ApiKey)[ \t]++(?=[^\s\"']))?+){_SECVAL}",
            re.IGNORECASE,
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
        re.compile(
            rf"(?<![A-Za-z0-9])({_ESCQ}{_NAME}"
            r"(?:TOKEN|SECRET|PASSWORD|PASSWD|"
            r"ACCESS_KEY|SECRET_KEY|ACCOUNT_KEY|PRIVATE_KEY|ENCRYPTION_KEY|"
            r"CREDENTIALS?)"
            rf"{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_CREDENTIAL_FIELD,
    ),
    # authorization header WITH a scheme (Bearer/Token/Basic/…): keep scheme.
    # Proxy-Authorization carries the same credentials, in every spelling
    # (proxy_authorization, proxyAuthorization, PROXY_AUTHORIZATION, the
    # HTTP/2 proxy-authorization).
    # Its name may follow a terminal colour code (`\033[1mAuthorization:`),
    # whose last letter starts no word (_NOT_AFTER_CSI), or an escape
    # that ends a word, as _WORD_START reads one (_NOT_AFTER_WORD_END_ESCAPE:
    # `\r\nAuthorization:` in a JSON copy of headers, `\tAuthorization:`,
    # `\u0026Authorization:`). An escape that writes a letter or a digit
    # (`\x41`, `\u0041`) continues a word, so no name starts after it.
    # Its scheme may follow escaped line breaks (`=\n\nBearer <token>` in a
    # json.dumps copy of a header on its own line). Its value is _AUTH_SECVAL.
    (
        re.compile(
            rf"({_AUTH_NAMED}{_ESCQ}\s*[:=]{_VGAP}"
            rf"(?:(?:\\{{1,8}}[nrt])++\s*)?{_ESCQ}(?:Bearer|Token|Basic|ApiKey){_SCHEME_GAP}){_AUTH_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
    # authorization header with a bare value (no recognized scheme). The negative
    # lookahead skips scheme-prefixed values already handled above, so
    # `Bearer <tok>` isn't double-redacted into `***REDACTED*** ***...`.
    (
        re.compile(
            rf"({_AUTH_NAMED}{_ESCQ}\s*[:=]{_SEP_GAP}{_ESCQ})"
            rf"(?!(?:Bearer|Token|Basic|ApiKey)\b|[rRuUbBfF]{{1,3}}[\"'])"
            rf"{_AUTH_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
        REDACTION_CATEGORY_BEARER_TOKEN,
    ),
]


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


# A replacement template with a group reference (`\1***REDACTED***`), as a
# function, made once per template. `Pattern.subn` looks a str template up in
# a cache keyed by the pattern, and a pattern's hash covers its whole compiled
# code, so on a long carrier (the value class) every call paid for it,
# matched or not; `Match.expand` reads the same template, only on a match.
_TEMPLATE_FUNCTIONS: dict[str, Any] = {}


def _replacement(template: Any) -> Any:
    if not isinstance(template, str) or "\\" not in template:
        return template
    function = _TEMPLATE_FUNCTIONS.get(template)
    if function is None:
        function = _TEMPLATE_FUNCTIONS[template] = lambda match: match.expand(template)
    return function


def redact_trajectory_text_with_count(text: str) -> tuple[str, int]:
    """Apply the canonical patterns and report how many matches were replaced."""
    text, categories = redact_trajectory_text_with_categories(text)
    return text, sum(categories.values())


def redact_trajectory_text_with_categories(text: str) -> tuple[str, dict[str, int]]:
    """Apply the canonical patterns and report replacements per category.

    Categories name the kind of secret each rule actually detects (API key,
    bearer token, private key block, URL credential, credential-bearing field
    value) so upload previews can itemize what was masked without inventing
    detection that does not exist. The total replacement count is the sum of
    the returned values.
    """
    memo = _redaction_memo.get()
    cached = memo.get(text) if memo is not None else None
    if cached is not None:
        return cached[0], dict(cached[1])
    original = text
    categories: dict[str, int] = {}
    for pattern, replacement, category in _REDACTION_PATTERNS:
        if isinstance(replacement, str):
            text, count = pattern.subn(_replacement(replacement), text)
        else:
            text, count = _subn_changed(pattern, replacement, text)
        if count:
            categories[category] = categories.get(category, 0) + count
    if memo is not None:
        memo.remember(original, text, categories)
    return text, categories


def _subn_changed(
    pattern: re.Pattern[str], replace: Callable[[re.Match[str]], str], text: str
) -> tuple[str, int]:
    """``pattern.subn`` with a function, counting only the matches it changed."""
    changed = 0

    def counted(match: re.Match[str]) -> str:
        nonlocal changed
        out = replace(match)
        changed += out != match[0]
        return out

    return pattern.subn(counted, text)[0], changed


@functools.lru_cache(maxsize=4096)
def _carrier_name(name: str) -> bool:
    """Whether a carrier masks this name's value on one line (`<name>: <v>`)."""
    probe = name + ": QzCarrierProbeValueQz"
    for pattern, replacement, _ in _REDACTION_PATTERNS:
        if not callable(replacement):
            probe = pattern.sub(replacement, probe)
    return "QzCarrierProbeValueQz" not in probe


def _yaml_block(match: re.Match[str]) -> str:
    if not _carrier_name(match["name"]):
        return match[0]
    return match["head"] + _BLOCK_LINE.sub(r"\1***REDACTED***", match["body"])


def _mask_string_leaves(value: Any) -> Any:
    """Every string leaf of a value held under a credential's name, masked."""
    if isinstance(value, dict):
        return {key: _mask_string_leaves(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_string_leaves(item) for item in value]
    if isinstance(value, str) and value:
        return "***REDACTED***"
    return value


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
    if isinstance(value, (dict, list)):
        # An object or list under a credential's name (`{"password": {"a":
        # "<v>"}}`, `{"secret": ["<v>"]}`): every string in it is the secret's.
        if isinstance(key, str) and _carrier_name(key):
            return _mask_string_leaves(value)
        return redact_trajectory_obj(value)
    if not isinstance(value, str):
        return value
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
    def total_input_tokens(self) -> int:
        """Uncached input tokens over the captured exchanges."""
        return sum(_exchange_token_usage(ex).input_tokens for ex in self.exchanges)

    @property
    def total_output_tokens(self) -> int:
        """Output tokens over the captured exchanges."""
        return sum(_exchange_token_usage(ex).output_tokens for ex in self.exchanges)

    @property
    def total_cache_read_tokens(self) -> int:
        """Prompt-cache read tokens over the captured exchanges."""
        return sum(_exchange_token_usage(ex).cache_read_tokens for ex in self.exchanges)

    @property
    def total_cache_creation_tokens(self) -> int | None:
        """Prompt-cache write tokens over the captured exchanges; None when an
        answered call named no cache-write count (unknown, not 0)."""
        if self.cache_writes_unreported():
            return None
        return sum(
            _exchange_token_usage(ex).cache_creation_tokens for ex in self.exchanges
        )

    def cache_writes_named(self) -> tuple[int, bool]:
        """(the writes the answered calls that named a count reported, whether
        every answered call that named none held its cache in its input
        count, so that its unknown writes sit inside ``input_tokens``)."""
        named, inside = 0, True
        for ex in self.exchanges:
            usage = _exchange_token_usage(ex)
            if not usage.split_reported:
                continue
            if usage.cache_write_reported:
                named += usage.cache_creation_tokens
            elif not usage.input_holds_cache:
                inside = False
        return named, inside

    def cache_writes_unreported(self) -> list[str]:
        """The models of the answered calls that named no cache-write count,
        once each, in order ("" for a call with no model named)."""
        models: list[str] = []
        for ex in self.exchanges:
            usage = _exchange_token_usage(ex)
            if usage.split_reported and not usage.cache_write_reported:
                metadata = ex.metadata if isinstance(ex.metadata, dict) else {}
                model = str(
                    metadata.get("provider_model")
                    or metadata.get("request_model")
                    or ex.response.body.get("model")
                    or ""
                )
                if model not in models:
                    models.append(model)
        return models

    @property
    def total_provider_tokens(self) -> int:
        """All tokens over the captured exchanges."""
        return sum(_exchange_token_usage(ex).total_tokens for ex in self.exchanges)

    def unreported_counts(self) -> set[str]:
        """``input`` and/or ``output`` when some answered call reported the
        other count but not this one (an output-only response): that total
        is then unknown, not the sum of the calls that named it."""
        missing: set[str] = set()
        for ex in self.exchanges:
            usage = _exchange_token_usage(ex)
            if not usage.split_reported:
                continue
            if not usage.input_reported:
                missing.add("input")
            if not usage.output_reported:
                missing.add("output")
        return missing

    @property
    def split_unreported(self) -> bool:
        """Whether some answered call named a total but no input or output
        count: the trial's split is then unknown."""
        for ex in self.exchanges:
            usage = _exchange_token_usage(ex)
            if not usage.split_reported and usage.total_tokens:
                return True
        return False

    def cache_write_split(self) -> tuple[int | None, int | None]:
        """(5-minute, 1-hour) cache-write tokens over the captured exchanges;
        (None, None) unless some exchange reported the split (only Anthropic
        does) and every exchange that wrote to the cache did."""
        five = hour = 0
        reported = False
        if self.cache_writes_unreported():
            return None, None
        for ex in self.exchanges:
            usage = _exchange_token_usage(ex)
            split = (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens)
            if split == (None, None):
                if usage.cache_creation_tokens:
                    return None, None
                continue
            reported = True
            five += split[0] or 0
            hour += split[1] or 0
        return (five, hour) if reported else (None, None)

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
