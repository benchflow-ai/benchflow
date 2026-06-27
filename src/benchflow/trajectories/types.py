"""Trajectory types — raw LLM API request/response pairs captured from providers."""

import re
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
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    provider_total_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        if self.provider_total_tokens is not None:
            return self.provider_total_tokens
        # ``input_tokens`` is normalized to already include cache reads/writes
        # (see ``_exchange_token_usage``), so the total is just input + output;
        # re-adding the cache breakdown here would double-count it.
        return self.input_tokens + self.output_tokens


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
    additive_cache_read = _first_int(
        usage.get("cache_read_input_tokens"),
        usage.get("cacheReadInputTokens"),
        usage.get("cacheReadInputTokenCount"),
    )
    additive_cache_creation = _first_int(
        usage.get("cache_creation_input_tokens"),
        usage.get("cacheWriteInputTokens"),
        usage.get("cacheWriteInputTokenCount"),
    )
    # Cache reported as a SUBSET already inside the input count — OpenAI
    # (`*_tokens_details.cached_tokens`) and Gemini (`cachedContentTokenCount`).
    inclusive_cache_read = _first_int(
        prompt_details.get("cached_tokens"),
        input_details.get("cached_tokens"),
        usage_metadata.get("cachedContentTokenCount"),
    )
    cache_read_tokens = additive_cache_read or inclusive_cache_read
    cache_creation_tokens = additive_cache_creation

    # Normalize `input_tokens` to mean the same thing across providers: the total
    # input the model processed, cache included. Anthropic/Bedrock report the
    # UNCACHED delta with cache as a separate additive component, so fold the
    # additive cache in; OpenAI/Gemini already report the cache-inclusive total
    # (their cache is a subset of it). This makes cross-provider usage and cost
    # apples-to-apples; cache_read/cache_creation stay broken out as subsets of
    # the input for pricing.
    raw_input = _first_int(
        usage.get("input_tokens"),
        usage.get("prompt_tokens"),
        usage.get("inputTokens"),
        usage_metadata.get("promptTokenCount"),
    )
    # Gemini reports tool-use prompt tokens (`toolUsePromptTokenCount`) separately
    # from `promptTokenCount` — it is additive input, NOT a subset — so fold it in
    # too, or tool-heavy Gemini runs underreport input/cost (and totalTokenCount
    # would exceed input + output). Absent for every other provider.
    additive_tool_use_prompt = _first_int(usage_metadata.get("toolUsePromptTokenCount"))
    input_tokens = (
        raw_input
        + additive_cache_read
        + additive_cache_creation
        + additive_tool_use_prompt
    )

    # Reasoning/thinking tokens are billed as output. Anthropic/OpenAI already
    # fold them into output_tokens/completion_tokens; Gemini reports them
    # separately as `thoughtsTokenCount`, so add them in for output parity.
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
    )


# Quote atom for the header/key-value carriers below: a run of 0-8 optional
# backslashes followed by an optional quote. This makes every carrier fire on raw
# text (``"k": "v"``), Python dict-repr (``'k': 'v'``), AND json.dumps-escaped
# JSON at any nesting depth (``\"k\": \"v\"``, ``\\\"k\\\": …``) — the escaped
# form is how a provider error body embedded in ``error.message`` reaches the
# redactor after ``Trajectory.to_jsonl`` runs ``json.dumps`` over the record
# before redacting it (#830). The ``{0,8}`` cap keeps it ReDoS-bounded.
_ESCQ = r'\\{0,8}["\']?'
# Secret value class for the carriers: stops at a quote (plain, or the backslash
# of an escaped closing quote), whitespace, and the ``,`` ``}`` ``&`` delimiters
# so a redacted value can never swallow a sibling JSON field or URL query param;
# excludes ``*`` so an already-redacted value isn't re-matched.
_SECVAL = r"[^\"'\s,}\\*&]+"
# Name-prefix class for the env/JSON carriers, LENGTH-CAPPED. An uncapped
# ``[A-Za-z0-9_]*`` before a required marker backtracks O(n²) on a long
# alphanumeric run (e.g. a base64 image field), so a benign trajectory could
# stall the redactor for tens of seconds. Real env-var/header names are short;
# ``{0,64}`` makes each match attempt O(1) → overall O(n) (#830 ReDoS fix).
_NAME = r"[A-Za-z0-9_]{0,64}"

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
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
    ),
    # --- Token families: redacted WHOLE (prefix included) so the v0.5 leak audit
    # greps (``AIzaSy``/``dtn_``…) see no live-key shape (#537/#585). ---
    # Anthropic: sk-ant-api03-...
    (re.compile(r"sk-ant-[a-zA-Z0-9_-]{12,}"), "***REDACTED***"),
    # BenchFlow proxy master key (sk-benchflow-<token_urlsafe(24)>), OpenAI service
    # account (sk-svcacct-), OpenRouter (sk-or-v1-): rare labels that won't appear
    # in a normal kebab slug, so the hyphen-permitting entropy class is safe here.
    (
        re.compile(r"sk-(?:benchflow|svcacct|or-v1)-[A-Za-z0-9_-]{12,}"),
        "***REDACTED***",
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
    ),
    # OpenAI / generic sk- (alphanumeric only — widening to include `-` would
    # match common slugs like `task-sk-us-east-1-...`)
    (re.compile(r"sk-[a-zA-Z0-9]{12,}"), "***REDACTED***"),
    # Google AI / Gemini: AIzaSy... (≥20 char suffix avoids matching `AIzaSy`
    # alone). Prefix is redacted too so the audit grep for `AIzaSy` is clean.
    (re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}"), "***REDACTED***"),
    # AWS access keys: AKIA/ASIA + exactly 16 chars; length anchor avoids
    # matching English words like "ASIAPACIFIC".
    (re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"), "***REDACTED***"),
    # Daytona SDK tokens: dtn_... — ≥16 char suffix avoids short ids (`dtn_v2`).
    (re.compile(r"dtn_[A-Za-z0-9_]{16,}"), "***REDACTED***"),
    # GitHub tokens: PATs / OAuth / app / refresh (ghp_/gho_/ghu_/ghs_/ghr_) and
    # fine-grained PATs (github_pat_...). ≥20 char suffix avoids short slugs.
    (re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"), "***REDACTED***"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "***REDACTED***"),
    # Slack tokens: bot/user/app/refresh/legacy (xoxb-/xoxp-/xoxa-/xoxr-/xoxs-).
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "***REDACTED***"),
    # Other provider key families with distinctive prefixes — caught here for the
    # bare-token-in-prose form that no NAME=value carrier sees (a provider
    # exception that echoes the key). Groq gsk_, xAI xai-, Replicate r8_,
    # HuggingFace hf_, Fireworks fw_. ≥20-char anchors avoid short ids (#830).
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "***REDACTED***"),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "***REDACTED***"),
    (re.compile(r"r8_[A-Za-z0-9]{20,}"), "***REDACTED***"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "***REDACTED***"),
    (re.compile(r"fw_[A-Za-z0-9]{20,}"), "***REDACTED***"),
    # JSON Web Tokens (session/bearer creds): three base64url segments split by
    # dots. The ``eyJ`` prefix is base64url of ``{"`` — a JWT header. The 3rd
    # (signature) segment requires ≥20 chars (real HS256/RS256 sigs are 43+) so
    # short dotted base64/method-chains that merely start ``eyJ`` aren't redacted;
    # a left boundary keeps it from starting mid-identifier. Segment upper bounds
    # keep it ReDoS-bounded (#830).
    (
        re.compile(
            r"(?<![A-Za-z0-9_-])"
            r"eyJ[A-Za-z0-9_-]{10,512}\.[A-Za-z0-9_-]{6,512}\.[A-Za-z0-9_-]{20,512}"
        ),
        "***REDACTED***",
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
    (
        re.compile(
            r"([?&](?:"
            r"api_?key|access_key|access_token|refresh_token|session_token|"
            r"session_id|sessionid|client_secret|aws_secret_access_key|account_key|"
            r"secret|password_hash|password|passwd|"
            r"signature|x-amz-signature|x-amz-credential|x-amz-security-token|sas"
            r")=)[^&\s\"']+",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
    ),
    # Azure SAS `?sig=<token>` — bare `sig` is excluded from the curated list above
    # (a `?sig=v2` scheme-version flag is a common non-secret), so gate on a
    # high-entropy value length: a real SAS signature is a 40+ char base64 HMAC,
    # while a version flag is short (#830).
    (
        re.compile(r"([?&]sig=)[^&\s\"']{16,}", re.IGNORECASE),
        r"\1***REDACTED***",
    ),
    # --- Header / key-value secret carriers ---
    # Match `name: value` / `name=value` in raw, JSON, Python dict-repr
    # (single-quoted), AND json.dumps-escaped (`\"name\": \"value\"`) forms — see
    # _ESCQ. The name is kept; only the value is dropped (#585/#830). No leading
    # boundary on the hyphenated header names — the hyphen keeps them from matching
    # inside underscore variable names.
    (
        re.compile(
            rf"((?:{_ESCQ}(?:x-api-key|x-goog-api-key|api-key){_ESCQ})\s*[:=]\s*{_ESCQ})"
            rf"{_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
    ),
    # Underscore form `api_key`. No leading boundary: namespaced env dumps like
    # `GEMINI_API_KEY=secret` / JSON keys such as `"openai_api_key"` must redact
    # too (#585). The `\1` capture preserves the matched name+separator.
    (
        re.compile(
            rf"({_ESCQ}api_key{_ESCQ}\s*[:=]\s*{_ESCQ}){_SECVAL}", re.IGNORECASE
        ),
        r"\1***REDACTED***",
    ),
    # `master_key`/`master-key` and `private_key`/`private-key` carriers — the
    # BenchFlow proxy master key and GCP/Vertex service-account private-key field
    # as labelled values (the PEM body itself is scrubbed by the block rule above).
    (
        re.compile(
            rf"({_ESCQ}(?:master|private)[_-]key{_ESCQ}\s*[:=]\s*{_ESCQ}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
    ),
    # Bearer-token env vars whose NAME has BEARER_TOKEN in the MIDDLE, e.g.
    # `AWS_BEARER_TOKEN_BEDROCK=` — the secret-suffix rule below anchors its marker
    # at the name end, so a TOKEN in the middle slips through. `BEARER_TOKEN` is an
    # unambiguous secret signal, safe as a substring (#830). Names length-capped.
    (
        re.compile(
            rf"(?<![A-Za-z0-9])"
            rf"({_ESCQ}{_NAME}BEARER_TOKEN{_NAME}{_ESCQ}\s*[:=]\s*{_ESCQ}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
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
            rf"{_ESCQ}\s*[:=]\s*{_ESCQ}){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
    ),
    # authorization header WITH a scheme (Bearer/Token/Basic/…): keep scheme.
    (
        re.compile(
            rf"(?<![A-Za-z0-9_-])({_ESCQ}authorization{_ESCQ}\s*[:=]\s*{_ESCQ}"
            rf"(?:Bearer|Token|Basic|ApiKey)\s+){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
    ),
    # authorization header with a bare value (no recognized scheme). The negative
    # lookahead skips scheme-prefixed values already handled above, so
    # `Bearer <tok>` isn't double-redacted into `***REDACTED*** ***...`.
    (
        re.compile(
            rf"(?<![A-Za-z0-9_-])({_ESCQ}authorization{_ESCQ}\s*[:=]\s*{_ESCQ})"
            rf"(?!(?:Bearer|Token|Basic|ApiKey)\b){_SECVAL}",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
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
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_acp_trajectory_jsonl(trajectory: list[dict[str, Any]]) -> str:
    """Serialize an ACP trajectory list to redacted JSONL.

    Each event is JSON-encoded and then run through ``redact_trajectory_text``
    so secrets the agent echoed into ``acp_trajectory.jsonl`` (env dumps, curl
    -v output, header logs) are stripped before the file is written or uploaded
    (#537/#585). Returns the joined lines without a trailing newline; callers
    that need one (e.g. uploaded copies) append it themselves.
    """
    import json

    return "\n".join(
        redact_trajectory_text(json.dumps(event, default=str)) for event in trajectory
    )


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
    verifiers_step: dict[str, Any] | None = None
    verifiers_tool_defs: list[dict[str, Any]] = Field(default_factory=list)


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
        return sum(_exchange_token_usage(ex).input_tokens for ex in self.exchanges)

    @property
    def total_output_tokens(self) -> int:
        return sum(_exchange_token_usage(ex).output_tokens for ex in self.exchanges)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(_exchange_token_usage(ex).cache_read_tokens for ex in self.exchanges)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(
            _exchange_token_usage(ex).cache_creation_tokens for ex in self.exchanges
        )

    @property
    def total_provider_tokens(self) -> int:
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
        """Export as JSONL (one exchange per line)."""
        import json

        lines = []
        for ex in self.exchanges:
            data = ex.model_dump(mode="json")
            raw = json.dumps(data, default=str)
            if redact_keys:
                raw = redact_trajectory_text(raw)
            lines.append(raw)
        return "\n".join(lines)
