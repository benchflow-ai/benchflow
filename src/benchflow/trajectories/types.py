"""Trajectory types — raw LLM API request/response pairs captured from providers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from benchflow.trajectories import redaction as _redaction

# Keep the historical ``benchflow.trajectories.types`` redaction imports
# working for downstream callers. The implementation lives in the standalone,
# stdlib-only module so the exact same code can be deployed beside LiteLLM.
REDACTION_CATEGORY_API_KEY = _redaction.REDACTION_CATEGORY_API_KEY
REDACTION_CATEGORY_BEARER_TOKEN = _redaction.REDACTION_CATEGORY_BEARER_TOKEN
REDACTION_CATEGORY_CREDENTIAL_FIELD = _redaction.REDACTION_CATEGORY_CREDENTIAL_FIELD
REDACTION_CATEGORY_PASSWORD = _redaction.REDACTION_CATEGORY_PASSWORD
REDACTION_CATEGORY_PRIVATE_KEY = _redaction.REDACTION_CATEGORY_PRIVATE_KEY
REDACTION_CATEGORY_URL_CREDENTIAL = _redaction.REDACTION_CATEGORY_URL_CREDENTIAL
redact_trajectory_obj = _redaction.redact_trajectory_obj
redact_trajectory_text = _redaction.redact_trajectory_text
redact_trajectory_text_with_categories = (
    _redaction.redact_trajectory_text_with_categories
)
redact_trajectory_text_with_count = _redaction.redact_trajectory_text_with_count

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
    reported_additive_cache_read = _first_int(
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
    cache_read_tokens = reported_additive_cache_read or inclusive_cache_read
    cache_creation_tokens = additive_cache_creation

    # Some normalized provider responses expose the same cache read in both
    # families (for example Gemini pass-through: ``prompt_tokens_details`` and
    # ``cache_read_input_tokens``). The inclusive marker proves the raw prompt
    # count already contains cache reads, so never add the duplicate field.
    additive_cache_read = 0 if inclusive_cache_read else reported_additive_cache_read

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
