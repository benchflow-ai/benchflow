"""Native-ACP token usage accounting for :mod:`benchflow.rollout`.

These helpers compute per-prompt usage deltas from a native-ACP agent's
cumulative snapshots and read a provider auth-failure status from a usage
runtime's captured HTTP exchanges. They are split out of ``rollout.py`` so the
token-accounting math is independently testable; the names are re-exported from
:mod:`benchflow.rollout` so existing imports keep resolving unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchflow.usage_tracking import (
    USAGE_SOURCE_AGENT_REPORTED,
    USAGE_SOURCE_NONE,
    is_token_usage_available,
    usage_unavailable,
)


@dataclass(frozen=True)
class ProviderFailure:
    """Sanitized provider failure recovered from captured LLM exchanges."""

    status: int
    marker: str

    @property
    def error_suffix(self) -> str:
        return f"{self.marker} (HTTP {self.status})"


_PROVIDER_FAILURES: dict[int, ProviderFailure] = {
    400: ProviderFailure(400, "provider rejected request"),
    401: ProviderFailure(401, "provider auth failed"),
    403: ProviderFailure(403, "provider auth failed"),
    429: ProviderFailure(429, "provider rate limited"),
    # A provider's server error, like its 503: the agent's ACP error then is
    # the provider's, retried as infrastructure, not the agent's own (a 500
    # read as acp_error, "the agent reported an error"). 529 is Anthropic's
    # overloaded_error.
    **{
        status: ProviderFailure(status, "provider unavailable")
        for status in (500, 502, 503, 504, 529)
    },
}


def _provider_failure_from_status(status: Any) -> ProviderFailure | None:
    try:
        return _PROVIDER_FAILURES.get(int(status))
    except (TypeError, ValueError):
        return None


def _provider_failure_from_runtime(runtime: Any) -> ProviderFailure | None:
    """Return the provider failure the trial's last answered call ended with.

    Reads the captured provider HTTP exchanges for the provider-owned failure
    an ACP agent often wraps as a generic ``-32603 Internal error``. Only the
    latest exchange with a status decides: a failure the agent's later calls
    got past (a 500 it retried, then 40 calls that answered) is not why it
    stopped, and its own error then stays its own. Only the sanitized status
    code and category marker are surfaced, never response bodies or headers,
    so no credential material reaches ``result.error`` (#546/#564). Returns
    ``None`` when there is no runtime, no trajectory, or the last call did not
    fail with a recognized provider-failure status.
    """
    server = getattr(runtime, "server", None)
    trajectory = getattr(server, "trajectory", None)
    exchanges = getattr(trajectory, "exchanges", None) or []
    for exchange in reversed(exchanges):
        status = getattr(getattr(exchange, "response", None), "status_code", None)
        if status is None:
            continue  # no answer at all: the call before it decides
        return _provider_failure_from_status(status)
    return None


def _provider_auth_status_from_runtime(runtime: Any) -> int | None:
    """Backward-compatible auth-only view of provider failure extraction.

    Kept for callers added by PR #564 that only care about the 401/403 status.
    """
    failure = _provider_failure_from_runtime(runtime)
    if failure is not None and failure.marker == "provider auth failed":
        return failure.status
    return None


def _api_error_subcategory(status: int) -> tuple[str, bool]:
    """Map a provider HTTP failure status to (subcategory, transient).

    Status-code-only by design — same #546/#564 security posture as the
    401/403 scan above (never read bodies or headers, so no credential
    material can leak into ``result.error``).
    """
    if status in (401, 403):
        return "auth", False
    if status == 402:
        return "quota", False
    if status == 404:
        return "model_not_found", False
    if status == 429:
        return "rate_limit", True
    if status >= 500 or status == 408:
        return "provider_error", True
    return "rejected_request", False


def _provider_api_failure_summary_from_runtime(runtime: Any) -> dict[str, Any] | None:
    """Summarize provider HTTP failures from a usage runtime's trajectory.

    Returns ``None`` when there is no runtime/trajectory or no captured
    exchanges; otherwise a dict with request totals, per-status failure
    counts, and the dominant failure's (subcategory, transient, fingerprint)
    classification. Reads only integer status codes (#546/#564).
    """
    server = getattr(runtime, "server", None)
    trajectory = getattr(server, "trajectory", None)
    exchanges = getattr(trajectory, "exchanges", None) or []
    total = 0
    failed: dict[int, int] = {}
    last_failed_status: int | None = None
    for exchange in exchanges:
        status = getattr(getattr(exchange, "response", None), "status_code", None)
        if not isinstance(status, int):
            continue
        total += 1
        if status >= 400:
            failed[status] = failed.get(status, 0) + 1
            last_failed_status = status
    if total == 0:
        return None
    summary: dict[str, Any] = {
        "total_requests": total,
        "failed_requests": sum(failed.values()),
    }
    if failed:
        dominant = max(
            failed.items(), key=lambda kv: (kv[1], kv[0] == last_failed_status)
        )[0]
        subcategory, transient = _api_error_subcategory(dominant)
        summary.update(
            status_counts={str(k): v for k, v in sorted(failed.items())},
            dominant_status=dominant,
            subcategory=subcategory,
            transient=transient,
            fingerprint=f"{subcategory}:{dominant}",
        )
    return summary


def classify_api_failure(
    summary: dict[str, Any] | None,
    *,
    total_tokens: int,
    n_tool_calls: int,
) -> tuple[str | None, dict[str, Any]]:
    """Decide the post-rollout API-error verdict for an error-free rollout.

    Returns ``("api_error", summary)`` when the proxy captured provider
    requests and every one of them failed while the agent produced zero
    tokens (proxy-proven); ``("suspected_api_error", {...})`` when there is
    no proxy failure evidence but the agent ended with zero tokens AND zero
    tool calls (zero-signal heuristic — e.g. an agent that validates the
    model id locally and never issues a request); ``(None, {})`` otherwise.
    A rollout with any real token usage or tool activity is never flagged.
    """
    summary = summary or {}
    failed = summary.get("failed_requests") or 0
    total = summary.get("total_requests") or 0
    if failed and failed == total and total_tokens == 0:
        return "api_error", summary
    if total_tokens == 0 and n_tool_calls == 0:
        return "suspected_api_error", {
            "total_requests": total,
            "failed_requests": failed,
        }
    return None, {}


_NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT = {
    "input_tokens": "n_input_tokens",
    "output_tokens": "n_output_tokens",
    "cached_read_tokens": "n_cache_read_tokens",
    "cached_write_tokens": "n_cache_creation_tokens",
    "total_tokens": "total_tokens",
}


def _zero_native_acp_usage_metrics() -> dict[str, Any]:
    return {**usage_unavailable(), "usage_details": {"thought_tokens": 0}}


def _zero_agent_reported_usage_metrics() -> dict[str, Any]:
    """No usage from an agent that routes its own model calls."""
    return {**_zero_native_acp_usage_metrics(), "usage_source": USAGE_SOURCE_NONE}


def _add_agent_reported_usage(
    metrics: dict[str, Any], delta: dict[str, int]
) -> dict[str, Any]:
    """``metrics`` plus one ACP usage delta of an agent that routes its own
    model calls: agent-reported, so unpriced and never trusted."""
    out = dict(metrics)
    for snapshot_field, result_field in _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT.items():
        out[result_field] = _as_nonnegative_int(out.get(result_field)) + (
            delta.get(snapshot_field) or 0
        )
    details = dict(out.get("usage_details") or {})
    details["thought_tokens"] = _as_nonnegative_int(details.get("thought_tokens")) + (
        delta.get("thought_tokens") or 0
    )
    out.update(
        usage_details=details,
        usage_source=USAGE_SOURCE_AGENT_REPORTED,
        cost_usd=None,
        price_source=None,
    )
    return out


def _with_agent_reported_usage(
    usage: dict[str, Any], reported: dict[str, Any]
) -> dict[str, Any]:
    """A trial's usage once an agent that routes its own model calls ran.

    Trusted usage of the trial's other roles (metered or native) keeps its
    totals: the agent's own report goes beside them in ``usage_details``,
    never into them, and the cost is unknown because those totals miss the
    agent's calls. Without trusted usage, the trial's usage is the agent's
    report: ``agent-reported``, or ``none`` when it reported nothing.
    """
    if not is_token_usage_available(usage):
        return dict(reported)
    details = {
        **(usage.get("usage_details") or {}),
        USAGE_SOURCE_AGENT_REPORTED: dict(reported),
    }
    return {**usage, "usage_details": details, "cost_usd": None, "price_source": None}


def _as_nonnegative_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float | str | bytes | bytearray):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    try:
        return max(int(str(value)), 0)
    except ValueError:
        return 0


def _native_acp_usage_delta(
    previous: dict[str, int | None] | None,
    current: dict[str, int | None],
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for usage_field in (
        "input_tokens",
        "output_tokens",
        "cached_read_tokens",
        "cached_write_tokens",
        "thought_tokens",
    ):
        current_value = _as_nonnegative_int(current.get(usage_field))
        previous_value = (
            _as_nonnegative_int(previous.get(usage_field)) if previous else 0
        )
        delta[usage_field] = max(current_value - previous_value, 0)

    current_total = current.get("total_tokens")
    if current_total is not None:
        current_value = _as_nonnegative_int(current_total)
        previous_value = (
            _as_nonnegative_int(previous.get("total_tokens"))
            if previous and previous.get("total_tokens") is not None
            else 0
        )
        delta["total_tokens"] = max(current_value - previous_value, 0)
    else:
        delta["total_tokens"] = (
            delta["input_tokens"]
            + delta["output_tokens"]
            + delta["cached_read_tokens"]
            + delta["cached_write_tokens"]
            + delta["thought_tokens"]
        )
    return delta
