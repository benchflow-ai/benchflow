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

from benchflow.usage_tracking import usage_unavailable


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
    503: ProviderFailure(503, "provider unavailable"),
}


def _provider_failure_from_status(status: Any) -> ProviderFailure | None:
    try:
        return _PROVIDER_FAILURES.get(int(status))
    except (TypeError, ValueError):
        return None


def _provider_failure_from_runtime(runtime: Any) -> ProviderFailure | None:
    """Return the latest provider failure from a usage runtime trajectory.

    Scans the captured provider HTTP exchanges for provider-owned failures that
    ACP agents often wrap as a generic ``-32603 Internal error``. Only the
    sanitized status code and category marker are surfaced, never response
    bodies or headers, so no credential material reaches ``result.error``
    (#546/#564). Returns ``None`` when there is no runtime, no trajectory, or no
    recognized provider-failure status.
    """
    server = getattr(runtime, "server", None)
    trajectory = getattr(server, "trajectory", None)
    exchanges = getattr(trajectory, "exchanges", None) or []
    for exchange in reversed(exchanges):
        status = getattr(getattr(exchange, "response", None), "status_code", None)
        failure = _provider_failure_from_status(status)
        if failure is not None:
            return failure
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
