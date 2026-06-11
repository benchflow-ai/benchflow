"""Native-ACP token usage accounting for :mod:`benchflow.rollout`.

These helpers compute per-prompt usage deltas from a native-ACP agent's
cumulative snapshots and read a provider auth-failure status from a usage
runtime's captured HTTP exchanges. They are split out of ``rollout.py`` so the
token-accounting math is independently testable; the names are re-exported from
:mod:`benchflow.rollout` so existing imports keep resolving unchanged.
"""

from __future__ import annotations

from typing import Any

from benchflow.usage_tracking import usage_unavailable


def _provider_auth_status_from_runtime(runtime: Any) -> int | None:
    """Return a provider 401/403 status from a usage runtime's trajectory.

    Scans the captured provider HTTP exchanges for an auth-failure status.
    Only the integer status code is read — never response bodies or headers —
    so no credential material can leak into ``result.error`` (#546/#564).
    Returns ``None`` when there is no runtime, no trajectory, or no 401/403.
    """
    server = getattr(runtime, "server", None)
    trajectory = getattr(server, "trajectory", None)
    exchanges = getattr(trajectory, "exchanges", None) or []
    for exchange in reversed(exchanges):
        status = getattr(getattr(exchange, "response", None), "status_code", None)
        if status in (401, 403):
            return status
    return None


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
