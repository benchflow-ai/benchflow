"""Per-rollout overrides of BenchFlow's startup timeouts.

Two startup waits are process-wide settings read from the environment: the
ACP handshake (``BENCHFLOW_ACP_HANDSHAKE_TIMEOUT``, default 60 s) and the
model gateway's health wait (``BENCHFLOW_LITELLM_HEALTH_TIMEOUT_SEC``,
default 180 s). A trainer running many rollouts at once needs them per
rollout, next to the sandbox setup timeout that ``RolloutConfig`` already
carries. :func:`startup_timeout_overrides` sets them for the code that runs
inside it (an asyncio task inherits them), and the waits read them first.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_ACP_HANDSHAKE_SEC: ContextVar[float | None] = ContextVar(
    "benchflow_acp_handshake_timeout_sec", default=None
)
_GATEWAY_SEC: ContextVar[float | None] = ContextVar(
    "benchflow_gateway_health_timeout_sec", default=None
)


def acp_handshake_override() -> float | None:
    """The ACP handshake timeout set for this rollout, if any."""
    return _ACP_HANDSHAKE_SEC.get()


def gateway_health_override() -> float | None:
    """The gateway health wait set for this rollout, if any."""
    return _GATEWAY_SEC.get()


def _positive(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not number > 0:
        raise ValueError(f"{name} must be a positive number of seconds, got {value!r}")
    return number


@contextmanager
def startup_timeout_overrides(
    *,
    acp_handshake_sec: float | None = None,
    gateway_sec: float | None = None,
) -> Iterator[None]:
    """Use these startup timeouts for rollouts started inside the block."""
    tokens = [
        _ACP_HANDSHAKE_SEC.set(_positive("acp_handshake_sec", acp_handshake_sec)),
        _GATEWAY_SEC.set(_positive("gateway_sec", gateway_sec)),
    ]
    try:
        yield
    finally:
        _GATEWAY_SEC.reset(tokens[1])
        _ACP_HANDSHAKE_SEC.reset(tokens[0])


__all__ = [
    "acp_handshake_override",
    "gateway_health_override",
    "startup_timeout_overrides",
]
