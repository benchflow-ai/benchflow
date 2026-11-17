"""Configuration contract for provider token-usage telemetry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, cast

UsageTrackingMode = Literal["auto", "required", "off"]
UsageSource = Literal[
    "provider_response",
    "agent_native_acp",
    "unavailable",
    "agent-reported",
    "none",
    "stream_log",
    "session_log",
    "end_of_turn",
    "proxy",
    "provider",
]
# Where a result's tokens came from, as result.json, ATIF and summary.json
# name it. None when no source counted them.
ResultUsageSource = Literal[
    "stream_log", "session_log", "end_of_turn", "proxy", "provider"
]
# Who set a result's cost_usd: the provider's own billed figure, or tokens
# at list prices (BenchFlow's gateway, its price table, an agent's own total).
CostSource = Literal["provider", "list_price_estimate"]

USAGE_TRACKING_ENV = "BENCHFLOW_USAGE_TRACKING"
USAGE_SOURCE_PROVIDER_RESPONSE = "provider_response"
USAGE_SOURCE_AGENT_NATIVE_ACP = "agent_native_acp"
USAGE_SOURCE_UNAVAILABLE = "unavailable"
# An agent that routes its own model calls (model_route = "agent") is
# never metered: its usage is what it reported over ACP, or none. Neither is
# trusted, so its trials never pool with metered ones in totals or summaries.
USAGE_SOURCE_AGENT_REPORTED = "agent-reported"
USAGE_SOURCE_NONE = "none"
# Result vocabulary. ``stream_log``: a native CLI's own output stream, kept on
# the host (``agent/claude-code.jsonl``); ``session_log``: the agent's session
# log, copied from the sandbox (``agent/claude-sessions/``,
# ``agent/codex-sessions/``); ``end_of_turn``: the agent's report at the end of
# each turn (ACP ``PromptResponse.usage``, a CLI's final event); ``proxy``:
# BenchFlow's model proxy, from each provider response; ``provider``: the
# provider's own record of the calls (no route records it yet).
USAGE_STREAM_LOG = "stream_log"
USAGE_SESSION_LOG = "session_log"
USAGE_END_OF_TURN = "end_of_turn"
USAGE_PROXY = "proxy"
USAGE_PROVIDER = "provider"
RESULT_USAGE_SOURCES: tuple[str, ...] = (
    USAGE_STREAM_LOG,
    USAGE_SESSION_LOG,
    USAGE_END_OF_TURN,
    USAGE_PROXY,
    USAGE_PROVIDER,
)
COST_SOURCE_PROVIDER = "provider"
COST_SOURCE_LIST_PRICE = "list_price_estimate"
COST_SOURCES: tuple[str, ...] = (COST_SOURCE_PROVIDER, COST_SOURCE_LIST_PRICE)
# The earlier values, kept in result.json as ``usage_source_legacy`` for the
# readers that tell a metered run from a login's or an agent-routed one.
_LEGACY_TO_RESULT: dict[str, str | None] = {
    USAGE_SOURCE_PROVIDER_RESPONSE: USAGE_PROXY,
    USAGE_SOURCE_AGENT_NATIVE_ACP: USAGE_END_OF_TURN,
    USAGE_SOURCE_AGENT_REPORTED: USAGE_END_OF_TURN,
    USAGE_SOURCE_UNAVAILABLE: None,
    USAGE_SOURCE_NONE: None,
}
_UNTRUSTED_LEGACY: frozenset[str] = frozenset(
    {USAGE_SOURCE_AGENT_REPORTED, USAGE_SOURCE_NONE, USAGE_SOURCE_UNAVAILABLE}
)
TRUSTED_USAGE_SOURCES: frozenset[str] = frozenset(
    {
        USAGE_SOURCE_PROVIDER_RESPONSE,
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        *RESULT_USAGE_SOURCES,
    }
)

_MODES: set[str] = {"auto", "required", "off"}
_USAGE_SOURCES: set[str] = {
    USAGE_SOURCE_PROVIDER_RESPONSE,
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_UNAVAILABLE,
    USAGE_SOURCE_AGENT_REPORTED,
    USAGE_SOURCE_NONE,
    *RESULT_USAGE_SOURCES,
}
_LEGACY_USAGE_PROXY_KEYS: frozenset[str] = frozenset(
    {
        "usage_proxy",
        "usage_proxy_advertised_base_url",
        "usage_proxy_bind_host",
        "usage_proxy_port",
        "usage_proxy_url",
    }
)


def normalize_usage_tracking_mode(value: str) -> UsageTrackingMode:
    mode = value.strip().lower()
    if mode not in _MODES:
        expected = ", ".join(sorted(_MODES))
        raise ValueError(f"usage_tracking must be one of: {expected}")
    return cast(UsageTrackingMode, mode)


def normalize_usage_source(value: str) -> UsageSource:
    if value not in _USAGE_SOURCES:
        expected = ", ".join(sorted(_USAGE_SOURCES))
        raise ValueError(f"usage_source must be one of: {expected}")
    return cast(UsageSource, value)


def _optional_mode(value: Any) -> UsageTrackingMode | None:
    if value is None:
        return None
    return normalize_usage_tracking_mode(str(value))


def is_trusted_usage_source(value: object, legacy: object = None) -> bool:
    """Return True for usage telemetry sources that satisfy required tracking.

    ``legacy`` is a result's ``usage_source_legacy``: an agent-routed trial
    records ``end_of_turn`` like a login's, and only its legacy value says it
    is the agent's own unmetered report.
    """
    if legacy is not None and str(legacy) in _UNTRUSTED_LEGACY:
        return False
    return str(value) in TRUSTED_USAGE_SOURCES


def is_token_usage_available(metrics: dict[str, Any] | None) -> bool:
    """Return True when a usage metrics payload has trusted token telemetry."""
    if not metrics:
        return False
    return is_trusted_usage_source(
        metrics.get("usage_source"), metrics.get("usage_source_legacy")
    )


def result_usage_source(value: object) -> str | None:
    """The result vocabulary's name for a usage source, old or new; None when
    no source counted the tokens."""
    if value is None:
        return None
    text = str(value)
    if text in RESULT_USAGE_SOURCES:
        return text
    return _LEGACY_TO_RESULT.get(text)


def legacy_usage_source(value: object, legacy: object = None) -> str:
    """The earlier name of a usage source (``usage_source_legacy``).

    A count salvaged from the agent's own logs is the agent's count, as
    ``agent_native_acp`` was; a result already carrying ``legacy`` keeps it.
    """
    if isinstance(legacy, str) and legacy in _LEGACY_TO_RESULT:
        return legacy
    text = USAGE_SOURCE_UNAVAILABLE if value is None else str(value)
    if text in _LEGACY_TO_RESULT:
        return text
    if text == USAGE_PROXY:
        return USAGE_SOURCE_PROVIDER_RESPONSE
    if text in RESULT_USAGE_SOURCES:
        return USAGE_SOURCE_AGENT_NATIVE_ACP
    return USAGE_SOURCE_UNAVAILABLE


def usage_unavailable() -> dict[str, Any]:
    """Return the canonical empty token-usage metrics payload.

    Unknown tokens and cost are None, never 0: a trial nothing
    counted is not a trial that used no tokens.
    """
    return {
        "n_input_tokens": None,
        "n_output_tokens": None,
        "n_cache_read_tokens": None,
        "n_cache_creation_tokens": None,
        "n_cache_write_5m_tokens": None,
        "n_cache_write_1h_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "usage_source": USAGE_SOURCE_UNAVAILABLE,
        "price_source": None,
    }


@dataclass(frozen=True, init=False)
class UsageTrackingConfig:
    """User-facing token/cost telemetry policy.

    ``mode`` is the operator *telemetry-enforcement* contract — it no longer
    decides whether the LiteLLM proxy runs. Every routable agent is always
    routed through the proxy (so usage, cost, and the raw LLM trajectory are
    always captured and the provider key never reaches the agent); ``mode`` only
    governs how hard BenchFlow insists on *trusted* telemetry:
    - ``auto`` records usage when LiteLLM or native ACP telemetry can be used.
    - ``required`` fails when no trusted token telemetry can be captured.
    - ``off`` still routes/captures, but does not *require* trusted telemetry
      (it no longer leaves provider traffic untouched).
    """

    _mode: UsageTrackingMode | None

    def __init__(
        self,
        mode: str | None = None,
    ) -> None:
        object.__setattr__(self, "_mode", _optional_mode(mode))

    @property
    def mode(self) -> UsageTrackingMode:
        return self._mode or "auto"

    @property
    def mode_is_explicit(self) -> bool:
        return self._mode is not None

    def overlay(self, override: UsageTrackingConfig) -> UsageTrackingConfig:
        """Return this config with explicitly supplied override fields applied."""
        return UsageTrackingConfig(
            mode=override._mode if override.mode_is_explicit else self._mode,
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> UsageTrackingConfig:
        legacy_keys = sorted(set(raw) & _LEGACY_USAGE_PROXY_KEYS)
        if legacy_keys:
            raise ValueError(
                f"{', '.join(legacy_keys)} is no longer supported; use usage_tracking="
                "auto|required|off instead."
            )
        return cls(mode=raw.get("usage_tracking"))

    @classmethod
    def coerce(
        cls, value: UsageTrackingConfig | dict[str, Any] | str | None
    ) -> UsageTrackingConfig:
        if value is None:
            return cls()
        if isinstance(value, UsageTrackingConfig):
            return value
        if isinstance(value, str):
            return cls(mode=value)
        if isinstance(value, dict):
            return cls.from_mapping(value)
        raise TypeError(f"invalid usage_tracking config: {type(value).__name__}")

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.mode_is_explicit:
            payload["usage_tracking"] = self.mode
        return payload

    def with_env_defaults(self) -> UsageTrackingConfig:
        env_mode = os.environ.get(USAGE_TRACKING_ENV)
        mode = self.mode
        if env_mode and not self.mode_is_explicit:
            mode = normalize_usage_tracking_mode(env_mode)

        return UsageTrackingConfig(mode=mode)

    def to_config_artifact(self) -> dict[str, Any]:
        return {"requested": self.mode}

    def to_result_metadata(
        self,
        *,
        environment: str,
        status: str,
        usage_source: str,
        usage_source_legacy: str | None = None,
    ) -> dict[str, Any]:
        # The proxy now always runs for routable agents, so the endpoint kind
        # follows where telemetry actually came from rather than the mode: a
        # live proxy reports host/sandbox, native-subscription agents report
        # agent_native, and only a genuinely un-routable agent (no telemetry at
        # all, e.g. a native-protocol agent) reports none.
        legacy = legacy_usage_source(usage_source, usage_source_legacy)
        endpoint_kind = "sandbox" if environment == "daytona" else "host"
        if legacy == USAGE_SOURCE_AGENT_NATIVE_ACP:
            endpoint_kind = "agent_native"
        elif legacy == USAGE_SOURCE_AGENT_REPORTED:
            endpoint_kind = "agent"
        elif legacy in (USAGE_SOURCE_UNAVAILABLE, USAGE_SOURCE_NONE):
            endpoint_kind = "none"
        return {
            "requested": self.mode,
            "status": status,
            "environment": environment,
            "endpoint_kind": endpoint_kind,
            "usage_source": result_usage_source(usage_source),
            "usage_source_legacy": legacy,
        }
