"""Configuration contract for provider token-usage telemetry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, cast

UsageTrackingMode = Literal["auto", "required", "off"]

USAGE_TRACKING_ENV = "BENCHFLOW_USAGE_TRACKING"

_MODES: set[str] = {"auto", "required", "off"}


def normalize_usage_tracking_mode(value: str) -> UsageTrackingMode:
    mode = value.strip().lower()
    if mode not in _MODES:
        expected = ", ".join(sorted(_MODES))
        raise ValueError(f"usage_tracking must be one of: {expected}")
    return cast(UsageTrackingMode, mode)


def _optional_mode(value: Any) -> UsageTrackingMode | None:
    if value is None:
        return None
    return normalize_usage_tracking_mode(str(value))


@dataclass(frozen=True, init=False)
class UsageTrackingConfig:
    """User-facing token/cost telemetry policy.

    ``mode`` is the operator contract:
    - ``auto`` records usage when the sandbox can reach a proxy.
    - ``required`` fails before the agent runs when telemetry cannot be wired.
    - ``off`` leaves provider traffic untouched.
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

    def validate_parallelism(self, *, concurrency: int, worker_count: int = 1) -> None:
        return None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> UsageTrackingConfig:
        proxy = raw.get("usage_proxy")
        if proxy is not None and not isinstance(proxy, dict):
            raise ValueError("usage_proxy must be a mapping")
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
    ) -> dict[str, Any]:
        endpoint_kind = "sandbox" if environment == "daytona" else "host"
        if self.mode == "off":
            endpoint_kind = "none"
        return {
            "requested": self.mode,
            "status": status,
            "environment": environment,
            "endpoint_kind": endpoint_kind,
            "usage_source": usage_source,
        }
