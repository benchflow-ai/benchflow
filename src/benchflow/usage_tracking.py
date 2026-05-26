"""Configuration contract for provider token-usage telemetry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit

UsageTrackingMode = Literal["auto", "required", "off"]

USAGE_TRACKING_ENV = "BENCHFLOW_USAGE_TRACKING"
USAGE_PROXY_ADVERTISED_BASE_URL_ENV = "BENCHFLOW_USAGE_PROXY_ADVERTISED_BASE_URL"
USAGE_PROXY_BIND_HOST_ENV = "BENCHFLOW_USAGE_PROXY_BIND_HOST"
USAGE_PROXY_PORT_ENV = "BENCHFLOW_USAGE_PROXY_PORT"

DEFAULT_USAGE_PROXY_BIND_HOST = "0.0.0.0"

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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_port(value: Any) -> int | None:
    if value is None or value == "":
        return None
    port = int(value)
    if port < 0 or port > 65535:
        raise ValueError("usage proxy port must be between 0 and 65535")
    return port


def normalize_advertised_base_url(value: str | None) -> str | None:
    url = _optional_str(value)
    if url is None:
        return None
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "usage_proxy.advertised_base_url must be an absolute http(s) URL"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            "usage_proxy.advertised_base_url must not include query or fragment"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError("usage_proxy.advertised_base_url must not include a path")
    return url.rstrip("/")


@dataclass(frozen=True, init=False)
class UsageTrackingConfig:
    """User-facing token/cost telemetry policy.

    ``mode`` is the operator contract:
    - ``auto`` records usage when the sandbox can reach a proxy.
    - ``required`` fails before the agent runs when telemetry cannot be wired.
    - ``off`` leaves provider traffic untouched.
    """

    _mode: UsageTrackingMode | None
    advertised_base_url: str | None = None
    bind_host: str | None = None
    port: int | None = None

    def __init__(
        self,
        mode: str | None = None,
        advertised_base_url: str | None = None,
        bind_host: str | None = None,
        port: int | str | None = None,
    ) -> None:
        object.__setattr__(self, "_mode", _optional_mode(mode))
        object.__setattr__(
            self,
            "advertised_base_url",
            normalize_advertised_base_url(advertised_base_url),
        )
        object.__setattr__(self, "bind_host", _optional_str(bind_host))
        object.__setattr__(self, "port", _optional_port(port))

    @property
    def mode(self) -> UsageTrackingMode:
        return self._mode or "auto"

    @property
    def mode_is_explicit(self) -> bool:
        return self._mode is not None

    @property
    def uses_external_proxy(self) -> bool:
        return self.advertised_base_url is not None

    @property
    def has_fixed_proxy_port(self) -> bool:
        return self.port is not None and self.port > 0

    @classmethod
    def from_values(
        cls,
        *,
        mode: str | None = None,
        advertised_base_url: str | None = None,
        bind_host: str | None = None,
        port: int | str | None = None,
    ) -> UsageTrackingConfig:
        return cls(
            mode=mode,
            advertised_base_url=advertised_base_url,
            bind_host=bind_host,
            port=port,
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> UsageTrackingConfig:
        proxy = raw.get("usage_proxy")
        if proxy is None:
            proxy = {}
        elif not isinstance(proxy, dict):
            raise ValueError("usage_proxy must be a mapping")
        return cls.from_values(
            mode=raw.get("usage_tracking"),
            advertised_base_url=(
                _first_present(
                    proxy.get("advertised_base_url"),
                    raw.get("usage_proxy_advertised_base_url"),
                    raw.get("usage_proxy_url"),
                )
            ),
            bind_host=_first_present(
                proxy.get("bind_host"),
                raw.get("usage_proxy_bind_host"),
            ),
            port=_first_present(proxy.get("port"), raw.get("usage_proxy_port")),
        )

    @classmethod
    def coerce(
        cls, value: UsageTrackingConfig | dict[str, Any] | str | None
    ) -> UsageTrackingConfig:
        if value is None:
            return cls()
        if isinstance(value, UsageTrackingConfig):
            return value
        if isinstance(value, str):
            return cls.from_values(mode=value)
        if isinstance(value, dict):
            return cls.from_mapping(value)
        raise TypeError(f"invalid usage_tracking config: {type(value).__name__}")

    def with_env_defaults(self) -> UsageTrackingConfig:
        env_mode = os.environ.get(USAGE_TRACKING_ENV)
        mode = self.mode
        if env_mode and not self.mode_is_explicit:
            mode = normalize_usage_tracking_mode(env_mode)

        return UsageTrackingConfig.from_values(
            mode=mode,
            advertised_base_url=(
                self.advertised_base_url
                or os.environ.get(USAGE_PROXY_ADVERTISED_BASE_URL_ENV)
            ),
            bind_host=self.bind_host or os.environ.get(USAGE_PROXY_BIND_HOST_ENV),
            port=(
                self.port
                if self.port is not None
                else os.environ.get(USAGE_PROXY_PORT_ENV)
            ),
        )

    def to_config_artifact(self) -> dict[str, Any]:
        return {
            "requested": self.mode,
            "advertised_base_url_configured": self.uses_external_proxy,
            "bind_host": self.bind_host,
            "port": self.port,
        }

    def to_result_metadata(
        self,
        *,
        environment: str,
        status: str,
        usage_source: str,
    ) -> dict[str, Any]:
        return {
            "requested": self.mode,
            "status": status,
            "environment": environment,
            "endpoint_kind": "external" if self.uses_external_proxy else "host",
            "usage_source": usage_source,
            "advertised_base_url_configured": self.uses_external_proxy,
        }
