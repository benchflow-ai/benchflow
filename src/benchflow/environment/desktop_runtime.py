"""Shared desktop-environment runtime artifact helpers.

Desktop/computer-use agent loops should not each invent their own artifact
envelope. This module owns the scrubbed trace shape for BenchFlow's 0.7
desktop adapter slices while leaving sandbox lifecycle to the sandbox provider
and decisions to the agent adapter.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DESKTOP_RUNTIME_TRACE_SCHEMA = "benchflow.desktop-runtime-trace.v1"


@dataclass(frozen=True)
class DesktopEnvironmentSnapshot:
    """Safe desktop environment metadata for trace artifacts."""

    sandbox_provider: str | None = None
    sandbox_provider_mode: str | None = None
    display: str | None = None
    dimensions: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "adapter": "desktop",
            "sandbox_provider": self.sandbox_provider,
            "sandbox_provider_mode": self.sandbox_provider_mode,
            "display": self.display,
        }
        if self.dimensions is not None:
            payload["dimensions"] = dict(self.dimensions)
        return payload


@dataclass(frozen=True)
class DesktopRuntimeSession:
    """First-class runtime evidence writer for a desktop environment slice."""

    snapshot: DesktopEnvironmentSnapshot

    @property
    def environment(self) -> dict[str, object]:
        return self.snapshot.to_dict()

    def build_trace_artifact(
        self,
        *,
        framework: str,
        steps: Sequence[object],
        final_result: str,
        screenshots_b64: list[str] | None = None,
        duration_sec: float | None = None,
        screenshot_method: str | None = None,
        screenshot_error: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        environment = self.environment
        if screenshot_method is not None:
            environment["screenshot_method"] = screenshot_method

        artifact: dict[str, object] = {
            "schema": DESKTOP_RUNTIME_TRACE_SCHEMA,
            "framework": framework,
            "steps": list(steps),
            "environment": environment,
            "screenshots_b64": [
                screenshot for screenshot in (screenshots_b64 or []) if screenshot
            ],
            "final_result": final_result,
        }
        if duration_sec is not None:
            artifact["duration_sec"] = duration_sec
        if screenshot_method is not None:
            artifact["screenshot_method"] = screenshot_method
        if screenshot_error is not None:
            artifact["screenshot_error"] = screenshot_error
        if extra:
            artifact.update(extra)
        return artifact

    def write_trace_artifact(
        self,
        path: Path,
        *,
        framework: str,
        steps: Sequence[object],
        final_result: str,
        screenshots_b64: list[str] | None = None,
        duration_sec: float | None = None,
        screenshot_method: str | None = None,
        screenshot_error: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        artifact = self.build_trace_artifact(
            framework=framework,
            steps=steps,
            final_result=final_result,
            screenshots_b64=screenshots_b64,
            duration_sec=duration_sec,
            screenshot_method=screenshot_method,
            screenshot_error=screenshot_error,
            extra=extra,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        return artifact


def desktop_runtime_session(
    *,
    sandbox_provider: str | None = None,
    sandbox_provider_mode: str | None = None,
    display: str | None = None,
    dimensions: Mapping[str, object] | None = None,
) -> DesktopRuntimeSession:
    """Create a desktop runtime artifact session.

    ``sandbox_provider_mode`` is intentionally optional: release evidence owns
    local/cloud provenance, while the trace records only runtime-local metadata
    that the agent can truthfully observe.
    """

    return DesktopRuntimeSession(
        snapshot=DesktopEnvironmentSnapshot(
            sandbox_provider=sandbox_provider,
            sandbox_provider_mode=sandbox_provider_mode,
            display=display if display is not None else os.environ.get("DISPLAY"),
            dimensions=dimensions,
        )
    )
