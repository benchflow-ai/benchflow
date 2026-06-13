from __future__ import annotations

import json
from pathlib import Path

from benchflow.environment.desktop_runtime import (
    DESKTOP_RUNTIME_TRACE_SCHEMA,
    desktop_runtime_session,
)


def test_desktop_runtime_session_writes_trace_artifact(tmp_path: Path) -> None:
    """Guards 0.7 desktop trace envelopes from living inside agent shims."""
    artifact_path = tmp_path / "artifacts" / "computer-use-smoke-trace.json"
    session = desktop_runtime_session(
        sandbox_provider="cua",
        sandbox_provider_mode="local",
        display=":1",
        dimensions={"width": 1024, "height": 768},
    )

    artifact = session.write_trace_artifact(
        artifact_path,
        framework="benchflow-desktop-test",
        steps=["write_file", {"action": "screenshot"}],
        screenshots_b64=["abc", ""],
        screenshot_method="gnome-screenshot",
        final_result="ready",
        duration_sec=0.5,
        extra={"custom": "value"},
    )

    stored = json.loads(artifact_path.read_text())
    assert stored == artifact
    assert artifact["schema"] == DESKTOP_RUNTIME_TRACE_SCHEMA
    assert artifact["framework"] == "benchflow-desktop-test"
    assert artifact["screenshots_b64"] == ["abc"]
    assert artifact["screenshot_method"] == "gnome-screenshot"
    assert artifact["environment"] == {
        "adapter": "desktop",
        "sandbox_provider": "cua",
        "sandbox_provider_mode": "local",
        "display": ":1",
        "dimensions": {"width": 1024, "height": 768},
        "screenshot_method": "gnome-screenshot",
    }
    assert artifact["custom"] == "value"
