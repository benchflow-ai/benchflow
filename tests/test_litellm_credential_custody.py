from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.runtime import ensure_litellm_runtime


@pytest.mark.asyncio
async def test_vertex_adc_provider_capture_remains_audit_only_on_host(monkeypatch):
    """Guards PR #1057 against trusting agent-accessible Vertex credentials."""

    async def fake_start(**_kwargs):
        return SimpleNamespace(base_url="http://127.0.0.1:4000")

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)

    updated, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"type":"authorized_user"}',
            "GOOGLE_APPLICATION_CREDENTIALS": (
                "/home/agent/.config/gcloud/application_default_credentials.json"
            ),
            "GOOGLE_CLOUD_PROJECT": "project",
            "GOOGLE_CLOUD_LOCATION": "global",
        },
        model="google-vertex/gemini-2.5-flash",
        runtime=None,
        environment="docker",
        session_id="run-vertex-adc",
    )

    assert provider_runtime is not None
    assert provider_runtime.capture_trusted is False
    assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in updated
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in updated


@pytest.mark.asyncio
async def test_vertex_adc_stays_audit_only_with_verified_sandbox_artifacts(monkeypatch):
    """Guards PR #1057 against equating file custody with ADC isolation."""

    async def fake_start(**_kwargs):
        return SimpleNamespace(
            base_url="http://127.0.0.1:4000",
            runtime_dir="/tmp/benchflow-litellm/test-runtime",
        )

    async def fake_artifact_custody(**_kwargs):
        return True

    monkeypatch.setattr(runtime_mod, "_start_sandbox_litellm", fake_start)
    monkeypatch.setattr(
        runtime_mod,
        "_provider_capture_has_verified_custody",
        fake_artifact_custody,
    )

    _, provider_runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"type":"authorized_user"}',
            "GOOGLE_CLOUD_PROJECT": "project",
            "GOOGLE_CLOUD_LOCATION": "global",
        },
        model="google-vertex/gemini-2.5-flash",
        runtime=None,
        environment="daytona",
        session_id="run-vertex-adc-sandbox",
        sandbox=SimpleNamespace(),
        sandbox_user="agent",
    )

    assert provider_runtime is not None
    assert provider_runtime.capture_trusted is False
