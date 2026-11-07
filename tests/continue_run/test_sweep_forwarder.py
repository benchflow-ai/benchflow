"""Guards FF-sw-evaluation continuation fixes for 363d815b."""

from types import SimpleNamespace

import pytest

from benchflow.continue_run.orchestrator import (
    LiteLLMLiveForwarder,
)


@pytest.mark.parametrize("provider", ["xiaomi", "azure", "custom"])
def test_forwarder_retains_complete_resolved_route(monkeypatch, provider):
    params = {
        "model": "openai/test",
        "api_base": "https://provider.invalid/v1",
        "api_key": "os.environ/DUMMY_PROVIDER_KEY",
        "api_version": "version",
        "reasoning_effort": "high",
    }
    monkeypatch.setattr(
        "benchflow.providers.litellm_config.resolve_litellm_route",
        lambda *args: SimpleNamespace(
            upstream_model=params["model"], litellm_params=params
        ),
    )
    forwarder = LiteLLMLiveForwarder(
        provider, env={"DUMMY_PROVIDER_KEY": "dummy-marker"}
    )
    kwargs = forwarder.build_kwargs(
        {
            "messages": [],
            "api_base": "https://wrong.invalid",
            "api_key": "wrong",
            "temperature": 0.2,
        }
    )
    assert kwargs == {
        **params,
        "api_key": "dummy-marker",
        "messages": [],
        "stream": False,
        "temperature": 0.2,
    }


@pytest.mark.parametrize("model", ["mimo-v2.6-flash", "xiaomi/mimo-v2.6-flash"])
def test_xiaomi_route_uses_selected_environment(model):
    forwarder = LiteLLMLiveForwarder(
        model,
        env={
            "XIAOMI_BASE_URL": "https://xiaomi.invalid/v1",
            "XIAOMI_API_KEY": "dummy-marker",
        },
    )
    kwargs = forwarder.build_kwargs({"messages": [], "stream": True})
    assert kwargs["api_base"] == "https://xiaomi.invalid/v1"
    assert kwargs["api_key"] == "dummy-marker"
    assert kwargs["stream"] is False
