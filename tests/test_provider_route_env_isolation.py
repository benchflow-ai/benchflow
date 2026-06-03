"""Regression tests for provider-route isolation after PR #598 review."""

from __future__ import annotations

import pytest

from benchflow.agents.env import resolve_agent_env
from benchflow.agents.provider_route import EXPLICIT_PROVIDER_BASE_URL_ENVS
from benchflow.providers.usage_proxy_runtime import _resolve_usage_proxy_target


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch, tmp_path):
    """Keep tests independent from the developer machine's real provider env."""
    for key in (
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_EXPLICIT_PROVIDER_BASE_URL_ENVS",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "GEMINI_API_BASE_URL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LITELLM_API_KEY",
        "LITELLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    empty_dotenv = tmp_path / "empty.env"
    empty_dotenv.write_text("")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(empty_dotenv))


def test_openhands_gemini_drops_inherited_cross_provider_env(monkeypatch):
    """Guards the PR #598 follow-up fix against shared .env route pollution."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://llm-proxy.example")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-proxy")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm-proxy.example")
    monkeypatch.setenv("LLM_API_KEY", "sk-proxy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy.example/v1")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm-proxy.example")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm")

    env = resolve_agent_env("openhands", "gemini-3.5-flash", {})

    assert env["LLM_MODEL"] == "gemini/gemini-3.5-flash"
    assert env["BENCHFLOW_PROVIDER_API_KEY"] == "gemini-key"
    assert env["LLM_API_KEY"] == "gemini-key"
    for key in (
        "BENCHFLOW_PROVIDER_BASE_URL",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "LITELLM_BASE_URL",
        "LITELLM_API_KEY",
    ):
        assert key not in env


def test_gemini_usage_proxy_ignores_inherited_generic_base_urls():
    """Guards the PR #598 follow-up fix for raw usage-proxy routing calls."""
    target = _resolve_usage_proxy_target(
        "openhands",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://llm-proxy.example",
            "LLM_BASE_URL": "https://llm-proxy.example",
            "OPENAI_BASE_URL": "https://openai-proxy.example/v1",
        },
        "gemini-3.5-flash",
    )

    assert target == "https://generativelanguage.googleapis.com"


def test_stale_explicit_marker_does_not_promote_inherited_base_url(monkeypatch):
    """Guards the PR #598 follow-up fix against forged explicit-route metadata."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy.example/v1")

    env = resolve_agent_env(
        "openhands",
        "gemini-3.5-flash",
        {EXPLICIT_PROVIDER_BASE_URL_ENVS: "OPENAI_BASE_URL"},
    )

    assert EXPLICIT_PROVIDER_BASE_URL_ENVS not in env
    assert "OPENAI_BASE_URL" not in env
    assert (
        _resolve_usage_proxy_target("openhands", env, "gemini-3.5-flash")
        == "https://generativelanguage.googleapis.com"
    )


def test_gemini_usage_proxy_prefers_gemini_specific_base_url():
    """Guards the PR #598 follow-up fix for Gemini-specific endpoint overrides."""
    target = _resolve_usage_proxy_target(
        "openhands",
        {
            "OPENAI_BASE_URL": "https://openai-proxy.example/v1",
            "GOOGLE_GEMINI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta",
        },
        "gemini-3.5-flash",
    )

    assert target == "https://generativelanguage.googleapis.com/v1beta"


def test_gemini_specific_base_url_survives_env_resolution(monkeypatch):
    """Guards the PR #598 follow-up fix for provider-specific Gemini routing."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv(
        "GOOGLE_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy.example/v1")

    env = resolve_agent_env("openhands", "gemini-3.5-flash", {})

    assert "OPENAI_BASE_URL" not in env
    assert (
        _resolve_usage_proxy_target("openhands", env, "gemini-3.5-flash")
        == "https://generativelanguage.googleapis.com/v1beta"
    )


def test_explicit_generic_base_url_still_overrides_gemini_route():
    """Guards the PR #598 follow-up fix without breaking explicit proxy routing."""
    env = resolve_agent_env(
        "openhands",
        "gemini-3.5-flash",
        {
            "GEMINI_API_KEY": "gemini-key",
            "BENCHFLOW_PROVIDER_BASE_URL": "https://explicit-proxy.example",
            "BENCHFLOW_PROVIDER_API_KEY": "sk-explicit",
        },
    )

    assert env[EXPLICIT_PROVIDER_BASE_URL_ENVS] == "BENCHFLOW_PROVIDER_BASE_URL"
    assert env["LLM_BASE_URL"] == "https://explicit-proxy.example"
    assert env["LLM_API_KEY"] == "sk-explicit"
    assert (
        _resolve_usage_proxy_target("openhands", env, "gemini-3.5-flash")
        == "https://explicit-proxy.example"
    )


def test_litellm_provider_prefix_keeps_litellm_route(monkeypatch):
    """Guards the PR #598 follow-up fix from breaking intentional proxy providers."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm-proxy.example")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy.example/v1")

    env = resolve_agent_env("openhands", "litellm/gemini-3.5-flash", {})

    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://litellm-proxy.example"
    assert env["BENCHFLOW_PROVIDER_API_KEY"] == "sk-litellm"
    assert env["LLM_BASE_URL"] == "https://litellm-proxy.example"
    assert env["LLM_API_KEY"] == "sk-litellm"
    assert (
        _resolve_usage_proxy_target("openhands", env, "litellm/gemini-3.5-flash")
        == "https://litellm-proxy.example"
    )
