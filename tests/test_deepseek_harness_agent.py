"""Regression coverage for commit 6429743f (native DeepSeek Harness ACP support)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchflow.acp.runtime import _model_selection_owned_by_env
from benchflow.agents import deepseek_harness_acp_launcher as launcher
from benchflow.agents.env import resolve_provider_env
from benchflow.agents.registry import AGENT_ALIASES, AGENTS


def test_deepseek_harness_registry_contract_is_pinned_and_native():
    cfg = AGENTS["deepseek-harness"]

    assert AGENT_ALIASES["dsh"] == "deepseek-harness"
    assert cfg.protocol == "acp"
    assert cfg.api_protocol == ""
    assert cfg.install_timeout == 600
    assert cfg.supports_acp_set_model is False
    assert cfg.acp_model_config_id == ""
    assert cfg.acp_effort_config_id == "reasoning_effort"
    assert "@deepseek-ai/dsh@0.1.6-alpha.2" in cfg.install_cmd
    assert "@deepseek-ai/dsh@latest" not in cfg.install_cmd
    assert (
        "sha512-PHR/3ZHpJNWXlDQ3U9weFb7calWbSMJd2GD3z2iPJ8zAKL7ipuzyPy5xGbaXf2OA8hc0SAGJeoUW7nfatCNOYw=="
        in cfg.install_cmd
    )
    assert (
        "env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy"
        in cfg.install_cmd
    )
    assert "/opt/benchflow/bin/dsh --profile acp --help" in cfg.install_cmd
    assert cfg.launch_cmd == "/opt/benchflow/bin/deepseek-harness-acp-launcher"


def test_deepseek_harness_launcher_writes_deterministic_isolated_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "agent-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BENCHFLOW_AGENT_HOME", str(home))
    monkeypatch.setenv("DSH_BENCHFLOW_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://gateway.test/v1/")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "super-secret")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")

    called: tuple[str, list[str]] | None = None

    def fake_execv(binary: str, argv: list[str]) -> None:
        nonlocal called
        called = (binary, argv)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(os, "execv", fake_execv)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        launcher.main()

    patch_path = home / ".dsh-benchflow/benchflow-acp.patch.yml"
    rendered = patch_path.read_text()
    assert 'model: "deepseek-v4-pro"' in rendered
    assert "protocol: chat-completions" in rendered
    assert 'baseURL: "http://gateway.test/v1"' in rendered
    assert "apiKeyEnv: DEEPSEEK_API_KEY" in rendered
    assert "super-secret" not in rendered
    assert "includeDefaultRoots: false" in rendered
    assert f'      - "{home}/.agents/skills"' in rendered
    assert "watch: false" in rendered
    assert "- id: session-telemetry-otel\n  disabled: true" in rendered
    assert "- id: tool-web\n  disabled: true" in rendered
    assert os.environ["DSH_HOME"] == str(home / ".dsh-benchflow")
    assert os.environ["DSH_PERMISSION_MODE"] == "danger-full-access"
    assert os.environ["DSH_TELEMETRY_MODE"] == "DISABLED"
    assert called == (
        "/opt/benchflow/bin/dsh",
        ["dsh", "--profile", "acp", "--patch", str(patch_path)],
    )


def test_deepseek_harness_launcher_selects_anthropic_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "anthropic-messages")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.test/anthropic/")
    monkeypatch.setattr(
        os,
        "execv",
        lambda _binary, _argv: (_ for _ in ()).throw(RuntimeError("exec intercepted")),
    )

    with pytest.raises(RuntimeError, match="exec intercepted"):
        launcher.main()

    rendered = (tmp_path / ".dsh-benchflow/benchflow-acp.patch.yml").read_text()
    assert "protocol: messages" in rendered
    assert 'baseURL: "https://api.deepseek.test/anthropic"' in rendered


def test_deepseek_harness_launcher_rejects_openai_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-responses")

    with pytest.raises(SystemExit, match=r"unsupported.*openai-responses"):
        launcher.main()


def test_deepseek_harness_model_is_launch_owned_for_direct_and_proxy_routes():
    direct = {
        "DSH_BENCHFLOW_MODEL": "deepseek-v4-pro",
    }
    proxy = {
        "DSH_BENCHFLOW_MODEL": "benchflow-deepseek-v4-pro",
        "BENCHFLOW_LITELLM_MODEL_ALIAS": "benchflow-deepseek-v4-pro",
    }

    assert _model_selection_owned_by_env(
        "deepseek-harness", "deepseek/deepseek-v4-pro", direct
    )
    assert _model_selection_owned_by_env(
        "deepseek-harness", "deepseek/deepseek-v4-pro", proxy
    )


def test_deepseek_harness_proxy_alias_overrides_direct_model_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DSH_BENCHFLOW_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("BENCHFLOW_LITELLM_MODEL_ALIAS", "benchflow-deepseek-v4-pro")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    called = False

    def fake_execv(_binary: str, _argv: list[str]) -> None:
        nonlocal called
        called = True
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(os, "execv", fake_execv)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        launcher.main()

    assert called
    rendered = (tmp_path / ".dsh-benchflow/benchflow-acp.patch.yml").read_text()
    assert 'model: "benchflow-deepseek-v4-pro"' in rendered
    assert 'model: "deepseek-v4-pro"' not in rendered


def test_launch_owned_model_rule_does_not_bypass_advertised_model_config():
    assert not _model_selection_owned_by_env(
        "claude-agent-acp",
        "anthropic/claude-sonnet-4-6",
        {"ANTHROPIC_MODEL": "claude-sonnet-4-6"},
    )


def test_deepseek_harness_provider_env_maps_direct_route():
    env = {
        "DEEPSEEK_API_KEY": "test-only",
        "DEEPSEEK_BASE_URL": "https://deepseek.test/v1",
    }

    resolve_provider_env(env, "deepseek/deepseek-v4-pro", "deepseek-harness")

    assert env["DEEPSEEK_API_KEY"] == "test-only"
    assert env["DEEPSEEK_BASE_URL"] == "https://deepseek.test/v1"
    assert env["DSH_BENCHFLOW_MODEL"] == "deepseek-v4-pro"


def test_deepseek_harness_provider_env_selects_official_anthropic_endpoint():
    env = {
        "DEEPSEEK_API_KEY": "test-only",
        "BENCHFLOW_PROVIDER_PROTOCOL": "anthropic-messages",
    }

    resolve_provider_env(env, "deepseek/deepseek-v4-pro", "deepseek-harness")

    assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"
    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert env["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com/anthropic"
