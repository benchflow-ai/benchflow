from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchflow.agents import openscience_acp_launcher as launcher
from benchflow.agents.env import resolve_provider_env
from benchflow.agents.registry import AGENTS


def test_openscience_registry_contract_is_pinned_and_native():
    cfg = AGENTS["openscience"]

    assert cfg.protocol == "acp"
    assert cfg.api_protocol == ""
    assert cfg.install_timeout == 600
    assert cfg.acp_model_format == "provider/model"
    assert cfg.acp_model_config_id == "model"
    assert cfg.supports_acp_set_model is True
    assert "OpenScience/contents/install?ref=v2.0.127" in cfg.install_cmd
    assert "Accept: application/vnd.github.raw+json" in cfg.install_cmd
    assert "OPENSCIENCE_SKIP_CHECKSUM=0" in cfg.install_cmd
    assert "--version 2.0.127 --no-modify-path" in cfg.install_cmd
    assert "curl -fsSL --connect-timeout 30 --max-time 60" in cfg.install_cmd
    assert (
        "env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy curl"
        in cfg.install_cmd
    )
    assert "--retry 2 --retry-delay 2 --retry-all-errors" in cfg.install_cmd
    assert "real_curl=$(command -v curl)" in cfg.install_cmd
    assert 'PATH="$tmp:$PATH"' in cfg.install_cmd
    assert (
        "XDG_CONFIG_HOME=/opt/benchflow/openscience-installer-home/.config"
        in cfg.install_cmd
    )
    assert (
        "OPENSCIENCE_DATA_DIR=/opt/benchflow/openscience-installer-home/.openscience"
        in cfg.install_cmd
    )
    assert (
        "HOME=/opt/benchflow/openscience-installer-home "
        "OPENSCIENCE_SKIP_CHECKSUM=0 bash" in cfg.install_cmd
    )
    assert "releases/latest" not in cfg.install_cmd
    assert cfg.launch_cmd == "/opt/benchflow/bin/openscience-acp-launcher"
    assert cfg.task_mcp_transport == "native-config"
    assert cfg.task_mcp_config_format == "openscience"


def test_openscience_launcher_writes_isolated_redacted_proxy_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "agent-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BENCHFLOW_AGENT_HOME", str(home))
    monkeypatch.setenv("BENCHFLOW_WORKSPACE", str(workspace))
    monkeypatch.setenv("BENCHFLOW_LITELLM_MODEL_ALIAS", "benchflow-test-model")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://gateway.test/v1/")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")
    monkeypatch.setenv("OPENSCIENCE_BENCHFLOW_API_KEY", "super-secret")

    called: tuple[str, list[str]] | None = None

    def fake_execv(binary: str, argv: list[str]) -> None:
        nonlocal called
        called = (binary, argv)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(os, "execv", fake_execv)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        launcher.main()

    config_path = home / ".openscience-benchflow/config/openscience.json"
    config = json.loads(config_path.read_text())
    rendered = config_path.read_text()
    assert config["model"] == "benchflow/benchflow-test-model"
    assert config["instructions"] == [
        str(home / ".openscience-benchflow/config/benchflow-workspace.md")
    ]
    provider = config["provider"]["benchflow"]
    assert provider["options"] == {
        "apiKey": "{env:OPENSCIENCE_BENCHFLOW_API_KEY}",
        "baseURL": "http://gateway.test/v1",
    }
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert "super-secret" not in rendered
    assert config["sandbox"] == {"enabled": False}
    assert config["agent"]["title"]["disable"] is True
    assert config["permission"]["*"] == "deny"
    assert config["permission"]["bash"] == "allow"
    assert config["permission"]["skill"] == "allow"
    assert config["permission"]["mcp"] == "allow"
    assert config["experimental"]["continue_loop_on_deny"] is True
    assert os.environ["OPENSCIENCE_DISABLE_BUNDLED_SKILLS"] == "1"
    assert os.environ["OPENSCIENCE_DISABLE_PROJECT_CONFIG"] == "1"
    instruction = (
        home / ".openscience-benchflow/config/benchflow-workspace.md"
    ).read_text()
    assert f"The benchmark task workspace is `{workspace}`." in instruction
    assert "session scratch is temporary internal state" in instruction
    assert called == (
        "/opt/benchflow/bin/openscience",
        ["openscience", "acp", "--cwd", str(workspace)],
    )


def test_openscience_launcher_uses_direct_deepseek_route_without_literal_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BENCHFLOW_LITELLM_MODEL_ALIAS", raising=False)
    monkeypatch.setenv("OPENSCIENCE_BENCHFLOW_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://api.deepseek.test")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_NAME", "deepseek")

    config = launcher._configuration(tmp_path)

    assert config["model"] == "deepseek/deepseek-v4-pro"
    assert set(config["provider"]) == {"deepseek"}
    assert config["provider"]["deepseek"]["npm"] == "@ai-sdk/openai-compatible"
    assert config["provider"]["deepseek"]["models"] == {
        "deepseek-v4-pro": {
            "name": "deepseek-v4-pro",
            "tool_call": True,
            "reasoning": True,
            "limit": {"context": 131072, "output": 8192},
        }
    }


def test_openscience_deepseek_provider_env_keeps_gateway_facts_and_maps_model_key():
    env = {
        "DEEPSEEK_API_KEY": "test-only",
        "DEEPSEEK_BASE_URL": "https://deepseek.test/v1",
    }

    resolve_provider_env(env, "deepseek/deepseek-v4-flash", "openscience")

    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://deepseek.test/v1"
    assert env["BENCHFLOW_PROVIDER_API_KEY"] == "test-only"
    assert env["OPENSCIENCE_BENCHFLOW_API_KEY"] == "test-only"
    assert env["OPENSCIENCE_BENCHFLOW_MODEL"] == "deepseek-v4-flash"


def test_openscience_launcher_selects_anthropic_messages_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENSCIENCE_BENCHFLOW_MODEL", "claude-test-model")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_NAME", "anthropic")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "anthropic-messages")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://anthropic.test/v1/")

    config = launcher._configuration(tmp_path)

    assert config["model"] == "anthropic/claude-test-model"
    provider = config["provider"]["anthropic"]
    assert provider["npm"] == "@ai-sdk/anthropic"
    assert provider["options"] == {
        "apiKey": "{env:OPENSCIENCE_BENCHFLOW_API_KEY}",
        "baseURL": "https://anthropic.test/v1",
    }


def test_openscience_launcher_merges_isolated_task_mcp_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / ".openscience-benchflow/config/task-mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "mcp": {
                    "task-server": {
                        "type": "remote",
                        "url": "http://task-mcp.test/mcp",
                        "headers": {},
                        "oauth": False,
                    }
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    config = launcher._configuration(tmp_path)

    assert config["mcp"] == {
        "task-server": {
            "type": "remote",
            "url": "http://task-mcp.test/mcp",
            "headers": {},
            "oauth": False,
        }
    }
    assert config["permission"]["task-server_*"] == "allow"


def test_openscience_launcher_rejects_openai_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-responses")

    with pytest.raises(SystemExit, match=r"unsupported.*openai-responses"):
        launcher._configuration(tmp_path)


def test_openscience_provider_env_selects_deepseek_anthropic_endpoint():
    env = {
        "DEEPSEEK_API_KEY": "test-only",
        "BENCHFLOW_PROVIDER_PROTOCOL": "anthropic-messages",
    }

    resolve_provider_env(env, "deepseek/deepseek-v4-flash", "openscience")

    assert env["BENCHFLOW_PROVIDER_NAME"] == "deepseek"
    assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"
    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert env["BENCHFLOW_PROVIDER_API_KEY"] == "test-only"
    assert env["OPENSCIENCE_BENCHFLOW_API_KEY"] == "test-only"
    assert env["OPENSCIENCE_BENCHFLOW_MODEL"] == "deepseek-v4-flash"
