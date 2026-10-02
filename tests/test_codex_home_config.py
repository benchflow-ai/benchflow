"""Every Codex thread uses the run's provider, the title thread included.

codex-acp 1.13.1 through 2.0.1 names a session after its first turn with an
ephemeral thread it starts without the session's CODEX_CONFIG, on the
hard-wired ``gpt-5.6-luna``, with the task prompt as input
(TitleGenerator.ts). Codex 0.156.1 picks such a thread's provider as
``requirements > explicit override > config.toml model_provider > "openai"``
(core/src/config/mod.rs) and builds the built-in ``openai`` provider from
``openai_base_url``, else api.openai.com (model-provider-info/src/lib.rs). So
the title thread went to api.openai.com with the task prompt on every
rollout. BenchFlow's launcher now writes the session's provider into Codex's
user config.toml. These tests run the real launcher shell against a fake
codex-acp; nothing is installed and no network is used.
"""

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from benchflow.agents.codex_config import (
    CODEX_CONFIG_ENV,
    CODEX_HOME_CONFIG_ENV,
    CODEX_HOME_CONFIG_MARKER,
    apply_codex_launch_config,
    apply_codex_provider_config,
    codex_home_config,
)
from benchflow.agents.registry import CODEX_ACP_BUILTIN_LAUNCH

PROXY = "http://127.0.0.1:4000/v1"
OPENAI = "https://api.openai.com/v1"


def _litellm_route_env() -> dict[str, str]:
    """The Codex env the LiteLLM route builds (litellm_runtime, codex-acp branch)."""
    env = {"OPENAI_BASE_URL": PROXY, "OPENAI_API_KEY": "sk-master"}
    apply_codex_provider_config(
        env, base_url=PROXY, model="gpt-6-astra", provider_name="litellm", strict=True
    )
    return env


def _thread_endpoint(
    home_config: str | None, *, explicit_provider: str | None = None
) -> str:
    """Where a Codex 0.156.1 thread sends its model requests.

    Mirrors ``Config`` loading for a thread started with no ``config`` (the
    title thread): ``required_model_provider().or(explicit).or(cfg.model_provider)
    .unwrap_or("openai")``, the configured providers merged into the built-in
    ones, and the built-in ``openai`` provider at ``openai_base_url`` or
    api.openai.com.
    """
    cfg = tomllib.loads(home_config) if home_config else {}
    providers = {"openai": {"base_url": cfg.get("openai_base_url") or OPENAI}}
    for key, provider in (cfg.get("model_providers") or {}).items():
        providers.setdefault(key, provider)
    provider_id = explicit_provider or cfg.get("model_provider") or "openai"
    return providers[provider_id]["base_url"]


def test_litellm_route_home_config_names_the_proxy():
    env, _ = apply_codex_launch_config(
        "codex-acp", _litellm_route_env(), model="gpt-6-astra", reasoning_effort=None
    )
    home = env[CODEX_HOME_CONFIG_ENV]
    assert home == codex_home_config(json.loads(env[CODEX_CONFIG_ENV]))
    assert home.splitlines()[0] == CODEX_HOME_CONFIG_MARKER
    parsed = tomllib.loads(home)
    assert parsed["model_provider"] == "benchflow-litellm"
    assert parsed["model"] == "gpt-6-astra"
    assert parsed["openai_base_url"] == PROXY
    assert parsed["model_providers"]["benchflow-litellm"] == {
        "name": "litellm",
        "base_url": PROXY,
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
        "supports_websockets": False,
    }


def test_title_thread_and_named_openai_thread_reach_only_the_proxy():
    """Before: api.openai.com for both. After: the run's proxy for both."""
    env, _ = apply_codex_launch_config(
        "codex-acp", _litellm_route_env(), model="gpt-6-astra", reasoning_effort=None
    )
    home = env[CODEX_HOME_CONFIG_ENV]
    assert _thread_endpoint(None) == OPENAI
    assert _thread_endpoint(None, explicit_provider="openai") == OPENAI
    assert _thread_endpoint(home) == PROXY
    assert _thread_endpoint(home, explicit_provider="openai") == PROXY


def test_no_provider_means_no_home_config_and_a_stale_one_is_dropped():
    """Subscription or plain OpenAI-key runs keep Codex's own provider."""
    stale = {CODEX_HOME_CONFIG_ENV: "left by another role", CODEX_CONFIG_ENV: "{}"}
    env, _ = apply_codex_launch_config(
        "codex-acp", stale, model="gpt-6-astra", reasoning_effort=None
    )
    assert CODEX_HOME_CONFIG_ENV not in env
    assert codex_home_config({"web_search": "disabled"}) is None
    assert codex_home_config({"model_provider": "x", "model_providers": {}}) is None
    assert codex_home_config(None) is None


def test_other_agents_are_untouched():
    env = {CODEX_HOME_CONFIG_ENV: "unrelated"}
    assert apply_codex_launch_config(
        "claude-agent-acp", env, model="m", reasoning_effort=None
    ) == (env, False)


def test_provider_values_survive_toml_quoting():
    tricky = 'quote " backslash \\ newline \n tab \t emoji \U0001f600 del \x7f'
    config = {
        "model_provider": "benchflow-azure foundry",
        "model": tricky,
        "model_providers": {
            "benchflow-azure foundry": {
                "name": tricky,
                "base_url": "https://example.test/openai/v1",
                "query_params": {"api-version": "2026-01-01-preview"},
                "http_headers": {"X-Title": tricky},
                "request_max_retries": 4,
                "stream_idle_timeout_ms": 30000.5,
                "supports_websockets": False,
                "unrenderable": [{"nested": True}],
            }
        },
    }
    rendered = codex_home_config(config)
    assert rendered is not None
    parsed = tomllib.loads(rendered)
    provider = parsed["model_providers"]["benchflow-azure foundry"]
    assert parsed["model"] == tricky
    assert provider["name"] == tricky
    assert provider["http_headers"] == {"X-Title": tricky}
    assert provider["query_params"] == {"api-version": "2026-01-01-preview"}
    assert provider["request_max_retries"] == 4
    assert provider["stream_idle_timeout_ms"] == 30000.5
    assert "unrenderable" not in provider


def _launcher(tmp_path: Path) -> tuple[str, Path]:
    """The built-in launcher with a fake codex-acp that records it ran."""
    ran = tmp_path / "codex-acp-ran"
    fake = tmp_path / "codex-acp"
    fake.write_text(f'#!/bin/sh\ntouch "{ran}"\n')
    fake.chmod(0o755)
    assert "/opt/benchflow/bin/codex-acp" in CODEX_ACP_BUILTIN_LAUNCH
    return CODEX_ACP_BUILTIN_LAUNCH.replace(
        "/opt/benchflow/bin/codex-acp", str(fake)
    ), ran


def _launch(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    command, _ = _launcher(tmp_path)
    return subprocess.run(
        ["sh", "-c", command],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
    )


def test_launcher_writes_the_home_config_where_codex_reads_it(tmp_path):
    home = tmp_path / "home"
    content = codex_home_config(json.loads(_litellm_route_env()[CODEX_CONFIG_ENV]))
    assert content is not None
    env = {"BENCHFLOW_AGENT_HOME": str(home), CODEX_HOME_CONFIG_ENV: content}
    assert _launch(tmp_path, env).returncode == 0
    assert (home / ".codex" / "config.toml").read_text() == content
    assert (tmp_path / "codex-acp-ran").exists()

    codex_home = tmp_path / "codex-home"
    env["CODEX_HOME"] = str(codex_home)
    assert _launch(tmp_path, env).returncode == 0
    assert (codex_home / "config.toml").read_text() == content


def test_launch_without_a_provider_removes_only_benchflows_file(tmp_path):
    home = tmp_path / "home"
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(f'{CODEX_HOME_CONFIG_MARKER}\nmodel_provider = "x"\n')
    assert _launch(tmp_path, {"BENCHFLOW_AGENT_HOME": str(home)}).returncode == 0
    assert not config.exists()

    config.write_text('model = "the image\'s own"\n')
    assert _launch(tmp_path, {"BENCHFLOW_AGENT_HOME": str(home)}).returncode == 0
    assert config.read_text() == 'model = "the image\'s own"\n'


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_launch_fails_closed_when_the_home_config_cannot_be_written(tmp_path):
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex").chmod(0o500)
    try:
        result = _launch(
            tmp_path,
            {"BENCHFLOW_AGENT_HOME": str(home), CODEX_HOME_CONFIG_ENV: "x = 1\n"},
        )
    finally:
        (home / ".codex").chmod(0o700)
    assert result.returncode == 1
    assert "cannot write" in result.stderr
    assert not (tmp_path / "codex-acp-ran").exists()
