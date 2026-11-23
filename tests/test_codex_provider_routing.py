"""Which first-party providers Codex can run on, decided by what each serves.

Codex speaks only the OpenAI Responses API. A provider entry's
``openai-responses`` endpoint is BenchFlow's claim that the provider serves
it, and the claim decides whether ``-a codex-acp`` (ACP or native) is refused
before a job exists. Each claim below was checked with an authenticated call
to the provider's own endpoint (status codes only):

- DeepSeek ``/v1/responses``: 200, and 200 with a ``custom`` tool, the kind
  every catalogued Codex model sends (``apply_patch``).
- Z.ai ``/api/paas/v4/responses`` and ``/api/coding/paas/v4/responses``: 404.
  Codex on Z.ai ran until the first model call failed and native Codex
  retried it.
- MiMo's token plan ``/v1/responses``: 200 for text, but 400
  ``responses_feature_not_supported`` for a ``custom`` tool, so a catalogued
  Codex model cannot run there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.agents.env import (
    resolve_agent_env,
    resolve_provider_env,
    unroutable_agent_model,
)
from benchflow.agents.providers import PROVIDERS, resolve_base_url
from benchflow.cli.main import app


@pytest.mark.parametrize(
    "model", ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"]
)
def test_codex_runs_deepseek_on_its_responses_endpoint(model):
    assert "openai-responses" in PROVIDERS["deepseek"].all_endpoints
    assert unroutable_agent_model("codex-acp", model) is None
    env = {"DEEPSEEK_API_KEY": "ds-test"}
    resolve_provider_env(env, model, "codex-acp")
    assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-responses"
    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.deepseek.com/v1"
    assert env["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"


def test_deepseek_keeps_chat_completions_for_other_agents():
    deepseek = PROVIDERS["deepseek"]
    assert deepseek.api_protocol == "openai-completions"
    env = {"DEEPSEEK_BASE_URL": "https://ds.example.test/v1"}
    assert resolve_base_url(deepseek, env, "openai-completions") == (
        "https://ds.example.test/v1"
    )
    assert resolve_base_url(deepseek, env, "openai-responses") == (
        "https://ds.example.test/v1"
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("zai", "zai/glm-5.1"),
        ("zai-coding", "zai-coding/glm-5.3-flash"),
        ("xiaomi", "xiaomi/mimo-v2.6-flash"),
    ],
)
def test_codex_is_refused_where_the_provider_serves_no_usable_responses(
    provider, model
):
    assert "openai-responses" not in PROVIDERS[provider].all_endpoints
    why = unroutable_agent_model("codex-acp", model)
    assert why is not None
    assert "requires provider protocol 'openai-responses'" in why
    assert f"provider {provider!r}" in why


@pytest.mark.parametrize("model", ["zai/glm-5.1", "zai-coding/glm-5.3-flash"])
def test_claude_keeps_its_zai_anthropic_route(model):
    env = resolve_agent_env("claude-agent-acp", model, {"ZAI_API_KEY": "zk-test"})
    assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"
    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/anthropic"


def _task(root: Path) -> Path:
    task = root / "task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task / "instruction.md").write_text("Do it.\n")
    return task


def _bench_run(tmp_path, monkeypatch, *args: str) -> tuple[object, list[object]]:
    started: list[object] = []
    monkeypatch.setattr(
        "benchflow.cli.main.run_batch_eval", lambda *a, **k: started.append(a)
    )
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(_task(tmp_path)),
            *args,
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    return result, started


@pytest.mark.parametrize("harness", ["acp", "native"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_codex_on_zai_coding_is_refused_before_a_job(
    tmp_path, monkeypatch, harness, dry_run
):
    result, started = _bench_run(
        tmp_path,
        monkeypatch,
        "-a",
        "codex-acp",
        "-m",
        "zai-coding/glm-5.3-flash",
        "--harness",
        harness,
        "--agent-env",
        "ZAI_API_KEY=zk-test",
        *(["--dry-run"] if dry_run else []),
    )
    assert result.exit_code == 1, result.output
    assert started == []
    assert not (tmp_path / "jobs").exists()
    assert "requires provider protocol 'openai-responses'" in result.stderr
    assert "No job was created" in result.stderr


@pytest.mark.parametrize("harness", ["acp", "native"])
def test_codex_on_deepseek_reaches_the_job(tmp_path, monkeypatch, harness):
    result, started = _bench_run(
        tmp_path,
        monkeypatch,
        "-a",
        "codex-acp",
        "-m",
        "deepseek/deepseek-flash",
        "--harness",
        harness,
        "--agent-env",
        "DEEPSEEK_API_KEY=ds-test",
    )
    assert "requires provider protocol" not in result.output
    assert len(started) == 1, result.output
