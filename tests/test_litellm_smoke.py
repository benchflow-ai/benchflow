"""Optional real-provider smoke test for provider usage telemetry.

Run explicitly with Docker:
    BENCHFLOW_RUN_TELEMETRY_SMOKE=1 uv run pytest tests/test_litellm_smoke.py -q

Run explicitly with Daytona:
    BENCHFLOW_RUN_DAYTONA_TELEMETRY_SMOKE=1 uv run pytest tests/test_litellm_smoke.py -q
"""

from __future__ import annotations

import json
import os

import pytest


def _smoke_dotenv() -> dict[str, str]:
    from benchflow._dotenv import load_dotenv_env

    return load_dotenv_env()


def _smoke_setting(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or _smoke_dotenv().get(key) or default


def _smoke_agent() -> str:
    configured = _smoke_setting("BENCHFLOW_TELEMETRY_SMOKE_AGENT")
    if configured:
        return configured
    env = {**_smoke_dotenv(), **os.environ}
    if env.get("OPENAI_API_KEY"):
        return "codex-acp"
    return "claude-agent-acp"


def _smoke_model(agent: str) -> str:
    configured = _smoke_setting("BENCHFLOW_TELEMETRY_SMOKE_MODEL")
    if configured:
        return configured
    if agent == "codex-acp":
        return "gpt-4.1-mini"
    return "claude-haiku-4-5-20251001"


def _smoke_agent_env() -> dict[str, str]:
    keys = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_PROMPT_CACHE_RETENTION",
    )
    dotenv = _smoke_dotenv()
    return {
        key: os.environ.get(key) or dotenv[key]
        for key in keys
        if os.environ.get(key) or dotenv.get(key)
    }


def _provider_auth_error(result_json):
    llm_traj = result_json.parent / "trajectory" / "llm_trajectory.jsonl"
    if not llm_traj.exists():
        return None
    for line in llm_traj.read_text().splitlines():
        exchange = json.loads(line)
        status = exchange.get("response", {}).get("status_code")
        if status not in {401, 403}:
            continue
        body = exchange.get("response", {}).get("body", {})
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if not message:
            detail = body.get("detail")
            message = detail.get("message") if isinstance(detail, dict) else detail
        if message and "insufficient permissions" in message.lower():
            return message
    return None


@pytest.mark.asyncio
async def test_real_acp_rollout_records_provider_usage(tmp_path):
    if os.environ.get("BENCHFLOW_RUN_TELEMETRY_SMOKE") != "1":
        pytest.skip("set BENCHFLOW_RUN_TELEMETRY_SMOKE=1 to run real telemetry smoke")

    from benchflow.sdk import SDK

    agent = _smoke_agent()
    result = await SDK().run(
        task_path="src/benchflow/demo_task",
        agent=agent,
        model=_smoke_model(agent),
        jobs_dir=tmp_path,
        job_name="telemetry-smoke",
        rollout_name="demo",
        environment=_smoke_setting("BENCHFLOW_TELEMETRY_SMOKE_ENV", "docker"),
        agent_env=_smoke_agent_env(),
        usage_tracking="required",
    )

    _assert_provider_usage_recorded(
        tmp_path / "telemetry-smoke" / "demo" / "result.json", result
    )


@pytest.mark.asyncio
async def test_real_daytona_acp_rollout_records_provider_usage(tmp_path):
    if os.environ.get("BENCHFLOW_RUN_DAYTONA_TELEMETRY_SMOKE") != "1":
        pytest.skip(
            "set BENCHFLOW_RUN_DAYTONA_TELEMETRY_SMOKE=1 to run Daytona telemetry smoke"
        )

    from benchflow.sdk import SDK

    agent = _smoke_agent()
    result = await SDK().run(
        task_path="src/benchflow/demo_task",
        agent=agent,
        model=_smoke_model(agent),
        jobs_dir=tmp_path,
        job_name="daytona-telemetry-smoke",
        rollout_name="demo",
        environment="daytona",
        agent_env=_smoke_agent_env(),
        usage_tracking="required",
    )

    _assert_provider_usage_recorded(
        tmp_path / "daytona-telemetry-smoke" / "demo" / "result.json",
        result,
    )


def _assert_provider_usage_recorded(result_json, result) -> None:
    data = json.loads(result_json.read_text())
    agent_result = data["agent_result"]

    auth_error = _provider_auth_error(result_json)
    if auth_error:
        pytest.skip(f"provider credentials do not have required scopes: {auth_error}")

    assert result.error is None
    assert agent_result["usage_source"] == "provider_response"
    assert agent_result["n_input_tokens"] > 0
    assert agent_result["n_output_tokens"] > 0
    assert agent_result["n_cache_read_tokens"] >= 0
    assert agent_result["n_cache_creation_tokens"] >= 0
    assert agent_result["total_tokens"] >= (
        agent_result["n_input_tokens"] + agent_result["n_output_tokens"]
    )
    assert agent_result["cost_usd"] is None or agent_result["cost_usd"] > 0

    llm_traj = result_json.parent / "trajectory" / "llm_trajectory.jsonl"
    assert llm_traj.exists()
    first = json.loads(llm_traj.read_text().splitlines()[0])
    assert "response" in first
