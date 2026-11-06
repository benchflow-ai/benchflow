"""Guards PR #78 explicit Claude Code direct endpoint routing."""

import pytest

from benchflow.providers.litellm_runtime import ensure_litellm_runtime


@pytest.mark.asyncio
async def test_explicit_direct_endpoint():
    env = {
        "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "test-token",
    }
    updated, runtime = await ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env=env,
        model="xiaomi/mimo-v2.6-pro",
        runtime=None,
        environment="docker",
        usage_tracking="off",
    )
    assert runtime is None
    assert updated["ANTHROPIC_BASE_URL"] == env["ANTHROPIC_BASE_URL"]
    assert updated["ANTHROPIC_MODEL"] == "mimo-v2.6-pro"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy", [{"usage_tracking": "required"}, {"disallow_web_tools": True}]
)
async def test_direct_endpoint_cannot_bypass_required_capture_or_restrictions(policy):
    with pytest.raises(RuntimeError, match="cannot enforce"):
        await ensure_litellm_runtime(
            agent="claude-agent-acp",
            model="xiaomi/mimo-v2.6-pro",
            runtime=None,
            agent_env={
                "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "test-token",
            },
            environment="docker",
            **policy,
        )


def test_direct_endpoint_credentials_survive_resolution(monkeypatch):
    from benchflow.agents.env import resolve_agent_env
    from benchflow.agents.launch_env import credential_policy

    env = {
        "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "test-token",
    }
    monkeypatch.setattr(
        "benchflow.agents.env.auto_inherit_env", lambda env, **kwargs: None
    )
    resolved = resolve_agent_env("claude-agent-acp", "xiaomi/mimo-v2.6-pro", env)
    policy = credential_policy("claude-agent-acp", "xiaomi/mimo-v2.6-pro", resolved)
    assert policy.route == "direct"
    assert policy.admitted_value("ANTHROPIC_AUTH_TOKEN", resolved) == "test-token"
    assert "BENCHFLOW_PROVIDER_NAME" not in resolved


@pytest.mark.parametrize(
    "patch",
    [
        {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        {"ANTHROPIC_BASE_URL": "file:///tmp/socket"},
        {"ANTHROPIC_AUTH_TOKEN": ""},
        {"ANTHROPIC_API_KEY": "api-key"},
        {"BENCHFLOW_PROVIDER_NAME": "litellm"},
    ],
)
def test_incomplete_or_proxy_owned_endpoints_are_not_direct(patch):
    from benchflow.agents.env import uses_direct_anthropic_endpoint

    env = {
        "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "test-token",
        **patch,
    }
    assert not uses_direct_anthropic_endpoint(
        "claude-agent-acp", "xiaomi/mimo-v2.6-pro", env
    )


def test_direct_endpoint_card_names_missing_capture(tmp_path):
    from benchflow.run_card import build_run_card, card_lines
    from tests.test_run_card import _evaluation, _tasks

    job = _evaluation(
        tmp_path,
        _tasks(tmp_path),
        model="xiaomi/mimo-v2.6-pro",
        agent_env={
            "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
            "ANTHROPIC_AUTH_TOKEN": "test-token",
        },
    )
    card = build_run_card(job, job._get_task_dirs())
    assert card["agent"]["direct_endpoint"] is True
    assert "no proxy call capture or cost accounting" in "\n".join(
        card_lines(card, tmp_path)
    )
    assert "test-token" not in str(card)
