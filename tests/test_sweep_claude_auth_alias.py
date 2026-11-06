"""Guards FF-sw-cli-other-C5 at 795cb12c and its pre-existing 7a2387ac path."""

import pytest
from typer.testing import CliRunner

from benchflow.agents.env import resolve_agent_env, uses_native_subscription_auth
from benchflow.cli.main import app
from benchflow.sandbox.native_oauth import native_oauth_egress_policy
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _run_refusal, _task

TOKENS = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN")
MODELS = ("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6")


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    for key in (
        *TOKENS,
        "ANTHROPIC_API_KEY",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    ):
        # CLI startup can introduce normalized aliases. Track even an
        # initially absent key so teardown removes aliases created by the CLI.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda *a: False
    )


@pytest.mark.parametrize("command", [("check",), ("tasks", "check")])
@pytest.mark.parametrize("agent", ["claude", "claude-agent-acp"])
@pytest.mark.parametrize("source", ["host", "dotenv"])
@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("model", MODELS)
def test_check_acceptance_survives_real_auth_resolution(
    tmp_path, monkeypatch, command, agent, source, token, model
):
    """Guards 795cb12c: every native token spelling must survive runtime setup."""
    if source == "host":
        monkeypatch.setenv(token, "dummy-native-login")
    else:
        (tmp_path / "dummy.env").write_text(f"{token}=dummy-native-login\n")
    task = _task(tmp_path / "task", "[sandbox]\nallow_internet=false\n")
    checked = CliRunner().invoke(
        app, [*command, str(task), "--agent", agent, "--model", model]
    )
    assert checked.exit_code == 0, checked.output
    actual = resolve_agent_env("claude-agent-acp", model, {})
    policy = native_oauth_egress_policy("claude-agent-acp", model, actual, no_web=True)
    assert policy is not None and policy.native_claude_model_only
    assert actual[token] == "dummy-native-login"
    assert "BENCHFLOW_PROVIDER_API_KEY" not in actual
    assert "BENCHFLOW_PROVIDER_BASE_URL" not in actual


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("policy", ["offline", "allowlist"])
@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("model", MODELS)
def test_explicit_native_alias_keeps_backend_plan(
    tmp_path, backend, policy, token, model
):
    """Guards 795cb12c: the real resolver preserves each backend's policy gate."""
    config = (
        "[sandbox]\nallow_internet=false\n"
        if policy == "offline"
        else '[sandbox]\nnetwork_mode="allowlist"\nallowed_hosts=["example.invalid"]\n'
    )
    task = _task(tmp_path / "task", config)
    overlay = {token: "dummy-native-login"}
    actual = resolve_agent_env("claude-agent-acp", model, overlay)
    assert uses_native_subscription_auth("claude-agent-acp", model, actual)
    assert "BENCHFLOW_PROVIDER_API_KEY" not in actual
    assert "BENCHFLOW_PROVIDER_BASE_URL" not in actual
    for env in (overlay, actual):
        if backend == "modal" and policy == "allowlist":
            with pytest.raises(RuntimeError, match="not enforced by modal"):
                validate_task_plan(
                    task,
                    sandbox=backend,
                    agent="claude-agent-acp",
                    model=model,
                    agent_env=env,
                )
        else:
            assert (
                validate_task_plan(
                    task,
                    sandbox=backend,
                    agent="claude-agent-acp",
                    model=model,
                    agent_env=env,
                )
                is None
            )


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("model", MODELS)
def test_runtime_setup_reaches_allocation_with_native_alias(tmp_path, token, model):
    """Guards 795cb12c: real auth resolution and setup agree before allocation."""
    task = _task(tmp_path / "task", "[sandbox]\nallow_internet=false\n")
    actual = resolve_agent_env("claude-agent-acp", model, {token: "dummy-native-login"})
    assert _run_refusal(task, tmp_path, model=model, resolved_env=actual) is None


@pytest.mark.parametrize("model", MODELS)
def test_missing_native_credential_still_fails_closed(model):
    """Guards 795cb12c: a native model name alone cannot bypass auth validation."""
    from benchflow.errors import MissingCredentialError

    with pytest.raises(MissingCredentialError, match="No Claude login found"):
        resolve_agent_env("claude-agent-acp", model, {})


@pytest.mark.parametrize("model", (*MODELS, None))
def test_host_native_login_does_not_create_provider_route(monkeypatch, model):
    """Guards 795cb12c's sibling: file-backed native auth has the same invariant."""
    monkeypatch.setattr("benchflow.agents.env.check_subscription_auth", lambda *a: True)
    actual = resolve_agent_env("claude-agent-acp", model, {})
    assert actual["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"
    policy = native_oauth_egress_policy("claude-agent-acp", model, actual, no_web=True)
    assert policy is not None and policy.native_claude_model_only


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("token", TOKENS)
def test_explicit_conflicts_are_still_refused(model, token):
    """Guards 795cb12c: native normalization cannot erase an explicit route."""
    for key, value in (
        ("BENCHFLOW_PROVIDER_API_KEY", "dummy-explicit-key"),
        ("BENCHFLOW_PROVIDER_BASE_URL", "https://proxy.example.invalid"),
        ("LLM_BASE_URL", "https://proxy.example.invalid"),
        ("ANTHROPIC_BASE_URL", "https://proxy.example.invalid"),
    ):
        actual = resolve_agent_env(
            "claude-agent-acp", model, {token: "dummy-native-login", key: value}
        )
        assert actual[key] == value
        if key == "BENCHFLOW_PROVIDER_API_KEY":
            # C7 at d3647cbf: an explicit provider key selects API auth before
            # native detection, even when a native token is also available.
            assert not uses_native_subscription_auth("claude-agent-acp", model, actual)
            assert (
                native_oauth_egress_policy(
                    "claude-agent-acp", model, actual, no_web=True
                )
                is None
            )
            continue
        with pytest.raises(ValueError, match=r"conflicts with|canonical first-party"):
            native_oauth_egress_policy("claude-agent-acp", model, actual, no_web=True)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("token", TOKENS)
def test_api_key_still_selects_provider_route(model, token):
    """Guards 795cb12c: an API key still wins over a native login."""
    actual = resolve_agent_env(
        "claude-agent-acp",
        model,
        {token: "dummy-native-login", "ANTHROPIC_API_KEY": "dummy-api-key"},
    )
    assert not uses_native_subscription_auth("claude-agent-acp", model, actual)
    assert actual["BENCHFLOW_PROVIDER_API_KEY"]
    assert (
        native_oauth_egress_policy("claude-agent-acp", model, actual, no_web=True)
        is None
    )
