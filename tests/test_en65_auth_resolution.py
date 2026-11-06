"""Auth regressions present in candidate ac3bedaa1."""

from pathlib import Path

import pytest

from benchflow.agents import env as auth
from benchflow.agents.launch_env import agent_launch_env, credential_policy
from benchflow.errors import MissingCredentialError


@pytest.fixture(autouse=True)
def isolated_logins(monkeypatch, tmp_path):
    for key in auth.inheritable_env_names():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(auth, "load_dotenv_env", lambda: {})
    monkeypatch.setattr(auth, "login_path", lambda name: tmp_path / Path(name).name)
    for name in ("auth.json", ".credentials.json"):
        (tmp_path / name).write_text('{"dummy": true}')
    return tmp_path


NATIVE_LOGINS = [
    ("codex-acp", "openai/gpt-5.5", "OPENAI_API_KEY", "auth.json"),
    (
        "claude-agent-acp",
        "anthropic/claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        ".credentials.json",
    ),
]


@pytest.mark.parametrize("agent,model,key,filename", NATIVE_LOGINS)
@pytest.mark.parametrize("with_model", [True, False])
def test_blank_key_selects_host_login_after_reresolution(
    monkeypatch, agent, model, key, filename, with_model
):
    """Guards ac3bedaa1: an explicit blank suppresses inheritance, not host login."""
    monkeypatch.setenv(key, "dummy-inherited-api-key")
    model = model if with_model else None
    resolved = auth.resolve_agent_env(agent, model, {key: ""})
    for _ in range(2):
        assert resolved[key] == ""
        assert resolved["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"
        assert credential_policy(agent, model, resolved).host_login
        assert not agent_launch_env(agent, model, resolved).get(key)
        resolved = auth.resolve_agent_env(agent, model, resolved)


@pytest.mark.parametrize("agent,model,key,filename", NATIVE_LOGINS)
def test_blank_key_without_host_login_is_missing(
    isolated_logins, agent, model, key, filename
):
    """Guards ac3bedaa1: an empty API slot cannot satisfy required authentication."""
    (isolated_logins / filename).unlink()
    with pytest.raises(MissingCredentialError):
        auth.resolve_agent_env(agent, model, {key: ""})


@pytest.mark.parametrize("agent,model,key,filename", NATIVE_LOGINS)
def test_previous_host_marker_does_not_authorize_missing_file(
    isolated_logins, agent, model, key, filename
):
    """Guards ac3bedaa1: host login must be rediscovered, never trusted from input."""
    resolved = auth.resolve_agent_env(agent, model, {})
    (isolated_logins / filename).unlink()
    with pytest.raises(MissingCredentialError):
        auth.resolve_agent_env(agent, model, {**resolved, key: ""})


def test_blank_key_does_not_send_host_login_to_custom_endpoint():
    """Guards ac3bedaa1: the host-login repair preserves native endpoint limits."""
    with pytest.raises(MissingCredentialError):
        auth.resolve_agent_env(
            "codex-acp",
            "openai/gpt-5.5",
            {
                "OPENAI_API_KEY": "",
                "BENCHFLOW_PROVIDER_BASE_URL": "https://custom.invalid/v1",
            },
        )


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-4-6", "claude-sonnet-4-6", "gemini/gemini-2.5-pro"],
)
def test_codex_host_login_does_not_authenticate_other_model_families(model):
    """Guards ac3bedaa1: unknown provider prefixes cannot select Codex host auth."""
    assert not auth.uses_native_subscription_auth("codex-acp", model, {})


@pytest.mark.parametrize("agent,model,key,filename", NATIVE_LOGINS)
def test_nonempty_api_key_precedes_host_login(agent, model, key, filename):
    """Guards ac3bedaa1: explicit API-key precedence is unchanged by blank fallback."""
    resolved = auth.resolve_agent_env(agent, model, {key: "dummy-api-key"})
    assert resolved[key] == "dummy-api-key"
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in resolved
    assert not credential_policy(agent, model, resolved).host_login


@pytest.mark.parametrize("explicit", [False, True])
def test_bearer_precedes_oauth_at_same_priority(monkeypatch, explicit):
    """Guards ac3bedaa1: documented bearer precedence must survive launch filtering."""
    inputs = {
        "ANTHROPIC_AUTH_TOKEN": "dummy-bearer",
        "CLAUDE_CODE_OAUTH_TOKEN": "dummy-oauth",
    }
    if not explicit:
        for key, value in inputs.items():
            monkeypatch.setenv(key, value)
    resolved = auth.resolve_agent_env(
        "claude-agent-acp", "claude-sonnet-4-6", inputs if explicit else {}
    )
    launch = agent_launch_env("claude-agent-acp", "claude-sonnet-4-6", resolved)
    assert launch.get("ANTHROPIC_AUTH_TOKEN") == "dummy-bearer"
    assert not launch.get("CLAUDE_CODE_OAUTH_TOKEN")


def test_explicit_oauth_precedes_inherited_bearer(monkeypatch):
    """Guards ac3bedaa1: caller-selected login still outranks inherited credentials."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "dummy-inherited-bearer")
    resolved = auth.resolve_agent_env(
        "claude-agent-acp",
        "claude-sonnet-4-6",
        {"CLAUDE_CODE_OAUTH_TOKEN": "dummy-explicit-oauth"},
    )
    for _ in range(2):
        launch = agent_launch_env("claude-agent-acp", "claude-sonnet-4-6", resolved)
        assert launch.get("CLAUDE_CODE_OAUTH_TOKEN") == "dummy-explicit-oauth"
        assert not launch.get("ANTHROPIC_AUTH_TOKEN")
        resolved = auth.resolve_agent_env(
            "claude-agent-acp", "claude-sonnet-4-6", resolved
        )


@pytest.mark.parametrize("native_url", [None, "https://native.invalid"])
def test_native_bearer_retains_explicit_provider_endpoint(native_url):
    """Guards ac3bedaa1: native routing maps a supplied URL without inventing a key."""
    inputs = {
        "ANTHROPIC_AUTH_TOKEN": "dummy-bearer",
        "BENCHFLOW_PROVIDER_BASE_URL": "https://generic.invalid",
    }
    if native_url:
        inputs["ANTHROPIC_BASE_URL"] = native_url
    resolved = auth.resolve_agent_env("claude-agent-acp", "claude-sonnet-4-6", inputs)
    assert resolved["ANTHROPIC_BASE_URL"] == (native_url or "https://generic.invalid")
    assert "BENCHFLOW_PROVIDER_API_KEY" not in resolved
    assert "ANTHROPIC_API_KEY" not in resolved


@pytest.fixture
def own_route_agent():
    from benchflow.agents.registry import (
        AGENT_INSTALLERS,
        AGENT_LAUNCH,
        AGENTS,
        register_agent,
    )

    register_agent(
        "en191-own-route",
        "true",
        "en191-own-route --acp",
        requires_env=["EN191_ENDPOINT_KEY"],
        env_mapping={"BENCHFLOW_PROVIDER_API_KEY": "EN191_ENDPOINT_KEY"},
        model_route="agent",
    )
    yield "en191-own-route"
    for registry in (AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH):
        registry.pop("en191-own-route", None)


@pytest.mark.parametrize(
    "label", ["deepseek/deepseek-v4-pro", "anthropic/claude-sonnet-4-6"]
)
def test_own_model_route_label_needs_and_admits_no_provider_key(
    monkeypatch, own_route_agent, label
):
    """A model label of an agent that routes its own calls
    asks for no provider key (a proxied agent would be refused without one)
    and resolves to the same direct table as no model: only its own key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-host-anthropic")
    with pytest.raises(MissingCredentialError):
        auth.resolve_agent_env("openhands", "deepseek/deepseek-v4-pro", {})
    explicit = {"EN191_ENDPOINT_KEY": "dummy-own-key"}
    labelled = auth.resolve_agent_env(own_route_agent, label, dict(explicit))
    unlabelled = auth.resolve_agent_env(own_route_agent, None, dict(explicit))
    assert labelled == unlabelled
    policy = credential_policy(own_route_agent, label, labelled)
    assert policy.route == "direct"
    launch = agent_launch_env(own_route_agent, label, labelled)
    assert launch["EN191_ENDPOINT_KEY"] == "dummy-own-key"
    assert "ANTHROPIC_API_KEY" not in launch
    assert "BENCHFLOW_PROVIDER_MODEL" not in launch
