"""A subscription is a login's contents, not a file's presence.

Fable's audit (CRED-1, CRED-4): any file at the Codex or Claude login path
made the run a "subscription", including a Codex auth.json that holds an API
key (``codex login --with-api-key``): the run went around the proxy with no
usage or cost, and the key was copied into the agent's home. Any
ANTHROPIC_AUTH_TOKEN did the same. Now an API key takes the API-key route,
through the proxy, and only a real login (OAuth tokens) is a subscription.
"""

from __future__ import annotations

import json

import pytest

_CLEARED = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "CODEX_AUTH_JSON",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "BENCHFLOW_PROVIDER_API_KEY",
    "BENCHFLOW_PROVIDER_BASE_URL",
)
CODEX_LOGIN = {
    "auth_mode": "chatgpt",
    "OPENAI_API_KEY": None,
    "tokens": {"access_token": "eyJ.fake.access", "refresh_token": "fake-refresh"},
}
CODEX_API_KEY_FILE = {"OPENAI_API_KEY": "sk-proj-FAKE-test-key-0000"}


def login_file_text(name: str, token: str) -> str:
    """A host login file in the shape its CLI writes, *token* in place of
     its secret: Claude's ``.credentials.json`` (an OAuth access token) or
     Codex's ``auth.json`` (a ChatGPT login). Only such contents are a login
    ; a test that tracks where a login goes follows *token*."""
    if name.endswith(".credentials.json"):
        return json.dumps({"claudeAiOauth": {"accessToken": token}})
    return json.dumps({"tokens": {"access_token": token}})


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host with no keys in its env, its logins under tmp_path."""
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)
    codex = tmp_path / "codex-home"
    claude = tmp_path / "claude-home"
    codex.mkdir()
    claude.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return codex, claude


def test_a_codex_auth_json_with_an_api_key_is_no_login(host):
    from benchflow.agents.env import check_subscription_auth

    codex, _ = host
    (codex / "auth.json").write_text(json.dumps(CODEX_API_KEY_FILE))
    assert check_subscription_auth("codex-acp", "OPENAI_API_KEY") is False
    (codex / "auth.json").write_text(json.dumps(CODEX_LOGIN))
    assert check_subscription_auth("codex-acp", "OPENAI_API_KEY") is True


def test_a_claude_file_without_an_oauth_token_is_no_login(host):
    from benchflow.agents.env import check_subscription_auth

    _, claude = host
    (claude / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {}}))
    assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY") is False
    (claude / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-FAKE"}})
    )
    assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY") is True


@pytest.mark.parametrize("inline", [False, True], ids=["host-file", "CODEX_AUTH_JSON"])
def test_an_api_key_in_a_codex_auth_json_goes_through_the_proxy(host, inline):
    """The key becomes OPENAI_API_KEY: the API-key route, proxied, with its
    usage and cost captured; no login is marked, so nothing copies the file
    into the agent's home."""
    from benchflow.agents.env import (
        _SUBSCRIPTION_AUTH_MARKER,
        resolve_agent_env,
        uses_native_subscription_auth,
    )

    codex, _ = host
    given: dict[str, str] = {}
    if inline:
        given["CODEX_AUTH_JSON"] = json.dumps(CODEX_API_KEY_FILE)
    else:
        (codex / "auth.json").write_text(json.dumps(CODEX_API_KEY_FILE))
    env = resolve_agent_env("codex-acp", "openai/gpt-6.1-sol", given)
    assert env.get("OPENAI_API_KEY") == CODEX_API_KEY_FILE["OPENAI_API_KEY"]
    assert "CODEX_AUTH_JSON" not in env
    assert env.get(_SUBSCRIPTION_AUTH_MARKER) != "1"
    assert not uses_native_subscription_auth("codex-acp", "openai/gpt-6.1-sol", env)


def test_a_chatgpt_login_stays_a_subscription(host):
    from benchflow.agents.env import uses_native_subscription_auth

    env = {"CODEX_AUTH_JSON": json.dumps(CODEX_LOGIN)}
    assert uses_native_subscription_auth("codex-acp", "openai/gpt-6.1-sol", env)


@pytest.mark.parametrize(
    ("token", "base_url", "login"),
    [
        ("sk-ant-oat01-FAKE-login-token", None, True),
        ("sk-ant-api03-FAKE-api-key", None, False),
        ("sk-ant-api03-FAKE-api-key", "https://gw.example.com", True),
        ("gateway-FAKE-token", None, True),
    ],
)
def test_an_api_key_in_the_token_slot_is_no_login_without_its_endpoint(
    token, base_url, login
):
    from benchflow.agents.env import uses_native_subscription_auth

    env = {"ANTHROPIC_AUTH_TOKEN": token}
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    assert (
        uses_native_subscription_auth(
            "claude-agent-acp", "anthropic/claude-opus-5-5", env
        )
        is login
    )


def test_an_anthropic_api_key_in_the_token_slot_goes_through_the_proxy(host):
    from benchflow.agents.env import resolve_agent_env, uses_native_subscription_auth

    env = resolve_agent_env(
        "claude-agent-acp",
        "anthropic/claude-opus-5-5",
        {"ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-FAKE-api-key"},
    )
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant-api03-FAKE-api-key"
    assert not uses_native_subscription_auth(
        "claude-agent-acp", "anthropic/claude-opus-5-5", env
    )
    # The same resolution as the key given as ANTHROPIC_API_KEY.
    direct = resolve_agent_env(
        "claude-agent-acp",
        "anthropic/claude-opus-5-5",
        {"ANTHROPIC_API_KEY": "sk-ant-api03-FAKE-api-key"},
    )
    assert env == direct


class TestAnthropicAuthTokenByContents:
    """Audit CRED-4 (the lead's ruling, as 0.9.3): ANTHROPIC_AUTH_TOKEN's
    contents decide its route, not its presence."""

    MODEL = "anthropic/claude-opus-5-5"

    def _resolve(self, env):
        from benchflow.agents.env import resolve_agent_env

        return resolve_agent_env("claude-agent-acp", self.MODEL, dict(env))

    def test_an_oauth_token_is_a_login(self, host):
        from benchflow.agents.env import uses_native_subscription_auth

        result = self._resolve({"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-FAKE"})
        assert result["ANTHROPIC_AUTH_TOKEN"] == "sk-ant-oat01-FAKE"
        assert not result.get("ANTHROPIC_API_KEY")
        assert uses_native_subscription_auth("claude-agent-acp", self.MODEL, result)

    @pytest.mark.parametrize("source", ["agent-env", "inherited"])
    def test_anything_else_is_refused_naming_the_variable(
        self, host, monkeypatch, source
    ):
        from benchflow.errors import MissingCredentialError

        given = {}
        if source == "inherited":
            monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-FAKE-123")
        else:
            given["ANTHROPIC_AUTH_TOKEN"] = "tok-FAKE-123"
        with pytest.raises(MissingCredentialError) as caught:
            self._resolve(given)
        assert "ANTHROPIC_AUTH_TOKEN" in str(caught.value)
        assert "tok-FAKE-123" not in str(caught.value)

    def test_a_gateway_s_bearer_with_its_endpoint_is_left_alone(self, host):
        result = self._resolve(
            {
                "ANTHROPIC_AUTH_TOKEN": "gw-FAKE-token",
                "ANTHROPIC_BASE_URL": "https://gateway.example/anthropic",
            }
        )
        assert result["ANTHROPIC_AUTH_TOKEN"] == "gw-FAKE-token"

    def test_an_exported_pair_is_refused(self, host, monkeypatch):
        """BenchFlow never takes ANTHROPIC_BASE_URL from the shell, so an
        exported token has no endpoint of its own: pass both with
        --agent-env."""
        from benchflow.errors import MissingCredentialError

        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gw-FAKE-token")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
        with pytest.raises(MissingCredentialError, match="ANTHROPIC_AUTH_TOKEN"):
            self._resolve({})

    @pytest.mark.parametrize(
        "login", ["CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]
    )
    def test_an_explicit_login_outranks_an_inherited_unknown_token(
        self, host, monkeypatch, login
    ):
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-FAKE-unknown")
        value = (
            "sk-ant-api03-FAKE-explicit"
            if login == "ANTHROPIC_API_KEY"
            else "sk-ant-oat01-FAKE-explicit"
        )
        result = self._resolve({login: value})
        assert result[login] == value

    def test_an_explicit_unknown_token_is_refused_beside_a_login(
        self, host, monkeypatch
    ):
        """The exemption is for an inherited token only: one the caller gave
        is still read by its contents."""
        from benchflow.errors import MissingCredentialError

        with pytest.raises(MissingCredentialError, match="ANTHROPIC_AUTH_TOKEN"):
            self._resolve(
                {
                    "ANTHROPIC_AUTH_TOKEN": "tok-FAKE-123",
                    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-FAKE",
                }
            )

    def test_another_model_s_route_is_not_read(self, host):
        """Only a Claude Code agent on an Anthropic model: a token for another
        endpoint (MiMo's, with its URL) is that route's business."""
        from benchflow.agents.env import resolve_agent_env

        result = resolve_agent_env(
            "claude-agent-acp",
            "xiaomi/mimo-v2.6-pro",
            {
                "ANTHROPIC_AUTH_TOKEN": "tok-FAKE-123",
                "ANTHROPIC_BASE_URL": "https://api.xiaomimimo.com/anthropic",
            },
        )
        assert result.get("ANTHROPIC_AUTH_TOKEN") == "tok-FAKE-123"
