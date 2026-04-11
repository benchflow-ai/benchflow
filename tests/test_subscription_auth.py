"""Tests for subscription auth — host CLI credentials as API key fallback."""

from pathlib import Path

import pytest

from benchflow.agents.registry import (
    AGENTS,
    AgentConfig,
    HostAuthFile,
    SubscriptionAuth,
    get_sandbox_home_dirs,
)

# ── SubscriptionAuth dataclass ──


class TestSubscriptionAuthDataclass:
    def test_construction(self):
        sa = SubscriptionAuth(
            replaces_env="ANTHROPIC_API_KEY",
            detect_file="~/.claude/.credentials.json",
            files=[
                HostAuthFile(
                    "~/.claude/.credentials.json", "{home}/.claude/.credentials.json"
                ),
            ],
        )
        assert sa.replaces_env == "ANTHROPIC_API_KEY"
        assert sa.detect_file == "~/.claude/.credentials.json"
        assert len(sa.files) == 1

    def test_agent_config_default_none(self):
        cfg = AgentConfig(name="t", install_cmd="", launch_cmd="")
        assert cfg.subscription_auth is None


# ── Agent config entries ──


class TestAgentSubscriptionAuth:
    def test_claude_subscription_auth(self):
        cfg = AGENTS["claude-agent-acp"]
        sa = cfg.subscription_auth
        assert sa is not None
        assert sa.replaces_env == "ANTHROPIC_API_KEY"
        assert ".claude/.credentials.json" in sa.detect_file
        assert len(sa.files) == 1

    def test_codex_subscription_auth(self):
        cfg = AGENTS["codex-acp"]
        sa = cfg.subscription_auth
        assert sa is not None
        assert sa.replaces_env == "OPENAI_API_KEY"
        assert ".codex/auth.json" in sa.detect_file
        assert len(sa.files) == 1

    def test_gemini_subscription_auth(self):
        cfg = AGENTS["gemini"]
        sa = cfg.subscription_auth
        assert sa is not None
        assert sa.replaces_env == "GEMINI_API_KEY"
        assert ".gemini/oauth_creds.json" in sa.detect_file
        # Gemini needs multiple files (oauth, settings, accounts)
        assert len(sa.files) == 3
        paths = [f.host_path for f in sa.files]
        assert any("oauth_creds.json" in p for p in paths)
        assert any("settings.json" in p for p in paths)
        assert any("google_accounts.json" in p for p in paths)

    def test_openclaw_no_subscription_auth(self):
        cfg = AGENTS["openclaw"]
        assert cfg.subscription_auth is None

    def test_pi_no_subscription_auth(self):
        cfg = AGENTS["pi-acp"]
        assert cfg.subscription_auth is None


# ── get_sandbox_home_dirs includes subscription auth dirs ──


class TestSandboxHomeDirsSubscription:
    def test_claude_dir_included(self):
        dirs = get_sandbox_home_dirs()
        assert ".claude" in dirs

    def test_codex_dir_included(self):
        dirs = get_sandbox_home_dirs()
        assert ".codex" in dirs

    def test_gemini_dir_included(self):
        dirs = get_sandbox_home_dirs()
        assert ".gemini" in dirs


# ── _resolve_agent_env subscription fallback ──


def _patch_expanduser(monkeypatch, tmp_path):
    """Patch Path.expanduser to redirect ~ to tmp_path."""
    orig = Path.expanduser

    def fake(self):
        s = str(self)
        if s.startswith("~"):
            return tmp_path / s[2:]  # strip ~/
        return orig(self)

    monkeypatch.setattr(Path, "expanduser", fake)


class TestResolveAgentEnvSubscription:
    def _resolve(self, agent="claude-agent-acp", model=None, agent_env=None):
        from benchflow.sdk import SDK

        return SDK._resolve_agent_env(agent, model, agent_env)

    def test_api_key_present_no_subscription_marker(self):
        """When API key is provided, no subscription auth marker is set."""
        result = self._resolve(
            model="claude-haiku-4-5-20251001",
            agent_env={"ANTHROPIC_API_KEY": "sk-test"},
        )
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_subscription_auth_detected(self, monkeypatch, tmp_path):
        """When host auth file exists and no API key, subscription auth is used."""
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text('{"claudeAiOauth": {}}')
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            model="claude-haiku-4-5-20251001",
            agent_env={},
        )
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_no_auth_file_raises(self, monkeypatch, tmp_path):
        """When no API key and no host auth file, raises ValueError."""
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="log in with the agent CLI"):
            self._resolve(
                model="claude-haiku-4-5-20251001",
                agent_env={},
            )

    def test_api_key_takes_precedence_over_host_auth(self, monkeypatch, tmp_path):
        """Explicit API key wins even when host auth file exists."""
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text("{}")
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            model="claude-haiku-4-5-20251001",
            agent_env={"ANTHROPIC_API_KEY": "sk-explicit"},
        )
        assert result["ANTHROPIC_API_KEY"] == "sk-explicit"
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_codex_subscription_auth(self, monkeypatch, tmp_path):
        """Codex subscription auth works with host ~/.codex/auth.json."""
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text('{"OPENAI_API_KEY": "from-login"}')
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="codex-acp",
            model="gpt-4o",
            agent_env={},
        )
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_gemini_subscription_auth(self, monkeypatch, tmp_path):
        """Gemini subscription auth works with host ~/.gemini/oauth_creds.json."""
        for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "oauth_creds.json").write_text('{"access_token": "at-test"}')
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="gemini",
            model="gemini-2.5-flash",
            agent_env={},
        )
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"
