"""Tests for subscription auth — host CLI credentials as API key fallback."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import (
    AGENTS,
    get_sandbox_home_dirs,
)

# Agent config entries


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


# Negative invariants ("agent X should NOT have subscription_auth") collapsed
# into the consolidated tripwire in test_registry_invariants.py.


# get_sandbox_home_dirs includes subscription auth dirs


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


# _resolve_agent_env subscription fallback


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
        from benchflow.agents.env import resolve_agent_env

        return resolve_agent_env(agent, model, agent_env)

    def test_api_key_present_no_subscription_marker(self):
        """When API key is provided, no subscription auth marker is set."""
        result = self._resolve(
            model="claude-haiku-4-5-20251001",
            agent_env={"ANTHROPIC_API_KEY": "sk-test"},
        )
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_claude_oauth_alias_satisfies_anthropic_key_requirement(self):
        """Guards PR #587: CLAUDE_OAUTH_TOKEN is accepted as a Claude Code alias."""
        result = self._resolve(
            model="claude-haiku-4-5-20251001",
            agent_env={"CLAUDE_OAUTH_TOKEN": "oauth-test"},
        )

        assert result["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-test"
        assert "ANTHROPIC_API_KEY" not in result

    def test_subscription_auth_detected(self, monkeypatch, tmp_path):
        """When host auth file exists and no API key, subscription auth is used."""
        for k in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_OAUTH_TOKEN",
            "CODEX_AUTH_JSON",
            "CODEX_ACCESS_TOKEN",
            "CODEX_API_KEY",
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
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_OAUTH_TOKEN",
            "CODEX_AUTH_JSON",
            "CODEX_ACCESS_TOKEN",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match=r"required for model .* but not set"):
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
        for k in (
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_JSON",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
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

    def test_codex_auth_json_auth(self, monkeypatch, tmp_path):
        """Guards PR #587: inline Codex auth.json can auth native Codex runs."""
        for k in ("CODEX_ACCESS_TOKEN", "CODEX_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="codex-acp",
            model="gpt-4o",
            agent_env={
                "CODEX_AUTH_JSON": '{"tokens": {"access_token": "access-token"}}'
            },
        )

        assert result["CODEX_AUTH_JSON"].startswith("{")
        assert "OPENAI_API_KEY" not in result
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_codex_auth_json_marks_native_subscription_usage_path(self):
        """Guards PR #613 follow-up: Codex subscription runs bypass LiteLLM."""
        from benchflow.agents.env import uses_native_subscription_auth

        assert uses_native_subscription_auth(
            "codex-acp",
            "gpt-4o",
            {"CODEX_AUTH_JSON": '{"tokens": {"access_token": "access-token"}}'},
        )

    def test_codex_api_key_prefers_litellm_usage_path(self):
        """Guards PR #613 follow-up: API-key Codex runs stay on LiteLLM."""
        from benchflow.agents.env import uses_native_subscription_auth

        assert not uses_native_subscription_auth(
            "codex-acp",
            "gpt-4o",
            {
                "CODEX_AUTH_JSON": '{"tokens": {"access_token": "access-token"}}',
                "OPENAI_API_KEY": "sk-openai",
            },
        )

    def test_codex_custom_base_url_not_native_subscription_usage_path(self):
        """Guards PR #613 follow-up: subscription auth is not proxy auth."""
        from benchflow.agents.env import uses_native_subscription_auth

        assert not uses_native_subscription_auth(
            "codex-acp",
            "gpt-4o",
            {
                "CODEX_AUTH_JSON": '{"tokens": {"access_token": "access-token"}}',
                "OPENAI_BASE_URL": "http://localhost:8765/v1",
            },
        )

    def test_codex_access_token_auth(self, monkeypatch, tmp_path):
        """Guards PR #296: Blocks-style Codex auth via CODEX_ACCESS_TOKEN."""
        for k in ("CODEX_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="codex-acp",
            model="gpt-4o",
            agent_env={"CODEX_ACCESS_TOKEN": "access-token"},
        )

        assert result["CODEX_ACCESS_TOKEN"] == "access-token"
        assert "OPENAI_API_KEY" not in result
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_claude_oauth_marks_native_subscription_usage_path(self):
        """Guards PR #613 follow-up: Claude Code OAuth bypasses LiteLLM."""
        from benchflow.agents.env import uses_native_subscription_auth

        assert uses_native_subscription_auth(
            "claude-agent-acp",
            "claude-haiku-4-5-20251001",
            {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"},
        )

    def test_claude_api_key_prefers_litellm_usage_path(self):
        """Guards PR #613 follow-up: Claude API-key runs stay on LiteLLM."""
        from benchflow.agents.env import uses_native_subscription_auth

        assert not uses_native_subscription_auth(
            "claude-agent-acp",
            "claude-haiku-4-5-20251001",
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "sk-ant",
            },
        )

    def test_claude_litellm_auth_token_is_not_subscription_usage_path(self):
        """Guards PR #613 follow-up: LiteLLM rewrites must not look like OAuth."""
        from benchflow.agents.env import uses_native_subscription_auth

        assert not uses_native_subscription_auth(
            "claude-agent-acp",
            "claude-haiku-4-5-20251001",
            {
                "ANTHROPIC_AUTH_TOKEN": "sk-benchflow-master",
                "BENCHFLOW_PROVIDER_NAME": "litellm",
                "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
            },
        )

    def test_codex_api_key_auth_alias(self, monkeypatch, tmp_path):
        """Guards PR #296: CODEX_API_KEY works for native Codex auth."""
        for k in ("CODEX_ACCESS_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="codex-acp",
            model="gpt-4o",
            agent_env={"CODEX_API_KEY": "codex-key"},
        )

        assert result["CODEX_API_KEY"] == "codex-key"
        assert result["OPENAI_API_KEY"] == "codex-key"
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "codex-key"
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_codex_access_token_does_not_auth_custom_provider(
        self, monkeypatch, tmp_path
    ):
        """Guards PR #296: access tokens are not proxy API keys."""
        for k in (
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_JSON",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            self._resolve(
                agent="codex-acp",
                model="vllm/mock-model",
                agent_env={"CODEX_ACCESS_TOKEN": "access-token"},
            )

    @pytest.mark.parametrize(
        "base_url_key",
        ["BENCHFLOW_PROVIDER_BASE_URL", "OPENAI_BASE_URL"],
    )
    def test_codex_subscription_auth_does_not_auth_custom_base_url(
        self, monkeypatch, tmp_path, base_url_key
    ):
        """Guards PR #296: subscription auth is not custom endpoint API-key auth."""
        for k in (
            "CODEX_API_KEY",
            "CODEX_AUTH_JSON",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        _patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            self._resolve(
                agent="codex-acp",
                model="gpt-4o",
                agent_env={
                    "CODEX_ACCESS_TOKEN": "access-token",
                    base_url_key: "http://localhost:8765/v1",
                },
            )

        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            self._resolve(
                agent="codex-acp",
                model="gpt-4o",
                agent_env={
                    "CODEX_AUTH_JSON": '{"tokens": {"access_token": "access-token"}}',
                    base_url_key: "http://localhost:8765/v1",
                },
            )

    @pytest.mark.parametrize(
        "base_url_key",
        ["BENCHFLOW_PROVIDER_BASE_URL", "OPENAI_BASE_URL"],
    )
    def test_codex_host_login_does_not_auth_custom_base_url(
        self, monkeypatch, tmp_path, base_url_key
    ):
        """Guards PR #296: host login is not custom endpoint API-key auth."""
        for k in (
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_JSON",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text('{"tokens": {"access_token": "at"}}')
        _patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            self._resolve(
                agent="codex-acp",
                model="gpt-4o",
                agent_env={base_url_key: "http://localhost:8765/v1"},
            )

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


class _FakeEnv:
    def __init__(self):
        self.exec_calls = []
        self.uploads = []

    async def exec(self, cmd: str, timeout_sec: int | None = None):
        self.exec_calls.append((cmd, timeout_sec))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source: str, dest: str):
        self.uploads.append((source, dest, Path(source).read_text()))


class TestUploadSubscriptionAuth:
    @pytest.mark.asyncio
    async def test_codex_auth_json_writes_auth_file(self):
        """Guards PR #587: inline Codex auth.json is uploaded for Daytona."""
        from benchflow.agents.credentials import write_credential_files

        env = _FakeEnv()
        await write_credential_files(
            env,
            "codex-acp",
            {"CODEX_AUTH_JSON": '{"tokens": {"access_token": "test"}}'},
            AGENTS["codex-acp"],
            "gpt-4o",
            "/home/agent",
        )

        assert len(env.uploads) == 1
        assert env.uploads[0][1:] == (
            "/home/agent/.codex/auth.json",
            '{"tokens": {"access_token": "test"}}',
        )

    @pytest.mark.asyncio
    async def test_openai_key_wins_over_codex_auth_json_file_write(self):
        """Guards PR #587: API-key auth keeps the existing Codex file shape."""
        from benchflow.agents.credentials import write_credential_files

        env = _FakeEnv()
        await write_credential_files(
            env,
            "codex-acp",
            {
                "OPENAI_API_KEY": "sk-test",
                "CODEX_AUTH_JSON": '{"tokens": {"access_token": "test"}}',
            },
            AGENTS["codex-acp"],
            "gpt-4o",
            "/home/agent",
        )

        assert len(env.uploads) == 1
        assert env.uploads[0][1] == "/home/agent/.codex/auth.json"
        assert env.uploads[0][2] == '{"OPENAI_API_KEY": "sk-test"}'

    @pytest.mark.asyncio
    async def test_subscription_auth_chowns_uploaded_home_file(
        self, monkeypatch, tmp_path
    ):
        """Guards the Codex ACP dogfood failure from 2026-05-19.

        Host auth files are staged as root-owned temp files. After upload, the
        sandbox user must own the credential directory and file, otherwise the
        ACP process exits with "Permission denied" while loading config.
        """
        from benchflow.agents.credentials import upload_subscription_auth

        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text('{"token": "test"}')
        _patch_expanduser(monkeypatch, tmp_path)

        env = _FakeEnv()
        await upload_subscription_auth(env, "codex-acp", "/home/agent")

        assert len(env.uploads) == 1
        assert env.uploads[0][1:] == (
            "/home/agent/.codex/auth.json",
            '{"token": "test"}',
        )
        assert (
            "chown -R agent:agent /home/agent/.codex "
            "&& chmod 600 /home/agent/.codex/auth.json",
            10,
        ) in env.exec_calls
