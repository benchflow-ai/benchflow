"""Tests for credential_files on AgentConfig and ProviderConfig."""

from benchflow.agents.providers import PROVIDERS, ProviderConfig
from benchflow.agents.registry import AGENTS, AgentConfig, CredentialFile


class TestCredentialFileDataclass:
    def test_template_rendering(self):
        cf = CredentialFile(
            path="/root/.codex/auth.json",
            env_source="OPENAI_API_KEY",
            template='{{"OPENAI_API_KEY": "{value}"}}',
        )
        value = "sk-test-123"
        content = cf.template.format(value=value)
        assert content == '{"OPENAI_API_KEY": "sk-test-123"}'

    def test_raw_value_when_no_template(self):
        cf = CredentialFile(
            path="/root/.config/creds.json",
            env_source="MY_CREDS",
        )
        assert cf.template == ""

    def test_mkdir_default_true(self):
        cf = CredentialFile(path="/x", env_source="Y")
        assert cf.mkdir is True

    def test_defaults(self):
        cfg = AgentConfig(name="t", install_cmd="", launch_cmd="")
        assert cfg.credential_files == []


class TestAgentCredentialFiles:
    def test_codex_has_auth_json(self):
        cfg = AGENTS["codex-acp"]
        assert len(cfg.credential_files) == 1
        cf = cfg.credential_files[0]
        assert cf.env_source == "OPENAI_API_KEY"
        assert ".codex/auth.json" in cf.path
        assert "{home}" in cf.path
        assert "{value}" in cf.template

    def test_claude_no_credential_files(self):
        cfg = AGENTS["claude-agent-acp"]
        assert cfg.credential_files == []

    def test_openclaw_no_credential_files(self):
        cfg = AGENTS["openclaw"]
        assert cfg.credential_files == []


class TestProviderCredentialFiles:
    def test_provider_config_has_field(self):
        cfg = ProviderConfig(
            name="t", base_url="", api_protocol="", auth_type="api_key"
        )
        assert cfg.credential_files == []

    def test_vertex_providers_have_adc(self):
        for name in ("google-vertex", "anthropic-vertex"):
            cfg = PROVIDERS[name]
            assert len(cfg.credential_files) == 1, (
                f"{name} should have 1 credential_file"
            )
            cf = cfg.credential_files[0]
            assert cf["env_source"] == "GOOGLE_APPLICATION_CREDENTIALS_JSON"
            assert "gcloud" in cf["path"]
            assert "GOOGLE_APPLICATION_CREDENTIALS" in cf.get("post_env", {})

    def test_zai_no_credential_files(self):
        cfg = PROVIDERS["zai"]
        assert cfg.credential_files == []
