"""Tests for AgentConfig + ProviderConfig registry shape:
env_mapping (BENCHFLOW_PROVIDER_* → agent-native vars) and credential_files."""

from benchflow.agents.providers import PROVIDERS
from benchflow.agents.registry import AGENTS, AgentConfig


class TestEnvMappingField:
    """env_mapping exists on AgentConfig and is populated for known agents."""

    def test_agentconfig_has_env_mapping(self):
        cfg = AgentConfig(name="t", install_cmd="", launch_cmd="")
        assert hasattr(cfg, "env_mapping")
        assert cfg.env_mapping == {}

    def test_claude_agent_has_mapping(self):
        cfg = AGENTS["claude-agent-acp"]
        assert "BENCHFLOW_PROVIDER_BASE_URL" in cfg.env_mapping
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "ANTHROPIC_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "ANTHROPIC_AUTH_TOKEN"

    def test_pi_acp_has_mapping(self):
        cfg = AGENTS["pi-acp"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "ANTHROPIC_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "ANTHROPIC_AUTH_TOKEN"

    def test_codex_acp_has_mapping(self):
        cfg = AGENTS["codex-acp"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "OPENAI_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "OPENAI_API_KEY"

    def test_gemini_has_mapping(self):
        cfg = AGENTS["gemini"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "GEMINI_API_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "GOOGLE_API_KEY"

    def test_openclaw_no_mapping(self):
        """Openclaw reads BENCHFLOW_PROVIDER_* directly via shim — no mapping needed."""
        cfg = AGENTS["openclaw"]
        assert cfg.env_mapping == {}


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
