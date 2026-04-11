"""Tests for AgentConfig + ProviderConfig registry shape:
env_mapping (BENCHFLOW_PROVIDER_* → agent-native vars) and credential_files.

Negative invariants ("agent X should NOT have feature Y configured") live in
test_registry_invariants.py — search there for the consolidated tripwire.
"""

from benchflow.agents.providers import PROVIDERS
from benchflow.agents.registry import AGENTS


class TestEnvMappingField:
    """env_mapping exists on AgentConfig and is populated for known agents."""

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


class TestAgentCredentialFiles:
    def test_codex_has_auth_json(self):
        cfg = AGENTS["codex-acp"]
        assert len(cfg.credential_files) == 1
        cf = cfg.credential_files[0]
        assert cf.env_source == "OPENAI_API_KEY"
        assert ".codex/auth.json" in cf.path
        assert "{home}" in cf.path
        assert "{value}" in cf.template


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
