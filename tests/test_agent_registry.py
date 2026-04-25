"""Tests for AgentConfig + ProviderConfig registry shape:
env_mapping (BENCHFLOW_PROVIDER_* → agent-native vars) and credential_files.

Negative invariants ("agent X should NOT have feature Y configured") live in
test_registry_invariants.py — search there for the consolidated tripwire.
"""

from benchflow._agent_env import resolve_provider_env
from benchflow.agents.providers import PROVIDERS
from benchflow.agents.registry import AGENTS


class TestEnvMappingField:
    """env_mapping exists on AgentConfig and is populated for known agents."""

    def test_claude_agent_has_mapping(self):
        cfg = AGENTS["claude-agent-acp"]
        assert "BENCHFLOW_PROVIDER_BASE_URL" in cfg.env_mapping
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "ANTHROPIC_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "ANTHROPIC_AUTH_TOKEN"

    def test_pi_acp_no_static_mapping(self):
        """pi-acp is multi-protocol — launch wrapper handles env translation."""
        cfg = AGENTS["pi-acp"]
        assert cfg.env_mapping == {}

    def test_codex_acp_has_mapping(self):
        cfg = AGENTS["codex-acp"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "OPENAI_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "OPENAI_API_KEY"

    def test_gemini_has_mapping(self):
        cfg = AGENTS["gemini"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "GEMINI_API_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "GOOGLE_API_KEY"

    def test_openhands_has_mapping(self):
        cfg = AGENTS["openhands"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "LLM_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "LLM_API_KEY"
        # OpenHands model is normalized in _normalize_openhands_model().
        assert "BENCHFLOW_PROVIDER_MODEL" not in cfg.env_mapping

    def test_openhands_normalizes_model(self):
        env = {}
        resolve_provider_env(
            agent="openhands",
            model="zai/glm-5",
            agent_env=env,
        )

        assert env["LLM_MODEL"] == "glm-5"


class TestOpenHandsConfig:
    def test_openhands_uses_agentskills_paths(self):
        cfg = AGENTS["openhands"]
        assert "$HOME/.agents/skills" in cfg.skill_paths
        assert "$WORKSPACE/.agents/skills" in cfg.skill_paths

    def test_openhands_install_cmd_has_uv_and_binary_fallbacks(self):
        cfg = AGENTS["openhands"]
        assert "apt-get install -y -qq curl ca-certificates" in cfg.install_cmd
        assert "uv tool install openhands --python 3.12" in cfg.install_cmd
        assert "install.openhands.dev/install.sh" in cfg.install_cmd

    def test_openhands_skips_acp_set_model(self):
        cfg = AGENTS["openhands"]
        assert cfg.supports_acp_set_model is False


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
