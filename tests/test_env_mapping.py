"""Tests for AgentConfig.env_mapping — translates BENCHFLOW_PROVIDER_* to agent-native vars."""

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
