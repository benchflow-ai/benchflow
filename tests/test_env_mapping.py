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


class TestEnvMappingApplication:
    """The mapping logic: translate source vars to target vars using setdefault."""

    @staticmethod
    def _apply_env_mapping(agent_env: dict, env_mapping: dict) -> dict:
        """Replicate the SDK's env_mapping logic for unit testing."""
        for src, dst in env_mapping.items():
            if src in agent_env:
                agent_env.setdefault(dst, agent_env[src])
        return agent_env

    def test_mapping_translates_provider_vars(self):
        env = {"BENCHFLOW_PROVIDER_BASE_URL": "https://api.z.ai/v1"}
        mapping = {"BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL"}
        result = self._apply_env_mapping(env, mapping)
        assert result["ANTHROPIC_BASE_URL"] == "https://api.z.ai/v1"

    def test_mapping_uses_setdefault(self):
        """Explicit --ae values are not overwritten by env_mapping."""
        env = {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://api.z.ai/v1",
            "ANTHROPIC_BASE_URL": "https://custom.example.com",
        }
        mapping = {"BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL"}
        result = self._apply_env_mapping(env, mapping)
        assert result["ANTHROPIC_BASE_URL"] == "https://custom.example.com"

    def test_no_mapping_when_source_missing(self):
        """If BENCHFLOW_PROVIDER_BASE_URL not in env, no target var created."""
        env = {"OTHER_VAR": "value"}
        mapping = {"BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL"}
        result = self._apply_env_mapping(env, mapping)
        assert "ANTHROPIC_BASE_URL" not in result

    def test_multiple_mappings(self):
        env = {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://api.z.ai/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "sk-test",
        }
        mapping = {
            "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
        }
        result = self._apply_env_mapping(env, mapping)
        assert result["ANTHROPIC_BASE_URL"] == "https://api.z.ai/v1"
        assert result["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
