"""Tests for AgentConfig + ProviderConfig registry shape:
env_mapping (BENCHFLOW_PROVIDER_* → agent-native vars) and credential_files.

Negative invariants ("agent X should NOT have feature Y configured") live in
test_registry_invariants.py — search there for the consolidated tripwire.
"""

import pytest

from benchflow.agents.env import resolve_provider_env
from benchflow.agents.providers import PROVIDERS
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    register_agent,
)


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
        assert cfg.acp_model_format == "registered-provider/model"

    def test_codex_acp_has_mapping(self):
        cfg = AGENTS["codex-acp"]
        assert cfg.api_protocol == "openai-responses"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "OPENAI_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "OPENAI_API_KEY"
        assert "openai_base_url=$OPENAI_BASE_URL" in cfg.launch_cmd

    def test_gemini_has_mapping(self):
        cfg = AGENTS["gemini"]
        assert (
            cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "GOOGLE_GEMINI_BASE_URL"
        )
        # #342: map BENCHFLOW_PROVIDER_API_KEY to the CLI-native var. The
        # bidirectional mirror in auto_inherit_env handles GOOGLE_API_KEY
        # callers transparently.
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "GEMINI_API_KEY"

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

    def test_openhands_install_cmd_forces_github_main(self):
        cfg = AGENTS["openhands"]
        assert "apt-get install -y -qq curl ca-certificates git" in cfg.install_cmd
        assert (
            "uv tool install --force --refresh "
            "--from 'git+https://github.com/OpenHands/OpenHands-CLI.git@main' "
            "openhands --python 3.12" in cfg.install_cmd
        )
        assert "command -v git" in cfg.install_cmd
        assert "install.openhands.dev/install.sh" not in cfg.install_cmd

    def test_openhands_skips_acp_set_model(self):
        cfg = AGENTS["openhands"]
        assert cfg.supports_acp_set_model is False

    def test_openhands_launch_cmd_writes_optional_azure_api_version(self):
        """Guards the fix from PR #559 against dropping Azure API version config."""
        cfg = AGENTS["openhands"]
        assert 'if [ -n "$LLM_API_VERSION" ]' in cfg.launch_cmd
        assert ',"api_version":"%s"' in cfg.launch_cmd
        assert '"$LLM_API_VERSION"' in cfg.launch_cmd

    def test_harvey_lab_installs_python_deps_in_venv(self):
        """Guards the v0.5 stress failure where pip hit PEP 668 in Ubuntu."""
        cfg = AGENTS["harvey-lab-harness"]
        assert "python3 -m venv /opt/benchflow/harvey-lab-venv" in cfg.install_cmd
        assert (
            "/opt/benchflow/harvey-lab-venv/bin/python -m pip install"
            in cfg.install_cmd
        )
        assert "pip3 install -q anthropic" not in cfg.install_cmd
        assert cfg.launch_cmd.startswith(
            "HARVEY_LABS_ROOT=/opt/harvey-labs "
            "/opt/benchflow/harvey-lab-venv/bin/python "
        )


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


class TestRegisterAgent:
    """register_agent() must pass through every AgentConfig field a runtime
    agent may need — including no-web-policy and provider-protocol routing.
    """

    @pytest.fixture
    def cleanup_agent(self):
        registered: list[str] = []
        yield registered
        for name in registered:
            AGENTS.pop(name, None)
            AGENT_INSTALLERS.pop(name, None)
            AGENT_LAUNCH.pop(name, None)

    def test_defaults(self, cleanup_agent):
        cleanup_agent.append("rt-defaults-agent")
        cfg = register_agent(
            name="rt-defaults-agent",
            install_cmd="install rt",
            launch_cmd="launch rt",
        )
        assert cfg.default_model == ""
        assert cfg.api_protocol == ""
        assert cfg.disallow_web_tools_setup_cmd == ""
        assert cfg.disallow_web_tools_launch_suffix == ""

    def test_passes_through_new_fields(self, cleanup_agent):
        cleanup_agent.append("rt-full-agent")
        cfg = register_agent(
            name="rt-full-agent",
            install_cmd="install rt",
            launch_cmd="launch rt",
            default_model="rt-model-1",
            api_protocol="openai-completions",
            disallow_web_tools_setup_cmd="printf 'no web' > /tmp/policy",
            disallow_web_tools_launch_suffix=" --no-web",
        )
        assert cfg.default_model == "rt-model-1"
        assert cfg.api_protocol == "openai-completions"
        assert cfg.disallow_web_tools_setup_cmd == "printf 'no web' > /tmp/policy"
        assert cfg.disallow_web_tools_launch_suffix == " --no-web"

        # And the registered entry reflects them.
        registered = AGENTS["rt-full-agent"]
        assert registered.default_model == "rt-model-1"
        assert registered.api_protocol == "openai-completions"
        assert registered.disallow_web_tools_launch_suffix == " --no-web"
