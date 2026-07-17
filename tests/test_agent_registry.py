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
        """Guards PR #846's Claude ACP pin and config-option invariant."""
        cfg = AGENTS["claude-agent-acp"]
        assert "BENCHFLOW_PROVIDER_BASE_URL" in cfg.env_mapping
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "ANTHROPIC_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "ANTHROPIC_AUTH_TOKEN"
        assert cfg.supports_acp_set_model is False
        assert cfg.acp_model_config_id == "model"
        assert cfg.acp_effort_config_id == "effort"
        assert "@agentclientprotocol/claude-agent-acp@0.55.0" in cfg.install_cmd

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

    def test_codex_acp_install_is_version_pinned(self):
        """Same @agentclientprotocol family as claude — pin so a floating latest
        can't silently break activation when upstream drops session/set_model."""
        cfg = AGENTS["codex-acp"]
        assert "@agentclientprotocol/codex-acp@0.0.45" in cfg.install_cmd

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

        assert env["LLM_MODEL"] == "openai/glm-5"

    def test_openhands_normalizes_github_models_model(self):
        env = {"GITHUB_TOKEN": "ghs_test_token"}
        resolve_provider_env(
            agent="openhands",
            model="github-models/openai/gpt-4.1-mini",
            agent_env=env,
        )

        assert env["BENCHFLOW_PROVIDER_NAME"] == "github-models"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "openai/gpt-4.1-mini"
        assert (
            env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://models.github.ai/inference"
        )
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "ghs_test_token"
        assert env["LLM_BASE_URL"] == "https://models.github.ai/inference"
        assert env["LLM_API_KEY"] == "ghs_test_token"
        assert env["LLM_MODEL"] == "openai/openai/gpt-4.1-mini"

    def test_openhands_bedrock_initial_env_marks_registered_provider(self):
        """Guards the LiteLLM runtime refactor: Bedrock is detected before runtime rewrite."""
        env = {
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-west-2",
        }

        resolve_provider_env(
            agent="openhands",
            model="aws-bedrock/us.anthropic.claude-opus-4-7",
            agent_env=env,
        )

        assert env["BENCHFLOW_PROVIDER_NAME"] == "aws-bedrock"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "us.anthropic.claude-opus-4-7"


class TestOpenHandsConfig:
    def test_openhands_uses_agentskills_paths(self):
        cfg = AGENTS["openhands"]
        assert "$HOME/.agents/skills" in cfg.skill_paths
        assert "$WORKSPACE/.agents/skills" in cfg.skill_paths

    def test_openhands_install_cmd_pins_cli_git_revision(self):
        cfg = AGENTS["openhands"]
        assert (
            "apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates git"
            in cfg.install_cmd
        )
        assert (
            "uv tool install --force --refresh "
            "--overrides /tmp/oh-sdk-overrides.txt "
            "--from "
            "'git+https://github.com/OpenHands/OpenHands-CLI.git@"
            "3ca17446c5d9c1e35e054803478a3501ec251ecf' "
            "openhands --python 3.12" in cfg.install_cmd
        )
        assert "OpenHands/OpenHands-CLI.git@main" not in cfg.install_cmd
        assert "openhands==1.16.0" not in cfg.install_cmd
        assert "command -v git" in cfg.install_cmd
        assert "install.openhands.dev/install.sh" not in cfg.install_cmd

    def test_openhands_install_cmd_overrides_buggy_sdk_pin(self):
        """Guards PR #644 against Opus timeouts from OpenHands SDK 1.21.0."""
        cfg = AGENTS["openhands"]

        assert "openhands-sdk==1.22.1" in cfg.install_cmd
        assert "openhands-tools==1.22.1" in cfg.install_cmd
        assert "openhands-sdk>=1.22.0" not in cfg.install_cmd
        assert "--overrides /tmp/oh-sdk-overrides.txt" in cfg.install_cmd

    def test_openhands_install_cmd_does_not_deploy_bedrock_shim(self):
        """Guards the LiteLLM runtime refactor: Bedrock patches live with LiteLLM."""
        cfg = AGENTS["openhands"]
        assert "oh_bedrock_opus_patch.py" not in cfg.install_cmd
        assert "zz_oh_bedrock_opus_patch.pth" not in cfg.install_cmd

    def test_openhands_install_cmd_does_not_self_test_provider_shim(self):
        """Provider patch self-tests belong to the LiteLLM runtime, not OpenHands."""
        cfg = AGENTS["openhands"]
        assert "_is_adaptive_thinking_model" not in cfg.install_cmd
        assert "us.anthropic.claude-opus-4-8" not in cfg.install_cmd
        assert "shim ACTIVE" not in cfg.install_cmd
        assert "shim NOT active" not in cfg.install_cmd

    def test_openhands_apt_bootstrap_retries_transient_mirror_failures(self):
        """Guards the local fix on v0.5-integration@e55219d against Ubuntu mirror signature flakiness."""
        cfg = AGENTS["openhands"]

        assert "rm -rf /var/lib/apt/lists/*" in cfg.install_cmd
        assert "apt-get clean" in cfg.install_cmd
        assert "Acquire::Retries=3" in cfg.install_cmd
        assert 'while [ "$attempt" -le 3 ]' in cfg.install_cmd
        assert 'case "$attempt"' in cfg.install_cmd

    def test_openhands_no_longer_installs_boto3_for_bedrock_provider(self):
        """LiteLLM owns Bedrock provider dependencies, so OpenHands stays provider-neutral."""
        cfg = AGENTS["openhands"]

        assert "--with 'boto3>=1.40'" not in cfg.install_cmd

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
        assert cfg.session_factory == ""
        assert cfg.disallow_web_tools_setup_cmd == ""
        assert cfg.disallow_web_tools_launch_suffix == ""

    def test_passes_through_new_fields(self, cleanup_agent):
        cleanup_agent.append("rt-full-agent")
        cfg = register_agent(
            name="rt-full-agent",
            install_cmd="install rt",
            launch_cmd="launch rt",
            protocol="session-factory",
            session_factory="my_agent.factory:create_agent",
            default_model="rt-model-1",
            api_protocol="openai-completions",
            disallow_web_tools_setup_cmd="printf 'no web' > /tmp/policy",
            disallow_web_tools_launch_suffix=" --no-web",
        )
        assert cfg.protocol == "session-factory"
        assert cfg.session_factory == "my_agent.factory:create_agent"
        assert cfg.default_model == "rt-model-1"
        assert cfg.api_protocol == "openai-completions"
        assert cfg.disallow_web_tools_setup_cmd == "printf 'no web' > /tmp/policy"
        assert cfg.disallow_web_tools_launch_suffix == " --no-web"

        # And the registered entry reflects them.
        registered = AGENTS["rt-full-agent"]
        assert registered.protocol == "session-factory"
        assert registered.session_factory == "my_agent.factory:create_agent"
        assert registered.default_model == "rt-model-1"
        assert registered.api_protocol == "openai-completions"
        assert registered.disallow_web_tools_launch_suffix == " --no-web"
