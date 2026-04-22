"""Tests for sandbox user config/auth directory derivation from agent registry."""

from benchflow.agents.registry import (
    AGENTS,
    AgentConfig,
    HostAuthFile,
    SubscriptionAuth,
    get_sandbox_home_dirs,
)


class TestSandboxDirs:
    def test_dirs_derived_from_skill_paths(self):
        """All $HOME skill_paths dirs from AGENTS registry are included."""
        dirs = get_sandbox_home_dirs()
        # claude-agent-acp has $HOME/.claude/skills
        assert ".claude" in dirs
        # gemini has $HOME/.gemini/skills
        assert ".gemini" in dirs
        # pi-acp has $HOME/.pi/agent/skills and $HOME/.agents/skills
        assert ".pi" in dirs
        assert ".agents" in dirs

    def test_credential_file_dirs_included(self):
        """Dirs from credential_files paths are included."""
        dirs = get_sandbox_home_dirs()
        # codex-acp has credential_file at {home}/.codex/auth.json
        assert ".codex" in dirs

    def test_home_dirs_included(self):
        """Explicit home_dirs from AgentConfig are included."""
        dirs = get_sandbox_home_dirs()
        # openclaw has home_dirs=[".openclaw"]
        assert ".openclaw" in dirs

    def test_does_not_include_legacy_local_tool_dir(self):
        """.local is not included unless an agent registry path derives it."""
        dirs = get_sandbox_home_dirs()
        assert ".local" not in dirs

    def test_only_includes_top_level_home_dirs(self):
        """Derived entries stay at $HOME top-level, not nested tool subpaths."""
        dirs = get_sandbox_home_dirs()
        assert ".local/bin" not in dirs

    def test_dirs_represent_registry_backed_home_config_or_auth(self):
        """Returned dirs are registry-derived user home config/auth roots."""
        dirs = get_sandbox_home_dirs()
        assert dirs == {".agents", ".claude", ".codex", ".gemini", ".openclaw", ".pi"}

    def test_new_agent_auto_included(self):
        """Adding an agent with skill_paths=$HOME/.newagent/skills includes .newagent."""
        AGENTS["_test_agent"] = AgentConfig(
            name="_test_agent",
            install_cmd="true",
            launch_cmd="true",
            skill_paths=["$HOME/.newagent/skills"],
        )
        try:
            dirs = get_sandbox_home_dirs()
            assert ".newagent" in dirs
        finally:
            del AGENTS["_test_agent"]

    def test_subscription_auth_file_dirs_included(self):
        """Dirs from subscription_auth.files container paths are included."""
        AGENTS["_test_agent_subscription_auth"] = AgentConfig(
            name="_test_agent_subscription_auth",
            install_cmd="true",
            launch_cmd="true",
            subscription_auth=SubscriptionAuth(
                replaces_env="TEST_API_KEY",
                detect_file="~/.subauth/login.json",
                files=[
                    HostAuthFile(
                        "~/.subauth/login.json",
                        "{home}/.subauth/login.json",
                    )
                ],
            ),
        )
        try:
            dirs = get_sandbox_home_dirs()
            assert ".subauth" in dirs
        finally:
            del AGENTS["_test_agent_subscription_auth"]

    def test_workspace_paths_excluded(self):
        """$WORKSPACE paths are not included (only $HOME paths)."""
        dirs = get_sandbox_home_dirs()
        # openclaw has $WORKSPACE/skills — should NOT produce a dir entry
        assert "skills" not in dirs

    def test_returns_set_of_strings(self):
        """Return type is a set of strings."""
        dirs = get_sandbox_home_dirs()
        assert isinstance(dirs, set)
        assert all(isinstance(d, str) for d in dirs)
        assert all(d.startswith(".") for d in dirs)
