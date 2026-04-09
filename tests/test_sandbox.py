"""Tests for sandbox user directory derivation from agent registry."""

from benchflow.agents.registry import (
    AGENTS,
    AgentConfig,
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

    def test_always_includes_local(self):
        """.local is always in the dir list."""
        dirs = get_sandbox_home_dirs()
        assert ".local" in dirs

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

    def test_no_duplicates_across_agents(self):
        """Multiple agents sharing a dir (e.g. .claude) don't cause issues."""
        # claude-agent-acp and openclaw both use $HOME/.claude/skills
        dirs = get_sandbox_home_dirs()
        assert ".claude" in dirs
        # set ensures no duplicates by nature; just verify count
        dir_list = list(dirs)
        assert len(dir_list) == len(set(dir_list))
