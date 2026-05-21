"""Tests for sandbox user config/auth directory derivation from agent registry."""

import pytest

from benchflow.agents.registry import (
    AGENTS,
    AgentConfig,
    HostAuthFile,
    SubscriptionAuth,
    get_sandbox_home_dirs,
)


class TestSandboxDirs:
    def test_skill_only_dirs_not_included(self):
        """Skill-only home dirs should not be copied during sandbox user setup."""
        dirs = get_sandbox_home_dirs()

        assert ".pi" not in dirs
        assert ".agents" not in dirs

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
        assert {".claude", ".codex", ".gemini", ".openclaw"}.issubset(dirs)
        assert ".agents" not in dirs
        assert ".pi" not in dirs

    def test_new_agent_skill_path_not_auto_included(self):
        """Skill-only home dirs should not become sandbox copy targets."""
        AGENTS["_test_agent"] = AgentConfig(
            name="_test_agent",
            install_cmd="true",
            launch_cmd="true",
            skill_paths=["$HOME/.newagent/skills"],
        )
        try:
            dirs = get_sandbox_home_dirs()
            assert ".newagent" not in dirs
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


class TestDockerExecEnvSecrecy:
    """DockerSandbox.exec must not leak env vars via `-e KEY=VALUE` flags.

    `-e` flags are visible in `ps aux` on the host. The verifier's
    [verifier.env] often carries LLM-judge API keys, so exec routes env
    through a sourced container file instead — matching DockerProcess.
    """

    def test_wrap_command_does_not_inline_secret_values(self):
        from benchflow.sandbox.docker import DockerSandbox

        env = {"OPENAI_API_KEY": "sk-secret-value", "FOO": "bar"}
        wrapped = DockerSandbox._wrap_command_with_env_file(env, "run-verifier")

        # The raw secret value must not appear verbatim in the command
        # string (it would otherwise show up in `ps aux`).
        assert "sk-secret-value" not in wrapped
        assert "bar" not in wrapped or "base64" in wrapped
        # The command sources a file and cleans it up.
        assert "base64 -d" in wrapped
        assert "rm -f" in wrapped
        assert wrapped.endswith("run-verifier")
        # Restrictive perms on the env file.
        assert "umask 077" in wrapped
        # Cleanup is via `trap ... EXIT`, so the env file is removed even if
        # the decode/source step fails and short-circuits the `&&` chain.
        assert wrapped.startswith("trap 'rm -f ")
        assert "EXIT" in wrapped

    @pytest.mark.asyncio
    async def test_exec_passes_no_dash_e_flags(self, monkeypatch):
        from unittest.mock import AsyncMock

        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        monkeypatch.setattr(sandbox, "_resolve_user", lambda u: u, raising=False)
        monkeypatch.setattr(sandbox, "_merge_env", lambda e: e or {}, raising=False)

        captured: dict = {}

        async def fake_run(command, check=True, timeout_sec=None):
            captured["command"] = command
            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(
            sandbox, "_run_docker_compose_command", AsyncMock(side_effect=fake_run)
        )

        await sandbox.exec("verify", env={"API_KEY": "sk-leak"})

        cmd = captured["command"]
        # No `-e KEY=VALUE` argument anywhere.
        assert "-e" not in cmd
        for arg in cmd:
            assert "sk-leak" not in arg
