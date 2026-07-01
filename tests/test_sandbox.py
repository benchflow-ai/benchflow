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

    @pytest.mark.asyncio
    async def test_docker_build_retries_transient_apt_signature_errors(
        self, monkeypatch
    ):
        """Guards v0.5-integration@e55219d against apt signature noise."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "apt-flake"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            if len(calls) == 1:
                raise RuntimeError(
                    "Docker compose command failed. Stdout: "
                    "At least one invalid signature was encountered. "
                    "The repository 'http://ports.ubuntu.com noble InRelease' "
                    "is not signed."
                )
            return ExecResult(stdout="", stderr="", return_code=0)

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        await sandbox._run_docker_compose_build()

        assert calls == [["build"], ["build"]]
        assert sleeps == [2.0]

    @pytest.mark.asyncio
    async def test_docker_build_retries_transient_pip_read_timeouts(self, monkeypatch):
        """Guards v0.5-integration@e55219d against pip download noise."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "pip-flake"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            if len(calls) == 1:
                raise RuntimeError(
                    "Docker compose command failed. Stdout: "
                    "pip._vendor.urllib3.exceptions.ReadTimeoutError: "
                    "HTTPSConnectionPool(host='files.pythonhosted.org', "
                    "port=443): Read timed out."
                )
            return ExecResult(stdout="", stderr="", return_code=0)

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        await sandbox._run_docker_compose_build()

        assert calls == [["build"], ["build"]]
        assert sleeps == [2.0]

    @pytest.mark.asyncio
    async def test_docker_build_does_not_retry_non_transient_errors(self, monkeypatch):
        """Guards v0.5-integration@e55219d against broad retry masking."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "real-build-bug"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            raise RuntimeError("Docker compose command failed. Stdout: syntax error")

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(RuntimeError, match="syntax error"):
            await sandbox._run_docker_compose_build()

        assert calls == [["build"]]
        assert sleeps == []

    def test_umask_scoped_to_env_file_write(self):
        """Guards bug I from PR #323: `umask 077` must not leak into the command.

        The restrictive umask that protects the temp env file is wrapped in a
        subshell `(umask 077 && ...)`, so the user's command runs under the
        default umask. Before the fix, `umask 077` ran at the top of the chain
        and persisted, making files the command created mode-0600.
        """
        from benchflow.sandbox.docker import DockerSandbox

        wrapped = DockerSandbox._wrap_command_with_env_file(
            {"FOO": "bar"}, "the-user-command"
        )
        # The umask is scoped to a `( ... )` subshell containing the env-file
        # write — it is not a bare top-level statement that bleeds through.
        assert "(umask 077 &&" in wrapped
        # `umask 077` only ever appears immediately inside the `(` subshell.
        assert wrapped.count("umask 077") == 1
        assert "(umask 077" in wrapped
        # The subshell closes before `set -a`/source/user command run, so the
        # umask does not affect anything after it.
        post_subshell = wrapped.split(")", 1)[1]
        assert "umask" not in post_subshell
        assert post_subshell.lstrip().startswith("&& set -a")
        assert wrapped.endswith("the-user-command")

    def test_non_identifier_env_keys_do_not_break_exec(self):
        """Guards bug H from PR #323: non-identifier env keys must not abort.

        Env keys that are valid process env names but not valid shell
        identifiers (e.g. containing `.` or `-`) cannot be assigned via
        `export NAME=...`. Emitting them would make `. {env_path}` fail and the
        user command would never run, so they are skipped — valid keys still
        flow through and the wrapped command stays intact.
        """
        import base64

        from benchflow.sandbox.docker import DockerSandbox

        env = {
            "VALID_KEY": "keep-me",
            "dotted.key": "drop-me",
            "dashed-key": "drop-me-too",
        }
        wrapped = DockerSandbox._wrap_command_with_env_file(env, "run-it")

        # Decode the base64 env body that gets sourced inside the container.
        token = wrapped.split("printf %s ", 1)[1].split(" |", 1)[0]
        encoded = token.strip("'")
        body = base64.b64decode(encoded).decode()

        # The valid identifier is exported; non-identifier keys are skipped so
        # sourcing the file cannot fail.
        assert "export VALID_KEY=" in body
        assert "dotted.key" not in body
        assert "dashed-key" not in body
        # The user command is still wrapped and reachable.
        assert wrapped.endswith("run-it")


class TestDockerComposeUpNetworkRaceRetry:
    """`compose up` retries the daemon-side network create/attach race.

    On Docker 29.x the daemon can report "Network ... Created" yet fail the
    container create/start that immediately follows with "network
    <project>_default not found". Only that signature retries — every other
    compose up failure must surface unchanged.
    """

    @pytest.mark.asyncio
    async def test_compose_up_retries_network_create_attach_race(
        self, monkeypatch, caplog
    ):
        """A 'network ... not found' failure right after compose up is retried."""
        import logging

        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "network-race"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            if len(calls) == 1:
                raise RuntimeError(
                    "Docker compose command failed. Stdout: "
                    "Network network-race_default Created ... "
                    "Error response from daemon: failed to set up container "
                    "networking: network network-race_default not found"
                )
            return ExecResult(stdout="", stderr="", return_code=0)

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        with caplog.at_level(logging.WARNING, logger="benchflow"):
            await sandbox._run_docker_compose_up()

        assert calls == [["up", "--detach", "--wait"]] * 2
        assert sleeps == [2.0]
        retry_logs = [r for r in caplog.records if "compose up" in r.message]
        assert len(retry_logs) == 1
        assert "network-race" in retry_logs[0].message
        assert "network create/attach race" in retry_logs[0].message

    @pytest.mark.asyncio
    async def test_compose_up_retries_unwrapped_network_not_found(self, monkeypatch):
        """Older daemons emit the race without the container-networking prefix."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "network-race-old"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            if len(calls) == 1:
                raise RuntimeError(
                    "Docker compose command failed. Stdout: "
                    "Error response from daemon: network "
                    "network-race-old_default not found"
                )
            return ExecResult(stdout="", stderr="", return_code=0)

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        await sandbox._run_docker_compose_up()

        assert calls == [["up", "--detach", "--wait"]] * 2
        assert sleeps == [2.0]

    @pytest.mark.asyncio
    async def test_compose_up_does_not_retry_non_race_failures(self, monkeypatch):
        """Only the network-race signature retries — not every compose up error."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "real-up-bug"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            raise RuntimeError(
                "Docker compose command failed. Stdout: "
                "container benchflow-main-1 exited (1) before becoming healthy"
            )

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(RuntimeError, match="before becoming healthy"):
            await sandbox._run_docker_compose_up()

        assert calls == [["up", "--detach", "--wait"]]
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_compose_up_retry_is_bounded(self, monkeypatch):
        """Guards PR #871's compose-up retries from outliving configured backoffs."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._compose import COMPOSE_UP_RETRY_DELAYS_SEC
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_name = "network-race-persistent"
        sandbox.logger = docker_module.logger

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command):
            calls.append(command)
            raise RuntimeError(
                "Docker compose command failed. Stdout: "
                "Error response from daemon: failed to set up container "
                "networking: network network-race-persistent_default not found"
            )

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(sandbox, "_run_docker_compose_command", fake_run)
        monkeypatch.setattr(docker_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(RuntimeError, match="not found"):
            await sandbox._run_docker_compose_up()

        assert calls == [["up", "--detach", "--wait"]] * (
            len(COMPOSE_UP_RETRY_DELAYS_SEC) + 1
        )
        assert sleeps == list(COMPOSE_UP_RETRY_DELAYS_SEC)

    def test_network_race_signature_matching(self):
        """The race matcher stays anchored to the daemon network-not-found error."""
        from benchflow.sandbox.docker import _is_compose_up_network_race_error

        assert _is_compose_up_network_race_error(
            "Error response from daemon: failed to set up container networking: "
            "network proj_default not found"
        )
        assert _is_compose_up_network_race_error(
            "Error response from daemon: network proj_default not found"
        )
        # External-network misconfiguration is a real error, not the race.
        assert not _is_compose_up_network_race_error(
            'network "shared" declared as external, but could not be found'
        )
        # A bare mention of a missing network without the daemon prefix is
        # ambiguous (e.g. user output) and must not trigger retries.
        assert not _is_compose_up_network_race_error("network proj_default not found")
        assert not _is_compose_up_network_race_error(
            "Error response from daemon: No such image: missing:latest"
        )
