"""Tests for sandbox user config/auth directory derivation from agent registry."""

import os
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import (
    AGENTS,
    AgentConfig,
    HostAuthFile,
    SubscriptionAuth,
    get_sandbox_home_dirs,
)
from benchflow.task.config import SandboxConfig
from tests.reachable_fixture import directory


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
        assert {".claude", ".codex", ".gemini"}.issubset(dirs)
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

    def test_named_agents_get_only_their_own_dirs(self):
        """Guards the fix from PR #1163 against copying every agent's home dirs
        into one sandbox.

        That union dates back to the hard-coded list in 28d9acdf and became
        registry-derived in 39d9b829. A sandbox running some agents must not
        be handed the config/auth dirs of every other registered agent.
        """
        AGENTS["_test_agent_home_dirs"] = AgentConfig(
            name="_test_agent_home_dirs",
            install_cmd="true",
            launch_cmd="true",
            home_dirs=[".testagent", ".config/testagent"],
        )
        try:
            assert get_sandbox_home_dirs(["codex-acp"]) == {".codex"}
            assert get_sandbox_home_dirs(["codex-acp", "_test_agent_home_dirs"]) == {
                ".codex",
                ".testagent",
                ".config/testagent",
            }
            assert get_sandbox_home_dirs(["oracle", "not-registered"]) == set()
        finally:
            del AGENTS["_test_agent_home_dirs"]

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
        assert "skills" not in dirs

    def test_returns_set_of_strings(self):
        """Return type is a set of strings."""
        dirs = get_sandbox_home_dirs()
        assert isinstance(dirs, set)
        assert all(isinstance(d, str) for d in dirs)
        assert all(d.startswith(".") for d in dirs)


class TestDockerVerifierLogMountProbe:
    @pytest.mark.asyncio
    async def test_untranslated_bind_mount_uses_copy_out(self, tmp_path):
        """Guards PR #1040 against raw Docker sockets losing WSL path translation."""
        from unittest.mock import AsyncMock

        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.rollout_paths = SimpleNamespace(
            verifier_dir=directory(tmp_path / "verifier")
        )
        sandbox._logs_are_mounted = True
        sandbox.logger = docker_module.logger
        sandbox.exec = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", return_code=1)
        )

        await sandbox._probe_verifier_log_mount()

        assert sandbox.is_mounted is False
        assert not list(sandbox.rollout_paths.verifier_dir.iterdir())
        command = sandbox.exec.await_args.args[0]
        assert command.startswith("test -f /logs/verifier/.runtime-mount-probe-")

    @pytest.mark.asyncio
    async def test_visible_bind_mount_keeps_mounted_fast_path(self, tmp_path):
        """Guards PR #1040 by preserving Docker's normal mounted fast path."""
        from unittest.mock import AsyncMock

        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.rollout_paths = SimpleNamespace(
            verifier_dir=directory(tmp_path / "verifier")
        )
        sandbox._logs_are_mounted = True
        sandbox.logger = docker_module.logger
        sandbox.exec = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", return_code=0)
        )

        await sandbox._probe_verifier_log_mount()

        assert sandbox.is_mounted is True
        assert not list(sandbox.rollout_paths.verifier_dir.iterdir())

    @staticmethod
    def _unmounted_sandbox(verifier_dir):
        from unittest.mock import AsyncMock

        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.rollout_paths = SimpleNamespace(verifier_dir=directory(verifier_dir))
        sandbox._logs_are_mounted = True
        sandbox.logger = docker_module.logger
        sandbox.exec = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", return_code=1)
        )
        return sandbox

    @pytest.mark.asyncio
    async def test_colima_gap_outside_home_is_debug_only(
        self, tmp_path, monkeypatch, caplog
    ):
        """Regression test: on a Mac,
        Colima shares only $HOME with its VM, so a rollout directory elsewhere
        (e.g. /tmp) is never visible in the container. That is the expected
        case, the verifier copies its outputs back, and it must not print a
        warning on every run."""
        import logging

        from benchflow.sandbox import docker as docker_module

        monkeypatch.setattr(docker_module, "_DOCKER_RUNS_IN_VM", True)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        sandbox = self._unmounted_sandbox(tmp_path / "jobs" / "verifier")

        with caplog.at_level(logging.DEBUG, logger="benchflow"):
            await sandbox._probe_verifier_log_mount()

        assert sandbox.is_mounted is False
        mount_logs = [r for r in caplog.records if "copied back" in r.message]
        assert [r.levelno for r in mount_logs] == [logging.DEBUG]
        assert "$HOME" in mount_logs[0].message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("in_vm", [True, False])
    async def test_unexpected_gap_stays_a_warning(
        self, tmp_path, monkeypatch, caplog, in_vm
    ):
        """Regression test: the gap stays loud where the daemon
        should see the path (under $HOME on a Mac, or any path on a host
        without a Docker VM), since it then means a remote daemon or a
        socket that bypasses path translation (PR #1040)."""
        import logging

        from benchflow.sandbox import docker as docker_module

        home = tmp_path / "home"
        monkeypatch.setattr(docker_module, "_DOCKER_RUNS_IN_VM", in_vm)
        monkeypatch.setenv("HOME", str(home))
        verifier_dir = (home if in_vm else tmp_path) / "jobs" / "verifier"
        sandbox = self._unmounted_sandbox(verifier_dir)

        with caplog.at_level(logging.DEBUG, logger="benchflow"):
            await sandbox._probe_verifier_log_mount()

        assert sandbox.is_mounted is False
        mount_logs = [r for r in caplog.records if "copied back" in r.message]
        assert [r.levelno for r in mount_logs] == [logging.WARNING]
        assert "bind mount is not visible" in mount_logs[0].message
        assert "DOCKER_HOST" in mount_logs[0].message


class TestDockerExecEnvSecrecy:
    """DockerSandbox.exec must not leak env vars via `-e KEY=VALUE` flags.

    `-e` flags are visible in `ps aux` on the host. The verifier's
    [verifier.env] often carries LLM-judge API keys, so exec routes env
    through a sourced container file instead — matching DockerProcess.
    """

    @pytest.mark.asyncio
    async def test_wrap_command_does_not_inline_secret_values(self, monkeypatch):
        from benchflow.sandbox._base import ENV_FROM_STDIN, ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._persistent_env = {}
        sandbox.default_user = None
        sandbox._main_container = "main-container"
        seen: dict = {}

        async def run(argv, check=False, timeout_sec=None, **kwargs):
            seen["argv"], seen["stdin"] = argv, kwargs.get("stdin_data")
            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(sandbox, "_run_docker_container_command", run)
        env = {"OPENAI_API_KEY": "sk-secret-value", "FOO": "bar"}
        await sandbox.exec("run-verifier", env=env)

        # The raw secret value must not appear in any argument (it would
        # otherwise show up in `ps aux`, on the host or in the container): the
        # env is the exec's stdin, read by the shell's own `.` (b156).
        argv = seen["argv"]
        assert "sk-secret-value" not in " ".join(argv) and "bar" not in argv[-1]
        assert argv[:2] == ["exec", "-i"]
        # PT's route may pin root's PATH first (C8); the stdin wrapper and the
        # command come last.
        assert argv[-1].endswith(ENV_FROM_STDIN + "run-verifier")
        assert "base64" not in argv[-1] and "/tmp/" not in argv[-1]
        assert "rm " not in argv[-1] and "trap" not in argv[-1]
        assert b"export OPENAI_API_KEY=sk-secret-value\n" in seen["stdin"]

    @pytest.mark.asyncio
    async def test_exec_passes_no_dash_e_flags(self, monkeypatch):
        from unittest.mock import AsyncMock

        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        monkeypatch.setattr(sandbox, "_resolve_user", lambda u: u, raising=False)
        monkeypatch.setattr(
            sandbox,
            "_merge_env",
            lambda e, user=None, plumbing=False, explicit_root=None: e or {},
            raising=False,
        )

        captured: dict = {}

        async def fake_run(command, check=True, timeout_sec=None, stdin_data=None):
            captured["command"] = command
            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(
            sandbox, "_run_docker_compose_command", AsyncMock(side_effect=fake_run)
        )

        await sandbox.exec("verify", env={"API_KEY": "sk-leak"})

        cmd = captured["command"]
        # No `-e KEY=VALUE` argument anywhere: only the empty loader values
        # of root's plumbing (C8).
        assert all(cmd[i + 1].endswith("=") for i, a in enumerate(cmd) if a == "-e")
        for arg in cmd:
            assert "sk-leak" not in arg

    @pytest.mark.asyncio
    async def test_pre_compose_hook_runs_with_compose_env(self, tmp_path, monkeypatch):
        """Task environments can prepare local Docker state before compose
        starts, once the operator lets the task run code on this machine."""
        from benchflow.sandbox.docker import DockerSandbox

        monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_dir = tmp_path
        sandbox.environment_name = "pre-compose"
        sandbox.task_env_config = SimpleNamespace(build_timeout_sec=60)
        sandbox._env_vars = SimpleNamespace(
            to_env_dict=lambda include_os_env=True: {
                "PATH": os.environ["PATH"],
                "BASE": "1",
            }
        )
        sandbox._compose_task_env = {"TASK": "2"}
        sandbox._persistent_env = {"PERSIST": "3"}

        hook = tmp_path / "benchflow-pre-compose.sh"
        hook.write_text(
            'printf \'%s:%s:%s:%s\\n\' "$PWD" "$BASE" "$TASK" "$PERSIST" > hook.out\n'
        )

        await sandbox._run_pre_compose_hook()

        assert (tmp_path / "hook.out").read_text() == f"{tmp_path}:1:2:3\n"

    @pytest.mark.asyncio
    async def test_pre_compose_hook_failure_is_reported(self, tmp_path, monkeypatch):
        from benchflow.sandbox.docker import DockerSandbox

        monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.environment_dir = tmp_path
        sandbox.environment_name = "bad-pre-compose"
        sandbox.task_env_config = SimpleNamespace(build_timeout_sec=60)
        sandbox._env_vars = SimpleNamespace(
            to_env_dict=lambda include_os_env=True: {"PATH": os.environ["PATH"]}
        )
        sandbox._compose_task_env = {}
        sandbox._persistent_env = {}
        (tmp_path / "benchflow-pre-compose.sh").write_text("echo hook-broke\nexit 42\n")

        with pytest.raises(RuntimeError, match="hook-broke"):
            await sandbox._run_pre_compose_hook()

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
        sandbox.task_env_config = SandboxConfig()

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command, timeout_sec=None):
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
        sandbox.task_env_config = SandboxConfig()

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command, timeout_sec=None):
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
        sandbox.task_env_config = SandboxConfig()

        calls: list[list[str]] = []
        sleeps: list[float] = []

        async def fake_run(command, timeout_sec=None):
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
        """Guards bug I from PR #323: no wrapper umask reaches the command.

        The wrapper once set ``umask 077`` for its temp env file; leaked, it
        made files the command created mode-0600. It writes no file now.
        """
        from benchflow.sandbox._base import wrap_command_with_env_stdin

        wrapped = wrap_command_with_env_stdin("the-user-command")
        # No env file is written any more (b143), so no umask is set at all:
        # the command runs under the caller's umask.
        assert "umask" not in wrapped
        assert wrapped.endswith("the-user-command")

    def test_non_identifier_env_keys_do_not_break_exec(self):
        """Guards bug H from PR #323: non-identifier env keys must not abort.

        Env keys that are valid process env names but not valid shell
        identifiers (e.g. containing `.` or `-`) cannot be assigned via
        `export NAME=...`. Emitting them would make `. {env_path}` fail and the
        user command would never run, so they are skipped — valid keys still
        flow through and the wrapped command stays intact.
        """

        from benchflow.sandbox._base import env_stdin_body

        env = {
            "VALID_KEY": "keep-me",
            "dotted.key": "drop-me",
            "dashed-key": "drop-me-too",
        }
        # The env body the container's own shell sources from its stdin.
        body = env_stdin_body(env).decode()
        wrapped = "run-it"

        # The valid identifier is exported; non-identifier keys are skipped so
        # sourcing the file cannot fail.
        assert "export VALID_KEY=" in body
        assert "dotted.key" not in body
        assert "dashed-key" not in body
        # The user command is still wrapped and reachable.
        assert wrapped.endswith("run-it")


class TestDockerComposeUpNetworkRaceRetry:
    """`compose up --wait` retries the daemon-side network create/attach race.

    On Docker 29.x the daemon can report "Network ... Created" yet fail the
    container create/start that immediately follows with "network
    <project>_default not found". Only that signature retries — every other
    compose up failure must surface unchanged.

    The sandboxes here have a task compose file, so start() goes straight to
    ``up --detach --wait`` (a Dockerfile-only sandbox first tries
    ``up --detach`` without the wait: tests/test_docker_direct_exec.py).
    """

    @staticmethod
    def _with_task_compose_file(sandbox, tmp_path):
        (tmp_path / "docker-compose.yaml").write_text("services: {}\n")
        sandbox.environment_dir = tmp_path
        return sandbox

    @pytest.mark.asyncio
    async def test_compose_up_retries_network_create_attach_race(
        self, monkeypatch, caplog, tmp_path
    ):
        """A 'network ... not found' failure right after compose up is retried."""
        import logging

        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = self._with_task_compose_file(
            DockerSandbox.__new__(DockerSandbox), tmp_path
        )
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

        assert calls == [["up", "--detach", "--wait", "--no-build"]] * 2
        assert sleeps == [2.0]
        retry_logs = [r for r in caplog.records if "compose up" in r.message]
        assert len(retry_logs) == 1
        assert "network-race" in retry_logs[0].message
        assert "network create/attach race" in retry_logs[0].message

    @pytest.mark.asyncio
    async def test_compose_up_retries_unwrapped_network_not_found(
        self, monkeypatch, tmp_path
    ):
        """Older daemons emit the race without the container-networking prefix."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = self._with_task_compose_file(
            DockerSandbox.__new__(DockerSandbox), tmp_path
        )
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

        assert calls == [["up", "--detach", "--wait", "--no-build"]] * 2
        assert sleeps == [2.0]

    @pytest.mark.asyncio
    async def test_compose_up_does_not_retry_non_race_failures(
        self, monkeypatch, tmp_path
    ):
        """Only the network-race signature retries — not every compose up error."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = self._with_task_compose_file(
            DockerSandbox.__new__(DockerSandbox), tmp_path
        )
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

        assert calls == [["up", "--detach", "--wait", "--no-build"]]
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_compose_up_retry_is_bounded(self, monkeypatch, tmp_path):
        """A persistent network-race signature re-raises after exactly the configured backoffs (bounded, not infinite)."""
        from benchflow.sandbox import docker as docker_module
        from benchflow.sandbox._compose import COMPOSE_UP_RETRY_DELAYS_SEC
        from benchflow.sandbox.docker import DockerSandbox

        sandbox = self._with_task_compose_file(
            DockerSandbox.__new__(DockerSandbox), tmp_path
        )
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

        assert calls == [["up", "--detach", "--wait", "--no-build"]] * (
            len(COMPOSE_UP_RETRY_DELAYS_SEC) + 1
        )
        assert sleeps == list(COMPOSE_UP_RETRY_DELAYS_SEC)
        # Concrete anti-revert guard: the assertions above derive from the constant
        # and would move with it, so a silent truncation back to the old two short
        # delays (2.0, 5.0) — the regression this bound exists to prevent — must
        # trip here independently.
        assert COMPOSE_UP_RETRY_DELAYS_SEC[-1] >= 10.0
        assert len(COMPOSE_UP_RETRY_DELAYS_SEC) >= 4

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
