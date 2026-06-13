"""Focused contract tests for setup_sandbox_user() shell command generation."""

import re
import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.registry import get_sandbox_home_dirs
from benchflow.sandbox.lockdown import setup_sandbox_user


async def _run_setup_sandbox_user(
    *, sandbox_user: str = "agent", workspace: str = "/app"
):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))

    await setup_sandbox_user(env, sandbox_user, workspace)

    env.exec.assert_awaited_once()
    return env.exec.call_args.args[0], env.exec.call_args.kwargs


def _assert_conditional_legacy_symlink(cmd: str, *, source: str, dest: str) -> None:
    """Legacy tool dirs should link only when a root-only install exists."""
    assert re.search(
        rf"if \[ -e [\"']?{re.escape(source)}[\"']? \].*ln -s(?:f|[a-zA-Z-])* [\"']?{re.escape(source)}[\"']? [\"']?{re.escape(dest)}[\"']?.*fi",
        cmd,
    ), f"expected explicit symlink from {source} to {dest} in setup command: {cmd}"


def _get_copy_loop_dirs(cmd: str) -> list[str]:
    """Extract the general home-dir copy loop payload from the shell command."""
    match = re.search(r"for d in (?P<dirs>.*?); do", cmd)
    assert match, f"expected general home-dir copy loop in setup command: {cmd}"
    return match.group("dirs").split()


class TestSetupSandboxUser:
    @pytest.mark.asyncio
    async def test_setup_command_avoids_recursive_root_tool_copies(self):
        """Heavy root-owned tool dirs should no longer be recursively copied."""
        cmd, kwargs = await _run_setup_sandbox_user()

        assert "cp -aL /root/.local/bin/." not in cmd
        assert "cp -a /root/.nvm/." not in cmd
        assert kwargs["timeout_sec"] == 120
        assert kwargs["user"] == "root"

    @pytest.mark.asyncio
    async def test_setup_command_still_creates_user_prepares_home_and_chowns_workspace(
        self,
    ):
        """The non-copy setup contract still creates the user and grants access."""
        cmd, _ = await _run_setup_sandbox_user()

        assert "id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent" in cmd
        assert "mkdir -p /home/agent/.local/bin" not in cmd
        assert "if [ -f /home/cua/.Xauthority ]; then" in cmd
        assert "cp /home/cua/.Xauthority /home/agent/.Xauthority" in cmd
        assert "chown -R agent:agent /home/agent" in cmd
        assert f"chown -R agent:agent {shlex.quote('/app')}" in cmd

    @pytest.mark.asyncio
    async def test_setup_command_keeps_heavy_root_tool_dirs_on_shared_paths(self):
        """Legacy root-only tool dirs should use conditional symlinks, not duplication."""
        cmd, _ = await _run_setup_sandbox_user()

        _assert_conditional_legacy_symlink(
            cmd,
            source="/root/.local/bin",
            dest="/home/agent/.local/bin",
        )
        _assert_conditional_legacy_symlink(
            cmd, source="/root/.nvm", dest="/home/agent/.nvm"
        )
        assert "cp -aL /root/.local/bin/. /home/agent/.local/bin/" not in cmd
        assert "cp -a /root/.nvm/. /home/agent/.nvm/" not in cmd

    @pytest.mark.asyncio
    async def test_setup_command_copy_loop_excludes_local_dir(self):
        """General home-dir copying should narrow to small config/auth dirs only."""
        cmd, _ = await _run_setup_sandbox_user()

        copy_loop_dirs = _get_copy_loop_dirs(cmd)

        assert copy_loop_dirs == sorted(
            d for d in get_sandbox_home_dirs() if d != ".local"
        )
        assert ".local" not in copy_loop_dirs
        assert "mkdir -p /home/agent/$d" in cmd
        assert "cp -a /root/$d/. /home/agent/$d/ 2>/dev/null || true" in cmd

    @pytest.mark.asyncio
    async def test_setup_command_does_not_copy_heavy_tool_trees_into_home(self):
        """BenchFlow-installed agents must not rely on sandbox-home tool copies.

        Agent binaries are placed in /usr/local/bin by the registered install_cmd
        values, so setup_sandbox_user() must not bulk-copy heavyweight tool trees
        (e.g. /root/.nvm, /root/.local/bin) to make them executable for the user.
        """
        cmd, _ = await _run_setup_sandbox_user()

        for heavy_source in ("/root/.nvm", "/root/.local/bin"):
            assert f"cp -a {heavy_source}/." not in cmd
            assert f"cp -aL {heavy_source}/." not in cmd

        copy_loop_dirs = _get_copy_loop_dirs(cmd)
        for heavy_dir in (".nvm", ".local"):
            assert heavy_dir not in copy_loop_dirs, (
                f"sandbox copy loop must not include heavy tool dir {heavy_dir!r}"
            )

    @pytest.mark.asyncio
    async def test_setup_raises_when_user_creation_command_fails(self):
        env = MagicMock()
        env.exec = AsyncMock(
            return_value=MagicMock(
                return_code=1,
                stdout="",
                stderr="useradd: failure",
            )
        )

        with pytest.raises(RuntimeError, match="Sandbox user setup failed"):
            await setup_sandbox_user(env, "agent", "/app")


class TestStageDockerfileDeps:
    """Tests for stage_dockerfile_deps COPY parsing (multi-source / JSON form)."""

    @staticmethod
    def _setup(tmp_path, dockerfile_body):
        """Create a task dir + context root, return (task_path, context_root)."""
        from pathlib import Path

        context_root = tmp_path / "repo"
        context_root.mkdir()
        task_path = context_root / "tasks" / "demo"
        env_dir = task_path / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text(dockerfile_body)
        return Path(task_path), Path(context_root)

    def test_multi_source_copy_stages_every_source(self, tmp_path):
        """COPY with multiple sources stages all of them into _deps/."""
        from benchflow.sandbox.setup import stage_dockerfile_deps

        task_path, context_root = self._setup(
            tmp_path,
            "FROM python:3.12\nCOPY pkg/a pkg/b pkg/c /app/\n",
        )
        for name in ("a", "b", "c"):
            (context_root / "pkg").mkdir(exist_ok=True)
            (context_root / "pkg" / name).write_text(f"file {name}\n")

        stage_dockerfile_deps(task_path, context_root)

        deps = task_path / "environment" / "_deps"
        for name in ("a", "b", "c"):
            assert (deps / name).exists(), f"source {name} not staged"

        rewritten = (task_path / "environment" / "Dockerfile").read_text()
        assert "_deps/a _deps/b _deps/c /app/" in rewritten

    def test_json_exec_form_copy_is_staged(self, tmp_path):
        """COPY in JSON/exec form stages its source and is rewritten."""
        from benchflow.sandbox.setup import stage_dockerfile_deps

        task_path, context_root = self._setup(
            tmp_path,
            'FROM python:3.12\nCOPY ["pkg/runtime", "/app/runtime"]\n',
        )
        (context_root / "pkg").mkdir()
        (context_root / "pkg" / "runtime").write_text("runtime\n")

        stage_dockerfile_deps(task_path, context_root)

        assert (task_path / "environment" / "_deps" / "runtime").exists()
        rewritten = (task_path / "environment" / "Dockerfile").read_text()
        assert "_deps/runtime" in rewritten
        assert rewritten.count("[") == 1  # still JSON form

    def test_single_source_copy_still_works(self, tmp_path):
        """The original two-arg COPY case is unchanged."""
        from benchflow.sandbox.setup import stage_dockerfile_deps

        task_path, context_root = self._setup(
            tmp_path,
            "FROM python:3.12\nCOPY pkg/only /app/\n",
        )
        (context_root / "pkg").mkdir()
        (context_root / "pkg" / "only").write_text("only\n")

        stage_dockerfile_deps(task_path, context_root)

        assert (task_path / "environment" / "_deps" / "only").exists()
        rewritten = (task_path / "environment" / "Dockerfile").read_text()
        assert "_deps/only /app/" in rewritten

    def test_malformed_json_copy_warns_and_leaves_unchanged(self, tmp_path, caplog):
        """An unparseable JSON-form COPY emits a warning, not silent skip."""
        import logging

        from benchflow.sandbox.setup import stage_dockerfile_deps

        body = 'FROM python:3.12\nCOPY ["pkg/lib", broken\n'
        task_path, context_root = self._setup(tmp_path, body)

        with caplog.at_level(logging.WARNING):
            stage_dockerfile_deps(task_path, context_root)

        assert any("COPY" in r.message for r in caplog.records)
        # Line left untouched so the build error is at least visible.
        assert "broken" in (task_path / "environment" / "Dockerfile").read_text()
