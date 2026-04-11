"""Tests for path lockdown — _validate_locked_path, _resolve_locked_paths, _lockdown_paths."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sdk import (
    SDK,
    _resolve_locked_paths,
    _validate_locked_path,
)

# ---------------------------------------------------------------------------
# _validate_locked_path
# ---------------------------------------------------------------------------


class TestValidateLockedPath:
    """Path validation rejects injection and traversal."""

    @pytest.mark.parametrize(
        "p",
        [
            "/solution",
            "/tests",
            "/logs/verifier",
            "/app-foo",
            "/data",
            "/app-*",
        ],
    )
    def test_valid_paths(self, p):
        _validate_locked_path(p)  # should not raise

    @pytest.mark.parametrize(
        "bad,match",
        [
            ("$(rm -rf /)", "must be absolute"),
            ("/solution; rm -rf /", None),
            ("/solution/`whoami`", None),
            ("/solution/../etc", None),
            ("/solution/./foo", "normalizes to"),
            ("//solution//foo", None),
            ("/solution/", None),
            ("solution", "must be absolute"),
            ("/solution | cat /etc/passwd", "must be absolute"),
            ("/", None),
        ],
        ids=[
            "dollar_injection",
            "semicolon",
            "backtick",
            "dotdot_traversal",
            "dot_normpath",
            "double_slash",
            "trailing_slash",
            "relative",
            "pipe",
            "bare_root",
        ],
    )
    def test_rejects_invalid(self, bad, match):
        with pytest.raises(ValueError, match=match):
            _validate_locked_path(bad)


# ---------------------------------------------------------------------------
# _resolve_locked_paths
# ---------------------------------------------------------------------------


class TestResolveLockedPaths:
    """Effective path resolution logic."""

    def test_defaults_with_sandbox_user(self):
        result = _resolve_locked_paths("agent", None)
        assert result == ["/solution", "/tests"]

    def test_union_with_caller_paths(self):
        result = _resolve_locked_paths("agent", ["/data"])
        assert result == ["/solution", "/tests", "/data"]

    def test_dedup_preserves_order(self):
        result = _resolve_locked_paths("agent", ["/solution", "/data"])
        assert result == ["/solution", "/tests", "/data"]

    def test_explicit_opt_out(self):
        result = _resolve_locked_paths("agent", [])
        assert result == []

    def test_no_sandbox_user_returns_empty(self):
        assert _resolve_locked_paths(None, None) == []

    def test_paths_without_sandbox_user_raises(self):
        with pytest.raises(ValueError, match="requires sandbox_user"):
            _resolve_locked_paths(None, ["/solution"])


# ---------------------------------------------------------------------------
# SDK._lockdown_paths
# ---------------------------------------------------------------------------


class TestLockdownPaths:
    """_lockdown_paths sends correct commands to env."""

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr=""))
        return env

    def _get_lockdown_cmd(self, mock_env):
        """Extract the lockdown command (single exec call)."""
        return mock_env.exec.call_args_list[0][0][0]

    def test_noop_empty_paths(self, mock_env):
        asyncio.run(SDK._lockdown_paths(mock_env, []))
        mock_env.exec.assert_not_called()

    def test_chown_before_chmod_and_symlink_skip(self, mock_env):
        asyncio.run(SDK._lockdown_paths(mock_env, ["/solution"]))
        assert mock_env.exec.call_count == 1
        cmd = self._get_lockdown_cmd(mock_env)
        assert cmd.index("chown root:root") < cmd.index("chmod 700")
        assert '[ -L "$d" ]' in cmd

    def test_multiple_paths(self, mock_env):
        asyncio.run(SDK._lockdown_paths(mock_env, ["/solution", "/tests", "/data"]))
        cmd = self._get_lockdown_cmd(mock_env)
        for p in ["/solution", "/tests", "/data"]:
            assert f"for d in {p}" in cmd

    def test_glob_expansion(self, mock_env):
        asyncio.run(SDK._lockdown_paths(mock_env, ["/app-*"]))
        assert "for d in /app-*" in self._get_lockdown_cmd(mock_env)

    def test_validation_rejects_bad_path(self, mock_env):
        with pytest.raises(ValueError):
            asyncio.run(SDK._lockdown_paths(mock_env, ["/solution/../etc"]))
        mock_env.exec.assert_not_called()


# ---------------------------------------------------------------------------
# JobConfig YAML parsing
# ---------------------------------------------------------------------------


class TestJobConfigYAML:
    """sandbox_locked_paths round-trips from YAML."""

    def test_native_yaml_with_locked_paths(self, tmp_path):
        yaml_content = (
            "tasks_dir: tasks\n"
            "sandbox_user: agent\n"
            "sandbox_locked_paths:\n"
            "  - /tasks\n"
            "  - /data\n"
        )
        (tmp_path / "config.yaml").write_text(yaml_content)
        (tmp_path / "tasks").mkdir()

        from benchflow.job import Job

        job = Job.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_locked_paths == ["/tasks", "/data"]
        assert job._config.sandbox_user == "agent"

    def test_native_yaml_without_locked_paths(self, tmp_path):
        yaml_content = "tasks_dir: tasks\nsandbox_user: agent\n"
        (tmp_path / "config.yaml").write_text(yaml_content)
        (tmp_path / "tasks").mkdir()

        from benchflow.job import Job

        job = Job.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_locked_paths is None

    def test_harbor_yaml_with_locked_paths(self, tmp_path):
        yaml_content = (
            "agents:\n"
            "  - name: claude-agent-acp\n"
            "datasets:\n"
            "  - path: tasks\n"
            "sandbox_user: agent\n"
            "sandbox_locked_paths:\n"
            "  - /data\n"
        )
        (tmp_path / "config.yaml").write_text(yaml_content)
        (tmp_path / "tasks").mkdir()

        from benchflow.job import Job

        job = Job.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_locked_paths == ["/data"]
        assert job._config.sandbox_user == "agent"

    def test_native_yaml_sandbox_user_defaults_to_agent(self, tmp_path):
        yaml_content = "tasks_dir: tasks\n"
        (tmp_path / "config.yaml").write_text(yaml_content)
        (tmp_path / "tasks").mkdir()

        from benchflow.job import Job

        job = Job.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_user == "agent"


# ---------------------------------------------------------------------------
# _write_config records locked paths
# ---------------------------------------------------------------------------


class TestWriteConfigRecordsPaths:
    """config.json includes effective locked paths."""

    def test_config_json_includes_locked_paths(self, tmp_path):
        import json

        SDK._write_config(
            tmp_path,
            task_path=tmp_path / "task",
            agent="test",
            model=None,
            environment="docker",
            skills_dir=None,
            sandbox_user="agent",
            context_root=None,
            sandbox_locked_paths=["/solution", "/tests"],
            timeout=300,
            started_at=__import__("datetime").datetime.now(),
            agent_env={},
        )
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["sandbox_locked_paths"] == ["/solution", "/tests"]


# ---------------------------------------------------------------------------
# Sandbox user defaults and warnings
# ---------------------------------------------------------------------------


class TestSandboxUserWarnings:
    """Default sandbox_user and root warnings."""

    def test_default_is_agent(self):
        """Default sandbox_user is 'agent', not None."""
        import inspect

        sig = inspect.signature(SDK.run)
        assert sig.parameters["sandbox_user"].default == "agent"


# ---------------------------------------------------------------------------
# Privilege dropping command construction
# ---------------------------------------------------------------------------


class TestPrivDropCommand:
    """SDK._build_priv_drop_cmd — setpriv/su-l command generation."""

    def test_contains_setpriv_and_su_fallback(self):
        cmd = SDK._build_priv_drop_cmd("my-agent --stdio", "agent")
        assert "setpriv --reuid=agent --regid=agent --init-groups" in cmd
        assert "su -l agent -c" in cmd

    def test_exec_prefix(self):
        """Both branches use exec to replace the shell (no lingering parent)."""
        cmd = SDK._build_priv_drop_cmd("my-agent", "agent")
        assert "exec setpriv" in cmd
        assert "exec su" in cmd

    def test_inner_command_is_shlex_quoted(self):
        import shlex

        cmd = SDK._build_priv_drop_cmd("agent --flag value", "agent")
        inner = "export HOME=/home/agent && cd /home/agent && agent --flag value"
        assert shlex.quote(inner) in cmd

    def test_single_quotes_in_launch(self):
        """Single quotes in agent_launch don't break the command."""
        cmd = SDK._build_priv_drop_cmd("agent --prompt 'hello world'", "agent")
        assert "hello world" in cmd
        assert cmd.count("if ") == 1
        assert cmd.count(" fi") == 1

    def test_custom_sandbox_user(self):
        cmd = SDK._build_priv_drop_cmd("my-agent", "bench-user")
        assert "--reuid=bench-user" in cmd
        assert "su -l bench-user" in cmd
        assert "/home/bench-user" in cmd
