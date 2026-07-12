"""Tests for path lockdown — _validate_locked_path, _resolve_locked_paths, lockdown_paths."""

import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.lockdown import (
    _DEFAULT_LOCKED,
    _resolve_locked_paths,
    _seed_verifier_workspace,
    _validate_locked_path,
    build_priv_drop_cmd,
    lockdown_paths,
)
from benchflow.sdk import SDK
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy

# _validate_locked_path


class TestValidateLockedPath:
    """Path validation rejects injection and traversal."""

    @pytest.mark.parametrize(
        "p",
        [
            "/oracle",
            "/solution",
            "/verifier",
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


# _resolve_locked_paths


class TestResolveLockedPaths:
    """Effective path resolution logic."""

    def test_defaults_with_sandbox_user(self):
        result = _resolve_locked_paths("agent", None)
        assert result == [
            "/oracle",
            "/solution",
            "/verifier",
            "/tests",
            "/testbed_verify",
        ]

    def test_union_with_caller_paths(self):
        result = _resolve_locked_paths("agent", ["/data"])
        assert result == [
            "/oracle",
            "/solution",
            "/verifier",
            "/tests",
            "/testbed_verify",
            "/data",
        ]

    def test_dedup_preserves_order(self):
        result = _resolve_locked_paths("agent", ["/solution", "/data"])
        assert result == [
            "/oracle",
            "/solution",
            "/verifier",
            "/tests",
            "/testbed_verify",
            "/data",
        ]

    def test_explicit_opt_out(self):
        result = _resolve_locked_paths("agent", [])
        assert result == []

    def test_no_sandbox_user_returns_empty(self):
        assert _resolve_locked_paths(None, None) == []

    def test_paths_without_sandbox_user_raises(self):
        with pytest.raises(ValueError, match="requires sandbox_user"):
            _resolve_locked_paths(None, ["/solution"])


# /testbed_verify grading-side isolation (verifier-home isolation)


class TestTestbedVerifyLockdown:
    """The root-owned verifier snapshot /testbed_verify must be locked from the agent.

    _seed_verifier_workspace makes /testbed_verify world-readable (chmod -R o+rX)
    so the root verifier can diff against it. Without lockdown the agent can read
    grading-side state there (judge config, rubrics, expected outputs, judge
    credentials) and forge a reward. These tests pin the path into the locked set
    and prove the resulting filesystem mode denies non-owner reads.
    """

    def test_testbed_verify_in_default_locked(self):
        # Mutation-killer: dropping /testbed_verify from _DEFAULT_LOCKED fails here.
        assert "/testbed_verify" in _DEFAULT_LOCKED

    def test_testbed_verify_in_resolved_defaults(self):
        # Mutation-killer at the resolution boundary the rollout actually calls.
        assert "/testbed_verify" in _resolve_locked_paths("agent", None)
        # Even when a caller passes custom paths, the grading snapshot stays locked.
        assert "/testbed_verify" in _resolve_locked_paths("agent", ["/data"])

    def test_validates_as_lockable_path(self):
        _validate_locked_path("/testbed_verify")  # underscore path must be accepted

    async def test_lockdown_chowns_root_and_chmods_700(self):
        """The emitted lockdown command for /testbed_verify denies group/other
        access (chown root:root + chmod 700), the mechanism that blocks reads."""
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr=""))
        await lockdown_paths(env, ["/testbed_verify"])
        cmd = env.exec.call_args_list[0][0][0]
        assert "for d in /testbed_verify" in cmd
        assert cmd.index("chown root:root") < cmd.index("chmod 700")

    async def test_seeded_world_readable_then_lockdown_blocks_other_read(
        self, tmp_path
    ):
        """Hermetic, sandbox-level behavioral check against a real filesystem.

        Reproduce the two-step container sequence on a temp dir: seed makes the
        snapshot world-readable (chmod -R o+rX), then lockdown's per-path body
        (chmod 700) strips it. After lockdown the directory grants no read/exec
        bits to group or other, so a non-root agent user cannot traverse in to
        read the judge key. chown root:root is omitted here (the test is not root,
        and root would bypass the mode anyway) — the read block is the mode.
        """
        import os
        import stat
        import subprocess

        snapshot = tmp_path / "testbed_verify"
        snapshot.mkdir()
        judge_key = snapshot / "judge_config.json"
        judge_key.write_text('{"GEMINI_API_KEY": "sk-secret-judge-key"}')

        # Step 1: mirror _seed_verifier_workspace making the snapshot world-readable.
        subprocess.run(["chmod", "-R", "o+rX", str(snapshot)], check=True)
        mode_after_seed = stat.S_IMODE(os.stat(snapshot).st_mode)
        assert mode_after_seed & stat.S_IROTH, "seed should leave the snapshot o+r"

        # Step 2: run the exact per-path lockdown body the production cmd emits.
        body = (
            f"for d in {snapshot}; do "
            f'  [ -L "$d" ] && continue; '
            f'  [ -e "$d" ] || continue; '
            f'  chmod 700 "$d"; '
            f"done"
        )
        subprocess.run(["sh", "-c", body], check=True)

        mode = stat.S_IMODE(os.stat(snapshot).st_mode)
        denied = stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
        denied |= stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH
        assert mode & denied == 0, (
            f"after lockdown the snapshot must grant no group/other bits, got {oct(mode)}"
        )
        # No other-execute bit means a non-owner cannot traverse in to the key.
        assert not (mode & stat.S_IXOTH), (
            "agent (other) must not traverse /testbed_verify"
        )


class TestSeedLockdownOrdering:
    """Seeding world-readable then locking must compose: lockdown wins."""

    async def test_seed_marks_world_readable_and_default_lock_targets_it(self):
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        await _seed_verifier_workspace(env, workspace="/testbed")
        seed_cmds = [c.args[0] for c in env.exec.call_args_list]
        assert any("chmod -R o+rX /testbed_verify" in c for c in seed_cmds), (
            "seed must make /testbed_verify world-readable (the leak lockdown closes)"
        )
        # The very path seed exposes is in the set lockdown subsequently locks.
        assert "/testbed_verify" in _DEFAULT_LOCKED


# lockdown_paths


class TestLockdownPaths:
    """lockdown_paths sends correct commands to env."""

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr=""))
        return env

    def _get_lockdown_cmd(self, mock_env):
        """Extract the lockdown command (single exec call)."""
        return mock_env.exec.call_args_list[0][0][0]

    async def test_noop_empty_paths(self, mock_env):
        await lockdown_paths(mock_env, [])
        mock_env.exec.assert_not_called()

    async def test_chown_before_chmod_and_symlink_skip(self, mock_env):
        await lockdown_paths(mock_env, ["/solution"])
        assert mock_env.exec.call_count == 1
        cmd = self._get_lockdown_cmd(mock_env)
        assert cmd.index("chown root:root") < cmd.index("chmod 700")
        assert '[ -L "$d" ]' in cmd

    async def test_multiple_paths(self, mock_env):
        await lockdown_paths(mock_env, ["/solution", "/tests", "/data"])
        cmd = self._get_lockdown_cmd(mock_env)
        for p in ["/solution", "/tests", "/data"]:
            assert f"for d in {p}" in cmd

    async def test_glob_expansion(self, mock_env):
        await lockdown_paths(mock_env, ["/app-*"])
        assert "for d in /app-*" in self._get_lockdown_cmd(mock_env)

    async def test_validation_rejects_bad_path(self, mock_env):
        with pytest.raises(ValueError):
            await lockdown_paths(mock_env, ["/solution/../etc"])
        mock_env.exec.assert_not_called()


# EvaluationConfig YAML parsing


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

        from benchflow.evaluation import Evaluation

        job = Evaluation.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_locked_paths == ["/tasks", "/data"]
        assert job._config.sandbox_user == "agent"

    def test_native_yaml_without_locked_paths(self, tmp_path):
        yaml_content = "tasks_dir: tasks\nsandbox_user: agent\n"
        (tmp_path / "config.yaml").write_text(yaml_content)
        (tmp_path / "tasks").mkdir()

        from benchflow.evaluation import Evaluation

        job = Evaluation.from_yaml(tmp_path / "config.yaml")
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

        from benchflow.evaluation import Evaluation

        job = Evaluation.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_locked_paths == ["/data"]
        assert job._config.sandbox_user == "agent"

    def test_native_yaml_sandbox_user_defaults_to_agent(self, tmp_path):
        yaml_content = "tasks_dir: tasks\n"
        (tmp_path / "config.yaml").write_text(yaml_content)
        (tmp_path / "tasks").mkdir()

        from benchflow.evaluation import Evaluation

        job = Evaluation.from_yaml(tmp_path / "config.yaml")
        assert job._config.sandbox_user == "agent"


# _write_config records locked paths


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
            skill_policy=resolve_task_skill_policy(
                task_path=tmp_path / "task",
                skill_mode=SKILL_MODE_NO_SKILL,
                runtime_skills_dir=None,
                declared_sandbox_skills_dir=None,
            ),
            sandbox_user="agent",
            context_root=None,
            sandbox_locked_paths=["/solution", "/tests"],
            timeout=300,
            started_at=__import__("datetime").datetime.now(),
            agent_env={},
        )
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["sandbox_locked_paths"] == ["/solution", "/tests"]


# Privilege dropping command construction


class TestPrivDropCommand:
    """build_priv_drop_cmd — setpriv/su-l command generation."""

    def test_contains_setpriv_and_su_fallback(self):
        cmd = build_priv_drop_cmd("my-agent --stdio", "agent")
        assert "setpriv --reuid=agent --regid=agent --init-groups" in cmd
        assert "su -l agent -c" in cmd

    def test_exec_prefix(self):
        """Both branches use exec to replace the shell (no lingering parent)."""
        cmd = build_priv_drop_cmd("my-agent", "agent")
        assert "exec setpriv" in cmd
        assert "exec su" in cmd

    def test_inner_command_is_shlex_quoted(self):
        import shlex

        cmd = build_priv_drop_cmd("agent --flag value", "agent")
        inner = "export HOME=/home/agent && agent --flag value"
        assert shlex.quote(inner) in cmd

    def test_single_quotes_in_launch(self):
        """Single quotes in agent_launch don't break the command."""
        cmd = build_priv_drop_cmd("agent --prompt 'hello world'", "agent")
        assert "hello world" in cmd
        subprocess.run(["bash", "-n", "-c", cmd], check=True)

    def test_custom_sandbox_user(self):
        cmd = build_priv_drop_cmd("my-agent", "bench-user")
        assert "--reuid=bench-user" in cmd
        assert "su -l bench-user" in cmd
        assert "/home/bench-user" in cmd

    def test_no_web_policy_installs_owner_firewall_before_privilege_drop(self):
        """Guards PR #921 against terminal-based oracle downloads."""
        cmd = build_priv_drop_cmd("my-agent", "agent")

        assert 'BENCHFLOW_DISALLOW_WEB_TOOLS:-}" = "1"' in cmd
        assert "apt-get install -y -qq iptables" in cmd
        assert "dnf -y install iptables" in cmd
        assert "apk add --no-cache iptables" in cmd
        assert "http://127.0.0.1:*|http://localhost:*" in cmd
        assert "agent_uid=$(id -u agent)" in cmd
        assert '--uid-owner "$agent_uid" --dport "$proxy_port" -j ACCEPT' in cmd
        assert '--uid-owner "$agent_uid" -j REJECT' in cmd
        assert cmd.index("iptables -C OUTPUT") < cmd.index("exec setpriv")
        subprocess.run(["bash", "-n", "-c", cmd], check=True)
