"""Tests for path lockdown — _validate_locked_path, _resolve_locked_paths, lockdown_paths."""

import json
import shutil
import subprocess
import sys
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

    def test_privilege_drop_does_not_install_firewall_during_acp_bootstrap(self):
        """Guards PR #921 against blocking OpenHands before ACP initialize."""
        cmd = build_priv_drop_cmd("my-agent", "agent")

        assert "iptables" not in cmd
        subprocess.run(["bash", "-n", "-c", cmd], check=True)


class TestAgentEgressFirewall:
    """External egress is blocked after ACP bootstrap but before prompting."""

    def test_firewall_allows_loopback_and_rejects_ipv4_and_ipv6_egress(self):
        from benchflow.sandbox.lockdown import _agent_egress_firewall_cmd

        cmd = _agent_egress_firewall_cmd("agent")

        assert "apt-get install -y -qq iptables" in cmd
        assert "dnf -y install iptables" in cmd
        assert "apk add --no-cache iptables" in cmd
        assert "agent_uid=$(id -u agent)" in cmd
        assert '-o lo -m owner --uid-owner "$agent_uid" -j ACCEPT' in cmd
        assert '--uid-owner "$agent_uid" -j REJECT' in cmd
        assert "ip6tables -C OUTPUT -o lo" in cmd
        assert "IPv6 enabled but ip6tables unavailable" in cmd
        subprocess.run(["bash", "-n", "-c", cmd], check=True)

    async def test_policy_is_skipped_when_web_tools_are_allowed(self):
        from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

        env = MagicMock()
        env.exec = AsyncMock()

        await enforce_agent_egress_firewall(
            env,
            "agent",
            {"LLM_BASE_URL": "http://127.0.0.1:1234"},
        )

        env.exec.assert_not_awaited()

    async def test_policy_requires_loopback_proxy_and_runs_as_root(self):
        from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(return_code=0))

        await enforce_agent_egress_firewall(
            env,
            "agent",
            {
                "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
                "LLM_BASE_URL": "http://127.0.0.1:1234",
            },
        )

        env.exec.assert_awaited_once()
        (_cmd,) = env.exec.await_args.args
        assert "iptables -C OUTPUT -o lo" in _cmd
        assert env.exec.await_args.kwargs == {"user": "root", "timeout_sec": 120}

    async def test_policy_accepts_canonical_loopback_provider_url(self):
        """Guards PR #942 remediation for native-protocol reviewer proxies."""

        from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(return_code=0))

        await enforce_agent_egress_firewall(
            env,
            "reviewer",
            {
                "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
                "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:45678/v1",
            },
        )

        env.exec.assert_awaited_once()

    async def test_policy_rejects_non_loopback_proxy(self):
        from benchflow.sandbox.lockdown import enforce_agent_egress_firewall

        env = MagicMock()
        env.exec = AsyncMock()

        with pytest.raises(RuntimeError, match="loopback provider base URL"):
            await enforce_agent_egress_firewall(
                env,
                "agent",
                {
                    "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
                    "LLM_BASE_URL": "https://api.openai.com/v1",
                },
            )

        env.exec.assert_not_awaited()


FAKE_IPTABLES = r'''#!{python}
"""A stateful iptables: rules per table and chain in a JSON file."""
import json, os, sys
state_path = os.environ["FW_STATE"]
name = os.path.basename(sys.argv[0])
args = [a for a in sys.argv[1:] if a != "-w"]
with open(os.environ["FW_LOG"], "a") as log:
    log.write(name + " " + " ".join(sys.argv[1:]) + "\n")
if os.environ.get("FW_BROKEN") == name:
    sys.exit(1)
state = json.load(open(state_path)) if os.path.exists(state_path) else {{}}
table = "filter"
if args[:1] == ["-t"]:
    table, args = args[1], args[2:]
key = name.replace("-legacy", "") + ":" + table
op, chain, rest = args[0], (args[1] if len(args) > 1 else ""), args[2:]
rules = state.setdefault(key, {{}}).setdefault(chain, [])
if op in ("-S", "-L", "-nL"):
    sys.exit(0)
if op == "-I" and rest and rest[0].isdigit():
    pos, rest = int(rest[0]) - 1, rest[1:]
else:
    pos = 0
spec = " ".join(rest)
if op == "-C":
    sys.exit(0 if spec in rules else 1)
if op == "-D":
    if spec in rules:
        rules.remove(spec)
        json.dump(state, open(state_path, "w"))
        sys.exit(0)
    sys.exit(1)
if op == "-I":
    rules.insert(pos, spec)
elif op == "-A":
    rules.append(spec)
json.dump(state, open(state_path, "w"))
'''


def _verdict(state, fam="iptables", uid=1000, dport=2280, out="lo"):
    """What the kernel does with the agent's packet: mangle, then filter."""

    def matches(spec):
        t = spec.split()
        ok = True
        for i, w in enumerate(t):
            if w == "-o" and t[i + 1] != out:
                ok = False
            if w == "--dport" and int(t[i + 1]) != dport:
                ok = False
            if w == "--uid-owner":
                neg = t[i - 1] == "!"
                ok = ok and ((int(t[i + 1]) == uid) != neg)
        return ok

    for table in ("mangle", "filter"):
        for spec in state.get(f"{fam}:{table}", {}).get("OUTPUT", []):
            if matches(spec):
                target = spec.split("-j", 1)[1].split()[0]
                if target in ("DROP", "REJECT"):
                    return "blocked"
                if target == "ACCEPT":
                    break
    return "passes"


FAKE_IPTABLES = r'''#!{python}
"""A stateful iptables: rules per table and chain in a JSON file."""
import json, os, sys
state_path = os.environ["FW_STATE"]
name = os.path.basename(sys.argv[0])
args = [a for a in sys.argv[1:] if a != "-w"]
with open(os.environ["FW_LOG"], "a") as log:
    log.write(name + " " + " ".join(sys.argv[1:]) + "\n")
if os.environ.get("FW_BROKEN") == name:
    sys.exit(1)
state = json.load(open(state_path)) if os.path.exists(state_path) else {{}}
table = "filter"
if args[:1] == ["-t"]:
    table, args = args[1], args[2:]
key = name.replace("-legacy", "") + ":" + table
op, chain, rest = args[0], (args[1] if len(args) > 1 else ""), args[2:]
rules = state.setdefault(key, {{}}).setdefault(chain, [])
if op in ("-S", "-L", "-nL"):
    sys.exit(0)
if op == "-I" and rest and rest[0].isdigit():
    pos, rest = int(rest[0]) - 1, rest[1:]
else:
    pos = 0
spec = " ".join(rest)
if op == "-C":
    sys.exit(0 if spec in rules else 1)
if op == "-D":
    if spec in rules:
        rules.remove(spec)
        json.dump(state, open(state_path, "w"))
        sys.exit(0)
    sys.exit(1)
if op == "-I":
    rules.insert(pos, spec)
elif op == "-A":
    rules.append(spec)
json.dump(state, open(state_path, "w"))
'''


def _verdict(state, fam="iptables", uid=1000, dport=2280, out="lo"):
    """What the kernel does with the agent's packet: mangle, then filter."""

    def matches(spec):
        t = spec.split()
        ok = True
        for i, w in enumerate(t):
            if w == "-o" and t[i + 1] != out:
                ok = False
            if w == "--dport" and int(t[i + 1]) != dport:
                ok = False
            if w == "--uid-owner":
                neg = t[i - 1] == "!"
                ok = ok and ((int(t[i + 1]) == uid) != neg)
        return ok

    for table in ("mangle", "filter"):
        for spec in state.get(f"{fam}:{table}", {}).get("OUTPUT", []):
            if matches(spec):
                target = spec.split("-j", 1)[1].split()[0]
                if target in ("DROP", "REJECT"):
                    return "blocked"
                if target == "ACCEPT":
                    break
    return "passes"


class TestSandboxDaemonGuard:
    """The agent user cannot reach Daytona's root daemon, or the run is refused."""

    @staticmethod
    def _bin(tmp_path, tools=("iptables", "ip6tables"), extra=None):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        for name in tools:
            p = bin_dir / name
            p.write_text(FAKE_IPTABLES.format(python=sys.executable))
            p.chmod(0o755)
        for name, body in (extra or {}).items():
            p = bin_dir / name
            p.write_text("#!/bin/sh\n" + body)
            p.chmod(0o755)
        idp = bin_dir / "id"
        idp.write_text("#!/bin/sh\necho 1000\n")
        idp.chmod(0o755)
        return bin_dir

    @staticmethod
    def _proc(tmp_path, pid1="/usr/local/bin/daytona", listen_port=None, ipv6=True):
        proc = tmp_path / "proc"
        (proc / "net").mkdir(parents=True)
        (proc / "1" / "fd").mkdir(parents=True)
        (proc / "1" / "exe").symlink_to(pid1)
        header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        lines = header
        if listen_port is not None:
            (proc / "1" / "fd" / "3").symlink_to("socket:[4242]")
            lines += f"   0: 00000000:{listen_port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 4242 1\n"
        (proc / "net" / "tcp").write_text(lines)
        (proc / "net" / "tcp6").write_text(header)
        if ipv6:
            (proc / "net" / "if_inet6").write_text(
                "00000000000000000000000000000001 01 80 10 80       lo\n"
            )
        return proc

    def _sh(self, tmp_path, cmd, bin_dir, shell="bash", broken=""):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "FW_STATE": str(tmp_path / "fw.json"),
            "FW_LOG": str(tmp_path / "fw.log"),
            "FW_BROKEN": broken,
        }
        return subprocess.run(
            [shell, "-c", cmd], capture_output=True, text=True, env=env, check=True
        ).stdout

    def _state(self, tmp_path):
        p = tmp_path / "fw.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _calls(self, tmp_path):
        p = tmp_path / "fw.log"
        return p.read_text().splitlines() if p.exists() else []

    def _guard(
        self,
        tmp_path,
        shell="bash",
        tools=("iptables", "ip6tables"),
        extra=None,
        broken="",
        **kw,
    ):
        from benchflow.sandbox.lockdown import _daemon_guard_cmd

        bin_dir = self._bin(tmp_path, tools, extra)
        proc = self._proc(tmp_path, **kw)
        return self._sh(
            tmp_path, _daemon_guard_cmd("agent", proc=str(proc)), bin_dir, shell, broken
        )

    def test_drops_the_agent_on_every_daemon_port_in_the_mangle_table(self, tmp_path):
        out = self._guard(tmp_path)
        assert "status=0" in out
        state = self._state(tmp_path)
        for fam in ("iptables", "ip6tables"):
            rules = state[f"{fam}:mangle"]["OUTPUT"]
            for port in (2280, 22220, 22222, 33333):
                # No `-o lo`: the gateway or another address is refused too.
                assert (
                    f"-p tcp --dport {port} -m owner --uid-owner 1000 -j DROP" in rules
                )
            assert len(rules) == 4
        # Running it again adds nothing and removes nothing.
        self._guard_again(tmp_path)
        assert len(self._state(tmp_path)["iptables:mangle"]["OUTPUT"]) == 4

    def _guard_again(self, tmp_path):
        from benchflow.sandbox.lockdown import _daemon_guard_cmd

        self._sh(
            tmp_path,
            _daemon_guard_cmd("agent", proc=str(tmp_path / "proc")),
            tmp_path / "bin",
        )

    def test_the_egress_firewall_never_opens_a_window(self, tmp_path):
        # The guard at setup, then the firewall with its loopback ACCEPT at
        # the top of filter: at no point does the agent's packet pass.
        from benchflow.sandbox.lockdown import _agent_egress_firewall_cmd

        self._guard(tmp_path)
        assert _verdict(self._state(tmp_path)) == "blocked"
        self._sh(tmp_path, _agent_egress_firewall_cmd("agent"), tmp_path / "bin")
        state = self._state(tmp_path)
        assert any("ACCEPT" in r for r in state["iptables:filter"]["OUTPUT"])
        assert _verdict(state) == "blocked"
        assert _verdict(state, fam="ip6tables") == "blocked"
        # Other loopback ports stay open to the agent.
        assert _verdict(state, dport=8080) == "passes"

    def test_a_port_the_daemon_listens_on_is_guarded_too(self, tmp_path):
        out = self._guard(tmp_path, listen_port=8080)
        assert "ports=2280 8080 22220 22222 33333" in out

    def test_a_process_1_that_is_not_the_daemon_adds_no_port(self, tmp_path):
        out = self._guard(tmp_path, pid1="/usr/bin/python3.11", listen_port=8000)
        assert "ports=2280 22220 22222 33333 " in out

    def test_discovery_works_under_dash(self, tmp_path):
        if not shutil.which("dash"):
            pytest.skip("no dash here")
        out = self._guard(tmp_path, shell="dash", listen_port=8080)
        assert "ports=2280 8080 22220 22222 33333" in out

    def test_no_iptables_and_no_package_manager_is_status_2(self, tmp_path):
        out = self._guard(tmp_path, tools=())
        assert "status=2" in out

    def test_a_failed_install_is_status_5_with_its_words(self, tmp_path):
        apt = "echo 'E: Could not get lock /var/lib/dpkg/lock-frontend' >&2; exit 100\n"
        out = self._guard(tmp_path, tools=(), extra={"apt-get": apt})
        assert "status=5" in out
        assert "BF daemon-guard install: E: Could not get lock" in out

    def test_alpine_gets_ip6tables_too(self, tmp_path):
        # Alpine ships ip6tables apart from iptables.
        apk = (
            'for t in iptables ip6tables; do cp "$(dirname "$0")/fw-template" "$(dirname "$0")/$t"; '
            'chmod 755 "$(dirname "$0")/$t"; done\n'
        )
        bin_dir = self._bin(tmp_path, tools=("iptables",), extra={"apk": apk})
        (bin_dir / "fw-template").write_text(
            FAKE_IPTABLES.format(python=sys.executable)
        )
        from benchflow.sandbox.lockdown import _daemon_guard_cmd

        proc = self._proc(tmp_path)
        out = self._sh(tmp_path, _daemon_guard_cmd("agent", proc=str(proc)), bin_dir)
        assert "status=0" in out
        assert len(self._state(tmp_path)["ip6tables:mangle"]["OUTPUT"]) == 4

    def test_the_legacy_backend_is_used_where_nft_does_not_work(self, tmp_path):
        out = self._guard(
            tmp_path,
            tools=("iptables", "iptables-legacy", "ip6tables", "ip6tables-legacy"),
            broken="iptables",
        )
        assert "status=0" in out
        assert any(
            c.startswith("iptables-legacy -w -t mangle -I")
            for c in self._calls(tmp_path)
        )

    def test_ipv6_without_ip6tables_is_reported(self, tmp_path):
        out = self._guard(tmp_path, tools=("iptables",))
        assert "status=3" in out

    @staticmethod
    def _env(mode, stdout="BF daemon-guard ports=2280 status=0", rc=0):
        env = MagicMock()
        env.sandbox_daemon_guard = mode
        env.exec = AsyncMock(
            return_value=MagicMock(return_code=rc, stdout=stdout, stderr="")
        )
        return env

    async def test_the_guard_runs_only_where_the_daemon_shares_the_network(self):
        from benchflow.sandbox.lockdown import guard_sandbox_daemon

        for mode in (None, "vm"):
            env = self._env(mode)
            await guard_sandbox_daemon(env, "agent")
            env.exec.assert_not_awaited()
        env = self._env("container")
        await guard_sandbox_daemon(env, "agent")
        assert "BF daemon-guard" in env.exec.await_args.args[0]
        assert env.exec.await_args.kwargs["user"] == "root"

    async def test_a_run_that_runs_nothing_as_the_user_needs_no_guard(self):
        # The oracle runs as root: an image without iptables must not refuse it.
        from benchflow.rollout_planes import DefaultRolloutPlanes
        from benchflow.sandbox.lockdown import setup_sandbox_user

        env = self._env("container", stdout="BF daemon-guard ports=2280 status=2")
        await setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
        assert all("BF daemon-guard" not in c.args[0] for c in env.exec.await_args_list)
        env = self._env("container", stdout="BF daemon-guard ports=2280 status=2")
        await DefaultRolloutPlanes().setup_sandbox_user(
            env, "agent", workspace="/app", guard_daemon=False
        )
        assert all("BF daemon-guard" not in c.args[0] for c in env.exec.await_args_list)

    @pytest.mark.parametrize(
        "out, why, lasting",
        [
            (
                "BF daemon-guard ports=2280 status=2",
                "no iptables and no package manager",
                True,
            ),
            ("BF daemon-guard ports=2280 status=3", "has IPv6 but no ip6tables", True),
            ("BF daemon-guard ports=2280 status=1", "rule could not be added", False),
            (
                "BF daemon-guard install: E: Could not get lock\nBF daemon-guard ports=2280 status=5",
                "installing iptables failed (E: Could not get lock)",
                False,
            ),
            ("", "did not report", False),
        ],
    )
    async def test_a_guard_that_cannot_be_installed_refuses_the_run(
        self, out, why, lasting
    ):
        from benchflow._utils.scoring import SANDBOX_SETUP, classify_error
        from benchflow.sandbox.lockdown import (
            SandboxDaemonGuardError,
            setup_sandbox_user,
        )

        env = self._env("container", stdout=out)
        with pytest.raises(SandboxDaemonGuardError) as e:
            await setup_sandbox_user(env, "agent", "/app")
        msg = str(e.value)
        assert why in msg
        assert classify_error(msg) == SANDBOX_SETUP
        assert _never_retried(msg) == lasting

    async def test_docker_gets_no_guard_and_no_iptables_install(self):
        from benchflow.sandbox.lockdown import setup_sandbox_user

        env = MagicMock(spec=["exec"])
        env.exec = AsyncMock(
            return_value=MagicMock(return_code=0, stdout="", stderr="")
        )
        await setup_sandbox_user(env, "agent", "/app")
        assert all("BF daemon-guard" not in c.args[0] for c in env.exec.await_args_list)


def _never_retried(msg):
    """Whether a refusal is the kind retrying cannot fix (the image's)."""
    try:
        from benchflow._utils.scoring import is_unrecoverable_startup_error
    except ImportError:
        from benchflow.evaluation import (
            _image_cannot_guard_daemon as is_unrecoverable_startup_error,
        )
    return is_unrecoverable_startup_error(msg)


class TestDinDDaemonGuard:
    """Compose tasks on Daytona: the VM refuses the containers' connections."""

    @staticmethod
    def _strategy(stdout, rc=0):
        from benchflow.sandbox.daytona_dind import _DaytonaDinD

        strat = _DaytonaDinD.__new__(_DaytonaDinD)
        strat._vm_exec = AsyncMock(
            return_value=MagicMock(return_code=rc, stdout=stdout, stderr="")
        )
        return strat

    async def _command(self):
        strat = self._strategy("BF dind-daemon-guard status=0\n")
        await strat._guard_daemon_from_containers()
        return strat._vm_exec.await_args.args[0]

    async def test_the_vm_refuses_bridges_and_host_network_processes(self, tmp_path):
        cmd = await self._command()
        t = TestSandboxDaemonGuard()
        bin_dir = t._bin(tmp_path)
        out = t._sh(tmp_path, cmd, bin_dir, shell="sh")
        assert "status=0" in out
        state = t._state(tmp_path)
        for fam in ("iptables", "ip6tables"):
            inp = state[f"{fam}:filter"]["INPUT"]
            for i in ("docker0", "br-+"):
                for p in (2280, 22220, 22222, 33333):
                    assert (
                        f"-i {i} -p tcp --dport {p} -j REJECT --reject-with tcp-reset"
                        in inp
                    )
            # A host-network container's agent (uid 1000) is dropped on the
            # VM itself; root (the daemon, dockerd, the runner) is not.
            assert _verdict(state, fam=fam, uid=1000) == "blocked"
            assert _verdict(state, fam=fam, uid=0) == "passes"

    async def test_the_vm_uses_the_legacy_backend_where_nft_does_not_work(
        self, tmp_path
    ):
        cmd = await self._command()
        t = TestSandboxDaemonGuard()
        bin_dir = t._bin(
            tmp_path,
            tools=("iptables", "iptables-legacy", "ip6tables", "ip6tables-legacy"),
        )
        out = t._sh(tmp_path, cmd, bin_dir, shell="sh", broken="iptables")
        assert "status=0" in out
        assert any(
            c.startswith("iptables-legacy -w -I INPUT") for c in t._calls(tmp_path)
        )

    async def test_rules_that_cannot_be_installed_refuse_the_run(self):
        from benchflow.sandbox.lockdown import SandboxDaemonGuardError

        strat = self._strategy("BF dind-daemon-guard status=1\n")
        with pytest.raises(SandboxDaemonGuardError, match="sandbox startup refused"):
            await strat._guard_daemon_from_containers()

    def test_daytona_says_where_its_daemon_is_guarded(self):
        from benchflow.sandbox.daytona import DaytonaSandbox

        env = DaytonaSandbox.__new__(DaytonaSandbox)
        env._compose_mode = False
        assert env.sandbox_daemon_guard == "container"
        env._compose_mode = True
        assert env.sandbox_daemon_guard == "vm"
