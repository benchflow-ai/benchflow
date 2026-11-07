"""Network and metadata regressions present at ac3bedaa1."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.integrity.trial import facts_from_task_config
from benchflow.rollout._setup import _task_disallows_internet, _task_egress_denylist
from benchflow.sandbox import lockdown
from benchflow.sandbox._compose import compose_needs_net_admin
from benchflow.sandbox.egress_denylist import (
    EgressDenylist,
    agent_network_sandbox_config,
    denylist_agent_env,
    egress_denylist_for,
)
from benchflow.sandbox.lockdown import (
    _cloud_metadata_reach_cmd,
    _cloud_metadata_reach_results,
    enforce_agent_egress_firewall,
)
from benchflow.task import TaskConfig
from benchflow.task.plan_validation import agent_egress_policy


@pytest.mark.parametrize(
    "sandbox",
    [
        {"network_mode": "public"},
        {"network_mode": "denylist", "blocked_hosts": ["blocked.example"]},
        {
            "network_mode": "allowlist",
            "allowed_hosts": ["allowed.example", "192.0.2.0/24"],
        },
    ],
)
def test_offline_agent_effective_policy_and_record_agree(sandbox):
    """Guards ac3bedaa1: agent no-network dominates the inherited sandbox list."""
    config = TaskConfig.model_validate(
        {"sandbox": sandbox, "agent": {"network_mode": "no-network"}}
    )
    original = config.model_dump()
    task = SimpleNamespace(config=config)
    assert _task_disallows_internet(task)
    # The ordinary rollout already suppresses the proxy. Helpers and evidence
    # must agree with that actual selection, rather than retain the allowlist.
    assert (
        agent_egress_policy(
            task,
            agent="codex-acp",
            sandbox_user="agent",
            disallow_web_tools=True,
        )
        is None
    )
    effective = agent_network_sandbox_config(config)
    assert effective.network_mode == "no-network"
    assert effective.allow_internet  # installs and root verifier stay online
    assert compose_needs_net_admin(effective)  # agent uid is held offline
    assert effective.allowed_hosts == []
    assert effective.blocked_hosts == []
    assert effective.blocked_urls == []
    assert _task_egress_denylist(task) is None
    facts = facts_from_task_config(config, task_id="offline", workspace="/app")
    assert facts.agent_network_mode == "no-network"
    assert facts.allowed_hosts == []
    assert config.model_dump() == original


@pytest.mark.parametrize(
    "mode,fields",
    [
        ("allowlist", {"allowed_hosts": ["allowed.example"]}),
        ("denylist", {"blocked_hosts": ["blocked.example"]}),
    ],
)
def test_sandbox_policy_without_offline_override_is_unchanged(mode, fields):
    """Guards ac3bedaa1: unchanged sandbox restrictions remain selected."""
    config = TaskConfig.model_validate({"sandbox": {"network_mode": mode, **fields}})
    assert agent_network_sandbox_config(config) is config.sandbox
    assert egress_denylist_for(config.sandbox).mode == mode


@pytest.mark.parametrize(
    "sandbox",
    [
        {"network_mode": "public"},
        {"network_mode": "allowlist", "allowed_hosts": ["allowed.example"]},
    ],
)
def test_provider_without_agent_keeps_original_sandbox_network(
    monkeypatch, tmp_path, sandbox
):
    """Guards ac3bedaa1: offline agent facts must not add oracle capabilities."""
    from benchflow.sandbox import setup

    config = TaskConfig.model_validate(
        {"sandbox": sandbox, "agent": {"network_mode": "no-network"}}
    )
    task = SimpleNamespace(config=config)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    staged = SimpleNamespace(release=lambda: None, remove=lambda: None)
    monkeypatch.setattr(
        setup, "_validate_task_runtime_for_launch", lambda *a, **k: None
    )
    monkeypatch.setattr(setup, "_stage_environment", lambda *a: staged)

    class Provider:
        pass

    def make(*args, **kwargs):
        provider = Provider()
        provider.task_env_config = args[5]
        return provider

    monkeypatch.setattr(setup, "_make_sandbox", make)
    provider = setup._create_sandbox_environment(
        "docker", task, tmp_path, "no-agent", None, preserve_agent_network=False
    )
    assert provider.task_env_config is config.sandbox
    assert compose_needs_net_admin(provider.task_env_config) == (
        sandbox["network_mode"] == "allowlist"
    )


@pytest.mark.asyncio
async def test_conflicting_no_web_and_allowlist_markers_are_refused():
    """Guards ac3bedaa1: mixed policy inputs cannot admit listed IPs or a web proxy."""
    env = SimpleNamespace(exec=AsyncMock())
    policy = EgressDenylist((), (), allowed_hosts=("allowed.example", "192.0.2.0/24"))
    agent_env = denylist_agent_env({"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}, policy)
    with pytest.raises(RuntimeError, match=r"[Nn]o-web.*allowlist"):
        await enforce_agent_egress_firewall(env, "agent", agent_env)
    env.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_model_only_transport_remains_compatible_with_no_web():
    """Guards ac3bedaa1: native model-only proxy still gets a loopback-only firewall."""
    env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))
    policy = EgressDenylist((), (), native_claude_model_only=True)
    agent_env = denylist_agent_env({"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}, policy)
    await enforce_agent_egress_firewall(env, "agent", agent_env)
    command = env.exec.await_args.args[0]
    assert "REDIRECT" not in command
    assert "192.0.2.0/24" not in command


@pytest.mark.parametrize("timeout_binary", ["/usr/bin/timeout", "/usr/bin/busybox"])
@pytest.mark.parametrize(
    "operation,wanted",
    [
        ("exec /bin/sleep 10", "timeout"),
        ("exit 143", "failed"),
        ("exit 0", "reached"),
        ("echo 'Connection refused' >&2; exit 1", "refused"),
        ("echo 'Network is unreachable' >&2; exit 1", "unreachable"),
    ],
)
def test_metadata_deadline_is_portable_and_distinct_from_signal_exit(
    tmp_path, timeout_binary, operation, wanted
):
    """Guards PR #9/ac3bedaa1 using real timeout tools and a socket-free stand-in."""
    if not Path(timeout_binary).is_file():
        pytest.skip(f"{timeout_binary} is not installed")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "timeout").symlink_to(timeout_binary)
    # Replace only the socket operation, including inside the deadline wrapper.
    # The real GNU/BusyBox tool and the production wrapper still run unchanged.
    bash = bindir / "bash"
    bash.write_text(
        '#!/bin/sh\ncase "$2" in\n'
        f"  'exec 3<>/dev/tcp/'*) {operation};;\n"
        '  *) exec /bin/bash "$@";;\nesac\n'
    )
    bash.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", _cloud_metadata_reach_cmd()],
        env={"PATH": f"{bindir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    probes = _cloud_metadata_reach_results(result.stdout)
    assert probes is not None, result.stdout
    assert set(probes.values()) == {wanted}


@pytest.mark.parametrize("timeout_binary", ["/usr/bin/timeout", "/usr/bin/busybox"])
def test_metadata_socket_probe_reaches_actual_loopback(
    monkeypatch, tmp_path, timeout_binary
):
    """Guards PR #9/ac3bedaa1: the wrapper retains the real Bash socket operation."""
    if not Path(timeout_binary).is_file():
        pytest.skip(f"{timeout_binary} is not installed")
    (tmp_path / "timeout").symlink_to(timeout_binary)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(lockdown, "CLOUD_METADATA_V4", ("127.0.0.1",))
        monkeypatch.setattr(lockdown, "CLOUD_METADATA_V6", ())
        monkeypatch.setattr(
            lockdown, "_CLOUD_METADATA_REACH_PORTS", (listener.getsockname()[1],)
        )
        result = subprocess.run(
            ["/bin/sh", "-c", _cloud_metadata_reach_cmd()],
            env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        probes = _cloud_metadata_reach_results(result.stdout)
        assert probes is not None, result.stdout
        assert set(probes.values()) == {"reached"}
